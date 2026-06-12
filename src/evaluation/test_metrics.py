"""
src/evaluation/test_metrics.py
==============================

Final held-out test-set evaluation for a trained (or QAT-wrapped) model.

This module is the authoritative source for the numbers that appear in the
thesis tables.  It is intentionally more thorough than
:mod:`src.evaluation.validate` (which only monitors training) and
:mod:`src.evaluation.auroc_eval` (which focuses on AUROC/PR-AUC curves):

Metrics computed
----------------
Image-level
    AUROC, PR-AUC (AUPRC), F1 at the best threshold, accuracy,
    precision, recall, specificity, confusion matrix, reconstruction loss.

Pixel-level  (when GT masks are available in the manifest)
    AUROC, PR-AUC, F1 at the best threshold.

Per-defect-type breakdown
    All image-level metrics repeated for every defect category found in the
    manifest (e.g. ``"crack"``, ``"scratch"``, ``"good"`` …).

Score diagnostics
    Mean, std, min, max, and per-class histograms of the raw anomaly scores
    (useful for threshold selection and calibration analysis).

Public interface
----------------
``compute_test_metrics(
        model: nn.Module,
        splits_dir: Path,
        category: str,
        device: str,
        split: str = "test",
        *,
        batch_size: int = 16,
        num_workers: int = 2,
        image_size: int | None = None,
        amp: bool = False,
        save_dir: Path | None = None,
        **kwargs) -> dict``

    Complete test evaluation.  Writes ``test_metrics.json`` to *save_dir*
    when provided.  Returns the full metrics dict.

``load_saved_metrics(save_dir: Path, category: str) -> dict``
    Utility: reload a previously saved ``test_metrics.json``.

Assumptions
-----------
- Split manifests were produced by ``build_splits.py``; records carry
  ``abs_path``, ``label`` (0 = normal, 1 = anomaly), ``defect`` (string),
  and optionally ``abs_mask`` / ``mask``.
- The model's ``forward`` follows the unified output schema:
  ``{"recon"?, "features"?, "anomaly_map"?, "score"?, "logits"?}``.
- ``auroc_eval.py`` and ``validate.py`` live in the same package and are
  imported for their lower-level helpers.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image as _PIL_Image  # type: ignore
    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _PIL_Image = None  # type: ignore
    _HAVE_PIL = False

try:
    from sklearn.metrics import (  # type: ignore
        roc_auc_score,
        average_precision_score,
        roc_curve,
    )
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    roc_auc_score = None            # type: ignore
    average_precision_score = None  # type: ignore
    roc_curve = None                # type: ignore
    _HAVE_SKLEARN = False

__all__ = ["compute_test_metrics", "load_saved_metrics"]

LOG = logging.getLogger(__name__)

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)

_METRICS_FILENAME = "test_metrics.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _TestDataset(Dataset):
    """ImageNet-normalised dataset that also loads GT masks and defect labels."""

    def __init__(self,
                 records: list[dict[str, Any]],
                 *,
                 image_size: int) -> None:
        if not _HAVE_PIL:
            raise RuntimeError("Pillow is required for image loading.")
        self.records    = list(records)
        self.image_size = int(image_size)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec   = self.records[idx]
        ipath = rec.get("abs_path") or rec.get("path")
        if ipath is None:
            raise KeyError(f"Record {idx} has no 'abs_path' or 'path' field.")

        with _PIL_Image.open(ipath) as img:
            img = img.convert("RGB").resize(
                (self.image_size, self.image_size), _PIL_Image.BILINEAR,
            )
            arr = (
                torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float() / 255.0
            )

        img_t = arr.permute(2, 0, 1).contiguous()
        img_t = (img_t - self._mean) / self._std

        # GT mask — zeros when absent or label==0.
        mpath = rec.get("abs_mask") or rec.get("mask")
        if mpath and Path(str(mpath)).is_file():
            with _PIL_Image.open(mpath) as m:
                m = m.convert("L").resize(
                    (self.image_size, self.image_size), _PIL_Image.NEAREST,
                )
                mask_t = torch.from_numpy(
                    (np.asarray(m, dtype=np.uint8) > 0).astype(np.float32)
                ).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, self.image_size, self.image_size,
                                 dtype=torch.float32)

        return {
            "image":  img_t,
            "label":  int(rec.get("label", 0)),
            "defect": str(rec.get("defect", "unknown")),
            "path":   str(ipath),
            "mask":   mask_t,
        }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_records(splits_dir: Path, category: str, split: str) -> list[dict]:
    path = Path(splits_dir) / f"{category}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. "
            "Run main_prepare.py / build_splits first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if split not in manifest.get("splits", {}):
        raise KeyError(
            f"Split {split!r} not in manifest "
            f"(available: {list(manifest['splits'])})"
        )
    return list(manifest["splits"][split])


def _build_loader(records: list[dict],
                  *,
                  image_size: int,
                  batch_size: int,
                  num_workers: int) -> DataLoader:
    bs = max(1, min(batch_size, len(records) or 1))
    return DataLoader(
        _TestDataset(records, image_size=image_size),
        batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _image_score(outputs: dict[str, torch.Tensor],
                 inputs: torch.Tensor) -> torch.Tensor | None:
    if "score" in outputs and isinstance(outputs["score"], torch.Tensor):
        s = outputs["score"]
        return s.flatten() if s.dim() <= 1 else s.flatten(1).amax(dim=1)
    if "anomaly_map" in outputs and isinstance(outputs["anomaly_map"], torch.Tensor):
        return outputs["anomaly_map"].flatten(1).amax(dim=1)
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return (recon - inputs).pow(2).mean(dim=1).flatten(1).amax(dim=1)
    return None


def _pixel_map(outputs: dict[str, torch.Tensor],
               inputs: torch.Tensor) -> torch.Tensor | None:
    """Return per-pixel anomaly map [B,1,H,W] aligned to input resolution."""
    if "anomaly_map" in outputs and isinstance(outputs["anomaly_map"], torch.Tensor):
        m = outputs["anomaly_map"]
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[-2:] != inputs.shape[-2:]:
            m = F.interpolate(m, size=inputs.shape[-2:],
                              mode="bilinear", align_corners=False)
        return m
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return (recon - inputs).pow(2).mean(dim=1, keepdim=True)
    return None


def _compute_recon_loss(outputs: dict[str, torch.Tensor],
                        inputs: torch.Tensor) -> float | None:
    """MSE reconstruction loss for the batch; None when not applicable."""
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return float(F.mse_loss(recon, inputs).item())
    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or not np.isfinite(scores).all():
        return None
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return None
    if _HAVE_SKLEARN:
        try:
            return float(roc_auc_score(labels, scores))
        except Exception:  # noqa: BLE001
            pass
    # Pure-numpy fallback
    order = np.argsort(-scores, kind="stable")
    y     = labels[order].astype(np.float64)
    n_neg = float(y.size - n_pos)
    tpr   = np.concatenate([[0.0], np.cumsum(y) / n_pos])
    fpr   = np.concatenate([[0.0], np.cumsum(1.0 - y) / max(n_neg, 1.0)])
    return float(np.trapezoid(tpr, fpr))


def _safe_prauc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or not np.isfinite(scores).all():
        return None
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == labels.size:
        return None
    if _HAVE_SKLEARN:
        try:
            return float(average_precision_score(labels, scores))
        except Exception:  # noqa: BLE001
            pass
    # Pure-numpy fallback
    order = np.argsort(-scores, kind="stable")
    y     = labels[order].astype(np.float64)
    tp    = np.cumsum(y)
    fp    = np.cumsum(1.0 - y)
    prec  = np.concatenate([[1.0], tp / (tp + fp)])
    rec   = np.concatenate([[0.0], tp / max(n_pos, 1.0)])
    return float(np.trapezoid(prec, rec))


def _best_threshold_metrics(labels: np.ndarray,
                             scores: np.ndarray,
                             n_thresholds: int = 200
                             ) -> dict[str, float | None]:
    """
    Return classification metrics at the F1-optimal threshold.

    Keys: threshold, f1, accuracy, precision, recall, specificity.
    """
    empty: dict[str, float | None] = {
        k: None for k in
        ("threshold", "f1", "accuracy", "precision", "recall", "specificity")
    }
    if labels.size == 0 or int(labels.sum()) == 0:
        return empty

    if _HAVE_SKLEARN and roc_curve is not None:
        try:
            fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
            n_pos = float(labels.sum())
            n_neg = float(labels.size - n_pos)
            tp = tpr_arr * n_pos
            fp = fpr_arr * n_neg
            fn = n_pos - tp
            tn = n_neg - fp
            denom = 2 * tp + fp + fn
            f1s = np.where(denom > 0, 2 * tp / denom, 0.0)
            best = int(np.argmax(f1s))
            thr  = float(thresholds[best])
            preds = (scores >= thr).astype(np.int64)
            return _confusion_metrics(labels, preds, thr)
        except Exception:  # noqa: BLE001
            pass

    # Quantile sweep fallback
    thresholds = np.unique(
        np.percentile(scores, np.linspace(0, 100, n_thresholds))
    )
    best_f1, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    preds = (scores >= best_thr).astype(np.int64)
    return _confusion_metrics(labels, preds, best_thr)


def _confusion_metrics(labels: np.ndarray,
                       preds: np.ndarray,
                       threshold: float) -> dict[str, float | None]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn
    f1_d  = 2 * tp + fp + fn
    return {
        "threshold":   threshold,
        "f1":          (2 * tp / f1_d)      if f1_d  > 0 else 0.0,
        "accuracy":    (tp + tn) / total     if total > 0 else 0.0,
        "precision":   tp / (tp + fp)        if (tp + fp) > 0 else 0.0,
        "recall":      tp / (tp + fn)        if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp)        if (tn + fp) > 0 else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _score_stats(scores: np.ndarray) -> dict[str, float | None]:
    if scores.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "p25": None, "p50": None, "p75": None, "p95": None}
    return {
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "min":  float(np.min(scores)),
        "max":  float(np.max(scores)),
        "p25":  float(np.percentile(scores, 25)),
        "p50":  float(np.percentile(scores, 50)),
        "p75":  float(np.percentile(scores, 75)),
        "p95":  float(np.percentile(scores, 95)),
    }


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


def _run_inference(model: nn.Module,
                   loader: DataLoader,
                   device_t: torch.device,
                   use_amp: bool
                   ) -> dict[str, Any]:
    """
    Single forward pass over the entire loader.

    Returns raw collected arrays plus totals.  Does not compute metrics.
    """
    img_scores:   list[np.ndarray] = []
    img_labels:   list[np.ndarray] = []
    img_defects:  list[list[str]]  = []
    pix_scores:   list[np.ndarray] = []
    pix_labels:   list[np.ndarray] = []
    loss_acc = 0.0
    loss_n   = 0
    n_seen = n_normal = n_anomaly = 0

    for batch in loader:
        x       = batch["image"].to(device_t, non_blocking=True)
        labels  = batch["label"]
        defects = list(batch["defect"])
        masks   = batch["mask"]  # [B,1,H,W] float

        ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if use_amp else _nullctx())
        with ctx:
            raw = model(x)

        outputs: dict[str, torch.Tensor] = (
            raw if isinstance(raw, dict) else {"recon": raw}
        )

        # Image scores
        iscore = _image_score(outputs, x)
        if iscore is not None:
            img_scores.append(iscore.detach().float().cpu().numpy())
            img_labels.append(labels.cpu().numpy().astype(np.int64))
            img_defects.append(defects)

        # Pixel maps vs GT masks
        pmap = _pixel_map(outputs, x)
        if pmap is not None:
            pmap_cpu = pmap.detach().float().cpu()
            for b in range(x.shape[0]):
                # Include all images; normal ones have all-zero masks.
                p_s = pmap_cpu[b].flatten().numpy()
                p_l = (masks[b].flatten() > 0.5).numpy().astype(np.int64)
                pix_scores.append(p_s)
                pix_labels.append(p_l)

        # Reconstruction loss (informational)
        loss_val = _compute_recon_loss(outputs, x)
        if loss_val is not None:
            loss_acc += loss_val * x.shape[0]
            loss_n   += x.shape[0]

        n_seen    += x.shape[0]
        n_normal  += int((labels == 0).sum().item())
        n_anomaly += int((labels == 1).sum().item())

    return {
        "img_scores":  img_scores,
        "img_labels":  img_labels,
        "img_defects": img_defects,
        "pix_scores":  pix_scores,
        "pix_labels":  pix_labels,
        "loss_acc":    loss_acc,
        "loss_n":      loss_n,
        "n_seen":      n_seen,
        "n_normal":    n_normal,
        "n_anomaly":   n_anomaly,
    }


# ---------------------------------------------------------------------------
# Per-defect breakdown
# ---------------------------------------------------------------------------

def _per_defect_metrics(all_scores:  np.ndarray,
                        all_labels:  np.ndarray,
                        all_defects: list[str],
                        best_thr:    float) -> dict[str, dict]:
    """
    Image-level AUROC, PR-AUC, and F1 broken down by defect type.

    Normal images are grouped under the key ``"good"``.
    """
    defect_arr = np.asarray(all_defects, dtype=object)
    unique     = sorted(set(all_defects))
    breakdown  = {}
    for dtype in unique:
        mask = defect_arr == dtype
        sub_scores = all_scores[mask]
        sub_labels = all_labels[mask]
        n_d        = int(mask.sum())
        n_anom_d   = int(sub_labels.sum())
        preds      = (sub_scores >= best_thr).astype(np.int64)
        cm         = _confusion_metrics(sub_labels, preds, best_thr)
        breakdown[dtype] = {
            "n":        n_d,
            "n_anomaly": n_anom_d,
            "auroc":    _safe_auroc(sub_labels, sub_scores),
            "pr_auc":   _safe_prauc(sub_labels, sub_scores),
            **{k: cm[k] for k in ("f1", "accuracy", "precision", "recall")},
            "score_stats": _score_stats(sub_scores),
        }
    return breakdown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_test_metrics(model: nn.Module,
                         splits_dir: Path,
                         category: str,
                         device: str,
                         split: str = "test",
                         *,
                         batch_size: int = 16,
                         num_workers: int = 2,
                         image_size: int | None = None,
                         amp: bool = False,
                         save_dir: Path | None = None,
                         **_unused) -> dict[str, Any]:
    """Run final test-set evaluation and return a comprehensive metrics dict.

    Parameters
    ----------
    model:
        Trained / QAT-wrapped ``nn.Module``.  Its ``forward`` may return a
        dict or a raw tensor (treated as ``{"recon": tensor}``).
    splits_dir:
        Directory containing per-category ``*.json`` manifests from
        ``build_splits.py``.
    category:
        MVTec category name, e.g. ``"bottle"``.
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    split:
        Manifest split key.  Defaults to ``"test"`` (held-out).
    batch_size, num_workers:
        DataLoader parameters.
    image_size:
        Spatial resolution override.  Falls back to
        ``model.arch_spec.input_size`` then 224.
    amp:
        Enable ``torch.autocast`` FP16 inference (CUDA only).
    save_dir:
        Optional directory; writes ``test_metrics.json`` atomically.

    Returns
    -------
    dict
        Full metrics structure — see module docstring for schema.
    """
    splits_dir = Path(splits_dir)
    records    = _load_records(splits_dir, category, split)

    if image_size is None:
        spec       = getattr(model, "arch_spec", None)
        image_size = int(getattr(spec, "input_size", 224)) if spec else 224

    loader = _build_loader(
        records, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )

    is_cuda  = str(device).startswith("cuda") and torch.cuda.is_available()
    device_t = torch.device(device if is_cuda else "cpu")
    use_amp  = amp and is_cuda

    was_training = model.training
    model.eval()
    model.to(device_t)

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            raw = _run_inference(model, loader, device_t, use_amp)
    finally:
        if was_training:
            model.train()
    elapsed = time.perf_counter() - t0

    n_seen    = raw["n_seen"]
    n_normal  = raw["n_normal"]
    n_anomaly = raw["n_anomaly"]
    loss_avg  = (raw["loss_acc"] / raw["loss_n"]) if raw["loss_n"] > 0 else None

    # ---- image-level aggregation ----
    if raw["img_scores"]:
        all_iscores  = np.concatenate(raw["img_scores"]).astype(np.float64)
        all_ilabels  = np.concatenate(raw["img_labels"]).astype(np.int64)
        all_idefects = [d for chunk in raw["img_defects"] for d in chunk]
        img_auroc    = _safe_auroc(all_ilabels, all_iscores)
        img_prauc    = _safe_prauc(all_ilabels, all_iscores)
        clf_metrics  = _best_threshold_metrics(all_ilabels, all_iscores)
        best_thr     = clf_metrics.get("threshold") or 0.0
        stats        = _score_stats(all_iscores)
        normal_stats = _score_stats(all_iscores[all_ilabels == 0])
        anomaly_stats = _score_stats(all_iscores[all_ilabels == 1])
        per_defect   = _per_defect_metrics(
            all_iscores, all_ilabels, all_idefects, best_thr
        )
    else:
        img_auroc = img_prauc = None
        clf_metrics = {k: None for k in
                       ("threshold", "f1", "accuracy", "precision",
                        "recall", "specificity", "tp", "tn", "fp", "fn")}
        best_thr     = 0.0
        stats        = _score_stats(np.array([]))
        normal_stats = anomaly_stats = stats.copy()
        per_defect   = {}

    # ---- pixel-level aggregation ----
    pix_auroc = pix_prauc = None
    pix_clf: dict[str, float | None] = {}
    n_pixels_total = n_pixels_pos = None

    if raw["pix_scores"]:
        all_pscores = np.concatenate(raw["pix_scores"]).astype(np.float64)
        all_plabels = np.concatenate(raw["pix_labels"]).astype(np.int64)
        n_pixels_total = int(all_plabels.size)
        n_pixels_pos   = int(all_plabels.sum())
        if n_pixels_pos > 0:
            pix_auroc = _safe_auroc(all_plabels, all_pscores)
            pix_prauc = _safe_prauc(all_plabels, all_pscores)
            pix_clf   = _best_threshold_metrics(all_plabels, all_pscores)

    LOG.info(
        "[%s] compute_test_metrics(split=%s) — "
        "img_auroc=%s  pr_auc=%s  f1=%s  "
        "pix_auroc=%s  n=%d (norm=%d, anom=%d)  %.2fs",
        category, split,
        f"{img_auroc:.4f}"  if img_auroc is not None else "n/a",
        f"{img_prauc:.4f}"  if img_prauc is not None else "n/a",
        f"{clf_metrics.get('f1', 0):.4f}"
            if clf_metrics.get("f1") is not None else "n/a",
        f"{pix_auroc:.4f}"  if pix_auroc is not None else "n/a",
        n_seen, n_normal, n_anomaly, elapsed,
    )

    metrics: dict[str, Any] = {
        # Identifiers
        "split":    str(split),
        "category": str(category),
        # Counts
        "n_samples":       int(n_seen),
        "n_normal":        int(n_normal),
        "n_anomaly":       int(n_anomaly),
        "n_pixels_total":  n_pixels_total,
        "n_pixels_pos":    n_pixels_pos,
        # Image-level metrics
        "auroc":           img_auroc,
        "image_auroc":     img_auroc,
        "pr_auc":          img_prauc,
        "auprc":           img_prauc,
        # Classification at best threshold
        "f1":              clf_metrics.get("f1"),
        "accuracy":        clf_metrics.get("accuracy"),
        "precision":       clf_metrics.get("precision"),
        "recall":          clf_metrics.get("recall"),
        "specificity":     clf_metrics.get("specificity"),
        "threshold":       clf_metrics.get("threshold"),
        "tp":              clf_metrics.get("tp"),
        "tn":              clf_metrics.get("tn"),
        "fp":              clf_metrics.get("fp"),
        "fn":              clf_metrics.get("fn"),
        # Pixel-level metrics
        "pixel_auroc":     pix_auroc,
        "pixel_pr_auc":    pix_prauc,
        "pixel_f1":        pix_clf.get("f1"),
        "pixel_threshold": pix_clf.get("threshold"),
        "pixel_precision": pix_clf.get("precision"),
        "pixel_recall":    pix_clf.get("recall"),
        # Loss
        "loss":            loss_avg,
        # Score diagnostics
        "score_stats":       stats,
        "normal_score_stats":  normal_stats,
        "anomaly_score_stats": anomaly_stats,
        # Per-defect breakdown
        "per_defect": per_defect,
        # Timing
        "image_size":      int(image_size),
        "elapsed_seconds": round(elapsed, 3),
    }

    if save_dir is not None:
        _write_metrics(metrics, Path(save_dir), category)

    return metrics


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _write_metrics(metrics: dict[str, Any],
                   save_dir: Path,
                   category: str) -> None:
    """Write metrics atomically as JSON under ``save_dir/<category>/``."""
    out_dir = save_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / _METRICS_FILENAME
    tmp  = dest.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(metrics, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(dest)
    LOG.info("Test metrics saved → %s", dest)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for numpy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def load_saved_metrics(save_dir: Path, category: str) -> dict[str, Any]:
    """Reload a previously saved ``test_metrics.json``.

    Parameters
    ----------
    save_dir:
        Same directory that was passed to :func:`compute_test_metrics`.
    category:
        MVTec category name.

    Returns
    -------
    dict
        Parsed metrics dict.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(save_dir) / category / _METRICS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved test metrics not found: {path}. "
            "Run compute_test_metrics with save_dir first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
