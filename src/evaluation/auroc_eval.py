"""
src/evaluation/auroc_eval.py
============================

AUROC evaluation entry point for anomaly detection models.

Computes:

- **Image-level AUROC / PR-AUC** — one score per image, one label per image
  (0 = normal, 1 = anomaly).
- **Pixel-level AUROC / PR-AUC** (optional) — one score per pixel from the
  model's ``anomaly_map``, matched against ground-truth binary masks supplied
  by the dataloader.  Skipped gracefully when masks are absent or all-zero.

Score extraction follows the same priority cascade used in
:mod:`src.evaluation.validate`:

1. Explicit ``"score"`` tensor in the forward output dict.
2. ``"anomaly_map"`` — max-pooled to obtain an image-level score; the raw map
   is also kept for pixel-level evaluation.
3. ``"recon"`` — per-pixel MSE, max-pooled for image level.

Public interface
----------------
``evaluate_auroc(model, dataloader, device, *, amp=False) -> dict``
    Runs inference over a pre-built DataLoader and returns metrics.

``evaluate_from_splits(model, splits_dir, category, device,
                       split="test", *, batch_size=16,
                       num_workers=2, image_size=None, amp=False) -> dict``
    Builds its own mask-aware DataLoader from a split manifest and delegates
    to :func:`evaluate_auroc`.  Preferred entry point for the test split.

Return schema (all keys always present; optional ones are ``None`` when
undefined)::

    {
        "auroc":          float | None,   # image-level ROC-AUC
        "image_auroc":    float | None,   # alias for "auroc"
        "pixel_auroc":    float | None,   # pixel-level ROC-AUC
        "pr_auc":         float | None,   # image-level PR-AUC
        "auprc":          float | None,   # alias for "pr_auc"
        "pixel_pr_auc":   float | None,   # pixel-level PR-AUC
        "f1_at_best":     float | None,   # image-level F1 at best threshold
        "best_threshold": float | None,
        "n_samples":      int,
        "n_normal":       int,
        "n_anomaly":      int,
        "n_pixels_pos":   int | None,     # anomaly pixels (pixel eval only)
        "n_pixels_total": int | None,
        "split":          str,
        "category":       str,
        "score_stats":    {"mean","std","min","max"},
        "elapsed_seconds": float,
    }
"""

from __future__ import annotations

import json
import logging
import time
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
        f1_score,
    )
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    roc_auc_score = None            # type: ignore
    average_precision_score = None  # type: ignore
    roc_curve = None                # type: ignore
    f1_score = None                 # type: ignore
    _HAVE_SKLEARN = False

__all__ = ["evaluate_auroc", "evaluate_from_splits"]

LOG = logging.getLogger(__name__)

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Mask-aware dataset (used by evaluate_from_splits)
# ---------------------------------------------------------------------------

class _TestDataset(Dataset):
    """ImageNet-normalised image dataset that also loads GT masks."""

    def __init__(self,
                 records: list[dict[str, Any]],
                 *,
                 image_size: int) -> None:
        if not _HAVE_PIL:
            raise RuntimeError("Pillow is required for image loading.")
        self.records   = list(records)
        self.image_size = int(image_size)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec   = self.records[idx]
        ipath = rec.get("abs_path") or rec.get("path")
        if ipath is None:
            raise KeyError(f"Record {idx} has no path field.")

        with _PIL_Image.open(ipath) as img:
            img = img.convert("RGB").resize(
                (self.image_size, self.image_size), _PIL_Image.BILINEAR,
            )
            arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float() / 255.0

        img_t = arr.permute(2, 0, 1).contiguous()
        img_t = (img_t - self._mean) / self._std

        sample: dict[str, Any] = {
            "image": img_t,
            "label": int(rec.get("label", 0)),
            "path":  str(ipath),
        }

        # GT mask: load as binary float32 tensor [1, H, W] or zeros.
        mpath = rec.get("abs_mask") or rec.get("mask")
        if mpath and Path(str(mpath)).is_file():
            with _PIL_Image.open(mpath) as m:
                m = m.convert("L").resize(
                    (self.image_size, self.image_size), _PIL_Image.NEAREST,
                )
                mask_arr = np.asarray(m, dtype=np.uint8)
            mask_t = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

        sample["mask"] = mask_t
        return sample


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


def _build_test_loader(records: list[dict],
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
# Score extraction helpers
# ---------------------------------------------------------------------------

def _image_score(outputs: dict[str, torch.Tensor],
                 inputs: torch.Tensor) -> torch.Tensor | None:
    """Per-image anomaly score, shape [B]."""
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
        return (recon - inputs).pow(2).mean(dim=1, keepdim=True).flatten(1).amax(dim=1)
    return None


def _pixel_map(outputs: dict[str, torch.Tensor],
               inputs: torch.Tensor) -> torch.Tensor | None:
    """Per-pixel anomaly map, shape [B, 1, H, W], resized to input spatial dims."""
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


# ---------------------------------------------------------------------------
# Metric helpers
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
    return _manual_auroc(labels, scores)


def _manual_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    y     = labels[order].astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    tp  = np.cumsum(y)
    fp  = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1.0)])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1.0)])
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
    return _manual_prauc(labels, scores)


def _manual_prauc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve via sorted thresholds."""
    order     = np.argsort(-scores, kind="stable")
    y         = labels[order].astype(np.float64)
    n_pos     = float(y.sum())
    tp        = np.cumsum(y)
    fp        = np.cumsum(1.0 - y)
    # precision and recall at each threshold
    prec      = tp / (tp + fp)
    rec       = tp / max(n_pos, 1.0)
    # prepend (rec=0, prec=1) boundary
    prec_full = np.concatenate([[1.0], prec])
    rec_full  = np.concatenate([[0.0], rec])
    return float(np.trapezoid(prec_full, rec_full))


def _best_f1_threshold(labels: np.ndarray,
                       scores: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, threshold) over the score range.  O(N log N)."""
    if labels.size == 0 or int(labels.sum()) == 0:
        return 0.0, float(np.nanmin(scores)) if scores.size else 0.0
    if _HAVE_SKLEARN and roc_curve is not None:
        try:
            fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
            n_pos = float(labels.sum())
            n_neg = float(labels.size - n_pos)
            tp = tpr_arr * n_pos
            fp = fpr_arr * n_neg
            fn = n_pos - tp
            denom = 2 * tp + fp + fn
            f1s = np.where(denom > 0, 2 * tp / denom, 0.0)
            best_idx = int(np.argmax(f1s))
            return float(f1s[best_idx]), float(thresholds[best_idx])
        except Exception:  # noqa: BLE001
            pass
    # fallback: sweep score quantiles
    thresholds = np.unique(np.percentile(scores, np.linspace(0, 100, 200)))
    best_f1, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        preds  = (scores >= thr).astype(np.int64)
        tp     = int(((preds == 1) & (labels == 1)).sum())
        fp     = int(((preds == 1) & (labels == 0)).sum())
        fn     = int(((preds == 0) & (labels == 1)).sum())
        denom  = 2 * tp + fp + fn
        f1     = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_auroc(model: nn.Module,
                   dataloader: DataLoader,
                   device: str,
                   *,
                   amp: bool = False,
                   split: str = "test",
                   category: str = "",
                   **_unused) -> dict[str, Any]:
    """Run full AUROC/PR-AUC evaluation over a DataLoader.

    Parameters
    ----------
    model:
        Any ``nn.Module`` whose ``forward`` returns a dict following the
        unified schema (or a raw tensor that is wrapped to ``{"recon": ...}``).
    dataloader:
        Yields batches with at least ``"image"`` (float tensor) and
        ``"label"`` (int tensor, 0=normal/1=anomaly).  Batches with a
        ``"mask"`` key (float tensor [B,1,H,W]) enable pixel-level metrics.
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    amp:
        Use ``torch.autocast`` for FP16 inference (GPU only).
    split:
        Label stored in the returned dict (informational).
    category:
        Label stored in the returned dict (informational).

    Returns
    -------
    dict
        See module docstring for the full schema.
    """
    is_cuda  = str(device).startswith("cuda") and torch.cuda.is_available()
    device_t = torch.device(device if is_cuda else "cpu")
    use_amp  = amp and is_cuda

    was_training = model.training
    model.eval()
    model.to(device_t)

    img_score_chunks:  list[np.ndarray] = []
    img_label_chunks:  list[np.ndarray] = []
    pix_score_chunks:  list[np.ndarray] = []
    pix_label_chunks:  list[np.ndarray] = []

    n_seen    = 0
    n_normal  = 0
    n_anomaly = 0
    has_pixel = False

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in dataloader:
                x      = batch["image"].to(device_t, non_blocking=True)
                labels = batch["label"]
                masks  = batch.get("mask")  # [B,1,H,W] float or None

                ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if use_amp else _nullctx())
                with ctx:
                    raw_out = model(x)

                if not isinstance(raw_out, dict):
                    outputs: dict[str, torch.Tensor] = {"recon": raw_out}
                else:
                    outputs = raw_out

                # ---- image-level scores ----
                iscore = _image_score(outputs, x)
                if iscore is not None:
                    img_score_chunks.append(iscore.detach().float().cpu().numpy())
                    img_label_chunks.append(
                        labels.cpu().numpy().astype(np.int64)
                    )

                # ---- pixel-level scores ----
                pmap = _pixel_map(outputs, x)
                if pmap is not None and masks is not None:
                    masks_t = masks.to(device_t, non_blocking=True)
                    # Only include pixels from images that have GT mask data.
                    # We evaluate pixel-level only for anomaly images (label==1).
                    anom_mask_bool = (labels == 1)
                    if anom_mask_bool.any():
                        has_pixel = True
                        pmap_cpu  = pmap.detach().float().cpu()
                        mask_cpu  = masks.float()
                        # Flatten to [B*H*W]
                        for b_idx in range(x.shape[0]):
                            if labels[b_idx].item() == 1 or mask_cpu[b_idx].sum() > 0:
                                p_scores = pmap_cpu[b_idx].flatten().numpy()
                                p_labels = (mask_cpu[b_idx].flatten() > 0.5).numpy().astype(np.int64)
                                pix_score_chunks.append(p_scores)
                                pix_label_chunks.append(p_labels)

                n_seen    += x.shape[0]
                n_normal  += int((labels == 0).sum().item())
                n_anomaly += int((labels == 1).sum().item())

    finally:
        if was_training:
            model.train()

    elapsed = time.perf_counter() - t0

    # ---- image-level metrics ----
    if img_score_chunks:
        all_iscores = np.concatenate(img_score_chunks).astype(np.float64)
        all_ilabels = np.concatenate(img_label_chunks).astype(np.int64)
        score_stats = {
            "mean": float(np.mean(all_iscores)),
            "std":  float(np.std(all_iscores)),
            "min":  float(np.min(all_iscores)),
            "max":  float(np.max(all_iscores)),
        }
        img_auroc  = _safe_auroc(all_ilabels, all_iscores)
        img_prauc  = _safe_prauc(all_ilabels, all_iscores)
        best_f1, best_thr = _best_f1_threshold(all_ilabels, all_iscores)
    else:
        score_stats = {"mean": None, "std": None, "min": None, "max": None}
        img_auroc = img_prauc = None
        best_f1 = best_thr = None

    # ---- pixel-level metrics ----
    n_pixels_pos = n_pixels_total = None
    pix_auroc = pix_prauc = None
    if has_pixel and pix_score_chunks:
        all_pscores = np.concatenate(pix_score_chunks).astype(np.float64)
        all_plabels = np.concatenate(pix_label_chunks).astype(np.int64)
        n_pixels_total = int(all_plabels.size)
        n_pixels_pos   = int(all_plabels.sum())
        pix_auroc = _safe_auroc(all_plabels, all_pscores)
        pix_prauc = _safe_prauc(all_plabels, all_pscores)

    LOG.info(
        "[%s] evaluate_auroc(split=%s) — img_auroc=%s, pix_auroc=%s, "
        "pr_auc=%s, n=%d (norm=%d, anom=%d) in %.2fs",
        category or "?", split,
        f"{img_auroc:.4f}"  if img_auroc  is not None else "n/a",
        f"{pix_auroc:.4f}"  if pix_auroc  is not None else "n/a",
        f"{img_prauc:.4f}"  if img_prauc  is not None else "n/a",
        n_seen, n_normal, n_anomaly, elapsed,
    )

    return {
        "auroc":           img_auroc,
        "image_auroc":     img_auroc,
        "pixel_auroc":     pix_auroc,
        "pr_auc":          img_prauc,
        "auprc":           img_prauc,
        "pixel_pr_auc":    pix_prauc,
        "f1_at_best":      float(best_f1)  if best_f1  is not None else None,
        "best_threshold":  float(best_thr) if best_thr is not None else None,
        "n_samples":       int(n_seen),
        "n_normal":        int(n_normal),
        "n_anomaly":       int(n_anomaly),
        "n_pixels_pos":    n_pixels_pos,
        "n_pixels_total":  n_pixels_total,
        "split":           str(split),
        "category":        str(category),
        "score_stats":     score_stats,
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: build loader from split manifest
# ---------------------------------------------------------------------------

def evaluate_from_splits(model: nn.Module,
                         splits_dir: Path,
                         category: str,
                         device: str,
                         split: str = "test",
                         *,
                         batch_size: int = 16,
                         num_workers: int = 2,
                         image_size: int | None = None,
                         amp: bool = False,
                         **_unused) -> dict[str, Any]:
    """Load the split manifest, build a mask-aware DataLoader, and evaluate.

    This is the preferred entry point for held-out test evaluation.  It
    mirrors the calling convention of :func:`src.evaluation.validate.validate`
    so that ``main_retrain.py`` can call both with the same signature.

    Parameters
    ----------
    model:
        Trained (or QAT-wrapped) ``nn.Module``.
    splits_dir:
        Directory containing per-category ``*.json`` manifests.
    category:
        MVTec category name, e.g. ``"bottle"``.
    device:
        PyTorch device string.
    split:
        Manifest split key, typically ``"test"``.
    batch_size, num_workers:
        DataLoader construction parameters.
    image_size:
        Override spatial resolution.  Defaults to ``model.arch_spec.input_size``
        if present, else 224.
    amp:
        Enable automatic mixed precision during inference.

    Returns
    -------
    dict
        Same schema as :func:`evaluate_auroc`.
    """
    splits_dir = Path(splits_dir)
    records    = _load_records(splits_dir, category, split)

    if image_size is None:
        spec       = getattr(model, "arch_spec", None)
        image_size = int(getattr(spec, "input_size", 224)) if spec else 224

    loader = _build_test_loader(
        records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return evaluate_auroc(
        model, loader, device,
        amp=amp, split=split, category=category,
    )


# ---------------------------------------------------------------------------
# Null context manager (avoids importing contextlib for a trivial shim)
# ---------------------------------------------------------------------------

class _nullctx:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
