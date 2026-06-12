"""
src/evaluation/validate.py
==========================

Standalone validation entry point. Runs a single forward pass over the
requested split (``"val"`` by default, ``"test"`` for held-out
evaluation) and returns a metrics dict.

Used in two places:

- ``main_retrain.py`` calls :func:`validate` once after the full-budget
  training finishes, on the val split, to record the final reconstruction
  error and (when both classes are present) AUROC.
- It can also be called ad hoc on a checkpointed model — e.g. by an
  early-stopping monitor or by an ablation script — without re-entering
  the training loop.

Design
------
- Dataloader construction is local to this module (mirrors the
  ``FitnessEvaluator`` convention from ``main_search.py`` so callers only
  need to pass ``splits_dir`` and ``category``).
- Loss adapts to the unified ``forward`` output schema produced by
  :mod:`src.models.model_factory`: ``recon`` → MSE; ``features`` →
  feature-norm fallback (the real feature-vs-teacher loss is owned by
  the trainer and is irrelevant for read-only validation).
- AUROC is computed only when the split contains *both* normal and
  anomaly samples; otherwise it is reported as ``None``. This keeps the
  function safe to call on the all-normal ``val`` split without raising.

Public interface (matches the contract pinned in ``main_retrain.py``)
---------------------------------------------------------------------
``validate(model: nn.Module,
           splits_dir: Path,
           category: str,
           device: str,
           split: str = "val",
           **kwargs) -> dict``

    Returns at least::

        {
            "loss":         float,
            "auroc":        float | None,
            "image_auroc":  float | None,
            "n_samples":    int,
            "n_normal":     int,
            "n_anomaly":    int,
            "split":        str,
            "category":     str,
            "score_stats":  {"mean", "std", "min", "max"},
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
    from PIL import Image  # type: ignore
    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    _HAVE_PIL = False

try:
    from sklearn.metrics import roc_auc_score  # type: ignore
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore
    _HAVE_SKLEARN = False

__all__ = ["validate"]

LOG = logging.getLogger(__name__)

# Same normalization as train_loop: matches a frozen ImageNet teacher.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class _ValDataset(Dataset):
    """Read-only image dataset over manifest records."""

    def __init__(self,
                 records: list[dict[str, Any]],
                 *,
                 image_size: int) -> None:
        if not _HAVE_PIL:
            raise RuntimeError("Pillow is required for image loading.")
        self.records = list(records)
        self.image_size = int(image_size)
        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        path = rec.get("abs_path") or rec.get("path")
        if path is None:
            raise KeyError(f"Record {idx} has no path field.")
        with Image.open(path) as img:
            img = img.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR,
            )
            arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).float() \
                / 255.0
        img_t = arr.permute(2, 0, 1).contiguous()
        img_t = (img_t - self._mean) / self._std
        return {
            "image": img_t,
            "label": int(rec.get("label", 0)),
            "path":  str(path),
        }


# ---------------------------------------------------------------------------
# Manifest / loader helpers
# ---------------------------------------------------------------------------
def _load_records(splits_dir: Path,
                  category: str,
                  split: str) -> list[dict[str, Any]]:
    """Read the per-category manifest and pull the requested split."""
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


def _build_loader(records: list[dict[str, Any]],
                  *,
                  image_size: int,
                  batch_size: int,
                  num_workers: int) -> DataLoader:
    bs = max(1, min(batch_size, len(records) or 1))
    return DataLoader(
        _ValDataset(records, image_size=image_size),
        batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Loss + score extractors
# ---------------------------------------------------------------------------
def _compute_loss(outputs: dict[str, torch.Tensor],
                  inputs: torch.Tensor) -> torch.Tensor:
    """MSE-style validation loss adapted to the model's output schema."""
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return F.mse_loss(recon, inputs)
    if "features" in outputs:
        feats = outputs["features"]
        if isinstance(feats, (list, tuple)):
            return sum(f.pow(2).mean() for f in feats) / max(len(feats), 1)
        return feats.pow(2).mean()
    if "logits" in outputs and isinstance(outputs["logits"], torch.Tensor):
        return outputs["logits"].pow(2).mean()
    raise RuntimeError(
        "Cannot compute loss: model forward returned none of "
        "{'recon', 'features', 'logits'}."
    )


def _per_sample_score(outputs: dict[str, torch.Tensor],
                      inputs: torch.Tensor) -> torch.Tensor | None:
    """Return a per-sample anomaly score tensor of shape ``[B]``.

    Priority: explicit ``score`` > ``anomaly_map`` (max-pooled) >
    ``recon`` (per-pixel MSE → max). For pure feature-only outputs we
    cannot derive a score without a teacher, so return ``None``.
    """
    if "score" in outputs and isinstance(outputs["score"], torch.Tensor):
        s = outputs["score"]
        return s.flatten() if s.dim() <= 1 else s.flatten(1).amax(dim=1)
    if "anomaly_map" in outputs and isinstance(
            outputs["anomaly_map"], torch.Tensor):
        m = outputs["anomaly_map"]
        return m.flatten(1).amax(dim=1)
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        per_pixel = (recon - inputs).pow(2).mean(dim=1, keepdim=True)
        return per_pixel.flatten(1).amax(dim=1)
    return None


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------
def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute AUROC robustly. Returns ``None`` when undefined."""
    if labels.size == 0 or scores.size == 0:
        return None
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    if not np.isfinite(scores).all():
        # NaNs or infs in scores → AUROC is meaningless.
        return None
    if _HAVE_SKLEARN:
        try:
            return float(roc_auc_score(labels, scores))
        except Exception:  # noqa: BLE001
            pass
    return _manual_auroc(labels, scores)


def _manual_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoid integration of the ROC curve (no scipy/sklearn)."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order].astype(np.float64)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1.0)])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1.0)])
    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate(model: nn.Module,
             splits_dir: Path,
             category: str,
             device: str,
             split: str = "val",
             *,
             batch_size: int = 16,
             num_workers: int = 2,
             image_size: int | None = None,
             **_unused) -> dict[str, Any]:
    """Run a single forward pass and return a validation metrics dict.

    See module docstring for the full return schema.
    """
    splits_dir = Path(splits_dir)
    records = _load_records(splits_dir, category, split)

    # Resolve image size: prefer explicit kwarg, then model.arch_spec,
    # then 224 as a safe ImageNet default.
    if image_size is None:
        spec = getattr(model, "arch_spec", None)
        image_size = int(getattr(spec, "input_size", 224)) if spec else 224

    loader = _build_loader(
        records, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    device_t = torch.device(device if is_cuda else "cpu")

    was_training = model.training
    model.eval()
    model.to(device_t)

    total_loss = 0.0
    n_seen = 0
    n_normal = 0
    n_anomaly = 0
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in loader:
                x = batch["image"].to(device_t, non_blocking=True)
                labels = batch["label"]
                outputs = model(x)
                if not isinstance(outputs, dict):
                    outputs = {"recon": outputs}
                try:
                    loss = _compute_loss(outputs, x)
                    total_loss += float(loss.item()) * x.shape[0]
                except RuntimeError as exc:
                    LOG.warning("Loss computation skipped: %s", exc)
                n_seen += x.shape[0]

                scores = _per_sample_score(outputs, x)
                if scores is not None:
                    score_chunks.append(
                        scores.detach().float().cpu().numpy()
                    )
                    label_chunks.append(
                        labels.detach().cpu().numpy().astype(np.int64)
                    )
                n_normal += int((labels == 0).sum().item())
                n_anomaly += int((labels == 1).sum().item())
    finally:
        if was_training:
            model.train()

    mean_loss = total_loss / max(n_seen, 1) if n_seen else float("nan")

    if score_chunks:
        all_scores = np.concatenate(score_chunks)
        all_labels = np.concatenate(label_chunks)
        score_stats = {
            "mean": float(np.mean(all_scores)),
            "std":  float(np.std(all_scores)),
            "min":  float(np.min(all_scores)),
            "max":  float(np.max(all_scores)),
        }
        auroc = _safe_auroc(all_labels, all_scores)
    else:
        score_stats = {"mean": None, "std": None, "min": None, "max": None}
        auroc = None

    elapsed = time.perf_counter() - t0
    LOG.info(
        "[%s] validate(split=%s) — loss=%.5f, AUROC=%s, n=%d (norm=%d, "
        "anom=%d) in %.2fs",
        category, split, mean_loss,
        f"{auroc:.4f}" if auroc is not None else "n/a",
        n_seen, n_normal, n_anomaly, elapsed,
    )
    return {
        "loss":            float(mean_loss),
        "auroc":           auroc,
        "image_auroc":     auroc,        # alias for downstream callers
        "n_samples":       int(n_seen),
        "n_normal":        int(n_normal),
        "n_anomaly":       int(n_anomaly),
        "split":           str(split),
        "category":        str(category),
        "image_size":      int(image_size),
        "score_stats":     score_stats,
        "elapsed_seconds": round(elapsed, 3),
    }
