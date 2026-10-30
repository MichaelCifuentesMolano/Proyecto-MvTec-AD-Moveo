"""
src/tasks/mvtec.py
==================

MVTec AD anomaly detection as a *task* of the hierarchical methodology.

A task is any object with this interface (duck-typed; RL tasks will provide
the same methods with episode return as ``evaluate``)::

    build(candidate)              -> nn.Module   (unquantized)
    input_shape(candidate)        -> tuple       (batch 1, for profiling)
    train(model, candidate, n_epochs)            (in place; QAT if wrapped)
    calibrate(model, candidate)                  (fake-quant observer ranges)
    evaluate(model, candidate)    -> float       (higher is better)

Performance here is the image-level AUROC on ``eval_split`` (``val_ad`` by
default: the defect-stratified slice of the official test set). The rest of
the official test set is never touched by the search.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.auroc_eval import evaluate_from_splits
from src.evaluation.train_loop import train as train_loop
from src.evaluation.validate import _ValDataset, _load_records
from src.models.model_factory import build_model
from src.quantization.qat_wrapper import FakeQuantize, calibrate

__all__ = ["MVTecTask"]


class MVTecTask:
    name = "mvtec_ad"
    performance_name = "auroc"

    def __init__(self, *, splits_dir: str | Path = "data/splits",
                 category: str = "bottle", eval_split: str = "val_ad",
                 device: str = "cuda", batch_size: int = 8, lr: float = 1e-3,
                 weight_decay: float = 1e-4, calibration_batches: int = 5,
                 seed: int = 0) -> None:
        if eval_split == "test":
            raise ValueError("eval_split='test' would leak the reported test set "
                             "into the search; use 'val_ad'.")
        self.splits_dir = Path(splits_dir)
        self.category = category
        self.eval_split = eval_split
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.calibration_batches = calibration_batches
        self.seed = seed

    @staticmethod
    def _size(candidate: dict[str, Any]) -> int:
        return int(candidate["arch_spec"].get("input_size", 224))

    def build(self, candidate: dict[str, Any]) -> nn.Module:
        return build_model(candidate)

    def input_shape(self, candidate: dict[str, Any]) -> tuple[int, ...]:
        s = self._size(candidate)
        return (1, 3, s, s)

    def train(self, model: nn.Module, candidate: dict[str, Any], n_epochs: int) -> None:
        if n_epochs > 0:
            info = train_loop(
                model, candidate, self.splits_dir, self.category, n_epochs,
                {"name": "adamw", "lr": self.lr, "weight_decay": self.weight_decay},
                {"name": "none"}, self.device, self.seed,
                batch_size=self.batch_size, num_workers=0,
                image_size=self._size(candidate),
            )
            best = info.get("best_val")
            loss = best.get("loss") if isinstance(best, dict) else best
            if loss is not None and not math.isfinite(loss):
                raise RuntimeError(f"training diverged (val loss={loss})")
        self.calibrate(model, candidate)

    def calibrate(self, model: nn.Module, candidate: dict[str, Any]) -> None:
        """Set fake-quant ranges on normal ``val`` images (no-op if unquantized)."""
        if not any(isinstance(m, FakeQuantize) for m in model.modules()):
            return
        records = _load_records(self.splits_dir, self.category, "val")
        n = min(len(records), self.batch_size * self.calibration_batches)
        loader = DataLoader(_ValDataset(records[:n], image_size=self._size(candidate)),
                            batch_size=self.batch_size, shuffle=False, num_workers=0)
        model.to(self.device)
        calibrate(model, loader, n_batches=self.calibration_batches, device=self.device)

    def evaluate(self, model: nn.Module, candidate: dict[str, Any]) -> float:
        result = evaluate_from_splits(
            model=model, splits_dir=self.splits_dir, category=self.category,
            device=self.device, split=self.eval_split,
            batch_size=self.batch_size * 2, num_workers=0,
            image_size=self._size(candidate),
        )
        auroc = result.get("auroc")
        if auroc is None or not math.isfinite(auroc):
            raise RuntimeError(f"AUROC undefined on split {self.eval_split!r}")
        return float(auroc)
