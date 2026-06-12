"""
src/evaluation/train_loop.py
============================

Self-contained training loop used by both the NSGA-II low-fidelity proxy
training (``main_search.py`` → ``src.nas.fitness``) and the full-budget
retraining stage (``main_retrain.py``).

Includes
--------
- Dataloader construction from the per-category split manifests written
  by ``src.data.build_splits``.
- Optimizer (``adam``, ``adamw``, ``sgd``) and scheduler (``cosine``,
  ``step``, ``plateau``, ``onecycle``, ``none``) builders driven by plain
  config dicts.
- Loss dispatcher that adapts to the architecture's ``forward`` output
  schema (reconstruction → MSE, feature-based → student/teacher MSE,
  with an optional frozen teacher backbone built lazily on demand).
- Mixed-precision training via ``torch.amp.autocast`` + ``GradScaler``.
- Validation-driven best-checkpoint saving with atomic write.
- Early stopping with configurable patience and minimum-delta.
- Console + CSV + (optional) TensorBoard logging.

Public interface (consumed by ``main_retrain.py``)
--------------------------------------------------
``train(model: nn.Module,
        candidate: dict,
        splits_dir: Path,
        category: str,
        n_epochs: int,
        optimizer_cfg: dict,
        scheduler_cfg: dict | None,
        device: str,
        seed: int,
        checkpoint_path: Path,
        log_dir: Path | None = None,
        **kwargs) -> dict``

Returned dict::

    {
        "best_epoch":       int,
        "best_val":         {"loss": float, "epoch": int},
        "history":          [{"epoch": int, "train_loss": ..., "val_loss":
                              ..., "lr": ...}, ...],
        "checkpoint_path":  str,
        "n_epochs_trained": int,
        "stopped_early":    bool,
        "elapsed_seconds":  float,
    }

Assumptions
-----------
- ``splits_dir / f"{category}.json"`` follows the schema written by
  :mod:`src.data.build_splits`. The training split is the unsupervised
  pool of normal images carved out of MVTec ``train/good``; validation
  uses the held-out slice of the same pool. Test is evaluated separately.
- Model ``forward`` returns a dict with at least one of: ``recon`` (image
  reconstruction), ``features`` (a tensor or a list of multi-scale
  tensors), or ``logits``. The dispatcher picks the appropriate loss.
- The caller has already wrapped the model for QAT (when desired) via
  :func:`src.quantization.qat_wrapper.wrap_for_qat` — this loop never
  toggles quantization state.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

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
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAVE_TB = True
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore
    _HAVE_TB = False

__all__ = ["train"]

LOG = logging.getLogger(__name__)

# ImageNet normalization — works well across MVTec categories and matches
# the expectations of any pretrained teacher backbone we might attach.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Configuration / state
# ---------------------------------------------------------------------------
@dataclass
class _TrainState:
    """Per-run mutable training state."""

    epoch: int = 0
    best_val: float = float("inf")
    best_epoch: int = -1
    epochs_without_improve: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class _ImageDataset(Dataset):
    """Minimal MVTec-AD image dataset reading from a manifest record list."""

    def __init__(self,
                 records: list[dict[str, Any]],
                 *,
                 image_size: int,
                 augment: bool = False) -> None:
        if not _HAVE_PIL:
            raise RuntimeError("Pillow is required for image loading.")
        self.records = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        # Vectorized normalization tensors (computed once).
        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        path = rec.get("abs_path") or rec.get("path")
        if path is None:
            raise KeyError(f"Record {idx} has no path: {rec}")
        with Image.open(path) as img:
            img = img.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR,
            )
            arr = torch.from_numpy(_pil_to_array(img).copy()).float() / 255.0
        img_t = arr.permute(2, 0, 1).contiguous()  # HWC -> CHW
        if self.augment and torch.rand(1).item() < 0.5:
            img_t = torch.flip(img_t, dims=(2,))   # horizontal flip
        img_t = (img_t - self._mean) / self._std
        return {
            "image": img_t,
            "label": int(rec.get("label", 0)),
            "path":  str(path),
        }


def _pil_to_array(img):
    """Convert a PIL image to a NumPy HxWxC array without bringing in numpy at import time."""
    import numpy as np  # local import keeps numpy optional at module-load time
    return np.asarray(img, dtype="uint8")


# ---------------------------------------------------------------------------
# Dataloader / optimizer / scheduler builders
# ---------------------------------------------------------------------------
def _load_manifest(splits_dir: Path, category: str) -> dict[str, Any]:
    path = Path(splits_dir) / f"{category}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. "
            "Run main_prepare.py / build_splits first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataloaders(*,
                       splits_dir: Path,
                       category: str,
                       image_size: int,
                       batch_size: int,
                       num_workers: int,
                       seed: int) -> tuple[DataLoader, DataLoader]:
    manifest = _load_manifest(splits_dir, category)
    train_records = manifest["splits"]["train"]
    val_records = manifest["splits"]["val"]
    if not train_records:
        raise RuntimeError(f"[{category}] empty train split.")
    train_ds = _ImageDataset(train_records, image_size=image_size,
                              augment=True)
    val_ds = _ImageDataset(val_records, image_size=image_size,
                            augment=False)
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(batch_size, len(val_ds) or 1)),
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def _build_optimizer(params: Iterable[torch.nn.Parameter],
                     cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = (cfg.get("name") or "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=wd,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=bool(cfg.get("nesterov", False)),
        )
    raise ValueError(f"unknown optimizer: {name!r}")


def _build_scheduler(optimizer: torch.optim.Optimizer,
                     cfg: dict[str, Any] | None,
                     *,
                     n_epochs: int,
                     n_steps_per_epoch: int):
    if cfg is None:
        return None
    name = (cfg.get("name") or "none").lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("T_max", n_epochs)),
            eta_min=float(cfg.get("eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(1, n_epochs // 3))),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            patience=int(cfg.get("patience", 5)),
            factor=float(cfg.get("factor", 0.5)),
            threshold=float(cfg.get("threshold", 1e-4)),
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("max_lr", 1e-2)),
            total_steps=int(cfg.get("total_steps",
                                     n_epochs * n_steps_per_epoch)),
        )
    raise ValueError(f"unknown scheduler: {name!r}")


# ---------------------------------------------------------------------------
# Loss dispatcher (model-output-aware)
# ---------------------------------------------------------------------------
def _compute_loss(outputs: dict[str, torch.Tensor],
                  inputs: torch.Tensor,
                  *,
                  teacher_features: list[torch.Tensor]
                  | torch.Tensor | None = None) -> torch.Tensor:
    """Return a scalar loss adapted to the model's forward output schema."""
    # 1. Reconstruction-style losses (autoencoder, unet, patch_cnn)
    if "recon" in outputs and isinstance(outputs["recon"], torch.Tensor):
        recon = outputs["recon"]
        if recon.shape != inputs.shape:
            recon = F.interpolate(recon, size=inputs.shape[-2:],
                                  mode="bilinear", align_corners=False)
        return F.mse_loss(recon, inputs)

    # 2. Feature-based losses (student-teacher, feature_recon)
    if "features" in outputs:
        feats = outputs["features"]
        if teacher_features is None:
            # No teacher available — regularize feature norm so optimization
            # has a well-defined direction. Acts as a pure "compactness"
            # objective on the student embedding.
            if isinstance(feats, (list, tuple)):
                return sum(f.pow(2).mean() for f in feats) / max(len(feats), 1)
            return feats.pow(2).mean()
        if isinstance(feats, (list, tuple)):
            tf = (teacher_features
                  if isinstance(teacher_features, (list, tuple))
                  else [teacher_features])
            n = min(len(feats), len(tf))
            losses = [F.mse_loss(_match_size(feats[i], tf[i]), tf[i])
                      for i in range(n)]
            return sum(losses) / max(len(losses), 1)
        tf_single = (teacher_features[0]
                     if isinstance(teacher_features, (list, tuple))
                     else teacher_features)
        return F.mse_loss(_match_size(feats, tf_single), tf_single)

    # 3. Logits-only fallback (treat as autoencoder of logits == 0)
    if "logits" in outputs and isinstance(outputs["logits"], torch.Tensor):
        return outputs["logits"].pow(2).mean()

    raise RuntimeError(
        "Model forward produced none of the expected keys "
        "{'recon', 'features', 'logits'}; loss cannot be computed."
    )


def _match_size(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Spatially align two feature maps before computing a feature loss."""
    if student.shape == teacher.shape:
        return student
    if student.dim() != 4 or teacher.dim() != 4:
        return student
    if student.shape[1] != teacher.shape[1]:
        # Channel mismatch — average-pool the larger one into the smaller's
        # spatial size and let the MSE absorb the residual channel diff.
        pass
    if student.shape[-2:] != teacher.shape[-2:]:
        student = F.adaptive_avg_pool2d(student, teacher.shape[-2:])
    return student


# ---------------------------------------------------------------------------
# Optional teacher hook
# ---------------------------------------------------------------------------
class _TeacherHook:
    """Capture multi-scale features from a frozen torchvision backbone."""

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._buffer: list[torch.Tensor] = []
        self._handles: list = []
        for name in layer_names:
            try:
                mod = self._get_submodule(model, name)
            except AttributeError:
                LOG.warning("Teacher missing layer %s — skipping", name)
                continue
            self._handles.append(mod.register_forward_hook(self._hook))

    @staticmethod
    def _get_submodule(model: nn.Module, dotted: str) -> nn.Module:
        cur = model
        for part in dotted.split("."):
            cur = getattr(cur, part)
        return cur

    def _hook(self, _mod, _inp, out) -> None:
        self._buffer.append(out)

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._buffer.clear()
        with torch.no_grad():
            self.model(x)
        return list(self._buffer)

    def to(self, device: str) -> "_TeacherHook":
        self.model.to(device)
        return self


def _maybe_build_teacher(candidate: dict[str, Any],
                         device: str) -> _TeacherHook | None:
    """Attach a frozen ImageNet ResNet-18 if the architecture needs one."""
    family = ((candidate or {}).get("architecture", {}) or candidate or {}) \
        .get("family", "")
    needs_teacher = family in {"feature_recon", "student_teacher"}
    if not needs_teacher:
        return None
    try:
        from torchvision.models import resnet18, ResNet18_Weights  # type: ignore
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:  # noqa: BLE001
        try:
            from torchvision.models import resnet18  # type: ignore
            backbone = resnet18(pretrained=True)
        except Exception:  # noqa: BLE001
            LOG.warning("Could not load torchvision ResNet-18 — feature loss "
                        "will fall back to feature-norm regularization.")
            return None
    layers = ["layer1", "layer2", "layer3"]
    return _TeacherHook(backbone, layers).to(device)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
class _CsvEpochLog:
    """Append-only per-epoch CSV log with a stable column order."""

    _COLS: tuple[str, ...] = (
        "epoch", "train_loss", "val_loss", "lr",
        "elapsed_seconds", "stopped_early",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=list(self._COLS))
        self._w.writeheader()
        self._fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._w.writerow({c: row.get(c) for c in self._COLS})
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _seed_workers(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _save_best_checkpoint(*,
                          path: Path,
                          model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          epoch: int,
                          val_loss: float,
                          candidate: dict[str, Any]) -> None:
    """Atomically write the best-so-far checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": float(val_loss),
        "candidate": candidate,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train(model: nn.Module,
          candidate: dict[str, Any],
          splits_dir: Path,
          category: str,
          n_epochs: int,
          optimizer_cfg: dict[str, Any],
          scheduler_cfg: dict[str, Any] | None,
          device: str,
          seed: int,
          checkpoint_path: Path | None = None,
          log_dir: Path | None = None,
          *,
          batch_size: int = 16,
          num_workers: int = 2,
          image_size: int | None = None,
          mixed_precision: bool = True,
          early_stop_patience: int = 15,
          early_stop_min_delta: float = 1e-5,
          grad_clip: float | None = 1.0,
          teacher: _TeacherHook | nn.Module | None = None,
          tensorboard: bool = False,
          **_unused) -> dict[str, Any]:
    """Train ``model`` on the given category for up to ``n_epochs``.

    See module docstring for the full return schema. The function is
    safe to call multiple times — each call seeds Python/Torch RNGs,
    reinstantiates the optimizer + scheduler, and overwrites
    ``checkpoint_path`` only when validation improves.
    """
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be >= 1; got {n_epochs}")
    splits_dir = Path(splits_dir)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

    _seed_workers(int(seed))

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    device_t = torch.device(device if is_cuda else "cpu")

    # Resolve image size from candidate when not explicitly provided.
    arch = (candidate or {}).get("architecture", candidate) or {}
    if image_size is None:
        image_size = int(arch.get("input_size", 224))

    train_loader, val_loader = _build_dataloaders(
        splits_dir=splits_dir, category=category,
        image_size=image_size, batch_size=batch_size,
        num_workers=num_workers, seed=int(seed),
    )

    model.to(device_t)
    optimizer = _build_optimizer(model.parameters(), optimizer_cfg)
    scheduler = _build_scheduler(
        optimizer, scheduler_cfg,
        n_epochs=n_epochs, n_steps_per_epoch=max(len(train_loader), 1),
    )
    is_plateau = (
        scheduler_cfg is not None
        and (scheduler_cfg.get("name") or "").lower() == "plateau"
    )
    is_onecycle = (
        scheduler_cfg is not None
        and (scheduler_cfg.get("name") or "").lower() == "onecycle"
    )

    scaler = torch.amp.GradScaler("cuda",
                                  enabled=mixed_precision and is_cuda)

    teacher_hook: _TeacherHook | None = None
    if isinstance(teacher, _TeacherHook):
        teacher_hook = teacher.to(device)
    elif isinstance(teacher, nn.Module):
        teacher_hook = _TeacherHook(teacher,
                                    ["layer1", "layer2", "layer3"]).to(device)
    else:
        teacher_hook = _maybe_build_teacher(candidate, device)

    # Logging setup ------------------------------------------------------
    csv_log: _CsvEpochLog | None = None
    tb: SummaryWriter | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_log = _CsvEpochLog(log_dir / "train_log.csv")
        if tensorboard and _HAVE_TB:
            tb = SummaryWriter(log_dir=str(log_dir / "tb"))

    state = _TrainState()
    t0 = time.perf_counter()
    LOG.info(
        "[%s] start training — n_epochs=%d, batch=%d, image=%d, mp=%s, "
        "device=%s, family=%s, teacher=%s",
        category, n_epochs, batch_size, image_size, mixed_precision,
        device_t, arch.get("family"), teacher_hook is not None,
    )

    try:
        for epoch in range(n_epochs):
            state.epoch = epoch
            train_loss = _run_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                scaler=scaler, device=device_t,
                mixed_precision=mixed_precision and is_cuda,
                teacher_hook=teacher_hook, training=True,
                grad_clip=grad_clip,
                scheduler=(scheduler if is_onecycle else None),
            )
            val_loss = _run_one_epoch(
                model=model, loader=val_loader, optimizer=None,
                scaler=scaler, device=device_t,
                mixed_precision=mixed_precision and is_cuda,
                teacher_hook=teacher_hook, training=False,
                grad_clip=None, scheduler=None,
            )
            lr = optimizer.param_groups[0]["lr"]

            # Step scheduler (cosine/step at epoch granularity, plateau
            # uses val_loss; onecycle stepped per-batch above).
            if scheduler is not None and not is_onecycle:
                if is_plateau:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            elapsed = time.perf_counter() - t0
            state.history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(lr),
                "elapsed_seconds": round(elapsed, 3),
            })
            LOG.info(
                "[%s] epoch %3d/%d — train=%.5f val=%.5f lr=%.2e (%.1fs)",
                category, epoch, n_epochs - 1,
                train_loss, val_loss, lr, elapsed,
            )
            if csv_log is not None:
                csv_log.write({**state.history[-1],
                               "stopped_early": state.stopped_early})
            if tb is not None:
                tb.add_scalar("loss/train", train_loss, epoch)
                tb.add_scalar("loss/val", val_loss, epoch)
                tb.add_scalar("lr", lr, epoch)

            # Best checkpoint --------------------------------------------
            improved = (val_loss < state.best_val - early_stop_min_delta)
            if improved or epoch == 0:
                state.best_val = float(val_loss)
                state.best_epoch = epoch
                state.epochs_without_improve = 0
                if checkpoint_path is not None:
                    _save_best_checkpoint(
                        path=checkpoint_path, model=model, optimizer=optimizer,
                        epoch=epoch, val_loss=val_loss, candidate=candidate,
                    )
            else:
                state.epochs_without_improve += 1
                if state.epochs_without_improve >= early_stop_patience:
                    LOG.info(
                        "[%s] early stop at epoch %d (no improvement for %d).",
                        category, epoch, state.epochs_without_improve,
                    )
                    state.stopped_early = True
                    break

            # Guard against numerical blow-up.
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                LOG.warning("[%s] non-finite loss — aborting training.",
                            category)
                state.stopped_early = True
                break
    finally:
        if csv_log is not None:
            csv_log.close()
        if tb is not None:
            tb.close()

    elapsed = time.perf_counter() - t0
    LOG.info(
        "[%s] done in %.1fs — best_val=%.5f at epoch %d, stopped_early=%s",
        category, elapsed, state.best_val, state.best_epoch,
        state.stopped_early,
    )
    return {
        "best_epoch": state.best_epoch,
        "best_val": {"loss": state.best_val, "epoch": state.best_epoch},
        "history": state.history,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_epochs_trained": state.epoch + 1,
        "stopped_early": bool(state.stopped_early),
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------
def _run_one_epoch(*,
                   model: nn.Module,
                   loader: DataLoader,
                   optimizer: torch.optim.Optimizer | None,
                   scaler: torch.amp.GradScaler,
                   device: torch.device,
                   mixed_precision: bool,
                   teacher_hook: _TeacherHook | None,
                   training: bool,
                   grad_clip: float | None,
                   scheduler) -> float:
    """One pass over ``loader``. Returns the mean loss across batches."""
    model.train(training)
    total_loss = 0.0
    n_seen = 0
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    ctx = (torch.enable_grad() if training else torch.no_grad())
    with ctx:
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(autocast_dev,
                                    enabled=mixed_precision):
                outputs = model(x)
                if not isinstance(outputs, dict):
                    outputs = {"recon": outputs}
                teacher_feats = (teacher_hook(x)
                                 if teacher_hook is not None else None)
                loss = _compute_loss(
                    outputs=outputs, inputs=x,
                    teacher_features=teacher_feats,
                )

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip,
                        )
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            bs = x.shape[0]
            total_loss += float(loss.detach().item()) * bs
            n_seen += bs

    return total_loss / max(n_seen, 1)
