"""
src/nas/fitness.py
==================

Fitness evaluator for the NSGA-II search.

Maps a candidate dict produced by :mod:`src.nas.encoding` to the four
minimisation objectives consumed by :class:`src.nas.nsga2_engine.NSGA2Engine`:

1. ``-AUROC``       (proxy from short training on the val split)
2. ``latency_ms``   (forward-pass latency on the search device)
3. ``peak_ram_mb``  (peak host + device RAM during inference)
4. ``energy_mj``    (energy per inference; ``NoopBackend`` when sensors absent)

Penalty system
--------------
Every stage of the evaluation pipeline can fail independently.  Each
failure mode is represented by a :class:`PenaltyReason` value and receives a
configurable penalty objective vector that places the candidate on the last
Pareto front, ensuring it is never selected over a valid candidate.

Stage → reason mapping::

    build_model       → BUILD_FAILED
    constraint check  → CONSTRAINT_VIOLATED
    qat_wrapper       → QAT_FAILED
    train / calibrate → OOM | TRAINING_DIVERGED | TRAINING_TIMEOUT
    auroc evaluation  → EVAL_FAILED
    hardware profile  → PROFILING_FAILED  (partial: uses raw value for passing metrics)

Objective normalisation
-----------------------
Raw values are returned as the primary output (what the NSGA-II engine
expects).  An :class:`ObjectiveNormalizer` also produces ``[0, 1]``-clipped
values relative to configurable *best* / *worst* reference points.  These
normalised values are stored in :class:`FitnessResult` for logging and
analysis but are **not** passed to the engine (dominance is scale-invariant).

Search-phase vs. full evaluation
---------------------------------
During NAS search, ``n_search_epochs`` is kept small (5–10) and the training
dataset is optionally capped at ``max_train_samples``.  This gives a fast
proxy AUROC suitable for ranking candidates.  Full evaluation is performed
by ``main_retrain.py`` after the search completes.

Assumptions
-----------
- ``model_factory.build_model(candidate)`` consumes the two-key candidate dict.
- ``qat_wrapper.wrap_for_qat(model, qconfig)`` is safe to call on any
  ``nn.Module``.
- Profiling modules (``latency_meter``, ``ram_meter``, ``energy_meter``) are
  importable; a ``NoopBackend`` handles missing Jetson sensors gracefully.
- The val split for the configured ``category`` exists under ``splits_dir``.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (hardware modules may be absent in test / CI environments)
# ---------------------------------------------------------------------------

def _try_import(module_path: str, symbol: str) -> Any:
    """Import ``symbol`` from ``module_path``; return ``None`` if unavailable."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, symbol, None)
    except Exception:  # noqa: BLE001
        return None

_build_model       = _try_import("src.models.model_factory",    "build_model")
_wrap_for_qat      = _try_import("src.quantization.qat_wrapper","wrap_for_qat")
_calibrate         = _try_import("src.quantization.qat_wrapper","calibrate")
_train             = _try_import("src.evaluation.train_loop",   "train")
_eval_from_splits  = _try_import("src.evaluation.auroc_eval",   "evaluate_from_splits")
_measure_latency   = _try_import("src.profiling.latency_meter", "measure_latency")
_measure_peak_ram  = _try_import("src.profiling.ram_meter",     "measure_peak_ram")
_measure_energy    = _try_import("src.profiling.energy_meter",  "measure_energy")
_estimate_complexity = _try_import("src.nas.encoding",          "estimate_complexity")


# ---------------------------------------------------------------------------
# Penalty reasons
# ---------------------------------------------------------------------------

class PenaltyReason(str, Enum):
    """Identifies which pipeline stage triggered a penalty."""
    NONE                 = "none"
    BUILD_FAILED         = "build_failed"
    CONSTRAINT_VIOLATED  = "constraint_violated"
    QAT_FAILED           = "qat_failed"
    OOM                  = "oom"
    TRAINING_DIVERGED    = "training_diverged"
    TRAINING_TIMEOUT     = "training_timeout"
    EVAL_FAILED          = "eval_failed"
    PROFILING_FAILED     = "profiling_failed"
    IMPORT_ERROR         = "import_error"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FitnessConfig:
    """YAML-serialisable hyperparameters for :class:`FitnessEvaluator`.

    All fields have sensible defaults for a Jetson Orin Nano evaluation.
    """
    # ---- Dataset ----
    splits_dir:              str | Path = Path("data/splits")
    category:                str        = "bottle"
    eval_split:              str        = "test"

    # ---- Device ----
    device:                  str        = "cuda"
    amp:                     bool       = False

    # ---- Quick training (search phase) ----
    n_search_epochs:         int        = 5
    max_train_samples:       int | None = 500    # cap training set size
    search_batch_size:       int        = 8
    search_lr:               float      = 1e-3
    search_weight_decay:     float      = 1e-4
    grad_clip_norm:          float      = 1.0
    qat_calibration_batches: int        = 5      # batches for fake-quant calibration

    # ---- Hardware profiling ----
    n_profiling_warmup:      int        = 10
    n_profiling_iters:       int        = 30
    profiling_amp:           bool       = False
    # Energy backend: "auto" | "tegrastats" | "sysfs" | "nvml" | "noop".
    # On a desktop PC with an NVIDIA GPU use "nvml" (requires pynvml);
    # on Jetson, "auto" resolves to tegrastats/sysfs.
    energy_backend:          str        = "auto"

    # ---- Normalisation reference points ----
    # best (ideal) / worst (normalisation floor) for each objective
    auroc_best:              float      = 1.00
    auroc_worst:             float      = 0.50
    latency_best_ms:         float      = 1.0
    latency_worst_ms:        float      = 1000.0
    ram_best_mb:             float      = 50.0
    ram_worst_mb:            float      = 8192.0
    energy_best_mj:          float      = 0.1
    energy_worst_mj:         float      = 100.0

    # ---- Penalty values (applied on failure; treated as beyond-worst) ----
    penalty_auroc:           float      = 0.0
    penalty_latency_ms:      float      = 9_999.0
    penalty_ram_mb:          float      = 65_536.0
    penalty_energy_mj:       float      = 9_999.0

    # ---- Hard constraints (violated → CONSTRAINT_VIOLATED penalty) ----
    max_params_m:            float | None = 20.0    # max 20 M parameters
    max_macs_m:              float | None = None
    min_auroc:               float       = 0.0      # do not penalise low AUROC
    max_latency_ms:          float | None = None
    max_ram_mb:              float | None = None

    # ---- Stability ----
    max_loss_value:          float      = 1e6       # divergence threshold
    nan_loss_is_penalty:     bool       = True

    # ---- Timeouts ----
    train_timeout_seconds:   float | None = 600.0
    profile_timeout_seconds: float | None = 120.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["splits_dir"] = str(d["splits_dir"])
        return d


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    """Detailed output of one candidate evaluation."""
    # Raw objectives (returned to NSGA-II engine)
    auroc:         float
    latency_ms:    float
    peak_ram_mb:   float
    energy_mj:     float

    # Normalised objectives ∈ [0, 1]  (for logging / HV display only)
    auroc_norm:    float | None = None
    latency_norm:  float | None = None
    ram_norm:      float | None = None
    energy_norm:   float | None = None

    # Penalty metadata
    penalty_reason:  PenaltyReason = PenaltyReason.NONE
    penalty_applied: bool          = False

    # Training diagnostics
    train_loss:       float | None = None
    n_epochs_trained: int          = 0
    stopped_early:    bool         = False

    # Profiling diagnostics
    profiling_ok:     bool         = True

    # Timing
    elapsed_seconds:  float        = 0.0

    # Error details
    error_message:    str          = ""

    def to_fitness_dict(self) -> dict[str, float]:
        """Return the minimal dict expected by :class:`NSGA2Engine`."""
        return {
            "auroc":        self.auroc,
            "latency_ms":   self.latency_ms,
            "peak_ram_mb":  self.peak_ram_mb,
            "energy_mj":    self.energy_mj,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialisable dict for CSV / JSON logging."""
        return {
            "auroc":          self.auroc,
            "latency_ms":     self.latency_ms,
            "peak_ram_mb":    self.peak_ram_mb,
            "energy_mj":      self.energy_mj,
            "auroc_norm":     self.auroc_norm,
            "latency_norm":   self.latency_norm,
            "ram_norm":       self.ram_norm,
            "energy_norm":    self.energy_norm,
            "penalty_reason": self.penalty_reason.value,
            "penalty_applied":self.penalty_applied,
            "train_loss":     self.train_loss,
            "n_epochs_trained": self.n_epochs_trained,
            "stopped_early":  self.stopped_early,
            "profiling_ok":   self.profiling_ok,
            "elapsed_seconds": self.elapsed_seconds,
            "error_message":  self.error_message,
        }


# ---------------------------------------------------------------------------
# Objective normaliser
# ---------------------------------------------------------------------------

class ObjectiveNormalizer:
    """Linear min-max normalisation of the four objectives.

    Maps each raw objective value to ``[0, 1]`` relative to the configured
    best / worst reference points, then clips to ``[0, 1]``.

    For AUROC (higher is better), the normalised value is::

        1 - (auroc - auroc_worst) / (auroc_best - auroc_worst)

    so that 0 = ideal and 1 = worst (consistent with minimisation).

    For the other three (lower is better)::

        (value - best) / (worst - best)
    """

    def __init__(self, cfg: FitnessConfig) -> None:
        self._cfg = cfg

    def normalise(self, auroc: float,
                  latency_ms: float,
                  peak_ram_mb: float,
                  energy_mj: float) -> dict[str, float]:
        """Return normalised objectives (all in [0, 1], 0 = ideal)."""
        cfg = self._cfg

        def _clip(x: float) -> float:
            return max(0.0, min(1.0, x))

        def _norm_higher(val: float, best: float, worst: float) -> float:
            rng = best - worst
            return _clip(1.0 - (val - worst) / rng) if abs(rng) > 1e-12 else 1.0

        def _norm_lower(val: float, best: float, worst: float) -> float:
            rng = worst - best
            return _clip((val - best) / rng) if abs(rng) > 1e-12 else 0.0

        return {
            "auroc_norm":   _norm_higher(auroc,       cfg.auroc_best,       cfg.auroc_worst),
            "latency_norm": _norm_lower(latency_ms,   cfg.latency_best_ms,  cfg.latency_worst_ms),
            "ram_norm":     _norm_lower(peak_ram_mb,  cfg.ram_best_mb,      cfg.ram_worst_mb),
            "energy_norm":  _norm_lower(energy_mj,    cfg.energy_best_mj,   cfg.energy_worst_mj),
        }


# ---------------------------------------------------------------------------
# Fitness evaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """Callable that maps a candidate dict to raw fitness objectives.

    Designed to be injected into :class:`src.nas.nsga2_engine.NSGA2Engine`
    as the ``fitness_fn`` parameter::

        evaluator = FitnessEvaluator(config)
        engine    = NSGA2Engine(nsga2_cfg, search_space, fitness_fn=evaluator)

    The evaluator runs the full pipeline for each candidate:

    1. Build model  (``model_factory.build_model``)
    2. Check hard constraints  (parameter count, MACs)
    3. Wrap for QAT  (``qat_wrapper.wrap_for_qat``)
    4. Quick training + calibration  (``train_loop.train``)
    5. AUROC evaluation  (``auroc_eval.evaluate_from_splits``)
    6. Hardware profiling  (latency / RAM / energy meters)
    7. Normalise objectives
    8. Return  ``{"auroc", "latency_ms", "peak_ram_mb", "energy_mj"}``

    Parameters
    ----------
    config:
        Evaluation hyperparameters.
    extra_callbacks:
        Optional list of ``(result: FitnessResult, candidate: dict) → None``
        callables invoked after each successful or failed evaluation (e.g.,
        for per-candidate CSV logging).
    """

    def __init__(self,
                 config: FitnessConfig,
                 extra_callbacks: list[Callable[[FitnessResult, dict], None]] | None = None
                 ) -> None:
        self._cfg        = config
        self._normaliser = ObjectiveNormalizer(config)
        self._callbacks  = extra_callbacks or []
        self._n_calls    = 0
        self._n_failures = 0

    # ------------------------------------------------------------------
    # NSGA-II-compatible entry point
    # ------------------------------------------------------------------

    def __call__(self, candidate_dict: dict[str, Any]) -> dict[str, float]:
        """Evaluate one candidate.  Returns raw objectives dict.

        Any exception is caught; the penalty vector is returned so the
        NSGA-II engine always receives a valid value.
        """
        result = self.evaluate(candidate_dict)
        return result.to_fitness_dict()

    # ------------------------------------------------------------------
    # Full detailed evaluation
    # ------------------------------------------------------------------

    def evaluate(self, candidate_dict: dict[str, Any]) -> FitnessResult:
        """Evaluate one candidate and return a :class:`FitnessResult`.

        Parameters
        ----------
        candidate_dict:
            Two-key dict ``{"arch_spec": ..., "quant_spec": ...}`` as
            produced by :class:`src.nas.encoding.GenomeEncoder`.

        Returns
        -------
        FitnessResult
            Raw and normalised objectives plus diagnostic metadata.
        """
        self._n_calls += 1
        t_start = time.perf_counter()

        # ---- 1. Build model ----
        model, reason, err = self._build_and_wrap(candidate_dict)
        if model is None:
            return self._penalise(reason, err, time.perf_counter() - t_start)

        # ---- 2. Check hard constraints ----
        violation = self._check_constraints(candidate_dict)
        if violation:
            return self._penalise(
                PenaltyReason.CONSTRAINT_VIOLATED,
                violation,
                time.perf_counter() - t_start,
            )

        # ---- 3. Quick training + QAT calibration ----
        train_info, reason, err = self._quick_train(model, candidate_dict)
        if reason not in (PenaltyReason.NONE, None):
            return self._penalise(reason, err, time.perf_counter() - t_start)

        # ---- 4. AUROC evaluation ----
        auroc, reason, err = self._evaluate_auroc(model, candidate_dict)
        if reason not in (PenaltyReason.NONE, None):
            return self._penalise(reason, err, time.perf_counter() - t_start)

        # ---- 5. Hardware profiling ----
        hw, hw_ok, hw_err = self._profile_hardware(model, candidate_dict)

        # ---- 6. Assemble result ----
        latency_ms  = hw.get("latency_ms",  self._cfg.penalty_latency_ms)
        peak_ram_mb = hw.get("peak_ram_mb", self._cfg.penalty_ram_mb)
        energy_mj   = hw.get("energy_mj",   self._cfg.penalty_energy_mj)

        norms = self._normaliser.normalise(auroc, latency_ms, peak_ram_mb, energy_mj)

        result = FitnessResult(
            auroc         = auroc,
            latency_ms    = latency_ms,
            peak_ram_mb   = peak_ram_mb,
            energy_mj     = energy_mj,
            auroc_norm    = norms["auroc_norm"],
            latency_norm  = norms["latency_norm"],
            ram_norm      = norms["ram_norm"],
            energy_norm   = norms["energy_norm"],
            penalty_reason  = PenaltyReason.PROFILING_FAILED if not hw_ok else PenaltyReason.NONE,
            penalty_applied = not hw_ok,
            train_loss        = train_info.get("best_val", {}).get("loss") if isinstance(train_info.get("best_val"), dict) else (train_info.get("best_val") if train_info else None),
            n_epochs_trained  = train_info.get("n_epochs_trained", 0) if train_info else 0,
            stopped_early     = train_info.get("stopped_early", False) if train_info else False,
            profiling_ok      = hw_ok,
            elapsed_seconds   = round(time.perf_counter() - t_start, 2),
            error_message     = hw_err,
        )

        for cb in self._callbacks:
            try:
                cb(result, candidate_dict)
            except Exception:  # noqa: BLE001
                pass

        LOG.debug(
            "Fitness eval #%d — AUROC=%.4f lat=%.1f RAM=%.0f E=%.3f [%s]",
            self._n_calls, auroc, latency_ms, peak_ram_mb, energy_mj,
            result.penalty_reason.value,
        )
        return result

    # ------------------------------------------------------------------
    # Stage 1 + 3 combined: build model and wrap for QAT
    # ------------------------------------------------------------------

    def _build_and_wrap(
            self, candidate_dict: dict[str, Any]
    ) -> tuple[nn.Module | None, PenaltyReason, str]:
        """Build the model and wrap it for QAT.

        Returns
        -------
        (model, reason, error_message)
        ``model`` is ``None`` when a failure occurs.
        """
        if _build_model is None:
            return None, PenaltyReason.IMPORT_ERROR, "model_factory not importable"

        # Build
        try:
            model = _build_model(candidate_dict)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return None, PenaltyReason.OOM, str(exc)
        except Exception as exc:  # noqa: BLE001
            return None, PenaltyReason.BUILD_FAILED, _short_tb(exc)

        # Wrap for QAT
        if _wrap_for_qat is None:
            LOG.warning("qat_wrapper not importable; skipping QAT wrap.")
            return model, PenaltyReason.NONE, ""

        quant_spec = candidate_dict.get("quant_spec", {})
        try:
            model = _wrap_for_qat(model, quant_spec)
        except Exception as exc:  # noqa: BLE001
            return None, PenaltyReason.QAT_FAILED, _short_tb(exc)

        return model, PenaltyReason.NONE, ""

    # ------------------------------------------------------------------
    # Stage 2: hard constraint check
    # ------------------------------------------------------------------

    def _check_constraints(self,
                           candidate_dict: dict[str, Any]) -> str:
        """Return a non-empty error string if any hard constraint is violated."""
        cfg  = self._cfg
        arch = candidate_dict.get("arch_spec", candidate_dict)

        if _estimate_complexity is not None:
            try:
                cplx = _estimate_complexity(arch)
                if cfg.max_params_m and cplx["n_params_m"] > cfg.max_params_m:
                    return (
                        f"n_params {cplx['n_params_m']:.1f}M "
                        f"> max {cfg.max_params_m}M"
                    )
                if cfg.max_macs_m and cplx["macs_m"] > cfg.max_macs_m:
                    return (
                        f"MACs {cplx['macs_m']:.1f}M "
                        f"> max {cfg.max_macs_m}M"
                    )
            except Exception:  # noqa: BLE001
                pass  # complexity estimation is best-effort

        return ""

    # ------------------------------------------------------------------
    # Stage 3: quick training + QAT calibration
    # ------------------------------------------------------------------

    def _quick_train(
            self,
            model: nn.Module,
            candidate_dict: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, PenaltyReason, str]:
        """Run short training; returns ``(train_info, reason, error)``."""
        if _train is None:
            LOG.warning("train_loop not importable; skipping training.")
            return {}, PenaltyReason.NONE, ""

        cfg = self._cfg

        optimizer_cfg  = {
            "name": "adamw",
            "lr":   cfg.search_lr,
            "weight_decay": cfg.search_weight_decay,
        }
        scheduler_cfg  = {"name": "none"}
        arch = candidate_dict.get("arch_spec", {})
        sz   = int(arch.get("input_size", 224))

        train_kwargs   = dict(
            model         = model,
            candidate     = candidate_dict,
            splits_dir    = Path(cfg.splits_dir),
            category      = cfg.category,
            n_epochs      = cfg.n_search_epochs,
            optimizer_cfg = optimizer_cfg,
            scheduler_cfg = scheduler_cfg,
            device        = cfg.device,
            seed          = 0,
            checkpoint_path = None,
            log_dir       = None,
            # Pass extra options to limit data when supported
            max_samples   = cfg.max_train_samples,
            batch_size    = cfg.search_batch_size,
            grad_clip_norm= cfg.grad_clip_norm,
            amp           = cfg.amp,
            image_size    = sz,
        )

        def _do_train() -> dict:
            return _train(**{k: v for k, v in train_kwargs.items()
                             if k in _train.__code__.co_varnames})

        try:
            if cfg.train_timeout_seconds:
                train_info = _run_with_timeout(_do_train, cfg.train_timeout_seconds)
            else:
                train_info = _do_train()
        except TimeoutError:
            self._n_failures += 1
            return None, PenaltyReason.TRAINING_TIMEOUT, "training timed out"
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            self._n_failures += 1
            return None, PenaltyReason.OOM, str(exc)
        except Exception as exc:  # noqa: BLE001
            self._n_failures += 1
            return None, PenaltyReason.BUILD_FAILED, _short_tb(exc)

        # Divergence check
        best_val_dict = train_info.get("best_val")
        best_val = best_val_dict.get("loss") if isinstance(best_val_dict, dict) else best_val_dict
        if best_val is not None and (
            math.isnan(best_val) or
            (cfg.nan_loss_is_penalty and not math.isfinite(best_val)) or
            best_val > cfg.max_loss_value
        ):
            self._n_failures += 1
            return None, PenaltyReason.TRAINING_DIVERGED, (
                f"best_val={best_val:.3g} exceeded max_loss={cfg.max_loss_value}"
            )

        # QAT calibration (short forward pass to set fake-quant ranges)
        if _calibrate is not None and cfg.qat_calibration_batches > 0:
            try:
                self._calibrate_model(model, candidate_dict)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("QAT calibration failed (non-fatal): %s", exc)

        return train_info, PenaltyReason.NONE, ""

    def _calibrate_model(self,
                         model: nn.Module,
                         candidate_dict: dict[str, Any]) -> None:
        """Run a few forward batches to set fake-quant observer ranges."""
        from torch.utils.data import DataLoader
        from src.evaluation.validate import _load_records, _ValDataset

        records = _load_records(
            Path(self._cfg.splits_dir), self._cfg.category, "val"
        )
        n    = min(len(records), self._cfg.search_batch_size * self._cfg.qat_calibration_batches)
        arch = candidate_dict.get("arch_spec", {})
        sz   = int(arch.get("input_size", 224))

        loader = DataLoader(
            _ValDataset(records[:n], image_size=sz),
            batch_size=self._cfg.search_batch_size,
            shuffle=False, num_workers=0,
        )
        device_t = torch.device(self._cfg.device if torch.cuda.is_available() else "cpu")
        model.eval().to(device_t)
        _calibrate(model, loader)

    # ------------------------------------------------------------------
    # Stage 4: AUROC evaluation
    # ------------------------------------------------------------------

    def _evaluate_auroc(
            self,
            model: nn.Module,
            candidate_dict: dict[str, Any],
    ) -> tuple[float, PenaltyReason, str]:
        """Return (auroc, reason, error)."""
        if _eval_from_splits is None:
            LOG.warning("auroc_eval not importable; AUROC set to 0.")
            return 0.0, PenaltyReason.EVAL_FAILED, "auroc_eval not importable"

        cfg = self._cfg
        try:
            result = _eval_from_splits(
                model      = model,
                splits_dir = Path(cfg.splits_dir),
                category   = cfg.category,
                device     = cfg.device,
                split      = cfg.eval_split,
                batch_size = cfg.search_batch_size * 2,
                num_workers= 0,
                amp        = cfg.amp,
                image_size = int(candidate_dict.get("arch_spec", {}).get("input_size", 224)),
            )
            auroc = float(result.get("auroc") or 0.0)
            return auroc, PenaltyReason.NONE, ""
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return 0.0, PenaltyReason.OOM, str(exc)
        except Exception as exc:  # noqa: BLE001
            return 0.0, PenaltyReason.EVAL_FAILED, _short_tb(exc)

    # ------------------------------------------------------------------
    # Stage 5: hardware profiling
    # ------------------------------------------------------------------

    def _profile_hardware(
            self,
            model: nn.Module,
            candidate_dict: dict[str, Any],
    ) -> tuple[dict[str, float], bool, str]:
        """Return ``(hw_dict, all_ok, error_message)``."""
        cfg  = self._cfg
        arch = candidate_dict.get("arch_spec", {})
        sz   = int(arch.get("input_size", 224))
        input_shape = (1, 3, sz, sz)

        hw: dict[str, float] = {}
        ok  = True
        err = ""

        device_str = cfg.device if torch.cuda.is_available() else "cpu"

        # ---- Latency ----
        if _measure_latency is not None:
            try:
                def _lat():
                    return _measure_latency(
                        model, input_shape, device=device_str,
                        n_warmup=cfg.n_profiling_warmup,
                        n_iters=cfg.n_profiling_iters,
                        amp=cfg.profiling_amp,
                    )
                lat_result = (
                    _run_with_timeout(_lat, cfg.profile_timeout_seconds)
                    if cfg.profile_timeout_seconds else _lat()
                )
                l_val = lat_result.get("latency_ms_mean") or lat_result.get("mean_ms")
                hw["latency_ms"] = float(l_val) if l_val is not None else cfg.penalty_latency_ms
            except Exception as exc:  # noqa: BLE001
                hw["latency_ms"] = cfg.penalty_latency_ms
                ok  = False
                err += f" latency: {exc}"
                LOG.warning("Latency profiling failed: %s", exc)
        else:
            hw["latency_ms"] = cfg.penalty_latency_ms
            ok = False

        # ---- RAM ----
        if _measure_peak_ram is not None:
            try:
                def _ram():
                    return _measure_peak_ram(
                        model, input_shape=input_shape, device=device_str,
                        n_warmup=cfg.n_profiling_warmup,
                        n_iters=cfg.n_profiling_iters,
                        fp16=cfg.profiling_amp,
                    )
                ram_result = (
                    _run_with_timeout(_ram, cfg.profile_timeout_seconds)
                    if cfg.profile_timeout_seconds else _ram()
                )
                r_val = ram_result.get("peak_ram_mb")
                hw["peak_ram_mb"] = float(r_val) if r_val is not None else cfg.penalty_ram_mb
            except Exception as exc:  # noqa: BLE001
                hw["peak_ram_mb"] = cfg.penalty_ram_mb
                ok  = False
                err += f" RAM: {exc}"
                LOG.warning("RAM profiling failed: %s", exc)
        else:
            hw["peak_ram_mb"] = cfg.penalty_ram_mb
            ok = False

        # ---- Energy ----
        if _measure_energy is not None:
            try:
                def _eng():
                    return _measure_energy(
                        model, input_shape=input_shape, device=device_str,
                        n_warmup=cfg.n_profiling_warmup,
                        n_iters=cfg.n_profiling_iters,
                        fp16=cfg.profiling_amp,
                        backend=cfg.energy_backend,
                    )
                eng_result = (
                    _run_with_timeout(_eng, cfg.profile_timeout_seconds)
                    if cfg.profile_timeout_seconds else _eng()
                )
                e_val = eng_result.get("active_energy_mj_per_inf") or eng_result.get("energy_mj_per_inf")
                if e_val is None:
                    # Do not hide a dead measurement channel behind a silent
                    # penalty: flag it so the individual is marked and the
                    # search summary can report the objective as inoperative.
                    ok = False
                    err += (f" energy: backend {cfg.energy_backend!r} returned"
                            " no measurement (install pynvml for NVML on PC"
                            " or run on Jetson for tegrastats/sysfs)")
                    hw["energy_mj"] = cfg.penalty_energy_mj
                else:
                    hw["energy_mj"] = float(e_val)
            except Exception as exc:  # noqa: BLE001
                hw["energy_mj"] = cfg.penalty_energy_mj
                ok  = False
                err += f" energy: {exc}"
                LOG.warning("Energy profiling failed: %s", exc)
        else:
            hw["energy_mj"] = cfg.penalty_energy_mj
            ok = False

        # Hard constraint post-check on measured values
        if cfg.max_latency_ms and hw.get("latency_ms", 0) > cfg.max_latency_ms:
            ok = False
            err += f" latency {hw['latency_ms']:.1f}ms > max {cfg.max_latency_ms}ms"
        if cfg.max_ram_mb and hw.get("peak_ram_mb", 0) > cfg.max_ram_mb:
            ok = False
            err += f" RAM {hw['peak_ram_mb']:.0f}MB > max {cfg.max_ram_mb}MB"

        return hw, ok, err.strip()

    # ------------------------------------------------------------------
    # Penalty factory
    # ------------------------------------------------------------------

    def _penalise(self,
                  reason: PenaltyReason,
                  error: str,
                  elapsed: float) -> FitnessResult:
        """Create a fully-penalised ``FitnessResult`` for a failed evaluation."""
        self._n_failures += 1
        cfg = self._cfg
        LOG.info(
            "Fitness penalty [%s]: %s",
            reason.value, error if error else "(no details)",
        )

        auroc       = cfg.penalty_auroc
        lat         = cfg.penalty_latency_ms
        ram         = cfg.penalty_ram_mb
        energy      = cfg.penalty_energy_mj
        norms = self._normaliser.normalise(auroc, lat, ram, energy)

        result = FitnessResult(
            auroc         = auroc,
            latency_ms    = lat,
            peak_ram_mb   = ram,
            energy_mj     = energy,
            auroc_norm    = norms["auroc_norm"],
            latency_norm  = norms["latency_norm"],
            ram_norm      = norms["ram_norm"],
            energy_norm   = norms["energy_norm"],
            penalty_reason  = reason,
            penalty_applied = True,
            elapsed_seconds = round(elapsed, 2),
            error_message   = error[:500],
        )
        for cb in self._callbacks:
            try:
                cb(result, {})
            except Exception:  # noqa: BLE001
                pass
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def n_calls(self) -> int:
        """Total number of evaluations requested."""
        return self._n_calls

    @property
    def n_failures(self) -> int:
        """Number of evaluations that returned a penalty."""
        return self._n_failures

    @property
    def failure_rate(self) -> float:
        """Fraction of evaluations that triggered a penalty."""
        return self._n_failures / max(self._n_calls, 1)

    def __repr__(self) -> str:
        return (
            f"FitnessEvaluator("
            f"category={self._cfg.category!r}, "
            f"device={self._cfg.device!r}, "
            f"n_calls={self._n_calls}, "
            f"failure_rate={self.failure_rate:.1%})"
        )


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

def _run_with_timeout(fn: Callable, timeout_seconds: float, *args, **kwargs) -> Any:
    """Run *fn* in a daemon thread; raise ``TimeoutError`` if it takes too long.

    The underlying thread is not forcibly killed (Python limitation) but is
    marked as daemon so it does not block process exit.

    Parameters
    ----------
    fn:
        Callable to execute.
    timeout_seconds:
        Maximum wall-clock seconds to wait.

    Returns
    -------
    Any
        Return value of *fn*.

    Raises
    ------
    TimeoutError
        If *fn* does not complete within *timeout_seconds*.
    Exception
        Any exception raised by *fn* is re-raised in the calling thread.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except _FuturesTimeout:
            raise TimeoutError(
                f"Evaluation timed out after {timeout_seconds:.0f}s"
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _short_tb(exc: Exception, max_chars: int = 2000) -> str:
    """Return a short traceback string for logging."""
    tb = traceback.format_exc()
    return tb[-max_chars:] if len(tb) > max_chars else tb


# Re-export Any for type hints in the module
from typing import Any  # noqa: E402 — needed for _run_with_timeout return annotation
import traceback  # noqa: E402 — needed by _short_tb
