"""
src/nas/fitness.py
==================

Task-agnostic fitness evaluator for the NSGA-II engine.

One evaluation of a candidate ``{"arch_spec", "quant_spec", "init_state"?}``:

1. ``task.build`` the float model; load ``init_state`` if given (Level 2
   starts from the Level-1 weights of its architecture).
2. ``wrap_for_qat`` with the candidate ``quant_spec`` (no-op for fp16).
3. Deterministic cost on one dummy input: MACs, BOPs, weight memory, params
   (``src.profiling.complexity``). Hard constraint: ``max_params_m``.
4. ``task.train`` for ``n_epochs`` (QAT when quantized) + observer calibration.
5. ``task.evaluate`` -> ``performance`` (higher is better).
6. Optional hardware profiling on the search device: latency (mean, p95,
   std), peak memory of the model (torch allocator) and energy.

Returns a flat dict with every metric (the engine picks its objectives by
name) or ``None`` when a constraint is violated. Exceptions propagate to the
engine, which logs them and assigns the penalty vector.

Latency / energy of *fake-quantized* models on a desktop GPU say nothing about
low-precision hardware (fake-quant runs in fp32), so Level 2 disables
profiling and optimises BOPs / weight memory; real latency and energy are
measured on the Jetson for the finalists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from src.profiling.complexity import measure_complexity
from src.quantization.qat_wrapper import wrap_for_qat

__all__ = ["FitnessConfig", "FitnessEvaluator", "candidate_id"]

LOG = logging.getLogger(__name__)


def candidate_id(candidate: dict[str, Any]) -> str:
    """Stable short id of a candidate (architecture + quantization)."""
    key = {k: candidate.get(k) for k in ("arch_spec", "quant_spec")}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class FitnessConfig:
    device:             str          = "cuda"
    n_epochs:           int          = 5
    profile_hardware:   bool         = True
    n_profiling_warmup: int          = 10
    n_profiling_iters:  int          = 50
    # Energy backend: "auto" | "tegrastats" | "sysfs" | "nvml" | "none".
    # "none" disables the objective explicitly (energy_mj = 0 for everyone).
    # Any other backend that cannot measure aborts at start-up.
    energy_backend:     str          = "none"
    max_params_m:       float | None = 20.0
    save_states_dir:    str | None   = None   # keep trained weights (Level 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FitnessEvaluator:
    """Callable ``candidate -> metrics dict`` for :class:`NSGA2Engine`."""

    def __init__(self, config: FitnessConfig, task: Any) -> None:
        self.cfg = config
        self.task = task
        if not torch.cuda.is_available() and config.device.startswith("cuda"):
            LOG.warning("CUDA not available — evaluating on CPU.")
            self.cfg.device = "cpu"
        self._check_energy_backend()

    def _check_energy_backend(self) -> None:
        """Fail at start-up instead of penalising every candidate's energy."""
        if not self.cfg.profile_hardware or self.cfg.energy_backend == "none":
            return
        from src.profiling.energy_meter import select_backend
        if select_backend(self.cfg.energy_backend, device=self.cfg.device).name == "noop":
            raise RuntimeError(
                f"Energy backend {self.cfg.energy_backend!r} cannot measure power "
                "on this machine (NVML 'Not Supported' on many laptop GPUs). Use "
                "energy_backend: none, or run on the Jetson.")

    def build(self, candidate: dict[str, Any]) -> torch.nn.Module:
        """Float model (+ init_state) wrapped with the candidate quantization."""
        model = self.task.build(candidate)
        if candidate.get("init_state"):
            state = torch.load(candidate["init_state"], map_location="cpu")
            model.load_state_dict(state)
        return wrap_for_qat(model, candidate.get("quant_spec")).to(self.cfg.device)

    def __call__(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        cfg = self.cfg
        model = None
        try:
            model = self.build(candidate)
            shape = self.task.input_shape(candidate)
            fp16 = candidate.get("quant_spec", {}).get("representation") == "fp16"
            out: dict[str, Any] = measure_complexity(
                model, shape, device=cfg.device, float_bits=16 if fp16 else 32)
            if cfg.max_params_m and out["n_params"] > cfg.max_params_m * 1e6:
                LOG.info("Constraint: %.1fM params > %.1fM", out["n_params"] / 1e6,
                         cfg.max_params_m)
                return None

            self.task.train(model, candidate, cfg.n_epochs)
            out["performance"] = self.task.evaluate(model, candidate)
            if not math.isfinite(out["performance"]):
                return None

            if cfg.save_states_dir:  # before profiling: fp16 profiling halves weights
                path = Path(cfg.save_states_dir) / f"{candidate_id(candidate)}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
                out["state_path"] = str(path)

            if cfg.profile_hardware:
                out.update(self._profile(model, shape, fp16))
            return out
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _profile(self, model: torch.nn.Module, shape: tuple[int, ...],
                 fp16: bool) -> dict[str, float]:
        from src.profiling.latency_meter import measure_latency
        from src.profiling.ram_meter import measure_peak_ram

        cfg = self.cfg
        common = dict(device=cfg.device, n_warmup=cfg.n_profiling_warmup,
                      n_iters=cfg.n_profiling_iters, fp16=fp16)
        lat = measure_latency(model, shape, **common)
        ram = measure_peak_ram(model, input_shape=shape, **common)
        hw = {
            "latency_ms": lat["latency_ms_mean"],
            "latency_ms_p95": lat["latency_ms_p95"],
            "latency_ms_std": lat["latency_ms_std"],
            "peak_ram_mb": ram["peak_ram_mb"],
            "energy_mj": 0.0,
        }
        if cfg.energy_backend != "none":
            from src.profiling.energy_meter import measure_energy
            eng = measure_energy(model, input_shape=shape,
                                 backend=cfg.energy_backend, **common)
            e = eng.get("active_energy_mj_per_inf") or eng.get("energy_mj_per_inf")
            if e is None:
                raise RuntimeError(f"energy backend {cfg.energy_backend!r} "
                                   "returned no measurement")
            hw["energy_mj"] = float(e)
        return hw
