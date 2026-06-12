"""
src/profiling/latency_meter.py
==============================

Inference-latency measurement utilities for the QAT/NAS pipeline.

Provides two complementary entry points:

1. :func:`measure_latency` — one-shot benchmark of a PyTorch ``nn.Module``
   on a target device. Used by ``src.nas.fitness`` (search stage) and
   ``main_retrain.py`` to obtain ``latency_ms_*`` and ``throughput_fps``
   statistics with warmup, CUDA-event-based timing, optional FP16, and
   deterministic settings.

2. :class:`LatencyTimer` — streaming, start/stop timer used by
   ``main_tracking.py`` for per-frame measurement during the closed-loop
   tracking task. Aggregates samples into a stats dict with ``mean``,
   ``p50``, ``p95``, ``p99``, ``std``, ``min``, ``max``, ``n``.

Devices supported
-----------------
- ``"cuda"`` / ``"cuda:N"`` — desktop GPU or Jetson (Orin Nano, Xavier,
  ...). CUDA event timing is used when ``use_cuda_events=True`` to avoid
  Python clock noise.
- ``"cpu"`` — uses ``time.perf_counter`` with thread / OMP knobs honored
  by the caller.
- The Jetson family is detected via ``/proc/device-tree/model``; when
  found, the device name is included in the report as ``"jetson"``.

Public interface
----------------
``measure_latency(model: nn.Module,
                  input_shape: tuple[int, ...] = (1, 3, 224, 224),
                  device: str = "cuda",
                  *,
                  n_warmup: int = 20,
                  n_iters: int = 200,
                  use_cuda_events: bool = True,
                  fp16: bool = False,
                  amp: bool = False,
                  input_factory: Callable | None = None) -> dict``

Returned dict (always present, NaN/None where not applicable)::

    {
        "latency_ms_mean": float,
        "latency_ms_p50":  float,
        "latency_ms_p95":  float,
        "latency_ms_p99":  float,
        "latency_ms_std":  float,
        "latency_ms_min":  float,
        "latency_ms_max":  float,
        "throughput_fps":  float,           # batch / mean_latency_s
        "n_iters":         int,
        "n_warmup":        int,
        "input_shape":     list[int],
        "device":          str,
        "device_kind":     "cuda" | "cpu" | "jetson",
        "device_name":     str,              # GPU / CPU / Jetson model
        "fp16":            bool,
        "amp":             bool,
        "synchronized":    bool,
    }

Assumptions
-----------
- The model's ``forward`` accepts a single ``Tensor`` argument by default;
  override with ``input_factory(device) -> Tensor | tuple | dict`` for
  models that need richer signatures.
- The caller is responsible for setting up power mode, fan curve, or
  deterministic CPU pinning on Jetson; this module only measures.
"""

from __future__ import annotations

import logging
import platform
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "measure_latency",
    "LatencyTimer",
    "time_block",
    "detect_device_info",
]

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _is_jetson() -> bool:
    """Detect a Jetson host via the device-tree model node (Linux only)."""
    if platform.system() != "Linux":
        return False
    try:
        text = Path("/proc/device-tree/model").read_text(
            encoding="utf-8", errors="ignore"
        )
        return "jetson" in text.lower() or "tegra" in text.lower()
    except OSError:
        return False


def detect_device_info(device: str) -> dict[str, Any]:
    """Return a small descriptor of the chosen execution device."""
    info: dict[str, Any] = {
        "device": str(device),
        "device_kind": "cpu",
        "device_name": platform.processor() or platform.machine(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in str(device):
            try:
                idx = int(str(device).split(":")[1])
            except ValueError:
                idx = 0
        info["device_kind"] = "jetson" if _is_jetson() else "cuda"
        info["device_name"] = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["cuda_index"] = idx
        try:
            info["device_total_memory_mb"] = round(
                torch.cuda.get_device_properties(idx).total_memory
                / (1024 ** 2), 1,
            )
        except Exception:  # noqa: BLE001
            pass
    return info


# ---------------------------------------------------------------------------
# One-shot benchmark
# ---------------------------------------------------------------------------
def measure_latency(model: nn.Module,
                    input_shape: tuple[int, ...] = (1, 3, 224, 224),
                    device: str = "cuda",
                    *,
                    n_warmup: int = 20,
                    n_iters: int = 200,
                    use_cuda_events: bool = True,
                    fp16: bool = False,
                    amp: bool = False,
                    input_factory: Callable[[str],
                                            torch.Tensor | tuple | dict]
                    | None = None,
                    **_unused) -> dict[str, Any]:
    """Benchmark a model's forward latency. See module docstring for the
    full return schema.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")
    n_warmup = max(0, int(n_warmup))

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not is_cuda and use_cuda_events:
        use_cuda_events = False
    if not is_cuda and (fp16 or amp):
        # FP16 / AMP only meaningful on CUDA for inference timing.
        fp16 = False
        amp = False

    # Prepare model -------------------------------------------------------
    was_training = model.training
    model.eval()
    model.to(device)
    if fp16:
        model = model.half()

    # Build a dummy input ------------------------------------------------
    try:
        sample = _build_input(input_shape, device, fp16=fp16,
                              factory=input_factory)
    except Exception as exc:  # noqa: BLE001
        if was_training:
            model.train()
        raise RuntimeError(
            f"Failed to construct input for shape {input_shape} on "
            f"{device}: {exc}"
        ) from exc

    fwd = _make_fwd(model, sample, amp=amp and is_cuda)

    info = detect_device_info(device)
    LOG.info(
        "Measuring latency on %s (%s) — input=%s, warmup=%d, iters=%d, "
        "fp16=%s, amp=%s",
        info["device_kind"], info.get("device_name", "?"),
        list(input_shape), n_warmup, n_iters, fp16, amp,
    )

    # Warmup --------------------------------------------------------------
    try:
        with torch.inference_mode():
            for _ in range(n_warmup):
                fwd()
            if is_cuda:
                torch.cuda.synchronize()

            # Timed loop --------------------------------------------------
            samples_ms = (
                _time_with_cuda_events(fwd, n_iters)
                if use_cuda_events
                else _time_with_perf_counter(fwd, n_iters,
                                             synchronize=is_cuda)
            )
    finally:
        if was_training:
            model.train()

    return _build_report(
        samples_ms=samples_ms,
        n_iters=n_iters,
        n_warmup=n_warmup,
        input_shape=input_shape,
        info=info,
        fp16=fp16,
        amp=amp,
        synchronized=is_cuda,
    )


# ---------------------------------------------------------------------------
# Streaming timer (used by main_tracking)
# ---------------------------------------------------------------------------
class LatencyTimer:
    """Streaming start/stop timer with rolling statistics.

    Use either as ``timer.start(); ...; timer.stop()`` or via the
    :func:`time_block` context manager. Sample units are milliseconds.
    """

    def __init__(self,
                 name: str = "latency",
                 *,
                 use_cuda_events: bool = False,
                 device: str = "cuda") -> None:
        self.name = name
        self.device = device
        self.use_cuda_events = (
            bool(use_cuda_events)
            and torch.cuda.is_available()
            and str(device).startswith("cuda")
        )
        self._samples: list[float] = []
        self._start_perf: float | None = None
        self._evt_start: torch.cuda.Event | None = None
        self._evt_end: torch.cuda.Event | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.use_cuda_events:
            self._evt_start = torch.cuda.Event(enable_timing=True)
            self._evt_end = torch.cuda.Event(enable_timing=True)
            self._evt_start.record()
        else:
            self._start_perf = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and append the elapsed milliseconds to the buffer."""
        if self.use_cuda_events:
            if self._evt_start is None or self._evt_end is None:
                raise RuntimeError("LatencyTimer.stop() called before start()")
            self._evt_end.record()
            torch.cuda.synchronize()
            dt_ms = float(self._evt_start.elapsed_time(self._evt_end))
            self._evt_start = self._evt_end = None
        else:
            if self._start_perf is None:
                raise RuntimeError("LatencyTimer.stop() called before start()")
            dt_ms = (time.perf_counter() - self._start_perf) * 1000.0
            self._start_perf = None
        self._samples.append(dt_ms)
        return dt_ms

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, float | int]:
        if not self._samples:
            return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                    "p95": float("nan"), "p99": float("nan"),
                    "std": float("nan"), "min": float("nan"),
                    "max": float("nan")}
        return _summarize_samples(self._samples)

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        mean_ms = statistics.fmean(self._samples)
        return 1000.0 / mean_ms if mean_ms > 0 else 0.0

    @property
    def n(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        self._samples.clear()
        self._start_perf = None
        self._evt_start = None
        self._evt_end = None

    # Convenience: record an externally-timed sample (e.g. from a remote
    # benchmark that already returned milliseconds).
    def add_sample(self, dt_ms: float) -> None:
        self._samples.append(float(dt_ms))


@contextmanager
def time_block(timer: LatencyTimer):
    """Context manager wrapper: ``with time_block(timer): ...``."""
    timer.start()
    try:
        yield
    finally:
        timer.stop()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _build_input(input_shape: tuple[int, ...],
                 device: str,
                 *,
                 fp16: bool,
                 factory: Callable | None) -> Any:
    """Materialize a dummy input from a shape or a user-supplied factory."""
    if factory is not None:
        x = factory(device)
        if isinstance(x, torch.Tensor) and fp16:
            x = x.half()
        return x
    dtype = torch.float16 if fp16 else torch.float32
    return torch.randn(*input_shape, dtype=dtype, device=device)


def _make_fwd(model: nn.Module,
              sample: Any,
              *,
              amp: bool) -> Callable[[], Any]:
    """Return a callable that runs one forward pass, honoring AMP."""
    if amp:
        autocast_dev = "cuda"

        def _fwd() -> Any:
            with torch.amp.autocast(autocast_dev, enabled=True):
                return _invoke(model, sample)
        return _fwd
    return lambda: _invoke(model, sample)


def _invoke(model: nn.Module, sample: Any) -> Any:
    if isinstance(sample, torch.Tensor):
        return model(sample)
    if isinstance(sample, (list, tuple)):
        return model(*sample)
    if isinstance(sample, dict):
        return model(**sample)
    return model(sample)


def _time_with_cuda_events(fwd: Callable[[], Any],
                           n_iters: int) -> list[float]:
    """High-precision per-iteration timing via CUDA events."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fwd()
        ends[i].record()
    torch.cuda.synchronize()
    return [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]


def _time_with_perf_counter(fwd: Callable[[], Any],
                            n_iters: int,
                            *,
                            synchronize: bool) -> list[float]:
    """Per-iteration timing via ``time.perf_counter`` (CPU or GPU)."""
    samples: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fwd()
        if synchronize:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _summarize_samples(samples: list[float]) -> dict[str, float | int]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n":   int(arr.size),
        "mean": float(arr.mean()),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "std":  float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


def _build_report(*,
                  samples_ms: list[float],
                  n_iters: int,
                  n_warmup: int,
                  input_shape: tuple[int, ...],
                  info: dict[str, Any],
                  fp16: bool,
                  amp: bool,
                  synchronized: bool) -> dict[str, Any]:
    s = _summarize_samples(samples_ms)
    batch = int(input_shape[0]) if input_shape else 1
    fps = (1000.0 * batch / s["mean"]) if s["mean"] > 0 else 0.0
    return {
        "latency_ms_mean": s["mean"],
        "latency_ms_p50":  s["p50"],
        "latency_ms_p95":  s["p95"],
        "latency_ms_p99":  s["p99"],
        "latency_ms_std":  s["std"],
        "latency_ms_min":  s["min"],
        "latency_ms_max":  s["max"],
        "throughput_fps":  float(fps),
        "n_iters":         int(n_iters),
        "n_warmup":        int(n_warmup),
        "input_shape":     list(input_shape),
        "fp16":            bool(fp16),
        "amp":             bool(amp),
        "synchronized":    bool(synchronized),
        **info,
    }
