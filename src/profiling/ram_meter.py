"""
src/profiling/ram_meter.py
==========================

Memory profiling utilities for the QAT/NAS pipeline.

Measures three complementary memory metrics:

1. **Peak host RAM** (process RSS). Sampled in a background thread so
   short-lived bursts during a forward pass are captured even when the
   target finishes faster than ``sample_hz``.
2. **Peak device VRAM** (CUDA allocator). For PyTorch targets the
   allocator is instrumented directly via ``torch.cuda.reset_peak_memory_stats``
   / ``torch.cuda.max_memory_allocated``. For non-PyTorch targets
   (TensorRT engine, raw callable that talks to CUDA outside PyTorch)
   memory is sampled through NVML when ``pynvml`` is available.
3. **Static model memory footprint** — params + buffers in MB,
   computed without running a forward pass.

Public interface (consumed by ``main_search.py`` / ``main_deploy.py``)
---------------------------------------------------------------------
``measure_peak_ram(target, *, input_shape=(1, 3, 224, 224),
                   device="cuda", n_warmup=5, n_iters=50,
                   sample_hz=100.0, input_factory=None,
                   fp16=False, runner_factory=None) -> dict``

    ``target`` may be one of:

    * an ``nn.Module``  – full instrumentation (most accurate path),
    * a callable ``() -> Any`` – run as-is for ``n_iters`` iterations,
    * a path or string pointing to a serialized engine – delegates engine
      loading to ``runner_factory(engine_path, input_shape, device)``;
      if no factory is given the function tries a lazy TensorRT
      loader and falls back to host-only metrics with a warning.

    Returns at least::

        {
            "peak_ram_mb":        float,   # max(host_peak, device_peak)
            "host_peak_mb":       float,
            "host_baseline_mb":   float,
            "device_peak_mb":     float | None,
            "device_reserved_mb": float | None,
            "device_baseline_mb": float | None,
            "model_total_mb":     float | None,  # only when target is Module
            "model_params_mb":    float | None,
            "model_buffers_mb":   float | None,
            "input_shape":        list[int],
            "device":             str,
            "device_kind":        str,
            "n_iters":            int,
            "sampler":            "psutil" | "none",
            "device_sampler":     "torch" | "pynvml" | "none",
        }

``model_memory_footprint(model: nn.Module) -> dict``
    Static analysis only — no forward pass. Returns
    ``{"params_mb", "buffers_mb", "total_mb", "n_parameters", "dtype_breakdown"}``.

``RamTracker``
    Streaming, context-manager-style tracker for long-running workloads
    (e.g. the closed-loop tracking session). Mirrors the API style of
    :class:`LatencyTimer` from :mod:`latency_meter`.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore
    _HAVE_PSUTIL = False

try:
    import pynvml  # type: ignore
    try:
        pynvml.nvmlInit()
        _HAVE_PYNVML = True
    except Exception:  # noqa: BLE001
        pynvml = None  # type: ignore
        _HAVE_PYNVML = False
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore
    _HAVE_PYNVML = False

__all__ = [
    "measure_peak_ram",
    "model_memory_footprint",
    "RamTracker",
    "tensor_dtype_size_bytes",
]

LOG = logging.getLogger(__name__)

_BYTES_PER_MB: int = 1024 * 1024


# ---------------------------------------------------------------------------
# Static model footprint
# ---------------------------------------------------------------------------
def tensor_dtype_size_bytes(dtype: torch.dtype) -> int:
    """Return the byte width of a torch ``dtype``."""
    try:
        # Available since PyTorch 1.13 — scalar tensor's element size.
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:  # noqa: BLE001
        return 4  # safe default (fp32)


def model_memory_footprint(model: nn.Module) -> dict[str, Any]:
    """Compute parameter + buffer footprint without running the model."""
    param_bytes = 0
    buffer_bytes = 0
    n_params = 0
    dtype_count: Counter[str] = Counter()

    for p in model.parameters():
        nb = p.numel() * p.element_size()
        param_bytes += nb
        n_params += p.numel()
        dtype_count[str(p.dtype)] += 1
    for b in model.buffers():
        buffer_bytes += b.numel() * b.element_size()

    return {
        "params_mb":   round(param_bytes / _BYTES_PER_MB, 4),
        "buffers_mb":  round(buffer_bytes / _BYTES_PER_MB, 4),
        "total_mb":    round((param_bytes + buffer_bytes) / _BYTES_PER_MB, 4),
        "n_parameters": int(n_params),
        "dtype_breakdown": dict(dtype_count),
    }


# ---------------------------------------------------------------------------
# Host RSS background sampler
# ---------------------------------------------------------------------------
class _HostRSSPeakSampler:
    """Track peak RSS of the current process in a daemon thread."""

    def __init__(self, *, sample_hz: float = 100.0, pid: int | None = None):
        self.interval = max(1.0 / max(sample_hz, 1.0), 0.001)
        self.pid = pid or os.getpid()
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not _HAVE_PSUTIL:
            return
        proc = psutil.Process(self.pid)
        self.baseline_bytes = int(proc.memory_info().rss)
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(proc,), daemon=True,
            name="HostRSSPeakSampler",
        )
        self._thread.start()

    def _loop(self, proc) -> None:
        while not self._stop.is_set():
            try:
                rss = int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # One last reading — the thread may have missed a final spike.
        if _HAVE_PSUTIL:
            try:
                rss = int(psutil.Process(self.pid).memory_info().rss)
                if rss > self.peak_bytes:
                    self.peak_bytes = rss
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# NVML peak sampler (used when PyTorch allocator stats aren't available)
# ---------------------------------------------------------------------------
class _NvmlPeakSampler:
    """Sample per-process GPU memory via NVML in a background thread."""

    def __init__(self, *, device_index: int = 0, sample_hz: float = 50.0):
        self.interval = max(1.0 / max(sample_hz, 1.0), 0.005)
        self.device_index = device_index
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return _HAVE_PYNVML

    def _proc_used_bytes(self, handle) -> int:
        """Sum used GPU memory of the current PID's processes on this device."""
        pid = os.getpid()
        total = 0
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in procs:
                if p.pid == pid and getattr(p, "usedGpuMemory", None):
                    total += int(p.usedGpuMemory)
        except Exception:  # noqa: BLE001
            return 0
        return total

    def start(self) -> None:
        if not self.available:
            return
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:  # noqa: BLE001
            return
        self.baseline_bytes = self._proc_used_bytes(handle)
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(handle,), daemon=True,
            name="NvmlPeakSampler",
        )
        self._thread.start()

    def _loop(self, handle) -> None:
        while not self._stop.is_set():
            used = self._proc_used_bytes(handle)
            if used > self.peak_bytes:
                self.peak_bytes = used
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def measure_peak_ram(target: Any,
                     *,
                     input_shape: tuple[int, ...] = (1, 3, 224, 224),
                     device: str = "cuda",
                     n_warmup: int = 5,
                     n_iters: int = 50,
                     sample_hz: float = 100.0,
                     input_factory: Callable[[str], Any] | None = None,
                     fp16: bool = False,
                     runner_factory: Callable[..., Callable[[], Any]]
                     | None = None,
                     **_unused) -> dict[str, Any]:
    """Measure peak host RAM and device VRAM for ``target``.

    See module docstring for the return schema.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    device_kind = _device_kind(device, is_cuda)
    device_index = _cuda_index(device) if is_cuda else 0

    # Resolve target into a callable runner + optional model footprint.
    runner: Callable[[], Any]
    model_footprint: dict[str, Any] | None = None
    use_torch_allocator = False

    if isinstance(target, nn.Module):
        model_footprint = model_memory_footprint(target)
        runner, use_torch_allocator = _make_module_runner(
            target, input_shape, device, fp16=fp16,
            input_factory=input_factory,
        )
    elif callable(target):
        runner = target
        use_torch_allocator = is_cuda  # allocator stats still capture deltas
    elif isinstance(target, (str, Path)):
        runner = _load_engine_runner(
            Path(target), input_shape=input_shape, device=device,
            runner_factory=runner_factory,
        )
        use_torch_allocator = False  # TRT runs outside the PyTorch allocator
    else:
        raise TypeError(
            f"Unsupported target type {type(target).__name__}; expected "
            "nn.Module, callable, or path-like."
        )

    # ----- device baseline + peak instrumentation ----------------------
    device_baseline_bytes: int | None = None
    if is_cuda:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            device_baseline_bytes = int(
                torch.cuda.memory_allocated(device_index)
            )
        except Exception:  # noqa: BLE001
            device_baseline_bytes = None

    # ----- start samplers ----------------------------------------------
    host_sampler = _HostRSSPeakSampler(sample_hz=sample_hz)
    host_sampler.start()

    nvml_sampler: _NvmlPeakSampler | None = None
    device_sampler_label = "none"
    if is_cuda and use_torch_allocator:
        device_sampler_label = "torch"
    elif is_cuda and _HAVE_PYNVML:
        nvml_sampler = _NvmlPeakSampler(
            device_index=device_index,
            sample_hz=min(sample_hz, 50.0),
        )
        nvml_sampler.start()
        device_sampler_label = "pynvml"

    # ----- run target ---------------------------------------------------
    LOG.info(
        "Measuring RAM (device=%s kind=%s) — warmup=%d, iters=%d, "
        "input=%s, host_sampler=%s, device_sampler=%s",
        device, device_kind, n_warmup, n_iters, list(input_shape),
        "psutil" if _HAVE_PSUTIL else "none",
        device_sampler_label,
    )
    try:
        with torch.inference_mode():
            for _ in range(max(0, n_warmup)):
                runner()
            if is_cuda:
                torch.cuda.synchronize()
            for _ in range(n_iters):
                runner()
            if is_cuda:
                torch.cuda.synchronize()
    finally:
        host_sampler.stop()
        if nvml_sampler is not None:
            nvml_sampler.stop()

    # ----- collect device peak -----------------------------------------
    device_peak_bytes: int | None = None
    device_reserved_bytes: int | None = None
    if is_cuda and use_torch_allocator:
        try:
            device_peak_bytes = int(
                torch.cuda.max_memory_allocated(device_index)
            )
            device_reserved_bytes = int(
                torch.cuda.max_memory_reserved(device_index)
            )
        except Exception:  # noqa: BLE001
            device_peak_bytes = None
            device_reserved_bytes = None
    elif nvml_sampler is not None:
        device_peak_bytes = nvml_sampler.peak_bytes or None
        device_baseline_bytes = nvml_sampler.baseline_bytes or device_baseline_bytes

    # ----- report ------------------------------------------------------
    host_peak_mb = (host_sampler.peak_bytes / _BYTES_PER_MB
                    if _HAVE_PSUTIL else float("nan"))
    host_baseline_mb = (host_sampler.baseline_bytes / _BYTES_PER_MB
                        if _HAVE_PSUTIL else float("nan"))
    device_peak_mb = (device_peak_bytes / _BYTES_PER_MB
                      if device_peak_bytes is not None else None)
    device_reserved_mb = (device_reserved_bytes / _BYTES_PER_MB
                          if device_reserved_bytes is not None else None)
    device_baseline_mb = (device_baseline_bytes / _BYTES_PER_MB
                          if device_baseline_bytes is not None else None)

    peak_components = [host_peak_mb if not _is_nan(host_peak_mb) else 0.0]
    if device_peak_mb is not None:
        peak_components.append(device_peak_mb)
    peak_ram_mb = max(peak_components) if peak_components else float("nan")

    report: dict[str, Any] = {
        "peak_ram_mb":        round(peak_ram_mb, 4)
        if not _is_nan(peak_ram_mb) else None,
        "host_peak_mb":       round(host_peak_mb, 4)
        if not _is_nan(host_peak_mb) else None,
        "host_baseline_mb":   round(host_baseline_mb, 4)
        if not _is_nan(host_baseline_mb) else None,
        "device_peak_mb":     round(device_peak_mb, 4)
        if device_peak_mb is not None else None,
        "device_reserved_mb": round(device_reserved_mb, 4)
        if device_reserved_mb is not None else None,
        "device_baseline_mb": round(device_baseline_mb, 4)
        if device_baseline_mb is not None else None,
        "model_total_mb":     model_footprint["total_mb"]
        if model_footprint else None,
        "model_params_mb":    model_footprint["params_mb"]
        if model_footprint else None,
        "model_buffers_mb":   model_footprint["buffers_mb"]
        if model_footprint else None,
        "model_n_parameters": model_footprint["n_parameters"]
        if model_footprint else None,
        "input_shape":        list(input_shape),
        "device":             str(device),
        "device_kind":        device_kind,
        "n_iters":            int(n_iters),
        "n_warmup":           int(n_warmup),
        "fp16":               bool(fp16),
        "sampler":            "psutil" if _HAVE_PSUTIL else "none",
        "device_sampler":     device_sampler_label,
    }
    return report


# ---------------------------------------------------------------------------
# Streaming tracker (parallels LatencyTimer)
# ---------------------------------------------------------------------------
class RamTracker:
    """Context-manager-style memory tracker for long-running workloads.

    Usage::

        tracker = RamTracker(device="cuda")
        with tracker:
            ...                # arbitrary workload
        result = tracker.result()
    """

    def __init__(self,
                 *,
                 device: str = "cuda",
                 sample_hz: float = 50.0) -> None:
        self.device = device
        self.sample_hz = sample_hz
        self._is_cuda = (str(device).startswith("cuda")
                         and torch.cuda.is_available())
        self._device_index = _cuda_index(device) if self._is_cuda else 0
        self._host_sampler = _HostRSSPeakSampler(sample_hz=sample_hz)
        self._nvml_sampler = (_NvmlPeakSampler(device_index=self._device_index,
                                                sample_hz=sample_hz)
                              if self._is_cuda and _HAVE_PYNVML else None)
        self._device_baseline: int | None = None
        self._device_peak: int | None = None
        self._device_reserved_peak: int | None = None
        self._active = False

    def __enter__(self) -> "RamTracker":
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self._device_index)
                self._device_baseline = int(
                    torch.cuda.memory_allocated(self._device_index)
                )
            except Exception:  # noqa: BLE001
                self._device_baseline = None
        self._host_sampler.start()
        if self._nvml_sampler is not None:
            self._nvml_sampler.start()
        self._active = True
        return self

    def __exit__(self, *_args) -> None:
        if not self._active:
            return
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                self._device_peak = int(
                    torch.cuda.max_memory_allocated(self._device_index)
                )
                self._device_reserved_peak = int(
                    torch.cuda.max_memory_reserved(self._device_index)
                )
            except Exception:  # noqa: BLE001
                pass
        self._host_sampler.stop()
        if self._nvml_sampler is not None:
            self._nvml_sampler.stop()
            if (self._device_peak is None and self._nvml_sampler.peak_bytes):
                self._device_peak = self._nvml_sampler.peak_bytes
        self._active = False

    def result(self) -> dict[str, Any]:
        host_peak = self._host_sampler.peak_bytes / _BYTES_PER_MB \
            if _HAVE_PSUTIL else None
        host_baseline = self._host_sampler.baseline_bytes / _BYTES_PER_MB \
            if _HAVE_PSUTIL else None
        device_peak = (self._device_peak / _BYTES_PER_MB
                       if self._device_peak is not None else None)
        device_baseline = (self._device_baseline / _BYTES_PER_MB
                           if self._device_baseline is not None else None)
        device_reserved = (self._device_reserved_peak / _BYTES_PER_MB
                           if self._device_reserved_peak is not None else None)
        peak_candidates = [v for v in (host_peak, device_peak)
                           if v is not None]
        return {
            "peak_ram_mb":        max(peak_candidates) if peak_candidates
            else None,
            "host_peak_mb":       host_peak,
            "host_baseline_mb":   host_baseline,
            "device_peak_mb":     device_peak,
            "device_reserved_mb": device_reserved,
            "device_baseline_mb": device_baseline,
            "device":             self.device,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_module_runner(model: nn.Module,
                        input_shape: tuple[int, ...],
                        device: str,
                        *,
                        fp16: bool,
                        input_factory: Callable[[str], Any] | None
                        ) -> tuple[Callable[[], Any], bool]:
    """Build a ``runner()`` callable for an ``nn.Module``."""
    model.eval()
    model.to(device)
    if fp16:
        model = model.half()

    sample = (input_factory(device) if input_factory is not None
              else torch.randn(
                  *input_shape,
                  dtype=torch.float16 if fp16 else torch.float32,
                  device=device,
              ))

    def _runner() -> Any:
        if isinstance(sample, torch.Tensor):
            return model(sample)
        if isinstance(sample, (list, tuple)):
            return model(*sample)
        if isinstance(sample, dict):
            return model(**sample)
        return model(sample)

    return _runner, True


def _load_engine_runner(engine_path: Path,
                        *,
                        input_shape: tuple[int, ...],
                        device: str,
                        runner_factory: Callable[..., Callable[[], Any]]
                        | None) -> Callable[[], Any]:
    """Resolve a callable runner for a serialized engine path."""
    if runner_factory is not None:
        return runner_factory(engine_path,
                              input_shape=input_shape, device=device)

    # Fallback: idle no-op runner so the function still records a
    # baseline and the file size, with an explicit warning. Callers
    # really should pass a runner_factory or a callable.
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")
    LOG.warning(
        "No runner_factory provided for engine %s — RAM measurement will "
        "only capture baseline memory. Pass runner_factory or wrap the "
        "engine in a callable for accurate metrics.", engine_path,
    )

    def _idle_runner() -> None:
        time.sleep(0.001)

    return _idle_runner


def _device_kind(device: str, is_cuda: bool) -> str:
    if not is_cuda:
        return "cpu"
    if platform.system() == "Linux":
        try:
            text = Path("/proc/device-tree/model").read_text(
                encoding="utf-8", errors="ignore"
            )
            if "jetson" in text.lower() or "tegra" in text.lower():
                return "jetson"
        except OSError:
            pass
    return "cuda"


def _cuda_index(device: str) -> int:
    s = str(device)
    if ":" not in s:
        return 0
    try:
        return int(s.split(":")[1])
    except ValueError:
        return 0


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN never equals itself
    except Exception:  # noqa: BLE001
        return False
