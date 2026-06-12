"""
src/profiling/energy_meter.py
=============================

Power and energy profiling for the QAT/NAS pipeline.

Measures:

* **Average power** (W) over a workload window.
* **Energy per inference** (mJ) — total integrated energy ÷ ``n_iters``.
* **Idle vs. active delta** — by sampling power without the workload
  immediately before / after the timed run, the meter reports
  ``avg_idle_power_w`` and ``active_delta_w`` so the workload's
  *incremental* energy can be separated from system baseline draw.

Sources (auto-selected; can be forced via ``backend=``)
-------------------------------------------------------
1. ``"tegrastats"`` — Jetson native. Spawns ``tegrastats`` and parses
   ``VDD_*`` / ``POM_*`` rails (``mW``); prefers a single "total" rail
   (``VDD_IN``, ``POM_5V_IN``, ``VDD_TOTAL``) and falls back to the sum
   of GPU/CPU/SOC/DDR component rails.
2. ``"sysfs"`` — Reads INA3221 / hwmon power channels directly from
   ``/sys`` (works on most Jetsons even when tegrastats is missing).
3. ``"nvml"`` — Desktop CUDA GPUs (and recent JetPacks): uses
   ``pynvml.nvmlDeviceGetPowerUsage``.
4. ``"noop"`` — Always available, returns ``None`` energy / power.
   Keeps callers crash-free on hosts without sensors.

Public interface (consumed by ``main_search.py``, ``main_deploy.py``,
``main_tracking.py``)
---------------------------------------------------------------------
``measure_energy(target, *, n_iters=200, n_warmup=20, device="cuda",
                 input_shape=(1, 3, 224, 224), sample_hz=10.0,
                 input_factory=None, fp16=False,
                 runner_factory=None, backend="auto",
                 include_idle_baseline=True,
                 idle_seconds=2.0) -> dict``

    Returns at least::

        {
            "energy_mj":          float | None,   # integrated active+idle
            "active_energy_mj":   float | None,   # idle-corrected
            "avg_power_w":        float | None,
            "avg_idle_power_w":   float | None,
            "active_delta_w":     float | None,
            "duration_s":         float,
            "n_samples":          int,
            "n_iters":            int,
            "energy_mj_per_inf":  float | None,
            "active_energy_mj_per_inf": float | None,
            "min_power_w":        float | None,
            "max_power_w":        float | None,
            "source":             "tegrastats" | "sysfs" | "nvml" | "noop",
            "device":             str,
            "device_kind":        str,
        }

``EnergyMeter(*, device="cuda", sample_hz=10.0, backend="auto")``
    Context manager for streaming workloads (e.g. tracking sessions)::

        with EnergyMeter(device="cuda") as em:
            ...                # arbitrary workload
        result = em.stop()
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

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
    "measure_energy",
    "EnergyMeter",
    "EnergyBackend",
    "TegrastatsBackend",
    "SysfsBackend",
    "NvmlBackend",
    "NoopBackend",
    "select_backend",
]

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class EnergyBackend(ABC):
    """Abstract sampling backend.

    Concrete backends append ``(timestamp_s, power_w)`` to ``self.samples``
    while the workload runs; ``stop()`` integrates them via the trapezoid
    rule and returns a result dict.
    """

    name: str = "base"

    def __init__(self, *, sample_hz: float = 10.0) -> None:
        self.sample_hz = max(float(sample_hz), 1.0)
        self.samples: list[tuple[float, float]] = []
        self._t_start: float = 0.0
        self._t_stop: float = 0.0

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> dict[str, Any]: ...

    # ------------------------------------------------------------------
    def _build_result(self) -> dict[str, Any]:
        duration_s = max(self._t_stop - self._t_start, 0.0)
        if not self.samples:
            return {
                "energy_mj":   None,
                "avg_power_w": None,
                "min_power_w": None,
                "max_power_w": None,
                "duration_s":  duration_s,
                "n_samples":   0,
                "source":      self.name,
            }
        ts = [s[0] for s in self.samples]
        ps = [s[1] for s in self.samples]
        if len(self.samples) >= 2:
            energy_j = 0.0
            for i in range(1, len(ts)):
                dt = ts[i] - ts[i - 1]
                if dt < 0:
                    continue
                energy_j += 0.5 * (ps[i] + ps[i - 1]) * dt
        else:
            energy_j = ps[0] * duration_s
        avg_p = energy_j / duration_s if duration_s > 0 else ps[0]
        return {
            "energy_mj":   float(energy_j * 1000.0),
            "avg_power_w": float(avg_p),
            "min_power_w": float(min(ps)),
            "max_power_w": float(max(ps)),
            "duration_s":  duration_s,
            "n_samples":   len(self.samples),
            "source":      self.name,
        }


class _PolledBackend(EnergyBackend):
    """Common loop for backends that read a scalar power on demand."""

    def __init__(self, *, sample_hz: float = 10.0) -> None:
        super().__init__(sample_hz=sample_hz)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    @abstractmethod
    def _read_power_w(self) -> float | None: ...

    def start(self) -> None:
        self.samples.clear()
        self._stop_evt.clear()
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"{self.name}_sampler",
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._t_stop = time.perf_counter()
        return self._build_result()

    def _loop(self) -> None:
        interval = 1.0 / self.sample_hz
        while not self._stop_evt.is_set():
            t = time.perf_counter() - self._t_start
            p = self._read_power_w()
            if p is not None:
                self.samples.append((t, p))
            self._stop_evt.wait(interval)


class NvmlBackend(_PolledBackend):
    """Read GPU power via NVML (``nvmlDeviceGetPowerUsage`` returns mW)."""

    name = "nvml"

    def __init__(self,
                 *,
                 sample_hz: float = 10.0,
                 device_index: int = 0) -> None:
        super().__init__(sample_hz=sample_hz)
        self.device_index = device_index
        self._handle = None
        if _HAVE_PYNVML:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:  # noqa: BLE001
                self._handle = None

    @classmethod
    def is_available(cls) -> bool:
        if not _HAVE_PYNVML:
            return False
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(handle)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _read_power_w(self) -> float | None:
        if self._handle is None:
            return None
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return float(mw) / 1000.0
        except Exception:  # noqa: BLE001
            return None


class SysfsBackend(_PolledBackend):
    """Read INA3221 / hwmon power channels from ``/sys`` directly."""

    name = "sysfs"

    _GLOB_PATTERNS: tuple[str, ...] = (
        "/sys/class/hwmon/hwmon*/power*_input",
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221/*/iio:device*/in_power*_input",
        "/sys/devices/platform/*/hwmon/hwmon*/power*_input",
    )

    def __init__(self, *, sample_hz: float = 10.0) -> None:
        super().__init__(sample_hz=sample_hz)
        self.paths: list[Path] = self._discover_paths()

    @classmethod
    def _discover_paths(cls) -> list[Path]:
        if platform.system() != "Linux":
            return []
        out: list[Path] = []
        for pattern in cls._GLOB_PATTERNS:
            with suppress(Exception):
                out.extend(Path("/").glob(pattern.lstrip("/")))
        # Filter unreadable entries (common when hwmon enumerates
        # write-only attributes, etc.).
        readable: list[Path] = []
        for p in out:
            try:
                int(p.read_text().strip())
                readable.append(p)
            except (OSError, ValueError):
                continue
        return readable

    @classmethod
    def is_available(cls) -> bool:
        return bool(cls._discover_paths())

    def _read_power_w(self) -> float | None:
        if not self.paths:
            return None
        total_uw = 0
        any_read = False
        for p in self.paths:
            try:
                v = int(p.read_text().strip())
            except (OSError, ValueError):
                continue
            # hwmon "power*_input" is uW; INA3221 "in_power*_input" is mW.
            if "iio:device" in str(p):
                total_uw += v * 1000     # mW -> uW
            else:
                total_uw += v
            any_read = True
        return float(total_uw) / 1_000_000.0 if any_read else None


class TegrastatsBackend(EnergyBackend):
    """Spawn ``tegrastats`` and parse its periodic output for power rails."""

    name = "tegrastats"
    _BIN_PATHS: tuple[str, ...] = ("/usr/bin/tegrastats",
                                   "/usr/local/bin/tegrastats")
    _RAIL_RE = re.compile(r"(VDD_[A-Z0-9_]+|POM_[A-Z0-9_]+)\s+(\d+)"
                          r"(?:/(\d+))?")
    _TOTAL_RAILS: tuple[str, ...] = (
        "VDD_IN", "POM_5V_IN", "VDD_TOTAL", "POM_TOTAL",
    )
    _COMPONENT_KEYS: tuple[str, ...] = (
        "GPU", "CPU", "SOC", "DDR", "CV", "MEM",
    )

    def __init__(self, *, sample_hz: float = 10.0) -> None:
        super().__init__(sample_hz=sample_hz)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._bin = next((p for p in self._BIN_PATHS if Path(p).is_file()),
                         None)

    @classmethod
    def is_available(cls) -> bool:
        return any(Path(p).is_file() for p in cls._BIN_PATHS)

    def start(self) -> None:
        if self._bin is None:
            raise RuntimeError("tegrastats binary not found")
        self.samples.clear()
        self._stop_evt.clear()
        self._t_start = time.perf_counter()
        interval_ms = max(int(1000.0 / self.sample_hz), 50)
        self._proc = subprocess.Popen(
            [self._bin, "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="tegrastats_reader",
        )
        self._reader.start()

    def stop(self) -> dict[str, Any]:
        self._stop_evt.set()
        if self._proc is not None:
            with suppress(Exception):
                self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with suppress(Exception):
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
            self._proc = None
        if self._reader is not None:
            self._reader.join(timeout=1.5)
            self._reader = None
        self._t_stop = time.perf_counter()
        return self._build_result()

    # ------------------------------------------------------------------
    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop_evt.is_set():
                break
            t = time.perf_counter() - self._t_start
            p = self._parse_total_power(line)
            if p is not None:
                self.samples.append((t, p))

    def _parse_total_power(self, line: str) -> float | None:
        rails: dict[str, float] = {}
        for m in self._RAIL_RE.finditer(line):
            key = m.group(1)
            try:
                mw = int(m.group(2))
            except ValueError:
                continue
            rails[key] = mw / 1000.0   # mW -> W
        if not rails:
            return None
        for k in self._TOTAL_RAILS:
            if k in rails:
                return rails[k]
        component_sum = sum(
            v for k, v in rails.items()
            if any(comp in k for comp in self._COMPONENT_KEYS)
        )
        return component_sum if component_sum > 0 else None


class NoopBackend(EnergyBackend):
    """Always-available fallback. Records duration only."""

    name = "noop"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def start(self) -> None:
        self.samples.clear()
        self._t_start = time.perf_counter()

    def stop(self) -> dict[str, Any]:
        self._t_stop = time.perf_counter()
        return self._build_result()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
_PRIORITY_JETSON: tuple[str, ...] = ("tegrastats", "sysfs", "nvml", "noop")
_PRIORITY_DESKTOP: tuple[str, ...] = ("nvml", "sysfs", "noop")
_PRIORITY_CPU: tuple[str, ...] = ("sysfs", "noop")

_BACKEND_REGISTRY: dict[str, type[EnergyBackend]] = {
    "tegrastats": TegrastatsBackend,
    "sysfs":      SysfsBackend,
    "nvml":       NvmlBackend,
    "noop":       NoopBackend,
}


def select_backend(name: str = "auto",
                   *,
                   sample_hz: float = 10.0,
                   device: str = "cuda") -> EnergyBackend:
    """Resolve ``name`` to a concrete backend instance."""
    name = (name or "auto").lower()
    if name in _BACKEND_REGISTRY:
        cls = _BACKEND_REGISTRY[name]
        if not cls.is_available():
            LOG.warning("Energy backend %r unavailable — falling back to noop.",
                        name)
            return NoopBackend(sample_hz=sample_hz)
        return _instantiate(cls, sample_hz=sample_hz, device=device)
    if name != "auto":
        LOG.warning("Unknown energy backend %r — using auto.", name)

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if _is_jetson():
        priority = _PRIORITY_JETSON
    elif is_cuda:
        priority = _PRIORITY_DESKTOP
    else:
        priority = _PRIORITY_CPU

    for candidate in priority:
        cls = _BACKEND_REGISTRY[candidate]
        if cls.is_available():
            LOG.info("Energy backend selected: %s", candidate)
            return _instantiate(cls, sample_hz=sample_hz, device=device)
    return NoopBackend(sample_hz=sample_hz)


def _instantiate(cls: type[EnergyBackend], *,
                 sample_hz: float, device: str) -> EnergyBackend:
    if cls is NvmlBackend:
        return NvmlBackend(sample_hz=sample_hz,
                           device_index=_cuda_index(device))
    return cls(sample_hz=sample_hz)


# ---------------------------------------------------------------------------
# Public: streaming meter
# ---------------------------------------------------------------------------
class EnergyMeter:
    """Context-manager streaming energy meter (used by ``main_tracking.py``)."""

    def __init__(self,
                 *,
                 device: str = "cuda",
                 sample_hz: float = 10.0,
                 backend: str = "auto") -> None:
        self.device = device
        self.sample_hz = sample_hz
        self.backend: EnergyBackend = select_backend(
            backend, sample_hz=sample_hz, device=device,
        )
        self._result: dict[str, Any] | None = None

    def __enter__(self) -> "EnergyMeter":
        self.backend.start()
        return self

    def __exit__(self, *_args) -> None:
        if self._result is None:
            self._result = self.backend.stop()

    def stop(self) -> dict[str, Any]:
        if self._result is None:
            self._result = self.backend.stop()
        return dict(self._result)


# ---------------------------------------------------------------------------
# Public: one-shot benchmark
# ---------------------------------------------------------------------------
def measure_energy(target: Any,
                   *,
                   n_iters: int = 200,
                   n_warmup: int = 20,
                   device: str = "cuda",
                   input_shape: tuple[int, ...] = (1, 3, 224, 224),
                   sample_hz: float = 10.0,
                   input_factory: Callable[[str], Any] | None = None,
                   fp16: bool = False,
                   runner_factory: Callable[..., Callable[[], Any]]
                   | None = None,
                   backend: str = "auto",
                   include_idle_baseline: bool = True,
                   idle_seconds: float = 2.0,
                   **_unused) -> dict[str, Any]:
    """Run ``target`` for ``n_iters`` while sampling power; integrate energy.

    See module docstring for the full return schema.
    """
    if n_iters < 1:
        raise ValueError(f"n_iters must be >= 1; got {n_iters}")

    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    device_kind = _device_kind(device, is_cuda)
    runner = _resolve_runner(target, input_shape=input_shape,
                             device=device, fp16=fp16,
                             input_factory=input_factory,
                             runner_factory=runner_factory)

    # ---- (optional) idle baseline ------------------------------------
    idle_payload: dict[str, Any] | None = None
    if include_idle_baseline and idle_seconds > 0:
        idle_backend = select_backend(backend, sample_hz=sample_hz,
                                      device=device)
        idle_backend.start()
        time.sleep(idle_seconds)
        idle_payload = idle_backend.stop()

    # ---- active workload --------------------------------------------
    active_backend = select_backend(backend, sample_hz=sample_hz,
                                    device=device)
    LOG.info(
        "Measuring energy via %s (device=%s kind=%s) — warmup=%d, "
        "iters=%d, sample_hz=%.1f Hz, idle_baseline=%s",
        active_backend.name, device, device_kind, n_warmup, n_iters,
        sample_hz, include_idle_baseline,
    )

    active_backend.start()
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
        active_payload = active_backend.stop()

    return _compose_report(
        active=active_payload,
        idle=idle_payload,
        n_iters=n_iters,
        n_warmup=n_warmup,
        device=device,
        device_kind=device_kind,
        input_shape=input_shape,
        backend_name=active_backend.name,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _resolve_runner(target: Any,
                    *,
                    input_shape: tuple[int, ...],
                    device: str,
                    fp16: bool,
                    input_factory: Callable[[str], Any] | None,
                    runner_factory: Callable[..., Callable[[], Any]] | None
                    ) -> Callable[[], Any]:
    """Convert a target into a ``runner()`` callable."""
    if isinstance(target, nn.Module):
        return _make_module_runner(target, input_shape, device,
                                   fp16=fp16, input_factory=input_factory)
    if callable(target):
        return target
    if isinstance(target, (str, Path)):
        if runner_factory is not None:
            return runner_factory(Path(target),
                                  input_shape=input_shape, device=device)
        if not Path(target).is_file():
            raise FileNotFoundError(f"Engine file not found: {target}")
        LOG.warning(
            "No runner_factory provided for engine %s — energy measurement "
            "will only capture idle baseline. Pass runner_factory or wrap "
            "the engine in a callable for accurate metrics.", target,
        )
        return lambda: time.sleep(0.001)
    raise TypeError(
        f"Unsupported target type {type(target).__name__}; expected "
        "nn.Module, callable, or path-like."
    )


def _make_module_runner(model: nn.Module,
                        input_shape: tuple[int, ...],
                        device: str,
                        *,
                        fp16: bool,
                        input_factory: Callable[[str], Any] | None
                        ) -> Callable[[], Any]:
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

    return _runner


def _compose_report(*,
                    active: dict[str, Any],
                    idle: dict[str, Any] | None,
                    n_iters: int,
                    n_warmup: int,
                    device: str,
                    device_kind: str,
                    input_shape: tuple[int, ...],
                    backend_name: str) -> dict[str, Any]:
    """Combine active + idle payloads into the final result schema."""
    energy_mj = active.get("energy_mj")
    avg_power_w = active.get("avg_power_w")
    duration_s = active.get("duration_s", 0.0)

    avg_idle_w: float | None = None
    if idle is not None:
        avg_idle_w = idle.get("avg_power_w")

    active_delta_w: float | None = None
    if avg_power_w is not None and avg_idle_w is not None:
        active_delta_w = float(avg_power_w - avg_idle_w)

    active_energy_mj: float | None = None
    if active_delta_w is not None and duration_s > 0:
        # Idle-corrected energy = (avg_active - avg_idle) * duration * 1000
        active_energy_mj = float(active_delta_w * duration_s * 1000.0)

    energy_per_inf = (energy_mj / n_iters
                       if energy_mj is not None and n_iters > 0 else None)
    active_energy_per_inf = (active_energy_mj / n_iters
                              if active_energy_mj is not None and n_iters > 0
                              else None)

    return {
        "energy_mj":          energy_mj,
        "active_energy_mj":   active_energy_mj,
        "avg_power_w":        avg_power_w,
        "avg_idle_power_w":   avg_idle_w,
        "active_delta_w":     active_delta_w,
        "min_power_w":        active.get("min_power_w"),
        "max_power_w":        active.get("max_power_w"),
        "duration_s":         duration_s,
        "n_samples":          active.get("n_samples", 0),
        "n_iters":            int(n_iters),
        "n_warmup":           int(n_warmup),
        "energy_mj_per_inf":  energy_per_inf,
        "active_energy_mj_per_inf": active_energy_per_inf,
        "source":             backend_name,
        "device":             str(device),
        "device_kind":        device_kind,
        "input_shape":        list(input_shape),
        "idle_baseline":      idle if idle is not None else None,
    }


def _is_jetson() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        text = Path("/proc/device-tree/model").read_text(
            encoding="utf-8", errors="ignore",
        )
    except OSError:
        return False
    low = text.lower()
    return "jetson" in low or "tegra" in low


def _device_kind(device: str, is_cuda: bool) -> str:
    if not is_cuda:
        return "cpu"
    return "jetson" if _is_jetson() else "cuda"


def _cuda_index(device: str) -> int:
    s = str(device)
    if ":" not in s:
        return 0
    try:
        return int(s.split(":")[1])
    except ValueError:
        return 0
