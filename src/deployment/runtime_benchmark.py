"""
src/deployment/runtime_benchmark.py

Real embedded inference benchmarks for TensorRT engines on the Jetson Orin Nano.

Four measurement regimes
------------------------
1. Cold-start latency
       Time from engine-file load → first valid inference output.
       Captures TRT kernel JIT compilation, CUDA context initialisation, and
       deserialization overhead — the worst-case latency a robot experiences
       at power-on.

2. Warm latency
       Steady-state latency after N warm-up iterations.
       Reported separately as GPU-only time (CUDA Events) and wall-clock time
       (end-to-end including host overhead and synchronisation cost).

3. Throughput
       Sustained images-per-second under continuous load, swept across
       multiple batch sizes to expose the optimal operating point.

4. Stability over time
       Long-running session that tracks latency drift (thermal throttling),
       jitter, and statistical outliers.  Temperature and clock readings from
       Jetson sysfs are sampled periodically and embedded in the report.

Timing accuracy
---------------
GPU-side latency uses ``torch.cuda.Event`` with microsecond resolution.
Output buffers are pre-allocated before the timed loop so allocation overhead
does not contaminate measurements.

Sysfs interfaces (Jetson-specific; silently return empty dicts elsewhere)
------------------------------------------------------------------------
* Temperature : /sys/class/thermal/thermal_zone*/temp
* GPU clock   : /sys/devices/gpu.0/devfreq/*/cur_freq
* CPU clock   : /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
* EMC clock   : /sys/class/devfreq/*/cur_freq  (memory controller)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: TensorRT + export helpers
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt  # type: ignore[import]
    from .export_tensorrt import (           # type: ignore[import]
        load_engine,
        _get_engine_io_names,
        _set_input_shape_ctx,
        _get_output_shape_ctx,
        _execute_ctx,
        _TRT_AVAILABLE,
        _TRT_VERSION,
    )
    _RUNNER_AVAILABLE = True
except Exception:
    trt = None  # type: ignore[assignment]
    _TRT_AVAILABLE = False
    _TRT_VERSION = (0, 0, 0)
    _RUNNER_AVAILABLE = False

    def load_engine(*_a, **_kw):  # type: ignore[misc]
        raise RuntimeError("tensorrt / export_tensorrt not available.")

# Type alias accepted by public functions.
EngineInput = Union[str, Path, "trt.ICudaEngine"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTLIER_SIGMA: float = 3.0          # z-score threshold for outlier classification
_DRIFT_STABLE_MS: float = 0.5        # drift ≤ 0.5 ms / 1 000 runs → stable
_JITTER_STABLE_CV: float = 0.05      # CV ≤ 5 % → stable
_OUTLIER_STABLE_RATE: float = 0.01   # ≤ 1 % outliers → stable


# ---------------------------------------------------------------------------
# Jetson sysfs helpers
# ---------------------------------------------------------------------------


def read_jetson_thermals() -> dict[str, float]:
    """
    Read zone temperatures from ``/sys/class/thermal/``.

    Returns a dict mapping zone type (e.g. "CPU-therm", "GPU-therm") to
    temperature in °C.  Returns an empty dict on non-Jetson machines.
    """
    root = Path("/sys/class/thermal")
    if not root.is_dir():
        return {}

    result: dict[str, float] = {}
    for zone_dir in sorted(root.glob("thermal_zone*")):
        try:
            temp_c = int((zone_dir / "temp").read_text().strip()) / 1_000.0
            zone_type = (zone_dir / "type").read_text().strip()
            result[zone_type] = temp_c
        except Exception:
            pass
    return result


def read_jetson_clocks() -> dict[str, int | float]:
    """
    Read current clock frequencies from sysfs.

    Returns a dict with keys such as ``gpu_hz``, ``cpu0_khz``, ``emc_hz``.
    Returns an empty dict on non-Jetson machines.
    """
    result: dict[str, int | float] = {}

    # GPU clock — Orin path; try multiple locations.
    for pattern in (
        "/sys/devices/gpu.0/devfreq/*/cur_freq",
        "/sys/class/devfreq/gpu/cur_freq",
        "/sys/bus/platform/drivers/gk20a/*/devfreq/*/cur_freq",
    ):
        for p in Path("/").glob(pattern.lstrip("/")):
            try:
                result["gpu_hz"] = int(p.read_text().strip())
                break
            except Exception:
                pass
        if "gpu_hz" in result:
            break

    # CPU clocks — first 8 cores (Orin has 8 × Cortex-A78AE).
    for i in range(8):
        p = Path(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
        if p.is_file():
            try:
                result[f"cpu{i}_khz"] = int(p.read_text().strip())
            except Exception:
                pass

    # EMC (memory controller) clock.
    for pattern in (
        "/sys/class/devfreq/*/cur_freq",
        "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
    ):
        for p in Path("/").glob(pattern.lstrip("/")):
            name = p.parent.name
            if "emc" in name.lower():
                try:
                    result["emc_hz"] = int(p.read_text().strip())
                except Exception:
                    pass

    return result


def read_system_info(device_id: int = 0) -> dict:
    """
    Collect host / device metadata for the benchmark report.

    Returns a dict with: cuda_device, driver_version, trt_version,
    torch_version, thermals, clocks, nvpmodel (if readable).
    """
    info: dict = {
        "trt_version": ".".join(str(v) for v in _TRT_VERSION),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(device_id)
        info["cuda_capability"] = list(
            torch.cuda.get_device_capability(device_id)
        )
        try:
            info["driver_version"] = torch.version.cuda
        except Exception:
            pass

    info["thermals"] = read_jetson_thermals()
    info["clocks"] = {k: v for k, v in read_jetson_clocks().items()}

    # Jetson power model.
    nvpmodel = Path("/etc/nvpmodel.conf")
    if nvpmodel.is_file():
        try:
            for line in nvpmodel.read_text().splitlines():
                if line.strip().startswith("PM_CONFIG DEFAULT"):
                    info["nvpmodel"] = line.strip()
                    break
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """
    Hyperparameters controlling all four benchmark regimes.

    Parameters
    ----------
    cold_start_trials : int
        Number of independent cold-start trials (engine reload + first
        inference per trial).  Mean and distribution are reported.
    n_warmup : int
        Warm-up inferences discarded before timed measurement begins.
    n_timed : int
        Number of timed inferences for the warm-latency regime.
    throughput_duration_s : float
        Wall-clock duration of each throughput measurement (per batch size).
    throughput_batch_sizes : list[int]
        Batch sizes swept during throughput measurement.
    stability_n_runs : int
        Total inferences for the stability regime.
    stability_thermal_every : int
        Sample thermals and clocks every N inferences during stability run.
    device_id : int
        CUDA device index.
    output_dir : Path or None
        When provided, JSON report and CSV timeseries are written here.
    save_timeseries : bool
        Write per-sample latency CSV to ``output_dir``.
    """

    cold_start_trials: int = 5
    n_warmup: int = 50
    n_timed: int = 200
    throughput_duration_s: float = 30.0
    throughput_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )
    stability_n_runs: int = 1_000
    stability_thermal_every: int = 50
    device_id: int = 0
    output_dir: Path | None = None
    save_timeseries: bool = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LatencyStats:
    """
    Descriptive statistics for a latency sample array.

    All times in milliseconds.  ``cv`` is the coefficient of variation
    (std / mean), a dimensionless jitter indicator.
    """

    n_samples: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    cv: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "LatencyStats":
        mean = float(arr.mean())
        return cls(
            n_samples=len(arr),
            mean_ms=round(mean, 3),
            std_ms=round(float(arr.std()), 3),
            min_ms=round(float(arr.min()), 3),
            max_ms=round(float(arr.max()), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p90_ms=round(float(np.percentile(arr, 90)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            cv=round(float(arr.std() / max(mean, 1e-9)), 4),
        )


@dataclass
class ColdStartResult:
    """
    Cold-start latency across multiple engine-reload trials.

    Attributes
    ----------
    load_times_ms : per-trial time from file-open to engine-ready (ms).
    first_infer_times_ms : per-trial first-inference time (ms).
    end_to_end_ms : load + first-inference combined (ms), one per trial.
    stats_load : statistics over ``load_times_ms``.
    stats_first_infer : statistics over ``first_infer_times_ms``.
    stats_end_to_end : statistics over ``end_to_end_ms``.
    """

    load_times_ms: list[float]
    first_infer_times_ms: list[float]
    end_to_end_ms: list[float]
    stats_load: LatencyStats
    stats_first_infer: LatencyStats
    stats_end_to_end: LatencyStats


@dataclass
class ThroughputResult:
    """
    Sustained throughput for one batch size.

    Attributes
    ----------
    batch_size : int
    duration_s : float — actual measurement window.
    total_batches : int — number of completed inference calls.
    total_images : int — total images processed.
    throughput_fps : float — images / second.
    mean_batch_ms : float — mean inference time per batch call.
    gpu_utilisation_pct : float — estimated GPU utilisation (0–100);
        0 if NVML unavailable.
    """

    batch_size: int
    duration_s: float
    total_batches: int
    total_images: int
    throughput_fps: float
    mean_batch_ms: float
    gpu_utilisation_pct: float


@dataclass
class StabilityResult:
    """
    Long-run stability analysis.

    Attributes
    ----------
    n_runs : int
    stats : LatencyStats — over the full run.
    drift_ms_per_1k : float — linear latency drift per 1 000 inferences;
        positive = slowing down (thermal throttle).
    outlier_count : int — samples beyond mean ± OUTLIER_SIGMA × std.
    outlier_rate : float — fraction of outlier samples.
    jitter_ratio : float — max_ms / mean_ms.
    thermal_samples : list[dict] — periodic {inference_idx, thermals, clocks}.
    is_stable : bool — True when drift, jitter, and outlier thresholds pass.
    verdict : str — "stable" | "jittery" | "throttling" | "unstable".
    """

    n_runs: int
    stats: LatencyStats
    drift_ms_per_1k: float
    outlier_count: int
    outlier_rate: float
    jitter_ratio: float
    thermal_samples: list[dict]
    is_stable: bool
    verdict: str


@dataclass
class BenchmarkReport:
    """
    Complete benchmark report for one TensorRT engine.

    All sub-results are plain dataclasses; ``as_dict()`` makes the whole
    structure JSON-serialisable.
    """

    timestamp: str
    engine_path: str
    trt_version: str
    device_name: str
    input_shape: tuple
    precision: str
    cold_start: ColdStartResult
    warm_gpu: LatencyStats         # CUDA Event timing
    warm_wall: LatencyStats        # end-to-end wall-clock timing
    throughput: list[ThroughputResult]
    stability: StabilityResult
    system_info: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d

    def summary(self) -> str:
        """Concise human-readable summary."""
        lines = [
            "── Runtime Benchmark ────────────────────────────────────",
            f"  Engine      : {self.engine_path}",
            f"  Precision   : {self.precision}",
            f"  Device      : {self.device_name}",
            f"  Input shape : {self.input_shape}",
            "  Cold start  : "
            f"load={self.cold_start.stats_load.mean_ms:.1f} ms  "
            f"first_infer={self.cold_start.stats_first_infer.mean_ms:.1f} ms",
            "  Warm GPU    : "
            f"mean={self.warm_gpu.mean_ms:.2f} ms  "
            f"p99={self.warm_gpu.p99_ms:.2f} ms  "
            f"CV={self.warm_gpu.cv:.3f}",
            "  Warm wall   : "
            f"mean={self.warm_wall.mean_ms:.2f} ms",
        ]
        for t in self.throughput:
            lines.append(
                f"  Throughput  : BS={t.batch_size}  "
                f"{t.throughput_fps:.1f} FPS  "
                f"({t.mean_batch_ms:.2f} ms/batch)"
            )
        lines += [
            f"  Stability   : {self.stability.verdict}  "
            f"drift={self.stability.drift_ms_per_1k:+.2f} ms/1k  "
            f"outliers={self.stability.outlier_rate*100:.1f}%",
            "─────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-allocated inference runner
# ---------------------------------------------------------------------------


class _TimedInferenceRunner:
    """
    Zero-overhead timed inference runner.

    Pre-allocates input and output CUDA buffers once so that no memory
    operations pollute the timed hot-path.  Uses ``torch.cuda.Event`` for
    microsecond-accurate GPU-side latency measurement.

    Parameters
    ----------
    engine     : Loaded TRT ICudaEngine.
    input_shape: (B, C, H, W) — fixed for the lifetime of this runner.
    device_id  : CUDA device index.
    """

    def __init__(
        self,
        engine: "trt.ICudaEngine",
        input_shape: tuple[int, ...],
        device_id: int = 0,
    ) -> None:
        if not _RUNNER_AVAILABLE:
            raise RuntimeError(
                "export_tensorrt helpers not available.  "
                "Ensure tensorrt is installed and export_tensorrt.py is present."
            )
        self._engine = engine
        self._device_id = device_id
        self._ctx = engine.create_execution_context()

        input_names, output_names = _get_engine_io_names(engine)
        self._input_name = input_names[0]
        self._output_names = output_names

        dev = f"cuda:{device_id}"
        self._input: torch.Tensor = torch.zeros(
            *input_shape, dtype=torch.float32, device=dev
        )
        _set_input_shape_ctx(self._ctx, self._input_name, input_shape)

        self._outputs: dict[str, torch.Tensor] = {
            name: torch.empty(
                _get_output_shape_ctx(self._ctx, name),
                dtype=torch.float32,
                device=dev,
            )
            for name in output_names
        }

        self._in_ptrs = {self._input_name: self._input.data_ptr()}
        self._out_ptrs = {n: t.data_ptr() for n, t in self._outputs.items()}

        self._stream = torch.cuda.Stream(device=dev)
        self._start_e = torch.cuda.Event(enable_timing=True)
        self._end_e = torch.cuda.Event(enable_timing=True)

    def run(self) -> tuple[float, float]:
        """
        Execute one inference.

        Returns
        -------
        (gpu_ms, wall_ms)
            ``gpu_ms``  — CUDA Event elapsed time (GPU-only, excludes host).
            ``wall_ms`` — wall-clock elapsed time (includes host overhead).
        """
        t0 = time.perf_counter()
        self._start_e.record(self._stream)
        _execute_ctx(
            self._ctx,
            self._in_ptrs,
            self._out_ptrs,
            self._stream.cuda_stream,
        )
        self._end_e.record(self._stream)
        torch.cuda.synchronize(self._device_id)
        wall_ms = (time.perf_counter() - t0) * 1_000.0
        gpu_ms = self._start_e.elapsed_time(self._end_e)
        return float(gpu_ms), float(wall_ms)

    def warmup(self, n: int = 50) -> None:
        """Run *n* warm-up inferences (results discarded)."""
        for _ in range(n):
            self.run()


# ---------------------------------------------------------------------------
# Internal analysis helpers
# ---------------------------------------------------------------------------


def _linear_drift(times_ms: np.ndarray) -> float:
    """
    Least-squares linear drift coefficient in ms per 1 000 inferences.

    Positive = latency increasing (thermal throttle / memory pressure).
    """
    if len(times_ms) < 10:
        return 0.0
    x = np.arange(len(times_ms), dtype=np.float64)
    # polyfit degree 1 → [slope, intercept]
    slope = float(np.polyfit(x, times_ms, 1)[0])
    return slope * 1_000.0


def _count_outliers(times_ms: np.ndarray) -> tuple[int, float]:
    """Return (count, rate) of samples beyond mean ± OUTLIER_SIGMA × std."""
    mean, std = times_ms.mean(), times_ms.std()
    mask = np.abs(times_ms - mean) > _OUTLIER_SIGMA * std
    count = int(mask.sum())
    return count, float(count / max(len(times_ms), 1))


def _stability_verdict(
    stats: LatencyStats,
    drift: float,
    outlier_rate: float,
) -> tuple[bool, str]:
    """
    Classify stability based on drift, jitter (CV), and outlier rate.

    Returns (is_stable, verdict_string).
    """
    throttling = drift > _DRIFT_STABLE_MS
    jittery = stats.cv > _JITTER_STABLE_CV
    outlier_heavy = outlier_rate > _OUTLIER_STABLE_RATE

    if not (throttling or jittery or outlier_heavy):
        return True, "stable"
    if throttling:
        return False, "throttling"
    if outlier_heavy and jittery:
        return False, "unstable"
    if jittery:
        return False, "jittery"
    return False, "noisy"


# ---------------------------------------------------------------------------
# Measurement 1 — Cold-start latency
# ---------------------------------------------------------------------------


def measure_cold_start(
    engine_path: str | Path,
    input_shape: tuple[int, ...],
    *,
    config: BenchmarkConfig | None = None,
) -> ColdStartResult:
    """
    Measure cold-start latency across multiple independent engine-reload trials.

    Each trial:
    1. Times deserialization + CUDA engine creation (``load_time_ms``).
    2. Times the very first inference without any warm-up (``first_infer_ms``).

    Parameters
    ----------
    engine_path : Path to the ``.trt`` engine file.
    input_shape : (B, C, H, W) shape for the dummy input.
    config      : BenchmarkConfig; defaults used if None.

    Returns
    -------
    ColdStartResult with per-trial and aggregate statistics.
    """
    if not _RUNNER_AVAILABLE:
        raise RuntimeError("tensorrt / export_tensorrt not available.")

    cfg = config or BenchmarkConfig()
    engine_path = Path(engine_path)
    dev = f"cuda:{cfg.device_id}"

    load_times: list[float] = []
    first_infer_times: list[float] = []
    end_to_end: list[float] = []

    dummy = torch.randn(*input_shape, device=dev, dtype=torch.float32)

    for trial in range(cfg.cold_start_trials):
        # ── Engine load ───────────────────────────────────────────────────
        t_load_start = time.perf_counter()
        engine = load_engine(engine_path, device_id=cfg.device_id)
        t_load_end = time.perf_counter()
        load_ms = (t_load_end - t_load_start) * 1_000.0

        # ── First inference (no warm-up) ──────────────────────────────────
        runner = _TimedInferenceRunner(engine, input_shape, cfg.device_id)
        gpu_ms, wall_ms = runner.run()

        load_times.append(load_ms)
        first_infer_times.append(gpu_ms)
        end_to_end.append(load_ms + gpu_ms)

        log.info(
            "Cold-start trial %d/%d | load=%.1f ms | first_infer=%.2f ms",
            trial + 1, cfg.cold_start_trials, load_ms, gpu_ms,
        )

        del engine  # release TRT engine before next trial

    l_arr = np.array(load_times)
    fi_arr = np.array(first_infer_times)
    e2e_arr = np.array(end_to_end)

    return ColdStartResult(
        load_times_ms=[round(v, 3) for v in load_times],
        first_infer_times_ms=[round(v, 3) for v in first_infer_times],
        end_to_end_ms=[round(v, 3) for v in end_to_end],
        stats_load=LatencyStats.from_array(l_arr),
        stats_first_infer=LatencyStats.from_array(fi_arr),
        stats_end_to_end=LatencyStats.from_array(e2e_arr),
    )


# ---------------------------------------------------------------------------
# Measurement 2 — Warm latency
# ---------------------------------------------------------------------------


def measure_warm_latency(
    engine: "trt.ICudaEngine",
    input_shape: tuple[int, ...],
    *,
    config: BenchmarkConfig | None = None,
) -> tuple[LatencyStats, LatencyStats]:
    """
    Measure steady-state GPU and wall-clock latency after warm-up.

    Parameters
    ----------
    engine      : Loaded TRT ICudaEngine.
    input_shape : (B, C, H, W) shape for dummy input.
    config      : BenchmarkConfig; defaults used if None.

    Returns
    -------
    (gpu_stats, wall_stats) : CUDA Event timing and wall-clock timing.
    """
    cfg = config or BenchmarkConfig()
    runner = _TimedInferenceRunner(engine, input_shape, cfg.device_id)

    log.info("Warm-up: %d inferences …", cfg.n_warmup)
    runner.warmup(cfg.n_warmup)

    gpu_times: list[float] = []
    wall_times: list[float] = []

    log.info("Timing %d inferences …", cfg.n_timed)
    for _ in range(cfg.n_timed):
        gpu_ms, wall_ms = runner.run()
        gpu_times.append(gpu_ms)
        wall_times.append(wall_ms)

    return (
        LatencyStats.from_array(np.array(gpu_times)),
        LatencyStats.from_array(np.array(wall_times)),
    )


# ---------------------------------------------------------------------------
# Measurement 3 — Throughput
# ---------------------------------------------------------------------------


def _read_gpu_utilisation(device_id: int) -> float:
    """Read GPU utilisation % via pynvml; returns 0.0 if unavailable."""
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return 0.0


def measure_throughput(
    engine: "trt.ICudaEngine",
    base_input_shape: tuple[int, ...],
    *,
    config: BenchmarkConfig | None = None,
) -> list[ThroughputResult]:
    """
    Measure sustained images-per-second across configured batch sizes.

    For each batch size, the engine is driven continuously for
    ``config.throughput_duration_s`` seconds and the completed images are
    counted.  A warm-up phase precedes each measurement.

    Parameters
    ----------
    engine           : Loaded TRT ICudaEngine.
    base_input_shape : (B, C, H, W) — B is overridden per batch size.
    config           : BenchmarkConfig; defaults used if None.

    Returns
    -------
    List of ThroughputResult, one per batch size.
    """
    cfg = config or BenchmarkConfig()
    c = base_input_shape[1]
    h = base_input_shape[2]
    w = base_input_shape[3]

    results: list[ThroughputResult] = []

    for bs in cfg.throughput_batch_sizes:
        shape = (bs, c, h, w)
        runner = _TimedInferenceRunner(engine, shape, cfg.device_id)
        runner.warmup(max(20, cfg.n_warmup // 2))

        log.info(
            "Throughput sweep | BS=%d | duration=%.0f s …",
            bs, cfg.throughput_duration_s,
        )

        batch_times: list[float] = []
        deadline = time.perf_counter() + cfg.throughput_duration_s

        while time.perf_counter() < deadline:
            gpu_ms, _ = runner.run()
            batch_times.append(gpu_ms)

        actual_duration = cfg.throughput_duration_s
        n_batches = len(batch_times)
        n_images = n_batches * bs
        mean_batch_ms = float(np.mean(batch_times)) if batch_times else 0.0
        fps = n_images / actual_duration if actual_duration > 0 else 0.0

        gpu_util = _read_gpu_utilisation(cfg.device_id)

        results.append(
            ThroughputResult(
                batch_size=bs,
                duration_s=round(actual_duration, 2),
                total_batches=n_batches,
                total_images=n_images,
                throughput_fps=round(fps, 2),
                mean_batch_ms=round(mean_batch_ms, 3),
                gpu_utilisation_pct=round(gpu_util, 1),
            )
        )
        log.info(
            "  BS=%d → %.1f FPS  (%.2f ms/batch)", bs, fps, mean_batch_ms
        )

    return results


# ---------------------------------------------------------------------------
# Measurement 4 — Stability over time
# ---------------------------------------------------------------------------


def measure_stability(
    engine: "trt.ICudaEngine",
    input_shape: tuple[int, ...],
    *,
    config: BenchmarkConfig | None = None,
) -> StabilityResult:
    """
    Long-running stability benchmark with thermal sampling.

    Runs ``config.stability_n_runs`` inferences and, every
    ``config.stability_thermal_every`` iterations, records temperatures and
    clock frequencies from Jetson sysfs.

    Parameters
    ----------
    engine      : Loaded TRT ICudaEngine.
    input_shape : (B, C, H, W).
    config      : BenchmarkConfig; defaults used if None.

    Returns
    -------
    StabilityResult with latency statistics, drift coefficient, outlier
    analysis, thermal trace, and a human-readable verdict.
    """
    cfg = config or BenchmarkConfig()
    runner = _TimedInferenceRunner(engine, input_shape, cfg.device_id)

    log.info("Stability run: %d inferences …", cfg.stability_n_runs)
    runner.warmup(cfg.n_warmup)

    gpu_times: list[float] = []
    thermal_trace: list[dict] = []

    for i in range(cfg.stability_n_runs):
        gpu_ms, _ = runner.run()
        gpu_times.append(gpu_ms)

        if i % cfg.stability_thermal_every == 0:
            sample = {
                "inference_idx": i,
                "thermals": read_jetson_thermals(),
                "clocks": {k: int(v) for k, v in read_jetson_clocks().items()},
                "latency_ms": round(gpu_ms, 3),
            }
            thermal_trace.append(sample)

        if (i + 1) % 200 == 0:
            log.debug("Stability progress: %d / %d", i + 1, cfg.stability_n_runs)

    arr = np.array(gpu_times, dtype=np.float64)
    stats = LatencyStats.from_array(arr)
    drift = _linear_drift(arr)
    outlier_count, outlier_rate = _count_outliers(arr)
    jitter_ratio = float(arr.max() / max(arr.mean(), 1e-9))
    is_stable, verdict = _stability_verdict(stats, drift, outlier_rate)

    log.info(
        "Stability | verdict=%s | drift=%+.2f ms/1k | CV=%.3f | outliers=%.1f%%",
        verdict, drift, stats.cv, outlier_rate * 100,
    )

    return StabilityResult(
        n_runs=cfg.stability_n_runs,
        stats=stats,
        drift_ms_per_1k=round(drift, 4),
        outlier_count=outlier_count,
        outlier_rate=round(outlier_rate, 5),
        jitter_ratio=round(jitter_ratio, 4),
        thermal_samples=thermal_trace,
        is_stable=is_stable,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Full benchmark orchestrator
# ---------------------------------------------------------------------------


def run_full_benchmark(
    engine_path: str | Path,
    input_shape: tuple[int, ...],
    *,
    config: BenchmarkConfig | None = None,
) -> BenchmarkReport:
    """
    Run all four benchmark regimes in sequence and return a unified report.

    Order of execution:
    1. Cold-start measurement (engine reloaded per trial).
    2. Warm latency on a persistent engine instance.
    3. Throughput sweep across batch sizes.
    4. Stability over time.

    Parameters
    ----------
    engine_path : Path to the ``.trt`` engine file.
    input_shape : (B, C, H, W) — B should be 1 for single-image benchmarks.
    config      : BenchmarkConfig; defaults used if None.

    Returns
    -------
    BenchmarkReport with all measurements.  If ``config.output_dir`` is set,
    the report JSON and (optionally) a latency CSV are written there.
    """
    if not _RUNNER_AVAILABLE:
        raise RuntimeError(
            "TensorRT is not available.  Install via JetPack or pip install tensorrt."
        )

    cfg = config or BenchmarkConfig()
    engine_path = Path(engine_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    device_name = (
        torch.cuda.get_device_name(cfg.device_id)
        if torch.cuda.is_available()
        else "unknown"
    )
    trt_ver = ".".join(str(v) for v in _TRT_VERSION)

    # Derive precision from JSON sidecar if available.
    sidecar = engine_path.with_suffix(".json")
    precision = "unknown"
    if sidecar.is_file():
        try:
            sd = json.loads(sidecar.read_text())
            precision = sd.get("build_info", {}).get("precision", "unknown")
        except Exception:
            pass

    log.info(
        "=== Full Benchmark | engine=%s | precision=%s ===",
        engine_path.name, precision,
    )

    # ── 1. Cold start ─────────────────────────────────────────────────────
    log.info("Phase 1/4 — Cold start (%d trials)", cfg.cold_start_trials)
    cold_start = measure_cold_start(engine_path, input_shape, config=cfg)

    # Load the engine once for the remaining phases.
    engine = load_engine(engine_path, device_id=cfg.device_id)

    # ── 2. Warm latency ───────────────────────────────────────────────────
    log.info("Phase 2/4 — Warm latency (%d runs)", cfg.n_timed)
    warm_gpu, warm_wall = measure_warm_latency(engine, input_shape, config=cfg)

    # ── 3. Throughput ─────────────────────────────────────────────────────
    log.info(
        "Phase 3/4 — Throughput (batch sizes %s, %.0f s each)",
        cfg.throughput_batch_sizes, cfg.throughput_duration_s,
    )
    throughput = measure_throughput(engine, input_shape, config=cfg)

    # ── 4. Stability ──────────────────────────────────────────────────────
    log.info("Phase 4/4 — Stability (%d runs)", cfg.stability_n_runs)
    stability = measure_stability(engine, input_shape, config=cfg)

    # ── Assemble report ───────────────────────────────────────────────────
    system_info = read_system_info(cfg.device_id)
    report = BenchmarkReport(
        timestamp=timestamp,
        engine_path=str(engine_path),
        trt_version=trt_ver,
        device_name=device_name,
        input_shape=input_shape,
        precision=precision,
        cold_start=cold_start,
        warm_gpu=warm_gpu,
        warm_wall=warm_wall,
        throughput=throughput,
        stability=stability,
        system_info=system_info,
    )

    log.info("\n%s", report.summary())

    # ── Persist ───────────────────────────────────────────────────────────
    if cfg.output_dir is not None:
        save_report(report, cfg.output_dir, save_timeseries=cfg.save_timeseries)

    return report


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------


def save_report(
    report: BenchmarkReport,
    output_dir: str | Path,
    *,
    save_timeseries: bool = True,
) -> dict[str, Path]:
    """
    Persist a BenchmarkReport to disk.

    Writes:
    * ``benchmark_report.json``   — full report as JSON.
    * ``latency_stability.csv``   — per-sample GPU latency during stability run
                                    (if ``save_timeseries`` is True).

    Parameters
    ----------
    report        : BenchmarkReport to persist.
    output_dir    : Destination directory (created if absent).
    save_timeseries: Write per-sample latency CSV.

    Returns
    -------
    dict mapping label → Path for each written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # JSON report (atomic write)
    json_path = out / "benchmark_report.json"
    tmp = json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report.as_dict(), indent=2, default=str))
    tmp.replace(json_path)
    written["json"] = json_path
    log.info("Report saved → %s", json_path)

    # Stability latency timeseries CSV
    if save_timeseries and report.stability.thermal_samples:
        csv_path = out / "latency_stability.csv"
        rows = [
            f"{s['inference_idx']},{s['latency_ms']}"
            for s in report.stability.thermal_samples
        ]
        csv_path.write_text("inference_idx,latency_ms\n" + "\n".join(rows))
        written["csv"] = csv_path
        log.info("Timeseries saved → %s", csv_path)

    return written


def load_report(path: str | Path) -> BenchmarkReport:
    """
    Load a BenchmarkReport from a JSON file written by :func:`save_report`.

    Parameters
    ----------
    path : Path to ``benchmark_report.json``.

    Returns
    -------
    BenchmarkReport (re-constructed from the dict representation).

    Notes
    -----
    Only the top-level scalar fields and nested dicts are restored;
    sub-dataclasses are returned as plain dicts within ``BenchmarkReport``.
    """
    data = json.loads(Path(path).read_text())

    def _ls(d: dict) -> LatencyStats:
        return LatencyStats(**{k: d[k] for k in LatencyStats.__dataclass_fields__})

    cold_raw = data["cold_start"]
    cold = ColdStartResult(
        load_times_ms=cold_raw["load_times_ms"],
        first_infer_times_ms=cold_raw["first_infer_times_ms"],
        end_to_end_ms=cold_raw["end_to_end_ms"],
        stats_load=_ls(cold_raw["stats_load"]),
        stats_first_infer=_ls(cold_raw["stats_first_infer"]),
        stats_end_to_end=_ls(cold_raw["stats_end_to_end"]),
    )
    stab_raw = data["stability"]
    stability = StabilityResult(
        n_runs=stab_raw["n_runs"],
        stats=_ls(stab_raw["stats"]),
        drift_ms_per_1k=stab_raw["drift_ms_per_1k"],
        outlier_count=stab_raw["outlier_count"],
        outlier_rate=stab_raw["outlier_rate"],
        jitter_ratio=stab_raw["jitter_ratio"],
        thermal_samples=stab_raw["thermal_samples"],
        is_stable=stab_raw["is_stable"],
        verdict=stab_raw["verdict"],
    )

    return BenchmarkReport(
        timestamp=data["timestamp"],
        engine_path=data["engine_path"],
        trt_version=data["trt_version"],
        device_name=data["device_name"],
        input_shape=tuple(data["input_shape"]),
        precision=data["precision"],
        cold_start=cold,
        warm_gpu=_ls(data["warm_gpu"]),
        warm_wall=_ls(data["warm_wall"]),
        throughput=[ThroughputResult(**t) for t in data["throughput"]],
        stability=stability,
        system_info=data.get("system_info", {}),
    )
