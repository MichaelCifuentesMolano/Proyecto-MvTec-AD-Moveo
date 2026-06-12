"""
src/deployment/export_tensorrt.py

Build TensorRT engines from ONNX models with FP32, FP16, or INT8 precision
for deployment on the Jetson Orin Nano.

Pipeline overview
-----------------
ONNX  ──►  TRT parser  ──►  BuilderConfig  ──►  Serialised engine  ──►  .trt file
                                  ▲
                           FP16 / INT8 flags
                           INT8 calibrator (entropy)

Precision modes
---------------
fp32  — Full floating-point; largest model, highest accuracy.
fp16  — Half-precision; ~2× throughput, minimal accuracy drop; recommended.
int8  — 8-bit integer quantisation; requires an image calibration dataset;
        best latency/energy; small accuracy drop if well-calibrated.

TensorRT version support
------------------------
Version detection at import time; version-adaptive helpers cover TRT 8.5–10.x:
  * Workspace API: max_workspace_size (≤8.4) vs set_memory_pool_limit (8.5+)
  * Build API:     build_engine (≤8.4) vs build_serialized_network (8.5+)
  * Binding API:   get_binding_* (8.x) vs get_tensor_* + set_tensor_address (9+)

Assumptions
-----------
* Models are exported to ONNX with explicit batch dimension (static or
  dynamic) by ``export_onnx.py``.
* Calibration tensors are ImageNet-normalised float32 on CPU or CUDA.
* A CUDA-capable device is required for INT8 calibration and engine building;
  engine files can be loaded and run on any compatible Jetson device.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: TensorRT
# ---------------------------------------------------------------------------

try:
    import tensorrt as trt  # type: ignore[import]
    _TRT_AVAILABLE = True
    _TRT_VERSION: tuple[int, ...] = tuple(
        int(x) for x in trt.__version__.split(".")[:3]
    )
    _TRT_MAJOR: int = _TRT_VERSION[0]
except ImportError:
    trt = None  # type: ignore[assignment]
    _TRT_AVAILABLE = False
    _TRT_VERSION = (0, 0, 0)
    _TRT_MAJOR = 0

# Optional: ONNX (for reading model metadata before build)
try:
    import onnx as _onnx_mod  # type: ignore[import]
    _ONNX_AVAILABLE = True
except ImportError:
    _onnx_mod = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False

# Base class for calibrator — must be object when TRT is absent.
_CalibBase = trt.IInt8EntropyCalibrator2 if _TRT_AVAILABLE else object


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TRTBuildConfig:
    """
    Hyperparameters for TensorRT engine compilation.

    Parameters
    ----------
    precision : {"fp32", "fp16", "int8"}
        Inference precision.  FP16 is the recommended default for Jetson.
        INT8 requires a calibration dataset (pass via ``build_engine``).
    workspace_mb : int
        GPU memory (MB) available to the TRT builder during optimisation.
        Does not affect runtime memory.  1 024 MB is a safe default.
    min_batch, opt_batch, max_batch : int
        Optimization-profile batch sizes.  ``opt_batch`` should match the
        deployment batch size for best tuning.  Set all three to the same
        value for a static-batch engine (recommended for Jetson).
    calibration_batches : int
        Number of calibration batches consumed by the INT8 calibrator.
        100–200 batches of 8–16 images is typically sufficient.
    calibration_cache : Path or None
        Path for the TRT calibration cache file.  When present and valid,
        calibration is skipped and the cache is loaded directly, saving
        several minutes on Jetson.
    dla_core : int
        DLA (Deep Learning Accelerator) core index.  -1 disables DLA and
        uses the Jetson GPU.  Valid values: 0, 1 (Orin has 2 DLA cores).
    strict_precision : bool
        When True, all layers are forced to the requested precision
        (PREFER_PRECISION_CONSTRAINTS / STRICT_TYPES).  May reduce accuracy.
    builder_opt_level : int
        Builder optimisation level [0, 5].  0 = fastest build, 5 = best
        runtime throughput.  Available in TRT 8.6+; silently ignored on
        older versions.
    verbose : bool
        Enable verbose TRT logger output (useful for debugging layer fusion).
    """

    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    workspace_mb: int = 1_024
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    calibration_batches: int = 100
    calibration_cache: Path | None = None
    dla_core: int = -1
    strict_precision: bool = False
    builder_opt_level: int = 3
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TRTBuildResult:
    """
    Outcome of a single TensorRT engine build.

    Attributes
    ----------
    success : bool
    engine_path : Path or None
    sidecar_path : Path or None  — JSON sidecar beside the engine file.
    precision : str
    input_name : str
    output_names : list[str]
    input_shape : tuple  — (B, C, H, W) with opt_batch.
    engine_size_mb : float
    calibration_cache_path : Path or None
    trt_version : str
    elapsed_seconds : float
    error_message : str or None
    """

    success: bool
    engine_path: Path | None
    sidecar_path: Path | None
    precision: str
    input_name: str
    output_names: list[str]
    input_shape: tuple
    engine_size_mb: float
    calibration_cache_path: Path | None
    trt_version: str
    elapsed_seconds: float
    error_message: str | None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["engine_path"] = str(self.engine_path) if self.engine_path else None
        d["sidecar_path"] = str(self.sidecar_path) if self.sidecar_path else None
        d["calibration_cache_path"] = (
            str(self.calibration_cache_path)
            if self.calibration_cache_path else None
        )
        d["input_shape"] = list(self.input_shape)
        return d


# ---------------------------------------------------------------------------
# INT8 calibrator
# ---------------------------------------------------------------------------


class Int8Calibrator(_CalibBase):  # type: ignore[misc]
    """
    Entropy-based INT8 calibrator backed by pre-loaded image tensors.

    Implements ``IInt8EntropyCalibrator2`` — the recommended calibrator for
    computer-vision models according to NVIDIA documentation.

    Parameters
    ----------
    batches : sequence of (B, C, H, W) float32 tensors.
        Calibration images.  Can be on CPU or CUDA; they are moved to CUDA
        automatically.  Use ``collect_calibration_batches()`` to build this
        list from a DataLoader.
    cache_path : Path or None
        When a valid cache file exists at this path, TRT reads it and skips
        running the model through calibration images.  After calibration the
        cache is written here for future re-use.
    """

    def __init__(
        self,
        batches: Sequence[torch.Tensor],
        cache_path: Path | None = None,
    ) -> None:
        if not _TRT_AVAILABLE:
            raise RuntimeError(
                "tensorrt is not installed.  Cannot create Int8Calibrator."
            )
        super().__init__()
        self._batches: list[torch.Tensor] = list(batches)
        self._idx: int = 0
        self._cache_path: Path | None = Path(cache_path) if cache_path else None
        # Pinned CUDA buffer — allocated on first call to get_batch().
        self._buf: torch.Tensor | None = None

    # TRT calls these in order: read_calibration_cache → (if None) get_batch
    # loop → write_calibration_cache.

    def get_batch_size(self) -> int:  # type: ignore[override]
        return int(self._batches[0].shape[0]) if self._batches else 1

    def get_batch(self, names: list[str]) -> list[int] | None:  # type: ignore[override]
        """
        Feed the next calibration batch to TRT.

        Returns a list containing the CUDA device pointer for the input tensor,
        or None when all batches have been consumed.
        """
        if self._idx >= len(self._batches):
            return None

        batch = self._batches[self._idx].float().contiguous().cuda()

        # Re-allocate the device buffer only when the shape changes.
        if self._buf is None or self._buf.shape != batch.shape:
            self._buf = torch.empty_like(batch, device="cuda")

        self._buf.copy_(batch)
        self._idx += 1
        log.debug(
            "Calibration batch %d / %d", self._idx, len(self._batches)
        )
        return [self._buf.data_ptr()]

    def read_calibration_cache(self) -> bytes | None:  # type: ignore[override]
        if self._cache_path and self._cache_path.is_file():
            log.info("Loading calibration cache from %s", self._cache_path)
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:  # type: ignore[override]
        if self._cache_path:
            self._cache_path.write_bytes(cache)
            log.info(
                "Calibration cache written (%d bytes) → %s",
                len(cache), self._cache_path,
            )


# ---------------------------------------------------------------------------
# Version-adaptive TRT helpers
# ---------------------------------------------------------------------------


def _make_logger(verbose: bool = False) -> "trt.Logger":
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    return trt.Logger(severity)


def _set_workspace(builder_config: "trt.IBuilderConfig", size_bytes: int) -> None:
    """Set builder workspace size — works on TRT 8.4 and 8.5+."""
    try:
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, size_bytes
        )
    except AttributeError:
        builder_config.max_workspace_size = size_bytes  # TRT ≤ 8.4


def _apply_precision(
    config: "trt.IBuilderConfig",
    precision: str,
    strict: bool,
) -> None:
    """Enable FP16 / INT8 builder flags."""
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    if strict:
        # PREFER_PRECISION_CONSTRAINTS replaces STRICT_TYPES in TRT 8.5+.
        for flag_name in ("PREFER_PRECISION_CONSTRAINTS", "STRICT_TYPES"):
            flag = getattr(trt.BuilderFlag, flag_name, None)
            if flag is not None:
                config.set_flag(flag)
                break


def _set_builder_opt_level(
    config: "trt.IBuilderConfig",
    level: int,
) -> None:
    """Set builder optimisation level (TRT 8.6+; silently ignored otherwise)."""
    try:
        config.builder_optimization_level = level
    except AttributeError:
        pass


def _apply_dla(
    config: "trt.IBuilderConfig",
    builder: "trt.Builder",
    dla_core: int,
) -> None:
    """Route layers to a DLA core when dla_core ≥ 0."""
    if dla_core < 0:
        return
    if not builder.platform_has_fast_dla:
        log.warning("Platform has no DLA; ignoring dla_core=%d.", dla_core)
        return
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    log.info("DLA core %d enabled with GPU fallback.", dla_core)


def _read_onnx_io(
    onnx_path: Path,
) -> tuple[str, list[int], list[str]]:
    """
    Read input name, input shape, and output names from an ONNX file.

    Dynamic dimensions (dim_value == 0) are returned as -1.
    Falls back to safe defaults when onnx is not installed.
    """
    if not _ONNX_AVAILABLE:
        log.warning("onnx not installed; using default I/O names.")
        return "images", [-1, 3, 224, 224], ["output"]

    proto = _onnx_mod.load(str(onnx_path))
    inp = proto.graph.input[0]
    shape = [
        d.dim_value if d.dim_value > 0 else -1
        for d in inp.type.tensor_type.shape.dim
    ]
    input_name = inp.name
    output_names = [o.name for o in proto.graph.output]
    return input_name, shape, output_names


def _build_serialized(
    builder: "trt.Builder",
    network: "trt.INetworkDefinition",
    config: "trt.IBuilderConfig",
) -> bytes:
    """
    Compile and serialise a TRT engine — compatible with TRT 8.x and 9.x/10.x.

    Returns raw engine bytes, or an empty bytes object on failure.
    """
    try:
        # Preferred API: TRT 8.5+
        serialized = builder.build_serialized_network(network, config)
        return bytes(serialized) if serialized is not None else b""
    except AttributeError:
        # Legacy API: TRT ≤ 8.4
        engine = builder.build_engine(network, config)
        if engine is None:
            return b""
        data = bytes(engine.serialize())
        del engine
        return data


def _get_engine_io_names(
    engine: "trt.ICudaEngine",
) -> tuple[list[str], list[str]]:
    """Return (input_names, output_names) — TRT 8/9/10 compatible."""
    try:
        # TRT 9+
        n = engine.num_io_tensors
        names = [engine.get_tensor_name(i) for i in range(n)]
        inputs = [
            n for n in names
            if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            n for n in names
            if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
    except AttributeError:
        # TRT 8
        inputs = [
            engine.get_binding_name(i)
            for i in range(engine.num_bindings)
            if engine.binding_is_input(i)
        ]
        outputs = [
            engine.get_binding_name(i)
            for i in range(engine.num_bindings)
            if not engine.binding_is_input(i)
        ]
    return inputs, outputs


def _set_input_shape_ctx(
    ctx: "trt.IExecutionContext",
    name: str,
    shape: tuple | torch.Size,
) -> None:
    """Set input shape in an execution context — TRT 8/9 compatible."""
    try:
        ctx.set_input_shape(name, tuple(shape))           # TRT 9+
    except AttributeError:
        engine = ctx.engine
        for i in range(engine.num_bindings):
            if engine.get_binding_name(i) == name:
                ctx.set_binding_shape(i, tuple(shape))
                break


def _get_output_shape_ctx(
    ctx: "trt.IExecutionContext",
    name: str,
) -> tuple[int, ...]:
    """Query output shape from execution context — TRT 8/9 compatible."""
    try:
        return tuple(int(d) for d in ctx.get_tensor_shape(name))  # TRT 9+
    except AttributeError:
        engine = ctx.engine
        for i in range(engine.num_bindings):
            if engine.get_binding_name(i) == name:
                return tuple(int(d) for d in ctx.get_binding_shape(i))
    return (-1,)


def _execute_ctx(
    ctx: "trt.IExecutionContext",
    input_ptrs: dict[str, int],
    output_ptrs: dict[str, int],
    stream_handle: int,
) -> None:
    """Execute asynchronously — TRT 8 (v2 API) and TRT 9+ (v3 API)."""
    all_ptrs = {**input_ptrs, **output_ptrs}
    try:
        # TRT 9+: set_tensor_address + execute_async_v3
        for name, ptr in all_ptrs.items():
            ctx.set_tensor_address(name, ptr)
        ctx.execute_async_v3(stream_handle)
    except AttributeError:
        # TRT 8: flat bindings list + execute_async_v2
        engine = ctx.engine
        bindings: list[int] = [0] * engine.num_bindings
        for i in range(engine.num_bindings):
            n = engine.get_binding_name(i)
            if n in all_ptrs:
                bindings[i] = all_ptrs[n]
        ctx.execute_async_v2(bindings, stream_handle)


# ---------------------------------------------------------------------------
# Sidecar / result helpers
# ---------------------------------------------------------------------------


def _write_sidecar(engine_path: Path, result: TRTBuildResult, extra: dict | None) -> Path:
    sidecar = engine_path.with_suffix(".json")
    payload = {
        "build_info": result.as_dict(),
        "metadata": extra or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(sidecar)
    return sidecar


def _failed(
    error: str,
    precision: str,
    elapsed: float,
) -> TRTBuildResult:
    return TRTBuildResult(
        success=False,
        engine_path=None,
        sidecar_path=None,
        precision=precision,
        input_name="",
        output_names=[],
        input_shape=(),
        engine_size_mb=0.0,
        calibration_cache_path=None,
        trt_version=".".join(str(v) for v in _TRT_VERSION),
        elapsed_seconds=elapsed,
        error_message=error,
    )


# ---------------------------------------------------------------------------
# Primary build function
# ---------------------------------------------------------------------------


def build_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    config: TRTBuildConfig | None = None,
    calibration_data: Sequence[torch.Tensor] | None = None,
    metadata: dict | None = None,
) -> TRTBuildResult:
    """
    Compile an ONNX model into a TensorRT engine file.

    Parameters
    ----------
    onnx_path        : Path to the source ``.onnx`` file.
    output_path      : Destination ``.trt`` engine file.
    config           : :class:`TRTBuildConfig`; defaults used if None.
    calibration_data : List of ``(B, C, H, W)`` float32 tensors for INT8
                       calibration.  Required when ``config.precision="int8"``
                       and no calibration cache exists.  Use
                       :func:`collect_calibration_batches` to build this list
                       from a DataLoader.
    metadata         : Optional dict embedded in the JSON sidecar.

    Returns
    -------
    TRTBuildResult with all outcome fields populated.

    Notes
    -----
    * Calibration cache at ``config.calibration_cache`` is read if present,
      skipping the calibration pass.  After calibration it is written there.
    * For INT8 without calibration data AND without an existing cache, the
      build falls back to FP16 with a warning.
    """
    if not _TRT_AVAILABLE:
        return _failed(
            "tensorrt is not installed.  Install via JetPack or pip.",
            (config or TRTBuildConfig()).precision,
            0.0,
        )

    t0 = time.perf_counter()
    cfg = config or TRTBuildConfig()
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        return _failed(
            f"ONNX file not found: {onnx_path}",
            cfg.precision,
            time.perf_counter() - t0,
        )

    # ── Read ONNX I/O metadata ─────────────────────────────────────────────
    input_name, onnx_shape, output_names = _read_onnx_io(onnx_path)
    # onnx_shape: [batch, C, H, W] — batch may be -1 for dynamic models.

    # ── INT8 calibration guard ────────────────────────────────────────────
    precision = cfg.precision
    if precision == "int8":
        cache_ok = (
            cfg.calibration_cache is not None
            and Path(cfg.calibration_cache).is_file()
        )
        data_ok = calibration_data is not None and len(calibration_data) > 0
        if not cache_ok and not data_ok:
            log.warning(
                "INT8 requested but no calibration_data or cache provided. "
                "Falling back to FP16."
            )
            precision = "fp16"

    log.info(
        "Building TRT engine | precision=%s | onnx=%s", precision, onnx_path.name
    )

    # ── TRT build context ─────────────────────────────────────────────────
    logger = _make_logger(cfg.verbose)
    runtime = trt.Runtime(logger)

    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder:
        with builder.create_network(EXPLICIT_BATCH) as network:
            with trt.OnnxParser(network, logger) as parser:

                # Parse ONNX
                if not parser.parse_from_file(str(onnx_path)):
                    errors = [
                        parser.get_error(i).desc()
                        for i in range(parser.num_errors)
                    ]
                    return _failed(
                        "ONNX parse errors: " + "; ".join(errors),
                        precision,
                        time.perf_counter() - t0,
                    )

                # Builder config
                build_cfg = builder.create_builder_config()
                _set_workspace(build_cfg, cfg.workspace_mb * (1 << 20))
                _apply_precision(build_cfg, precision, cfg.strict_precision)
                _set_builder_opt_level(build_cfg, cfg.builder_opt_level)
                _apply_dla(build_cfg, builder, cfg.dla_core)

                # Optimisation profile (required for dynamic or explicit batch)
                profile = builder.create_optimization_profile()
                c, h, w = (
                    onnx_shape[1] if len(onnx_shape) > 1 else 3,
                    onnx_shape[2] if len(onnx_shape) > 2 else 224,
                    onnx_shape[3] if len(onnx_shape) > 3 else 224,
                )
                profile.set_shape(
                    input_name,
                    min=(cfg.min_batch, c, h, w),
                    opt=(cfg.opt_batch, c, h, w),
                    max=(cfg.max_batch, c, h, w),
                )
                build_cfg.add_optimization_profile(profile)

                # INT8 calibrator
                calibrator: Int8Calibrator | None = None
                if precision == "int8":
                    cal_data = list(calibration_data) if calibration_data else []
                    calibrator = Int8Calibrator(
                        batches=cal_data[: cfg.calibration_batches],
                        cache_path=cfg.calibration_cache,
                    )
                    build_cfg.int8_calibrator = calibrator

                # Compile
                engine_bytes = _build_serialized(builder, network, build_cfg)

    if not engine_bytes:
        return _failed(
            "TRT builder returned None — check GPU memory and ONNX validity.",
            precision,
            time.perf_counter() - t0,
        )

    # ── Serialise to disk ─────────────────────────────────────────────────
    output_path.write_bytes(engine_bytes)
    engine_size_mb = len(engine_bytes) / (1024 ** 2)

    cal_cache_path: Path | None = None
    if calibrator is not None and cfg.calibration_cache is not None:
        cal_cache_path = Path(cfg.calibration_cache)

    input_shape = (cfg.opt_batch, c, h, w)

    result = TRTBuildResult(
        success=True,
        engine_path=output_path,
        sidecar_path=None,
        precision=precision,
        input_name=input_name,
        output_names=output_names,
        input_shape=input_shape,
        engine_size_mb=round(engine_size_mb, 3),
        calibration_cache_path=cal_cache_path,
        trt_version=".".join(str(v) for v in _TRT_VERSION),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
        error_message=None,
    )

    result.sidecar_path = _write_sidecar(output_path, result, metadata)

    log.info(
        "Engine built | %.2f MB | precision=%s | %.1f s",
        engine_size_mb, precision, result.elapsed_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# Load engine
# ---------------------------------------------------------------------------


def load_engine(
    engine_path: str | Path,
    *,
    device_id: int = 0,
) -> "trt.ICudaEngine":
    """
    Deserialise and load a TensorRT engine from disk.

    Parameters
    ----------
    engine_path : Path to the ``.trt`` engine file.
    device_id   : CUDA device index (use the same device used during build).

    Returns
    -------
    ``trt.ICudaEngine`` ready for inference.

    Raises
    ------
    RuntimeError
        When tensorrt is not installed or deserialisation fails.
    """
    if not _TRT_AVAILABLE:
        raise RuntimeError("tensorrt is not installed.")

    engine_path = Path(engine_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine file not found: {engine_path}")

    torch.cuda.set_device(device_id)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(
            f"TRT deserialisation failed for {engine_path}.  "
            "The engine may have been built for a different TRT / GPU version."
        )
    log.info(
        "Engine loaded from %s  (%.2f MB)",
        engine_path, len(engine_bytes) / (1024 ** 2),
    )
    return engine


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    engine: "trt.ICudaEngine",
    images: torch.Tensor,
    *,
    context: "trt.IExecutionContext | None" = None,
    device_id: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Run forward inference on a TensorRT engine.

    Parameters
    ----------
    engine    : Loaded ``ICudaEngine`` (from :func:`load_engine`).
    images    : ``(B, C, H, W)`` float32 tensor on CUDA.
    context   : Optional pre-created execution context.  When None a new
                context is created per call (small overhead).
    device_id : CUDA device index.

    Returns
    -------
    dict mapping each output name → output tensor on the same CUDA device.
    """
    if not _TRT_AVAILABLE:
        raise RuntimeError("tensorrt is not installed.")

    images = images.float().contiguous().cuda(device_id)
    input_names, output_names = _get_engine_io_names(engine)

    ctx = context if context is not None else engine.create_execution_context()

    # Set input shape for dynamic profiles.
    _set_input_shape_ctx(ctx, input_names[0], images.shape)

    # Allocate output tensors using shapes from the context.
    outputs: dict[str, torch.Tensor] = {}
    for name in output_names:
        shape = _get_output_shape_ctx(ctx, name)
        outputs[name] = torch.empty(
            shape, dtype=torch.float32, device=f"cuda:{device_id}"
        )

    # Execute
    input_ptrs = {input_names[0]: images.data_ptr()}
    output_ptrs = {name: t.data_ptr() for name, t in outputs.items()}
    stream = torch.cuda.current_stream(device_id)
    _execute_ctx(ctx, input_ptrs, output_ptrs, stream.cuda_stream)
    torch.cuda.synchronize(device_id)

    return outputs


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def benchmark_engine(
    engine: "trt.ICudaEngine",
    input_shape: tuple[int, ...],
    *,
    n_warmup: int = 10,
    n_runs: int = 100,
    device_id: int = 0,
) -> dict:
    """
    Measure inference latency and throughput on the current device.

    Parameters
    ----------
    engine       : Loaded TRT engine.
    input_shape  : ``(B, C, H, W)`` — shape of the dummy input.
    n_warmup     : Warm-up iterations (discarded).
    n_runs       : Timed iterations.
    device_id    : CUDA device index.

    Returns
    -------
    dict with keys:
      mean_ms, std_ms, min_ms, max_ms, p50_ms, p95_ms, p99_ms,
      throughput_fps, batch_size, n_runs, trt_version, device.
    """
    if not _TRT_AVAILABLE:
        raise RuntimeError("tensorrt is not installed.")

    dummy = torch.randn(*input_shape, device=f"cuda:{device_id}")
    ctx = engine.create_execution_context()

    # Warm-up
    for _ in range(n_warmup):
        run_inference(engine, dummy, context=ctx, device_id=device_id)

    # Timed runs
    latencies: list[float] = []
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)

    for _ in range(n_runs):
        start_e.record()
        run_inference(engine, dummy, context=ctx, device_id=device_id)
        end_e.record()
        torch.cuda.synchronize(device_id)
        latencies.append(start_e.elapsed_time(end_e))  # milliseconds

    arr = np.array(latencies, dtype=np.float64)
    batch_size = int(input_shape[0])
    mean_ms = float(arr.mean())

    return {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(float(arr.std()), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "throughput_fps": round(batch_size * 1_000 / mean_ms, 2),
        "batch_size": batch_size,
        "n_runs": n_runs,
        "trt_version": ".".join(str(v) for v in _TRT_VERSION),
        "device": torch.cuda.get_device_name(device_id),
    }


# ---------------------------------------------------------------------------
# High-level pipeline wrapper
# ---------------------------------------------------------------------------


def export_from_onnx_result(
    onnx_result: object,
    output_dir: str | Path,
    *,
    config: TRTBuildConfig | None = None,
    calibration_data: Sequence[torch.Tensor] | None = None,
    run_benchmark: bool = True,
) -> TRTBuildResult:
    """
    Build a TRT engine from an :class:`~deployment.export_onnx.ONNXExportResult`.

    Derives the engine output path and metadata automatically from the ONNX
    result.  Optionally benchmarks the engine after building.

    Parameters
    ----------
    onnx_result      : ``ONNXExportResult`` from ``export_onnx.export_to_onnx``
                       or ``export_from_candidate``.
    output_dir       : Root directory for TRT engine files.  The engine is
                       placed at ``<output_dir>/<fingerprint>/model.trt``.
    config           : Build configuration.
    calibration_data : INT8 calibration tensors.
    run_benchmark    : If True and build succeeded, run a quick latency
                       benchmark and append results to the sidecar JSON.

    Returns
    -------
    TRTBuildResult.
    """
    onnx_path = getattr(onnx_result, "onnx_path", None)
    if onnx_path is None:
        raise ValueError("onnx_result.onnx_path is None — export must have succeeded.")

    onnx_path = Path(onnx_path)
    sidecar_onnx = getattr(onnx_result, "sidecar_path", None)

    # Reconstruct metadata from the ONNX sidecar if available.
    meta: dict = {}
    if sidecar_onnx and Path(sidecar_onnx).is_file():
        try:
            meta = json.loads(Path(sidecar_onnx).read_text()).get("metadata", {})
        except Exception:
            pass

    fingerprint = meta.get("fingerprint", onnx_path.parent.name)
    output_dir = Path(output_dir)
    engine_dir = output_dir / fingerprint
    engine_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or TRTBuildConfig()
    engine_path = engine_dir / f"model_{cfg.precision}.trt"

    if cfg.calibration_cache is None and cfg.precision == "int8":
        cfg = TRTBuildConfig(
            **{**asdict(cfg), "calibration_cache": engine_dir / "calibration.cache"}
        )

    result = build_engine(
        onnx_path=onnx_path,
        output_path=engine_path,
        config=cfg,
        calibration_data=calibration_data,
        metadata=meta,
    )

    # Optional post-build benchmark
    if result.success and run_benchmark and torch.cuda.is_available():
        try:
            engine = load_engine(engine_path)
            bm = benchmark_engine(engine, result.input_shape)
            log.info(
                "Benchmark | mean=%.2f ms | p95=%.2f ms | %.1f FPS",
                bm["mean_ms"], bm["p95_ms"], bm["throughput_fps"],
            )
            # Append benchmark to sidecar
            if result.sidecar_path and Path(result.sidecar_path).is_file():
                sidecar_data = json.loads(Path(result.sidecar_path).read_text())
                sidecar_data["benchmark"] = bm
                Path(result.sidecar_path).write_text(
                    json.dumps(sidecar_data, indent=2, default=str)
                )
        except Exception as exc:
            log.warning("Post-build benchmark failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def collect_calibration_batches(
    dataloader: "torch.utils.data.DataLoader",
    n_batches: int = 100,
    *,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    Collect up to *n_batches* image batches from a DataLoader.

    Images are kept on *device* (use "cpu" to minimise GPU memory during
    calibration setup; the calibrator moves them to CUDA on demand).

    Parameters
    ----------
    dataloader : PyTorch DataLoader yielding tensors or (tensor, label) tuples.
    n_batches  : Maximum number of batches to collect.
    device     : Target device ("cpu" or "cuda").

    Returns
    -------
    list of ``(B, C, H, W)`` float32 tensors.
    """
    batches: list[torch.Tensor] = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        batches.append(images.to(device).float())
    log.info(
        "Collected %d calibration batches (total %d images).",
        len(batches),
        sum(b.shape[0] for b in batches),
    )
    return batches


def check_requirements() -> dict[str, bool | str]:
    """
    Report TensorRT availability and version.

    Returns
    -------
    dict with "tensorrt" (bool), "trt_version" (str), "cuda_available" (bool).
    """
    status: dict[str, bool | str] = {
        "tensorrt": _TRT_AVAILABLE,
        "trt_version": ".".join(str(v) for v in _TRT_VERSION),
        "cuda_available": torch.cuda.is_available(),
        "onnx_available": _ONNX_AVAILABLE,
    }
    for k, v in status.items():
        log.info("%-20s %s", k, v)
    return status
