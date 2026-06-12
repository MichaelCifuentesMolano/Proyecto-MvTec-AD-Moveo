"""
src/deployment/export_onnx.py

ONNX export pipeline for quantised PyTorch anomaly-detection models.

Responsibilities
----------------
* Convert trained models (including QAT models with fake-quantise nodes) to
  ONNX by tracing with ``torch.onnx.export``.
* Disable fake-quantise modules before tracing so the exported graph is clean
  FP32 (suitable as input for TensorRT FP16 / INT8 compilation downstream).
* Wrap dict-output models into a tensor-output shim compatible with ONNX
  tracing; the output key (anomaly_map / score / all) is resolved at export
  time by probing the model once.
* Optionally simplify the graph with onnx-simplifier (onnxsim).
* Optionally validate numerical consistency against PyTorch via ONNXRuntime.
* Embed key-value metadata into the ONNX ``metadata_props`` field and write a
  JSON sidecar alongside each exported file.
* Expose a high-level ``export_from_candidate`` entry-point that derives paths
  and metadata automatically from a NAS candidate dict.

Assumptions
-----------
* Models follow the unified forward schema, returning either a plain tensor or
  a dict with any subset of {"recon", "features", "anomaly_map", "score",
  "logits"}.  Absent keys are tolerated.
* Input tensors are ImageNet-normalised float32, shape (B, 3, H, W).
* QAT fake-quantise modules expose the standard
  ``torch.ao.quantization.FakeQuantize`` interface or a compatible
  ``disable_fake_quant()`` method.
* onnx (≥1.14), onnxruntime (≥1.16), and onnx-simplifier (≥0.4) are optional;
  missing packages produce logged warnings, not hard failures.

Downstream usage
----------------
The ONNX files produced here are intended as inputs to ``export_tensorrt.py``
for TRT engine compilation on the Jetson Orin Nano.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import onnx
    import onnx.checker
    _ONNX_AVAILABLE = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

try:
    from onnxsim import simplify as _onnxsim_fn
    _ONNXSIM_AVAILABLE = True
except ImportError:
    _onnxsim_fn = None  # type: ignore[assignment]
    _ONNXSIM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPSET: int = 17           # TensorRT 8.6+ supports up to opset 17
INPUT_NAME: str = "images"        # canonical ONNX input node name

# Priority order for selecting a fallback output key.
_KEY_PRIORITY = ("anomaly_map", "score", "logits", "recon", "features")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ONNXExportConfig:
    """
    Hyperparameters controlling the ONNX export process.

    Parameters
    ----------
    opset : int
        ONNX opset version.  TensorRT 8.x supports opcodes up to opset 17.
    output_mode : {"anomaly_map", "score", "all"}
        Which model output(s) to include in the exported graph.

        "anomaly_map" — single spatial heatmap (B, 1, H, W); recommended for
                        the TensorRT tracking pipeline.
        "score"       — single image-level score (B,) or (B, 1).
        "all"         — every non-None output from the forward pass, exported
                        as separate named ONNX outputs.

    dynamic_batch : bool
        Export with a dynamic batch-size axis (dim 0).  Slightly reduces
        TensorRT optimisation quality; use False for fixed single-image
        inference on Jetson.
    simplify : bool
        Run onnx-simplifier after export.  Reduces operator count and can
        improve TRT compilation time.
    validate : bool
        Run ONNXRuntime inference on a random input and compare outputs
        against PyTorch (tolerance controlled by ``rtol`` / ``atol``).
    copy_model : bool
        Deep-copy the model before modification.  Set False only when GPU RAM
        is insufficient to hold two copies simultaneously.
    rtol, atol : float
        Relative / absolute tolerances for numerical validation.
    do_constant_folding : bool
        Fold constant sub-expressions during tracing.
    verbose : bool
        Enable verbose output from ``torch.onnx.export``.
    """

    opset: int = DEFAULT_OPSET
    output_mode: Literal["anomaly_map", "score", "all"] = "anomaly_map"
    dynamic_batch: bool = False
    simplify: bool = True
    validate: bool = True
    copy_model: bool = True
    rtol: float = 1e-3
    atol: float = 1e-4
    do_constant_folding: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ONNXExportResult:
    """
    Outcome of a single ONNX export attempt.

    Attributes
    ----------
    success : bool
    onnx_path : Path or None
    sidecar_path : Path or None
        JSON sidecar written beside the ONNX file.
    input_name : str
    output_names : list[str]
    input_shape : tuple — (B, C, H, W) as exported.
    output_shapes : list[tuple]
    model_size_mb : float
    validation_passed : bool
    validation_max_diff : float
        Maximum absolute element-wise difference between PyTorch and
        ONNXRuntime outputs.
    simplified : bool
    error_message : str or None
    elapsed_seconds : float
    """

    success: bool
    onnx_path: Path | None
    sidecar_path: Path | None
    input_name: str
    output_names: list[str]
    input_shape: tuple
    output_shapes: list[tuple]
    model_size_mb: float
    validation_passed: bool
    validation_max_diff: float
    simplified: bool
    error_message: str | None
    elapsed_seconds: float

    def as_dict(self) -> dict:
        """JSON-serialisable representation."""
        d = asdict(self)
        d["onnx_path"] = str(self.onnx_path) if self.onnx_path else None
        d["sidecar_path"] = str(self.sidecar_path) if self.sidecar_path else None
        d["input_shape"] = list(self.input_shape)
        d["output_shapes"] = [list(s) for s in self.output_shapes]
        return d


# ---------------------------------------------------------------------------
# ONNX-compatible model wrappers
# ---------------------------------------------------------------------------


class _SingleOutputWrapper(nn.Module):
    """Return a single tensor from a dict-output model — tracing-safe."""

    def __init__(self, model: nn.Module, key: str) -> None:
        super().__init__()
        self.model = model
        self._key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.model(x)
        return out[self._key] if isinstance(out, dict) else out


class _MultiOutputWrapper(nn.Module):
    """Return a fixed-length tuple from a dict-output model — tracing-safe."""

    def __init__(self, model: nn.Module, keys: list[str]) -> None:
        super().__init__()
        self.model = model
        self._keys = keys

    def forward(self, x: torch.Tensor) -> tuple:  # type: ignore[override]
        out = self.model(x)
        if isinstance(out, dict):
            return tuple(out[k] for k in self._keys)
        return (out,)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _disable_fake_quantize(model: nn.Module) -> None:
    """
    Set all fake-quantise modules to pass-through (identity) mode.

    Handles three interfaces in order of preference:
    1. ``module.disable_fake_quant()`` — standard ``torch.ao`` / custom QAT.
    2. ``module.disable_observer()``   — paired observer disablement.
    3. Direct ``buffer[0] = 0`` on ``fake_quant_enabled`` / ``observer_enabled``
       buffers — legacy / custom QAT modules.
    """
    for module in model.modules():
        for method in ("disable_fake_quant", "disable_observer"):
            fn = getattr(module, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        for attr in ("fake_quant_enabled", "observer_enabled"):
            buf = getattr(module, attr, None)
            if buf is not None and hasattr(buf, "__setitem__"):
                try:
                    buf[0] = 0
                except Exception:
                    pass


def _prepare_model(
    model: nn.Module,
    config: ONNXExportConfig,
    device: torch.device,
) -> nn.Module:
    """
    Return an eval-mode, QAT-disabled model ready for ONNX tracing.

    Deep-copies first (unless ``config.copy_model`` is False) to avoid
    mutating the caller's model.
    """
    m = copy.deepcopy(model) if config.copy_model else model
    m.eval()
    _disable_fake_quantize(m)
    m.to(device)
    return m


def _probe_forward(
    model: nn.Module,
    dummy: torch.Tensor,
) -> dict | torch.Tensor:
    """Single no-grad forward pass to inspect the output structure."""
    with torch.no_grad():
        return model(dummy)


def _resolve_output_keys(
    probe: dict | torch.Tensor,
    mode: str,
) -> list[str]:
    """
    Determine which output keys to export given the probed model output.

    Returns a list of dict keys.  The sentinel ``["__tensor__"]`` means the
    model already returns a plain tensor and no wrapping is needed.

    Raises
    ------
    RuntimeError
        When no usable key can be identified (empty dict, all None values).
    """
    if not isinstance(probe, dict):
        return ["__tensor__"]

    available = {k for k, v in probe.items() if v is not None}
    if not available:
        raise RuntimeError("Model forward returned a dict with all-None values.")

    if mode == "all":
        # Emit keys in canonical priority order, then alphabetically for the rest.
        ordered = [k for k in _KEY_PRIORITY if k in available]
        ordered += sorted(available - set(ordered))
        return ordered

    # Single-key modes: "anomaly_map" or "score".
    if mode in available:
        return [mode]

    # Fall back through the priority list.
    for k in _KEY_PRIORITY:
        if k in available:
            log.warning(
                "Requested output key %r absent; exporting %r instead.", mode, k
            )
            return [k]

    raise RuntimeError(
        f"No usable output key found.  Available keys: {sorted(available)}"
    )


def _build_wrapper(model: nn.Module, keys: list[str]) -> nn.Module:
    """
    Wrap the model so ONNX tracing receives a tensor or fixed-length tuple.
    """
    if keys == ["__tensor__"]:
        return model
    if len(keys) == 1:
        return _SingleOutputWrapper(model, keys[0])
    return _MultiOutputWrapper(model, keys)


def _run_pytorch_forward(
    wrapper: nn.Module,
    dummy: torch.Tensor,
) -> list[np.ndarray]:
    """Collect PyTorch reference outputs as CPU numpy arrays."""
    with torch.no_grad():
        out = wrapper(dummy)
    if isinstance(out, torch.Tensor):
        return [out.cpu().numpy()]
    return [t.cpu().numpy() for t in out]


def _dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {INPUT_NAME: {0: "batch_size"}}
    for name in output_names:
        axes[name] = {0: "batch_size"}
    return axes


def _validate_onnx(
    onnx_path: Path,
    dummy_np: np.ndarray,
    torch_outputs: list[np.ndarray],
    rtol: float,
    atol: float,
) -> tuple[bool, float]:
    """
    Validate ONNX model with ONNXRuntime against PyTorch reference outputs.

    Returns
    -------
    (passed, max_absolute_difference)
        ``passed`` is False when onnx / onnxruntime are unavailable or when
        numerical comparison fails.
    """
    if not _ORT_AVAILABLE:
        log.warning("onnxruntime not installed — skipping numerical validation.")
        return False, float("nan")

    if _ONNX_AVAILABLE:
        try:
            onnx.checker.check_model(str(onnx_path))
        except Exception as exc:
            log.error("ONNX graph check failed: %s", exc)
            return False, float("nan")

    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        ort_out = session.run(None, {session.get_inputs()[0].name: dummy_np})
    except Exception as exc:
        log.error("ONNXRuntime inference failed: %s", exc)
        return False, float("nan")

    max_diff = 0.0
    passed = True
    for pt, rt in zip(torch_outputs, ort_out):
        diff = float(np.max(np.abs(pt - rt)))
        max_diff = max(max_diff, diff)
        if not np.allclose(pt, rt, rtol=rtol, atol=atol):
            log.warning(
                "Output mismatch: max_abs_diff=%.6f (rtol=%.1e, atol=%.1e)",
                diff, rtol, atol,
            )
            passed = False

    if passed:
        log.info("ONNX validation passed (max_abs_diff=%.6f).", max_diff)
    return passed, max_diff


def _simplify_onnx(onnx_path: Path) -> bool:
    """
    Run onnx-simplifier on the ONNX file in-place.

    Returns True on success, False when unavailable or when simplification
    does not pass its own internal check.
    """
    if not (_ONNXSIM_AVAILABLE and _ONNX_AVAILABLE):
        log.warning("onnx-simplifier or onnx not installed — skipping simplification.")
        return False

    try:
        model_proto = onnx.load(str(onnx_path))
        simplified, ok = _onnxsim_fn(model_proto)
        if ok:
            onnx.save(simplified, str(onnx_path))
            log.info("onnxsim simplification succeeded.")
            return True
        log.warning("onnxsim internal check failed — retaining original graph.")
        return False
    except Exception as exc:
        log.warning("onnxsim error: %s", exc)
        return False


def _embed_metadata(onnx_path: Path, metadata: dict) -> None:
    """Embed key-value string pairs into the ONNX model's metadata_props."""
    if not _ONNX_AVAILABLE:
        return
    try:
        proto = onnx.load(str(onnx_path))
        for k, v in metadata.items():
            entry = proto.metadata_props.add()
            entry.key = str(k)
            entry.value = str(v)
        onnx.save(proto, str(onnx_path))
    except Exception as exc:
        log.warning("Could not embed ONNX metadata: %s", exc)


def _write_sidecar(
    onnx_path: Path,
    result: ONNXExportResult,
    extra: dict | None,
) -> Path:
    """Write a JSON sidecar atomically beside the ONNX file."""
    sidecar = onnx_path.with_suffix(".json")
    payload = {
        "export_info": result.as_dict(),
        "metadata": extra or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(sidecar)
    return sidecar


def _failed_result(
    error: str,
    input_shape: tuple,
    elapsed: float,
) -> ONNXExportResult:
    return ONNXExportResult(
        success=False,
        onnx_path=None,
        sidecar_path=None,
        input_name=INPUT_NAME,
        output_names=[],
        input_shape=input_shape,
        output_shapes=[],
        model_size_mb=0.0,
        validation_passed=False,
        validation_max_diff=float("nan"),
        simplified=False,
        error_message=error,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Primary export function
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    config: ONNXExportConfig | None = None,
    device: str | torch.device = "cpu",
    metadata: dict | None = None,
) -> ONNXExportResult:
    """
    Convert a PyTorch anomaly-detection model to ONNX.

    Pipeline
    --------
    1. Deep-copy and set model to eval mode; disable all fake-quantise nodes.
    2. Probe the model with a random input to determine output structure.
    3. Wrap the model so ``torch.onnx.export`` receives a tensor output.
    4. Collect PyTorch reference outputs for later validation.
    5. Export via ``torch.onnx.export`` (tracing-based).
    6. Optionally simplify with onnx-simplifier.
    7. Optionally validate with ONNXRuntime against reference outputs.
    8. Embed metadata; write JSON sidecar.

    Parameters
    ----------
    model       : Trained PyTorch model (QAT-aware or plain).
    output_path : Destination ``.onnx`` file path.
    input_shape : Export input shape ``(B, C, H, W)``; default ``(1,3,224,224)``.
    config      : :class:`ONNXExportConfig`; defaults used if None.
    device      : Device for tracing (use "cpu" for reproducibility; the
                  exported graph runs on any device via TRT/ORT).
    metadata    : Optional dict of string key-value pairs embedded in the ONNX
                  file and sidecar (e.g. fingerprint, AUROC, category).

    Returns
    -------
    ONNXExportResult with all outcome fields populated.
    """
    t0 = time.perf_counter()
    cfg = config or ONNXExportConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_ = torch.device(device)

    # ── 1. Prepare model ───────────────────────────────────────────────────
    try:
        prepared = _prepare_model(model, cfg, device_)
    except Exception as exc:
        return _failed_result(
            f"Model preparation failed: {exc}", input_shape,
            time.perf_counter() - t0,
        )

    dummy = torch.randn(*input_shape, device=device_)

    # ── 2. Probe output structure ──────────────────────────────────────────
    try:
        probe = _probe_forward(prepared, dummy)
        keys = _resolve_output_keys(probe, cfg.output_mode)
    except Exception as exc:
        return _failed_result(
            f"Model probe failed: {exc}", input_shape,
            time.perf_counter() - t0,
        )

    output_names = keys if keys != ["__tensor__"] else ["output"]

    # ── 3. Wrap model ──────────────────────────────────────────────────────
    wrapper = _build_wrapper(prepared, keys)

    # ── 4. Collect PyTorch reference outputs ──────────────────────────────
    try:
        torch_outputs = _run_pytorch_forward(wrapper, dummy)
    except Exception as exc:
        return _failed_result(
            f"Reference forward pass failed: {exc}", input_shape,
            time.perf_counter() - t0,
        )

    output_shapes = [tuple(o.shape) for o in torch_outputs]

    # ── 5. torch.onnx.export ───────────────────────────────────────────────
    export_kwargs: dict = {
        "export_params": True,
        "opset_version": cfg.opset,
        "do_constant_folding": cfg.do_constant_folding,
        "input_names": [INPUT_NAME],
        "output_names": output_names,
        "verbose": cfg.verbose,
    }
    if cfg.dynamic_batch:
        export_kwargs["dynamic_axes"] = _dynamic_axes(output_names)

    # TrainingMode.EVAL is stable through PyTorch 2.x; guard for future removal.
    try:
        export_kwargs["training"] = torch.onnx.TrainingMode.EVAL
    except AttributeError:
        pass

    try:
        with torch.no_grad():
            torch.onnx.export(wrapper, (dummy,), str(output_path), **export_kwargs)
        log.info("ONNX model written: %s", output_path)
    except Exception as exc:
        return _failed_result(
            f"torch.onnx.export failed: {exc}", input_shape,
            time.perf_counter() - t0,
        )

    # ── 6. Simplification ─────────────────────────────────────────────────
    simplified = _simplify_onnx(output_path) if cfg.simplify else False

    # ── 7. Validation ─────────────────────────────────────────────────────
    val_passed, max_diff = False, float("nan")
    if cfg.validate:
        val_passed, max_diff = _validate_onnx(
            output_path,
            dummy.cpu().numpy(),
            torch_outputs,
            cfg.rtol,
            cfg.atol,
        )

    # ── 8. Metadata + sidecar ─────────────────────────────────────────────
    model_size_mb = output_path.stat().st_size / (1024 ** 2)

    embed = {
        "framework": "PyTorch-NSGA2-NAS",
        "opset": str(cfg.opset),
        "output_mode": cfg.output_mode,
        "input_shape": "x".join(str(d) for d in input_shape),
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }
    _embed_metadata(output_path, embed)

    result = ONNXExportResult(
        success=True,
        onnx_path=output_path,
        sidecar_path=None,           # filled in after sidecar write
        input_name=INPUT_NAME,
        output_names=output_names,
        input_shape=input_shape,
        output_shapes=output_shapes,
        model_size_mb=round(model_size_mb, 3),
        validation_passed=val_passed,
        validation_max_diff=float(max_diff),
        simplified=simplified,
        error_message=None,
        elapsed_seconds=round(time.perf_counter() - t0, 3),
    )

    result.sidecar_path = _write_sidecar(output_path, result, metadata)

    log.info(
        "Export complete | size=%.2f MB | validated=%s | simplified=%s | %.1f s",
        model_size_mb, val_passed, simplified, result.elapsed_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# High-level pipeline wrappers
# ---------------------------------------------------------------------------


def export_from_candidate(
    model: nn.Module,
    candidate_dict: dict,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    config: ONNXExportConfig | None = None,
) -> ONNXExportResult:
    """
    Export a model associated with a NAS candidate dict.

    Output layout::

        <output_dir>/<fingerprint>/model.onnx
        <output_dir>/<fingerprint>/model.json

    The input resolution and metadata are derived automatically from
    ``candidate_dict`` (fields: ``fingerprint``, ``arch.input_size``,
    ``arch.family``, ``auroc``, ``latency_ms``, ``peak_ram_mb``,
    ``energy_mj``, ``generation``).

    Parameters
    ----------
    model          : Trained PyTorch model corresponding to *candidate_dict*.
    candidate_dict : NAS candidate dict produced by ``encoding.py`` or
                     ``nsga2_engine.pareto_candidates``.
    output_dir     : Root directory for ONNX artefacts.
    device         : Tracing device.
    config         : Export configuration; defaults used if None.

    Returns
    -------
    ONNXExportResult.
    """
    output_dir = Path(output_dir)
    fingerprint = str(candidate_dict.get("fingerprint", "unknown"))

    arch = candidate_dict.get("arch", {})
    input_size = int(arch.get("input_size", 224))
    input_shape = (1, 3, input_size, input_size)

    onnx_dir = output_dir / fingerprint
    onnx_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "fingerprint": fingerprint,
        "arch_family": str(arch.get("family", "")),
        "input_size": str(input_size),
        "auroc": str(candidate_dict.get("auroc", "")),
        "latency_ms": str(candidate_dict.get("latency_ms", "")),
        "peak_ram_mb": str(candidate_dict.get("peak_ram_mb", "")),
        "energy_mj": str(candidate_dict.get("energy_mj", "")),
        "generation": str(candidate_dict.get("generation", "")),
        "is_pareto": str(candidate_dict.get("is_pareto", "")),
    }

    return export_to_onnx(
        model=model,
        output_path=onnx_dir / "model.onnx",
        input_shape=input_shape,
        config=config,
        device=device,
        metadata=metadata,
    )


def batch_export(
    models_and_candidates: list[tuple[nn.Module, dict]],
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    config: ONNXExportConfig | None = None,
    stop_on_failure: bool = False,
) -> list[ONNXExportResult]:
    """
    Export multiple models sequentially.

    Parameters
    ----------
    models_and_candidates : List of ``(model, candidate_dict)`` pairs.
    output_dir            : Root output directory shared by all exports.
    device                : Tracing device.
    config                : Shared export configuration.
    stop_on_failure       : Raise ``RuntimeError`` on the first failed export.

    Returns
    -------
    List of :class:`ONNXExportResult`, one per input pair, in order.
    """
    output_dir = Path(output_dir)
    results: list[ONNXExportResult] = []
    n = len(models_and_candidates)

    for i, (model, cand) in enumerate(models_and_candidates):
        fp = cand.get("fingerprint", f"candidate_{i}")
        log.info("Exporting %d/%d  fingerprint=%s", i + 1, n, fp)
        res = export_from_candidate(
            model, cand, output_dir, device=device, config=config
        )
        results.append(res)

        if not res.success:
            log.error("Export failed for %s: %s", fp, res.error_message)
            if stop_on_failure:
                raise RuntimeError(
                    f"Export failed for candidate {fp}: {res.error_message}"
                )

    n_ok = sum(r.success for r in results)
    log.info("Batch export complete: %d / %d succeeded.", n_ok, n)
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def check_requirements() -> dict[str, bool]:
    """
    Report which optional ONNX-related packages are available.

    Returns
    -------
    dict with boolean values for keys ``"onnx"``, ``"onnxruntime"``,
    ``"onnxsim"``.
    """
    status = {
        "onnx": _ONNX_AVAILABLE,
        "onnxruntime": _ORT_AVAILABLE,
        "onnxsim": _ONNXSIM_AVAILABLE,
    }
    for pkg, ok in status.items():
        level = logging.INFO if ok else logging.WARNING
        log.log(level, "%-15s %s", pkg, "✓" if ok else "✗ (not installed)")
    return status


def load_sidecar(onnx_path: str | Path) -> dict:
    """
    Load the JSON sidecar written alongside an exported ONNX file.

    Parameters
    ----------
    onnx_path : Path to the ``.onnx`` file (not the sidecar itself).

    Returns
    -------
    Parsed sidecar dict, or an empty dict if the sidecar does not exist.
    """
    sidecar = Path(onnx_path).with_suffix(".json")
    if not sidecar.is_file():
        log.warning("No sidecar found at %s", sidecar)
        return {}
    return json.loads(sidecar.read_text())
