"""
src/quantization/quant_error.py
===============================

Quantization-sensitivity / robustness measurement utilities.

Computes three families of diagnostics for a model that has already been
wrapped by ``src.quantization.qat_wrapper.wrap_for_qat``:

1. **Layer-wise weight error** — per ``QConv2d`` / ``QLinear``: MSE,
   max-abs error, and cosine similarity between the floating-point weight
   tensor and its fake-quantized version.
2. **Activation drift** — per layer: MSE, max-abs error, and cosine
   similarity between the layer's output activations under FP behavior
   and under fake-quantized behavior, on a small calibration batch.
3. **Accuracy drop after fake quantization** — task-level metric (e.g.
   AUROC) computed once with quantization enabled and once disabled,
   returning the drop as a robustness signal.

The aggregated output is consumable by the NSGA-II fitness function as a
stability/robustness metric — see :func:`scalar_robustness_score`.

Public interface
----------------
- :func:`measure_layerwise_error`        — per-layer weight + activation drift.
- :func:`measure_accuracy_drop`          — task-level fp ↔ quantized comparison.
- :func:`measure_quant_sensitivity`      — full report (combines both).
- :func:`scalar_robustness_score`        — single scalar derived from the report.

Assumptions
-----------
- The model has been wrapped by :func:`wrap_for_qat`. The FP behavior is
  recovered by temporarily disabling every :class:`FakeQuantize` module
  in the wrapped model (no separate FP twin is required).
- Calibration / accuracy loaders yield batches that
  :func:`_extract_input` knows how to read (tensor, tuple/list of
  tensors, or dict with ``image`` / ``input`` / ``x`` / ``data`` key).
- Robustness is "lower is better": small drift, small weight error, and
  small accuracy drop yield a score near zero.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qat_wrapper import FakeQuantize, QConv2d, QLinear

__all__ = [
    "LayerError",
    "QuantSensitivityReport",
    "measure_layerwise_error",
    "measure_accuracy_drop",
    "measure_quant_sensitivity",
    "scalar_robustness_score",
]

LOG = logging.getLogger(__name__)

# Default weights for the scalar robustness summary.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "activation_drift": 0.5,   # mean output MSE across layers (already in [0, ~))
    "weight_error":     0.2,   # mean weight MSE across layers
    "accuracy_drop":    1.0,   # raw AUROC / task-metric drop
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class LayerError:
    """Per-layer quantization-induced error metrics."""

    name: str
    layer_type: str
    bits_weight: int | None = None
    bits_activation: int | None = None
    n_params: int = 0
    weight_mse: float | None = None
    weight_max_abs: float | None = None
    weight_cosine: float | None = None
    output_mse: float | None = None
    output_max_abs: float | None = None
    output_cosine: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantSensitivityReport:
    """Aggregated quantization-sensitivity report."""

    n_quantized_layers: int = 0
    layerwise: list[LayerError] = field(default_factory=list)
    activation_drift_mean: float | None = None
    activation_drift_max: float | None = None
    weight_error_mean: float | None = None
    weight_error_max: float | None = None
    fp_accuracy: float | None = None
    q_accuracy: float | None = None
    accuracy_drop: float | None = None
    n_eval_batches: int = 0
    n_calib_batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layerwise"] = [le.to_dict() if not isinstance(le, dict) else le
                          for le in self.layerwise]
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)


# ---------------------------------------------------------------------------
# Context: temporarily run the model in FP-equivalent mode
# ---------------------------------------------------------------------------
@contextmanager
def _quantization_disabled(model: nn.Module):
    """Temporarily disable every :class:`FakeQuantize` in ``model``."""
    saved: list[tuple[FakeQuantize, bool]] = []
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            saved.append((m, m.cfg.enabled))
            m.cfg.enabled = False
    try:
        yield
    finally:
        for m, prev in saved:
            m.cfg.enabled = prev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_input(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)) and batch:
        return _extract_input(batch[0])
    if isinstance(batch, dict):
        for key in ("image", "input", "x", "data"):
            if key in batch:
                return _extract_input(batch[key])
    raise TypeError(
        f"Cannot extract input tensor from batch of type {type(batch)}"
    )


def _layer_weight(layer: nn.Module) -> torch.Tensor | None:
    if isinstance(layer, QConv2d):
        return layer.conv.weight
    if isinstance(layer, QLinear):
        return layer.linear.weight
    return None


def _bits_of(layer: nn.Module) -> tuple[int | None, int | None]:
    bw = layer.weight_fq.cfg.bits if getattr(layer, "weight_fq", None) \
        else None
    ba = layer.act_fq.cfg.bits if getattr(layer, "act_fq", None) \
        else None
    return bw, ba


def _collect_quantized_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, (QConv2d, QLinear))
    ]


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Return MSE, max-abs, and cosine similarity between two tensors."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    diff = a - b
    mse = float(diff.pow(2).mean().item())
    max_abs = float(diff.abs().max().item())
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    cos = float((a.dot(b) / denom).item()) if a.numel() > 0 else 1.0
    return {"mse": mse, "max_abs": max_abs, "cosine": cos}


# ---------------------------------------------------------------------------
# Layer-wise weight error
# ---------------------------------------------------------------------------
def _measure_weight_error(layer: nn.Module) -> dict[str, float] | None:
    """Compare the FP weight to its fake-quantized round-trip."""
    weight = _layer_weight(layer)
    if weight is None:
        return None
    fq: FakeQuantize | None = getattr(layer, "weight_fq", None)
    if fq is None or not fq.cfg.enabled:
        return {"mse": 0.0, "max_abs": 0.0, "cosine": 1.0}

    # Bootstrap weight observer if it has never seen the weight.
    with torch.no_grad():
        if not torch.isfinite(fq.running_min).all():
            was_training = fq.training
            fq.train(True)
            fq._observe(weight)
            fq.train(was_training)
        was_obs = fq.observe_only
        fq.observe_only = False
        w_q = fq(weight.detach())
        fq.observe_only = was_obs

    return _diff_stats(weight, w_q)


# ---------------------------------------------------------------------------
# Activation drift
# ---------------------------------------------------------------------------
def _capture_layer_outputs(model: nn.Module,
                           layers: list[tuple[str, nn.Module]],
                           x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the model once and capture each ``layers``' output."""
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def _hook(_mod, _inp, out):
            captures[name] = out.detach()
        return _hook

    for name, mod in layers:
        handles.append(mod.register_forward_hook(make_hook(name)))
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(x)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()
    return captures


def _measure_activation_drift(model: nn.Module,
                              loader: Iterable,
                              n_batches: int,
                              device: str) -> dict[str, dict[str, float]]:
    """Per-layer activation drift averaged over ``n_batches`` batches.

    Each batch is forwarded twice through the *same* model — once with
    every fake-quant disabled (FP equivalent) and once with quantization
    active — so no separate FP twin is needed.
    """
    layers = _collect_quantized_layers(model)
    if not layers:
        return {}

    accum: dict[str, dict[str, list[float]]] = {
        name: {"mse": [], "max_abs": [], "cosine": []}
        for name, _ in layers
    }

    seen = 0
    for batch in loader:
        x = _extract_input(batch).to(device)

        with _quantization_disabled(model):
            fp_outs = _capture_layer_outputs(model, layers, x)
        q_outs = _capture_layer_outputs(model, layers, x)

        for name, _ in layers:
            if name not in fp_outs or name not in q_outs:
                continue
            stats = _diff_stats(fp_outs[name], q_outs[name])
            for k, v in stats.items():
                accum[name][k].append(v)

        seen += 1
        if seen >= n_batches:
            break

    averaged: dict[str, dict[str, float]] = {}
    for name, sub in accum.items():
        if not sub["mse"]:
            continue
        averaged[name] = {
            k: float(sum(v) / len(v)) for k, v in sub.items()
        }
    return averaged


# ---------------------------------------------------------------------------
# Public: layer-wise error
# ---------------------------------------------------------------------------
def measure_layerwise_error(model: nn.Module,
                            calibration_loader: Iterable | None = None,
                            *,
                            n_batches: int = 4,
                            device: str = "cuda") -> list[LayerError]:
    """Combine weight error and activation drift into one record per layer."""
    layers = _collect_quantized_layers(model)
    if not layers:
        LOG.warning("No quantized layers found — model not wrapped for QAT?")
        return []

    drift = (
        _measure_activation_drift(model, calibration_loader,
                                  n_batches=n_batches, device=device)
        if calibration_loader is not None else {}
    )

    out: list[LayerError] = []
    for name, mod in layers:
        bw, ba = _bits_of(mod)
        weight = _layer_weight(mod)
        n_params = int(weight.numel()) if weight is not None else 0
        we = _measure_weight_error(mod) or {}
        ad = drift.get(name, {})
        out.append(LayerError(
            name=name,
            layer_type=type(mod).__name__,
            bits_weight=bw,
            bits_activation=ba,
            n_params=n_params,
            weight_mse=we.get("mse"),
            weight_max_abs=we.get("max_abs"),
            weight_cosine=we.get("cosine"),
            output_mse=ad.get("mse"),
            output_max_abs=ad.get("max_abs"),
            output_cosine=ad.get("cosine"),
        ))
    return out


# ---------------------------------------------------------------------------
# Public: accuracy drop
# ---------------------------------------------------------------------------
def measure_accuracy_drop(model: nn.Module,
                          eval_loader: Iterable,
                          accuracy_fn: Callable[[nn.Module, Iterable], float],
                          *,
                          higher_is_better: bool = True) -> dict[str, float]:
    """Run ``accuracy_fn`` twice — fp-mode and quantized-mode — return drop.

    ``accuracy_fn`` must accept ``(model, eval_loader)`` and return a
    scalar (e.g. AUROC). When ``higher_is_better`` is ``True`` (default),
    ``accuracy_drop = fp - q``; otherwise ``q - fp`` so the drop stays
    non-negative when quantization degrades the metric.
    """
    with _quantization_disabled(model):
        fp_acc = float(accuracy_fn(model, eval_loader))
    q_acc = float(accuracy_fn(model, eval_loader))
    drop = (fp_acc - q_acc) if higher_is_better else (q_acc - fp_acc)
    return {"fp_accuracy": fp_acc, "q_accuracy": q_acc,
            "accuracy_drop": float(drop)}


# ---------------------------------------------------------------------------
# Public: composite report
# ---------------------------------------------------------------------------
def measure_quant_sensitivity(model: nn.Module,
                              calibration_loader: Iterable | None = None,
                              *,
                              accuracy_loader: Iterable | None = None,
                              accuracy_fn: Callable | None = None,
                              n_calib_batches: int = 4,
                              device: str = "cuda",
                              higher_is_better: bool = True
                              ) -> QuantSensitivityReport:
    """Build the full :class:`QuantSensitivityReport` for a quantized model."""
    report = QuantSensitivityReport()

    layers = _collect_quantized_layers(model)
    report.n_quantized_layers = len(layers)
    if not layers:
        LOG.warning("Model has no QConv2d / QLinear layers — empty report.")
        return report

    layerwise = measure_layerwise_error(
        model, calibration_loader,
        n_batches=n_calib_batches, device=device,
    )
    report.layerwise = layerwise
    report.n_calib_batches = n_calib_batches if calibration_loader else 0

    drift_mse = [le.output_mse for le in layerwise
                 if le.output_mse is not None]
    weight_mse = [le.weight_mse for le in layerwise
                  if le.weight_mse is not None]
    if drift_mse:
        report.activation_drift_mean = float(sum(drift_mse) / len(drift_mse))
        report.activation_drift_max = float(max(drift_mse))
    if weight_mse:
        report.weight_error_mean = float(sum(weight_mse) / len(weight_mse))
        report.weight_error_max = float(max(weight_mse))

    if accuracy_loader is not None and accuracy_fn is not None:
        try:
            acc = measure_accuracy_drop(
                model, accuracy_loader, accuracy_fn,
                higher_is_better=higher_is_better,
            )
            report.fp_accuracy = acc["fp_accuracy"]
            report.q_accuracy = acc["q_accuracy"]
            report.accuracy_drop = acc["accuracy_drop"]
        except Exception:  # noqa: BLE001
            LOG.exception("accuracy_fn raised — leaving accuracy_drop unset")

    return report


# ---------------------------------------------------------------------------
# Robustness scalarizer
# ---------------------------------------------------------------------------
def scalar_robustness_score(report: QuantSensitivityReport,
                            *,
                            weights: dict[str, float] | None = None) -> float:
    """Reduce a :class:`QuantSensitivityReport` to a single scalar.

    Lower values indicate a more robust quantization (smaller drift,
    smaller weight error, smaller accuracy drop). Components missing from
    the report (e.g. accuracy drop when no accuracy_fn was provided)
    contribute zero to the score.

    The default weighting privileges accuracy drop, which is the
    end-to-end signal that the search ultimately cares about; layer-wise
    drift and weight error are kept as low-cost early indicators.
    """
    w = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        w.update(weights)

    score = 0.0
    if report.activation_drift_mean is not None:
        score += w["activation_drift"] * float(report.activation_drift_mean)
    if report.weight_error_mean is not None:
        score += w["weight_error"] * float(report.weight_error_mean)
    if report.accuracy_drop is not None:
        score += w["accuracy_drop"] * max(0.0, float(report.accuracy_drop))
    return float(score)
