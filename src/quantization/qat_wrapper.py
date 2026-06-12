"""
src/quantization/qat_wrapper.py
===============================

Quantization-Aware Training (QAT) wrapper.

Implements a self-contained, eager-mode QAT toolkit so the NSGA-II search
can freely mix bit widths beyond what PyTorch's built-in qconfigs support
(``torch.ao.quantization`` is INT8-only out of the box).

Capabilities
------------
- Arbitrary bit widths per tensor (default support: ``2..8``; ``>= 16`` is
  treated as a no-op / fp pass-through).
- Symmetric or asymmetric quantization, independently for weights and
  activations.
- Per-channel weight quantization (per output channel for Conv / Linear)
  or per-tensor.
- Mixed precision: per-layer bit-width overrides via name patterns and an
  optional first/last layer override (a common technique to retain
  accuracy at very low bit widths).
- Two observer flavors: plain min/max and EMA min/max.
- Straight-through-estimator backward pass for gradient flow during QAT.

Public interface (consumed by ``main_search.py`` / ``main_retrain.py``)
----------------------------------------------------------------------
``wrap_for_qat(model: nn.Module, qconfig: dict) -> nn.Module``
    In-place replacement of every eligible ``nn.Conv2d`` / ``nn.Linear``
    with a quantization-aware variant. Returns the same ``model`` for
    fluent chaining. Pass an empty dict, ``None``, or
    ``{"enabled": False}`` to disable quantization (returns the model
    untouched — quantization-free training keeps working).

``calibrate(model: nn.Module, calibration_loader, *,
            n_batches: int = 32, device: str = "cuda") -> nn.Module``
    Runs a few forward passes in eval mode to populate observer
    statistics; required before QAT fine-tuning starts.

``calibrate_and_finetune(model, calibration_loader, *, n_batches=32,
                         device="cuda", **_unused) -> nn.Module``
    Convenience alias for compatibility with the orchestrator interface
    (the actual fine-tune loop lives in ``src.evaluation.train_loop``).

Configuration schema
--------------------
::

    qconfig = {
        "enabled":           True,
        "weight": {
            "bits":         8,
            "symmetric":    True,
            "per_channel":  True,
            "observer":     "minmax" | "ema",
            "ema_momentum": 0.01,
        },
        "activation": {
            "bits":         8,
            "symmetric":    False,
            "per_channel":  False,
            "observer":     "ema" | "minmax",
            "ema_momentum": 0.01,
        },
        "skip_layers":       ["head", "score_head"],   # name fragments
        "first_last_bits":    8,                        # mixed precision
        "per_layer": {
            "stem":         {"weight": {"bits": 8}, "activation": {"bits": 8}},
            "bottleneck":   {"weight": {"bits": 6}},
        },
        "quantize_input":     True,    # add fake-quant on each layer input
    }

Assumptions
-----------
- This module is *eager-mode*: it walks ``named_modules`` and substitutes
  layers in place. It does not require FX tracing and works on any
  architecture produced by ``model_factory.build_model``.
- Real INT8 / INT4 export is the responsibility of
  ``src.deployment.export_onnx`` / ``export_tensorrt`` — here we only
  *simulate* low-precision behavior via fake-quant + STE.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

__all__ = [
    "wrap_for_qat",
    "calibrate",
    "calibrate_and_finetune",
    "FakeQuantize",
    "QConv2d",
    "QLinear",
    "TensorQuantConfig",
    "set_observer_only",
    "set_frozen",
]

LOG = logging.getLogger(__name__)

_DEFAULT_WEIGHT_BITS: int = 8
_DEFAULT_ACT_BITS: int = 8
_NO_QUANT_BITS_THRESHOLD: int = 16   # >= 16 bits is treated as fp pass-through


# ---------------------------------------------------------------------------
# Per-tensor quantization configuration
# ---------------------------------------------------------------------------
@dataclass
class TensorQuantConfig:
    """Resolved quantization configuration for a single tensor (weight or act)."""

    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    observer: str = "ema"            # 'minmax' | 'ema'
    ema_momentum: float = 0.01
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None,
                  *, default_bits: int = 8) -> "TensorQuantConfig":
        data = dict(data or {})
        if "bits" not in data:
            data["bits"] = default_bits
        cfg = cls(
            bits=int(data.get("bits", default_bits)),
            symmetric=bool(data.get("symmetric", True)),
            per_channel=bool(data.get("per_channel", False)),
            observer=str(data.get("observer", "ema")),
            ema_momentum=float(data.get("ema_momentum", 0.01)),
            enabled=bool(data.get("enabled", True)),
        )
        if cfg.bits < 2:
            raise ValueError(f"bits must be >= 2 (got {cfg.bits}); "
                             "binary quantization is not supported.")
        if cfg.bits >= _NO_QUANT_BITS_THRESHOLD:
            cfg.enabled = False
        if cfg.observer not in {"minmax", "ema"}:
            raise ValueError(f"unknown observer: {cfg.observer!r}")
        return cfg

    def qrange(self) -> tuple[int, int]:
        """Return the integer ``(qmin, qmax)`` range for this config."""
        if self.symmetric:
            qmax = (1 << (self.bits - 1)) - 1
            return -qmax, qmax
        return 0, (1 << self.bits) - 1


# ---------------------------------------------------------------------------
# Fake-quantize core (STE)
# ---------------------------------------------------------------------------
class _FakeQuantSTE(torch.autograd.Function):
    """Quantize-dequantize forward + identity backward (Straight-Through)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor,
                scale: torch.Tensor,
                zero_point: torch.Tensor,
                qmin: int, qmax: int) -> torch.Tensor:
        x_int = torch.round(x / scale + zero_point).clamp(qmin, qmax)
        x_dq = (x_int - zero_point) * scale
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.save_for_backward(x, scale, zero_point)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D401
        # Pass gradient through (the STE assumption). The clamp region's
        # gradient could be masked here, but the standard QAT recipe
        # passes everything for stability.
        return grad_output, None, None, None, None


class FakeQuantize(nn.Module):
    """Configurable fake-quantization module with optional running observer."""

    def __init__(self, cfg: TensorQuantConfig, *, num_channels: int = 1) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_channels = max(1, int(num_channels))
        shape = (self.num_channels,) if cfg.per_channel else (1,)

        self.register_buffer("scale", torch.ones(shape, dtype=torch.float32))
        self.register_buffer(
            "zero_point", torch.zeros(shape, dtype=torch.float32),
        )
        self.register_buffer("running_min", torch.full(shape, float("inf")))
        self.register_buffer("running_max", torch.full(shape, float("-inf")))

        # Mode flags (controlled by helpers in this module).
        self.observe_only: bool = False  # update stats but skip quantization
        self.frozen: bool = False        # stop updating observer

    # ------------------------------------------------------------------
    def _observe(self, x: torch.Tensor) -> None:
        """Update running_min / running_max from the current batch."""
        if self.frozen:
            return
        if self.cfg.per_channel:
            # x is expected to have shape [C, ...] (weights) or [B, C, ...]
            # — caller chooses the channel axis upstream.
            x_flat = x.reshape(self.num_channels, -1)
            cur_min = x_flat.amin(dim=1).detach()
            cur_max = x_flat.amax(dim=1).detach()
        else:
            cur_min = x.detach().min().reshape(1)
            cur_max = x.detach().max().reshape(1)

        if self.cfg.observer == "ema" and torch.isfinite(self.running_min).all():
            m = self.cfg.ema_momentum
            self.running_min.mul_(1 - m).add_(cur_min * m)
            self.running_max.mul_(1 - m).add_(cur_max * m)
        else:
            # First update or plain min/max observer.
            self.running_min.copy_(torch.minimum(self.running_min, cur_min))
            self.running_max.copy_(torch.maximum(self.running_max, cur_max))

        self._recompute_qparams()

    # ------------------------------------------------------------------
    def _recompute_qparams(self) -> None:
        """Compute scale & zero_point from the latest running min/max."""
        qmin, qmax = self.cfg.qrange()
        if self.cfg.symmetric:
            r = torch.maximum(self.running_max.abs(), self.running_min.abs())
            r = r.clamp(min=1e-8)
            self.scale.copy_(r / qmax)
            self.zero_point.zero_()
        else:
            r = (self.running_max - self.running_min).clamp(min=1e-8)
            self.scale.copy_(r / (qmax - qmin))
            zp = qmin - (self.running_min / self.scale).round()
            self.zero_point.copy_(zp.clamp(qmin, qmax))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enabled:
            return x

        if self.training:
            self._observe(x)

        if self.observe_only:
            return x

        qmin, qmax = self.cfg.qrange()
        scale, zp = self._broadcast(x)
        return _FakeQuantSTE.apply(x, scale, zp, qmin, qmax)

    # ------------------------------------------------------------------
    def _broadcast(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape per-channel scale / zp for broadcasting onto ``x``."""
        if not self.cfg.per_channel or self.num_channels == 1:
            return self.scale, self.zero_point
        # By convention we apply per-channel quant on dim 0 of the tensor
        # (output channels for weights). Activation per-channel is not
        # commonly used and is treated as per-tensor here.
        view_shape = [self.num_channels] + [1] * (x.dim() - 1)
        return self.scale.view(view_shape), self.zero_point.view(view_shape)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"bits={self.cfg.bits}, sym={self.cfg.symmetric}, "
                f"per_ch={self.cfg.per_channel}, obs={self.cfg.observer}, "
                f"ch={self.num_channels}, enabled={self.cfg.enabled}")


# ---------------------------------------------------------------------------
# Quantized layer wrappers
# ---------------------------------------------------------------------------
class QConv2d(nn.Module):
    """Quantization-aware ``nn.Conv2d`` replacement.

    Holds the original conv, plus optional fake-quant modules for the
    input activations and the weight tensor.
    """

    def __init__(self,
                 conv: nn.Conv2d,
                 weight_cfg: TensorQuantConfig,
                 act_cfg: TensorQuantConfig,
                 *,
                 quantize_input: bool = True) -> None:
        super().__init__()
        self.conv = conv
        # Per-channel weight quant uses out_channels as the channel axis.
        self.weight_fq = FakeQuantize(
            weight_cfg, num_channels=conv.out_channels,
        ) if weight_cfg.enabled else None
        self.act_fq = FakeQuantize(
            act_cfg, num_channels=1,
        ) if (act_cfg.enabled and quantize_input) else None

    @classmethod
    def from_float(cls,
                   conv: nn.Conv2d,
                   weight_cfg: TensorQuantConfig,
                   act_cfg: TensorQuantConfig,
                   *,
                   quantize_input: bool = True) -> "QConv2d":
        return cls(conv, weight_cfg, act_cfg, quantize_input=quantize_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fq is not None:
            x = self.act_fq(x)
        w = self.conv.weight
        if self.weight_fq is not None:
            # Initialize weight observer the first time we see the weight
            # so that calibrate() does not need a forward pass to populate
            # weight ranges (weights are static; one observation suffices).
            if not torch.isfinite(self.weight_fq.running_min).all():
                with torch.no_grad():
                    was_training = self.weight_fq.training
                    self.weight_fq.train(True)
                    self.weight_fq._observe(w)
                    self.weight_fq.train(was_training)
            w = self.weight_fq(w)
        return nn.functional.conv2d(
            x, w, self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QLinear(nn.Module):
    """Quantization-aware ``nn.Linear`` replacement."""

    def __init__(self,
                 linear: nn.Linear,
                 weight_cfg: TensorQuantConfig,
                 act_cfg: TensorQuantConfig,
                 *,
                 quantize_input: bool = True) -> None:
        super().__init__()
        self.linear = linear
        self.weight_fq = FakeQuantize(
            weight_cfg, num_channels=linear.out_features,
        ) if weight_cfg.enabled else None
        self.act_fq = FakeQuantize(
            act_cfg, num_channels=1,
        ) if (act_cfg.enabled and quantize_input) else None

    @classmethod
    def from_float(cls,
                   linear: nn.Linear,
                   weight_cfg: TensorQuantConfig,
                   act_cfg: TensorQuantConfig,
                   *,
                   quantize_input: bool = True) -> "QLinear":
        return cls(linear, weight_cfg, act_cfg, quantize_input=quantize_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fq is not None:
            x = self.act_fq(x)
        w = self.linear.weight
        if self.weight_fq is not None:
            if not torch.isfinite(self.weight_fq.running_min).all():
                with torch.no_grad():
                    was_training = self.weight_fq.training
                    self.weight_fq.train(True)
                    self.weight_fq._observe(w)
                    self.weight_fq.train(was_training)
            w = self.weight_fq(w)
        return nn.functional.linear(x, w, self.linear.bias)


# ---------------------------------------------------------------------------
# Wrapping logic
# ---------------------------------------------------------------------------
@dataclass
class _LayerOverride:
    """Resolved per-layer quantization override (parsed from qconfig)."""
    weight: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)


def wrap_for_qat(model: nn.Module, qconfig: dict | None,
                 **_unused) -> nn.Module:
    """Wrap an FP32 model with quantization-aware modules in place."""
    if not qconfig or not qconfig.get("enabled", True):
        LOG.debug("Quantization disabled — returning model unchanged.")
        return model

    weight_defaults = qconfig.get("weight", {}) or {}
    act_defaults = qconfig.get("activation", {}) or {}
    skip_patterns: list[str] = list(qconfig.get("skip_layers", []) or [])
    first_last_bits = qconfig.get("first_last_bits")
    per_layer = qconfig.get("per_layer", {}) or {}
    quantize_input = bool(qconfig.get("quantize_input", True))

    # Collect candidate layers (Conv2d, Linear) in graph order.
    candidates: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        if _name_matches_any(name, skip_patterns):
            continue
        candidates.append((name, mod))

    if not candidates:
        LOG.warning("No quantizable layers found — model returned as-is.")
        return model

    first_name = candidates[0][0]
    last_name = candidates[-1][0]

    LOG.info("Quantizing %d layer(s) (skip=%s, first_last_bits=%s)",
             len(candidates), skip_patterns, first_last_bits)

    for name, mod in candidates:
        w_cfg, a_cfg = _resolve_layer_config(
            name=name,
            weight_defaults=weight_defaults,
            act_defaults=act_defaults,
            per_layer=per_layer,
            first_last_bits=first_last_bits,
            is_first=(name == first_name),
            is_last=(name == last_name),
        )
        if isinstance(mod, nn.Conv2d):
            new_mod = QConv2d.from_float(mod, w_cfg, a_cfg,
                                         quantize_input=quantize_input)
        else:
            new_mod = QLinear.from_float(mod, w_cfg, a_cfg,
                                         quantize_input=quantize_input)
        _set_submodule(model, name, new_mod)
        LOG.debug("  [%s] %s -> w(b=%d,sym=%s,ch=%s) a(b=%d,sym=%s)",
                  name, type(mod).__name__,
                  w_cfg.bits, w_cfg.symmetric, w_cfg.per_channel,
                  a_cfg.bits, a_cfg.symmetric)
    return model


def _resolve_layer_config(*,
                          name: str,
                          weight_defaults: dict[str, Any],
                          act_defaults: dict[str, Any],
                          per_layer: dict[str, dict[str, Any]],
                          first_last_bits: int | None,
                          is_first: bool,
                          is_last: bool
                          ) -> tuple[TensorQuantConfig, TensorQuantConfig]:
    """Compose the effective TensorQuantConfig pair for a single layer."""
    w = dict(weight_defaults)
    a = dict(act_defaults)

    for pattern, override in per_layer.items():
        if pattern in name or re.search(pattern, name) is not None:
            if "weight" in override:
                w.update(override["weight"])
            if "activation" in override:
                a.update(override["activation"])

    if first_last_bits is not None and (is_first or is_last):
        # Mixed precision: keep edge layers at the "safe" bit width.
        w["bits"] = int(first_last_bits)
        a["bits"] = int(first_last_bits)

    w_cfg = TensorQuantConfig.from_dict(w, default_bits=_DEFAULT_WEIGHT_BITS)
    a_cfg = TensorQuantConfig.from_dict(a, default_bits=_DEFAULT_ACT_BITS)
    return w_cfg, a_cfg


def _name_matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if pat in name:
            return True
        try:
            if re.search(pat, name):
                return True
        except re.error:
            continue
    return False


def _set_submodule(model: nn.Module, dotted_name: str,
                   new_module: nn.Module) -> None:
    """Replace ``model.<dotted_name>`` with ``new_module`` in place."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(model: nn.Module,
              calibration_loader,
              *,
              n_batches: int = 32,
              device: str = "cuda",
              observe_only: bool = True) -> nn.Module:
    """Populate observer statistics by running ``n_batches`` through the model.

    By default ``observe_only=True`` skips the actual fake-quant transform
    so calibration captures clean floating-point ranges; switch to
    ``observe_only=False`` to apply quantization during calibration (PTQ
    flavor).
    """
    model.eval()
    set_observer_only(model, observe_only)
    set_frozen(model, False)

    seen = 0
    with torch.no_grad():
        for batch in calibration_loader:
            x = _extract_input(batch).to(device)
            model(x)
            seen += 1
            if seen >= n_batches:
                break

    set_observer_only(model, False)
    LOG.info("Calibration complete (%d batches).", seen)
    return model


def calibrate_and_finetune(model: nn.Module,
                           calibration_loader,
                           *,
                           n_batches: int = 32,
                           device: str = "cuda",
                           **_unused) -> nn.Module:
    """Convenience alias kept for the orchestrator's interface contract."""
    return calibrate(model, calibration_loader,
                     n_batches=n_batches, device=device)


def _extract_input(batch: Any) -> torch.Tensor:
    """Pull the input tensor out of a heterogeneous batch object."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)) and batch:
        return _extract_input(batch[0])
    if isinstance(batch, dict):
        for key in ("image", "input", "x", "data"):
            if key in batch:
                return _extract_input(batch[key])
    raise TypeError(
        f"Cannot extract an input tensor from batch of type {type(batch)}; "
        "supply (tensor,) / (tensor, label) / dict-with-'image'."
    )


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------
def set_observer_only(model: nn.Module, value: bool) -> None:
    """Toggle ``observe_only`` on every :class:`FakeQuantize` in ``model``."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.observe_only = bool(value)


def set_frozen(model: nn.Module, value: bool) -> None:
    """Freeze or unfreeze every :class:`FakeQuantize` observer in ``model``."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.frozen = bool(value)
