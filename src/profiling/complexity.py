"""
src/profiling/complexity.py
===========================

Deterministic computational-cost metrics of a (possibly fake-quantized) model,
measured with forward hooks on one dummy input:

- ``macs_m``    — multiply-accumulates of Conv2d / Linear layers (millions).
- ``bops_g``    — bit-operations: sum of ``MACs * w_bits * a_bits`` (billions).
                  Captures the cost reduction of low precision that latency on
                  a desktop GPU cannot show (fake-quant runs in fp32).
- ``weight_kb`` — weight storage at the layers' bit widths (kilobytes).
- ``n_params``  — trainable parameters.

Layers without fake-quant count as ``float_bits`` (32 by default, 16 for an
fp16 representation). Power-of-two weights are counted at their bit width.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.quantization.qat_wrapper import QConv2d, QLinear

__all__ = ["measure_complexity"]


def _bits(fq, float_bits: int) -> int:
    return fq.cfg.bits if fq is not None and fq.cfg.enabled else float_bits


def measure_complexity(model: nn.Module, input_shape: tuple[int, ...], *,
                       device: str = "cpu", float_bits: int = 32) -> dict[str, Any]:
    """Return ``{macs_m, bops_g, weight_kb, n_params}`` for one forward pass."""
    totals = {"macs": 0.0, "bops": 0.0, "wbits": 0.0}
    hooks = []

    def account(macs: float, n_weights: int, w_bits: int, a_bits: int) -> None:
        totals["macs"] += macs
        totals["bops"] += macs * w_bits * a_bits
        totals["wbits"] += n_weights * w_bits

    def conv_hook(mod, _inp, out):
        conv = mod.conv if isinstance(mod, QConv2d) else mod
        w = conv.weight
        macs = out.numel() / out.shape[0] * w[0].numel()   # per sample
        wq = getattr(mod, "weight_fq", None)
        aq = getattr(mod, "act_fq", None)
        account(macs, w.numel(), _bits(wq, float_bits), _bits(aq, float_bits))

    def linear_hook(mod, _inp, out):
        lin = mod.linear if isinstance(mod, QLinear) else mod
        w = lin.weight
        macs = out.numel() / out.shape[0] * w.shape[1]
        wq = getattr(mod, "weight_fq", None)
        aq = getattr(mod, "act_fq", None)
        account(macs, w.numel(), _bits(wq, float_bits), _bits(aq, float_bits))

    inner = set()
    for m in model.modules():
        if isinstance(m, QConv2d):
            inner.add(id(m.conv))
        elif isinstance(m, QLinear):
            inner.add(id(m.linear))
    for m in model.modules():
        if id(m) in inner:
            continue  # counted through its Q-wrapper
        if isinstance(m, (QConv2d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, (QLinear, nn.Linear)):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    return {
        "macs_m": totals["macs"] / 1e6,
        "bops_g": totals["bops"] / 1e9,
        "weight_kb": totals["wbits"] / 8 / 1024,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
