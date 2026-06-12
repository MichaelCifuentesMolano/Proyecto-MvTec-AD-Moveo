"""
src/models/model_factory.py
===========================

Architecture factory for the NSGA-II search. Builds a PyTorch ``nn.Module``
from a genome / candidate configuration produced by ``src.nas.encoding``.

Supported architecture families
-------------------------------
- ``"autoencoder"``       – Encoder–decoder reconstruction CNN.
- ``"unet"``              – Lightweight U-Net (encoder–decoder + skip conns).
- ``"feature_recon"``     – Feature-reconstruction network operating in the
                            feature space of an external pretrained backbone.
- ``"student_teacher"``   – Compact student that mimics multi-scale teacher
                            features (teacher is loaded externally).
- ``"patch_cnn"``         – Fully-convolutional patch-level anomaly scorer.

Public interface (consumed by ``main_search.py`` / ``main_retrain.py``)
----------------------------------------------------------------------
``build_model(candidate: dict) -> torch.nn.Module``

The ``candidate`` may be either the architecture sub-dict directly or the
full genome dict containing an ``"architecture"`` key (and an optional
``"quantization"`` key — ignored here). Sensible defaults fill any
missing fields so partial genomes still build.

Forward output convention
-------------------------
Every model returns a ``dict`` with a consistent (but family-dependent)
schema::

    {
        "recon":       Tensor [B, C, H, W]   # reconstruction (when applicable)
        "features":    list[Tensor] | Tensor  # feature maps (when applicable)
        "anomaly_map": Tensor [B, 1, H, W]   # pixel-wise score (if internal)
        "score":       Tensor [B]            # image-level score  (if internal)
    }

Models that need an external signal (teacher features, target features) leave
``anomaly_map`` / ``score`` out of the dict; the training loop computes them.

Genome schema (architecture sub-dict)
-------------------------------------
::

    {
        "family":           "autoencoder" | "unet" | ...
        "input_channels":   3,
        "input_size":       224,
        "depth":            4,                  # encoder stages
        "base_width":       16,                 # channels at stage 0
        "width_mult":       2.0,                # channel growth per stage
        "kernel_size":      3,
        "bottleneck_dim":   128,
        "use_skip":         true,               # u-net only
        "norm":             "batch" | "instance" | "group" | "none"
        "activation":       "relu" | "leaky_relu" | "gelu" | "silu"
        "layer_type":       "conv" | "ds" | "ir"   # standard / depthwise-sep /
                                                   # inverted-residual
        "expansion":        4,                  # ir-block expansion factor
        "feature_dim":      256,                # feature_recon / student_teacher
        "n_scales":         3,                  # student_teacher
        "patch_size":       32,                 # patch_cnn
    }

Assumptions
-----------
- Quantization is applied later by ``src.quantization.qat_wrapper``;
  this factory produces ordinary FP32 models.
- Spatial dimensions are halved at every encoder stage; choose ``depth``
  such that ``input_size / 2**depth >= 4`` (validated at build time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_model",
    "AutoencoderCNN",
    "LightweightUNet",
    "FeatureReconNet",
    "StudentTeacherStudent",
    "PatchCNN",
    "ArchSpec",
]


# ---------------------------------------------------------------------------
# Architecture spec (parsed from the genome with sane defaults)
# ---------------------------------------------------------------------------
@dataclass
class ArchSpec:
    """Normalized architecture specification derived from a genome dict."""

    family: str = "autoencoder"
    input_channels: int = 3
    input_size: int = 224
    depth: int = 4
    base_width: int = 16
    width_mult: float = 2.0
    kernel_size: int = 3
    bottleneck_dim: int = 128
    use_skip: bool = True
    norm: str = "batch"
    activation: str = "relu"
    layer_type: str = "conv"        # 'conv' | 'ds' | 'ir'
    expansion: int = 4              # 'ir' expansion factor
    feature_dim: int = 256          # feature_recon / student_teacher
    n_scales: int = 3               # student_teacher
    patch_size: int = 32            # patch_cnn
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchSpec":
        """Build an ``ArchSpec`` from a genome dict, accepting partial fields."""
        if "architecture" in data and isinstance(data["architecture"], dict):
            data = data["architecture"]
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        spec = cls(**kwargs)
        if extra:
            spec.extra.update(extra)
        spec._validate()
        return spec

    # ------------------------------------------------------------------
    def widths(self) -> list[int]:
        """Per-stage channel counts (rounded to a multiple of 4)."""
        out: list[int] = []
        w = float(self.base_width)
        for _ in range(self.depth):
            out.append(_round_to_multiple(int(round(w)), 4, minimum=4))
            w *= self.width_mult
        return out

    def smallest_spatial(self) -> int:
        return self.input_size // (2 ** self.depth)

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.input_size < 16:
            raise ValueError(f"input_size too small: {self.input_size}")
        if self.smallest_spatial() < 2:
            raise ValueError(
                f"depth={self.depth} too large for input_size="
                f"{self.input_size}: bottleneck spatial would be "
                f"{self.smallest_spatial()}"
            )
        if self.layer_type not in {"conv", "ds", "ir"}:
            raise ValueError(f"unknown layer_type={self.layer_type!r}")
        if self.norm not in {"batch", "instance", "group", "none"}:
            raise ValueError(f"unknown norm={self.norm!r}")
        if self.activation not in {"relu", "leaky_relu", "gelu", "silu"}:
            raise ValueError(f"unknown activation={self.activation!r}")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[["ArchSpec"], nn.Module]] = {}


def _register(name: str) -> Callable:
    def deco(fn: Callable[["ArchSpec"], nn.Module]) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(candidate: dict) -> nn.Module:
    """Construct a PyTorch model from a genome / candidate dict.

    Raises
    ------
    ValueError
        If the requested ``family`` is unknown or the spec is invalid.
    """
    spec = ArchSpec.from_dict(candidate)
    if spec.family not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture family: {spec.family!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    model = _REGISTRY[spec.family](spec)
    # Stash the resolved spec for downstream introspection (export, logging).
    model.arch_spec = spec  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _make_norm(channels: int, kind: str) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "group":
        groups = max(1, min(32, channels // 4))
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def _make_activation(kind: str) -> nn.Module:
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if kind == "gelu":
        return nn.GELU()
    if kind == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"unknown activation: {kind}")


def _round_to_multiple(value: int, multiple: int, minimum: int = 1) -> int:
    return max(minimum, ((value + multiple - 1) // multiple) * multiple)


# ---- Convolutional blocks -------------------------------------------------
class ConvBNAct(nn.Module):
    """Plain Conv → Norm → Activation block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1,
                 norm: str = "batch", activation: str = "relu",
                 use_act: bool = True) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=pad, groups=groups,
                              bias=norm == "none")
        self.norm = _make_norm(out_ch, norm)
        self.act = _make_activation(activation) if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise + pointwise (MobileNetV1 style) — efficient on Jetson."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, norm: str = "batch",
                 activation: str = "relu") -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, kernel=kernel, stride=stride,
                            groups=in_ch, norm=norm, activation=activation)
        self.pw = ConvBNAct(in_ch, out_ch, kernel=1, stride=1,
                            norm=norm, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted-residual block (with optional skip)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, expansion: int = 4,
                 norm: str = "batch", activation: str = "relu") -> None:
        super().__init__()
        hidden = max(in_ch * expansion, 8)
        self.use_skip = (stride == 1 and in_ch == out_ch)
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel=1, stride=1,
                                    norm=norm, activation=activation))
        layers += [
            ConvBNAct(hidden, hidden, kernel=kernel, stride=stride,
                      groups=hidden, norm=norm, activation=activation),
            ConvBNAct(hidden, out_ch, kernel=1, stride=1,
                      norm=norm, activation=activation, use_act=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_skip else out


def _make_block(in_ch: int, out_ch: int, *,
                spec: ArchSpec, stride: int = 1) -> nn.Module:
    """Pick the conv block flavor based on ``spec.layer_type``."""
    if spec.layer_type == "conv":
        return ConvBNAct(in_ch, out_ch, kernel=spec.kernel_size,
                         stride=stride, norm=spec.norm,
                         activation=spec.activation)
    if spec.layer_type == "ds":
        return DepthwiseSeparable(in_ch, out_ch, kernel=spec.kernel_size,
                                  stride=stride, norm=spec.norm,
                                  activation=spec.activation)
    if spec.layer_type == "ir":
        return InvertedResidual(in_ch, out_ch, kernel=spec.kernel_size,
                                stride=stride, expansion=spec.expansion,
                                norm=spec.norm, activation=spec.activation)
    raise ValueError(spec.layer_type)


# ---- Down / Up stages ------------------------------------------------------
class DownStage(nn.Module):
    """Two conv blocks; the second one downsamples (stride=2)."""

    def __init__(self, in_ch: int, out_ch: int, *, spec: ArchSpec) -> None:
        super().__init__()
        self.refine = _make_block(in_ch, out_ch, spec=spec, stride=1)
        self.down = _make_block(out_ch, out_ch, spec=spec, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.refine(x)
        out = self.down(skip)
        return out, skip


class UpStage(nn.Module):
    """Upsample (×2) + optional skip concat + refinement block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 *, spec: ArchSpec, use_skip: bool) -> None:
        super().__init__()
        self.use_skip = use_skip
        # Bilinear upsample is quantization-friendly and TRT-compatible.
        merge_in = in_ch + (skip_ch if use_skip else 0)
        self.refine = _make_block(merge_in, out_ch, spec=spec, stride=1)

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear",
                          align_corners=False)
        if self.use_skip and skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.refine(x)


# ---------------------------------------------------------------------------
# Architecture: AutoencoderCNN
# ---------------------------------------------------------------------------
class AutoencoderCNN(nn.Module):
    """Reconstruction-based anomaly detector (encoder → bottleneck → decoder)."""

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.widths()

        self.stem = ConvBNAct(spec.input_channels, widths[0],
                              kernel=spec.kernel_size, stride=1,
                              norm=spec.norm, activation=spec.activation)
        # Encoder
        self.encoder: nn.ModuleList = nn.ModuleList()
        prev = widths[0]
        for w in widths:
            self.encoder.append(DownStage(prev, w, spec=spec))
            prev = w

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNAct(prev, spec.bottleneck_dim, kernel=1, stride=1,
                      norm=spec.norm, activation=spec.activation),
            ConvBNAct(spec.bottleneck_dim, prev, kernel=1, stride=1,
                      norm=spec.norm, activation=spec.activation),
        )

        # Decoder mirrors the encoder; no skips for plain autoencoder.
        self.decoder: nn.ModuleList = nn.ModuleList()
        rev = list(reversed(widths))
        for i, w in enumerate(rev):
            in_ch = prev
            out_ch = rev[i + 1] if i + 1 < len(rev) else widths[0]
            self.decoder.append(UpStage(in_ch, skip_ch=0, out_ch=out_ch,
                                        spec=spec, use_skip=False))
            prev = out_ch

        self.head = nn.Conv2d(prev, spec.input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(x)
        for enc in self.encoder:
            h, _ = enc(h)
        h = self.bottleneck(h)
        for dec in self.decoder:
            h = dec(h, None)
        recon = self.head(h)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        anomaly_map = (recon - x).pow(2).mean(dim=1, keepdim=True)
        score = anomaly_map.flatten(1).amax(dim=1)
        return {"recon": recon, "anomaly_map": anomaly_map, "score": score}


@_register("autoencoder")
def _build_autoencoder(spec: ArchSpec) -> nn.Module:
    return AutoencoderCNN(spec)


# ---------------------------------------------------------------------------
# Architecture: LightweightUNet
# ---------------------------------------------------------------------------
class LightweightUNet(nn.Module):
    """U-Net with parameterizable depth/width and quant-friendly upsample."""

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.widths()

        self.stem = ConvBNAct(spec.input_channels, widths[0],
                              kernel=spec.kernel_size, stride=1,
                              norm=spec.norm, activation=spec.activation)
        # Encoder
        self.encoder: nn.ModuleList = nn.ModuleList()
        prev = widths[0]
        for w in widths:
            self.encoder.append(DownStage(prev, w, spec=spec))
            prev = w

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNAct(prev, spec.bottleneck_dim, kernel=spec.kernel_size,
                      stride=1, norm=spec.norm, activation=spec.activation),
            ConvBNAct(spec.bottleneck_dim, prev, kernel=spec.kernel_size,
                      stride=1, norm=spec.norm, activation=spec.activation),
        )

        # Decoder with skip connections
        self.decoder: nn.ModuleList = nn.ModuleList()
        rev_widths = list(reversed(widths))
        for i, w in enumerate(rev_widths):
            in_ch = prev
            out_ch = rev_widths[i + 1] if i + 1 < len(rev_widths) \
                else widths[0]
            skip_ch = w
            self.decoder.append(UpStage(in_ch, skip_ch=skip_ch,
                                        out_ch=out_ch, spec=spec,
                                        use_skip=spec.use_skip))
            prev = out_ch

        self.head = nn.Conv2d(prev, spec.input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for enc in self.encoder:
            h, skip = enc(h)
            skips.append(skip)
        h = self.bottleneck(h)
        for dec, skip in zip(self.decoder, reversed(skips)):
            h = dec(h, skip if self.spec.use_skip else None)
        recon = self.head(h)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:],
                                  mode="bilinear", align_corners=False)
        anomaly_map = (recon - x).pow(2).mean(dim=1, keepdim=True)
        score = anomaly_map.flatten(1).amax(dim=1)
        return {"recon": recon, "anomaly_map": anomaly_map, "score": score}


@_register("unet")
def _build_unet(spec: ArchSpec) -> nn.Module:
    return LightweightUNet(spec)


# ---------------------------------------------------------------------------
# Architecture: FeatureReconNet
# ---------------------------------------------------------------------------
class FeatureReconNet(nn.Module):
    """Reconstructs target features from the input.

    The training loop typically extracts target features via a frozen
    pretrained backbone and supervises ``model.forward(x)["features"]``
    to match them. Anomaly score = ‖f_target − f_recon‖² in feature space.
    """

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.widths()

        layers: list[nn.Module] = [
            ConvBNAct(spec.input_channels, widths[0],
                      kernel=spec.kernel_size, stride=2,
                      norm=spec.norm, activation=spec.activation),
        ]
        prev = widths[0]
        for w in widths[1:]:
            layers.append(_make_block(prev, w, spec=spec, stride=2))
            prev = w

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(prev, spec.feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        feats = self.head(h)
        # Anomaly map / score require the target features and are computed
        # by the training loop / evaluator.
        return {"features": feats}


@_register("feature_recon")
def _build_feature_recon(spec: ArchSpec) -> nn.Module:
    return FeatureReconNet(spec)


# ---------------------------------------------------------------------------
# Architecture: StudentTeacherStudent (student only)
# ---------------------------------------------------------------------------
class StudentTeacherStudent(nn.Module):
    """Compact student that produces multi-scale features.

    The teacher is loaded externally (e.g. a frozen ResNet/EfficientNet);
    the training loop aligns ``student.forward(x)["features"]`` with the
    teacher's features at matching scales.
    """

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.widths()
        n_scales = max(1, min(spec.n_scales, spec.depth))
        self._scale_indices: list[int] = list(
            range(spec.depth - n_scales, spec.depth)
        )

        self.stem = ConvBNAct(spec.input_channels, widths[0],
                              kernel=spec.kernel_size, stride=1,
                              norm=spec.norm, activation=spec.activation)
        self.encoder: nn.ModuleList = nn.ModuleList()
        prev = widths[0]
        for w in widths:
            self.encoder.append(DownStage(prev, w, spec=spec))
            prev = w

        # Per-scale projection heads → unified feature_dim for KD.
        self.projections: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(widths[i], spec.feature_dim, kernel_size=1)
            for i in self._scale_indices
        ])

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for enc in self.encoder:
            h, skip = enc(h)
            skips.append(skip)
        feats: list[torch.Tensor] = []
        for proj, idx in zip(self.projections, self._scale_indices):
            feats.append(proj(skips[idx]))
        return {"features": feats}


@_register("student_teacher")
def _build_student_teacher(spec: ArchSpec) -> nn.Module:
    return StudentTeacherStudent(spec)


# ---------------------------------------------------------------------------
# Architecture: PatchCNN
# ---------------------------------------------------------------------------
class PatchCNN(nn.Module):
    """Fully-convolutional patch-level scorer.

    Behaves like a sliding-window CNN: each spatial position of the output
    corresponds to a receptive field of approximately ``patch_size``.
    The internal anomaly map is the per-position score (sigmoid of logits).
    """

    def __init__(self, spec: ArchSpec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.widths()
        # Choose strides so the cumulative stride matches patch_size when
        # depth is large enough; cap stride product to ≤ patch_size.
        max_strides = max(1, int(round(spec.patch_size).bit_length() - 1))
        n_strides = min(spec.depth, max_strides)

        layers: list[nn.Module] = [
            ConvBNAct(spec.input_channels, widths[0],
                      kernel=spec.kernel_size, stride=1,
                      norm=spec.norm, activation=spec.activation),
        ]
        prev = widths[0]
        for i, w in enumerate(widths):
            stride = 2 if i < n_strides else 1
            layers.append(_make_block(prev, w, spec=spec, stride=stride))
            prev = w

        self.backbone = nn.Sequential(*layers)
        self.score_head = nn.Conv2d(prev, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.score_head(feats)
        anomaly_map = torch.sigmoid(logits)
        # Upsample to the input resolution for pixel-level metrics.
        anomaly_map_full = F.interpolate(
            anomaly_map, size=x.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        score = anomaly_map.flatten(1).amax(dim=1)
        return {
            "logits": logits,
            "features": feats,
            "anomaly_map": anomaly_map_full,
            "score": score,
        }


@_register("patch_cnn")
def _build_patch_cnn(spec: ArchSpec) -> nn.Module:
    return PatchCNN(spec)
