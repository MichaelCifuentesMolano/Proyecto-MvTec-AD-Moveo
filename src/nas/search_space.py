"""
src/nas/search_space.py
=======================

Defines the joint architecture + quantisation search space consumed by the
NSGA-II evolutionary engine.

Every candidate in the NSGA-II population is represented as a flat integer
gene vector of length ``GENOME_LENGTH = 48``.  Each gene is an *index* into a
discrete choice list (not a raw parameter value), so the entire vector lies in
a fixed integer hypercube — ideal for integer-mutation operators.

Gene layout (authoritative, all indices inclusive)
--------------------------------------------------
.. code-block:: none

    ─────────────────────────────────────────────────────────────────
    Gene  0    arch_family      index → ARCH_FAMILIES
    Gene  1    input_size       index → ARCH_INPUT_SIZES
    Gene  2    n_stages         raw   ∈ [MIN_STAGES, MAX_STAGES]
    ─────────────────────────────────────────────────────────────────
    Genes  3– 7  channels[5]   index → ARCH_CHANNELS   (per slot)
    Genes  8–12  kernel[5]     index → ARCH_KERNELS
    Genes 13–17  block[5]      index → ARCH_BLOCKS
    Genes 18–22  bottleneck[5] index → ARCH_BOTTLENECKS
    Genes 23–27  skip[5]       binary ∈ {0, 1}
    Genes 28–32  attention[5]  index → ARCH_ATTENTIONS
    ─────────────────────────────────────────────────────────────────
    Gene 33    global_w_bits    index → QUANT_BITS
    Gene 34    global_a_bits    index → QUANT_BITS
    Gene 35    symmetric        binary ∈ {0, 1}
    Gene 36    per_channel      binary ∈ {0, 1}
    Gene 37    mixed_precision  binary ∈ {0, 1}
    ─────────────────────────────────────────────────────────────────
    Genes 38–42  layer_w_bits[5] index → QUANT_BITS   (mixed-prec)
    Genes 43–47  layer_a_bits[5] index → QUANT_BITS
    ─────────────────────────────────────────────────────────────────

Only the first ``n_stages`` slots of each per-stage section are active.
Per-layer quant genes 38-47 are used only when ``mixed_precision == 1``;
otherwise they are ignored during decode (though they still exist in the
vector and are subject to mutation, which is harmless).

Public interface
----------------
``SearchSpace``
    Main class.  Instantiate once and pass to the NSGA-II engine.

``Genome``
    Decoded, human-readable representation of one candidate.
    ``Genome.to_candidate_dict()`` produces the dict expected by
    ``model_factory.build_model`` and ``qat_wrapper.wrap_for_qat``.

``ArchSearchConfig`` / ``QuantSearchConfig``
    YAML-serialisable config dataclasses that govern which choices are
    available.  Pass to ``SearchSpace.__init__`` to restrict the space.

Assumptions
-----------
- ``model_factory.build_model(candidate)`` consumes the dict returned by
  ``Genome.to_candidate_dict()``.
- ``qat_wrapper.wrap_for_qat(model, qconfig)`` accepts the ``quant_spec``
  sub-dict from the same dict.
- Maximum depth is ``MAX_STAGES = 5``; the genome is always length 48
  regardless of the active stage count.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Sequence

import numpy as np

__all__ = [
    "ArchSearchConfig",
    "QuantSearchConfig",
    "SearchSpaceConfig",
    "Genome",
    "SearchSpace",
    "GENOME_LENGTH",
    "MIN_STAGES",
    "MAX_STAGES",
    "ARCH_FAMILIES",
    "ARCH_INPUT_SIZES",
    "ARCH_CHANNELS",
    "ARCH_KERNELS",
    "ARCH_BLOCKS",
    "ARCH_BOTTLENECKS",
    "ARCH_ATTENTIONS",
    "QUANT_BITS",
]


# ---------------------------------------------------------------------------
# Choice registries  (edit here to expand / restrict the search space)
# ---------------------------------------------------------------------------

ARCH_FAMILIES:     list[str]   = ["autoencoder", "unet", "feature_recon",
                                   "student_teacher", "patch_cnn"]
ARCH_INPUT_SIZES:  list[int]   = [128, 192, 224, 256]
ARCH_CHANNELS:     list[int]   = [16, 24, 32, 48, 64, 96, 128, 192, 256]
ARCH_KERNELS:      list[int]   = [3, 5, 7]
ARCH_BLOCKS:       list[str]   = ["conv", "ds", "ir"]
ARCH_BOTTLENECKS:  list[float] = [0.5, 1.0, 2.0, 4.0, 6.0]
ARCH_ATTENTIONS:   list[str]   = ["none", "se", "eca", "spatial", "cbam"]
QUANT_BITS:        list[int]   = [4, 6, 8]

MIN_STAGES: int = 2
MAX_STAGES: int = 5   # max encoder stages; genome is always this wide

# ---------------------------------------------------------------------------
# Gene index constants  (single source of truth for all gene positions)
# ---------------------------------------------------------------------------

G_FAMILY:       int   = 0
G_INPUT_SIZE:   int   = 1
G_N_STAGES:     int   = 2
G_CHANNELS:     slice = slice(3,  8)    # 5 genes
G_KERNELS:      slice = slice(8,  13)   # 5 genes
G_BLOCKS:       slice = slice(13, 18)   # 5 genes
G_BOTTLENECKS:  slice = slice(18, 23)   # 5 genes
G_SKIP:         slice = slice(23, 28)   # 5 genes
G_ATTN:         slice = slice(28, 33)   # 5 genes
G_W_BITS:       int   = 33
G_A_BITS:       int   = 34
G_SYMMETRIC:    int   = 35
G_PER_CHANNEL:  int   = 36
G_MIXED_PREC:   int   = 37
G_LAYER_W:      slice = slice(38, 43)   # 5 genes
G_LAYER_A:      slice = slice(43, 48)   # 5 genes
GENOME_LENGTH:  int   = 48


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchSearchConfig:
    """Restricts which architecture choices are reachable.

    All lists must be non-empty subsets of the corresponding module-level
    choice registry.  Pass to :class:`SearchSpace` to narrow the space.
    """
    families:         list[str]   = field(default_factory=lambda: list(ARCH_FAMILIES))
    input_sizes:      list[int]   = field(default_factory=lambda: list(ARCH_INPUT_SIZES))
    channels:         list[int]   = field(default_factory=lambda: list(ARCH_CHANNELS))
    kernels:          list[int]   = field(default_factory=lambda: list(ARCH_KERNELS))
    blocks:           list[str]   = field(default_factory=lambda: list(ARCH_BLOCKS))
    bottlenecks:      list[float] = field(default_factory=lambda: list(ARCH_BOTTLENECKS))
    attentions:       list[str]   = field(default_factory=lambda: list(ARCH_ATTENTIONS))
    min_stages:       int         = MIN_STAGES
    max_stages:       int         = MAX_STAGES

    def __post_init__(self) -> None:
        if self.min_stages < 1 or self.max_stages > MAX_STAGES:
            raise ValueError(
                f"stages must be in [1, {MAX_STAGES}]; "
                f"got [{self.min_stages}, {self.max_stages}]."
            )
        if self.min_stages > self.max_stages:
            raise ValueError("min_stages must be ≤ max_stages.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuantSearchConfig:
    """Restricts which quantisation choices are reachable."""
    weight_bits:           list[int] = field(default_factory=lambda: list(QUANT_BITS))
    act_bits:              list[int] = field(default_factory=lambda: list(QUANT_BITS))
    allow_mixed_precision: bool      = True
    allow_asymmetric:      bool      = True
    allow_per_channel:     bool      = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchSpaceConfig:
    """Top-level config combining arch and quant sub-spaces."""
    arch:  ArchSearchConfig  = field(default_factory=ArchSearchConfig)
    quant: QuantSearchConfig = field(default_factory=QuantSearchConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"arch": self.arch.to_dict(), "quant": self.quant.to_dict()}


# ---------------------------------------------------------------------------
# Genome dataclass
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """Human-readable, decoded representation of one search candidate.

    Attributes
    ----------
    arch_family:
        Architecture family string (one of :data:`ARCH_FAMILIES`).
    input_size:
        Spatial input resolution.
    n_stages:
        Number of active encoder stages.
    channels:
        Channel counts for each active stage.  Length == ``n_stages``.
    kernel_sizes:
        Kernel sizes per stage.
    block_types:
        Block type per stage (``"conv"``, ``"ds"``, ``"ir"``).
    bottleneck_ratios:
        Expansion/squeeze ratio per stage.
    skip_connections:
        Whether each stage uses an additive skip.
    attentions:
        Attention mechanism per stage.
    global_weight_bits:
        Global default weight quantisation precision.
    global_act_bits:
        Global default activation quantisation precision.
    symmetric:
        Use symmetric (zero-point = 0) quantisation.
    per_channel:
        Per-channel weight quantisation (vs. per-tensor).
    mixed_precision:
        When ``True``, ``layer_weight_bits`` / ``layer_act_bits`` override
        the global setting per stage.
    layer_weight_bits:
        Per-stage weight bits (length ``n_stages``; meaningful only when
        ``mixed_precision`` is ``True``).
    layer_act_bits:
        Per-stage activation bits.
    """
    # Architecture
    arch_family:        str
    input_size:         int
    n_stages:           int
    channels:           list[int]
    kernel_sizes:       list[int]
    block_types:        list[str]
    bottleneck_ratios:  list[float]
    skip_connections:   list[bool]
    attentions:         list[str]
    # Quantisation
    global_weight_bits: int
    global_act_bits:    int
    symmetric:          bool
    per_channel:        bool
    mixed_precision:    bool
    layer_weight_bits:  list[int]
    layer_act_bits:     list[int]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_candidate_dict(self) -> dict[str, Any]:
        """Return the dict consumed by ``model_factory.build_model``.

        The returned structure has two top-level keys:

        - ``"arch_spec"`` — consumed by the model factory.
        - ``"quant_spec"`` — consumed by ``qat_wrapper.wrap_for_qat``.

        Both are plain dicts (JSON-serialisable, no custom types).
        """
        # Per-layer weight bits: use layer-specific when mixed_precision.
        if self.mixed_precision:
            w_bits: int | list[int] = list(self.layer_weight_bits)
            a_bits: int | list[int] = list(self.layer_act_bits)
        else:
            w_bits = self.global_weight_bits
            a_bits = self.global_act_bits

        return {
            "arch_spec": {
                "family":            self.arch_family,
                "input_size":        self.input_size,
                "depth":             self.n_stages,
                "channels":          list(self.channels),
                "kernel_size":       list(self.kernel_sizes),
                "block_type":        list(self.block_types),
                "bottleneck_ratio":  list(self.bottleneck_ratios),
                "skip_connections":  list(self.skip_connections),
                "attention":         list(self.attentions),
            },
            "quant_spec": {
                "weight_bits":       w_bits,
                "act_bits":          a_bits,
                "symmetric":         self.symmetric,
                "per_channel":       self.per_channel,
                "mixed_precision":   self.mixed_precision,
                # Flat copies for downstream introspection
                "global_weight_bits": self.global_weight_bits,
                "global_act_bits":    self.global_act_bits,
                "layer_weight_bits":  list(self.layer_weight_bits),
                "layer_act_bits":     list(self.layer_act_bits),
            },
        }

    def summary(self) -> dict[str, Any]:
        """Compact summary dict for logging / CSV export."""
        return {
            "arch_family":       self.arch_family,
            "input_size":        self.input_size,
            "n_stages":          self.n_stages,
            "channels":          "-".join(str(c) for c in self.channels),
            "kernel_sizes":      "-".join(str(k) for k in self.kernel_sizes),
            "block_types":       "-".join(self.block_types),
            "bottleneck_ratios": "-".join(f"{b:.1f}" for b in self.bottleneck_ratios),
            "skip_connections":  "".join("1" if s else "0" for s in self.skip_connections),
            "attentions":        "-".join(self.attentions),
            "global_weight_bits": self.global_weight_bits,
            "global_act_bits":    self.global_act_bits,
            "symmetric":         int(self.symmetric),
            "per_channel":       int(self.per_channel),
            "mixed_precision":   int(self.mixed_precision),
            "layer_weight_bits": "-".join(str(b) for b in self.layer_weight_bits)
                                 if self.mixed_precision else "",
            "layer_act_bits":    "-".join(str(b) for b in self.layer_act_bits)
                                 if self.mixed_precision else "",
        }

    def n_params_upper_bound(self) -> int:
        """Very rough estimate of total parameters (useful for filtering)."""
        total = 0
        prev_ch = 3  # RGB input
        for ch, ks, bt in zip(self.channels, self.kernel_sizes, self.bottleneck_ratios):
            mid_ch = max(1, int(ch * bt))
            # Approximate: two conv layers per stage
            total += prev_ch * mid_ch * ks * ks + mid_ch * ch * 1 * 1 + ch
            prev_ch = ch
        return total

    def __repr__(self) -> str:
        mp_str = " MP" if self.mixed_precision else ""
        sym    = "S" if self.symmetric else "A"
        return (
            f"Genome({self.arch_family}, d={self.n_stages}, "
            f"ch={self.channels}, "
            f"W{self.global_weight_bits}A{self.global_act_bits}{sym}{mp_str})"
        )


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

class SearchSpace:
    """Defines, samples, encodes, and decodes the NAS search space.

    Parameters
    ----------
    config:
        Optional :class:`SearchSpaceConfig` to restrict choices.  Defaults
        to the full space defined by the module-level choice registries.

    Examples
    --------
    >>> ss = SearchSpace()
    >>> gene_vec = ss.sample(rng=np.random.default_rng(0))
    >>> genome   = ss.decode(gene_vec)
    >>> candidate = genome.to_candidate_dict()
    >>> gene_back = ss.encode(genome)
    >>> assert np.array_equal(gene_vec, gene_back)
    """

    def __init__(self, config: SearchSpaceConfig | None = None) -> None:
        cfg  = config or SearchSpaceConfig()
        acfg = cfg.arch
        qcfg = cfg.quant

        # Validate subsets against global registries
        self._families    = _subset(acfg.families,    ARCH_FAMILIES,    "families")
        self._input_sizes = _subset(acfg.input_sizes,  ARCH_INPUT_SIZES,  "input_sizes")
        self._channels    = _subset(acfg.channels,     ARCH_CHANNELS,     "channels")
        self._kernels     = _subset(acfg.kernels,      ARCH_KERNELS,      "kernels")
        self._blocks      = _subset(acfg.blocks,       ARCH_BLOCKS,       "blocks")
        self._bottlenecks = _subset(acfg.bottlenecks,  ARCH_BOTTLENECKS,  "bottlenecks")
        self._attentions  = _subset(acfg.attentions,   ARCH_ATTENTIONS,   "attentions")
        self._min_stages  = acfg.min_stages
        self._max_stages  = acfg.max_stages

        self._w_bits = _subset(qcfg.weight_bits, QUANT_BITS, "weight_bits")
        self._a_bits = _subset(qcfg.act_bits,    QUANT_BITS, "act_bits")
        self._allow_mixed = qcfg.allow_mixed_precision
        self._allow_asym  = qcfg.allow_asymmetric
        self._allow_perch = qcfg.allow_per_channel

        # Pre-compute bounds arrays (inclusive)
        self._lo, self._hi = self._build_bounds()
        self._config = cfg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def genome_length(self) -> int:
        return GENOME_LENGTH

    @property
    def lower_bounds(self) -> np.ndarray:
        """Integer lower bound for each gene (inclusive)."""
        return self._lo.copy()

    @property
    def upper_bounds(self) -> np.ndarray:
        """Integer upper bound for each gene (inclusive)."""
        return self._hi.copy()

    @property
    def config(self) -> SearchSpaceConfig:
        return self._config

    # ------------------------------------------------------------------
    # Bound construction
    # ------------------------------------------------------------------

    def _build_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(GENOME_LENGTH, dtype=np.int32)
        hi = np.zeros(GENOME_LENGTH, dtype=np.int32)

        hi[G_FAMILY]      = len(self._families)    - 1
        hi[G_INPUT_SIZE]  = len(self._input_sizes) - 1
        lo[G_N_STAGES]    = self._min_stages
        hi[G_N_STAGES]    = self._max_stages

        for sl, choices in [
            (G_CHANNELS,    self._channels),
            (G_KERNELS,     self._kernels),
            (G_BLOCKS,      self._blocks),
            (G_BOTTLENECKS, self._bottlenecks),
            (G_ATTN,        self._attentions),
        ]:
            hi[sl] = len(choices) - 1

        hi[G_SKIP] = 1  # binary

        hi[G_W_BITS]      = len(self._w_bits) - 1
        hi[G_A_BITS]      = len(self._a_bits) - 1
        hi[G_SYMMETRIC]   = 1 if self._allow_asym  else 0
        hi[G_PER_CHANNEL] = 1 if self._allow_perch else 0
        hi[G_MIXED_PREC]  = 1 if self._allow_mixed else 0

        hi[G_LAYER_W] = len(self._w_bits) - 1
        hi[G_LAYER_A] = len(self._a_bits) - 1

        return lo, hi

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a uniformly random valid gene vector.

        Parameters
        ----------
        rng:
            NumPy random generator for reproducibility.  Creates a new
            default generator when ``None``.

        Returns
        -------
        np.ndarray
            Integer gene vector of length :data:`GENOME_LENGTH`.
        """
        rng = rng or np.random.default_rng()
        vec = np.array(
            [rng.integers(int(lo), int(hi) + 1) for lo, hi in zip(self._lo, self._hi)],
            dtype=np.int32,
        )
        return vec

    def sample_population(self,
                          size: int,
                          rng: np.random.Generator | None = None
                          ) -> np.ndarray:
        """Sample a full population matrix of shape ``[size, genome_length]``."""
        rng = rng or np.random.default_rng()
        return np.vstack([self.sample(rng) for _ in range(size)])

    # ------------------------------------------------------------------
    # Decode: gene vector → Genome
    # ------------------------------------------------------------------

    def decode(self, gene_vec: np.ndarray) -> Genome:
        """Decode an integer gene vector into a :class:`Genome`.

        Only the first ``n_stages`` slots of each per-stage section are
        used; the rest are silently ignored.

        Parameters
        ----------
        gene_vec:
            Integer array of length :data:`GENOME_LENGTH`.

        Returns
        -------
        Genome
        """
        v = np.asarray(gene_vec, dtype=np.int32)
        if v.shape != (GENOME_LENGTH,):
            raise ValueError(
                f"Expected gene vector of length {GENOME_LENGTH}; "
                f"got {v.shape}."
            )

        n = int(np.clip(v[G_N_STAGES], self._min_stages, self._max_stages))

        def _idx(sl: slice | int, choices: list, clip: bool = True) -> Any:
            """Read one or five gene(s) and map to choice(s)."""
            if isinstance(sl, int):
                idx = int(v[sl])
                if clip:
                    idx = int(np.clip(idx, 0, len(choices) - 1))
                return choices[idx]
            idxs = v[sl][:n]
            return [choices[int(np.clip(i, 0, len(choices) - 1))] for i in idxs]

        mixed = bool(int(np.clip(v[G_MIXED_PREC], 0, self._hi[G_MIXED_PREC])))

        layer_w = [self._w_bits[int(np.clip(i, 0, len(self._w_bits)-1))]
                   for i in v[G_LAYER_W][:n]]
        layer_a = [self._a_bits[int(np.clip(i, 0, len(self._a_bits)-1))]
                   for i in v[G_LAYER_A][:n]]

        return Genome(
            arch_family       = _idx(G_FAMILY,      self._families),
            input_size        = _idx(G_INPUT_SIZE,   self._input_sizes),
            n_stages          = n,
            channels          = _idx(G_CHANNELS,     self._channels),
            kernel_sizes      = _idx(G_KERNELS,      self._kernels),
            block_types       = _idx(G_BLOCKS,       self._blocks),
            bottleneck_ratios = _idx(G_BOTTLENECKS,  self._bottlenecks),
            skip_connections  = [bool(x) for x in v[G_SKIP][:n]],
            attentions        = _idx(G_ATTN,         self._attentions),
            global_weight_bits= _idx(G_W_BITS,       self._w_bits),
            global_act_bits   = _idx(G_A_BITS,       self._a_bits),
            symmetric         = bool(int(np.clip(v[G_SYMMETRIC],   0, self._hi[G_SYMMETRIC]))),
            per_channel       = bool(int(np.clip(v[G_PER_CHANNEL], 0, self._hi[G_PER_CHANNEL]))),
            mixed_precision   = mixed,
            layer_weight_bits = layer_w,
            layer_act_bits    = layer_a,
        )

    # ------------------------------------------------------------------
    # Encode: Genome → gene vector
    # ------------------------------------------------------------------

    def encode(self, genome: Genome) -> np.ndarray:
        """Encode a :class:`Genome` back to an integer gene vector.

        This is the exact inverse of :meth:`decode` for genomes produced
        by :meth:`decode`.  Genes for inactive stages are set to their
        lower bound.

        Parameters
        ----------
        genome:
            A :class:`Genome` instance produced (or compatible with) this
            search space.

        Returns
        -------
        np.ndarray
            Integer array of length :data:`GENOME_LENGTH`.
        """
        v = np.zeros(GENOME_LENGTH, dtype=np.int32)

        def _rev(val: Any, choices: list) -> int:
            try:
                return choices.index(val)
            except ValueError:
                # Nearest by value for numeric choices
                dists = [abs(float(val) - float(c)) for c in choices]
                return int(np.argmin(dists))

        v[G_FAMILY]      = _rev(genome.arch_family,  self._families)
        v[G_INPUT_SIZE]  = _rev(genome.input_size,   self._input_sizes)
        v[G_N_STAGES]    = int(np.clip(genome.n_stages, self._min_stages, self._max_stages))

        n = genome.n_stages
        for slot in range(MAX_STAGES):
            active = slot < n
            ch_idx    = _rev(genome.channels[slot],          self._channels)    if active else 0
            ks_idx    = _rev(genome.kernel_sizes[slot],      self._kernels)     if active else 0
            bl_idx    = _rev(genome.block_types[slot],       self._blocks)      if active else 0
            bn_idx    = _rev(genome.bottleneck_ratios[slot], self._bottlenecks) if active else 0
            sk_val    = int(genome.skip_connections[slot])                       if active else 0
            at_idx    = _rev(genome.attentions[slot],         self._attentions)  if active else 0
            lw_idx    = _rev(genome.layer_weight_bits[slot],  self._w_bits)      if active else 0
            la_idx    = _rev(genome.layer_act_bits[slot],     self._a_bits)      if active else 0

            v[G_CHANNELS.start    + slot] = ch_idx
            v[G_KERNELS.start     + slot] = ks_idx
            v[G_BLOCKS.start      + slot] = bl_idx
            v[G_BOTTLENECKS.start + slot] = bn_idx
            v[G_SKIP.start        + slot] = sk_val
            v[G_ATTN.start        + slot] = at_idx
            v[G_LAYER_W.start     + slot] = lw_idx
            v[G_LAYER_A.start     + slot] = la_idx

        v[G_W_BITS]      = _rev(genome.global_weight_bits, self._w_bits)
        v[G_A_BITS]      = _rev(genome.global_act_bits,    self._a_bits)
        v[G_SYMMETRIC]   = int(genome.symmetric)
        v[G_PER_CHANNEL] = int(genome.per_channel)
        v[G_MIXED_PREC]  = int(genome.mixed_precision)

        return v

    # ------------------------------------------------------------------
    # Validation and repair
    # ------------------------------------------------------------------

    def is_valid(self, gene_vec: np.ndarray) -> bool:
        """Return ``True`` if every gene is within its defined bounds."""
        v = np.asarray(gene_vec, dtype=np.int32)
        if v.shape != (GENOME_LENGTH,):
            return False
        return bool(np.all(v >= self._lo) and np.all(v <= self._hi))

    def clip(self, gene_vec: np.ndarray) -> np.ndarray:
        """Project a gene vector onto the feasible integer hypercube."""
        return np.clip(
            np.asarray(gene_vec, dtype=np.int32), self._lo, self._hi
        ).astype(np.int32)

    def repair(self, gene_vec: np.ndarray) -> np.ndarray:
        """Clip + enforce stage consistency.

        Ensures ``n_stages`` is within ``[min_stages, max_stages]`` and
        that per-stage genes for inactive slots are set to their lower
        bound (cosmetic — decode ignores them anyway).
        """
        v = self.clip(gene_vec)
        n = int(v[G_N_STAGES])
        for slot in range(n, MAX_STAGES):
            for sl in (G_CHANNELS, G_KERNELS, G_BLOCKS, G_BOTTLENECKS,
                       G_SKIP, G_ATTN, G_LAYER_W, G_LAYER_A):
                v[sl.start + slot] = self._lo[sl.start + slot]
        return v

    # ------------------------------------------------------------------
    # Crossover and mutation operators
    # ------------------------------------------------------------------

    def crossover_single_point(
            self,
            parent_a: np.ndarray,
            parent_b: np.ndarray,
            rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-point crossover that respects gene section boundaries.

        The cut point is drawn uniformly from the set of section boundaries
        ``{2, 7, 12, 17, 22, 27, 32, 37, 42}`` so that stage-slots are
        never split mid-section.

        Returns
        -------
        tuple of two offspring gene vectors.
        """
        # Section boundary cut points
        boundaries = [2, 7, 12, 17, 22, 27, 32, 37, 42]
        cut = int(rng.choice(boundaries))
        child_a = np.concatenate([parent_a[:cut], parent_b[cut:]])
        child_b = np.concatenate([parent_b[:cut], parent_a[cut:]])
        return self.repair(child_a), self.repair(child_b)

    def crossover_uniform(
            self,
            parent_a: np.ndarray,
            parent_b: np.ndarray,
            rng: np.random.Generator,
            swap_prob: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform crossover: each gene independently swapped with probability
        ``swap_prob``.

        Returns
        -------
        tuple of two offspring gene vectors.
        """
        mask    = rng.random(GENOME_LENGTH) < swap_prob
        child_a = np.where(mask, parent_b, parent_a).astype(np.int32)
        child_b = np.where(mask, parent_a, parent_b).astype(np.int32)
        return self.repair(child_a), self.repair(child_b)

    def mutate(
            self,
            gene_vec: np.ndarray,
            rng: np.random.Generator,
            mutation_rate: float = 1.0 / GENOME_LENGTH,
    ) -> np.ndarray:
        """Integer random-resetting mutation.

        Each gene is independently replaced by a uniformly random value
        within its bounds with probability ``mutation_rate``.

        Parameters
        ----------
        gene_vec:
            Original gene vector.
        rng:
            NumPy random generator.
        mutation_rate:
            Per-gene mutation probability.  Default is the classic
            ``1/L`` rule (expected one mutation per genome).

        Returns
        -------
        np.ndarray
            Mutated gene vector (new object; original unmodified).
        """
        v    = gene_vec.copy()
        mask = rng.random(GENOME_LENGTH) < mutation_rate
        for i in np.where(mask)[0]:
            v[i] = int(rng.integers(int(self._lo[i]), int(self._hi[i]) + 1))
        return self.repair(v)

    def mutate_neighbourhood(
            self,
            gene_vec: np.ndarray,
            rng: np.random.Generator,
            mutation_rate: float = 1.0 / GENOME_LENGTH,
            step: int = 1,
    ) -> np.ndarray:
        """Neighbourhood mutation: mutated genes move ±``step`` (clamped).

        Preserves locality — useful for fine-grained exploration near the
        Pareto front in later generations.
        """
        v    = gene_vec.copy()
        mask = rng.random(GENOME_LENGTH) < mutation_rate
        for i in np.where(mask)[0]:
            delta = rng.choice([-step, +step])
            v[i]  = int(np.clip(v[i] + delta, self._lo[i], self._hi[i]))
        return self.repair(v)

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------

    def genome_to_candidate(self, gene_vec: np.ndarray) -> dict[str, Any]:
        """Shortcut: decode + ``Genome.to_candidate_dict()`` in one call."""
        return self.decode(gene_vec).to_candidate_dict()

    def iter_dimensions(self) -> Iterator[tuple[int, str, int, int]]:
        """Iterate over ``(gene_index, description, lo, hi)`` for all genes.

        Useful for building solver variable lists (e.g., pymoo IntegerVar).
        """
        names: list[str] = (
            ["arch_family", "input_size", "n_stages"]
            + [f"channels_{i}"    for i in range(MAX_STAGES)]
            + [f"kernel_{i}"      for i in range(MAX_STAGES)]
            + [f"block_{i}"       for i in range(MAX_STAGES)]
            + [f"bottleneck_{i}"  for i in range(MAX_STAGES)]
            + [f"skip_{i}"        for i in range(MAX_STAGES)]
            + [f"attention_{i}"   for i in range(MAX_STAGES)]
            + ["global_w_bits", "global_a_bits",
               "symmetric", "per_channel", "mixed_precision"]
            + [f"layer_w_bits_{i}" for i in range(MAX_STAGES)]
            + [f"layer_a_bits_{i}" for i in range(MAX_STAGES)]
        )
        for i, name in enumerate(names):
            yield i, name, int(self._lo[i]), int(self._hi[i])

    def n_configurations(self) -> int:
        """Upper bound on the number of distinct configurations.

        The true count is smaller because inactive-stage genes are masked
        and mixed-precision genes are conditional.
        """
        total = 1
        for lo, hi in zip(self._lo, self._hi):
            total *= int(hi - lo + 1)
        return total

    def describe(self) -> str:
        """Return a human-readable summary of the search space."""
        nc = self.n_configurations()
        lines = [
            "SearchSpace",
            f"  genome_length      : {GENOME_LENGTH}",
            f"  upper bound config : {nc:.3e}",
            f"  stages             : [{self._min_stages}, {self._max_stages}]",
            f"  families           : {self._families}",
            f"  input_sizes        : {self._input_sizes}",
            f"  channels           : {self._channels}",
            f"  kernels            : {self._kernels}",
            f"  blocks             : {self._blocks}",
            f"  bottlenecks        : {self._bottlenecks}",
            f"  attentions         : {self._attentions}",
            f"  weight_bits        : {self._w_bits}",
            f"  act_bits           : {self._a_bits}",
            f"  allow_mixed_prec   : {self._allow_mixed}",
            f"  allow_asymmetric   : {self._allow_asym}",
            f"  allow_per_channel  : {self._allow_perch}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SearchSpace(genome_length={GENOME_LENGTH}, "
            f"stages=[{self._min_stages},{self._max_stages}], "
            f"families={len(self._families)}, "
            f"bits={self._w_bits})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _subset(values: list, registry: list, name: str) -> list:
    """Validate that ``values`` is a non-empty subset of ``registry``."""
    unknown = [v for v in values if v not in registry]
    if unknown:
        warnings.warn(
            f"SearchSpace: unknown {name} entries will be ignored: {unknown}",
            UserWarning, stacklevel=3,
        )
        values = [v for v in values if v in registry]
    if not values:
        raise ValueError(
            f"SearchSpace: '{name}' choice list is empty after filtering. "
            f"Registry: {registry}."
        )
    return list(values)
