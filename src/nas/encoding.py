"""
src/nas/encoding.py
===================

Genome ↔ architecture/quantisation configuration codec.

This module is the low-level translation layer between the three
representations used by the pipeline:

1. **Gene vector** — ``np.ndarray`` of shape ``(GENOME_LENGTH,)`` with
   ``int32`` dtype.  The integer hypercube representation consumed by
   the NSGA-II engine (sampling, crossover, mutation).

2. **Candidate dict** — plain Python dict with two sub-dicts
   ``{"arch_spec": {...}, "quant_spec": {...}}``, consumed by
   ``model_factory.build_model`` and ``qat_wrapper.wrap_for_qat``.

3. **String / JSON** — compact serialisable forms for logging, CSV
   columns, checkpoints, and deduplication.

Relationship to ``search_space.py``
------------------------------------
``search_space.SearchSpace`` *defines* what is valid (choice lists,
bounds, sampling, crossover, mutation) and exposes ``decode`` / ``encode``
via the ``Genome`` dataclass.

``encoding.GenomeEncoder`` wraps a ``SearchSpace`` and adds:

- Direct gene-vector ↔ candidate-dict conversion (no intermediate ``Genome``
  object required at call sites).
- JSON serialisation with metadata (space fingerprint, timestamp).
- Compact hex encoding for CSV / filename use.
- SHA-256 fingerprinting for deduplication.
- Hamming distance for population diversity metrics.
- Semantic validation of both gene vectors and candidate dicts.
- Architectural complexity estimation (parameter count, MACs) that does
  not require building a PyTorch model.

Standalone functions
--------------------
``validate_arch_spec``, ``validate_quant_spec``
    Return a list of error strings (empty = valid).  Callable without a
    ``SearchSpace`` instance.

``complete_arch_defaults``, ``complete_quant_defaults``
    Fill missing optional fields with sensible defaults so
    ``model_factory`` never receives an incomplete dict.

``estimate_complexity``
    Approximate parameter count and MACs from an ``arch_spec`` dict
    without instantiating a model.

Assumptions
-----------
- Gene-vector layout is governed by the constants in ``search_space.py``
  (``G_FAMILY``, ``G_CHANNELS``, etc.).
- A ``SearchSpace`` with the *full* default choices is used when none is
  provided to :class:`GenomeEncoder`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any, Sequence

import numpy as np

from src.nas.search_space import (
    ARCH_ATTENTIONS, ARCH_BLOCKS, ARCH_BOTTLENECKS, ARCH_CHANNELS,
    ARCH_FAMILIES, ARCH_INPUT_SIZES, ARCH_KERNELS,
    GENOME_LENGTH, MAX_STAGES, MIN_STAGES, QUANT_BITS,
    SearchSpace, SearchSpaceConfig,
)

__all__ = [
    "GenomeEncoder",
    "validate_arch_spec",
    "validate_quant_spec",
    "complete_arch_defaults",
    "complete_quant_defaults",
    "estimate_complexity",
]

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field contracts  (what model_factory / qat_wrapper require)
# ---------------------------------------------------------------------------

#: Mandatory keys for an arch_spec dict.
_ARCH_REQUIRED: frozenset[str] = frozenset({
    "family", "input_size", "depth",
    "channels", "kernel_size", "block_type",
})

#: Mandatory keys for a quant_spec dict.
_QUANT_REQUIRED: frozenset[str] = frozenset({
    "weight_bits", "act_bits", "symmetric", "per_channel",
})

#: Default values for optional arch_spec fields.
_ARCH_DEFAULTS: dict[str, Any] = {
    "bottleneck_ratio":  1.0,
    "skip_connections":  False,
    "attention":         "none",
}

#: Default values for optional quant_spec fields.
_QUANT_DEFAULTS: dict[str, Any] = {
    "mixed_precision":    False,
    "layer_weight_bits":  None,
    "layer_act_bits":     None,
    "global_weight_bits": None,
    "global_act_bits":    None,
}

#: Valid string sets for categorical arch_spec fields.
_VALID_FAMILIES:    frozenset = frozenset(ARCH_FAMILIES)
_VALID_BLOCKS:      frozenset = frozenset(ARCH_BLOCKS)
_VALID_ATTENTIONS:  frozenset = frozenset(ARCH_ATTENTIONS)


# ---------------------------------------------------------------------------
# GenomeEncoder
# ---------------------------------------------------------------------------

class GenomeEncoder:
    """Bidirectional codec between gene vectors and candidate dicts.

    Parameters
    ----------
    search_space:
        The :class:`~src.nas.search_space.SearchSpace` that defines the
        valid choices.  When ``None`` the full default space is used.

    Examples
    --------
    >>> enc = GenomeEncoder()
    >>> v   = enc.search_space.sample(np.random.default_rng(0))
    >>> d   = enc.gene_to_dict(v)
    >>> v2  = enc.dict_to_gene(d)
    >>> assert np.array_equal(v, v2)
    >>> enc.fingerprint(v)
    'a3f9d12b'
    """

    def __init__(self,
                 search_space: SearchSpace | None = None,
                 strict: bool = False) -> None:
        """
        Parameters
        ----------
        search_space:
            Optional restricted search space.
        strict:
            When ``True``, ``dict_to_gene`` and validation functions raise
            ``ValueError`` on invalid input instead of returning error lists.
        """
        self._ss     = search_space or SearchSpace()
        self._strict = strict

    @property
    def search_space(self) -> SearchSpace:
        return self._ss

    # ------------------------------------------------------------------
    # Core conversions
    # ------------------------------------------------------------------

    def gene_to_dict(self, gene_vec: np.ndarray) -> dict[str, Any]:
        """Gene vector → full candidate dict (arch_spec + quant_spec).

        Equivalent to ``SearchSpace.decode(v).to_candidate_dict()``
        but callable without an intermediate ``Genome`` object.

        Parameters
        ----------
        gene_vec:
            Integer array of length :data:`~src.nas.search_space.GENOME_LENGTH`.

        Returns
        -------
        dict
            ``{"arch_spec": {...}, "quant_spec": {...}}``.
        """
        genome = self._ss.decode(gene_vec)
        return genome.to_candidate_dict()

    def gene_to_arch_spec(self, gene_vec: np.ndarray) -> dict[str, Any]:
        """Extract only the ``arch_spec`` sub-dict from a gene vector."""
        return self.gene_to_dict(gene_vec)["arch_spec"]

    def gene_to_quant_spec(self, gene_vec: np.ndarray) -> dict[str, Any]:
        """Extract only the ``quant_spec`` sub-dict from a gene vector."""
        return self.gene_to_dict(gene_vec)["quant_spec"]

    def dict_to_gene(self, candidate_dict: dict[str, Any]) -> np.ndarray:
        """Candidate dict → gene vector.

        Accepts the two-key ``{"arch_spec": ..., "quant_spec": ...}`` form
        produced by :meth:`gene_to_dict`, or a flat dict with all keys at
        the top level (legacy / manual configs).

        Unknown or out-of-range values are mapped to the nearest valid
        choice via the ``SearchSpace.encode`` nearest-neighbour fallback.

        Parameters
        ----------
        candidate_dict:
            Candidate dict.

        Returns
        -------
        np.ndarray
            Integer gene vector of length :data:`GENOME_LENGTH`.
        """
        # Normalise flat vs. nested dict
        if "arch_spec" in candidate_dict:
            arch  = dict(candidate_dict["arch_spec"])
            quant = dict(candidate_dict.get("quant_spec", {}))
        else:
            arch  = {k: v for k, v in candidate_dict.items()
                     if k not in _QUANT_REQUIRED}
            quant = {k: v for k, v in candidate_dict.items()
                     if k in _QUANT_REQUIRED | set(_QUANT_DEFAULTS)}

        arch  = complete_arch_defaults(arch)
        quant = complete_quant_defaults(quant)

        errs = validate_arch_spec(arch) + validate_quant_spec(quant)
        if errs and self._strict:
            raise ValueError("Invalid candidate dict:\n" + "\n".join(errs))
        if errs:
            LOG.debug("dict_to_gene: %d validation issue(s): %s", len(errs), errs)

        from src.nas.search_space import Genome  # local import to avoid circularity
        n = int(arch.get("depth", 3))

        def _list(val: Any, n: int, default: Any) -> list:
            if isinstance(val, (list, tuple)):
                lst = list(val)
                return (lst + [lst[-1]] * n)[:n] if lst else [default] * n
            return [val] * n

        channels    = _list(arch.get("channels", 32),       n, 32)
        kernels     = _list(arch.get("kernel_size", 3),     n, 3)
        blocks      = _list(arch.get("block_type", "conv"), n, "conv")
        bottlenecks = _list(arch.get("bottleneck_ratio", 1.0), n, 1.0)
        skips       = _list(arch.get("skip_connections", False), n, False)
        attentions  = _list(arch.get("attention", "none"),  n, "none")

        mixed   = bool(quant.get("mixed_precision", False))
        g_wbits = quant.get("weight_bits", quant.get("global_weight_bits", 8))
        g_abits = quant.get("act_bits",    quant.get("global_act_bits",    8))
        if isinstance(g_wbits, list):
            g_wbits = g_wbits[0]
        if isinstance(g_abits, list):
            g_abits = g_abits[0]

        lw = _list(quant.get("layer_weight_bits") or g_wbits, n, int(g_wbits))
        la = _list(quant.get("layer_act_bits")    or g_abits, n, int(g_abits))

        genome = Genome(
            arch_family       = str(arch.get("family", "autoencoder")),
            input_size        = int(arch.get("input_size", 224)),
            n_stages          = n,
            channels          = [int(c) for c in channels],
            kernel_sizes      = [int(k) for k in kernels],
            block_types       = [str(b) for b in blocks],
            bottleneck_ratios = [float(b) for b in bottlenecks],
            skip_connections  = [bool(s) for s in skips],
            attentions        = [str(a) for a in attentions],
            global_weight_bits= int(g_wbits),
            global_act_bits   = int(g_abits),
            symmetric         = bool(quant.get("symmetric", True)),
            per_channel       = bool(quant.get("per_channel", False)),
            mixed_precision   = mixed,
            layer_weight_bits = [int(b) for b in lw],
            layer_act_bits    = [int(b) for b in la],
        )
        return self._ss.encode(genome)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_gene_to_dict(self,
                           population: np.ndarray
                           ) -> list[dict[str, Any]]:
        """Decode a population matrix ``[N, GENOME_LENGTH]`` to a list of dicts."""
        return [self.gene_to_dict(population[i]) for i in range(len(population))]

    def batch_fingerprint(self, population: np.ndarray) -> list[str]:
        """Return a fingerprint for every row of a population matrix."""
        return [self.fingerprint(population[i]) for i in range(len(population))]

    # ------------------------------------------------------------------
    # Fingerprinting and distance
    # ------------------------------------------------------------------

    def fingerprint(self, gene_vec: np.ndarray) -> str:
        """Return an 8-hex-char SHA-256 fingerprint of a gene vector.

        Suitable for deduplication and as a short unique candidate ID.

        Parameters
        ----------
        gene_vec:
            Gene vector to hash.

        Returns
        -------
        str
            8-character lowercase hexadecimal string.
        """
        raw = np.asarray(gene_vec, dtype=np.int32).tobytes()
        return hashlib.sha256(raw).hexdigest()[:8]

    @staticmethod
    def hamming(a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two gene vectors (number of differing genes).

        Parameters
        ----------
        a, b:
            Integer gene vectors of equal length.

        Returns
        -------
        int
            Number of positions where ``a`` and ``b`` differ.
        """
        return int(np.sum(np.asarray(a, dtype=np.int32) != np.asarray(b, dtype=np.int32)))

    @staticmethod
    def population_diversity(population: np.ndarray) -> dict[str, float]:
        """Compute pairwise Hamming diversity statistics for a population.

        Parameters
        ----------
        population:
            Integer matrix of shape ``[N, GENOME_LENGTH]``.

        Returns
        -------
        dict
            Keys: ``mean``, ``min``, ``max``, ``std`` of pairwise distances.
        """
        n = len(population)
        if n < 2:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        dists: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(GenomeEncoder.hamming(population[i], population[j]))
        arr = np.asarray(dists, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "std":  float(arr.std()),
        }

    # ------------------------------------------------------------------
    # String / hex encoding
    # ------------------------------------------------------------------

    def gene_to_hex(self, gene_vec: np.ndarray) -> str:
        """Encode a gene vector as a compact hexadecimal string.

        The string is exactly ``GENOME_LENGTH * 2`` hex characters (one
        byte per gene value in ``[0, 255]``).  Gene values are stored as
        ``uint8``; values exceeding 255 are clamped (should not occur with
        the current 48-gene layout, where max index is at most 8).

        Parameters
        ----------
        gene_vec:
            Integer gene vector.

        Returns
        -------
        str
            Hex string, e.g. ``"0204030101…"``.
        """
        arr = np.clip(np.asarray(gene_vec, dtype=np.int32), 0, 255).astype(np.uint8)
        return arr.tobytes().hex()

    def hex_to_gene(self, hex_str: str) -> np.ndarray:
        """Decode a hex string produced by :meth:`gene_to_hex`.

        Parameters
        ----------
        hex_str:
            Hexadecimal string of length ``GENOME_LENGTH * 2``.

        Returns
        -------
        np.ndarray
            Integer gene vector of length :data:`GENOME_LENGTH`.
        """
        expected_len = GENOME_LENGTH * 2
        if len(hex_str) != expected_len:
            raise ValueError(
                f"hex_to_gene: expected string of length {expected_len}, "
                f"got {len(hex_str)}."
            )
        raw = bytes.fromhex(hex_str)
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int32)

    def gene_to_csv_row(self, gene_vec: np.ndarray) -> str:
        """Return a comma-separated string of gene integers for CSV logging."""
        return ",".join(str(int(x)) for x in gene_vec)

    def csv_row_to_gene(self, row_str: str) -> np.ndarray:
        """Parse a comma-separated gene string produced by :meth:`gene_to_csv_row`."""
        parts = row_str.strip().split(",")
        if len(parts) != GENOME_LENGTH:
            raise ValueError(
                f"csv_row_to_gene: expected {GENOME_LENGTH} comma-separated integers, "
                f"got {len(parts)}."
            )
        return np.array([int(p) for p in parts], dtype=np.int32)

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def to_json(self,
                gene_vec: np.ndarray,
                *,
                extra: dict[str, Any] | None = None) -> str:
        """Serialise a gene vector to a JSON string.

        The envelope includes:

        - ``"genes"``        — list of integers.
        - ``"fingerprint"``  — 8-char hex ID.
        - ``"hex"``          — compact hex encoding.
        - ``"candidate"``    — decoded candidate dict (for human inspection).
        - ``"timestamp"``    — UNIX epoch float.
        - any extra key-value pairs from ``extra``.

        Parameters
        ----------
        gene_vec:
            Gene vector to serialise.
        extra:
            Optional additional metadata to embed.

        Returns
        -------
        str
            JSON string.
        """
        candidate = self.gene_to_dict(gene_vec)
        doc: dict[str, Any] = {
            "genes":       [int(x) for x in gene_vec],
            "fingerprint": self.fingerprint(gene_vec),
            "hex":         self.gene_to_hex(gene_vec),
            "timestamp":   time.time(),
            "candidate":   _jsonify(candidate),
        }
        if extra:
            doc.update({k: _jsonify(v) for k, v in extra.items()})
        return json.dumps(doc, indent=2)

    def from_json(self, json_str: str) -> np.ndarray:
        """Deserialise a gene vector from a JSON string produced by :meth:`to_json`.

        Accepts both the envelope form (with ``"genes"`` key) and a bare
        list of integers.

        Parameters
        ----------
        json_str:
            JSON string.

        Returns
        -------
        np.ndarray
            Integer gene vector of length :data:`GENOME_LENGTH`.
        """
        doc = json.loads(json_str)
        if isinstance(doc, list):
            genes = doc
        elif "genes" in doc:
            genes = doc["genes"]
        elif "hex" in doc:
            return self.hex_to_gene(doc["hex"])
        else:
            raise ValueError(
                "from_json: could not find 'genes' or 'hex' key in JSON."
            )
        if len(genes) != GENOME_LENGTH:
            raise ValueError(
                f"from_json: expected {GENOME_LENGTH} genes, got {len(genes)}."
            )
        return np.array(genes, dtype=np.int32)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_gene(self, gene_vec: np.ndarray) -> list[str]:
        """Return a list of constraint-violation strings for a gene vector.

        An empty list means the vector is valid.

        Parameters
        ----------
        gene_vec:
            Integer array to check.

        Returns
        -------
        list[str]
            Human-readable error messages (empty = valid).
        """
        errors: list[str] = []
        v = np.asarray(gene_vec, dtype=np.int32)

        if v.shape != (GENOME_LENGTH,):
            errors.append(
                f"Wrong gene vector length: expected {GENOME_LENGTH}, "
                f"got {v.shape}."
            )
            return errors  # cannot continue checking

        lo = self._ss.lower_bounds
        hi = self._ss.upper_bounds
        oor = np.where((v < lo) | (v > hi))[0]
        for i in oor:
            errors.append(
                f"Gene[{i}] = {v[i]} is out of range [{lo[i]}, {hi[i]}]."
            )

        # Semantic: n_stages consistency
        n = int(v[3 - 1])  # G_N_STAGES == 2
        from src.nas.search_space import G_N_STAGES
        n = int(v[G_N_STAGES])
        if not (MIN_STAGES <= n <= MAX_STAGES):
            errors.append(
                f"n_stages = {n} is outside [{MIN_STAGES}, {MAX_STAGES}]."
            )

        return errors

    def validate_candidate(self, candidate_dict: dict[str, Any]) -> list[str]:
        """Validate a candidate dict against model_factory's requirements.

        Parameters
        ----------
        candidate_dict:
            Two-key dict ``{"arch_spec": ..., "quant_spec": ...}`` or flat.

        Returns
        -------
        list[str]
            Error messages (empty = valid).
        """
        if "arch_spec" in candidate_dict:
            arch  = candidate_dict["arch_spec"]
            quant = candidate_dict.get("quant_spec", {})
        else:
            arch  = candidate_dict
            quant = {}
        return validate_arch_spec(arch) + validate_quant_spec(quant)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def describe(self, gene_vec: np.ndarray) -> str:
        """Return a multi-line human-readable description of a candidate."""
        genome = self._ss.decode(gene_vec)
        s      = genome.summary()
        cplx   = estimate_complexity(self.gene_to_arch_spec(gene_vec))
        lines  = [
            f"Fingerprint : {self.fingerprint(gene_vec)}",
            f"Family      : {s['arch_family']}",
            f"Input size  : {s['input_size']}",
            f"Stages      : {s['n_stages']}",
            f"Channels    : {s['channels']}",
            f"Kernels     : {s['kernel_sizes']}",
            f"Blocks      : {s['block_types']}",
            f"Bottlenecks : {s['bottleneck_ratios']}",
            f"Skip        : {s['skip_connections']}",
            f"Attention   : {s['attentions']}",
            f"W_bits (g)  : {s['global_weight_bits']}",
            f"A_bits (g)  : {s['global_act_bits']}",
            f"Symmetric   : {s['symmetric']}",
            f"Per-channel : {s['per_channel']}",
            f"Mixed-prec  : {s['mixed_precision']}",
        ]
        if s["layer_weight_bits"]:
            lines.append(f"W_bits/layer: {s['layer_weight_bits']}")
            lines.append(f"A_bits/layer: {s['layer_act_bits']}")
        lines += [
            f"~Params     : {cplx['n_params'] / 1e6:.3f} M",
            f"~MACs       : {cplx['macs'] / 1e6:.3f} M",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone validation
# ---------------------------------------------------------------------------

def validate_arch_spec(arch_spec: dict[str, Any]) -> list[str]:
    """Validate an ``arch_spec`` dict against model_factory requirements.

    Parameters
    ----------
    arch_spec:
        Dict with architecture parameters.

    Returns
    -------
    list[str]
        Error messages (empty list = valid).
    """
    errors: list[str] = []
    missing = _ARCH_REQUIRED - set(arch_spec.keys())
    if missing:
        errors.append(f"arch_spec missing required keys: {sorted(missing)}.")

    family = arch_spec.get("family")
    if family is not None and str(family) not in _VALID_FAMILIES:
        errors.append(
            f"arch_spec.family={family!r} is not one of {sorted(_VALID_FAMILIES)}."
        )

    depth = arch_spec.get("depth")
    if depth is not None:
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            errors.append(f"arch_spec.depth must be an integer; got {depth!r}.")
            depth = None

    if depth is not None:
        channels = arch_spec.get("channels")
        if isinstance(channels, (list, tuple)) and len(channels) != depth:
            errors.append(
                f"arch_spec.channels has length {len(channels)} but depth={depth}."
            )

        for list_key in ("kernel_size", "block_type",
                         "bottleneck_ratio", "skip_connections", "attention"):
            val = arch_spec.get(list_key)
            if isinstance(val, (list, tuple)) and len(val) != depth:
                errors.append(
                    f"arch_spec.{list_key} has length {len(val)} but depth={depth}."
                )

    block_type = arch_spec.get("block_type")
    if block_type is not None:
        bts = [block_type] if isinstance(block_type, str) else block_type
        invalid = [b for b in bts if str(b) not in _VALID_BLOCKS]
        if invalid:
            errors.append(
                f"arch_spec.block_type has invalid values: {invalid}. "
                f"Valid: {sorted(_VALID_BLOCKS)}."
            )

    attention = arch_spec.get("attention")
    if attention is not None:
        atns = [attention] if isinstance(attention, str) else attention
        invalid = [a for a in atns if str(a) not in _VALID_ATTENTIONS]
        if invalid:
            errors.append(
                f"arch_spec.attention has invalid values: {invalid}. "
                f"Valid: {sorted(_VALID_ATTENTIONS)}."
            )

    input_size = arch_spec.get("input_size")
    if input_size is not None:
        try:
            sz = int(input_size)
            if sz < 32 or sz > 1024:
                errors.append(
                    f"arch_spec.input_size={sz} is outside the practical range [32, 1024]."
                )
        except (TypeError, ValueError):
            errors.append(f"arch_spec.input_size must be an integer; got {input_size!r}.")

    return errors


def validate_quant_spec(quant_spec: dict[str, Any]) -> list[str]:
    """Validate a ``quant_spec`` dict against qat_wrapper requirements.

    Parameters
    ----------
    quant_spec:
        Dict with quantisation parameters.

    Returns
    -------
    list[str]
        Error messages (empty list = valid).
    """
    errors: list[str] = []
    missing = _QUANT_REQUIRED - set(quant_spec.keys())
    if missing:
        errors.append(f"quant_spec missing required keys: {sorted(missing)}.")

    for bits_key in ("weight_bits", "act_bits"):
        val = quant_spec.get(bits_key)
        if val is None:
            continue
        vals = val if isinstance(val, (list, tuple)) else [val]
        for b in vals:
            try:
                bi = int(b)
                if not (2 <= bi <= 32):
                    errors.append(
                        f"quant_spec.{bits_key} value {bi} is outside [2, 32]."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"quant_spec.{bits_key} contains non-integer value: {b!r}."
                )

    mixed = quant_spec.get("mixed_precision", False)
    if mixed:
        for lkey in ("layer_weight_bits", "layer_act_bits"):
            lval = quant_spec.get(lkey)
            if lval is not None and not isinstance(lval, (list, tuple)):
                errors.append(
                    f"quant_spec.{lkey} should be a list when mixed_precision=True."
                )

    return errors


# ---------------------------------------------------------------------------
# Default completion
# ---------------------------------------------------------------------------

def complete_arch_defaults(arch_spec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``arch_spec`` with missing optional fields filled in.

    Does not modify the original dict.

    Parameters
    ----------
    arch_spec:
        Possibly partial architecture specification.

    Returns
    -------
    dict
        Complete arch_spec ready for ``model_factory.build_model``.
    """
    completed = dict(arch_spec)
    depth = int(completed.get("depth", 3))
    for key, default in _ARCH_DEFAULTS.items():
        if key not in completed:
            completed[key] = [default] * depth
    return completed


def complete_quant_defaults(quant_spec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``quant_spec`` with missing optional fields filled in.

    Parameters
    ----------
    quant_spec:
        Possibly partial quantisation specification.

    Returns
    -------
    dict
        Complete quant_spec ready for ``qat_wrapper.wrap_for_qat``.
    """
    completed = dict(quant_spec)
    for key, default in _QUANT_DEFAULTS.items():
        if key not in completed:
            completed[key] = default
    # Propagate global bits to layer lists when mixed_precision and lists absent
    if completed.get("mixed_precision"):
        if completed.get("layer_weight_bits") is None:
            wb = completed.get("weight_bits", 8)
            completed["layer_weight_bits"] = wb if isinstance(wb, list) else None
        if completed.get("layer_act_bits") is None:
            ab = completed.get("act_bits", 8)
            completed["layer_act_bits"] = ab if isinstance(ab, list) else None
    return completed


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

def estimate_complexity(arch_spec: dict[str, Any]) -> dict[str, float]:
    """Estimate parameter count and MACs without building a PyTorch model.

    Uses the ``arch_spec`` dict to compute an analytical approximation
    based on the dominant convolution layers.  Decoder/upsampling paths
    are approximated as a mirror of the encoder.

    Parameters
    ----------
    arch_spec:
        Architecture specification dict.

    Returns
    -------
    dict
        Keys: ``n_params`` (int), ``macs`` (float), ``n_params_m`` (float,
        params in millions), ``macs_m`` (float, MACs in millions).
    """
    depth   = int(arch_spec.get("depth", 3))
    in_size = int(arch_spec.get("input_size", 224))
    family  = str(arch_spec.get("family", "autoencoder"))

    channels_raw = arch_spec.get("channels", 32)
    if isinstance(channels_raw, (list, tuple)):
        channels = [int(c) for c in channels_raw][:depth]
        if len(channels) < depth:
            channels += [channels[-1]] * (depth - len(channels))
    else:
        # Doubling schedule
        base = int(channels_raw)
        channels = [base * (2 ** i) for i in range(depth)]

    kernel_raw = arch_spec.get("kernel_size", 3)
    if isinstance(kernel_raw, (list, tuple)):
        kernels = [int(k) for k in kernel_raw][:depth]
        kernels += [kernels[-1]] * (depth - len(kernels))
    else:
        kernels = [int(kernel_raw)] * depth

    block_raw = arch_spec.get("block_type", "conv")
    if isinstance(block_raw, (list, tuple)):
        blocks = [str(b) for b in block_raw][:depth]
        blocks += [blocks[-1]] * (depth - len(blocks))
    else:
        blocks = [str(block_raw)] * depth

    bn_raw = arch_spec.get("bottleneck_ratio", 1.0)
    if isinstance(bn_raw, (list, tuple)):
        bottlenecks = [float(b) for b in bn_raw][:depth]
        bottlenecks += [bottlenecks[-1]] * (depth - len(bottlenecks))
    else:
        bottlenecks = [float(bn_raw)] * depth

    total_params = 0
    total_macs   = 0.0

    in_ch  = 3
    H = W  = in_size

    for i, (out_ch, ks, block, bn) in enumerate(
            zip(channels, kernels, blocks, bottlenecks)):
        mid_ch = max(1, int(out_ch * bn))
        p, m   = _stage_complexity(in_ch, out_ch, mid_ch, ks, block, H, W)
        total_params += p
        total_macs   += m
        in_ch = out_ch
        H = max(1, H // 2)
        W = max(1, W // 2)

    # Decoder approximation (for autoencoder / unet families)
    if family in ("autoencoder", "unet"):
        dec_channels = list(reversed(channels[:-1])) + [3]
        dec_blocks   = list(reversed(blocks))
        dec_kernels  = list(reversed(kernels))
        dec_bottlenecks = list(reversed(bottlenecks))
        for out_ch, ks, block, bn in zip(
                dec_channels, dec_blocks, dec_kernels, dec_bottlenecks):
            mid_ch = max(1, int(out_ch * bn))
            p, m   = _stage_complexity(in_ch, out_ch, mid_ch, ks, block, H, W)
            total_params += p
            total_macs   += m
            in_ch = out_ch
            H = min(in_size, H * 2)
            W = min(in_size, W * 2)

    return {
        "n_params":   total_params,
        "macs":       total_macs,
        "n_params_m": total_params / 1e6,
        "macs_m":     total_macs / 1e6,
    }


def _stage_complexity(in_ch: int, out_ch: int, mid_ch: int,
                      ks: int, block: str,
                      H: int, W: int) -> tuple[int, float]:
    """Analytical parameter and MAC count for a single encoder stage."""
    if block == "conv":
        # Standard conv: in → out, stride 2
        params = out_ch * in_ch * ks * ks + out_ch          # weight + bias
        macs   = float(out_ch * in_ch * ks * ks * H * W)    # pre-stride spatial
    elif block == "ds":
        # Depthwise separable: DW (in→in) + PW (in→out)
        dw_p   = in_ch * ks * ks + in_ch
        pw_p   = out_ch * in_ch + out_ch
        params = dw_p + pw_p
        dw_m   = float(in_ch * ks * ks * H * W)
        pw_m   = float(out_ch * in_ch * H * W)
        macs   = dw_m + pw_m
    elif block == "ir":
        # Inverted residual: PW expand (in→mid) + DW (mid→mid) + PW project (mid→out)
        pw1_p  = mid_ch * in_ch + mid_ch
        dw_p   = mid_ch * ks * ks + mid_ch
        pw2_p  = out_ch * mid_ch + out_ch
        params = pw1_p + dw_p + pw2_p
        pw1_m  = float(mid_ch * in_ch * H * W)
        dw_m   = float(mid_ch * ks * ks * H * W)
        pw2_m  = float(out_ch * mid_ch * H * W)
        macs   = pw1_m + dw_m + pw2_m
    else:
        # Fallback: treat as standard conv
        params = out_ch * in_ch * ks * ks + out_ch
        macs   = float(out_ch * in_ch * ks * ks * H * W)

    # BN parameters
    params += out_ch * 2

    return int(params), macs


# ---------------------------------------------------------------------------
# JSON utility
# ---------------------------------------------------------------------------

def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
