"""
src/nas/space.py
================

Generic discrete search space for the NSGA-II engine, plus the three spaces of
the hierarchical methodology:

- **Level 1 (architecture)** — ``arch_space``: one gene per architecture
  hyperparameter. Candidates are trained in floating point.
- **Level 2 (numeric representation + quantization)** — ``quant_space``: for a
  *fixed* architecture, the representation (``int`` / ``pot`` / ``fp16``),
  symmetry, per-channel flag and per-segment weight / activation bits.
- **Flat (ablation baseline)** — ``flat_space``: Level-1 and Level-2 genes in a
  single genome, as in a non-hierarchical joint search.

A genome is an integer vector; gene ``i`` indexes the choice list of gene
name ``i``. ``to_candidate`` maps the decoded ``{name: value}`` dict to the
candidate dict consumed by the fitness evaluator::

    {"arch_spec": {...}, "quant_spec": {...}, "init_state": "path" | absent}

``arch_spec`` keys are those of ``src.models.model_factory.ArchSpec``;
``quant_spec`` follows the search schema of ``src.quantization.qat_wrapper``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np

__all__ = ["DiscreteSpace", "arch_space", "quant_space", "flat_space",
           "quant_genes_to_spec"]


class DiscreteSpace:
    """Named categorical genes with NSGA-II variation operators."""

    def __init__(self, genes: dict[str, list],
                 to_candidate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if not genes or any(len(c) == 0 for c in genes.values()):
            raise ValueError("every gene needs at least one choice")
        self.names = list(genes)
        self.choices = [list(genes[n]) for n in self.names]
        self._hi = np.array([len(c) - 1 for c in self.choices], dtype=np.int32)
        self._to_candidate = to_candidate

    @property
    def genome_length(self) -> int:
        return len(self.names)

    # ---- encoding ----------------------------------------------------------
    def decode(self, vec: np.ndarray) -> dict[str, Any]:
        return {n: c[int(i)] for n, c, i in zip(self.names, self.choices, vec)}

    def encode(self, values: dict[str, Any]) -> np.ndarray:
        return np.array([c.index(values[n]) for n, c in zip(self.names, self.choices)],
                        dtype=np.int32)

    def genome_to_candidate(self, vec: np.ndarray) -> dict[str, Any]:
        return self._to_candidate(self.decode(vec))

    def fingerprint(self, vec: np.ndarray) -> str:
        """Hash of the *candidate*: genomes that decode to the same model (e.g.
        any bits under fp16) share one evaluation in the engine cache."""
        cand = json.dumps(self.genome_to_candidate(vec), sort_keys=True)
        return hashlib.sha1(cand.encode()).hexdigest()[:10]

    # ---- sampling / variation ---------------------------------------------
    def sample_population(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self._hi + 1, size=(size, self.genome_length),
                            dtype=np.int32)

    def crossover_single_point(self, a, b, rng):
        if self.genome_length < 2:
            return a.copy(), b.copy()
        k = int(rng.integers(1, self.genome_length))
        return (np.concatenate([a[:k], b[k:]]), np.concatenate([b[:k], a[k:]]))

    def crossover_uniform(self, a, b, rng):
        mask = rng.random(self.genome_length) < 0.5
        return np.where(mask, a, b), np.where(mask, b, a)

    def mutate(self, vec, rng, rate: float):
        out = vec.copy()
        hit = rng.random(self.genome_length) < rate
        out[hit] = rng.integers(0, self._hi[hit] + 1)
        return out

    def mutate_neighbourhood(self, vec, rng, rate: float):
        out = vec.copy()
        hit = rng.random(self.genome_length) < rate
        out[hit] += rng.choice([-1, 1], size=int(hit.sum()))
        return np.clip(out, 0, self._hi)


# ---------------------------------------------------------------------------
# Level spaces
# ---------------------------------------------------------------------------

def _seg_names(n_segments: int) -> list[str]:
    return ([f"w_bits_{s}" for s in range(n_segments)]
            + [f"a_bits_{s}" for s in range(n_segments)])


def quant_genes_to_spec(d: dict[str, Any], n_segments: int) -> dict[str, Any]:
    """Decoded Level-2 genes -> ``quant_spec`` (search schema)."""
    if d["representation"] == "fp16":   # bits are meaningless for fp16
        return {"representation": "fp16", "weight_bits": [16] * n_segments,
                "act_bits": [16] * n_segments, "symmetric": True,
                "per_channel": False}
    return {
        "representation": d["representation"],
        "symmetric": d["symmetric"],
        "per_channel": d["per_channel"],
        "weight_bits": [d[f"w_bits_{s}"] for s in range(n_segments)],
        "act_bits": [d[f"a_bits_{s}"] for s in range(n_segments)],
    }


def arch_space(arch_choices: dict[str, list], float_representation: str = "fp16"
               ) -> DiscreteSpace:
    """Level 1: architecture genes; candidates are trained unquantized."""
    return DiscreteSpace(
        arch_choices,
        lambda d: {"arch_spec": dict(d),
                   "quant_spec": {"representation": float_representation}},
    )


def quant_space(arch_spec: dict[str, Any], quant_choices: dict[str, list],
                n_segments: int, init_state: str | None = None) -> DiscreteSpace:
    """Level 2: representation + per-segment bits for a fixed architecture."""
    genes = {"representation": quant_choices["representations"],
             "symmetric": quant_choices["symmetric"],
             "per_channel": quant_choices["per_channel"]}
    for name in _seg_names(n_segments):
        genes[name] = quant_choices["weight_bits" if name[0] == "w" else "act_bits"]

    def to_candidate(d: dict[str, Any]) -> dict[str, Any]:
        cand = {"arch_spec": dict(arch_spec),
                "quant_spec": quant_genes_to_spec(d, n_segments)}
        if init_state:
            cand["init_state"] = init_state
        return cand

    return DiscreteSpace(genes, to_candidate)


def flat_space(arch_choices: dict[str, list], quant_choices: dict[str, list],
               n_segments: int) -> DiscreteSpace:
    """Ablation: Level-1 and Level-2 genes in one genome (non-hierarchical)."""
    q = quant_space({}, quant_choices, n_segments)
    genes = dict(arch_choices)
    genes.update(dict(zip(q.names, q.choices)))

    def to_candidate(d: dict[str, Any]) -> dict[str, Any]:
        return {"arch_spec": {k: d[k] for k in arch_choices},
                "quant_spec": quant_genes_to_spec(d, n_segments)}

    return DiscreteSpace(genes, to_candidate)
