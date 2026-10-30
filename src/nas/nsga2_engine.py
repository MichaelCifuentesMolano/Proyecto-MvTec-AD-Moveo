"""
src/nas/nsga2_engine.py
=======================

Task-agnostic NSGA-II engine used at every level of the hierarchical search.

Algorithm reference
-------------------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A fast and elitist multiobjective genetic algorithm: NSGA-II."
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Objectives
----------
``NSGA2Config.objectives`` names the keys read from the fitness dict, and
``NSGA2Config.maximize`` the subset that is maximised (negated internally);
everything is minimised by the engine. A fitness function that raises or
returns ``None`` gets ``PENALTY`` on every minimised objective (``0`` on
maximised ones), which keeps it on the last front without NaNs in the
crowding distance.

Public interface
----------------
``NSGA2Engine(config, space, fitness_fn)`` where ``space`` provides
``sample_population``, ``crossover_single_point`` / ``crossover_uniform``,
``mutate`` / ``mutate_neighbourhood``, ``genome_length``,
``genome_to_candidate`` and ``fingerprint`` (see ``src.nas.space``).
Identical genomes are evaluated once (results are cached by fingerprint).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "NSGA2Config",
    "GenerationResult",
    "NSGA2Engine",
    "fast_non_dominated_sort",
    "crowding_distance",
    "PENALTY",
]

LOG = logging.getLogger(__name__)

PENALTY: float = 1e12


# ---------------------------------------------------------------------------
# Configuration / results
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Config:
    """Hyperparameters for the NSGA-II engine (YAML-ready)."""
    population_size:        int   = 16
    objectives:             list[str] = field(
        default_factory=lambda: ["performance", "macs_m"])
    maximize:               list[str] = field(
        default_factory=lambda: ["performance"])
    crossover_prob:         float = 0.90
    crossover_type:         str   = "uniform"        # "single_point" | "uniform"
    mutation_rate:          float | None = None      # None -> 1 / genome_length
    neighbourhood_mutation: bool  = False
    tournament_size:        int   = 2
    seed:                   int   = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    """Snapshot of one completed generation."""
    generation:      int
    n_pareto:        int
    best:            dict[str, float]   # best raw value per objective on the front
    n_evaluated:     int                # new (non-cached) evaluations
    n_failed_evals:  int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Core algorithmic primitives (pure functions, testable independently)
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return ``True`` if objective vector *a* strictly dominates *b*.

    Dominance (minimisation): *a* dominates *b* iff
    ``a[i] ≤ b[i]`` for all i and ``a[i] < b[i]`` for at least one i.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002, Algorithm 1).

    Returns the ordered list of fronts; ``fronts[0]`` is the Pareto front and
    each element is an index into *objectives* (``[N, M]``, minimised).
    """
    n = len(objectives)
    dom_count   = np.zeros(n, dtype=np.int32)
    dom_set:    list[list[int]] = [[] for _ in range(n)]
    fronts:     list[list[int]] = [[]]

    for i in range(n):
        obj_i = objectives[i]
        for j in range(i + 1, n):
            obj_j = objectives[j]
            if _dom_fast(obj_i, obj_j):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif _dom_fast(obj_j, obj_i):
                dom_set[j].append(i)
                dom_count[i] += 1

    for i in range(n):
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        k += 1

    return [f for f in fronts if f]


def _dom_fast(a: np.ndarray, b: np.ndarray) -> bool:
    """Inlined dominance check — avoids Python overhead for the inner loop."""
    lt_one = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            lt_one = True
    return lt_one


def crowding_distance(objectives: np.ndarray,
                      front: Sequence[int]) -> dict[int, float]:
    """Crowding distance of the individuals on one front (boundaries = inf)."""
    k = len(front)
    distances: dict[int, float] = {idx: 0.0 for idx in front}

    if k <= 2:
        for idx in front:
            distances[idx] = math.inf
        return distances

    front_list = list(front)
    obj_sub    = objectives[front_list]          # [k, M]
    n_obj      = obj_sub.shape[1]

    for m in range(n_obj):
        col       = obj_sub[:, m]
        sort_idx  = np.argsort(col, kind="stable")
        obj_range = float(col[sort_idx[-1]] - col[sort_idx[0]])

        distances[front_list[sort_idx[0]]]  = math.inf
        distances[front_list[sort_idx[-1]]] = math.inf

        if obj_range < 1e-12:
            continue

        for r in range(1, k - 1):
            i_prev = front_list[sort_idx[r - 1]]
            i_next = front_list[sort_idx[r + 1]]
            i_curr = front_list[sort_idx[r]]
            delta  = (objectives[i_next, m] - objectives[i_prev, m]) / obj_range
            if distances[i_curr] != math.inf:
                distances[i_curr] += delta

    return distances


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NSGA2Engine:
    """NSGA-II multi-objective evolutionary optimiser (μ + λ, elitist)."""

    def __init__(self, config: NSGA2Config, space: Any,
                 fitness_fn: Callable[[dict[str, Any]], dict[str, Any] | None]
                 ) -> None:
        unknown = set(config.maximize) - set(config.objectives)
        if unknown:
            raise ValueError(f"maximize keys not in objectives: {sorted(unknown)}")
        self._cfg = config
        self._ss = space
        self._fit_fn = fitness_fn
        self._rng = np.random.default_rng(config.seed)
        self._mut_rate = (config.mutation_rate if config.mutation_rate is not None
                          else 1.0 / space.genome_length)
        self._signs = np.array([-1.0 if k in config.maximize else 1.0
                                for k in config.objectives])
        self._penalty = np.where(self._signs < 0, 0.0, PENALTY)

        self._population: np.ndarray | None = None
        self._objectives: np.ndarray | None = None
        self._generation = 0
        self._history: list[GenerationResult] = []
        self._cache: dict[str, np.ndarray] = {}
        # Every new evaluation, in order (written to evaluations.csv).
        self.evaluations: list[dict[str, Any]] = []
        self._n_failed = 0
        self._n_new = 0

    # ---- main loop ------------------------------------------------------------
    def initialize(self, population: np.ndarray | None = None) -> None:
        """Evaluate generation 0: ``population`` rows first, random fill after."""
        n = self._cfg.population_size
        pop = self._ss.sample_population(n, self._rng)
        if population is not None:
            seed = np.asarray(population, dtype=np.int32)[:n]
            pop[:len(seed)] = seed
        self._population = pop
        self._objectives = self._evaluate_population(pop)
        self._generation = 0

    def run(self, n_generations: int,
            on_generation_end: Callable[[GenerationResult, "NSGA2Engine"], None]
            | None = None) -> None:
        if self._population is None:
            self.initialize()
        for _ in range(n_generations):
            result = self.step()
            if on_generation_end is not None:
                on_generation_end(result, self)

    def step(self) -> GenerationResult:
        if self._population is None:
            raise RuntimeError("Call initialize() first.")
        t0 = time.perf_counter()
        self._n_failed = self._n_new = 0
        self._generation += 1

        offspring = self._create_offspring()
        off_obj = self._evaluate_population(offspring)
        self._population, self._objectives = self._select_next_generation(
            np.vstack([self._population, offspring]),
            np.vstack([self._objectives, off_obj]),
            self._cfg.population_size,
        )

        front = self.pareto_indices()
        raw = self.raw_objectives[front]
        best = ({k: float(raw[:, j].max() if self._signs[j] < 0 else raw[:, j].min())
                 for j, k in enumerate(self._cfg.objectives)} if front else {})
        result = GenerationResult(
            generation=self._generation, n_pareto=len(front), best=best,
            n_evaluated=self._n_new, n_failed_evals=self._n_failed,
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )
        self._history.append(result)
        LOG.info("Gen %3d | Pareto=%2d | new evals=%2d | failed=%d | best=%s | %.1fs",
                 self._generation, len(front), self._n_new, self._n_failed,
                 {k: round(v, 4) for k, v in best.items()}, result.elapsed_seconds)
        return result

    # ---- variation -------------------------------------------------------------
    def _create_offspring(self) -> np.ndarray:
        n = self._cfg.population_size
        ranks, dists = self._assign_ranks_and_distances(self._objectives)
        mutate = (self._ss.mutate_neighbourhood if self._cfg.neighbourhood_mutation
                  else self._ss.mutate)
        offspring: list[np.ndarray] = []
        while len(offspring) < n:
            p_a = self._population[self._tournament_select(ranks, dists)]
            p_b = self._population[self._tournament_select(ranks, dists)]
            if self._rng.random() < self._cfg.crossover_prob:
                if self._cfg.crossover_type == "uniform":
                    c_a, c_b = self._ss.crossover_uniform(p_a, p_b, self._rng)
                else:
                    c_a, c_b = self._ss.crossover_single_point(p_a, p_b, self._rng)
            else:
                c_a, c_b = p_a.copy(), p_b.copy()
            offspring.append(mutate(c_a, self._rng, self._mut_rate))
            if len(offspring) < n:
                offspring.append(mutate(c_b, self._rng, self._mut_rate))
        return np.vstack(offspring)

    def _assign_ranks_and_distances(self, objectives: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]:
        ranks = np.zeros(len(objectives), dtype=np.int32)
        dists = np.zeros(len(objectives), dtype=np.float64)
        for rank, front in enumerate(fast_non_dominated_sort(objectives)):
            ranks[front] = rank
            for idx, d in crowding_distance(objectives, front).items():
                dists[idx] = d
        return ranks, dists

    def _tournament_select(self, ranks: np.ndarray, distances: np.ndarray) -> int:
        cands = self._rng.integers(0, len(ranks), size=self._cfg.tournament_size)
        best = int(cands[0])
        for c in map(int, cands[1:]):
            if ranks[c] < ranks[best] or (ranks[c] == ranks[best]
                                          and distances[c] > distances[best]):
                best = c
        return best

    def _select_next_generation(self, pop: np.ndarray, obj: np.ndarray,
                                n_select: int) -> tuple[np.ndarray, np.ndarray]:
        # Drop duplicate genomes first: cached copies of the same individual
        # would otherwise crowd the front and stall diversity.
        _, first = np.unique(pop, axis=0, return_index=True)
        first = np.sort(first)
        if len(first) >= n_select:
            pop, obj = pop[first], obj[first]
        selected: list[int] = []
        for front in fast_non_dominated_sort(obj):
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                cd = crowding_distance(obj, front)
                selected.extend(sorted(front, key=lambda i: cd[i], reverse=True)
                                [:n_select - len(selected)])
            if len(selected) >= n_select:
                break
        sel = np.asarray(selected, dtype=np.int32)
        return pop[sel].copy(), obj[sel].copy()

    # ---- evaluation ------------------------------------------------------------
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.vstack([self._evaluate_one(g) for g in population])

    def _evaluate_one(self, gene_vec: np.ndarray) -> np.ndarray:
        fp = self._ss.fingerprint(gene_vec)
        if fp in self._cache:
            return self._cache[fp]
        self._n_new += 1
        candidate = self._ss.genome_to_candidate(gene_vec)
        try:
            raw = self._fit_fn(candidate)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Evaluation failed [%s]: %s", fp, exc)
            raw = None
        valid = raw is not None and all(raw.get(k) is not None
                                        for k in self._cfg.objectives)
        if valid:
            obj = np.array([float(raw[k]) for k in self._cfg.objectives]) * self._signs
        else:
            self._n_failed += 1
            obj = self._penalty.copy()
        self._cache[fp] = obj
        self.evaluations.append({"generation": self._generation, "fingerprint": fp,
                                 "genome": [int(g) for g in gene_vec],
                                 "candidate": candidate, "metrics": raw or {},
                                 "valid": valid})
        return obj

    # ---- accessors ---------------------------------------------------------------
    @property
    def population(self) -> np.ndarray:
        return self._population

    @property
    def objectives(self) -> np.ndarray:
        """Minimisation-form objective matrix of the current population."""
        return self._objectives

    @property
    def raw_objectives(self) -> np.ndarray:
        """Objective matrix in natural units (maximised keys un-negated)."""
        return self._objectives * self._signs

    @property
    def history(self) -> list[GenerationResult]:
        return self._history

    @property
    def generation(self) -> int:
        return self._generation

    def is_valid(self, i: int) -> bool:
        return not np.array_equal(self._objectives[i], self._penalty)

    def pareto_indices(self) -> list[int]:
        """Rank-0 indices among valid individuals of the current population."""
        valid = [i for i in range(len(self._population)) if self.is_valid(i)]
        if not valid:
            return []
        front = fast_non_dominated_sort(self._objectives[valid])[0]
        return [valid[i] for i in front]
