"""
src/nas/nsga2_engine.py
=======================

NSGA-II multi-objective evolutionary engine for joint architecture +
quantisation search.

Algorithm reference
-------------------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A fast and elitist multiobjective genetic algorithm: NSGA-II."
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

Four objectives (all *minimised* internally)
--------------------------------------------
1.  ``-AUROC``      — negated so maximisation becomes minimisation.
2.  ``latency_ms``  — inference latency in milliseconds.
3.  ``peak_ram_mb`` — peak RAM in megabytes.
4.  ``energy_mj``   — energy per inference in millijoules.

The fitness function injected by the caller returns raw measurements
(AUROC ∈ [0,1], latency > 0, …).  The engine applies the sign flip and
handles the reference-point normalisation internally.

Public interface
----------------
``NSGA2Config``
    Hyperparameter dataclass (YAML-serialisable).

``GenerationResult``
    Per-generation snapshot returned by :meth:`NSGA2Engine.step` and
    accumulated in :meth:`NSGA2Engine.run`.

``NSGA2Engine``
    Main class.  Inject a ``SearchSpace`` and a fitness callable, then
    call :meth:`run`.

Assumptions
-----------
- The fitness callable has the signature
  ``fitness_fn(candidate_dict: dict) -> dict[str, float]``
  and returns at least the keys ``{"auroc", "latency_ms",
  "peak_ram_mb", "energy_mj"}``.  Missing keys are replaced by the
  configured penalty values.
- ``SearchSpace`` exposes ``sample_population``, ``crossover_*``,
  ``mutate``, ``clip``, and ``genome_length``.
- Population sizes of 30–200 individuals are the intended operating range.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.nas.search_space import GENOME_LENGTH, SearchSpace
from src.nas.encoding import GenomeEncoder

import hashlib
from src.nas import search_space as _ss_mod


def search_space_hash() -> str:
    """Deterministic fingerprint of the ACTIVE search-space definition.

    Genes are indices into option lists, so two checkpoints are only
    compatible if every list (and the genome geometry) is identical. Any
    edit to the lists silently re-interprets old genomes as different
    architectures — this hash makes that incompatibility detectable.
    """
    sig = repr((
        _ss_mod.ARCH_FAMILIES, _ss_mod.ARCH_INPUT_SIZES,
        _ss_mod.ARCH_CHANNELS, _ss_mod.ARCH_KERNELS,
        _ss_mod.ARCH_BLOCKS, _ss_mod.ARCH_BOTTLENECKS,
        _ss_mod.ARCH_ATTENTIONS, _ss_mod.QUANT_BITS,
        _ss_mod.MIN_STAGES, _ss_mod.MAX_STAGES, GENOME_LENGTH,
    ))
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "NSGA2Config",
    "GenerationResult",
    "NSGA2Engine",
    "fast_non_dominated_sort",
    "crowding_distance",
    "hypervolume_2d",
    "hypervolume_mc",
]

LOG = logging.getLogger(__name__)

# Objective index → name (display) mapping
OBJECTIVE_NAMES: list[str] = ["-AUROC", "Latency (ms)", "RAM (MB)", "Energy (mJ)"]
FITNESS_KEYS:    list[str] = ["auroc", "latency_ms", "peak_ram_mb", "energy_mj"]
# Sign applied to raw fitness to convert to minimisation objective.
# AUROC is maximised, so it is negated; the rest are already minimised.
OBJECTIVE_SIGNS: np.ndarray = np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Config:
    """Hyperparameters for the NSGA-II engine.

    All fields are YAML-ready (plain Python scalars / lists).
    """
    # Population
    population_size:        int   = 50
    n_objectives:           int   = 4

    # Genetic operators
    crossover_prob:         float = 0.90
    crossover_type:         str   = "single_point"  # "single_point" | "uniform"
    mutation_rate:          float | None = None       # None → 1/GENOME_LENGTH
    neighbourhood_mutation: bool  = False  # True → ±1 step instead of random reset
    tournament_size:        int   = 2

    # Hypervolume reference point (for each minimised objective in order)
    # [-AUROC reference=0.0, latency_ms ref, peak_ram_mb ref, energy_mj ref]
    hv_reference:           list[float] = field(
        default_factory=lambda: [0.0, 5000.0, 16384.0, 1000.0]
    )
    hv_n_samples:           int   = 50_000   # Monte Carlo samples

    # Penalty vector used when evaluation fails (same order as hv_reference)
    penalty_objectives:     list[float] = field(
        default_factory=lambda: [0.0, 5000.0, 16384.0, 1000.0]
    )

    # Reproducibility
    seed:                   int   = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def effective_mutation_rate(self) -> float:
        return self.mutation_rate if self.mutation_rate is not None else 1.0 / GENOME_LENGTH


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Snapshot of one completed generation."""
    generation:           int
    n_pareto:             int
    hypervolume:          float
    best_auroc:           float           # highest AUROC on Pareto front
    best_latency_ms:      float           # lowest latency on Pareto front
    best_ram_mb:          float
    best_energy_mj:       float
    mean_auroc:           float           # mean over entire population
    elapsed_seconds:      float
    n_failed_evals:       int
    pareto_fingerprints:  list[str]       # for deduplication tracking


# ---------------------------------------------------------------------------
# Core algorithmic primitives (pure functions, testable independently)
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return ``True`` if objective vector *a* strictly dominates *b*.

    Dominance (minimisation): *a* dominates *b* iff
    ``a[i] ≤ b[i]`` for all i and ``a[i] < b[i]`` for at least one i.

    Parameters
    ----------
    a, b:
        1-D objective vectors of equal length.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002, Algorithm 1).

    Parameters
    ----------
    objectives:
        2-D array of shape ``[N, M]`` with minimisation objective values.

    Returns
    -------
    list[list[int]]
        Ordered list of fronts.  ``fronts[0]`` is the Pareto front;
        each element is an index into *objectives*.
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
    le_all = True
    lt_one = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            lt_one = True
    return lt_one


def crowding_distance(objectives: np.ndarray,
                      front: Sequence[int]) -> dict[int, float]:
    """Compute crowding distance for individuals on a single front.

    Parameters
    ----------
    objectives:
        Full objective matrix ``[N, M]``.
    front:
        Indices of individuals on the front.

    Returns
    -------
    dict[int, float]
        Mapping from individual index to crowding distance.
        Boundary individuals receive ``+inf``.
    """
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

        # Boundary points
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


def hypervolume_2d(objectives: np.ndarray,
                   reference: Sequence[float]) -> float:
    """Exact 2-D hypervolume by sweep (minimisation).

    Useful for quick monitoring plots on the two most important objectives.

    Parameters
    ----------
    objectives:
        Array of shape ``[N, 2]`` (first two objectives only).
    reference:
        Reference point ``[r0, r1]``; must be weakly dominated by all
        solutions.

    Returns
    -------
    float
        Hypervolume indicator value.
    """
    obj  = np.asarray(objectives[:, :2], dtype=np.float64)
    ref  = np.asarray(reference[:2],     dtype=np.float64)

    # Keep non-dominated points only (sort ascending by obj[0])
    order  = np.argsort(obj[:, 0], kind="stable")
    sorted_obj = obj[order]
    pareto: list[np.ndarray] = []
    min_y  = math.inf
    for p in sorted_obj:
        if p[1] < min_y:
            pareto.append(p)
            min_y = p[1]

    if not pareto:
        return 0.0

    hv = 0.0
    for i, p in enumerate(pareto):
        x_right = pareto[i + 1][0] if i + 1 < len(pareto) else ref[0]
        width   = x_right - p[0]
        height  = ref[1]  - p[1]
        if width > 0 and height > 0:
            hv += width * height
    return float(hv)


def hypervolume_mc(objectives: np.ndarray,
                   reference: Sequence[float],
                   n_samples: int = 50_000,
                   rng: np.random.Generator | None = None) -> float:
    """Monte Carlo hypervolume approximation for arbitrary dimension.

    Estimates the volume of the objective space that is dominated by at
    least one solution in *objectives* and bounded by *reference*.

    Parameters
    ----------
    objectives:
        Array of shape ``[N, M]``.
    reference:
        Reference point of length ``M``; must weakly dominate all rows.
    n_samples:
        Number of Monte Carlo samples.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    float
        Approximated hypervolume.  Returns 0.0 when the bounding box is
        degenerate.
    """
    rng  = rng or np.random.default_rng()
    obj  = np.asarray(objectives, dtype=np.float64)
    if len(obj) == 0:
        return 0.0

    ref  = np.array(reference, dtype=np.float64)
    lb   = obj.min(axis=0)

    # Adjust reference point for any objective where all individuals are worse than or equal to the reference
    max_vals = obj.max(axis=0)
    for i in range(len(ref)):
        if lb[i] >= ref[i]:
            ref[i] = float(max_vals[i] + abs(max_vals[i]) * 0.1 + 1.0)

    if np.any(lb >= ref):
        return 0.0

    box_vol = float(np.prod(ref - lb))
    if box_vol <= 0.0:
        return 0.0

    samples    = rng.uniform(lb, ref, size=(n_samples, obj.shape[1]))
    dominated  = np.zeros(n_samples, dtype=bool)
    for p in obj:
        # Sample is dominated by p if p[i] ≤ sample[i] for all i
        dominated |= np.all(samples >= p, axis=1)

    return float(dominated.mean() * box_vol)


# ---------------------------------------------------------------------------
# NSGA-II Engine
# ---------------------------------------------------------------------------

class NSGA2Engine:
    """NSGA-II multi-objective evolutionary optimiser.

    Parameters
    ----------
    config:
        Hyperparameter configuration.
    search_space:
        Search space that provides sampling, crossover, and mutation.
    fitness_fn:
        Callable ``(candidate_dict) -> dict[str, float]``.  The returned
        dict must contain at least ``{"auroc", "latency_ms",
        "peak_ram_mb", "energy_mj"}``.  Raise any exception or return
        ``None`` / missing keys to signal evaluation failure; the engine
        substitutes the penalty objective vector.

    Examples
    --------
    >>> def my_fitness(c):
    ...     return {"auroc": 0.85, "latency_ms": 12.0,
    ...             "peak_ram_mb": 400.0, "energy_mj": 5.0}
    >>> engine = NSGA2Engine(NSGA2Config(), SearchSpace(), my_fitness)
    >>> result = engine.run(n_generations=10)
    """

    def __init__(self,
                 config: NSGA2Config,
                 search_space: SearchSpace,
                 fitness_fn: Callable[[dict[str, Any]], dict[str, float]]) -> None:
        self._cfg      = config
        self._ss       = search_space
        self._fit_fn   = fitness_fn
        self._encoder  = GenomeEncoder(search_space)
        self._rng      = np.random.default_rng(config.seed)
        self._mut_rate = config.effective_mutation_rate()
        self._ref      = np.asarray(config.hv_reference,       dtype=np.float64)
        self._penalty  = np.asarray(config.penalty_objectives, dtype=np.float64)

        # State (set by initialize or load_checkpoint)
        self._population:  np.ndarray | None = None  # [N, G]
        self._objectives:  np.ndarray | None = None  # [N, M]
        self._generation:  int  = 0
        self._history:     list[GenerationResult] = []
        self._n_failed:    int  = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Sample a fresh random population and evaluate it.

        Must be called before :meth:`step` or :meth:`run` unless a
        checkpoint is loaded.
        """
        LOG.info("NSGA-II: initialising population (N=%d) …",
                 self._cfg.population_size)
        pop = self._ss.sample_population(self._cfg.population_size, self._rng)
        obj = self._evaluate_population(pop)
        self._population = pop
        self._objectives = obj
        self._generation = 0
        LOG.info("NSGA-II: initialisation complete.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self,
            n_generations: int,
            on_generation_end: Callable[[int, GenerationResult,
                                         "NSGA2Engine"], None] | None = None,
            ) -> dict[str, Any]:
        """Run the search for *n_generations* and return a results dict.

        Parameters
        ----------
        n_generations:
            Number of generations to evolve after the initial population.
        on_generation_end:
            Optional callback invoked after each generation with
            ``(generation_index, GenerationResult, engine)``.  Useful for
            logging and checkpointing.

        Returns
        -------
        dict
            Keys: ``"history"``, ``"pareto_population"``,
            ``"pareto_objectives"``, ``"final_generation"``,
            ``"total_elapsed_seconds"``.
        """
        if self._population is None:
            self.initialize()

        t_total = time.perf_counter()

        for _ in range(n_generations):
            result = self.step()
            if on_generation_end is not None:
                on_generation_end(self._generation, result, self)

        elapsed = time.perf_counter() - t_total

        fronts  = fast_non_dominated_sort(self._objectives)
        p_idx   = np.asarray(fronts[0], dtype=np.int32)

        return {
            "history":              self._history,
            "pareto_population":    self._population[p_idx],
            "pareto_objectives":    self._objectives[p_idx],
            "final_generation":     self._generation,
            "total_elapsed_seconds": round(elapsed, 2),
        }

    def step(self) -> GenerationResult:
        """Advance the population by one generation.

        Returns
        -------
        GenerationResult
            Statistics for the completed generation.

        Raises
        ------
        RuntimeError
            If the engine has not been initialised.
        """
        if self._population is None:
            raise RuntimeError("Call initialize() or load_checkpoint() first.")

        t0 = time.perf_counter()
        self._n_failed = 0

        # ---- Create offspring (μ + λ strategy: N offspring) ----
        offspring_pop = self._create_offspring()
        offspring_obj = self._evaluate_population(offspring_pop)

        # ---- Environmental selection on combined pool [2N] ----
        combined_pop = np.vstack([self._population, offspring_pop])
        combined_obj = np.vstack([self._objectives, offspring_obj])
        self._population, self._objectives = self._select_next_generation(
            combined_pop, combined_obj, self._cfg.population_size
        )
        self._generation += 1

        # ---- Pareto front statistics ----
        fronts  = fast_non_dominated_sort(self._objectives)
        p_idx   = fronts[0]
        p_obj   = self._objectives[p_idx]

        # Raw AUROC = -obj[:, 0]
        pareto_auroc = -p_obj[:, 0]
        hv = hypervolume_mc(
            p_obj, self._ref,
            n_samples=self._cfg.hv_n_samples,
            rng=self._rng,
        )

        fingerprints = [self._encoder.fingerprint(self._population[i]) for i in p_idx]

        result = GenerationResult(
            generation          = self._generation,
            n_pareto            = len(p_idx),
            hypervolume         = hv,
            best_auroc          = float(pareto_auroc.max()),
            best_latency_ms     = float(p_obj[:, 1].min()),
            best_ram_mb         = float(p_obj[:, 2].min()),
            best_energy_mj      = float(p_obj[:, 3].min()),
            mean_auroc          = float((-self._objectives[:, 0]).mean()),
            elapsed_seconds     = round(time.perf_counter() - t0, 2),
            n_failed_evals      = self._n_failed,
            pareto_fingerprints = fingerprints,
        )
        self._history.append(result)

        LOG.info(
            "Gen %3d | Pareto=%2d | HV=%.5f | Best AUROC=%.4f "
            "| Lat=%.1fms | RAM=%.0fMB | E=%.3fmJ | %.1fs",
            self._generation, result.n_pareto, hv,
            result.best_auroc, result.best_latency_ms,
            result.best_ram_mb, result.best_energy_mj,
            result.elapsed_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Offspring creation
    # ------------------------------------------------------------------

    def _create_offspring(self) -> np.ndarray:
        """Generate N offspring via tournament selection + crossover + mutation."""
        n  = self._cfg.population_size
        ranks, distances = self._assign_ranks_and_distances(
            self._population, self._objectives
        )
        offspring: list[np.ndarray] = []

        while len(offspring) < n:
            # Binary tournament selection of two parents
            idx_a = self._tournament_select(ranks, distances)
            idx_b = self._tournament_select(ranks, distances)

            p_a = self._population[idx_a]
            p_b = self._population[idx_b]

            # Crossover
            if self._rng.random() < self._cfg.crossover_prob:
                if self._cfg.crossover_type == "uniform":
                    child_a, child_b = self._ss.crossover_uniform(p_a, p_b, self._rng)
                else:
                    child_a, child_b = self._ss.crossover_single_point(p_a, p_b, self._rng)
            else:
                child_a, child_b = p_a.copy(), p_b.copy()

            # Mutation
            if self._cfg.neighbourhood_mutation:
                child_a = self._ss.mutate_neighbourhood(child_a, self._rng, self._mut_rate)
                child_b = self._ss.mutate_neighbourhood(child_b, self._rng, self._mut_rate)
            else:
                child_a = self._ss.mutate(child_a, self._rng, self._mut_rate)
                child_b = self._ss.mutate(child_b, self._rng, self._mut_rate)

            offspring.append(child_a)
            if len(offspring) < n:
                offspring.append(child_b)

        return np.vstack(offspring[:n])

    # ------------------------------------------------------------------
    # Tournament selection
    # ------------------------------------------------------------------

    def _assign_ranks_and_distances(
            self,
            population: np.ndarray,
            objectives: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rank, crowding_distance) arrays for the current population."""
        fronts = fast_non_dominated_sort(objectives)
        n      = len(population)
        ranks  = np.zeros(n, dtype=np.int32)
        dists  = np.zeros(n, dtype=np.float64)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            cd = crowding_distance(objectives, front)
            for idx, d in cd.items():
                dists[idx] = d

        return ranks, dists

    def _tournament_select(self,
                           ranks: np.ndarray,
                           distances: np.ndarray) -> int:
        """Binary (or k-ary) tournament selection.

        Selects the individual with the lowest rank; ties broken by
        highest crowding distance (promotes diversity).

        Returns
        -------
        int
            Index of the selected individual in the current population.
        """
        k         = self._cfg.tournament_size
        n         = len(ranks)
        candidates = self._rng.integers(0, n, size=k)
        best       = int(candidates[0])
        for c in candidates[1:]:
            c = int(c)
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and distances[c] > distances[best]:
                best = c
        return best

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    def _select_next_generation(self,
                                combined_pop: np.ndarray,
                                combined_obj: np.ndarray,
                                n_select: int,
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Select the best *n_select* individuals from the combined pool.

        Fills the next generation with complete fronts (Pareto elitism).
        When a front does not fit entirely, individuals are chosen by
        descending crowding distance to maintain diversity.

        Parameters
        ----------
        combined_pop:
            Gene matrix of shape ``[2N, G]``.
        combined_obj:
            Objective matrix of shape ``[2N, M]``.
        n_select:
            Target next-generation size (``N``).

        Returns
        -------
        tuple
            ``(next_population, next_objectives)`` each of size ``n_select``.
        """
        fronts   = fast_non_dominated_sort(combined_obj)
        selected: list[int] = []

        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                # Partial front: sort by crowding distance (descending)
                remaining  = n_select - len(selected)
                cd         = crowding_distance(combined_obj, front)
                sorted_sub = sorted(front, key=lambda i: cd[i], reverse=True)
                selected.extend(sorted_sub[:remaining])
                break

            if len(selected) >= n_select:
                break

        sel = np.asarray(selected[:n_select], dtype=np.int32)
        return combined_pop[sel].copy(), combined_obj[sel].copy()

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate all individuals and return the objective matrix ``[N, M]``."""
        n   = len(population)
        obj = np.empty((n, self._cfg.n_objectives), dtype=np.float64)
        for i in range(n):
            obj[i] = self._evaluate_one(population[i])
        return obj

    def _evaluate_one(self, gene_vec: np.ndarray) -> np.ndarray:
        """Evaluate a single gene vector.  Returns the objective vector.

        On any exception from the fitness function, the penalty objective
        vector is returned and the failure counter is incremented.
        """
        try:
            candidate = self._encoder.gene_to_dict(gene_vec)
            raw       = self._fit_fn(candidate)
            if raw is None:
                raise ValueError("fitness_fn returned None")
            return self._raw_to_objectives(raw)
        except Exception as exc:  # noqa: BLE001
            self._n_failed += 1
            fp = self._encoder.fingerprint(gene_vec)
            LOG.warning("Evaluation failed [%s]: %s", fp, exc)
            return self._penalty.copy()

    def _raw_to_objectives(self, raw: dict[str, float]) -> np.ndarray:
        """Convert a raw fitness dict to a minimisation objective vector.

        Missing keys are replaced by their penalty values.

        Parameters
        ----------
        raw:
            Dict with at least some of ``{"auroc", "latency_ms",
            "peak_ram_mb", "energy_mj"}``.

        Returns
        -------
        np.ndarray
            Objective vector of length ``n_objectives`` in minimisation form.
        """
        vals = np.array(
            [float(raw.get(k, self._penalty[i]))
             for i, k in enumerate(FITNESS_KEYS)],
            dtype=np.float64,
        )
        return vals * OBJECTIVE_SIGNS  # negate AUROC

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def population(self) -> np.ndarray:
        """Current gene population matrix ``[N, GENOME_LENGTH]``."""
        if self._population is None:
            raise RuntimeError("Engine not initialised.")
        return self._population

    @property
    def objectives(self) -> np.ndarray:
        """Current objective matrix ``[N, M]`` in minimisation form."""
        if self._objectives is None:
            raise RuntimeError("Engine not initialised.")
        return self._objectives

    @property
    def pareto_indices(self) -> list[int]:
        """Indices of Pareto-optimal individuals in the current population."""
        if self._objectives is None:
            raise RuntimeError("Engine not initialised.")
        return fast_non_dominated_sort(self._objectives)[0]

    @property
    def pareto_population(self) -> np.ndarray:
        """Gene matrix of current Pareto-optimal individuals."""
        return self._population[self.pareto_indices]

    @property
    def pareto_objectives(self) -> np.ndarray:
        """Objective matrix of current Pareto-optimal individuals."""
        return self._objectives[self.pareto_indices]

    @property
    def pareto_candidates(self) -> list[dict[str, Any]]:
        """Decoded candidate dicts for the current Pareto front."""
        idx = self.pareto_indices
        candidates = []
        for i in idx:
            c = self._encoder.gene_to_dict(self._population[i])
            # Attach objectives in display form
            raw_obj = self._objectives[i] * OBJECTIVE_SIGNS  # un-negate
            c["auroc"]        = float(raw_obj[0])
            c["latency_ms"]   = float(raw_obj[1])
            c["peak_ram_mb"]  = float(raw_obj[2])
            c["energy_mj"]    = float(raw_obj[3])
            c["is_pareto"]    = True
            c["fingerprint"]  = self._encoder.fingerprint(self._population[i])
            c["generation"]   = self._generation
            candidates.append(c)
        return candidates

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def history(self) -> list[GenerationResult]:
        return list(self._history)

    @property
    def hypervolume_history(self) -> dict[str, list]:
        """Dict suitable for :func:`src.evaluation.plots.plot_hypervolume_evolution`."""
        return {
            "generation": [r.generation for r in self._history],
            "hypervolume": [r.hypervolume for r in self._history],
        }

    # ------------------------------------------------------------------
    # Full population snapshot (for CSV / JSON export)
    # ------------------------------------------------------------------

    def population_snapshot(self) -> list[dict[str, Any]]:
        """Return all individuals in the current population as candidate dicts.

        Each dict is extended with objective values, Pareto rank, and
        crowding distance for downstream CSV writing.
        """
        if self._population is None:
            raise RuntimeError("Engine not initialised.")

        fronts = fast_non_dominated_sort(self._objectives)
        rank_map: dict[int, int] = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                rank_map[idx] = rank

        all_cd: dict[int, float] = {}
        for front in fronts:
            cd = crowding_distance(self._objectives, front)
            all_cd.update(cd)

        snapshot = []
        for i in range(len(self._population)):
            c   = self._encoder.gene_to_dict(self._population[i])
            raw = self._objectives[i] * OBJECTIVE_SIGNS
            c.update({
                "auroc":           float(raw[0]),
                "latency_ms":      float(raw[1]),
                "peak_ram_mb":     float(raw[2]),
                "energy_mj":       float(raw[3]),
                "pareto_rank":     int(rank_map.get(i, -1)),
                "crowding_dist":   float(all_cd.get(i, 0.0)),
                "is_pareto":       rank_map.get(i, -1) == 0,
                "fingerprint":     self._encoder.fingerprint(self._population[i]),
                "generation":      self._generation,
            })
            snapshot.append(c)
        return snapshot

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(self, path: Path | str) -> None:
        """Save the current engine state to a ``.npz`` + ``.json`` pair.

        Parameters
        ----------
        path:
            File path without extension.  Two files are written:
            ``<path>.npz`` (numpy arrays) and ``<path>.json`` (metadata).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(p) + ".npz",
            population=self._population,
            objectives=self._objectives,
        )

        meta = {
            "generation":  self._generation,
            "config":      self._cfg.to_dict(),
            # --- compatibility guards (validated at load time) ---
            "genome_length":   int(self._population.shape[1]),
            "population_size": int(self._population.shape[0]),
            "search_space_hash": search_space_hash(),
            # Full numpy RNG state: a resumed run continues the SAME random
            # stream instead of silently diverging from an uninterrupted one.
            "rng_state": self._rng.bit_generator.state,
            "history":     [
                {k: v for k, v in vars(r).items()} for r in self._history
            ],
        }
        (p.parent / (p.name + ".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        LOG.info("Checkpoint saved -> %s.{npz,json}", p)

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore engine state from a checkpoint.

        Parameters
        ----------
        path:
            File path without extension (same as used in :meth:`checkpoint`).
        """
        p      = Path(path)
        arrays = np.load(str(p) + ".npz")
        self._population = arrays["population"].astype(np.int32)
        self._objectives = arrays["objectives"].astype(np.float64)

        # ---- hard structural guards (always enforceable) -----------------
        if self._population.shape[1] != GENOME_LENGTH:
            raise RuntimeError(
                f"Checkpoint incompatible: genome_length="
                f"{self._population.shape[1]} en el checkpoint vs "
                f"{GENOME_LENGTH} en el código actual. Reanudar mezclaría "
                "dos espacios de búsqueda distintos. Inicia una corrida "
                "nueva o restaura la versión del código que lo generó.")
        if self._objectives.shape[1] != self._cfg.n_objectives:
            raise RuntimeError(
                f"Checkpoint incompatible: n_objectives="
                f"{self._objectives.shape[1]} vs {self._cfg.n_objectives}.")

        meta_path = p.parent / (p.name + ".json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            # ---- metadata guards (checkpoints nuevos los traen) ----------
            ck_hash = meta.get("search_space_hash")
            if ck_hash is not None and ck_hash != search_space_hash():
                raise RuntimeError(
                    "Checkpoint incompatible: el ESPACIO DE BÚSQUEDA cambió "
                    f"desde que se guardó (hash {ck_hash} vs "
                    f"{search_space_hash()}). Los genomas son índices en "
                    "listas de opciones: reanudar re-interpretaría genomas "
                    "viejos como arquitecturas distintas SIN error visible. "
                    "Prohibido reanudar.")
            ck_pop = meta.get("population_size")
            if ck_pop is not None and int(ck_pop) != self._cfg.population_size:
                raise RuntimeError(
                    f"Checkpoint incompatible: population_size={ck_pop} en el "
                    f"checkpoint vs {self._cfg.population_size} en la config "
                    "actual. Ajusta la config o inicia una corrida nueva.")
            ck_gl = meta.get("genome_length")
            if ck_gl is not None and int(ck_gl) != GENOME_LENGTH:
                raise RuntimeError(
                    f"Checkpoint incompatible: genome_length={ck_gl} vs "
                    f"{GENOME_LENGTH}.")
            if ck_hash is None:
                LOG.warning(
                    "Checkpoint legacy sin metadatos de compatibilidad "
                    "(anterior a las guardas): no se puede verificar el "
                    "espacio de búsqueda. Continúa bajo tu responsabilidad.")

            self._generation = int(meta.get("generation", 0))
            self._history    = [
                GenerationResult(**r) for r in meta.get("history", [])
            ]
            # Restore the RNG stream so resumed == uninterrupted.
            rng_state = meta.get("rng_state")
            if rng_state is not None:
                try:
                    self._rng.bit_generator.state = rng_state
                    LOG.info("RNG state restaurado desde el checkpoint.")
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("No se pudo restaurar el estado RNG: %s "
                                "(la reanudación usará un stream nuevo).", exc)
        LOG.info("Checkpoint loaded <- %s (gen=%d)", p, self._generation)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        init = self._population is not None
        return (
            f"NSGA2Engine(N={self._cfg.population_size}, "
            f"M={self._cfg.n_objectives}, "
            f"gen={self._generation}, "
            f"initialised={init})"
        )
