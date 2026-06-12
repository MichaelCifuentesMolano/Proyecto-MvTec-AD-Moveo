"""
src/nas/pareto.py

Pareto-front analysis utilities for the NSGA-II NAS pipeline.

Conventions
-----------
* All objectives are **minimised**.  Callers must negate AUROC before passing
  (consistent with nsga2_engine.py, where OBJECTIVE_SIGNS = [-1, 1, 1, 1]).
* "reference point" means the WORST acceptable objective vector (upper bound).
  It must strictly dominate (be larger than) all non-dominated solutions in
  every dimension for the hypervolume indicator to be positive.

This module has no imports from the NAS engine or hardware layers; it can be
used standalone in post-processing, plotting, and statistical pipelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ArrayLike = np.ndarray | list | tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_array(x: ArrayLike, dtype: type = float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _check_objectives(objectives: np.ndarray) -> None:
    if objectives.ndim != 2:
        raise ValueError(
            f"objectives must be 2-D (n_solutions, n_objectives), "
            f"got shape {objectives.shape}"
        )
    if len(objectives) == 0:
        raise ValueError("objectives array must not be empty")


def _pareto_front_2d_mask(pts: np.ndarray) -> np.ndarray:
    """Boolean mask selecting the 2-D Pareto front (both objectives minimised)."""
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        le = (pts <= pts[i]).all(axis=1)
        lt = (pts < pts[i]).any(axis=1)
        dominated = le & lt
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False
    return is_pareto


# ---------------------------------------------------------------------------
# Pareto-front extraction
# ---------------------------------------------------------------------------


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Return True if solution *a* Pareto-dominates solution *b*.

    Dominance (all objectives minimised): a ≤ b element-wise AND a < b in at
    least one dimension.

    Parameters
    ----------
    a, b : 1-D float arrays of equal length.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Return row indices of non-dominated (rank-0) solutions.

    Uses a vectorised O(N²·M) sweep; adequate for population sizes ≤ 2 000.

    Parameters
    ----------
    objectives : (n, m) float array — all objectives minimised.

    Returns
    -------
    indices : 1-D int array of non-dominated row indices (unsorted).
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)
    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        # Remove i if any other solution dominates it.
        le = (objectives <= objectives[i]).all(axis=1)   # j ≤ i in all dims
        lt = (objectives < objectives[i]).any(axis=1)    # j < i in some dim
        dominated = le & lt
        dominated[i] = False
        if dominated.any():
            is_pareto[i] = False

    return np.where(is_pareto)[0]


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Decompose the population into successive Pareto fronts F₁, F₂, …

    Parameters
    ----------
    objectives : (n, m) float array — all objectives minimised.

    Returns
    -------
    fronts : list of lists; fronts[0] is the Pareto front (rank 0),
             fronts[k] is rank k.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)

    remaining = np.arange(len(objectives))
    fronts: list[list[int]] = []

    while len(remaining):
        local_idx = pareto_front(objectives[remaining])
        global_idx = remaining[local_idx]
        fronts.append(global_idx.tolist())
        keep = np.ones(len(remaining), dtype=bool)
        keep[local_idx] = False
        remaining = remaining[keep]

    return fronts


def crowding_distance(
    objectives: np.ndarray,
    front: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute normalised crowding distance (Deb et al. 2002).

    Parameters
    ----------
    objectives : (n, m) float array.
    front      : optional 1-D int array selecting a subset of rows.
                 If None, all rows are treated as a single front.

    Returns
    -------
    distances : (n,) float array.  Boundary points → +inf.
                Solutions not in *front* remain 0.0.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)

    n = len(objectives)
    distances = np.zeros(n)
    indices = np.arange(n) if front is None else np.asarray(front, dtype=int)

    if len(indices) <= 2:
        distances[indices] = np.inf
        return distances

    sub = objectives[indices]
    sub_dist = np.zeros(len(indices))

    for k in range(objectives.shape[1]):
        order = np.argsort(sub[:, k])
        sub_dist[order[0]] = np.inf
        sub_dist[order[-1]] = np.inf
        f_range = sub[order[-1], k] - sub[order[0], k]
        if f_range == 0.0:
            continue
        sub_dist[order[1:-1]] += (
            (sub[order[2:], k] - sub[order[:-2], k]) / f_range
        )

    distances[indices] = sub_dist
    return distances


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def hypervolume_2d(objectives: np.ndarray, reference: np.ndarray) -> float:
    """
    Exact 2-D hypervolume indicator via sweep line.  O(N log N).

    The dominated hypervolume is the area of the staircase region bounded by
    the Pareto front below-left and the reference point above-right.

    Parameters
    ----------
    objectives : (n, 2) float array — both objectives minimised.
    reference  : (2,) reference / worst-case point; must be strictly worse
                 than all non-dominated solutions in both dimensions.

    Returns
    -------
    hv : float ≥ 0.
    """
    objectives = _to_array(objectives)
    reference = _to_array(reference)
    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    # Keep only solutions strictly dominated by the reference.
    valid = (objectives < reference).all(axis=1)
    if not valid.any():
        return 0.0

    pts = objectives[valid, :2]
    # Work on the 2-D Pareto front only (necessary for the sweep to be correct).
    mask = _pareto_front_2d_mask(pts)
    pts = pts[mask]

    # Sort by first objective ascending; the Pareto front then has second
    # objective descending.
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    # Staircase area: sum_i width_i × height_i
    xs = np.concatenate([pts[:, 0], [reference[0]]])
    widths = np.diff(xs)                           # x_{i+1} - x_i
    heights = reference[1] - pts[:, 1]             # r_y - y_i
    return float(np.sum(widths * heights))


def _hypervolume_3d(objectives: np.ndarray, reference: np.ndarray) -> float:
    """
    Exact 3-D hypervolume via z-sweep.  O(N² log N).

    Slices the volume along the third objective; each slice's cross-section
    is a 2-D hypervolume problem.

    Parameters
    ----------
    objectives : (n, 3) float array.
    reference  : (3,) reference point.
    """
    valid = (objectives < reference).all(axis=1)
    if not valid.any():
        return 0.0

    pts = objectives[valid]
    order = np.argsort(pts[:, 2])
    pts = pts[order]

    hv = 0.0
    for i in range(len(pts)):
        z_next = reference[2] if i == len(pts) - 1 else pts[i + 1, 2]
        dz = z_next - pts[i, 2]
        if dz <= 0.0:
            continue
        # 2-D HV of the active set up to slice i in the x-y plane.
        hv += hypervolume_2d(pts[: i + 1, :2], reference[:2]) * dz

    return hv


def hypervolume_mc(
    objectives: np.ndarray,
    reference: np.ndarray,
    *,
    n_samples: int = 50_000,
    seed: int | np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo hypervolume approximation — works for any number of objectives.

    Uniformly samples points in [ideal, reference] and estimates the fraction
    dominated by at least one solution in the set.  Accuracy improves with
    *n_samples*; 50 000 gives ~1 % relative error for typical Pareto fronts.

    Parameters
    ----------
    objectives : (n, m) float array — all objectives minimised.
    reference  : (m,) reference / worst-case point.
    n_samples  : number of Monte Carlo samples.
    seed       : int seed or ``numpy.random.Generator``.

    Returns
    -------
    hv : float approximation ≥ 0.
    """
    objectives = _to_array(objectives)
    reference = _to_array(reference)

    valid = (objectives < reference).all(axis=1)
    if not valid.any():
        return 0.0

    pts = objectives[valid]
    ideal = pts.min(axis=0)
    volume = float(np.prod(reference - ideal))
    if volume <= 0.0:
        return 0.0

    rng = np.random.default_rng(seed)
    samples = rng.uniform(ideal, reference, size=(n_samples, reference.shape[0]))

    # A sample is dominated if ANY Pareto point p satisfies p ≤ sample (element-wise).
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= (p[np.newaxis, :] <= samples).all(axis=1)

    return float(volume * dominated.mean())


def hypervolume(
    objectives: np.ndarray,
    reference: np.ndarray,
    *,
    method: Literal["auto", "exact", "mc"] = "auto",
    n_samples: int = 50_000,
    seed: int | np.random.Generator | None = None,
) -> float:
    """
    Compute the hypervolume indicator, automatically choosing the algorithm.

    Dispatch table (``method="auto"``):

    * 2 objectives → exact sweep             (``hypervolume_2d``)
    * 3 objectives → exact z-sweep           (``_hypervolume_3d``)
    * 4+ objectives → Monte Carlo            (``hypervolume_mc``)

    Parameters
    ----------
    objectives : (n, m) float array — all objectives minimised.
    reference  : (m,) reference point.
    method     : "auto" | "exact" | "mc".
                 "exact" raises ``ValueError`` for m > 3.
    n_samples  : Monte Carlo sample count (ignored for exact methods).
    seed       : RNG seed for MC.
    """
    objectives = _to_array(objectives)
    reference = _to_array(reference)
    _check_objectives(objectives)

    m = objectives.shape[1]

    if method == "mc":
        return hypervolume_mc(objectives, reference, n_samples=n_samples, seed=seed)

    if method == "exact":
        if m == 2:
            return hypervolume_2d(objectives, reference)
        if m == 3:
            return _hypervolume_3d(objectives, reference)
        raise ValueError(
            f"Exact hypervolume not implemented for m={m}; use method='mc'."
        )

    # auto
    if m <= 2:
        return hypervolume_2d(objectives, reference)
    if m == 3:
        return _hypervolume_3d(objectives, reference)
    return hypervolume_mc(objectives, reference, n_samples=n_samples, seed=seed)


def hypervolume_contributions(
    objectives: np.ndarray,
    reference: np.ndarray,
    *,
    n_samples: int = 50_000,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Exclusive hypervolume contribution of each solution: HV(S) − HV(S \\ {i}).

    Solutions with contribution > 0 are on the Pareto front.  Boundary
    solutions typically have the largest contributions.

    Complexity: O(N × HV_compute).  For N = 50, m = 4, n_samples = 50 000
    this takes a few seconds; intended for post-processing, not inner loops.

    Parameters
    ----------
    objectives : (n, m) float array — all objectives minimised.
    reference  : (m,) reference point.

    Returns
    -------
    contributions : (n,) float array.
    """
    objectives = _to_array(objectives)
    reference = _to_array(reference)
    _check_objectives(objectives)

    n = len(objectives)
    total_hv = hypervolume(objectives, reference, n_samples=n_samples, seed=seed)
    contributions = np.zeros(n)

    for i in range(n):
        reduced = np.delete(objectives, i, axis=0)
        if len(reduced) == 0:
            contributions[i] = total_hv
        else:
            contributions[i] = total_hv - hypervolume(
                reduced, reference, n_samples=n_samples, seed=seed
            )

    return contributions


# ---------------------------------------------------------------------------
# Objective normalisation
# ---------------------------------------------------------------------------


def normalise_objectives(
    objectives: np.ndarray,
    *,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Min-max normalise objectives column-wise to [0, 1].

    Parameters
    ----------
    objectives : (n, m) float array.
    ideal      : (m,) best achievable values; defaults to column-wise minimum.
    nadir      : (m,) worst reference values; defaults to column-wise maximum.
    eps        : guard against zero-range columns.

    Returns
    -------
    normalised : (n, m) float array in [0, 1].
    """
    objectives = _to_array(objectives)
    ideal_ = objectives.min(axis=0) if ideal is None else _to_array(ideal)
    nadir_ = objectives.max(axis=0) if nadir is None else _to_array(nadir)
    ranges = np.where((nadir_ - ideal_) > eps, nadir_ - ideal_, eps)
    return np.clip((objectives - ideal_) / ranges, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


def spacing(objectives: np.ndarray) -> float:
    """
    Spacing metric (Schott 1995).

    Measures uniformity of solution distribution as the standard deviation of
    nearest-neighbour distances in objective space.  A value of 0 indicates
    perfectly uniform spacing; lower is better.

    Parameters
    ----------
    objectives : (n, m) float array.

    Returns
    -------
    s : float ≥ 0.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)
    n = len(objectives)

    if n < 2:
        return 0.0

    d = np.empty(n)
    for i in range(n):
        dists = np.linalg.norm(objectives - objectives[i], axis=1)
        dists[i] = np.inf
        d[i] = dists.min()

    d_mean = d.mean()
    return float(np.sqrt(np.mean((d - d_mean) ** 2)))


def maximum_spread(objectives: np.ndarray) -> np.ndarray:
    """
    Per-objective range (max − min) across all solutions.

    Parameters
    ----------
    objectives : (n, m) float array.

    Returns
    -------
    spreads : (m,) float array.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)
    return objectives.max(axis=0) - objectives.min(axis=0)


def generational_distance(
    approx: np.ndarray,
    reference_front: np.ndarray,
    *,
    p: float = 2.0,
) -> float:
    """
    Generational Distance (GD): average distance from the approximation set
    to the true (or best-known) Pareto front.  Lower is better.

    GD(S, PF*) = (Σ_{s∈S} min_{r∈PF*} ‖s − r‖_p^p)^{1/p} / |S|

    Parameters
    ----------
    approx          : (n, m) float array — candidate solutions.
    reference_front : (k, m) float array — true/best Pareto front.
    p               : Minkowski exponent (default 2 = Euclidean).

    Returns
    -------
    gd : float ≥ 0.
    """
    approx = _to_array(approx)
    reference_front = _to_array(reference_front)
    if approx.ndim == 1:
        approx = approx.reshape(1, -1)

    total = 0.0
    for pt in approx:
        dists = np.linalg.norm(reference_front - pt, axis=1)
        total += float(dists.min() ** p)

    return float((total ** (1.0 / p)) / len(approx))


def inverted_generational_distance(
    approx: np.ndarray,
    reference_front: np.ndarray,
    *,
    p: float = 2.0,
) -> float:
    """
    Inverted Generational Distance (IGD): average distance from the reference
    Pareto front to the approximation set.  Measures coverage; lower is better.

    IGD(S, PF*) = GD(PF*, S).

    Parameters
    ----------
    approx          : (n, m) float array — candidate solutions.
    reference_front : (k, m) float array — true/best Pareto front.
    p               : Minkowski exponent.
    """
    return generational_distance(reference_front, approx, p=p)


def epsilon_indicator(
    approx: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """
    Additive ε-indicator: minimum ε such that *approx* ε-dominates every point
    in *reference_front*.  Lower = closer approximation.

    For each reference point r, the best approximation solution s achieves
    ε(r) = max_k (s_k − r_k).  The overall ε = max over all r of ε(r).

    Parameters
    ----------
    approx          : (n, m) float array.
    reference_front : (k, m) float array.

    Returns
    -------
    eps : float.
    """
    approx = _to_array(approx)
    reference_front = _to_array(reference_front)

    eps = -np.inf
    for ref_pt in reference_front:
        # Per-approximation-solution ε for this reference point.
        per_approx = np.max(approx - ref_pt[np.newaxis, :], axis=1)
        eps = max(eps, float(per_approx.min()))

    return float(eps)


def population_diversity(objectives: np.ndarray) -> dict[str, float]:
    """
    Compute an aggregated diversity summary for a set of solutions.

    Parameters
    ----------
    objectives : (n, m) float array.

    Returns
    -------
    dict with keys:
      spacing, mean_nn_distance,
      max_spread_{k}  (one per objective k),
      crowding_distance_mean, crowding_distance_std.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)

    spreads = maximum_spread(objectives)
    sp = spacing(objectives)

    nn_dists = []
    for i in range(len(objectives)):
        d = np.linalg.norm(objectives - objectives[i], axis=1)
        d[i] = np.inf
        nn_dists.append(float(d.min()))
    nn_arr = np.array(nn_dists)

    cd = crowding_distance(objectives)
    finite_cd = cd[np.isfinite(cd)]

    result: dict[str, float] = {
        "spacing": float(sp),
        "mean_nn_distance": float(nn_arr.mean()),
    }
    for k, s in enumerate(spreads):
        result[f"max_spread_{k}"] = float(s)
    result["crowding_distance_mean"] = float(finite_cd.mean()) if len(finite_cd) else 0.0
    result["crowding_distance_std"] = float(finite_cd.std()) if len(finite_cd) else 0.0

    return result


# ---------------------------------------------------------------------------
# Knee-point detection — internal rankers
# ---------------------------------------------------------------------------


def _rank_chebyshev(
    norm_obj: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    """
    Rank by weighted Chebyshev achievement scalarisation.

    score_i = max_k(w_k · norm_obj[i, k]).
    Smaller score → better balanced trade-off → ranked first.
    """
    m = norm_obj.shape[1]
    w = np.ones(m) / m if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    scores = (norm_obj * w[np.newaxis, :]).max(axis=1)
    return np.argsort(scores)                          # ascending: best knee first


def _rank_bend_angle(norm_obj: np.ndarray) -> np.ndarray:
    """
    Rank by perpendicular (NBI) distance from the utopian diagonal.

    The utopian diagonal connects the ideal point to the nadir in normalised
    objective space.  The knee solution is farthest from this diagonal.
    Larger distance → better knee → ranked first (descending sort).
    """
    ideal = norm_obj.min(axis=0)
    nadir = norm_obj.max(axis=0)
    diagonal = nadir - ideal
    diag_len = float(np.linalg.norm(diagonal))

    if diag_len < 1e-12:
        return np.arange(len(norm_obj))

    unit_diag = diagonal / diag_len
    vecs = norm_obj - ideal[np.newaxis, :]
    proj = (vecs @ unit_diag)[:, np.newaxis] * unit_diag[np.newaxis, :]
    perp = np.linalg.norm(vecs - proj, axis=1)

    return np.argsort(-perp)                           # descending: largest distance first


def _rank_curvature_2d(
    objectives: np.ndarray,
    x_idx: int,
    y_idx: int,
) -> np.ndarray:
    """
    Rank by discrete three-point angle curvature on a 2-D Pareto projection.

    Sorts the projection by the x-axis objective, then at each interior point
    computes the angle between the incoming and outgoing segments.  Sharper
    bends (smaller angles) are ranked first.  Boundary points are ranked last.

    Returns indices into the original *objectives* array.
    """
    pts = objectives[:, [x_idx, y_idx]]
    order = np.argsort(pts[:, 0])          # sort by x: indices into objectives
    pts_sorted = pts[order]
    n = len(pts_sorted)

    if n < 3:
        return order

    angles = np.full(n, np.inf)            # boundary points → +inf (ranked last)
    for i in range(1, n - 1):
        a = pts_sorted[i - 1] - pts_sorted[i]
        b = pts_sorted[i + 1] - pts_sorted[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        cos_angle = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
        angles[i] = math.acos(cos_angle)   # smaller → sharper bend → better knee

    # Map sorted-position ranks back to original objective indices.
    return order[np.argsort(angles)]       # ascending: sharpest bend first


# ---------------------------------------------------------------------------
# Knee-point detection — public interface
# ---------------------------------------------------------------------------


def find_knee_points(
    objectives: np.ndarray,
    *,
    method: Literal["chebyshev", "bend_angle", "curvature_2d"] = "chebyshev",
    weights: np.ndarray | None = None,
    n_knees: int = 1,
    x_idx: int = 0,
    y_idx: int = 1,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
) -> np.ndarray:
    """
    Find knee-point solutions on the Pareto front.

    The input *objectives* should be the Pareto front only (pass
    ``objectives[pareto_front(objectives)]`` if needed).  All objectives are
    treated as minimised; normalisation is applied internally.

    Parameters
    ----------
    objectives : (n, m) float array — Pareto-front solutions, all minimised.
    method     : Knee detection strategy.

                 "chebyshev"    — Weighted Chebyshev scalarisation: finds the
                                  most balanced trade-off across all objectives.
                                  Recommended for ≥ 3 objectives.

                 "bend_angle"   — NBI perpendicular distance from the utopian
                                  diagonal: finds the point of maximum global
                                  curvature in normalised objective space.
                                  Good for 2–4 objectives.

                 "curvature_2d" — Discrete three-point angle curvature on a
                                  2-D Pareto projection (use *x_idx* / *y_idx*
                                  to choose the axes).  Useful for visually
                                  inspecting specific objective pairs.

    weights    : (m,) array for "chebyshev"; uniform if None.
    n_knees    : number of knee candidates to return (best → worst).
    x_idx      : x-axis column for "curvature_2d".
    y_idx      : y-axis column for "curvature_2d".
    ideal      : (m,) ideal point for normalisation; computed if None.
    nadir      : (m,) nadir point for normalisation; computed if None.

    Returns
    -------
    knee_indices : (n_knees,) int array — row indices into *objectives*,
                   ordered best → worst knee.
    """
    objectives = _to_array(objectives)
    _check_objectives(objectives)

    norm = normalise_objectives(objectives, ideal=ideal, nadir=nadir)

    if method == "chebyshev":
        ranked = _rank_chebyshev(norm, weights)
    elif method == "bend_angle":
        ranked = _rank_bend_angle(norm)
    elif method == "curvature_2d":
        ranked = _rank_curvature_2d(objectives, x_idx, y_idx)
    else:
        raise ValueError(
            f"Unknown knee method: {method!r}. "
            "Choose 'chebyshev', 'bend_angle', or 'curvature_2d'."
        )

    return ranked[: min(n_knees, len(objectives))]


# ---------------------------------------------------------------------------
# ParetoSummary dataclass
# ---------------------------------------------------------------------------


@dataclass
class ParetoSummary:
    """
    Aggregated Pareto-front statistics for one generation or experiment.

    Attributes
    ----------
    n_total                    : total population size.
    n_pareto                   : number of non-dominated solutions.
    hypervolume                : HV indicator (higher = better).
    hypervolume_contributions  : exclusive HV contribution per Pareto solution.
    spacing                    : spacing metric (lower = more uniform).
    max_spread                 : per-objective range of the Pareto front.
    mean_nn_distance           : mean nearest-neighbour distance.
    crowding_distance_mean/std : finite crowding distances statistics.
    knee_indices               : row indices into *pareto_objectives* of knee
                                 solutions, ordered best → worst.
    pareto_indices             : row indices into the full *objectives* array.
    ideal                      : column-wise minimum of the Pareto front.
    nadir                      : column-wise maximum of the Pareto front.
    objective_names            : labels for each objective column.
    """

    n_total: int
    n_pareto: int
    hypervolume: float
    hypervolume_contributions: np.ndarray          # (n_pareto,)
    spacing: float
    max_spread: np.ndarray                         # (m,)
    mean_nn_distance: float
    crowding_distance_mean: float
    crowding_distance_std: float
    knee_indices: np.ndarray                       # indices into pareto_objectives
    pareto_indices: np.ndarray                     # indices into full objectives
    ideal: np.ndarray                              # (m,)
    nadir: np.ndarray                              # (m,)
    objective_names: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ParetoSummary(n_pareto={self.n_pareto}/{self.n_total}, "
            f"hv={self.hypervolume:.4f}, spacing={self.spacing:.4f})"
        )

    def as_dict(self) -> dict:
        """Serialisable dict (numpy arrays converted to plain lists)."""
        return {
            "n_total": self.n_total,
            "n_pareto": self.n_pareto,
            "hypervolume": self.hypervolume,
            "hypervolume_contributions": self.hypervolume_contributions.tolist(),
            "spacing": self.spacing,
            "max_spread": self.max_spread.tolist(),
            "mean_nn_distance": self.mean_nn_distance,
            "crowding_distance_mean": self.crowding_distance_mean,
            "crowding_distance_std": self.crowding_distance_std,
            "knee_indices": self.knee_indices.tolist(),
            "pareto_indices": self.pareto_indices.tolist(),
            "ideal": self.ideal.tolist(),
            "nadir": self.nadir.tolist(),
            "objective_names": self.objective_names,
        }


# ---------------------------------------------------------------------------
# ParetoAnalyzer — high-level stateful interface
# ---------------------------------------------------------------------------


class ParetoAnalyzer:
    """
    Stateful Pareto-front analyser wrapping all pure-function utilities.

    Caches the Pareto-front indices after the first call so that repeated
    queries (hypervolume, diversity, knee points) do not re-run the O(N²)
    non-dominated sort.

    Parameters
    ----------
    objectives      : (n, m) float array — all objectives minimised.
                      For the 4-objective NAS problem pass objectives with
                      AUROC already negated (column 0).
    reference       : (m,) reference / worst-case point for hypervolume.
    objective_names : optional m-element list of objective labels.
    hv_n_samples    : MC sample count for hypervolume when m > 3.
    hv_seed         : RNG seed for MC hypervolume.

    Example
    -------
    >>> import numpy as np
    >>> obj = np.array([[-0.92, 12.3, 420., 35.],
    ...                 [-0.85,  8.1, 380., 28.],
    ...                 [-0.78,  5.9, 310., 21.]])
    >>> ref = np.array([0.0, 5000., 16384., 1000.])
    >>> ana = ParetoAnalyzer(obj, ref, objective_names=["-AUROC","lat","ram","energy"])
    >>> summary = ana.summarise()
    >>> print(summary)
    """

    def __init__(
        self,
        objectives: np.ndarray,
        reference: np.ndarray,
        *,
        objective_names: Sequence[str] | None = None,
        hv_n_samples: int = 50_000,
        hv_seed: int = 42,
    ) -> None:
        self._obj = _to_array(objectives)
        _check_objectives(self._obj)
        self._ref = _to_array(reference)
        m = self._obj.shape[1]
        self._names: list[str] = (
            list(objective_names)
            if objective_names is not None
            else [f"f{k}" for k in range(m)]
        )
        self._hv_n_samples = hv_n_samples
        self._hv_seed = hv_seed

        # Lazily computed cache.
        self._pareto_idx: np.ndarray | None = None
        self._fronts: list[list[int]] | None = None

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def objectives(self) -> np.ndarray:
        """Full (n, m) objective array."""
        return self._obj

    @property
    def pareto_indices(self) -> np.ndarray:
        """1-D int array of non-dominated row indices (cached)."""
        if self._pareto_idx is None:
            self._pareto_idx = pareto_front(self._obj)
        return self._pareto_idx

    @property
    def pareto_objectives(self) -> np.ndarray:
        """(n_pareto, m) objective values of the Pareto front."""
        return self._obj[self.pareto_indices]

    @property
    def fronts(self) -> list[list[int]]:
        """All Pareto fronts (ranks 0, 1, …) as lists of global row indices."""
        if self._fronts is None:
            self._fronts = non_dominated_sort(self._obj)
        return self._fronts

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def compute_hypervolume(self, *, pareto_only: bool = True) -> float:
        """
        Hypervolume indicator.

        Parameters
        ----------
        pareto_only : if True (default), compute HV of the Pareto front only;
                      otherwise use the full population.
        """
        obj = self.pareto_objectives if pareto_only else self._obj
        return hypervolume(
            obj, self._ref,
            n_samples=self._hv_n_samples,
            seed=self._hv_seed,
        )

    def compute_hypervolume_contributions(self) -> np.ndarray:
        """
        Exclusive HV contribution for each Pareto solution.

        Returns
        -------
        contributions : (n_pareto,) float array.
        """
        return hypervolume_contributions(
            self.pareto_objectives,
            self._ref,
            n_samples=self._hv_n_samples,
            seed=self._hv_seed,
        )

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def compute_diversity(self) -> dict[str, float]:
        """Diversity metrics for the Pareto front (delegates to population_diversity)."""
        return population_diversity(self.pareto_objectives)

    # ------------------------------------------------------------------
    # Knee points
    # ------------------------------------------------------------------

    def find_knee_points(
        self,
        *,
        method: str = "chebyshev",
        n_knees: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Knee-point indices into the **full population** (not pareto_objectives).

        Parameters
        ----------
        method  : "chebyshev" | "bend_angle" | "curvature_2d".
        n_knees : number of knee solutions to return.
        kwargs  : forwarded to ``find_knee_points()``.

        Returns
        -------
        global_indices : (n_knees,) int array — row indices into self.objectives.
        """
        local_idx = find_knee_points(
            self.pareto_objectives,
            method=method,      # type: ignore[arg-type]
            n_knees=n_knees,
            **kwargs,
        )
        return self.pareto_indices[local_idx]

    # ------------------------------------------------------------------
    # Full summary
    # ------------------------------------------------------------------

    def summarise(
        self,
        *,
        knee_method: Literal["chebyshev", "bend_angle", "curvature_2d"] = "chebyshev",
        n_knees: int = 3,
    ) -> ParetoSummary:
        """
        Compute and return a :class:`ParetoSummary` for the current population.

        Parameters
        ----------
        knee_method : method passed to ``find_knee_points()``.
        n_knees     : number of knee candidates to identify.

        Returns
        -------
        summary : :class:`ParetoSummary` with all metrics filled.

        Notes
        -----
        ``ParetoSummary.knee_indices`` are indices into ``pareto_objectives``
        (the Pareto-front sub-array).  To retrieve full-population indices use
        ``summary.pareto_indices[summary.knee_indices]``.
        """
        pf_obj = self.pareto_objectives
        pf_idx = self.pareto_indices

        hv = self.compute_hypervolume()
        hv_contrib = self.compute_hypervolume_contributions()
        div = self.compute_diversity()
        m = self._obj.shape[1]

        knee_local = find_knee_points(
            pf_obj,
            method=knee_method,
            n_knees=n_knees,
        )

        return ParetoSummary(
            n_total=len(self._obj),
            n_pareto=len(pf_idx),
            hypervolume=hv,
            hypervolume_contributions=hv_contrib,
            spacing=div["spacing"],
            max_spread=np.array([div[f"max_spread_{k}"] for k in range(m)]),
            mean_nn_distance=div["mean_nn_distance"],
            crowding_distance_mean=div["crowding_distance_mean"],
            crowding_distance_std=div["crowding_distance_std"],
            knee_indices=knee_local,
            pareto_indices=pf_idx,
            ideal=pf_obj.min(axis=0),
            nadir=pf_obj.max(axis=0),
            objective_names=self._names,
        )
