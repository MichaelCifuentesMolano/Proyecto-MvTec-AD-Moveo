"""
src/nas/convergence.py

Convergence measurement for the NSGA-II NAS search.

Responsibilities
----------------
* Record per-generation statistics (hypervolume, Pareto size, best objectives,
  diversity, elapsed time) as the search progresses.
* Detect plateau conditions via sliding-window improvement thresholds.
* Compute convergence speed (generation at which N % of total HV improvement
  is achieved) and area-under-the-HV-curve efficiency.
* Provide early-stopping signals compatible with the NSGA-II engine's
  ``on_generation_end`` callback interface.
* Summarise multi-run statistics for repeatability analysis.

Assumptions
-----------
* All raw NSGA-II objectives are minimised; AUROC is stored **un-negated**
  (positive) in ``GenerationSnapshot.best_auroc`` — the caller must un-negate.
* Hypervolume is non-decreasing across generations (standard NSGA-II property).
* ``pareto_fingerprints`` are opaque strings (SHA-256 prefixes from encoding.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Stopping-reason enum
# ---------------------------------------------------------------------------


class StoppingReason(str, Enum):
    """Why the convergence tracker recommends stopping (or does not)."""

    NOT_CONVERGED = "not_converged"
    HV_PLATEAU = "hv_plateau"                  # HV improvement < threshold
    PARETO_STABLE = "pareto_stable"            # Pareto fingerprints unchanged
    DIVERSITY_COLLAPSED = "diversity_collapsed"  # spacing near zero
    COMBINED = "combined"                      # multiple criteria simultaneously
    MAX_GENERATIONS = "max_generations"        # external budget exhausted


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceConfig:
    """
    Hyperparameters controlling convergence detection and early stopping.

    Parameters
    ----------
    plateau_window : int
        Number of consecutive generations considered for plateau detection.
    plateau_rel_tol : float
        Relative HV improvement threshold.  The window is declared a plateau
        when ``(hv[t] - hv[t - window]) / max(hv[t - window], eps) < rel_tol``.
    min_generations : int
        Do not trigger early stopping before this many generations regardless
        of other criteria.
    pareto_stability_window : int
        Window (in generations) over which Pareto Jaccard similarity is
        averaged for the stability check.
    pareto_stability_threshold : float
        Mean Jaccard similarity above which the Pareto front is considered
        stable.
    diversity_collapse_threshold : float
        Pareto-front spacing value below which diversity is considered to have
        collapsed (premature convergence warning).
    smoothing_window : int
        Moving-average window for the smoothed HV series (display only).
    require_hv_plateau : bool
        Include HV plateau in the stopping criterion.
    require_pareto_stability : bool
        Also require Pareto stability for stopping.
    require_diversity_ok : bool
        Raise a warning (but do not stop) when diversity collapses.
    """

    plateau_window: int = 10
    plateau_rel_tol: float = 1e-3
    min_generations: int = 20
    pareto_stability_window: int = 5
    pareto_stability_threshold: float = 0.90
    diversity_collapse_threshold: float = 0.01
    smoothing_window: int = 5
    require_hv_plateau: bool = True
    require_pareto_stability: bool = False
    require_diversity_ok: bool = False

    # Small epsilon for safe division.
    _eps: float = field(default=1e-12, repr=False)


# ---------------------------------------------------------------------------
# Per-generation data container
# ---------------------------------------------------------------------------


@dataclass
class GenerationSnapshot:
    """
    Lightweight record of one generation's key statistics.

    Decoupled from ``nsga2_engine.GenerationResult`` so this module can be used
    standalone.  Use ``GenerationSnapshot.from_dict`` to build from the engine's
    output dict, or ``GenerationSnapshot.from_generation_result`` for the
    dataclass form.

    Parameters
    ----------
    generation : int
        Zero-based generation index.
    hypervolume : float
        Hypervolume indicator for this generation (higher = better).
    n_pareto : int
        Number of non-dominated solutions.
    best_auroc : float
        Best (highest) AUROC seen so far — stored **positive** (un-negated).
    best_latency_ms : float
        Best (lowest) inference latency in milliseconds.
    best_ram_mb : float
        Best (lowest) peak RAM usage in megabytes.
    best_energy_mj : float
        Best (lowest) energy per inference in millijoules.
    mean_auroc : float
        Population-mean AUROC (un-negated).
    n_failed_evals : int
        Number of fitness evaluations that returned a penalty this generation.
    elapsed_seconds : float
        Wall-clock time for this generation.
    pareto_fingerprints : list[str]
        Genome fingerprints (hex strings) of the current Pareto front.
    spacing : float
        Pareto-front spacing metric (0 = unknown / not computed).
    diversity_score : float
        Population diversity scalar (e.g. mean Hamming distance; 0 = unknown).
    """

    generation: int
    hypervolume: float
    n_pareto: int
    best_auroc: float
    best_latency_ms: float
    best_ram_mb: float
    best_energy_mj: float
    mean_auroc: float
    n_failed_evals: int
    elapsed_seconds: float
    pareto_fingerprints: list[str] = field(default_factory=list)
    spacing: float = 0.0
    diversity_score: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationSnapshot":
        """
        Build from a plain dict (e.g. loaded from JSON checkpoint history).

        Unknown keys are silently ignored.
        """
        return cls(
            generation=int(d.get("generation", 0)),
            hypervolume=float(d.get("hypervolume", 0.0)),
            n_pareto=int(d.get("n_pareto", 0)),
            best_auroc=float(d.get("best_auroc", 0.0)),
            best_latency_ms=float(d.get("best_latency_ms", 0.0)),
            best_ram_mb=float(d.get("best_ram_mb", 0.0)),
            best_energy_mj=float(d.get("best_energy_mj", 0.0)),
            mean_auroc=float(d.get("mean_auroc", 0.0)),
            n_failed_evals=int(d.get("n_failed_evals", 0)),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            pareto_fingerprints=list(d.get("pareto_fingerprints", [])),
            spacing=float(d.get("spacing", 0.0)),
            diversity_score=float(d.get("diversity_score", 0.0)),
        )

    @classmethod
    def from_generation_result(cls, gr: object) -> "GenerationSnapshot":
        """
        Build from ``nsga2_engine.GenerationResult`` (dataclass or dict-like).

        Accepts both attribute access and dict subscript so it works with the
        dataclass, a plain ``dict``, or any mapping.
        """
        def _get(key: str, default=0.0):
            try:
                return getattr(gr, key)
            except AttributeError:
                try:
                    return gr[key]  # type: ignore[index]
                except (KeyError, TypeError):
                    return default

        return cls(
            generation=int(_get("generation", 0)),
            hypervolume=float(_get("hypervolume", 0.0)),
            n_pareto=int(_get("n_pareto", 0)),
            best_auroc=float(_get("best_auroc", 0.0)),
            best_latency_ms=float(_get("best_latency_ms", 0.0)),
            best_ram_mb=float(_get("best_ram_mb", 0.0)),
            best_energy_mj=float(_get("best_energy_mj", 0.0)),
            mean_auroc=float(_get("mean_auroc", 0.0)),
            n_failed_evals=int(_get("n_failed_evals", 0)),
            elapsed_seconds=float(_get("elapsed_seconds", 0.0)),
            pareto_fingerprints=list(_get("pareto_fingerprints", [])),
            spacing=float(_get("spacing", 0.0)),
            diversity_score=float(_get("diversity_score", 0.0)),
        )


# ---------------------------------------------------------------------------
# Analysis result containers
# ---------------------------------------------------------------------------


@dataclass
class PlateauResult:
    """
    Outcome of a single plateau-detection check.

    Attributes
    ----------
    is_plateau : bool
        True when the series improvement over the window is below the threshold.
    plateau_start_gen : int or None
        Index of the first generation in the detected plateau window, or None.
    window_relative_improvement : float
        ``(series[t] - series[t-window]) / max(|series[t-window]|, eps)``
        for the most recent window.
    """

    is_plateau: bool
    plateau_start_gen: int | None
    window_relative_improvement: float


@dataclass
class SpeedResult:
    """
    Generation indices at which given fractions of total HV improvement occur.

    ``None`` means the threshold was never reached within the recorded history.
    """

    gen_at_50pct: int | None
    gen_at_90pct: int | None
    gen_at_95pct: int | None
    gen_at_99pct: int | None
    auc_fraction: float        # area under HV curve / ideal area (max possible)


@dataclass
class ConvergenceReport:
    """
    Full convergence analysis for a completed (or ongoing) NSGA-II run.

    All ``*_series`` lists are indexed by generation (0-based).

    Attributes
    ----------
    n_generations : int
        Total number of generations recorded.
    converged : bool
        Whether the stopping criterion was triggered.
    convergence_generation : int or None
        First generation at which the stopping criterion was satisfied, or None.
    stopping_reason : StoppingReason
        The criterion responsible for convergence (or NOT_CONVERGED).
    hv_series : list[float]
        Raw hypervolume per generation.
    hv_smoothed : list[float]
        Moving-average-smoothed HV series.
    hv_relative_improvement : list[float]
        Per-generation relative HV improvement; 0.0 at generation 0.
    hv_cumulative_fraction : list[float]
        Fraction of total HV improvement achieved by each generation.
    hv_plateau : PlateauResult
        Plateau check result for the final window.
    speed : SpeedResult
        Convergence speed milestones.
    pareto_size_series : list[int]
        Number of Pareto solutions per generation.
    pareto_jaccard_series : list[float]
        Jaccard similarity between consecutive Pareto fronts; 0.0 at gen 0.
    pareto_stable : bool
        Whether the Pareto front was stable over the final stability window.
    best_auroc_series : list[float]
        Best AUROC (un-negated, positive) per generation.
    best_latency_series : list[float]
        Best latency (ms) per generation.
    best_ram_series : list[float]
        Best RAM (MB) per generation.
    best_energy_series : list[float]
        Best energy (mJ) per generation.
    mean_auroc_series : list[float]
        Population-mean AUROC per generation.
    spacing_series : list[float]
        Pareto-front spacing metric per generation (0 if not provided).
    diversity_series : list[float]
        Population diversity scalar per generation (0 if not provided).
    diversity_collapsed : bool
        True when diversity fell below the collapse threshold.
    n_failed_evals_series : list[int]
        Failed evaluations per generation.
    total_failed_evals : int
        Sum of failed evaluations over all generations.
    elapsed_per_gen : list[float]
        Wall-clock seconds per generation.
    total_elapsed_seconds : float
        Total wall-clock time.
    """

    n_generations: int
    converged: bool
    convergence_generation: int | None
    stopping_reason: StoppingReason

    hv_series: list[float]
    hv_smoothed: list[float]
    hv_relative_improvement: list[float]
    hv_cumulative_fraction: list[float]
    hv_plateau: PlateauResult
    speed: SpeedResult

    pareto_size_series: list[int]
    pareto_jaccard_series: list[float]
    pareto_stable: bool

    best_auroc_series: list[float]
    best_latency_series: list[float]
    best_ram_series: list[float]
    best_energy_series: list[float]
    mean_auroc_series: list[float]

    spacing_series: list[float]
    diversity_series: list[float]
    diversity_collapsed: bool

    n_failed_evals_series: list[int]
    total_failed_evals: int

    elapsed_per_gen: list[float]
    total_elapsed_seconds: float

    def __repr__(self) -> str:
        hv_final = self.hv_series[-1] if self.hv_series else 0.0
        return (
            f"ConvergenceReport(generations={self.n_generations}, "
            f"converged={self.converged}, "
            f"reason={self.stopping_reason.value}, "
            f"hv_final={hv_final:.4f}, "
            f"elapsed={self.total_elapsed_seconds:.1f}s)"
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        hv_i = self.hv_series[0] if self.hv_series else 0.0
        hv_f = self.hv_series[-1] if self.hv_series else 0.0
        lines = [
            "── Convergence Report ──────────────────────────────",
            f"  Generations      : {self.n_generations}",
            f"  Converged        : {self.converged}"
            + (f"  @ gen {self.convergence_generation}" if self.convergence_generation is not None else ""),
            f"  Stopping reason  : {self.stopping_reason.value}",
            f"  HV  initial      : {hv_i:.6f}",
            f"  HV  final        : {hv_f:.6f}",
            f"  HV  improvement  : {((hv_f - hv_i) / max(hv_i, 1e-12)) * 100:.2f} %",
            f"  HV  AUC fraction : {self.speed.auc_fraction:.4f}",
            f"  Speed 50 % HV    : gen {self.speed.gen_at_50pct}",
            f"  Speed 90 % HV    : gen {self.speed.gen_at_90pct}",
            f"  Speed 95 % HV    : gen {self.speed.gen_at_95pct}",
            f"  Pareto stable    : {self.pareto_stable}",
            f"  Diversity colps. : {self.diversity_collapsed}",
            f"  Failed evals     : {self.total_failed_evals}",
            f"  Total time       : {self.total_elapsed_seconds:.1f} s",
            "─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """JSON-serialisable dict (all numpy values cast to Python scalars)."""
        return {
            "n_generations": self.n_generations,
            "converged": self.converged,
            "convergence_generation": self.convergence_generation,
            "stopping_reason": self.stopping_reason.value,
            "hv_series": self.hv_series,
            "hv_smoothed": self.hv_smoothed,
            "hv_relative_improvement": self.hv_relative_improvement,
            "hv_cumulative_fraction": self.hv_cumulative_fraction,
            "hv_plateau": {
                "is_plateau": self.hv_plateau.is_plateau,
                "plateau_start_gen": self.hv_plateau.plateau_start_gen,
                "window_relative_improvement": self.hv_plateau.window_relative_improvement,
            },
            "speed": {
                "gen_at_50pct": self.speed.gen_at_50pct,
                "gen_at_90pct": self.speed.gen_at_90pct,
                "gen_at_95pct": self.speed.gen_at_95pct,
                "gen_at_99pct": self.speed.gen_at_99pct,
                "auc_fraction": self.speed.auc_fraction,
            },
            "pareto_size_series": self.pareto_size_series,
            "pareto_jaccard_series": self.pareto_jaccard_series,
            "pareto_stable": self.pareto_stable,
            "best_auroc_series": self.best_auroc_series,
            "best_latency_series": self.best_latency_series,
            "best_ram_series": self.best_ram_series,
            "best_energy_series": self.best_energy_series,
            "mean_auroc_series": self.mean_auroc_series,
            "spacing_series": self.spacing_series,
            "diversity_series": self.diversity_series,
            "diversity_collapsed": self.diversity_collapsed,
            "n_failed_evals_series": self.n_failed_evals_series,
            "total_failed_evals": self.total_failed_evals,
            "elapsed_per_gen": self.elapsed_per_gen,
            "total_elapsed_seconds": self.total_elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Pure analysis functions
# ---------------------------------------------------------------------------


def smooth(series: list[float], window: int) -> list[float]:
    """
    Centred moving-average smooth of *series* with the given *window* width.

    Boundary values replicate the nearest fully-covered average so the output
    has the same length as the input.

    Parameters
    ----------
    series : list of floats.
    window : averaging window (must be ≥ 1; 1 = no smoothing).

    Returns
    -------
    smoothed : list of floats, same length as *series*.
    """
    if window <= 1 or len(series) <= 1:
        return list(series)

    arr = np.asarray(series, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode="edge")
    out = np.convolve(padded, kernel, mode="valid")[: len(arr)]
    return out.tolist()


def relative_improvement(series: list[float], eps: float = 1e-12) -> list[float]:
    """
    Per-step relative improvement: ``(s[t] - s[t-1]) / max(|s[t-1]|, eps)``.

    The first element is always 0.0.

    Parameters
    ----------
    series : list of floats (e.g. HV per generation).
    eps    : guard against division by zero.

    Returns
    -------
    improvements : list of floats, same length as *series*.
    """
    if len(series) < 2:
        return [0.0] * len(series)

    arr = np.asarray(series, dtype=float)
    prev = np.maximum(np.abs(arr[:-1]), eps)
    deltas = (arr[1:] - arr[:-1]) / prev
    return [0.0] + deltas.tolist()


def cumulative_improvement_fraction(
    series: list[float],
    eps: float = 1e-12,
) -> list[float]:
    """
    Fraction of total improvement achieved by each generation.

    ``frac[t] = (series[t] - series[0]) / max(series[-1] - series[0], eps)``

    Useful for convergence-speed analysis.  Returns all zeros when the series
    is flat.

    Parameters
    ----------
    series : list of floats (assumed non-decreasing, e.g. HV).
    eps    : guard against flat series.

    Returns
    -------
    fractions : list of floats in [0, 1], same length as *series*.
    """
    if not series:
        return []

    arr = np.asarray(series, dtype=float)
    total_gain = float(arr[-1] - arr[0])
    if abs(total_gain) < eps:
        return [0.0] * len(series)

    fracs = np.clip((arr - arr[0]) / total_gain, 0.0, 1.0)
    return fracs.tolist()


def detect_plateau(
    series: list[float],
    window: int,
    rel_tol: float,
    *,
    min_length: int = 0,
    eps: float = 1e-12,
) -> PlateauResult:
    """
    Check whether the tail of *series* constitutes a plateau.

    A plateau is declared when the relative improvement over the last *window*
    steps is below *rel_tol*:

        ``(series[-1] - series[-1-window]) / max(|series[-1-window]|, eps) < rel_tol``

    Parameters
    ----------
    series     : list of floats (e.g. HV values, non-decreasing).
    window     : look-back window in steps.
    rel_tol    : relative improvement threshold.
    min_length : minimum series length before any plateau can be declared.
    eps        : division guard.

    Returns
    -------
    PlateauResult with ``is_plateau``, ``plateau_start_gen``, and
    ``window_relative_improvement``.
    """
    n = len(series)
    if n < window + 1 or n < min_length:
        return PlateauResult(
            is_plateau=False,
            plateau_start_gen=None,
            window_relative_improvement=float("nan"),
        )

    v_now = series[-1]
    v_then = series[-1 - window]
    rel_imp = (v_now - v_then) / max(abs(v_then), eps)
    is_plateau = rel_imp < rel_tol

    return PlateauResult(
        is_plateau=is_plateau,
        plateau_start_gen=(n - 1 - window) if is_plateau else None,
        window_relative_improvement=float(rel_imp),
    )


def pareto_jaccard(fp_a: Sequence[str], fp_b: Sequence[str]) -> float:
    """
    Jaccard similarity between two Pareto-front fingerprint sets.

    Returns 1.0 when both sets are identical, 0.0 when disjoint.
    Returns 0.0 when both sets are empty.

    Parameters
    ----------
    fp_a, fp_b : sequences of genome fingerprint strings.

    Returns
    -------
    similarity : float in [0, 1].
    """
    set_a, set_b = set(fp_a), set(fp_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(len(set_a & set_b) / union)


def convergence_speed(
    hv_series: list[float],
    fractions: Sequence[float] = (0.50, 0.90, 0.95, 0.99),
    eps: float = 1e-12,
) -> dict[float, int | None]:
    """
    Find the first generation at which each fraction of total HV improvement
    is achieved.

    Parameters
    ----------
    hv_series : list of HV values per generation.
    fractions : thresholds to locate (values in (0, 1]).
    eps       : guard for flat series.

    Returns
    -------
    dict mapping each fraction to the generation index (0-based) or None if
    the threshold was never reached.
    """
    cum_frac = np.asarray(cumulative_improvement_fraction(hv_series, eps=eps))
    result: dict[float, int | None] = {}
    for f in fractions:
        idx = np.searchsorted(cum_frac, f, side="left")
        result[f] = int(idx) if idx < len(cum_frac) else None
    return result


def area_under_curve_fraction(hv_series: list[float]) -> float:
    """
    Normalised area under the HV curve.

    ``AUC_fraction = AUC(hv_series) / (n_generations * hv_final)``

    A value of 1.0 means the HV was at its final value from generation 0
    (instant convergence).  Values closer to 0 indicate slow convergence.
    Returns 0.0 for empty or all-zero series.

    Parameters
    ----------
    hv_series : list of HV values per generation.

    Returns
    -------
    auc_fraction : float in [0, 1].
    """
    if not hv_series or hv_series[-1] <= 0.0:
        return 0.0
    n = len(hv_series)
    ideal_area = n * hv_series[-1]
    actual_area = float(np.trapezoid(hv_series))
    return float(np.clip(actual_area / ideal_area, 0.0, 1.0))


def _build_speed_result(hv_series: list[float]) -> SpeedResult:
    speed_map = convergence_speed(hv_series, fractions=(0.50, 0.90, 0.95, 0.99))
    return SpeedResult(
        gen_at_50pct=speed_map.get(0.50),
        gen_at_90pct=speed_map.get(0.90),
        gen_at_95pct=speed_map.get(0.95),
        gen_at_99pct=speed_map.get(0.99),
        auc_fraction=area_under_curve_fraction(hv_series),
    )


# ---------------------------------------------------------------------------
# Standalone report builder
# ---------------------------------------------------------------------------


def compute_convergence_report(
    snapshots: list[GenerationSnapshot],
    config: ConvergenceConfig | None = None,
) -> ConvergenceReport:
    """
    Build a complete :class:`ConvergenceReport` from a list of generation
    snapshots.

    Parameters
    ----------
    snapshots : ordered list of :class:`GenerationSnapshot` objects.
    config    : convergence parameters; uses defaults if None.

    Returns
    -------
    ConvergenceReport with all fields populated.
    """
    cfg = config or ConvergenceConfig()

    if not snapshots:
        plateau = PlateauResult(False, None, float("nan"))
        speed = SpeedResult(None, None, None, None, 0.0)
        return ConvergenceReport(
            n_generations=0,
            converged=False,
            convergence_generation=None,
            stopping_reason=StoppingReason.NOT_CONVERGED,
            hv_series=[], hv_smoothed=[], hv_relative_improvement=[],
            hv_cumulative_fraction=[], hv_plateau=plateau, speed=speed,
            pareto_size_series=[], pareto_jaccard_series=[], pareto_stable=False,
            best_auroc_series=[], best_latency_series=[], best_ram_series=[],
            best_energy_series=[], mean_auroc_series=[],
            spacing_series=[], diversity_series=[], diversity_collapsed=False,
            n_failed_evals_series=[], total_failed_evals=0,
            elapsed_per_gen=[], total_elapsed_seconds=0.0,
        )

    # ── Extract series ─────────────────────────────────────────────────────
    hv_series = [s.hypervolume for s in snapshots]
    pareto_size = [s.n_pareto for s in snapshots]
    best_auroc = [s.best_auroc for s in snapshots]
    best_lat = [s.best_latency_ms for s in snapshots]
    best_ram = [s.best_ram_mb for s in snapshots]
    best_energy = [s.best_energy_mj for s in snapshots]
    mean_auroc = [s.mean_auroc for s in snapshots]
    spacing_s = [s.spacing for s in snapshots]
    diversity_s = [s.diversity_score for s in snapshots]
    failed_s = [s.n_failed_evals for s in snapshots]
    elapsed_s = [s.elapsed_seconds for s in snapshots]

    # ── HV analysis ────────────────────────────────────────────────────────
    hv_smoothed = smooth(hv_series, cfg.smoothing_window)
    hv_rel_imp = relative_improvement(hv_series, eps=cfg._eps)
    hv_cum_frac = cumulative_improvement_fraction(hv_series, eps=cfg._eps)
    hv_plateau = detect_plateau(
        hv_series,
        cfg.plateau_window,
        cfg.plateau_rel_tol,
        min_length=cfg.min_generations,
        eps=cfg._eps,
    )
    speed = _build_speed_result(hv_series)

    # ── Pareto Jaccard ─────────────────────────────────────────────────────
    jaccard_series: list[float] = [0.0]
    for i in range(1, len(snapshots)):
        j = pareto_jaccard(
            snapshots[i - 1].pareto_fingerprints,
            snapshots[i].pareto_fingerprints,
        )
        jaccard_series.append(j)

    # Stability: mean Jaccard over the last stability window.
    win = cfg.pareto_stability_window
    recent_jaccard = jaccard_series[-win:] if len(jaccard_series) >= win else jaccard_series
    mean_jaccard = float(np.mean(recent_jaccard)) if recent_jaccard else 0.0
    pareto_stable = mean_jaccard >= cfg.pareto_stability_threshold

    # ── Diversity collapse ─────────────────────────────────────────────────
    recent_spacing = [s for s in spacing_s[-win:] if s > 0.0]
    diversity_collapsed = bool(
        recent_spacing
        and float(np.mean(recent_spacing)) < cfg.diversity_collapse_threshold
    )

    # ── Stopping criterion ─────────────────────────────────────────────────
    n_gen = len(snapshots)
    past_min = n_gen >= cfg.min_generations

    hv_ok = hv_plateau.is_plateau if cfg.require_hv_plateau else True
    par_ok = pareto_stable if cfg.require_pareto_stability else True
    div_ok = (not diversity_collapsed) if cfg.require_diversity_ok else True

    converged = past_min and hv_ok and par_ok and div_ok

    # Determine stopping reason and first convergence generation.
    stopping_reason = StoppingReason.NOT_CONVERGED
    convergence_generation: int | None = None

    if converged:
        active_reasons: list[StoppingReason] = []
        if cfg.require_hv_plateau and hv_plateau.is_plateau:
            active_reasons.append(StoppingReason.HV_PLATEAU)
        if cfg.require_pareto_stability and pareto_stable:
            active_reasons.append(StoppingReason.PARETO_STABLE)
        if len(active_reasons) == 1:
            stopping_reason = active_reasons[0]
        else:
            stopping_reason = StoppingReason.COMBINED

        # Approximate: use the plateau start generation if available.
        if hv_plateau.plateau_start_gen is not None:
            convergence_generation = hv_plateau.plateau_start_gen + cfg.plateau_window
        else:
            convergence_generation = n_gen - 1

    return ConvergenceReport(
        n_generations=n_gen,
        converged=converged,
        convergence_generation=convergence_generation,
        stopping_reason=stopping_reason,
        hv_series=hv_series,
        hv_smoothed=hv_smoothed,
        hv_relative_improvement=hv_rel_imp,
        hv_cumulative_fraction=hv_cum_frac,
        hv_plateau=hv_plateau,
        speed=speed,
        pareto_size_series=pareto_size,
        pareto_jaccard_series=jaccard_series,
        pareto_stable=pareto_stable,
        best_auroc_series=best_auroc,
        best_latency_series=best_lat,
        best_ram_series=best_ram,
        best_energy_series=best_energy,
        mean_auroc_series=mean_auroc,
        spacing_series=spacing_s,
        diversity_series=diversity_s,
        diversity_collapsed=diversity_collapsed,
        n_failed_evals_series=failed_s,
        total_failed_evals=sum(failed_s),
        elapsed_per_gen=elapsed_s,
        total_elapsed_seconds=sum(elapsed_s),
    )


# ---------------------------------------------------------------------------
# Stateful tracker — primary runtime interface
# ---------------------------------------------------------------------------


class ConvergenceTracker:
    """
    Stateful per-generation convergence tracker.

    Designed to plug directly into ``nsga2_engine.NSGA2Engine.run()`` via the
    ``on_generation_end`` callback (see :meth:`as_callback`).

    Parameters
    ----------
    config : ConvergenceConfig — detection hyperparameters.

    Example
    -------
    ::

        tracker = ConvergenceTracker()
        engine.run(
            n_generations=100,
            on_generation_end=tracker.as_callback(),
        )
        report = tracker.compute_report()
        print(report.summary())
    """

    def __init__(self, config: ConvergenceConfig | None = None) -> None:
        self._cfg = config or ConvergenceConfig()
        self._snapshots: list[GenerationSnapshot] = []
        self._stopped: bool = False
        self._stopping_reason: StoppingReason = StoppingReason.NOT_CONVERGED

        # Rolling window for cheap online checks (avoids recomputing full series).
        self._hv_window: deque[float] = deque(maxlen=self._cfg.plateau_window + 1)
        self._jaccard_window: deque[float] = deque(
            maxlen=self._cfg.pareto_stability_window
        )
        self._last_fingerprints: list[str] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, snapshot: GenerationSnapshot) -> bool:
        """
        Register one generation's snapshot and check stopping criteria.

        Parameters
        ----------
        snapshot : :class:`GenerationSnapshot` for the completed generation.

        Returns
        -------
        should_stop : bool — True when early stopping is recommended.
        """
        self._snapshots.append(snapshot)
        self._hv_window.append(snapshot.hypervolume)

        # Online Jaccard update.
        j = pareto_jaccard(self._last_fingerprints, snapshot.pareto_fingerprints)
        if self._snapshots:    # skip first generation (no previous fingerprints)
            self._jaccard_window.append(j)
        self._last_fingerprints = list(snapshot.pareto_fingerprints)

        # Online stopping check.
        if not self._stopped:
            self._stopped = self._check_stopping()
        return self._stopped

    def record_from_dict(self, d: dict) -> bool:
        """
        Convenience wrapper: build a :class:`GenerationSnapshot` from a dict
        and call :meth:`record`.

        Returns
        -------
        should_stop : bool.
        """
        return self.record(GenerationSnapshot.from_dict(d))

    def as_callback(self) -> Callable[[object], None]:
        """
        Return a callable compatible with the NSGA-II engine's
        ``on_generation_end(generation_result)`` signature.

        The callable calls :meth:`record` internally.  It does NOT raise on
        convergence — the caller must inspect :attr:`should_stop` or use the
        engine's built-in max-generations limit.

        Returns
        -------
        callback : ``Callable[[GenerationResult | dict], None]``
        """
        def _callback(gen_result: object) -> None:
            snap = GenerationSnapshot.from_generation_result(gen_result)
            self.record(snap)

        return _callback

    # ------------------------------------------------------------------
    # Online stopping check
    # ------------------------------------------------------------------

    def _check_stopping(self) -> bool:
        """Run the fast online stopping check.  No full series recomputation."""
        cfg = self._cfg
        n = len(self._snapshots)

        if n < cfg.min_generations:
            return False

        # HV plateau check using the rolling window.
        hv_ok = True
        if cfg.require_hv_plateau:
            if len(self._hv_window) == self._hv_window.maxlen:
                v_now = self._hv_window[-1]
                v_then = self._hv_window[0]
                rel_imp = (v_now - v_then) / max(abs(v_then), cfg._eps)
                hv_ok = rel_imp < cfg.plateau_rel_tol
            else:
                hv_ok = False

        # Pareto stability via rolling Jaccard window.
        par_ok = True
        if cfg.require_pareto_stability:
            if len(self._jaccard_window) >= cfg.pareto_stability_window:
                par_ok = float(np.mean(self._jaccard_window)) >= cfg.pareto_stability_threshold
            else:
                par_ok = False

        converged = hv_ok and par_ok

        if converged:
            active: list[StoppingReason] = []
            if cfg.require_hv_plateau and hv_ok:
                active.append(StoppingReason.HV_PLATEAU)
            if cfg.require_pareto_stability and par_ok:
                active.append(StoppingReason.PARETO_STABLE)
            self._stopping_reason = (
                active[0] if len(active) == 1 else StoppingReason.COMBINED
            )

        return converged

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def should_stop(self) -> bool:
        """True when the stopping criterion has been triggered."""
        return self._stopped

    @property
    def stopping_reason(self) -> StoppingReason:
        """The reason that triggered early stopping (NOT_CONVERGED if running)."""
        return self._stopping_reason

    @property
    def n_generations(self) -> int:
        """Number of generations recorded so far."""
        return len(self._snapshots)

    @property
    def snapshots(self) -> list[GenerationSnapshot]:
        """Immutable view of all recorded snapshots."""
        return list(self._snapshots)

    @property
    def hypervolume_series(self) -> list[float]:
        return [s.hypervolume for s in self._snapshots]

    @property
    def pareto_size_series(self) -> list[int]:
        return [s.n_pareto for s in self._snapshots]

    @property
    def best_auroc_series(self) -> list[float]:
        return [s.best_auroc for s in self._snapshots]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def compute_report(self) -> ConvergenceReport:
        """
        Build and return a full :class:`ConvergenceReport` from the recorded
        snapshots.

        Can be called at any time (including mid-run) to get an up-to-date
        analysis.

        Returns
        -------
        ConvergenceReport with all metrics filled.
        """
        return compute_convergence_report(self._snapshots, self._cfg)

    # ------------------------------------------------------------------
    # Reconstruction from saved history
    # ------------------------------------------------------------------

    @classmethod
    def from_history(
        cls,
        history: list[dict],
        config: ConvergenceConfig | None = None,
    ) -> "ConvergenceTracker":
        """
        Reconstruct a tracker from a saved NSGA-II history list.

        Useful for post-processing runs loaded from ``engine.checkpoint()``
        JSON files without re-running the search.

        Parameters
        ----------
        history : list of generation result dicts (as written by the engine).
        config  : convergence configuration; uses defaults if None.

        Returns
        -------
        ConvergenceTracker with all generations replayed.
        """
        tracker = cls(config)
        for d in history:
            tracker.record(GenerationSnapshot.from_dict(d))
        return tracker


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------


def compare_runs(
    runs: list[list[GenerationSnapshot]],
    *,
    names: list[str] | None = None,
    config: ConvergenceConfig | None = None,
) -> dict:
    """
    Compare convergence behaviour across multiple independent NSGA-II runs.

    Computes per-run reports then aggregates statistics (mean, std, min, max)
    for key convergence metrics.  Useful for repeatability analysis in the
    thesis.

    Parameters
    ----------
    runs  : list of snapshot lists, one list per run.
    names : optional run labels (e.g. ["run_0", "run_1", ...]).
    config: shared convergence configuration.

    Returns
    -------
    dict with keys:

    * ``"n_runs"``               — number of runs.
    * ``"run_names"``            — list of run labels.
    * ``"reports"``              — list of :class:`ConvergenceReport` objects.
    * ``"hv_final"``             — per-run final HV.
    * ``"hv_stats"``             — {mean, std, min, max} over final HVs.
    * ``"convergence_gens"``     — per-run convergence generation (or None).
    * ``"speed_50pct"``          — per-run gen-at-50%-HV.
    * ``"speed_90pct"``          — per-run gen-at-90%-HV.
    * ``"best_auroc_final"``     — per-run best AUROC at last generation.
    * ``"total_elapsed"``        — per-run total elapsed seconds.
    * ``"converged_count"``      — number of runs that triggered convergence.
    * ``"failed_evals_total"``   — total failed evaluations per run.
    * ``"auc_fraction"``         — per-run AUC fraction (convergence efficiency).
    """
    cfg = config or ConvergenceConfig()
    n_runs = len(runs)
    run_names = names if names is not None else [f"run_{i}" for i in range(n_runs)]
    if len(run_names) != n_runs:
        raise ValueError(
            f"len(names)={len(run_names)} does not match len(runs)={n_runs}"
        )

    reports = [compute_convergence_report(r, cfg) for r in runs]

    hv_final = [r.hv_series[-1] if r.hv_series else 0.0 for r in reports]
    hv_arr = np.asarray(hv_final, dtype=float)

    def _first_or_none(lst: list, key: str):
        return [getattr(r, key) for r in lst]

    conv_gens = [r.convergence_generation for r in reports]
    speed_50 = [r.speed.gen_at_50pct for r in reports]
    speed_90 = [r.speed.gen_at_90pct for r in reports]
    best_auroc_final = [
        r.best_auroc_series[-1] if r.best_auroc_series else 0.0 for r in reports
    ]
    elapsed = [r.total_elapsed_seconds for r in reports]
    converged_count = sum(r.converged for r in reports)
    failed_totals = [r.total_failed_evals for r in reports]
    auc_fracs = [r.speed.auc_fraction for r in reports]

    return {
        "n_runs": n_runs,
        "run_names": run_names,
        "reports": reports,
        "hv_final": hv_final,
        "hv_stats": {
            "mean": float(hv_arr.mean()),
            "std": float(hv_arr.std()),
            "min": float(hv_arr.min()),
            "max": float(hv_arr.max()),
        },
        "convergence_gens": conv_gens,
        "speed_50pct": speed_50,
        "speed_90pct": speed_90,
        "best_auroc_final": best_auroc_final,
        "total_elapsed": elapsed,
        "converged_count": converged_count,
        "failed_evals_total": failed_totals,
        "auc_fraction": auc_fracs,
    }
