"""
src/evaluation/statistics.py
============================

Statistical analysis utilities for comparing quantized model configurations
across multiple objectives and repeated experimental runs.

Designed to support three distinct analysis workflows:

1. **Pairwise comparisons** (Wilcoxon signed-rank test)
   Compare two model configurations on matched observations
   (e.g., AUROC scores across the 15 MVTec categories).

2. **Multi-group comparisons** (Friedman test + post-hoc Nemenyi)
   Compare k ≥ 3 configurations simultaneously on the same block structure.
   Friedman's non-parametric analogue of repeated-measures ANOVA is
   appropriate here because metrics across categories are correlated blocks,
   and metric distributions are generally non-Gaussian.

3. **Repeatability / reproducibility analysis**
   Given repeated runs of the same configuration (different seeds), report
   intraclass correlation coefficient (ICC), coefficient of variation (CV),
   within-subject SD, and standard error of measurement (SEM).
   These are the numbers that justify claims like "results are stable
   across seeds with ICC > 0.95".

4. **Confidence intervals**
   Both bootstrap (BCa — bias-corrected accelerated) and parametric
   (Student t) CIs for any scalar statistic.

Public interface
----------------
``wilcoxon_test(a, b, *, alternative, zero_method, correction, alpha)``
    → ``WilcoxonResult``

``friedman_test(*groups, names, alpha)``
    → ``FriedmanResult``

``nemenyi_posthoc(groups, names)``
    → ``PosthocResult``

``bootstrap_ci(data, *, stat_fn, confidence, n_resamples, seed)``
    → ``ConfidenceInterval``

``parametric_ci(data, *, confidence)``
    → ``ConfidenceInterval``

``repeatability_stats(runs, *, names)``
    → ``RepeatabilityResult``

``compare_models(results, metric, *, alpha, n_bootstrap)``
    → ``ModelComparisonReport``

Assumptions
-----------
- scipy ≥ 1.7 is the primary backend; pure-numpy fallbacks cover the most
  critical paths (Wilcoxon exact p-value is replaced by normal approximation,
  Nemenyi by a Bonferroni-Dunn bound) so the module is importable on
  environments where scipy is absent.
- All input sequences are 1-D array-like of finite floats.
- Group sizes for Friedman must be equal (balanced blocks).
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

try:
    from scipy import stats as _sp_stats  # type: ignore
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _sp_stats = None  # type: ignore
    _HAVE_SCIPY = False

__all__ = [
    "ConfidenceInterval",
    "WilcoxonResult",
    "FriedmanResult",
    "PosthocResult",
    "RepeatabilityResult",
    "ModelComparisonReport",
    "wilcoxon_test",
    "friedman_test",
    "nemenyi_posthoc",
    "bootstrap_ci",
    "parametric_ci",
    "repeatability_stats",
    "compare_models",
]

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """A scalar confidence interval for a statistic."""
    statistic:  float
    lower:      float
    upper:      float
    confidence: float
    method:     str            # "bootstrap_bca" | "parametric_t"
    n:          int
    se:         float | None = None   # standard error (parametric only)

    def __str__(self) -> str:
        pct = self.confidence * 100
        return (f"{self.statistic:.4f} "
                f"[{pct:.0f}% CI: {self.lower:.4f}–{self.upper:.4f}] "
                f"(n={self.n}, {self.method})")


@dataclass
class WilcoxonResult:
    """Output of a Wilcoxon signed-rank test."""
    statistic:   float
    p_value:     float
    significant: bool
    alpha:       float
    alternative: str          # "two-sided" | "greater" | "less"
    n_pairs:     int
    effect_size: float        # rank-biserial correlation r
    method:      str          # "scipy" | "normal_approx"
    interpretation: str       # human-readable one-liner

    def __str__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (f"Wilcoxon W={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({sig}), r={self.effect_size:.3f}")


@dataclass
class FriedmanResult:
    """Output of a Friedman test."""
    statistic:   float
    p_value:     float
    significant: bool
    alpha:       float
    df:          int
    n_groups:    int
    n_blocks:    int           # number of observations per group (categories)
    method:      str
    names:       list[str]
    mean_ranks:  dict[str, float]

    def __str__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (f"Friedman χ²({self.df})={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({sig}), k={self.n_groups}")


@dataclass
class PosthocResult:
    """Post-hoc pairwise comparison table (after a significant Friedman)."""
    p_values:   dict[tuple[str, str], float]   # {(a,b): p_adjusted}
    significant: dict[tuple[str, str], bool]
    method:     str     # "nemenyi" | "bonferroni_dunn"
    alpha:      float
    cd:         float | None = None    # critical difference (Nemenyi)


@dataclass
class RepeatabilityResult:
    """Repeatability / reproducibility metrics across repeated runs."""
    n_runs:             int
    n_subjects:         int    # number of items measured per run (categories)
    mean:               float
    grand_std:          float
    within_subject_sd:  float  # sqrt(MS_within) — pure measurement noise
    icc:                float  # ICC(2,1) — two-way mixed, absolute agreement
    icc_ci:             tuple[float, float]  # 95 % CI
    cv:                 float  # coefficient of variation (%)
    sem:                float  # standard error of measurement
    mdc95:              float  # minimal detectable change at 95 % level
    names:              list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"ICC={self.icc:.3f} {self.icc_ci}, "
                f"CV={self.cv:.2f}%, "
                f"within-SD={self.within_subject_sd:.4f}, "
                f"MDC95={self.mdc95:.4f} "
                f"(n_runs={self.n_runs}, n_subjects={self.n_subjects})")


@dataclass
class ModelComparisonReport:
    """Aggregated pairwise and group comparison report for multiple models."""
    metric:     str
    n_blocks:   int          # e.g. 15 MVTec categories
    friedman:   FriedmanResult | None
    posthoc:    PosthocResult | None
    pairwise:   dict[tuple[str, str], WilcoxonResult]
    cis:        dict[str, ConfidenceInterval]
    summary:    dict[str, dict[str, float]]   # name → {mean, median, std, …}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x: Sequence) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array; got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or Inf values.")
    return arr


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared — scipy or pure approximation."""
    if _HAVE_SCIPY:
        return float(_sp_stats.chi2.sf(x, df))
    # Wilson-Hilferty normal approximation (accurate for df > 1)
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def _t_ppf(p: float, df: int) -> float:
    """Percent-point of Student t — scipy or fallback via normal for df > 30."""
    if _HAVE_SCIPY:
        return float(_sp_stats.t.ppf(p, df))
    # For df > 30 the t-distribution is very close to normal.
    if df > 30:
        return float(_norm_ppf(p))
    raise RuntimeError(
        "scipy is required for parametric CI with small samples (df ≤ 30). "
        "Install scipy or use bootstrap_ci instead."
    )


def _norm_ppf(p: float) -> float:
    """Rational approximation of the normal quantile (Abramowitz & Stegun)."""
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0,1); got {p}.")
    # Two-sided: find z such that Phi(z) = p
    sign = 1.0 if p > 0.5 else -1.0
    q = min(p, 1 - p)
    t = math.sqrt(-2.0 * math.log(q))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t ** 2
    den = 1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
    return sign * (t - num / den)


def _rank_biserial(n: int, w_plus: float) -> float:
    """Rank-biserial correlation from Wilcoxon W+ statistic."""
    max_w = n * (n + 1) / 2
    return (2 * w_plus / max_w) - 1 if max_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_test(a: Sequence[float],
                  b: Sequence[float],
                  *,
                  alternative: str = "two-sided",
                  zero_method: str = "wilcox",
                  correction: bool = True,
                  alpha: float = 0.05) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired samples.

    Parameters
    ----------
    a, b:
        Paired observations (e.g., AUROC of model A and model B on the same
        set of MVTec categories).  Must have the same length.
    alternative:
        ``"two-sided"`` | ``"greater"`` (a > b) | ``"less"`` (a < b).
    zero_method:
        How to handle zero differences: ``"wilcox"`` (drop) | ``"pratt"``.
    correction:
        Apply continuity correction to the normal approximation (ignored when
        scipy computes the exact distribution).
    alpha:
        Significance level.

    Returns
    -------
    WilcoxonResult
    """
    a_arr = _to_array(a)
    b_arr = _to_array(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same length.")

    diff = a_arr - b_arr

    if _HAVE_SCIPY:
        try:
            res = _sp_stats.wilcoxon(
                diff, alternative=alternative,
                zero_method=zero_method, correction=correction,
            )
            stat  = float(res.statistic)
            pval  = float(res.pvalue)
            method = "scipy"
        except Exception as exc:  # noqa: BLE001
            LOG.warning("scipy Wilcoxon failed (%s); falling back.", exc)
            stat, pval, method = _wilcoxon_normal_approx(diff, correction)
    else:
        stat, pval, method = _wilcoxon_normal_approx(diff, correction)

    non_zero = diff[diff != 0]
    n_pairs  = len(non_zero)
    r        = _rank_biserial(n_pairs, stat)

    if alternative == "greater":
        interp = (f"a > b is {'supported' if pval < alpha else 'not supported'} "
                  f"at α={alpha}")
    elif alternative == "less":
        interp = (f"a < b is {'supported' if pval < alpha else 'not supported'} "
                  f"at α={alpha}")
    else:
        interp = (f"difference is {'significant' if pval < alpha else 'not significant'} "
                  f"at α={alpha}")

    return WilcoxonResult(
        statistic=stat, p_value=pval,
        significant=(pval < alpha), alpha=alpha,
        alternative=alternative,
        n_pairs=n_pairs, effect_size=r,
        method=method, interpretation=interp,
    )


def _wilcoxon_normal_approx(diff: np.ndarray,
                             correction: bool
                             ) -> tuple[float, float, str]:
    """Normal-approximation fallback for Wilcoxon (no scipy)."""
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return 0.0, 1.0, "normal_approx"
    abs_diff = np.abs(non_zero)
    ranks    = _rank_data(abs_diff)
    w_plus   = float(ranks[non_zero > 0].sum())
    n        = len(ranks)
    mu       = n * (n + 1) / 4
    # Variance with tie correction
    ties     = _tie_sum(ranks)
    sigma2   = (n * (n + 1) * (2 * n + 1) / 24) - ties / 48
    sigma    = math.sqrt(max(sigma2, 1e-12))
    corr     = 0.5 if correction else 0.0
    z        = (w_plus - mu - corr) / sigma
    pval     = 2 * (1 - _norm_cdf(abs(z)))
    return w_plus, pval, "normal_approx"


def _rank_data(x: np.ndarray) -> np.ndarray:
    """Average ranks (handles ties)."""
    order  = np.argsort(x, kind="stable")
    ranks  = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Tie averaging
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


def _tie_sum(ranks: np.ndarray) -> float:
    """Sum of t^3 - t for tie correction in Wilcoxon variance."""
    _, counts = np.unique(ranks, return_counts=True)
    return float(sum(c ** 3 - c for c in counts if c > 1))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def friedman_test(*groups: Sequence[float],
                  names: Sequence[str] | None = None,
                  alpha: float = 0.05) -> FriedmanResult:
    """Friedman non-parametric repeated-measures test.

    Parameters
    ----------
    *groups:
        k sequences of equal length n.  Each sequence is one model
        configuration; each position corresponds to one block (category).
    names:
        Labels for the k groups (used in mean_ranks and downstream posthoc).
    alpha:
        Significance level.

    Returns
    -------
    FriedmanResult
    """
    k = len(groups)
    if k < 3:
        raise ValueError("Friedman test requires at least 3 groups.")
    arrs = [_to_array(g) for g in groups]
    n = len(arrs[0])
    if not all(len(a) == n for a in arrs):
        raise ValueError("All groups must have equal length (balanced blocks).")

    names_list = list(names) if names else [f"group_{i}" for i in range(k)]
    if len(names_list) != k:
        raise ValueError("len(names) must equal number of groups.")

    # Build (n × k) matrix; rank within each block row
    mat    = np.column_stack(arrs)           # [n, k]
    ranked = np.apply_along_axis(_rank_data, axis=1, arr=mat)  # [n, k]
    col_means = ranked.mean(axis=0)          # mean rank per group

    if _HAVE_SCIPY:
        try:
            res  = _sp_stats.friedmanchisquare(*[ranked[:, j] for j in range(k)])
            stat = float(res.statistic)
            pval = float(res.pvalue)
            method = "scipy"
        except Exception as exc:  # noqa: BLE001
            LOG.warning("scipy Friedman failed (%s); falling back.", exc)
            stat, pval, method = _friedman_manual(ranked, n, k)
    else:
        stat, pval, method = _friedman_manual(ranked, n, k)

    mean_ranks = {names_list[j]: float(col_means[j]) for j in range(k)}
    return FriedmanResult(
        statistic=stat, p_value=pval,
        significant=(pval < alpha), alpha=alpha,
        df=(k - 1), n_groups=k, n_blocks=n,
        method=method, names=names_list,
        mean_ranks=mean_ranks,
    )


def _friedman_manual(ranked: np.ndarray,
                     n: int,
                     k: int) -> tuple[float, float, str]:
    col_sums = ranked.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1))) * float((col_sums ** 2).sum()) - 3 * n * (k + 1)
    pval = _chi2_sf(stat, k - 1)
    return stat, pval, "manual"


# ---------------------------------------------------------------------------
# Post-hoc Nemenyi (after a significant Friedman)
# ---------------------------------------------------------------------------

def nemenyi_posthoc(groups: Sequence[Sequence[float]],
                    names: Sequence[str] | None = None,
                    *,
                    alpha: float = 0.05) -> PosthocResult:
    """Nemenyi post-hoc test following a significant Friedman.

    Uses the critical difference (CD) approach with the Studentised range
    distribution when scipy is available; falls back to Bonferroni-Dunn
    (conservative) otherwise.

    Parameters
    ----------
    groups:
        Same k groups as passed to :func:`friedman_test`.
    names:
        Labels for the k groups.
    alpha:
        Family-wise error rate.

    Returns
    -------
    PosthocResult
    """
    arrs = [_to_array(g) for g in groups]
    k    = len(arrs)
    n    = len(arrs[0])
    if not all(len(a) == n for a in arrs):
        raise ValueError("All groups must have equal length.")
    names_list = list(names) if names else [f"group_{i}" for i in range(k)]

    mat    = np.column_stack(arrs)
    ranked = np.apply_along_axis(_rank_data, axis=1, arr=mat)
    mean_ranks = ranked.mean(axis=0)

    cd: float | None = None
    method: str

    if _HAVE_SCIPY:
        try:
            # Studentised range critical value q_alpha(k, inf) / sqrt(2)
            # Nemenyi CD = q * sqrt(k*(k+1) / (6*n))
            q_alpha = float(_sp_stats.studentized_range.ppf(
                1 - alpha, k=k, df=np.inf,
            ))
            cd = q_alpha / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))
            method = "nemenyi"
        except Exception:  # noqa: BLE001
            cd, method = None, "bonferroni_dunn"
    else:
        method = "bonferroni_dunn"

    pairs = list(itertools.combinations(range(k), 2))
    n_comparisons = len(pairs)
    p_values: dict[tuple[str, str], float] = {}
    significant: dict[tuple[str, str], bool] = {}

    if method == "nemenyi" and cd is not None:
        for i, j in pairs:
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            # p-value approximation from rank difference vs CD at alpha
            # CD corresponds to alpha; scale linearly (conservative approx)
            p_approx = min(1.0, alpha * (cd / max(rank_diff, 1e-12)) ** 2)
            key = (names_list[i], names_list[j])
            p_values[key]    = p_approx
            significant[key] = (rank_diff > cd)
    else:
        # Bonferroni-Dunn: CD = z_alpha/n_comp * sqrt(k*(k+1)/(6*n))
        z_bonf = _norm_ppf(1 - alpha / (2 * n_comparisons))
        cd = z_bonf * math.sqrt(k * (k + 1) / (6 * n))
        for i, j in pairs:
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            key = (names_list[i], names_list[j])
            p_values[key]    = min(1.0, 2 * n_comparisons * (1 - _norm_cdf(
                rank_diff / math.sqrt(k * (k + 1) / (6 * n))
            )))
            significant[key] = (rank_diff > cd)

    return PosthocResult(
        p_values=p_values, significant=significant,
        method=method, alpha=alpha, cd=cd,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence interval (BCa)
# ---------------------------------------------------------------------------

def bootstrap_ci(data: Sequence[float],
                 *,
                 stat_fn: Callable[[np.ndarray], float] = np.mean,
                 confidence: float = 0.95,
                 n_resamples: int = 10_000,
                 seed: int = 0) -> ConfidenceInterval:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Parameters
    ----------
    data:
        1-D sample.
    stat_fn:
        Statistic to bootstrap (default: ``np.mean``).
    confidence:
        Coverage probability, e.g. 0.95 for 95 % CI.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    ConfidenceInterval
    """
    arr = _to_array(data)
    n   = len(arr)
    if n < 2:
        raise ValueError("Bootstrap CI requires at least 2 observations.")

    rng           = np.random.default_rng(seed)
    observed_stat = stat_fn(arr)

    # Bootstrap distribution
    indices  = rng.integers(0, n, size=(n_resamples, n))
    boot_stats = np.array([stat_fn(arr[idx]) for idx in indices],
                          dtype=np.float64)

    # BCa bias correction (z0)
    prop_below = float(np.mean(boot_stats < observed_stat))
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = _norm_ppf(prop_below)

    # BCa acceleration (a) via jackknife
    jack_stats = np.array(
        [stat_fn(np.delete(arr, i)) for i in range(n)], dtype=np.float64
    )
    jack_mean = jack_stats.mean()
    num   = float(((jack_mean - jack_stats) ** 3).sum())
    denom = 6.0 * float(((jack_mean - jack_stats) ** 2).sum()) ** 1.5
    a = num / denom if abs(denom) > 1e-12 else 0.0

    alpha2 = (1 - confidence) / 2
    z_lo   = _norm_ppf(alpha2)
    z_hi   = _norm_ppf(1 - alpha2)

    def _adj(z: float) -> float:
        denom_ = 1 - a * (z0 + z)
        if abs(denom_) < 1e-12:
            return 0.5
        return _norm_cdf(z0 + (z0 + z) / denom_)

    lo_q = _adj(z_lo)
    hi_q = _adj(z_hi)

    lower = float(np.percentile(boot_stats, lo_q * 100))
    upper = float(np.percentile(boot_stats, hi_q * 100))

    return ConfidenceInterval(
        statistic=float(observed_stat), lower=lower, upper=upper,
        confidence=confidence, method="bootstrap_bca", n=n,
    )


# ---------------------------------------------------------------------------
# Parametric confidence interval (Student t)
# ---------------------------------------------------------------------------

def parametric_ci(data: Sequence[float],
                  *,
                  confidence: float = 0.95) -> ConfidenceInterval:
    """Parametric confidence interval assuming approximately normal distribution.

    Uses the Student t-distribution with n-1 degrees of freedom.

    Parameters
    ----------
    data:
        1-D sample.
    confidence:
        Coverage probability.

    Returns
    -------
    ConfidenceInterval
    """
    arr  = _to_array(data)
    n    = len(arr)
    if n < 2:
        raise ValueError("Parametric CI requires at least 2 observations.")
    mean = float(arr.mean())
    se   = float(arr.std(ddof=1) / math.sqrt(n))
    t    = _t_ppf(1 - (1 - confidence) / 2, n - 1)
    return ConfidenceInterval(
        statistic=mean, lower=mean - t * se, upper=mean + t * se,
        confidence=confidence, method="parametric_t", n=n, se=se,
    )


# ---------------------------------------------------------------------------
# Repeatability statistics
# ---------------------------------------------------------------------------

def repeatability_stats(runs: Sequence[Sequence[float]],
                        *,
                        names: Sequence[str] | None = None) -> RepeatabilityResult:
    """Compute intraclass correlation and repeatability indices.

    Treats each run as a rater and each subject (e.g., MVTec category) as a
    case in a two-way mixed-effects ANOVA.  Computes ICC(2,1): single-measure,
    absolute agreement, two-way mixed model.

    Parameters
    ----------
    runs:
        k sequences of equal length n, where k = number of repeated runs
        and n = number of items measured per run (e.g., categories).
    names:
        Optional labels for the n subjects.

    Returns
    -------
    RepeatabilityResult

    Notes
    -----
    ICC(2,1) formulation (Shrout & Fleiss 1979):
        ICC = (MSb - MSe) / (MSb + (k-1)*MSe + k*(MSr - MSe)/n)

    where MSb = between-subjects, MSr = between-raters, MSe = residual.
    """
    arrs = [_to_array(r) for r in runs]
    k    = len(arrs)         # n_runs / raters
    if k < 2:
        raise ValueError("repeatability_stats requires at least 2 runs.")
    n = len(arrs[0])
    if not all(len(a) == n for a in arrs):
        raise ValueError("All runs must have equal length.")

    names_list = list(names) if names else [f"subject_{i}" for i in range(n)]
    mat        = np.column_stack(arrs)   # [n, k]

    grand_mean  = float(mat.mean())
    row_means   = mat.mean(axis=1)       # subject means [n]
    col_means   = mat.mean(axis=0)       # rater means   [k]

    ss_between  = k * float(((row_means - grand_mean) ** 2).sum())
    ss_raters   = n * float(((col_means - grand_mean) ** 2).sum())
    ss_total    = float(((mat - grand_mean) ** 2).sum())
    ss_residual = ss_total - ss_between - ss_raters

    df_between  = n - 1
    df_raters   = k - 1
    df_residual = (n - 1) * (k - 1)

    ms_between  = ss_between  / max(df_between,  1)
    ms_raters   = ss_raters   / max(df_raters,   1)
    ms_residual = ss_residual / max(df_residual, 1)

    # ICC(2,1) — two-way mixed, absolute agreement, single measure
    denom = ms_between + (k - 1) * ms_residual + k * max(ms_raters - ms_residual, 0) / n
    icc   = (ms_between - ms_residual) / max(denom, 1e-12)
    icc   = float(np.clip(icc, -1.0, 1.0))

    # 95 % CI for ICC via F-distribution (Shrout & Fleiss)
    icc_ci = _icc_ci(ms_between, ms_residual, n, k, confidence=0.95)

    within_sd = math.sqrt(max(ms_residual, 0.0))
    cv        = (within_sd / max(abs(grand_mean), 1e-12)) * 100.0
    sem_val   = within_sd * math.sqrt(1 - icc)
    mdc95     = sem_val * math.sqrt(2) * 1.96

    return RepeatabilityResult(
        n_runs=k, n_subjects=n,
        mean=grand_mean,
        grand_std=float(mat.std()),
        within_subject_sd=within_sd,
        icc=icc, icc_ci=icc_ci,
        cv=cv, sem=sem_val, mdc95=mdc95,
        names=names_list,
    )


def _icc_ci(ms_b: float,
            ms_e: float,
            n: int,
            k: int,
            confidence: float = 0.95) -> tuple[float, float]:
    """95 % CI for ICC(2,1) via F-ratio method (Shrout & Fleiss 1979)."""
    alpha2 = (1 - confidence) / 2
    df1    = n - 1
    df2    = (n - 1) * (k - 1)
    f_obs  = ms_b / max(ms_e, 1e-12)

    if _HAVE_SCIPY:
        try:
            f_lo = float(_sp_stats.f.ppf(alpha2,     df1, df2))
            f_hi = float(_sp_stats.f.ppf(1 - alpha2, df1, df2))
            lo   = (f_obs / f_hi - 1) / (f_obs / f_hi + k - 1)
            hi   = (f_obs / f_lo - 1) / (f_obs / f_lo + k - 1)
            return float(np.clip(lo, -1, 1)), float(np.clip(hi, -1, 1))
        except Exception:  # noqa: BLE001
            pass
    # Fallback: return point estimate ± normal CI on logit scale
    eps = 1e-6
    icc = (ms_b - ms_e) / max(ms_b + (k - 1) * ms_e, 1e-12)
    icc = float(np.clip(icc, eps, 1 - eps))
    se  = math.sqrt(2 * k * (1 - icc) ** 2 / (k * n * (k - 1)))
    z   = 1.96
    return max(-1.0, icc - z * se), min(1.0, icc + z * se)


# ---------------------------------------------------------------------------
# High-level multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, Sequence[float]],
                   metric: str,
                   *,
                   alpha: float = 0.05,
                   n_bootstrap: int = 5_000,
                   seed: int = 0) -> ModelComparisonReport:
    """Compare k model configurations on a scalar metric across n blocks.

    Parameters
    ----------
    results:
        Mapping from model name to a sequence of n per-block metric values
        (e.g., ``{"modelA": [0.91, 0.88, …], "modelB": [0.89, 0.87, …]}``).
    metric:
        Human-readable name of the metric (stored in the report).
    alpha:
        Significance level for all tests.
    n_bootstrap:
        Bootstrap resamples for CIs.
    seed:
        RNG seed.

    Returns
    -------
    ModelComparisonReport
    """
    if len(results) < 2:
        raise ValueError("compare_models requires at least 2 model entries.")

    names  = list(results.keys())
    groups = [_to_array(results[n]) for n in names]
    n_blocks = len(groups[0])
    if not all(len(g) == n_blocks for g in groups):
        raise ValueError("All model result sequences must have equal length.")

    # Summary statistics per model
    summary: dict[str, dict[str, float]] = {}
    for name, arr in zip(names, groups):
        summary[name] = {
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std(ddof=1)),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "p25":    float(np.percentile(arr, 25)),
            "p75":    float(np.percentile(arr, 75)),
        }

    # Bootstrap CIs per model
    cis: dict[str, ConfidenceInterval] = {}
    for name, arr in zip(names, groups):
        cis[name] = bootstrap_ci(arr, confidence=1 - alpha,
                                 n_resamples=n_bootstrap, seed=seed)

    # Friedman (k ≥ 3) or skip (k == 2 — Wilcoxon suffices)
    friedman_res: FriedmanResult | None  = None
    posthoc_res:  PosthocResult  | None  = None
    if len(groups) >= 3:
        friedman_res = friedman_test(*groups, names=names, alpha=alpha)
        if friedman_res.significant:
            posthoc_res = nemenyi_posthoc(groups, names=names, alpha=alpha)
        else:
            LOG.info(
                "Friedman p=%.4f ≥ α=%.2f; post-hoc not warranted.",
                friedman_res.p_value, alpha,
            )

    # All pairwise Wilcoxon tests (Bonferroni-corrected alpha)
    pairs = list(itertools.combinations(range(len(names)), 2))
    bonf_alpha = alpha / max(len(pairs), 1)
    pairwise: dict[tuple[str, str], WilcoxonResult] = {}
    for i, j in pairs:
        key = (names[i], names[j])
        try:
            pairwise[key] = wilcoxon_test(
                groups[i], groups[j], alpha=bonf_alpha,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Wilcoxon(%s, %s) failed: %s", names[i], names[j], exc)

    return ModelComparisonReport(
        metric=metric, n_blocks=n_blocks,
        friedman=friedman_res, posthoc=posthoc_res,
        pairwise=pairwise, cis=cis, summary=summary,
    )
