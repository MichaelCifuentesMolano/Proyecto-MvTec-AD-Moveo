"""
src/evaluation/plots.py
=======================

Publication-quality figure generation for the NSGA-II multi-objective
optimisation and tracking pipeline.

All functions write self-contained PNG/PDF figures to disk (or return
``matplotlib.figure.Figure`` objects for embedding).  There is no
interactive display dependency — the module forces the ``Agg`` backend when
``$DISPLAY`` is unavailable, making it safe to run on headless Jetson devices
and CI servers.

Figures generated
-----------------
``plot_pareto_front``
    2-D projection of the 4-objective Pareto front, with dominated solutions
    shown in a muted colour and Pareto-optimal solutions highlighted.

``plot_pareto_matrix``
    Lower-triangle scatter matrix showing all 6 pairwise projections of the
    4 objectives simultaneously (similar to a pair plot).

``plot_hypervolume_evolution``
    Hypervolume indicator vs. NSGA-II generation, with optional mean ± std
    band when multiple independent runs are provided.

``plot_tradeoff_scatter``
    Single scatter with a continuous colour axis and optional bubble-size
    axis — useful for spotting Pareto trade-offs along a third or fourth
    dimension.

``plot_tracking_curves``
    Precision/Recall/IoU (and optionally F1 and anomaly score) curves over
    frames or over threshold, per scenario.

``plot_score_distributions``
    KDE + histogram of per-image anomaly scores split by class (normal vs
    anomaly), with a vertical decision threshold line.

``generate_all_plots``
    Orchestrator that calls all of the above from the data structures
    produced by ``main_search.py``, ``main_retrain.py``, and
    ``main_tracking.py``.

Assumptions
-----------
- ``matplotlib ≥ 3.5`` is the only hard dependency.
- ``scipy`` is used for KDE smoothing when available; falls back to
  numpy histogram otherwise.
- Candidate dicts follow the schema produced by ``main_search.py`` /
  ``main_retrain.py``:
  ``{"auroc", "latency_ms", "peak_ram_mb", "energy_mj", "is_pareto"?}``.
- Tracking dicts follow the schema from ``main_tracking.py``:
  ``{"frame", "iou", "precision", "recall", "anomaly_score"?}``.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Force non-interactive backend before importing pyplot.
if not os.environ.get("DISPLAY") and os.name != "nt":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

try:
    from scipy.stats import gaussian_kde  # type: ignore
    _HAVE_KDE = True
except ImportError:  # pragma: no cover
    _HAVE_KDE = False

__all__ = [
    "PlotStyle",
    "plot_pareto_front",
    "plot_pareto_matrix",
    "plot_hypervolume_evolution",
    "plot_tradeoff_scatter",
    "plot_tracking_curves",
    "plot_score_distributions",
    "generate_all_plots",
]

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Objective metadata
# ---------------------------------------------------------------------------

#: Centralised display properties for each tracked objective.
OBJECTIVE_META: dict[str, dict[str, Any]] = {
    "auroc": {
        "label":   "AUROC",
        "unit":    "",
        "scale":   1.0,
        "better":  "higher",
        "fmt":     ".3f",
    },
    "latency_ms": {
        "label":   "Latency",
        "unit":    "ms",
        "scale":   1.0,
        "better":  "lower",
        "fmt":     ".1f",
    },
    "peak_ram_mb": {
        "label":   "Peak RAM",
        "unit":    "MB",
        "scale":   1.0,
        "better":  "lower",
        "fmt":     ".0f",
    },
    "energy_mj": {
        "label":   "Energy",
        "unit":    "mJ/inf",
        "scale":   1.0,
        "better":  "lower",
        "fmt":     ".2f",
    },
}

_ALL_OBJECTIVES = ["auroc", "latency_ms", "peak_ram_mb", "energy_mj"]


def _axis_label(key: str) -> str:
    meta = OBJECTIVE_META.get(key, {})
    label = meta.get("label", key)
    unit  = meta.get("unit", "")
    return f"{label} ({unit})" if unit else label


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

@dataclass
class PlotStyle:
    """Matplotlib style parameters shared across all figures."""
    # Colour scheme
    pareto_color:    str = "#E63946"     # red — highlighted Pareto front
    dominated_color: str = "#A8DADC"    # teal — dominated solutions
    normal_color:    str = "#457B9D"     # blue — normal class scores
    anomaly_color:   str = "#E63946"    # red  — anomaly class scores
    threshold_color: str = "#2D6A4F"    # green — decision threshold
    cmap:            str = "plasma"     # continuous colour maps

    # Figure size presets (width, height) in inches
    figsize_single:  tuple[float, float] = (6.5, 4.5)
    figsize_matrix:  tuple[float, float] = (10.0, 9.5)
    figsize_wide:    tuple[float, float] = (10.0, 4.5)
    figsize_tall:    tuple[float, float] = (6.5, 8.0)

    # Rendering
    dpi:             int   = 150        # screen-quality; use 300 for print
    font_size:       float = 11.0
    title_size:      float = 13.0
    linewidth:       float = 1.8
    markersize:      float = 7.0
    alpha_dominated: float = 0.45
    alpha_pareto:    float = 0.90
    grid_alpha:      float = 0.25

    # File format
    fmt:             str   = "png"      # "png" | "pdf" | "svg"


_DEFAULT_STYLE = PlotStyle()


def _apply_rcparams(style: PlotStyle) -> None:
    """Set global matplotlib rcParams to match the requested style."""
    plt.rcParams.update({
        "figure.dpi":           style.dpi,
        "font.size":            style.font_size,
        "axes.titlesize":       style.title_size,
        "axes.labelsize":       style.font_size,
        "xtick.labelsize":      style.font_size - 1,
        "ytick.labelsize":      style.font_size - 1,
        "legend.fontsize":      style.font_size - 1,
        "axes.grid":            True,
        "grid.alpha":           style.grid_alpha,
        "grid.linestyle":       "--",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "lines.linewidth":      style.linewidth,
        "lines.markersize":     style.markersize,
        "figure.autolayout":    True,
        "savefig.bbox":         "tight",
        "savefig.dpi":          style.dpi,
    })


def _save(fig: Figure,
          save_path: Path | None,
          style: PlotStyle) -> Figure:
    """Save the figure to *save_path* (with format from style) and close it."""
    if save_path is None:
        return fig
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Honour explicit suffix; add style.fmt only when no suffix given.
    if save_path.suffix == "":
        save_path = save_path.with_suffix(f".{style.fmt}")
    fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved figure → %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _extract(candidates: list[dict[str, Any]],
             key: str,
             scale: float = 1.0) -> np.ndarray:
    return np.array(
        [c.get(key, float("nan")) * scale for c in candidates],
        dtype=np.float64,
    )


def _pareto_mask(candidates: list[dict[str, Any]]) -> np.ndarray:
    """Return boolean array — True where ``is_pareto`` is truthy."""
    return np.array(
        [bool(c.get("is_pareto", False)) for c in candidates],
        dtype=bool,
    )


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask: True where all arrays are finite at the same index."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


# ---------------------------------------------------------------------------
# 1. Pareto front — 2-D projection
# ---------------------------------------------------------------------------

def plot_pareto_front(
        candidates: list[dict[str, Any]],
        x_obj: str = "latency_ms",
        y_obj: str = "auroc",
        *,
        style: PlotStyle | None = None,
        title: str | None = None,
        annotate_pareto: bool = True,
        save_path: Path | str | None = None,
) -> Figure:
    """2-D scatter of all candidates with the Pareto front highlighted.

    Parameters
    ----------
    candidates:
        List of candidate dicts from the search or retrain pipeline.
    x_obj, y_obj:
        Objective keys (must appear in ``OBJECTIVE_META`` or the candidate
        dicts).
    style:
        Visual style; uses ``_DEFAULT_STYLE`` when ``None``.
    annotate_pareto:
        Label each Pareto-optimal point with its rank index.
    save_path:
        File path without extension (extension is appended from ``style.fmt``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = style or _DEFAULT_STYLE
    _apply_rcparams(s)

    x_all = _extract(candidates, x_obj)
    y_all = _extract(candidates, y_obj)
    is_pareto = _pareto_mask(candidates)
    valid = _finite_mask(x_all, y_all)

    fig, ax = plt.subplots(figsize=s.figsize_single)

    # Dominated points
    dom = valid & ~is_pareto
    ax.scatter(x_all[dom], y_all[dom],
               c=s.dominated_color, alpha=s.alpha_dominated,
               edgecolors="white", linewidths=0.4,
               label=f"Dominated (n={dom.sum()})", zorder=2)

    # Pareto-optimal points
    par = valid & is_pareto
    ax.scatter(x_all[par], y_all[par],
               c=s.pareto_color, alpha=s.alpha_pareto,
               edgecolors="white", linewidths=0.6,
               s=s.markersize ** 2 * 1.4,
               label=f"Pareto front (n={par.sum()})", zorder=3)

    # Connect Pareto points with a step line (sorted by x)
    if par.sum() > 1:
        order = np.argsort(x_all[par])
        ax.plot(x_all[par][order], y_all[par][order],
                color=s.pareto_color, linewidth=s.linewidth * 0.7,
                linestyle="--", alpha=0.5, zorder=2)

    # Optional annotations
    if annotate_pareto and par.sum() <= 20:
        par_indices = np.where(par)[0]
        for rank, idx in enumerate(
            par_indices[np.argsort(x_all[par_indices])], start=1
        ):
            ax.annotate(
                str(rank),
                (x_all[idx], y_all[idx]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=s.font_size - 3, color=s.pareto_color,
            )

    ax.set_xlabel(_axis_label(x_obj))
    ax.set_ylabel(_axis_label(y_obj))
    ax.set_title(title or f"Pareto Front: {_axis_label(y_obj)} vs {_axis_label(x_obj)}")
    ax.legend(framealpha=0.85)

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 2. Pareto scatter matrix (all pairwise projections)
# ---------------------------------------------------------------------------

def plot_pareto_matrix(
        candidates: list[dict[str, Any]],
        *,
        objectives: list[str] | None = None,
        style: PlotStyle | None = None,
        title: str = "Multi-objective Trade-off Matrix",
        save_path: Path | str | None = None,
) -> Figure:
    """Lower-triangle pairwise scatter matrix across all objectives.

    Parameters
    ----------
    candidates:
        List of candidate dicts.
    objectives:
        Subset of objective keys to include; defaults to all four.
    style, save_path:
        As in :func:`plot_pareto_front`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = style or _DEFAULT_STYLE
    _apply_rcparams(s)

    objs = objectives or _ALL_OBJECTIVES
    k    = len(objs)
    if k < 2:
        raise ValueError("At least 2 objectives required for matrix plot.")

    data      = {o: _extract(candidates, o) for o in objs}
    is_pareto = _pareto_mask(candidates)

    fig, axes = plt.subplots(k - 1, k - 1, figsize=s.figsize_matrix)
    # Ensure axes is always 2-D
    if k == 2:
        axes = np.array([[axes]])

    fig.suptitle(title, fontsize=s.title_size + 1, y=1.01)

    for row in range(k - 1):           # y-axis objective index
        for col in range(k - 1):       # x-axis objective index
            ax = axes[row, col]
            if col > row:              # upper triangle: hide
                ax.set_visible(False)
                continue
            x_key = objs[col]
            y_key = objs[row + 1]
            x_arr = data[x_key]
            y_arr = data[y_key]
            valid = _finite_mask(x_arr, y_arr)

            dom = valid & ~is_pareto
            par = valid & is_pareto

            ax.scatter(x_arr[dom], y_arr[dom],
                       c=s.dominated_color, alpha=s.alpha_dominated,
                       edgecolors="none", s=s.markersize ** 2 * 0.7, zorder=2)
            ax.scatter(x_arr[par], y_arr[par],
                       c=s.pareto_color, alpha=s.alpha_pareto,
                       edgecolors="white", linewidths=0.4,
                       s=s.markersize ** 2, zorder=3)

            if col == 0:
                ax.set_ylabel(_axis_label(y_key), fontsize=s.font_size - 1)
            else:
                ax.set_yticklabels([])
            if row == k - 2:
                ax.set_xlabel(_axis_label(x_key), fontsize=s.font_size - 1)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=s.font_size - 2)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=s.pareto_color,    label="Pareto-optimal"),
        mpatches.Patch(color=s.dominated_color, label="Dominated"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    fig.tight_layout()

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 3. Hypervolume evolution
# ---------------------------------------------------------------------------

def plot_hypervolume_evolution(
        hv_history: dict[str, Any],
        *,
        style: PlotStyle | None = None,
        title: str = "Hypervolume Convergence",
        save_path: Path | str | None = None,
) -> Figure:
    """Line plot of the hypervolume indicator over NSGA-II generations.

    Parameters
    ----------
    hv_history:
        Dict with keys:

        - ``"generation"`` — list/array of generation indices (required).
        - ``"hypervolume"`` — list/array of HV values (required).
        - ``"hypervolume_std"`` — per-generation std across runs (optional).
        - ``"runs"`` — list of {generation, hypervolume} dicts, one per
          independent run (optional; shown as faint lines).

    style, save_path:
        As in :func:`plot_pareto_front`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = style or _DEFAULT_STYLE
    _apply_rcparams(s)

    gen = np.asarray(hv_history["generation"], dtype=np.float64)
    hv  = np.asarray(hv_history["hypervolume"], dtype=np.float64)
    hv_std = hv_history.get("hypervolume_std")
    runs   = hv_history.get("runs", [])

    fig, ax = plt.subplots(figsize=s.figsize_single)

    # Individual run traces (faint background)
    run_color = "#457B9D"
    for run in runs:
        ax.plot(run["generation"], run["hypervolume"],
                color=run_color, alpha=0.25, linewidth=s.linewidth * 0.5,
                zorder=1)

    # Mean (or single run) main line
    ax.plot(gen, hv,
            color=s.pareto_color, linewidth=s.linewidth,
            label="Hypervolume" if not runs else "Mean HV",
            zorder=3)
    ax.scatter(gen, hv, color=s.pareto_color, s=25, zorder=4, edgecolors="white",
               linewidths=0.5)

    # Std band
    if hv_std is not None:
        hv_std_arr = np.asarray(hv_std, dtype=np.float64)
        ax.fill_between(gen, hv - hv_std_arr, hv + hv_std_arr,
                        color=s.pareto_color, alpha=0.15,
                        label="±1 std")

    # Mark final value
    ax.axhline(hv[-1], color=s.pareto_color, linewidth=0.8,
               linestyle=":", alpha=0.6)
    ax.annotate(
        f"Final: {hv[-1]:.4f}",
        xy=(gen[-1], hv[-1]),
        xytext=(-60, 8), textcoords="offset points",
        fontsize=s.font_size - 1.5, color=s.pareto_color,
        arrowprops=dict(arrowstyle="->", color=s.pareto_color, lw=0.8),
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume Indicator")
    ax.set_title(title)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if hv_std is not None or runs:
        ax.legend(framealpha=0.85)

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 4. Tradeoff scatter with continuous colour axis
# ---------------------------------------------------------------------------

def plot_tradeoff_scatter(
        candidates: list[dict[str, Any]],
        x_obj: str = "latency_ms",
        y_obj: str = "auroc",
        color_obj: str = "energy_mj",
        size_obj: str | None = "peak_ram_mb",
        *,
        style: PlotStyle | None = None,
        title: str | None = None,
        mark_pareto: bool = True,
        save_path: Path | str | None = None,
) -> Figure:
    """Scatter plot with colour- and (optionally) size-encoded third/fourth objectives.

    Parameters
    ----------
    candidates:
        List of candidate dicts.
    x_obj, y_obj:
        Primary axes.
    color_obj:
        Objective mapped to the continuous colour axis.
    size_obj:
        Objective mapped to marker area (``None`` = uniform size).
    mark_pareto:
        Draw a star marker over each Pareto-optimal point.
    style, save_path:
        As in :func:`plot_pareto_front`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = style or _DEFAULT_STYLE
    _apply_rcparams(s)

    x   = _extract(candidates, x_obj)
    y   = _extract(candidates, y_obj)
    c   = _extract(candidates, color_obj)
    sz  = _extract(candidates, size_obj) if size_obj else None
    par = _pareto_mask(candidates)
    valid = _finite_mask(x, y, c)
    if sz is not None:
        valid &= np.isfinite(sz)

    # Normalise marker sizes to [30, 250]
    if sz is not None:
        sz_v   = sz[valid]
        sz_min, sz_range = float(sz_v.min()), float(sz_v.max() - sz_v.min())
        marker_sz = 30 + 220 * (sz_v - sz_min) / max(sz_range, 1e-6)
    else:
        marker_sz = s.markersize ** 2

    fig, ax = plt.subplots(figsize=s.figsize_single)

    sc = ax.scatter(
        x[valid], y[valid],
        c=c[valid], s=marker_sz,
        cmap=s.cmap, alpha=0.80,
        edgecolors="white", linewidths=0.4, zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(_axis_label(color_obj), fontsize=s.font_size - 1)

    # Star markers on Pareto-optimal
    if mark_pareto:
        par_valid = valid & par
        if par_valid.any():
            ax.scatter(x[par_valid], y[par_valid],
                       marker="*", s=(s.markersize * 2) ** 2,
                       c="white", edgecolors=s.pareto_color,
                       linewidths=1.2, zorder=4,
                       label="Pareto-optimal")
            ax.legend(framealpha=0.85)

    # Size legend (3 representative values)
    if sz is not None and sz_range > 0:
        for q in (0.1, 0.5, 0.9):
            val = float(np.quantile(sz[valid], q))
            s_norm = 30 + 220 * (val - sz_min) / sz_range
            ax.scatter([], [], s=s_norm,
                       c="grey", alpha=0.6, edgecolors="white", linewidths=0.3,
                       label=f"{_axis_label(size_obj)} ≈ {val:{OBJECTIVE_META.get(size_obj or '', {}).get('fmt', '.1f')}}")
        ax.legend(title=_axis_label(size_obj), framealpha=0.85,
                  fontsize=s.font_size - 2)

    ax.set_xlabel(_axis_label(x_obj))
    ax.set_ylabel(_axis_label(y_obj))
    ax.set_title(
        title or f"{_axis_label(y_obj)} vs {_axis_label(x_obj)} "
                 f"[colour: {_axis_label(color_obj)}]"
    )

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 5. Tracking performance curves
# ---------------------------------------------------------------------------

def plot_tracking_curves(
        tracking_data: dict[str, Any],
        *,
        metrics: list[str] | None = None,
        style: PlotStyle | None = None,
        title: str = "Tracking Performance",
        save_path: Path | str | None = None,
) -> Figure:
    """Per-frame or per-threshold tracking metric curves.

    Parameters
    ----------
    tracking_data:
        Dict accepted in two forms:

        **Flat form** — single scenario::

            {
                "frame":         [0, 1, 2, …],      # or "threshold"
                "iou":           [0.72, 0.85, …],
                "precision":     [0.91, 0.88, …],
                "recall":        [0.84, 0.80, …],
                "f1":            […],                # optional
                "anomaly_score": […],                # optional
            }

        **Scenario form** — multiple named scenarios::

            {
                "scenarios": {
                    "nominal":  {"frame": […], "iou": […], …},
                    "blur":     {"frame": […], "iou": […], …},
                    "occlusion":{"frame": […], "iou": […], …},
                }
            }

        When ``"mean"`` / ``"std"`` sub-keys are present inside a scenario,
        a shaded band is drawn around the mean curve.

    metrics:
        Subset of metric keys to plot.  Defaults to
        ``["iou", "precision", "recall"]``.
    style, save_path:
        As in :func:`plot_pareto_front`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s       = style or _DEFAULT_STYLE
    metrics = metrics or ["iou", "precision", "recall"]
    _apply_rcparams(s)

    # Normalise to scenario dict
    if "scenarios" in tracking_data:
        scenarios: dict[str, dict] = tracking_data["scenarios"]
    else:
        scenarios = {"all": tracking_data}

    x_key = "threshold" if "threshold" in next(iter(scenarios.values())) else "frame"
    x_label = "Detection Threshold" if x_key == "threshold" else "Frame"

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(s.figsize_single[0] * n_metrics / 1.5, s.figsize_single[1]),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    cmap_fn = cm.get_cmap("tab10")
    scenario_names = list(scenarios.keys())
    colors = [cmap_fn(i / max(len(scenario_names), 1))
              for i in range(len(scenario_names))]

    for ax, metric in zip(axes, metrics):
        for color, (scen_name, scen_data) in zip(colors, scenarios.items()):
            if metric not in scen_data:
                continue
            x_raw = np.asarray(scen_data.get(x_key, range(len(scen_data[metric]))),
                                dtype=np.float64)
            y_raw = np.asarray(scen_data[metric], dtype=np.float64)

            # Mean / std form
            if y_raw.ndim == 2:
                y_mean = y_raw.mean(axis=0)
                y_std  = y_raw.std(axis=0)
            elif "mean" in scen_data.get(metric, {}):
                y_mean = np.asarray(scen_data[metric]["mean"])
                y_std  = np.asarray(scen_data[metric].get("std", np.zeros_like(y_mean)))
            else:
                y_mean = y_raw
                y_std  = None

            valid = np.isfinite(x_raw) & np.isfinite(y_mean)
            label = scen_name if len(scenario_names) > 1 else metric.upper()
            ax.plot(x_raw[valid], y_mean[valid],
                    color=color, linewidth=s.linewidth, label=label, zorder=3)

            if y_std is not None:
                ax.fill_between(x_raw[valid],
                                (y_mean - y_std)[valid],
                                (y_mean + y_std)[valid],
                                color=color, alpha=0.15, zorder=2)

            # Annotate final value
            if valid.any():
                final_y = float(y_mean[valid][-1])
                ax.annotate(f"{final_y:.2f}",
                            xy=(x_raw[valid][-1], final_y),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=s.font_size - 3, color=color)

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if len(scenario_names) > 1:
            ax.legend(fontsize=s.font_size - 2, framealpha=0.85)

    fig.suptitle(title, fontsize=s.title_size)
    fig.tight_layout()

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 6. Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
        scores_normal: Sequence[float],
        scores_anomaly: Sequence[float],
        *,
        threshold: float | None = None,
        category: str = "",
        n_bins: int = 40,
        style: PlotStyle | None = None,
        save_path: Path | str | None = None,
) -> Figure:
    """Histogram + KDE overlay of anomaly score distributions.

    Parameters
    ----------
    scores_normal:
        Per-image anomaly scores for the normal class.
    scores_anomaly:
        Per-image anomaly scores for the anomaly class.
    threshold:
        Optional decision threshold vertical line.
    category:
        MVTec category name (used in title).
    n_bins:
        Number of histogram bins.
    style, save_path:
        As in :func:`plot_pareto_front`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = style or _DEFAULT_STYLE
    _apply_rcparams(s)

    norm_arr = np.asarray(scores_normal,  dtype=np.float64)
    anom_arr = np.asarray(scores_anomaly, dtype=np.float64)
    all_vals = np.concatenate([norm_arr, anom_arr])
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    x_range = x_max - x_min
    x_pad   = x_range * 0.05
    bins    = np.linspace(x_min - x_pad, x_max + x_pad, n_bins + 1)

    fig, ax = plt.subplots(figsize=s.figsize_single)

    for arr, color, label, alpha in [
        (norm_arr, s.normal_color,  f"Normal (n={len(norm_arr)})",  0.40),
        (anom_arr, s.anomaly_color, f"Anomaly (n={len(anom_arr)})", 0.35),
    ]:
        if arr.size == 0:
            continue
        ax.hist(arr, bins=bins, density=True, color=color, alpha=alpha,
                edgecolor="white", linewidth=0.3, label=label, zorder=2)

        if _HAVE_KDE and arr.size > 3:
            xs = np.linspace(x_min - x_pad, x_max + x_pad, 400)
            try:
                kde = gaussian_kde(arr, bw_method="scott")
                ax.plot(xs, kde(xs), color=color, linewidth=s.linewidth,
                        alpha=0.9, zorder=3)
            except Exception:  # noqa: BLE001
                pass
        else:
            # Fallback: smooth with numpy histogram + linear interp
            hist, edges = np.histogram(arr, bins=bins, density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centres, hist, color=color, linewidth=s.linewidth,
                    alpha=0.9, zorder=3)

    if threshold is not None:
        ax.axvline(threshold, color=s.threshold_color,
                   linewidth=s.linewidth * 0.9, linestyle="--",
                   label=f"Threshold = {threshold:.3f}", zorder=4)

    cat_str = f" — {category}" if category else ""
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution{cat_str}")
    ax.legend(framealpha=0.85)

    return _save(fig, save_path, s)


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(
        output_dir: Path | str,
        *,
        candidates: list[dict[str, Any]] | None = None,
        hv_history: dict[str, Any] | None = None,
        tracking_data: dict[str, Any] | None = None,
        scores_by_category: dict[str, dict[str, Sequence[float]]] | None = None,
        style: PlotStyle | None = None,
        fmt: str = "png",
) -> dict[str, Path]:
    """Generate all standard figures and save them to *output_dir*.

    Parameters
    ----------
    output_dir:
        Root directory; sub-directories are created automatically.
    candidates:
        Full population of candidates (must include ``"is_pareto"`` key).
    hv_history:
        Dict as expected by :func:`plot_hypervolume_evolution`.
    tracking_data:
        Dict as expected by :func:`plot_tracking_curves`.
    scores_by_category:
        Mapping from category name to
        ``{"normal": [...], "anomaly": [...], "threshold": float?}``.
    style:
        Shared visual style; defaults to ``_DEFAULT_STYLE``.
    fmt:
        Output format: ``"png"`` | ``"pdf"`` | ``"svg"``.

    Returns
    -------
    dict[str, Path]
        Mapping from figure name to absolute file path.
    """
    out    = Path(output_dir)
    s      = style or PlotStyle(fmt=fmt)
    s.fmt  = fmt
    saved: dict[str, Path] = {}

    def _p(name: str) -> Path:
        return out / f"{name}.{fmt}"

    # ---- Pareto front projections ----
    if candidates:
        pareto_dir = out / "pareto"
        # Full scatter matrix
        plot_pareto_matrix(
            candidates, style=s,
            save_path=pareto_dir / f"pareto_matrix.{fmt}",
        )
        saved["pareto_matrix"] = pareto_dir / f"pareto_matrix.{fmt}"

        # Key 2-D projections
        projections = [
            ("latency_ms",  "auroc"),
            ("energy_mj",   "auroc"),
            ("peak_ram_mb", "auroc"),
            ("latency_ms",  "energy_mj"),
        ]
        for x_k, y_k in projections:
            fname = f"pareto_{y_k}_vs_{x_k}"
            plot_pareto_front(
                candidates, x_k, y_k, style=s,
                save_path=pareto_dir / f"{fname}.{fmt}",
            )
            saved[fname] = pareto_dir / f"{fname}.{fmt}"

        # Tradeoff coloured scatters
        tradeoff_dir = out / "tradeoff"
        combos = [
            ("latency_ms", "auroc", "energy_mj",   "peak_ram_mb"),
            ("energy_mj",  "auroc", "latency_ms",  "peak_ram_mb"),
            ("latency_ms", "auroc", "peak_ram_mb",  None),
        ]
        for x_k, y_k, c_k, sz_k in combos:
            fname = f"tradeoff_{y_k}_vs_{x_k}_c_{c_k}"
            plot_tradeoff_scatter(
                candidates, x_k, y_k, c_k, sz_k,
                style=s, save_path=tradeoff_dir / f"{fname}.{fmt}",
            )
            saved[fname] = tradeoff_dir / f"{fname}.{fmt}"

    # ---- Hypervolume evolution ----
    if hv_history:
        plot_hypervolume_evolution(
            hv_history, style=s,
            save_path=_p("hypervolume_evolution"),
        )
        saved["hypervolume_evolution"] = _p("hypervolume_evolution")

    # ---- Tracking curves ----
    if tracking_data:
        track_dir = out / "tracking"
        plot_tracking_curves(
            tracking_data, style=s,
            save_path=track_dir / f"tracking_curves.{fmt}",
        )
        saved["tracking_curves"] = track_dir / f"tracking_curves.{fmt}"

        # F1 + anomaly score if available
        first_scen = (
            next(iter(tracking_data["scenarios"].values()))
            if "scenarios" in tracking_data else tracking_data
        )
        extra_metrics = [m for m in ("f1", "anomaly_score") if m in first_scen]
        if extra_metrics:
            plot_tracking_curves(
                tracking_data, metrics=extra_metrics, style=s,
                title="Tracking — Anomaly Score & F1",
                save_path=track_dir / f"tracking_score_f1.{fmt}",
            )
            saved["tracking_score_f1"] = track_dir / f"tracking_score_f1.{fmt}"

    # ---- Score distributions per category ----
    if scores_by_category:
        dist_dir = out / "distributions"
        for cat, score_dict in scores_by_category.items():
            norm   = score_dict.get("normal",  [])
            anom   = score_dict.get("anomaly", [])
            thr    = score_dict.get("threshold")
            fname  = f"scores_{cat}"
            plot_score_distributions(
                norm, anom, threshold=thr if isinstance(thr, float) else None,
                category=cat, style=s,
                save_path=dist_dir / f"{fname}.{fmt}",
            )
            saved[f"scores_{cat}"] = dist_dir / f"{fname}.{fmt}"

    LOG.info("generate_all_plots: wrote %d figures to %s", len(saved), out)
    return saved
