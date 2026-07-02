"""
main_report.py
==============

Entry-point orchestration script for the *reporting* stage of the
quantized-NN / MVTec-AD pipeline. Consolidates the artifacts produced by
``main_search.py``, ``main_retrain.py``, ``main_deploy.py``, and
``main_tracking.py`` into a single thesis-ready bundle of tables, plots,
Pareto curves, and a compiled summary document.

Responsibilities
----------------
1. Discover and load (when available) the CSV/JSON outputs of every
   upstream stage. Each stage is *optional*: the report is built from
   whatever data exists at the time of the run.
2. Run statistical summaries via ``src.evaluation.statistics`` — Pareto
   metrics, convergence indicators (best/median per generation,
   hypervolume), descriptive statistics, and pairwise comparisons across
   precisions / scenarios.
3. Render plots via ``src.evaluation.plots`` — Pareto front (2-D
   projections + parallel coordinates), convergence curves, and boxplots
   of the on-device metrics.
4. Export LaTeX tables via ``src.evaluation.export_tables`` — top-K
   candidates, deploy ranking, tracking summary, and per-stage descriptive
   tables — concatenated into a single ``latex_tables.tex`` file.
5. Compose a self-contained ``final_summary.pdf`` that bundles the plots
   and the LaTeX tables (rendered via the export-tables module's PDF
   backend, with a matplotlib ``PdfPages`` fallback so the script always
   produces a deliverable PDF).

Expected module interfaces (downstream contract)
------------------------------------------------
``src.evaluation.statistics``
    ``summarize_search(history: pd.DataFrame,
                       pareto: pd.DataFrame | None,
                       objectives: list[str]) -> dict``
        Returns a JSON-friendly dict with keys: ``"per_generation"`` (list
        of records: gen, n_valid, best, median, worst per objective,
        hypervolume), ``"final_pareto"`` (n, ranges, hypervolume),
        ``"descriptive"`` (mean/std/min/max per objective).

    ``summarize_retrain(metrics: pd.DataFrame,
                        ranked: pd.DataFrame | None) -> dict``
    ``summarize_deploy(runtime: pd.DataFrame,
                       ranked: pd.DataFrame | None) -> dict``
    ``summarize_tracking(metrics: pd.DataFrame,
                         failures: pd.DataFrame | None) -> dict``

    ``compare_groups(values_by_group: dict[str, list[float]],
                     test: str = "mannwhitney") -> dict``

``src.evaluation.plots``
    ``plot_pareto(pareto: pd.DataFrame, output_path: Path,
                  objectives: list[str], history: pd.DataFrame | None = None,
                  title: str | None = None) -> Path``

    ``plot_convergence(per_generation: list[dict] | pd.DataFrame,
                       output_path: Path,
                       metrics: list[str] | None = None,
                       title: str | None = None) -> Path``

    ``plot_boxplots(df: pd.DataFrame, output_path: Path,
                    columns: list[str], group_by: str | None = None,
                    title: str | None = None) -> Path``

``src.evaluation.export_tables``
    ``to_latex_table(df: pd.DataFrame, caption: str, label: str,
                     columns: list[str] | None = None,
                     precision: int = 4) -> str``

    ``export_latex_bundle(sections: list[tuple[str, str]],
                          output_path: Path,
                          title: str | None = None) -> Path``
        Concatenates ``(section_title, latex_body)`` pairs into a single
        ``.tex`` file, returns the path.

    ``export_pdf_summary(payload: dict, output_path: Path,
                         template: str | None = None) -> Path``
        Generates ``final_summary.pdf`` from a structured payload
        ``{"title": str, "sections": [{"heading", "text", "images",
        "tables"}]}``. Implementations may use reportlab, weasyprint, or
        pylatex; this script uses ``PdfPages`` as a fallback if the call
        fails.

Assumptions
-----------
- Upstream stages have written their canonical CSV/JSON artifacts under
  ``results/{search,retrain,deploy,tracking}/``. The script tolerates
  missing stages and only includes available data in the report.
- ``pandas`` and ``matplotlib`` are available; the export modules may
  bring additional optional dependencies (reportlab, etc.) but the
  fallback path keeps the script runnable on a minimal stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore
    PdfPages = None  # type: ignore
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# NOTE: the idealized interfaces this script previously imported
# (summarize_*, plot_pareto/plot_convergence/plot_boxplots, to_latex_table,
# export_latex_bundle, export_pdf_summary) were never implemented under
# src.evaluation — the script could not even be imported. The report is now
# self-contained: summaries are computed here with pandas, plots/LaTeX/PDF
# use the built-in fallback renderers, and statistical comparisons use scipy
# when available.
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scistats  # type: ignore
except ImportError:  # pragma: no cover
    _scistats = None


# ---------------------------------------------------------------------------
# Self-contained statistical summaries
# ---------------------------------------------------------------------------
def _describe(df: "pd.DataFrame | None",
              cols: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if df is None:
        return out
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(s):
            continue
        out[c] = {"n": int(len(s)), "mean": float(s.mean()),
                  "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                  "min": float(s.min()), "median": float(s.median()),
                  "max": float(s.max())}
    return out


def summarize_search(history: "pd.DataFrame | None",
                     pareto: "pd.DataFrame | None",
                     objectives: list[str],
                     generations: "pd.DataFrame | None" = None) -> dict:
    out: dict[str, Any] = {}
    if generations is not None and len(generations):
        out["per_generation"] = generations.to_dict("records")
    if history is not None and len(history):
        out["n_evaluations"] = int(len(history))
        if "valid" in history.columns:
            out["valid_rate"] = float(
                history["valid"].astype(str).str.lower().eq("true").mean())
        out["descriptive"] = _describe(history, objectives)
    if pareto is not None and len(pareto):
        out["final_pareto"] = {"n": int(len(pareto)),
                               "descriptive": _describe(pareto, objectives)}
    return out


def summarize_retrain(metrics: "pd.DataFrame | None",
                      ranked: "pd.DataFrame | None") -> dict:
    out: dict[str, Any] = {}
    if metrics is None or not len(metrics):
        return out
    ok = metrics[metrics["status"].astype(str).str.startswith("ok")] \
        if "status" in metrics.columns else metrics
    out["n_candidates"] = int(len(metrics))
    out["n_ok"] = int(len(ok))
    out["descriptive"] = _describe(
        ok, ["test_auroc", "test_auprc", "test_f1",
             "val_auroc", "train_seconds", "best_epoch"])
    if ranked is not None and len(ranked):
        out["best"] = ranked.iloc[0].to_dict()
    return out


def summarize_deploy(runtime: "pd.DataFrame | None",
                     ranked: "pd.DataFrame | None") -> dict:
    out: dict[str, Any] = {}
    if runtime is None or not len(runtime):
        return out
    ok = runtime[runtime["status"].astype(str).str.startswith("ok")] \
        if "status" in runtime.columns else runtime
    out["n_artifacts"] = int(len(runtime))
    out["n_ok"] = int(len(ok))
    cols = ["latency_ms_mean", "latency_ms_p95", "latency_ms_p99",
            "throughput_fps", "energy_mj_per_inf", "peak_ram_mb",
            "on_device_auroc"]
    out["descriptive"] = _describe(ok, cols)
    if "precision" in ok.columns:
        out["per_precision"] = {
            str(p): _describe(g, cols)
            for p, g in ok.groupby("precision")
        }
    if ranked is not None and len(ranked):
        out["best"] = ranked.iloc[0].to_dict()
    return out


def summarize_tracking(metrics: "pd.DataFrame | None",
                       failures: "pd.DataFrame | None") -> dict:
    out: dict[str, Any] = {}
    if metrics is None or not len(metrics):
        return out
    cols = ["achieved_fps", "detection_latency_ms_p95",
            "end_to_end_latency_ms_p95", "tracker_success_rate",
            "n_lost", "mean_iou", "mean_tracking_error_px",
            "energy_mj_per_frame", "control_saturation_rate"]
    out["n_sessions"] = int(len(metrics))
    out["descriptive"] = _describe(metrics, cols)
    if "scenario" in metrics.columns:
        out["per_scenario"] = {
            str(s): _describe(g, cols)
            for s, g in metrics.groupby("scenario")
        }
    if failures is not None and len(failures) and \
            "failure_type" in failures.columns:
        out["failures_by_type"] = (
            failures["failure_type"].value_counts().to_dict())
    return out


def compare_groups(values_by_group: dict[str, list[float]],
                   test: str = "mannwhitney") -> dict:
    """Pairwise nonparametric comparison across groups (scipy-backed)."""
    if _scistats is None:
        return {"error": "scipy not installed — pip install scipy"}
    names = list(values_by_group.keys())
    pairs: dict[str, dict[str, float]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = values_by_group[names[i]], values_by_group[names[j]]
            try:
                stat, p = _scistats.mannwhitneyu(a, b,
                                                 alternative="two-sided")
                pairs[f"{names[i]} vs {names[j]}"] = {
                    "U": float(stat), "p_value": float(p),
                    "n_a": len(a), "n_b": len(b)}
            except ValueError as exc:
                pairs[f"{names[i]} vs {names[j]}"] = {"error": str(exc)}
    return {"test": test, "pairs": pairs}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# ``config/`` (singular) — ``configs/`` silently fell back to defaults.
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config" / "report.yaml"
DEFAULT_RESULTS_ROOT: Path = PROJECT_ROOT / "results"
DEFAULT_REPORT_DIR: Path = DEFAULT_RESULTS_ROOT / "report"

OBJECTIVES: tuple[str, ...] = (
    "neg_auroc", "latency_ms", "peak_ram_mb", "energy_mj",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ReportConfig:
    """Configuration for the reporting stage."""

    # I/O
    results_root: Path = DEFAULT_RESULTS_ROOT
    report_dir: Path = DEFAULT_REPORT_DIR

    # Per-stage subdirectories (relative to ``results_root`` unless absolute).
    search_dir: Path = Path("search")
    retrain_dir: Path = Path("retrain")
    deploy_dir: Path = Path("deploy")
    tracking_dir: Path = Path("tracking")

    # Output filenames (placed under ``report_dir``).
    pareto_png: str = "pareto_front.png"
    convergence_png: str = "convergence.png"
    boxplots_png: str = "boxplots.png"
    latex_tex: str = "latex_tables.tex"
    summary_pdf: str = "final_summary.pdf"
    summary_json: str = "report_summary.json"

    # Document metadata
    title: str = ("Quantized Neural Networks for Embedded Anomaly Detection "
                  "— Experimental Report")
    author: str = "Doctoral Research Project"

    # Plot / table behavior
    objectives: tuple[str, ...] = OBJECTIVES
    pareto_axes: tuple[str, str] = ("latency_ms", "neg_auroc")
    # Column names must exist in results/search/generations.csv
    # (written per generation by main_search.py).
    convergence_metrics: tuple[str, ...] = (
        "hypervolume", "best_auroc", "best_latency_ms", "n_pareto",
    )
    boxplot_columns: tuple[str, ...] = (
        "latency_ms_p95", "energy_mj_per_inf",
        "peak_ram_mb", "on_device_auroc",
    )
    boxplot_group_by: str = "precision"
    table_precision: int = 4
    top_k_candidates: int = 10

    # Behavior flags
    include_search: bool = True
    include_retrain: bool = True
    include_deploy: bool = True
    include_tracking: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path) -> "ReportConfig":
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to parse YAML configs.")
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportConfig":
        kwargs = dict(data)
        for key in ("results_root", "report_dir", "search_dir",
                    "retrain_dir", "deploy_dir", "tracking_dir"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        for key in ("objectives", "pareto_axes", "convergence_metrics",
                    "boxplot_columns"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    # ------------------------------------------------------------------
    def resolve(self, sub: Path) -> Path:
        """Resolve a sub-directory against ``results_root`` if relative."""
        return sub if sub.is_absolute() else self.results_root / sub


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging(log_path: Path | None,
                       level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("report")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_csv_safe(path: Path,
                   logger: logging.Logger) -> pd.DataFrame | None:
    if not path.is_file():
        logger.info("Skipping (not found): %s", path)
        return None
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s — %d row(s), %d col(s)",
                    path, len(df), len(df.columns))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _read_json_safe(path: Path,
                    logger: logging.Logger) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str),
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Data loaders (one per upstream stage)
# ---------------------------------------------------------------------------
@dataclass
class StageData:
    """Container for the artifacts loaded from a single upstream stage."""

    name: str
    path: Path
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    json_blobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    available: bool = False


def _load_search(cfg: ReportConfig,
                 logger: logging.Logger) -> StageData:
    base = cfg.resolve(cfg.search_dir)
    s = StageData(name="search", path=base)
    s.frames["history"] = _read_csv_safe(base / "history.csv", logger)
    s.frames["pareto"] = _read_csv_safe(base / "pareto_front.csv", logger)
    s.frames["generations"] = _read_csv_safe(base / "generations.csv", logger)
    s.json_blobs["best"] = _read_json_safe(
        base / "best_candidates.json", logger,
    ) or {}
    s.json_blobs["summary"] = _read_json_safe(
        base / "search_summary.json", logger,
    ) or {}
    s.available = s.frames["history"] is not None or \
                  s.frames["pareto"] is not None
    return s


def _load_retrain(cfg: ReportConfig,
                  logger: logging.Logger) -> StageData:
    base = cfg.resolve(cfg.retrain_dir)
    s = StageData(name="retrain", path=base)
    s.frames["metrics"] = _read_csv_safe(base / "final_metrics.csv", logger)
    s.frames["ranked"]  = _read_csv_safe(base / "model_ranked.csv", logger)
    s.json_blobs["summary"] = _read_json_safe(
        base / "retrain_summary.json", logger,
    ) or {}
    s.available = s.frames["metrics"] is not None
    return s


def _load_deploy(cfg: ReportConfig,
                 logger: logging.Logger) -> StageData:
    base = cfg.resolve(cfg.deploy_dir)
    s = StageData(name="deploy", path=base)
    s.frames["runtime"] = _read_csv_safe(base / "runtime_metrics.csv", logger)
    s.frames["ranked"]  = _read_csv_safe(
        base / "final_embedded_rank.csv", logger,
    )
    s.json_blobs["summary"] = _read_json_safe(
        base / "deploy_summary.json", logger,
    ) or {}
    s.available = s.frames["runtime"] is not None
    return s


def _load_tracking(cfg: ReportConfig,
                   logger: logging.Logger) -> StageData:
    base = cfg.resolve(cfg.tracking_dir)
    s = StageData(name="tracking", path=base)
    s.frames["metrics"]  = _read_csv_safe(
        base / "tracking_metrics.csv", logger,
    )
    s.frames["failures"] = _read_csv_safe(
        base / "failure_cases.csv", logger,
    )
    s.json_blobs["summary"] = _read_json_safe(
        base / "tracking_summary.json", logger,
    ) or {}
    s.available = s.frames["metrics"] is not None
    return s


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class ReportPipeline:
    """Builds the consolidated report from upstream-stage artifacts."""

    def __init__(self, cfg: ReportConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("report")
        cfg.report_dir.mkdir(parents=True, exist_ok=True)

        self._stages: dict[str, StageData] = {}
        self._stats: dict[str, Any] = {}
        self._artifacts: dict[str, Path | None] = {
            "pareto_png": None,
            "convergence_png": None,
            "boxplots_png": None,
            "latex_tex": None,
            "summary_pdf": None,
        }
        self._latex_sections: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        if self.cfg.include_search:
            self._stages["search"] = _load_search(self.cfg, self.log)
        if self.cfg.include_retrain:
            self._stages["retrain"] = _load_retrain(self.cfg, self.log)
        if self.cfg.include_deploy:
            self._stages["deploy"] = _load_deploy(self.cfg, self.log)
        if self.cfg.include_tracking:
            self._stages["tracking"] = _load_tracking(self.cfg, self.log)

        avail = [n for n, s in self._stages.items() if s.available]
        self.log.info("Stages with available data: %s",
                      avail if avail else "<none>")
        if not avail:
            raise RuntimeError(
                "No upstream stage artifacts found under "
                f"{self.cfg.results_root}. Run main_search.py / "
                "main_retrain.py / main_deploy.py / main_tracking.py first."
            )

    # ------------------------------------------------------------------
    def _run_statistics(self) -> None:
        s = self._stages
        if "search" in s and s["search"].available:
            try:
                self._stats["search"] = summarize_search(
                    history=s["search"].frames.get("history"),
                    pareto=s["search"].frames.get("pareto"),
                    objectives=list(self.cfg.objectives),
                    generations=s["search"].frames.get("generations"),
                )
            except Exception as exc:  # noqa: BLE001
                self.log.exception("summarize_search failed")
                self._stats["search"] = {"error": str(exc)}
        if "retrain" in s and s["retrain"].available:
            try:
                self._stats["retrain"] = summarize_retrain(
                    metrics=s["retrain"].frames.get("metrics"),
                    ranked=s["retrain"].frames.get("ranked"),
                )
            except Exception as exc:  # noqa: BLE001
                self.log.exception("summarize_retrain failed")
                self._stats["retrain"] = {"error": str(exc)}
        if "deploy" in s and s["deploy"].available:
            try:
                self._stats["deploy"] = summarize_deploy(
                    runtime=s["deploy"].frames.get("runtime"),
                    ranked=s["deploy"].frames.get("ranked"),
                )
            except Exception as exc:  # noqa: BLE001
                self.log.exception("summarize_deploy failed")
                self._stats["deploy"] = {"error": str(exc)}
        if "tracking" in s and s["tracking"].available:
            try:
                self._stats["tracking"] = summarize_tracking(
                    metrics=s["tracking"].frames.get("metrics"),
                    failures=s["tracking"].frames.get("failures"),
                )
            except Exception as exc:  # noqa: BLE001
                self.log.exception("summarize_tracking failed")
                self._stats["tracking"] = {"error": str(exc)}

        # Pairwise comparison: deploy precisions on key metrics.
        deploy_df = s.get("deploy", StageData("deploy", self.cfg.deploy_dir)) \
            .frames.get("runtime")
        if deploy_df is not None and "precision" in deploy_df.columns:
            comparisons: dict[str, dict[str, Any]] = {}
            for col in ("latency_ms_p95", "energy_mj_per_inf",
                        "on_device_auroc"):
                if col not in deploy_df.columns:
                    continue
                groups = {
                    str(p): deploy_df.loc[deploy_df["precision"] == p,
                                          col].dropna().tolist()
                    for p in deploy_df["precision"].dropna().unique()
                }
                groups = {k: v for k, v in groups.items() if len(v) >= 2}
                if len(groups) >= 2:
                    try:
                        comparisons[col] = compare_groups(
                            values_by_group=groups, test="mannwhitney",
                        )
                    except Exception as exc:  # noqa: BLE001
                        comparisons[col] = {"error": str(exc)}
            if comparisons:
                self._stats.setdefault("deploy", {})["comparisons"] = comparisons

    # ------------------------------------------------------------------
    def _generate_plots(self) -> None:
        # --- Pareto front (search-stage) ---------------------------------
        pareto_df = (self._stages.get("search").frames.get("pareto")
                     if "search" in self._stages else None)
        if pareto_df is not None and len(pareto_df):
            out = self.cfg.report_dir / self.cfg.pareto_png
            self._artifacts["pareto_png"] = self._fallback_scatter_pareto(
                pareto_df, out,
            )
            if self._artifacts["pareto_png"]:
                self.log.info("Pareto plot -> %s", out)

        # --- Convergence -------------------------------------------------
        per_gen = (self._stats.get("search", {}) or {}).get("per_generation")
        if per_gen:
            out = self.cfg.report_dir / self.cfg.convergence_png
            self._artifacts["convergence_png"] = (
                self._fallback_convergence(per_gen, out)
            )
            if self._artifacts["convergence_png"]:
                self.log.info("Convergence plot -> %s", out)

        # --- Boxplots (deploy-stage) ------------------------------------
        runtime_df = (self._stages.get("deploy").frames.get("runtime")
                      if "deploy" in self._stages else None)
        if runtime_df is not None and len(runtime_df):
            out = self.cfg.report_dir / self.cfg.boxplots_png
            cols = [c for c in self.cfg.boxplot_columns
                    if c in runtime_df.columns]
            if cols:
                self._artifacts["boxplots_png"] = (
                    self._fallback_boxplots(runtime_df, cols, out)
                )
                if self._artifacts["boxplots_png"]:
                    self.log.info("Boxplot -> %s", out)

    # ------------------------------------------------------------------
    def _generate_latex(self) -> None:
        sections: list[tuple[str, str]] = []
        s = self._stages

        if "search" in s and s["search"].frames.get("pareto") is not None:
            df = s["search"].frames["pareto"]
            cols = [c for c in (["pareto_rank", "individual_index",
                                 "neg_auroc", "latency_ms",
                                 "peak_ram_mb", "energy_mj"])
                    if c in df.columns]
            sections.append((
                "Final Pareto Front",
                self._safe_latex(df.head(self.cfg.top_k_candidates),
                                 cols, "Final Pareto Front (top-K)",
                                 "tab:pareto"),
            ))

        if "retrain" in s and s["retrain"].frames.get("ranked") is not None:
            df = s["retrain"].frames["ranked"]
            cols = [c for c in (["rank", "candidate_id", "score",
                                 "test_auroc", "test_auprc", "test_f1",
                                 "val_auroc", "best_epoch",
                                 "train_seconds"])
                    if c in df.columns]
            sections.append((
                "Retrain Ranking",
                self._safe_latex(df.head(self.cfg.top_k_candidates),
                                 cols, "Full-budget retraining ranking",
                                 "tab:retrain"),
            ))

        if "deploy" in s and s["deploy"].frames.get("ranked") is not None:
            df = s["deploy"].frames["ranked"]
            cols = [c for c in (["rank", "candidate_id", "precision",
                                 "score", "on_device_auroc",
                                 "retrain_test_auroc",
                                 "latency_ms_p95", "throughput_fps",
                                 "energy_mj_per_inf", "peak_ram_mb"])
                    if c in df.columns]
            sections.append((
                "Embedded Deployment Ranking",
                self._safe_latex(df.head(self.cfg.top_k_candidates),
                                 cols, "Embedded ranking after on-device "
                                 "benchmark", "tab:deploy"),
            ))

        if "tracking" in s and s["tracking"].frames.get("metrics") is not None:
            df = s["tracking"].frames["metrics"]
            cols = [c for c in (["scenario", "status", "n_frames",
                                 "achieved_fps",
                                 "detection_latency_ms_p95",
                                 "tracker_success_rate",
                                 "n_lost", "n_recoveries",
                                 "energy_mj_per_frame"])
                    if c in df.columns]
            sections.append((
                "Tracking Validation",
                self._safe_latex(df, cols,
                                 "Closed-loop tracking metrics per scenario",
                                 "tab:tracking"),
            ))

        if not sections:
            self.log.warning("No tables to export — skipping LaTeX bundle.")
            return

        out = self.cfg.report_dir / self.cfg.latex_tex
        out.write_text(self._fallback_latex_bundle(sections),
                       encoding="utf-8")
        self._latex_sections = sections
        self._artifacts["latex_tex"] = out
        self.log.info("LaTeX bundle -> %s", out)

    # ------------------------------------------------------------------
    def _safe_latex(self, df: pd.DataFrame, cols: list[str],
                    caption: str, label: str) -> str:
        try:
            return self._fallback_latex_table(df, cols, caption, label)
        except Exception:  # noqa: BLE001
            self.log.exception("LaTeX table failed for %s", label)
            return f"% table {label} could not be rendered\n"

    # ------------------------------------------------------------------
    def _generate_pdf(self) -> None:
        out = self.cfg.report_dir / self.cfg.summary_pdf
        # The payload is persisted so a proper LaTeX/reportlab template can
        # consume it later; the PDF itself is rendered with PdfPages.
        _write_json_atomic(self.cfg.report_dir / "pdf_payload.json",
                           self._build_pdf_payload())

        # PdfPages: stitch the available PNGs into a single PDF.
        if not _HAVE_MPL:
            self.log.warning("matplotlib unavailable — skipping PDF.")
            return
        images = [p for k in ("pareto_png", "convergence_png", "boxplots_png")
                  for p in [self._artifacts.get(k)] if p and Path(p).is_file()]
        if not images:
            self.log.warning("No plots found — skipping PDF.")
            return
        with PdfPages(str(out)) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.suptitle(self.cfg.title, fontsize=14, y=0.95)
            ax = fig.add_subplot(111)
            ax.axis("off")
            text = "\n".join(self._title_page_lines())
            ax.text(0.05, 0.85, text, fontsize=10,
                    family="monospace", verticalalignment="top")
            pdf.savefig(fig)
            plt.close(fig)
            # One image per page
            for img_path in images:
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_subplot(111)
                ax.axis("off")
                img = plt.imread(str(img_path))
                ax.imshow(img)
                ax.set_title(Path(img_path).stem.replace("_", " ").title(),
                             fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        self._artifacts["summary_pdf"] = out
        self.log.info("Summary PDF (fallback) -> %s", out)

    # ------------------------------------------------------------------
    def _build_pdf_payload(self) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        # 1. Overview
        sections.append({
            "heading": "Overview",
            "text": ("Consolidated experimental report for the NSGA-II "
                     "search, full-budget retraining, embedded deployment, "
                     "and closed-loop tracking validation stages."),
            "tables": [],
            "images": [],
        })
        # 2. Search
        if self._stats.get("search"):
            sections.append({
                "heading": "Search (NSGA-II)",
                "text": json.dumps(self._stats["search"].get("final_pareto",
                                                              {}),
                                   indent=2, default=str),
                "tables": [s for s in self._latex_sections
                           if "Pareto" in s[0]],
                "images": ([str(self._artifacts["pareto_png"])]
                           if self._artifacts.get("pareto_png") else []) +
                          ([str(self._artifacts["convergence_png"])]
                           if self._artifacts.get("convergence_png") else []),
            })
        # 3. Retrain
        if self._stats.get("retrain"):
            sections.append({
                "heading": "Full-budget Retraining",
                "text": json.dumps(self._stats["retrain"], indent=2,
                                   default=str)[:4000],
                "tables": [s for s in self._latex_sections
                           if "Retrain" in s[0]],
                "images": [],
            })
        # 4. Deploy
        if self._stats.get("deploy"):
            sections.append({
                "heading": "Embedded Deployment",
                "text": json.dumps(self._stats["deploy"], indent=2,
                                   default=str)[:4000],
                "tables": [s for s in self._latex_sections
                           if "Deployment" in s[0]],
                "images": ([str(self._artifacts["boxplots_png"])]
                           if self._artifacts.get("boxplots_png") else []),
            })
        # 5. Tracking
        if self._stats.get("tracking"):
            sections.append({
                "heading": "Closed-loop Tracking",
                "text": json.dumps(self._stats["tracking"], indent=2,
                                   default=str)[:4000],
                "tables": [s for s in self._latex_sections
                           if "Tracking" in s[0]],
                "images": [],
            })
        return {
            "title": self.cfg.title,
            "author": self.cfg.author,
            "sections": sections,
        }

    # ------------------------------------------------------------------
    def _title_page_lines(self) -> list[str]:
        lines = [self.cfg.title, "", f"Author: {self.cfg.author}", ""]
        for stage_name, stage in self._stages.items():
            if not stage.available:
                continue
            lines.append(f"[{stage_name}] artifacts under: {stage.path}")
            for k, df in stage.frames.items():
                if df is not None:
                    lines.append(f"  - {k}.csv : "
                                 f"{len(df)} rows, {len(df.columns)} cols")
        return lines

    # ------------------------------------------------------------------
    # Fallback implementations (used only when downstream modules are
    # missing or fail). They keep the script always-deliverable.
    # ------------------------------------------------------------------
    def _fallback_scatter_pareto(self, pareto_df: pd.DataFrame,
                                 out: Path) -> Path | None:
        if not _HAVE_MPL:
            return None
        x_col, y_col = self.cfg.pareto_axes
        if x_col not in pareto_df or y_col not in pareto_df:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pareto_df[x_col], pareto_df[y_col], s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Pareto front (fallback scatter)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _fallback_convergence(self, per_gen: list[dict],
                              out: Path) -> Path | None:
        if not _HAVE_MPL or not per_gen:
            return None
        df = pd.DataFrame(per_gen)
        if "generation" not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        for col in self.cfg.convergence_metrics:
            if col in df.columns:
                ax.plot(df["generation"], df[col], label=col, marker="o")
        ax.set_xlabel("generation")
        ax.set_title("Convergence (fallback)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _fallback_boxplots(self, df: pd.DataFrame, cols: list[str],
                           out: Path) -> Path | None:
        if not _HAVE_MPL:
            return None
        fig, axes = plt.subplots(1, len(cols),
                                 figsize=(3 * len(cols), 4),
                                 sharey=False)
        if len(cols) == 1:
            axes = [axes]
        gby = self.cfg.boxplot_group_by
        for ax, col in zip(axes, cols):
            if gby and gby in df.columns:
                groups = [g[col].dropna().values
                          for _, g in df.groupby(gby)]
                labels = [str(k) for k, _ in df.groupby(gby)]
                ax.boxplot(groups, labels=labels)
            else:
                ax.boxplot([df[col].dropna().values], labels=[col])
            ax.set_title(col, fontsize=10)
        fig.suptitle("On-device metrics (fallback)")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    @staticmethod
    def _fallback_latex_table(df: pd.DataFrame, cols: list[str],
                              caption: str, label: str) -> str:
        try:
            view = df[cols] if cols else df
        except KeyError:
            view = df
        body = view.to_latex(index=False, escape=True, na_rep="--")
        return ("\\begin{table}[h]\n\\centering\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
                f"{body}\n\\end{{table}}\n")

    def _fallback_latex_bundle(self,
                               sections: list[tuple[str, str]]) -> str:
        parts = [
            "% Auto-generated by main_report.py (fallback)",
            f"% Title: {self.cfg.title}",
            f"% Author: {self.cfg.author}", "",
        ]
        for title, body in sections:
            parts.append(f"% ---- {title} ----")
            parts.append(body)
            parts.append("")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.log.info("=== Report generation — START ===")
        self.log.info("Config: %s",
                      json.dumps(self.cfg.to_dict(), indent=2, default=str))

        _write_json_atomic(
            self.cfg.report_dir / "report_config.json",
            self.cfg.to_dict(),
        )

        self._load_all()

        for stage_method in (self._run_statistics,
                             self._generate_plots,
                             self._generate_latex,
                             self._generate_pdf):
            try:
                stage_method()
            except Exception:  # noqa: BLE001
                self.log.exception("%s failed", stage_method.__name__)

        elapsed = time.perf_counter() - t0
        summary = {
            "title": self.cfg.title,
            "report_dir": str(self.cfg.report_dir),
            "artifacts": {
                k: (str(v) if v else None) for k, v in self._artifacts.items()
            },
            "stages_loaded": [n for n, s in self._stages.items()
                              if s.available],
            "stats": self._stats,
            "elapsed_seconds": round(elapsed, 3),
        }
        _write_json_atomic(self.cfg.report_dir / self.cfg.summary_json,
                           summary)
        self.log.info("=== Report generated in %.2f s ===", elapsed)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Generate thesis-ready tables, plots, Pareto curves, "
                     "and a compiled summary PDF from upstream-stage "
                     "artifacts."),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--author", type=str, default=None)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--no-retrain", action="store_true")
    p.add_argument("--no-deploy", action="store_true")
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def _apply_cli_overrides(cfg: ReportConfig,
                         args: argparse.Namespace) -> ReportConfig:
    overrides: dict[str, Any] = {
        "results_root": args.results_root,
        "report_dir":   args.report_dir,
        "title":        args.title,
        "author":       args.author,
        "top_k_candidates": args.top_k,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.no_search:
        cfg.include_search = False
    if args.no_retrain:
        cfg.include_retrain = False
    if args.no_deploy:
        cfg.include_deploy = False
    if args.no_tracking:
        cfg.include_tracking = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = (ReportConfig.from_file(args.config)
           if args.config.is_file() else ReportConfig())
    cfg = _apply_cli_overrides(cfg, args)

    log_path = cfg.report_dir / "report.log"
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    try:
        ReportPipeline(cfg, logger=logger).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Reporting failure: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during reporting")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
