"""
main_retrain.py
===============

Entry-point orchestration script for the *full-budget retraining* stage of
the quantized-NN / MVTec-AD pipeline.

The NSGA-II search (``main_search.py``) uses a low-fidelity training budget
to keep the multi-objective sweep tractable. Once the search produces a
Pareto front, the most promising candidates must be retrained with the full
training budget to obtain definitive AUROC / metric values and a
deployable checkpoint.

Responsibilities
----------------
1. Load the search artifacts (``best_candidates.json`` and/or
   ``pareto_front.csv``) and pick the candidate set to retrain.
2. For every candidate:
   * Build the architecture from its configuration.
   * Wrap it for quantization-aware training (QAT).
   * Run the *full-budget* training loop with validation-driven early stop.
   * Run a final validation pass and a held-out test-metrics pass.
   * Save the best checkpoint under ``checkpoints/final_models/``.
   * Stream a row into ``results/retrain/final_metrics.csv``.
3. After all candidates are processed, rank them by a configurable scalar
   utility (AUROC-dominant by default) and persist
   ``results/retrain/model_ranked.csv``.

Expected module interfaces (downstream contract)
------------------------------------------------
``src.models.model_factory``
    ``build_model(candidate: dict) -> torch.nn.Module``

``src.quantization.qat_wrapper``
    ``wrap_for_qat(model: nn.Module, qconfig: dict) -> nn.Module``

``src.evaluation.train_loop``
    ``train(model: nn.Module,
            candidate: dict,
            splits_dir: Path,
            category: str,
            n_epochs: int,
            optimizer_cfg: dict,
            scheduler_cfg: dict | None,
            device: str,
            seed: int,
            checkpoint_path: Path,
            log_dir: Path | None = None,
            **kwargs) -> dict``
        Trains the model with full budget and val-driven checkpointing.
        Saves the best model at ``checkpoint_path``.
        Returns ``{"best_epoch": int, "best_val": dict,
                   "history": list[dict], "checkpoint_path": str}``.

``src.evaluation.validate``
    ``validate(model: nn.Module,
               splits_dir: Path,
               category: str,
               device: str,
               split: str = "val",
               **kwargs) -> dict``
        Runs a single forward pass over the requested split and returns a
        metrics dict (``loss``, ``auroc``, ...).

``src.evaluation.test_metrics``
    ``compute_test_metrics(model: nn.Module,
                           splits_dir: Path,
                           category: str,
                           device: str,
                           **kwargs) -> dict``
        Computes the held-out test metrics. Returns a dict with at least
        ``auroc``, ``image_auroc``, ``pixel_auroc`` (when applicable),
        ``auprc``, ``f1``.

Assumptions
-----------
- ``main_prepare.py`` and ``main_search.py`` have been executed; the search
  results live under ``results/search/`` and split manifests under
  ``data/splits/``.
- Each search candidate carries (at minimum) keys ``"candidate"`` containing
  the architecture/quantization configuration consumable by ``build_model``
  and ``wrap_for_qat``. The ``qconfig`` is read from
  ``candidate["candidate"]["quantization"]`` if present, else ``{}``.
- The downstream training/evaluation modules own dataloader construction
  (mirroring the convention used by ``FitnessEvaluator``), so this script
  only forwards ``splits_dir`` and ``category``.
- One candidate failure must not abort the batch; partial results are
  always written to disk before the run terminates.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

import torch

# ---------------------------------------------------------------------------
# Project module imports (interfaces declared above).
# ---------------------------------------------------------------------------
from src.models.model_factory import build_model
from src.quantization.qat_wrapper import wrap_for_qat
from src.evaluation.train_loop import train as train_full
from src.evaluation.validate import validate as validate_model
from src.evaluation.test_metrics import compute_test_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_SEARCH_RESULTS: Path = PROJECT_ROOT / "results" / "search"
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "retrain"
DEFAULT_CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints" / "final_models"
DEFAULT_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
# NOTE: the config folder is ``config/`` (singular); ``configs/`` silently
# fell back to built-in defaults.
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config" / "retrain.yaml"

# Columns guaranteed to be present in final_metrics.csv (extra metrics are
# accepted and serialized as JSON).
CORE_METRIC_COLUMNS: tuple[str, ...] = (
    "candidate_id", "checkpoint_path", "status",
    "test_auroc", "test_auprc", "test_f1",
    "val_auroc", "val_loss",
    "best_epoch", "train_seconds",
    "search_neg_auroc", "search_latency_ms",
    "search_peak_ram_mb", "search_energy_mj",
    "extra_metrics", "candidate_config", "error",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrainConfig:
    """Configuration for the full-budget retraining stage."""

    # I/O
    search_results_dir: Path = DEFAULT_SEARCH_RESULTS
    results_dir: Path = DEFAULT_RESULTS_DIR
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    splits_dir: Path = DEFAULT_SPLITS_DIR
    category: str = "bottle"

    # Candidate selection
    candidates_source: str = "best_candidates"  # 'best_candidates' | 'pareto'
    max_candidates: int | None = None           # None -> use all

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"

    # Full-budget training hyperparameters
    n_epochs: int = 200
    optimizer: dict[str, Any] = field(default_factory=lambda: {
        "name": "adam", "lr": 1e-3, "weight_decay": 1e-5,
    })
    scheduler: dict[str, Any] | None = field(default_factory=lambda: {
        "name": "cosine", "T_max": 200, "eta_min": 1e-6,
    })

    # Resume / behavior
    skip_existing: bool = True   # skip candidates whose checkpoint already exists
    fail_fast: bool = False      # stop the batch on the first failed candidate

    # Ranking — weighted scalar utility used to break ties / produce a single
    # ordering. All metrics are normalized via min-max within the cohort
    # before being weighted. AUROC is treated as "higher is better"; latency,
    # RAM, and energy as "lower is better".
    ranking: dict[str, float] = field(default_factory=lambda: {
        "test_auroc":           1.0,
        "search_latency_ms":    -0.20,
        "search_peak_ram_mb":   -0.10,
        "search_energy_mj":     -0.10,
    })

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path) -> "RetrainConfig":
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
    def from_dict(cls, data: dict[str, Any]) -> "RetrainConfig":
        kwargs = dict(data)
        for key in ("search_results_dir", "results_dir",
                    "checkpoints_dir", "splits_dir"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
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
        return d


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging(log_path: Path | None,
                       level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("retrain")
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
# Reproducibility
# ---------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    """Seed all RNG subsystems via the project-wide utility."""
    from src.utils.set_seed import set_seed
    set_seed(seed, deterministic_torch=True)


def _backup_existing(path: Path, logger: logging.Logger) -> None:
    """Rename an existing results file instead of silently truncating it."""
    if path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
        path.rename(backup)
        logger.info("Existing %s backed up to %s", path.name, backup.name)


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------
def _load_candidates(cfg: RetrainConfig,
                     logger: logging.Logger) -> list[dict[str, Any]]:
    """Load and normalize the candidate set from the search artifacts."""
    if cfg.candidates_source == "best_candidates":
        path = cfg.search_results_dir / "best_candidates.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"best_candidates.json not found at {path}. "
                "Run main_search.py first or pass --candidates-source pareto."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        items: Iterable[dict[str, Any]] = payload.get("candidates", [])
        candidates = [_normalize_candidate(it, source=path) for it in items]
    elif cfg.candidates_source == "pareto":
        path = cfg.search_results_dir / "pareto_front.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"pareto_front.csv not found at {path}."
            )
        candidates = _load_pareto_csv(path)
    else:
        raise ValueError(
            f"Unsupported candidates_source: {cfg.candidates_source!r}"
        )

    if cfg.max_candidates is not None:
        candidates = candidates[: cfg.max_candidates]
    logger.info("Loaded %d candidate(s) from %s",
                len(candidates), cfg.candidates_source)
    return candidates


def _normalize_candidate(item: dict[str, Any],
                         source: Path) -> dict[str, Any]:
    """Coerce a heterogeneous record into the internal candidate schema."""
    objectives = item.get("objectives", {}) or {}
    return {
        "candidate_id": _candidate_id_from(item, source),
        "config": item.get("candidate", {}) or {},
        "search_objectives": {
            "neg_auroc":    objectives.get("neg_auroc"),
            "latency_ms":   objectives.get("latency_ms"),
            "peak_ram_mb":  objectives.get("peak_ram_mb"),
            "energy_mj":    objectives.get("energy_mj"),
        },
        "search_metrics": item.get("metrics", {}) or {},
        "raw": item,
    }


def _candidate_id_from(item: dict[str, Any], source: Path) -> str:
    """Build a stable, filesystem-safe identifier for the candidate."""
    if "candidate_id" in item:
        return str(item["candidate_id"])
    rank = item.get("rank")
    idx = item.get("individual_index")
    if rank is not None:
        return f"rank{int(rank):03d}"
    if idx is not None:
        return f"ind{int(idx):05d}"
    # Last-resort hash of the config dict.
    import hashlib
    digest = hashlib.sha1(
        json.dumps(item.get("candidate", {}), sort_keys=True,
                   default=str).encode("utf-8")
    ).hexdigest()[:10]
    return f"cand_{digest}"


def _load_pareto_csv(path: Path) -> list[dict[str, Any]]:
    """Parse ``pareto_front.csv`` produced by main_search.py."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cfg_blob = row.get("candidate") or "{}"
            metrics_blob = row.get("metrics") or "{}"
            try:
                config = json.loads(cfg_blob)
            except json.JSONDecodeError:
                config = {}
            try:
                metrics = json.loads(metrics_blob)
            except json.JSONDecodeError:
                metrics = {}
            item = {
                "rank": int(row.get("pareto_rank", 0) or 0),
                "individual_index": int(row.get("individual_index", -1) or -1),
                "candidate": config,
                "metrics": metrics,
                "objectives": {
                    "neg_auroc":   _maybe_float(row.get("neg_auroc")),
                    "latency_ms":  _maybe_float(row.get("latency_ms")),
                    "peak_ram_mb": _maybe_float(row.get("peak_ram_mb")),
                    "energy_mj":   _maybe_float(row.get("energy_mj")),
                },
            }
            out.append(_normalize_candidate(item, source=path))
    return out


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Metrics CSV writer
# ---------------------------------------------------------------------------
class MetricsWriter:
    """Crash-safe append-only CSV writer for retraining results."""

    def __init__(self, path: Path,
                 columns: tuple[str, ...] = CORE_METRIC_COLUMNS) -> None:
        self.path = path
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(self.columns))
        self._writer.writeheader()
        self._fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        clean: dict[str, Any] = {}
        for col in self.columns:
            v = row.get(col)
            if isinstance(v, (dict, list, tuple)):
                clean[col] = json.dumps(v, default=str)
            else:
                clean[col] = v
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RetrainPipeline:
    """Full-budget retraining of Pareto-optimal candidates."""

    def __init__(self, cfg: RetrainConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("retrain")

        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = cfg.results_dir / "final_metrics.csv"
        self._ranked_path = cfg.results_dir / "model_ranked.csv"

        # Cache previous metrics BEFORE the writer truncates the file, so a
        # re-run with skip_existing=True can carry forward real test metrics
        # instead of producing an empty ranking.
        self._previous: dict[str, dict[str, Any]] = {}
        if self._metrics_path.is_file():
            try:
                with self._metrics_path.open("r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        cid = row.get("candidate_id")
                        if cid and str(row.get("status", "")).startswith("ok"):
                            self._previous[cid] = dict(row)
                self.log.info("Cached %d previous metric row(s) for reuse.",
                              len(self._previous))
            except Exception:  # noqa: BLE001
                self.log.warning("Could not parse previous final_metrics.csv")

        _backup_existing(self._metrics_path, self.log)
        _backup_existing(self._ranked_path, self.log)

        self._writer = MetricsWriter(self._metrics_path)
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _checkpoint_path(self, candidate_id: str) -> Path:
        return self.cfg.checkpoints_dir / f"{candidate_id}.pt"

    # ------------------------------------------------------------------
    def _retrain_one(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Retrain a single candidate end-to-end and return its result row."""
        cid = candidate["candidate_id"]
        config = candidate["config"]
        qconfig = (config.get("quantization", {})
                   if isinstance(config, dict) else {})
        ckpt_path = self._checkpoint_path(cid)

        record: dict[str, Any] = {
            "candidate_id": cid,
            "checkpoint_path": str(ckpt_path),
            "status": "pending",
            "candidate_config": config,
            "search_neg_auroc":   candidate["search_objectives"].get("neg_auroc"),
            "search_latency_ms":  candidate["search_objectives"].get("latency_ms"),
            "search_peak_ram_mb": candidate["search_objectives"].get("peak_ram_mb"),
            "search_energy_mj":   candidate["search_objectives"].get("energy_mj"),
        }

        if self.cfg.skip_existing and ckpt_path.is_file():
            prev = self._previous.get(cid)
            if prev is not None:
                # Carry forward the previously measured metrics so the
                # ranking still sees this candidate.
                for col in ("test_auroc", "test_auprc", "test_f1",
                            "val_auroc", "val_loss", "best_epoch",
                            "train_seconds", "extra_metrics"):
                    record[col] = prev.get(col)
                record["status"] = "ok_cached"
                self.log.info("[%s] checkpoint exists — reusing previous "
                              "metrics (ok_cached)", cid)
            else:
                record["status"] = "skipped_existing"
                self.log.warning(
                    "[%s] checkpoint exists but no previous metrics found — "
                    "candidate will be EXCLUDED from the ranking. Use "
                    "--no-skip-existing to re-train it.", cid)
            return record

        self.log.info("[%s] building model + applying QAT wrapper", cid)
        try:
            model = build_model(config)
            model = wrap_for_qat(model, qconfig)
            model = model.to(self.cfg.device)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] failed during model construction", cid)
            record.update(status="failed_build", error=str(exc))
            return record

        # ---- training -----------------------------------------------------
        t0 = time.perf_counter()
        try:
            train_result = train_full(
                model=model,
                candidate=config,
                splits_dir=self.cfg.splits_dir,
                category=self.cfg.category,
                n_epochs=self.cfg.n_epochs,
                optimizer_cfg=self.cfg.optimizer,
                scheduler_cfg=self.cfg.scheduler,
                device=self.cfg.device,
                seed=self.cfg.seed,
                checkpoint_path=ckpt_path,
                log_dir=self.cfg.results_dir / "logs" / cid,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] training failed", cid)
            record.update(status="failed_train",
                          train_seconds=round(time.perf_counter() - t0, 3),
                          error=str(exc))
            return record
        record["train_seconds"] = round(time.perf_counter() - t0, 3)
        record["best_epoch"] = train_result.get("best_epoch")

        # ---- reload best checkpoint -------------------------------------
        if ckpt_path.is_file():
            try:
                state = torch.load(ckpt_path, map_location=self.cfg.device)
                state_dict = state.get("model_state_dict", state)
                model.load_state_dict(state_dict, strict=False)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("[%s] could not reload checkpoint: %s",
                                 cid, exc)

        # ---- final validation -------------------------------------------
        try:
            val_metrics = validate_model(
                model=model,
                splits_dir=self.cfg.splits_dir,
                category=self.cfg.category,
                device=self.cfg.device,
                split="val",
            )
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] validation failed", cid)
            record.update(status="failed_val", error=str(exc))
            return record
        record["val_auroc"] = val_metrics.get("auroc")
        record["val_loss"] = val_metrics.get("loss")

        # ---- test metrics -----------------------------------------------
        try:
            test_metrics = compute_test_metrics(
                model=model,
                splits_dir=self.cfg.splits_dir,
                category=self.cfg.category,
                device=self.cfg.device,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] test metrics failed", cid)
            record.update(status="failed_test", error=str(exc))
            return record
        record["test_auroc"] = test_metrics.get("auroc")
        record["test_auprc"] = test_metrics.get("auprc")
        record["test_f1"]    = test_metrics.get("f1")

        # Stash any extra metrics for completeness without expanding columns.
        known = {"auroc", "auprc", "f1", "loss"}
        record["extra_metrics"] = {
            k: v for k, v in {**val_metrics, **test_metrics}.items()
            if k not in known
        }
        record["status"] = "ok"

        self.log.info(
            "[%s] DONE — test AUROC=%.4f, val AUROC=%.4f, "
            "best_epoch=%s, train_s=%.1f",
            cid,
            float(record["test_auroc"] or float("nan")),
            float(record["val_auroc"] or float("nan")),
            record.get("best_epoch"),
            record["train_seconds"],
        )
        return record

    # ------------------------------------------------------------------
    def _rank_models(self, records: list[dict[str, Any]]) -> None:
        """Compute a weighted scalar score and write ``model_ranked.csv``."""
        successful = [r for r in records
                      if str(r.get("status", "")).startswith("ok")]
        if not successful:
            self.log.warning("No successful retrains — skipping ranking")
            self._ranked_path.write_text(
                "candidate_id,score,status\n", encoding="utf-8"
            )
            return

        weights = self.cfg.ranking
        keys = list(weights.keys())

        # Min-max normalize each ranked metric within the cohort.
        norm: dict[str, list[float]] = {}
        for k in keys:
            values = [self._safe_float(r.get(k)) for r in successful]
            valid = [v for v in values if v is not None]
            if not valid:
                norm[k] = [0.0] * len(successful)
                continue
            vmin, vmax = min(valid), max(valid)
            span = vmax - vmin or 1.0
            norm[k] = [
                ((v - vmin) / span) if v is not None else 0.0
                for v in values
            ]

        scored: list[tuple[float, dict[str, Any]]] = []
        for i, rec in enumerate(successful):
            score = sum(weights[k] * norm[k][i] for k in keys)
            rec_copy = dict(rec)
            rec_copy["score"] = float(score)
            scored.append((score, rec_copy))

        scored.sort(key=lambda x: x[0], reverse=True)

        fieldnames = (["rank", "score"]
                      + [c for c in CORE_METRIC_COLUMNS if c != "extra_metrics"])
        with self._ranked_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for rank, (_, rec) in enumerate(scored):
                row = {"rank": rank, "score": round(rec["score"], 6)}
                for col in fieldnames:
                    if col in ("rank", "score"):
                        continue
                    v = rec.get(col)
                    if isinstance(v, (dict, list, tuple)):
                        v = json.dumps(v, default=str)
                    row[col] = v
                w.writerow(row)
        self.log.info("Wrote ranking for %d model(s) to %s",
                      len(scored), self._ranked_path)

    @staticmethod
    def _safe_float(v: Any) -> float | None:
        try:
            if v is None or v == "":
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute the retraining pipeline end-to-end."""
        t0 = time.perf_counter()
        self.log.info("=== Full-budget retraining — START ===")
        self.log.info("Config: %s",
                      json.dumps(self.cfg.to_dict(), indent=2, default=str))

        (self.cfg.results_dir / "retrain_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        _seed_everything(self.cfg.seed)
        candidates = _load_candidates(self.cfg, self.log)

        try:
            for i, cand in enumerate(candidates):
                self.log.info("---- Candidate %d/%d (%s) ----",
                              i + 1, len(candidates), cand["candidate_id"])
                try:
                    record = self._retrain_one(cand)
                except KeyboardInterrupt:
                    raise
                except Exception:  # noqa: BLE001
                    record = {
                        "candidate_id": cand["candidate_id"],
                        "status": "failed_unhandled",
                        "error": traceback.format_exc(limit=3),
                        "candidate_config": cand.get("config", {}),
                    }
                    self.log.exception("Unhandled error retraining %s",
                                       cand["candidate_id"])
                self._records.append(record)
                self._writer.write(record)
                # Skipped/cached candidates are not failures.
                if self.cfg.fail_fast and not str(
                        record.get("status", "")).startswith(("ok", "skipped")):
                    self.log.error(
                        "fail_fast=True and candidate %s failed (%s) — aborting.",
                        cand["candidate_id"], record.get("status"),
                    )
                    break
        finally:
            self._writer.close()

        self._rank_models(self._records)

        elapsed = time.perf_counter() - t0
        n_ok = sum(1 for r in self._records
                   if str(r.get("status", "")).startswith("ok"))
        summary = {
            "n_candidates": len(candidates),
            "n_ok": n_ok,
            "n_failed": len(self._records) - n_ok,
            "elapsed_seconds": round(elapsed, 3),
            "final_metrics_csv": str(self._metrics_path),
            "model_ranked_csv": str(self._ranked_path),
            "checkpoints_dir": str(self.cfg.checkpoints_dir),
        }
        (self.cfg.results_dir / "retrain_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self.log.info("=== Retraining finished in %.2f s — ok=%d/%d ===",
                      elapsed, n_ok, len(candidates))
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Retrain Pareto-optimal candidates from main_search.py "
                     "with the full training budget."),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--search-results-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--candidates-source", type=str, default=None,
                   choices=[None, "best_candidates", "pareto"])
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   choices=[None, "cpu", "cuda"])
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-train even if a checkpoint already exists.")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _apply_cli_overrides(cfg: RetrainConfig,
                         args: argparse.Namespace) -> RetrainConfig:
    overrides: dict[str, Any] = {
        "search_results_dir": args.search_results_dir,
        "results_dir":        args.results_dir,
        "checkpoints_dir":    args.checkpoints_dir,
        "splits_dir":         args.splits_dir,
        "category":           args.category,
        "candidates_source":  args.candidates_source,
        "max_candidates":     args.max_candidates,
        "n_epochs":           args.n_epochs,
        "seed":               args.seed,
        "device":             args.device,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.no_skip_existing:
        cfg.skip_existing = False
    if args.fail_fast:
        cfg.fail_fast = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = (RetrainConfig.from_file(args.config)
           if args.config.is_file() else RetrainConfig())
    cfg = _apply_cli_overrides(cfg, args)

    log_path = cfg.results_dir / "retrain.log"
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    try:
        RetrainPipeline(cfg, logger=logger).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Retrain failure: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during retraining")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
