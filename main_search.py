"""
main_search.py
==============

Entry-point orchestration script for the NSGA-II multi-objective search over
neural-architecture and quantization parameters, targeting MVTec AD anomaly
detection on embedded GPU devices (Jetson Orin Nano).

Responsibilities
----------------
1. Load search configuration (YAML/JSON) and seed the RNG state.
2. Build the search space and genotype/phenotype encoder.
3. Construct the fitness evaluator that wires together: model factory →
   QAT wrapper → latency / RAM / energy profilers → AUROC evaluator.
4. Run the NSGA-II engine for ``n_generations`` with population checkpointing.
5. Persist per-generation populations, the global evaluation history, the
   final Pareto front, and the top-K best candidates.

Optimization objectives (all minimized internally)
--------------------------------------------------
    f1 = -AUROC          (maximize AUROC)
    f2 =  latency_ms     (minimize)
    f3 =  peak_ram_mb    (minimize)
    f4 =  energy_mj      (minimize)

Expected module interfaces (downstream contract)
------------------------------------------------
``src.nas.search_space``
    ``class SearchSpace``
        ``__init__(self, spec: dict)``
        ``sample(self, rng: np.random.Generator) -> dict``        # candidate dict
        ``bounds(self) -> dict[str, Any]``
        ``validate(self, candidate: dict) -> bool``
        ``n_variables(self) -> int``

``src.nas.encoding``
    ``class Encoder``
        ``__init__(self, search_space: SearchSpace)``
        ``decode(self, genome: np.ndarray) -> dict``
        ``encode(self, candidate: dict) -> np.ndarray``
        ``random_genome(self, rng: np.random.Generator) -> np.ndarray``
        ``genome_length: int``
        ``variable_types: list[str]``                              # 'int' | 'float' | 'cat'
        ``lower: np.ndarray``
        ``upper: np.ndarray``

``src.nas.fitness``
    ``class FitnessEvaluator``
        Composes model_factory, qat_wrapper, profilers, AUROC eval.
        ``__call__(self, candidate: dict) -> dict``
            returns {"objectives": tuple[float, ...],   # (-auroc, lat, ram, energy)
                     "metrics":    dict,                # raw measurements
                     "valid":      bool,
                     "error":      str | None,
                     "candidate":  dict}

``src.nas.nsga2_engine``
    ``class NSGA2Engine``
        ``__init__(self, encoder, evaluator, population_size, n_objectives,
                   crossover_prob, mutation_prob, eta_c, eta_m, seed,
                   n_workers=1)``
        ``run(self, n_generations: int,
              on_generation_end: callable | None = None,
              initial_population: np.ndarray | None = None) -> dict``
            Returns ``{"history": list[dict],
                       "final_population": np.ndarray,
                       "final_objectives": np.ndarray,
                       "final_meta":       list[dict]}``

``src.nas.pareto``
    ``compute_pareto_front(objectives: np.ndarray) -> np.ndarray``  # indices
    ``crowding_distance(front_objectives: np.ndarray) -> np.ndarray``

``src.models.model_factory``        ``build_model(candidate: dict) -> torch.nn.Module``
``src.quantization.qat_wrapper``    ``wrap_for_qat(model, qconfig: dict) -> torch.nn.Module``
                                    ``calibrate_and_finetune(model, ...) -> torch.nn.Module``
``src.profiling.latency_meter``     ``measure_latency(model, input_shape, device, **kw) -> dict``
``src.profiling.ram_meter``         ``measure_peak_ram(model, input_shape, device, **kw) -> dict``
``src.profiling.energy_meter``      ``measure_energy(model, input_shape, device, **kw) -> dict``
``src.evaluation.auroc_eval``       ``evaluate_auroc(model, dataloader, device) -> dict``

Assumptions
-----------
- ``main_prepare.py`` has already been executed; ``data/splits/`` contains the
  per-category split manifests required by the dataloaders constructed inside
  ``FitnessEvaluator``.
- The fitness evaluator owns dataloader construction so the search loop stays
  agnostic to dataset specifics.
- Configuration is supplied via a YAML file (default ``config/search.yaml``);
  CLI flags override individual fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# Project module imports (interfaces declared in the docstring above).
# ---------------------------------------------------------------------------
from src.nas.search_space import SearchSpace, SearchSpaceConfig, ArchSearchConfig, QuantSearchConfig
from src.nas.encoding import GenomeEncoder as Encoder
from src.nas.nsga2_engine import NSGA2Engine, NSGA2Config
from src.nas.fitness import FitnessEvaluator, FitnessConfig, PenaltyReason
from src.nas.pareto import pareto_front as compute_pareto_front, crowding_distance
from src.models.model_factory import build_model
from src.quantization.qat_wrapper import wrap_for_qat
from src.profiling.latency_meter import measure_latency
from src.profiling.ram_meter import measure_peak_ram
from src.profiling.energy_meter import measure_energy
from src.evaluation.auroc_eval import evaluate_auroc
from src.utils.set_seed import set_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# NOTE: the config folder is ``config/`` (singular). A previous version
# pointed at ``configs/`` and silently fell back to built-in defaults.
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config" / "search.yaml"
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "search"
DEFAULT_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"

OBJECTIVE_NAMES: tuple[str, ...] = ("neg_auroc", "latency_ms",
                                    "peak_ram_mb", "energy_mj")
N_OBJECTIVES: int = len(OBJECTIVE_NAMES)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class SearchConfig:
    """Top-level configuration for the NSGA-II search stage."""

    # I/O
    splits_dir: Path = DEFAULT_SPLITS_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    category: str = "bottle"

    # Reproducibility
    seed: int = 42

    # NSGA-II hyperparameters
    population_size: int = 32
    n_generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float | None = None  # None -> 1/genome_length
    eta_c: float = 15.0                 # SBX index (reserved; engine uses single_point/uniform)
    eta_m: float = 20.0                 # polynomial mutation index (reserved)
    tournament_size: int = 2

    # Hypervolume reference point (order: neg_auroc, latency, ram, energy).
    # None -> derived from the fitness penalty values so that every feasible
    # solution strictly dominates the reference point.
    hv_reference: list[float] | None = None

    # Search space + evaluator specs (free-form dicts, validated downstream)
    search_space: dict[str, Any] = field(default_factory=dict)
    fitness: dict[str, Any] = field(default_factory=dict)

    # Compute / runtime
    device: str = "cuda"
    n_workers: int = 1                  # parallel evaluations
    top_k: int = 5                      # how many "best" candidates to dump

    # Resume / checkpointing
    resume_from: Path | None = None     # pickle/npz with population genomes
    force_finalize: bool = False        # overwrite final artifacts on a 0-gen resume

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path) -> "SearchConfig":
        """Load a SearchConfig from a YAML or JSON file."""
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
    def from_dict(cls, data: dict[str, Any]) -> "SearchConfig":
        """Build a SearchConfig from a plain dict, coercing path fields.

        Accepts both the flat schema (fields of this dataclass at top level)
        and the legacy nested schema (``algorithm:``, ``operators:``,
        ``hypervolume:``, ``parallelism:``), which is flattened here so that
        an older YAML cannot silently be ignored again.
        """
        kwargs = dict(data)

        # ---- Legacy nested-schema support ------------------------------
        algo = kwargs.pop("algorithm", None) or {}
        for key in ("seed", "population_size", "n_generations"):
            if key in algo:
                kwargs.setdefault(key, algo[key])
        if algo.get("warm_start_checkpoint"):
            kwargs.setdefault("resume_from", algo["warm_start_checkpoint"])

        ops = kwargs.pop("operators", None) or {}
        xover = ops.get("crossover", {}) or {}
        mut = ops.get("mutation", {}) or {}
        if "probability" in xover:
            kwargs.setdefault("crossover_prob", xover["probability"])
        if "eta" in xover:
            kwargs.setdefault("eta_c", xover["eta"])
        if "probability" in mut:
            kwargs.setdefault("mutation_prob", mut["probability"])
        if "eta" in mut:
            kwargs.setdefault("eta_m", mut["eta"])
        if "tournament_size" in ops:
            kwargs.setdefault("tournament_size", ops["tournament_size"])

        hv = kwargs.pop("hypervolume", None) or {}
        if "reference_point" in hv:
            kwargs.setdefault("hv_reference", hv["reference_point"])

        par = kwargs.pop("parallelism", None) or {}
        if "n_workers" in par:
            kwargs.setdefault("n_workers", par["n_workers"])
        # -----------------------------------------------------------------

        for key in ("splits_dir", "results_dir", "resume_from"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        # Drop unknown keys into ``extra`` for forward compatibility.
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
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
    """Configure stdout + file logging for the search run."""
    logger = logging.getLogger()
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
# CSV / JSON helpers
# ---------------------------------------------------------------------------
class HistoryWriter:
    """Append-only CSV writer for the global evaluation history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh,
                                          fieldnames=list(row.keys()))
            self._writer.writeheader()
        # Coerce non-scalar values to JSON for round-trippable storage.
        clean = {k: (json.dumps(v, default=str)
                     if isinstance(v, (dict, list, tuple)) else v)
                 for k, v in row.items()}
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _write_population_csv(path: Path,
                          gen: int,
                          objectives: np.ndarray,
                          meta: list[dict[str, Any]]) -> None:
    """Persist a single generation's population to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["generation", "individual"]
                  + list(OBJECTIVE_NAMES)
                  + ["valid", "error", "candidate", "metrics"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, (obj, m) in enumerate(zip(objectives, meta)):
            row = {
                "generation": gen,
                "individual": i,
                "valid": m.get("valid", True),
                "error": m.get("error"),
                "candidate": json.dumps(m.get("candidate", {}), default=str),
                "metrics": json.dumps(m.get("metrics", {}), default=str),
            }
            for j, name in enumerate(OBJECTIVE_NAMES):
                row[name] = float(obj[j]) if obj is not None else None
            w.writerow(row)


def _write_pareto_csv(path: Path,
                      objectives: np.ndarray,
                      meta: list[dict[str, Any]],
                      indices: np.ndarray) -> None:
    """Persist the Pareto front (and crowding distance) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    front_obj = objectives[indices]
    cd = crowding_distance(front_obj) if len(indices) else np.array([])
    fieldnames = (["pareto_rank", "individual_index", "crowding_distance"]
                  + list(OBJECTIVE_NAMES)
                  + ["candidate", "metrics"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rank, (idx, obj) in enumerate(zip(indices, front_obj)):
            m = meta[idx]
            row = {
                "pareto_rank": rank,
                "individual_index": int(idx),
                "crowding_distance": float(cd[rank]) if len(cd) else None,
                "candidate": json.dumps(m.get("candidate", {}), default=str),
                "metrics": json.dumps(m.get("metrics", {}), default=str),
            }
            for j, name in enumerate(OBJECTIVE_NAMES):
                row[name] = float(obj[j])
            w.writerow(row)


def _write_best_candidates_json(path: Path,
                                objectives: np.ndarray,
                                meta: list[dict[str, Any]],
                                pareto_idx: np.ndarray,
                                top_k: int) -> None:
    """Persist top-K Pareto candidates.

    Ranking: normalised Euclidean distance to the ideal point, computed only
    over *operative* objectives (columns with non-zero spread). A measurement
    channel that returned its penalty value for every individual (e.g. energy
    on a platform without a power sensor) is thereby excluded instead of
    silently distorting — or being ignored by — the selection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pareto_idx) == 0:
        path.write_text(json.dumps({"candidates": []}, indent=2),
                        encoding="utf-8")
        return
    front = objectives[pareto_idx]
    span = front.max(axis=0) - front.min(axis=0)
    active = span > 1e-12
    if not active.any():                      # degenerate: all columns constant
        active = np.ones(front.shape[1], dtype=bool)
        span = np.ones(front.shape[1])
    norm = (front[:, active] - front[:, active].min(axis=0)) / span[active]
    dist = np.sqrt((norm ** 2).sum(axis=1))
    order = np.argsort(dist, kind="stable")
    chosen = pareto_idx[order][:top_k]
    payload = {
        "objective_names": list(OBJECTIVE_NAMES),
        "ranking": ("normalised Euclidean distance to the ideal point over "
                    "operative objectives"),
        "active_objectives": [n for n, a in zip(OBJECTIVE_NAMES, active) if a],
        "candidates": [
            {
                "rank": r,
                "individual_index": int(idx),
                "objectives": {n: float(objectives[idx, j])
                               for j, n in enumerate(OBJECTIVE_NAMES)},
                "candidate": meta[idx].get("candidate", {}),
                "metrics": meta[idx].get("metrics", {}),
            }
            for r, idx in enumerate(chosen)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class SearchPipeline:
    """Orchestrates an NSGA-II search over architecture + quantization."""

    def __init__(self, cfg: SearchConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("search")
        self.cfg.results_dir.mkdir(parents=True, exist_ok=True)

        # Seed *all* RNG subsystems (python/numpy/torch/cuda), not only numpy:
        # the fitness evaluator trains models, so torch must be seeded too.
        seed_state = set_seed(cfg.seed, deterministic_torch=True)
        try:
            (self.cfg.results_dir / "seed_state.json").write_text(
                json.dumps(asdict(seed_state), indent=2, default=str),
                encoding="utf-8")
        except TypeError:  # set_seed returned a plain dict
            (self.cfg.results_dir / "seed_state.json").write_text(
                json.dumps(seed_state, indent=2, default=str),
                encoding="utf-8")

        self._rng = np.random.default_rng(cfg.seed)
        self._history_writer = HistoryWriter(
            self.cfg.results_dir / "history.csv"
        )
        self._global_eval_idx = 0
        self._current_gen = 0

        # Real per-candidate evaluation records, keyed by candidate hash.
        # Populated by the FitnessEvaluator callback so that population CSVs
        # and the Pareto front carry *measured* metrics, not placeholders.
        self._results_by_key: dict[str, dict[str, Any]] = {}
        self._encoder: Encoder | None = None
        self._fitness_config: FitnessConfig | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_key(candidate: dict[str, Any]) -> str:
        blob = json.dumps(candidate, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def _on_eval_result(self, result: Any, candidate: dict[str, Any]) -> None:
        """FitnessEvaluator callback — persist every *real* evaluation.

        This is the traceability contract of the testbed: each row in
        ``history.csv`` corresponds to one actual pipeline execution
        (build → QAT → train → AUROC → hardware profiling).
        """
        reason = getattr(result, "penalty_reason", PenaltyReason.NONE)
        # Profiling failures keep a real AUROC → usable but flagged.
        valid = reason in (PenaltyReason.NONE, PenaltyReason.PROFILING_FAILED)
        metrics = result.to_dict() if hasattr(result, "to_dict") else {}
        error = getattr(result, "error_message", "") or None

        self._results_by_key[self._candidate_key(candidate)] = {
            "valid": valid,
            "error": error,
            "metrics": metrics,
        }

        row: dict[str, Any] = {
            "eval_id": self._global_eval_idx,
            "generation": self._current_gen,
            "valid": valid,
            "penalty_reason": getattr(reason, "value", str(reason)),
            "neg_auroc": -float(getattr(result, "auroc", 0.0)),
            "latency_ms": float(getattr(result, "latency_ms", float("nan"))),
            "peak_ram_mb": float(getattr(result, "peak_ram_mb", float("nan"))),
            "energy_mj": float(getattr(result, "energy_mj", float("nan"))),
            "profiling_ok": bool(getattr(result, "profiling_ok", False)),
            "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
            "error": error,
            "candidate": candidate,
            "metrics": metrics,
        }
        self._history_writer.write(row)
        self._global_eval_idx += 1

    # ------------------------------------------------------------------
    def _population_meta(self,
                         population: np.ndarray) -> list[dict[str, Any]]:
        """Attach real evaluation records to each individual in a population."""
        meta: list[dict[str, Any]] = []
        for gene in population:
            cand = self._encoder.gene_to_dict(gene)
            rec = self._results_by_key.get(self._candidate_key(cand))
            if rec is None:
                rec = {"valid": True, "error": "no evaluation record found",
                       "metrics": {}}
            meta.append({**rec, "candidate": cand})
        return meta

    # ------------------------------------------------------------------
    def _build_components(self) -> tuple[Encoder, FitnessEvaluator]:
        """Instantiate search space, encoder, and fitness evaluator."""
        self.log.info("Building search space and encoder")
        if self.cfg.search_space:
            arch_cfg = ArchSearchConfig(**self.cfg.search_space.get("arch", {}))
            quant_cfg = QuantSearchConfig(**self.cfg.search_space.get("quant", {}))
            space_config = SearchSpaceConfig(arch=arch_cfg, quant=quant_cfg)
        else:
            space_config = None
        space = SearchSpace(config=space_config)
        encoder = Encoder(search_space=space)

        # Device sanity check: fall back to CPU explicitly (and loudly)
        # instead of relying on downstream silent fallbacks.
        device = self.cfg.device
        if device == "cuda" and (torch is None or not torch.cuda.is_available()):
            self.log.warning("CUDA requested but not available — "
                             "falling back to device='cpu'.")
            device = "cpu"
            self.cfg.device = device

        if self.cfg.n_workers > 1:
            self.log.warning("n_workers=%d requested, but the NSGA-II engine "
                             "evaluates candidates serially; running with 1.",
                             self.cfg.n_workers)

        self.log.info("Building fitness evaluator (device=%s)", device)
        fitness_cfg_dict = {
            "splits_dir": str(self.cfg.splits_dir),
            "category": self.cfg.category,
            "device": device,
            **self.cfg.fitness
        }
        # Filter kwargs to match FitnessConfig fields
        from dataclasses import fields
        valid_keys = {f.name for f in fields(FitnessConfig)}
        unknown = set(fitness_cfg_dict) - valid_keys
        if unknown:
            self.log.warning("Ignoring unknown fitness config keys: %s",
                             sorted(unknown))
        filtered_cfg = {k: v for k, v in fitness_cfg_dict.items() if k in valid_keys}
        fitness_config = FitnessConfig(**filtered_cfg)
        self._fitness_config = fitness_config
        evaluator = FitnessEvaluator(
            config=fitness_config,
            extra_callbacks=[self._on_eval_result],
        )
        return encoder, evaluator

    # ------------------------------------------------------------------
    def _load_initial_population(self,
                                 encoder: Encoder) -> np.ndarray | None:
        """Load a checkpointed population if ``resume_from`` is provided."""
        if self.cfg.resume_from is None:
            return None
        path = self.cfg.resume_from
        if not path.is_file():
            self.log.warning("resume_from=%s not found; starting fresh", path)
            return None
        self.log.info("Resuming initial population from %s", path)
        if path.suffix == ".npz":
            data = np.load(path)
            return data["population"]
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return np.asarray(data["population"], dtype=float)
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                genomes = []
                for row in reader:
                    c_dict = json.loads(row["candidate"])
                    genomes.append(encoder.encode(c_dict))
            return np.vstack(genomes)
        raise ValueError(f"Unsupported resume format: {path.suffix}")

    # ------------------------------------------------------------------
    def _on_generation_end(self,
                           gen: int,
                           result: Any,
                           engine: NSGA2Engine) -> None:
        """Callback invoked by the NSGA-II engine after each generation.

        Per-individual measurements are *not* fabricated here: they come from
        the real evaluation records captured by :meth:`_on_eval_result`.
        (Per-evaluation rows are appended to ``history.csv`` by that callback
        at evaluation time, so nothing is duplicated here.)
        """
        self._current_gen = gen
        population = engine.population
        objectives = engine.objectives
        meta = self._population_meta(population)

        # Per-generation population CSV (with real metrics attached).
        gen_path = self.cfg.results_dir / f"population_gen_{gen:03d}.csv"
        _write_population_csv(gen_path, gen, objectives, meta)

        # Per-generation convergence statistics (hypervolume, Pareto size…).
        gen_stats_path = self.cfg.results_dir / "generations.csv"
        stats_row = {
            "generation": gen,
            "n_pareto": getattr(result, "n_pareto", None),
            "hypervolume": getattr(result, "hypervolume", None),
            "best_auroc": getattr(result, "best_auroc", None),
            "best_latency_ms": getattr(result, "best_latency_ms", None),
            "best_ram_mb": getattr(result, "best_ram_mb", None),
            "best_energy_mj": getattr(result, "best_energy_mj", None),
            "mean_auroc": getattr(result, "mean_auroc", None),
            "n_failed_evals": getattr(result, "n_failed_evals", None),
            "elapsed_seconds": getattr(result, "elapsed_seconds", None),
        }
        write_header = not gen_stats_path.is_file()
        with gen_stats_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stats_row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(stats_row)

        self.log.info(
            "Gen %03d done — best AUROC=%.4f, min latency=%.2f ms, "
            "valid=%d/%d",
            gen, result.best_auroc, result.best_latency_ms,
            len(meta) - result.n_failed_evals, len(meta),
        )

        # Auto-checkpointing for easy resume
        ckpt_path = self.cfg.results_dir / "latest_checkpoint"
        engine.checkpoint(ckpt_path)

    # ------------------------------------------------------------------
    def _finalize(self,
                  final_population: np.ndarray,
                  final_objectives: np.ndarray,
                  final_meta: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute Pareto front and persist final artifacts."""
        valid_mask = np.array([m.get("valid", True) for m in final_meta],
                              dtype=bool)
        if not valid_mask.any():
            self.log.error("No valid individuals at end of search.")
            return {"pareto_indices": [], "n_valid": 0}

        valid_idx = np.where(valid_mask)[0]
        valid_obj = final_objectives[valid_idx]

        # Detect inoperative objectives: a column that is constant across all
        # valid individuals never influenced dominance. The typical cause is a
        # measurement backend returning its penalty value for every candidate
        # (e.g. energy_mj = penalty on a PC without a power sensor).
        inoperative: list[str] = []
        if len(valid_idx) > 1:
            span = valid_obj.max(axis=0) - valid_obj.min(axis=0)
            inoperative = [n for n, s in zip(OBJECTIVE_NAMES, span)
                           if s <= 1e-12]
            for name in inoperative:
                self.log.warning(
                    "Objective %r was CONSTANT across all valid individuals — "
                    "it did not influence the search. Check the corresponding "
                    "measurement backend before claiming multi-objective "
                    "results on this axis.", name)

        local_pareto = compute_pareto_front(valid_obj)
        pareto_idx = valid_idx[local_pareto]

        pareto_path = self.cfg.results_dir / "pareto_front.csv"
        _write_pareto_csv(pareto_path, final_objectives, final_meta, pareto_idx)
        self.log.info("Pareto front (%d points) written to %s",
                      len(pareto_idx), pareto_path)

        best_path = self.cfg.results_dir / "best_candidates.json"
        _write_best_candidates_json(best_path, final_objectives, final_meta,
                                    pareto_idx, top_k=self.cfg.top_k)
        self.log.info("Top-%d candidates written to %s",
                      self.cfg.top_k, best_path)

        # Persist the final genome population for potential resume.
        np.savez(self.cfg.results_dir / "final_population.npz",
                 population=final_population,
                 objectives=final_objectives)

        return {"pareto_indices": pareto_idx.tolist(),
                "n_valid": int(valid_mask.sum()),
                "n_invalid": int((~valid_mask).sum()),
                "inoperative_objectives": inoperative}

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute the full search loop and return a summary dict."""
        t0 = time.perf_counter()
        self.log.info("=== NSGA-II search — START ===")
        self.log.info("Config: %s", json.dumps(self.cfg.to_dict(),
                                               default=str, indent=2))

        # Persist the resolved config alongside the results.
        (self.cfg.results_dir / "search_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        encoder, evaluator = self._build_components()
        self._encoder = encoder
        mutation_prob = (self.cfg.mutation_prob
                         if self.cfg.mutation_prob is not None
                         else 1.0 / max(encoder.search_space.genome_length, 1))
        self.log.info("Genome length=%d, mutation_prob=%.4f",
                      encoder.search_space.genome_length, mutation_prob)

        # Align the engine's penalty vector and hypervolume reference with
        # the fitness penalty values. The engine defaults (e.g. 1000 mJ) are
        # NOT dominated by a penalised fitness value (9999 mJ), which would
        # corrupt the hypervolume indicator.
        fc = self._fitness_config
        penalty_vec = [-fc.penalty_auroc, fc.penalty_latency_ms,
                       fc.penalty_ram_mb, fc.penalty_energy_mj]
        hv_ref = (list(self.cfg.hv_reference)
                  if self.cfg.hv_reference is not None else list(penalty_vec))
        self.log.info("HV reference=%s, penalty=%s", hv_ref, penalty_vec)

        nsga_cfg = NSGA2Config(
            population_size=self.cfg.population_size,
            n_objectives=N_OBJECTIVES,
            crossover_prob=self.cfg.crossover_prob,
            mutation_rate=mutation_prob,
            tournament_size=self.cfg.tournament_size,
            hv_reference=hv_ref,
            penalty_objectives=penalty_vec,
            seed=self.cfg.seed,
        )

        engine = NSGA2Engine(
            config=nsga_cfg,
            search_space=encoder.search_space,
            fitness_fn=evaluator,
        )

        try:
            resume_path = self.cfg.resume_from
            if resume_path is not None and resume_path.with_suffix(".json").is_file():
                self.log.info("Resuming full checkpoint from %s", resume_path)
                engine.load_checkpoint(resume_path.with_suffix(""))
            else:
                initial_pop = self._load_initial_population(encoder)
                if initial_pop is not None:
                    engine._population = initial_pop
                    engine._objectives = engine._evaluate_population(initial_pop)
                    engine._generation = 0
            
            remaining_generations = max(0, self.cfg.n_generations - engine.generation)
            self.log.info(
                "Current generation: %d. Target generations: %d. Remaining to run: %d",
                engine.generation, self.cfg.n_generations, remaining_generations
            )

            # Guard: a resume that has nothing left to run must NOT silently
            # overwrite final artifacts produced by the real search (this
            # previously replaced a full run's Pareto front with a 0.02 s
            # no-op re-finalisation lacking all measured metrics).
            pareto_exists = (self.cfg.results_dir / "pareto_front.csv").is_file()
            if (remaining_generations == 0 and pareto_exists
                    and not self.cfg.force_finalize):
                self.log.warning(
                    "Checkpoint already at the target generation and final "
                    "artifacts exist — skipping re-finalisation to protect "
                    "them. Use --force-finalize to overwrite deliberately.")
                return {"skipped": True,
                        "reason": "already finalized at target generation",
                        "n_generations": self.cfg.n_generations,
                        "population_size": self.cfg.population_size}

            result = engine.run(
                n_generations=remaining_generations,
                on_generation_end=self._on_generation_end,
            )
        finally:
            self._history_writer.close()

        final_meta = self._population_meta(engine.population)

        summary = self._finalize(
            final_population=engine.population,
            final_objectives=engine.objectives,
            final_meta=final_meta,
        )

        elapsed = time.perf_counter() - t0
        summary["elapsed_seconds"] = round(elapsed, 3)
        summary["n_generations"] = self.cfg.n_generations
        summary["population_size"] = self.cfg.population_size

        (self.cfg.results_dir / "search_summary.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        self.log.info("=== Search finished in %.2f s ===", elapsed)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run NSGA-II NAS + quantization-aware multi-objective search."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="Path to a YAML/JSON search configuration.")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--category", type=str, default=None,
                   help="MVTec AD category to optimize for.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--n-generations", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   choices=[None, "cpu", "cuda"])
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--force-finalize", action="store_true",
                   help="Allow a 0-generation resume to overwrite existing "
                        "final artifacts (pareto_front.csv, summary…).")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce log verbosity to WARNING.")
    return p


def _apply_cli_overrides(cfg: SearchConfig,
                         args: argparse.Namespace) -> SearchConfig:
    """Override config fields with non-None CLI values."""
    overrides: dict[str, Any] = {
        "results_dir": args.results_dir,
        "splits_dir": args.splits_dir,
        "category": args.category,
        "seed": args.seed,
        "population_size": args.population_size,
        "n_generations": args.n_generations,
        "device": args.device,
        "n_workers": args.n_workers,
        "resume_from": args.resume_from,
        "top_k": args.top_k,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.force_finalize:
        cfg.force_finalize = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    config_loaded = args.config.is_file()
    if config_loaded:
        cfg = SearchConfig.from_file(args.config)
    else:
        # Allow running with pure CLI args + defaults if no config file exists.
        cfg = SearchConfig()
    cfg = _apply_cli_overrides(cfg, args)

    log_path = cfg.results_dir / "search.log"
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )
    if config_loaded:
        logger.info("Loaded configuration from %s", args.config)
    else:
        logger.warning(
            "Config file %s NOT found — running with built-in defaults + CLI "
            "flags. (This silent fallback previously masked a configs/ vs "
            "config/ path typo.)", args.config)

    try:
        SearchPipeline(cfg, logger=logger).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Search failure: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during NSGA-II search")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
