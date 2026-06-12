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
- Configuration is supplied via a YAML file (default ``configs/search.yaml``);
  CLI flags override individual fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
from src.nas.fitness import FitnessEvaluator, FitnessConfig
from src.nas.pareto import pareto_front as compute_pareto_front, crowding_distance
from src.models.model_factory import build_model
from src.quantization.qat_wrapper import wrap_for_qat
from src.profiling.latency_meter import measure_latency
from src.profiling.ram_meter import measure_peak_ram
from src.profiling.energy_meter import measure_energy
from src.evaluation.auroc_eval import evaluate_auroc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
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
    eta_c: float = 15.0                 # SBX distribution index (UNUSED by the
    eta_m: float = 20.0                 # current integer-genome engine, which
                                        # uses single-point crossover + random-
                                        # reset mutation; kept for compatibility)

    # Hypervolume reference point, order: [neg_auroc, latency_ms,
    # peak_ram_mb, energy_mj]. Must STRICTLY dominate (be worse than) every
    # feasible measurement, otherwise valid points contribute zero HV (this
    # silently happened in the pilot run: energy ref 1000 mJ < real PC
    # energy). Penalised evaluations must fall OUTSIDE the reference so
    # failures never inflate the indicator. Calibrate after generation 1
    # using evaluations.csv.
    hv_reference: list[float] = field(
        default_factory=lambda: [0.0, 1000.0, 8192.0, 5000.0]
    )
    # Engine penalty vector for failed evaluations — aligned with
    # FitnessConfig penalties so a failure is encoded identically whether it
    # happens inside the evaluator or as an exception in the engine.
    penalty_objectives: list[float] = field(
        default_factory=lambda: [0.0, 9999.0, 65536.0, 9999.0]
    )

    # Search space + evaluator specs (free-form dicts, validated downstream)
    search_space: dict[str, Any] = field(default_factory=dict)
    fitness: dict[str, Any] = field(default_factory=dict)

    # Compute / runtime
    device: str = "cuda"
    n_workers: int = 1                  # parallel evaluations
    top_k: int = 5                      # how many "best" candidates to dump

    # Resume / checkpointing
    resume_from: Path | None = None     # pickle/npz with population genomes

    # Run hygiene / safety
    keep_old: bool = False              # True -> do NOT delete previous outputs
    allow_noop_energy: bool = False     # True -> tolerate energy backend 'noop'

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Range/shape validation runs at construction time so an invalid
        # config can never start a multi-hour search. Path checks are
        # deferred to validate(check_paths=True), called in main() AFTER
        # CLI overrides (a CLI flag may legitimately fix a default path).
        self.validate(check_paths=False)

    def validate(self, check_paths: bool = True) -> None:
        """Fail fast on invalid, inconsistent or incomplete configuration.

        Raises
        ------
        ValueError / FileNotFoundError
            With an explicit message. No silent fallbacks.
        """
        if self.population_size < 4:
            raise ValueError(
                f"population_size={self.population_size} inválido (mínimo 4).")
        if self.n_generations < 1:
            raise ValueError(
                f"n_generations={self.n_generations} inválido (mínimo 1). "
                "Un valor <= 0 produciría una 'búsqueda' vacía que termina "
                "con éxito aparente sin buscar nada.")
        if self.top_k < 1:
            raise ValueError(f"top_k={self.top_k} inválido (mínimo 1).")
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} inválido (debe ser >= 0).")
        if not (0.0 <= self.crossover_prob <= 1.0):
            raise ValueError(
                f"crossover_prob={self.crossover_prob} fuera de [0, 1].")
        if self.mutation_prob is not None and not (0.0 < self.mutation_prob <= 1.0):
            raise ValueError(
                f"mutation_prob={self.mutation_prob} fuera de (0, 1].")
        if len(self.hv_reference) != N_OBJECTIVES:
            raise ValueError(
                f"hv_reference debe tener {N_OBJECTIVES} elementos "
                f"(orden {OBJECTIVE_NAMES}); tiene {len(self.hv_reference)}.")
        if len(self.penalty_objectives) != N_OBJECTIVES:
            raise ValueError(
                f"penalty_objectives debe tener {N_OBJECTIVES} elementos; "
                f"tiene {len(self.penalty_objectives)}.")
        # Cost objectives (latency, RAM, energy): the penalty must lie
        # STRICTLY OUTSIDE the hypervolume reference box, otherwise failed
        # evaluations inflate the hypervolume indicator.
        for i in range(1, N_OBJECTIVES):
            if self.penalty_objectives[i] <= self.hv_reference[i]:
                raise ValueError(
                    f"penalty_objectives[{i}]={self.penalty_objectives[i]} "
                    f"debe ser > hv_reference[{i}]={self.hv_reference[i]} "
                    f"({OBJECTIVE_NAMES[i]}): si el penalti cae dentro de la "
                    "caja de referencia, los fallos inflan el hipervolumen.")
        if check_paths:
            if not Path(self.splits_dir).is_dir():
                raise FileNotFoundError(
                    f"splits_dir no existe: {self.splits_dir} — ejecuta "
                    "main_prepare.py o corrige la ruta.")
            split_manifest = Path(self.splits_dir) / f"{self.category}.json"
            if not split_manifest.is_file():
                raise FileNotFoundError(
                    f"No hay manifest de splits para la categoría "
                    f"'{self.category}': {split_manifest}")
            try:
                Path(self.results_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(
                    f"results_dir no es escribible: {self.results_dir} ({exc})")

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
        """Build a SearchConfig from a plain dict, coercing path fields."""
        kwargs = dict(data)
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
            # A typo in a YAML key must never become a silent default.
            logging.getLogger("search").warning(
                "Claves de configuración NO reconocidas (¿typo?): %s — "
                "se ignoran y se usa el default para el campo real.",
                sorted(extra),
            )
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
# Stale-output cleanup
# ---------------------------------------------------------------------------
# Artifacts produced by a previous search run. They are removed before a
# fresh run so that old and new results can never be mixed in the same
# directory. Log files are intentionally NOT removed (the logger keeps an
# open handle on them).
_STALE_PATTERNS: tuple[str, ...] = (
    "population_gen_*.csv", "history.csv", "evaluations.csv",
    "pareto_front.csv",
    "best_candidates.json", "final_population.npz",
    "latest_checkpoint.json", "latest_checkpoint.npz",
    "search_config.json", "search_summary.json", "run_metadata.json",
    "pareto_front.png", "convergence.png", "convergence.csv",
)


def _clean_previous_outputs(results_dir: Path, logger: logging.Logger) -> int:
    """Delete artifacts from a previous run inside ``results_dir``.

    Returns the number of files removed. Never raises: a file that cannot
    be removed (e.g. locked) is reported and skipped.
    """
    n_removed = 0
    for pattern in _STALE_PATTERNS:
        for f in sorted(results_dir.glob(pattern)):
            try:
                f.unlink()
                n_removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale file %s: %s", f, exc)
    if n_removed:
        logger.info("Cleanup: removed %d stale artifact(s) from %s "
                    "(use --keep-old to disable).", n_removed, results_dir)
    return n_removed


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------
class HistoryWriter:
    """Append-only CSV writer for the global evaluation history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lazy open: the file is only opened (in append mode) on the first
        # actual write. Opening eagerly with mode "w" truncated history.csv
        # to 0 bytes whenever a resumed run had no generations left to run.
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            is_new = (not self.path.is_file()
                      or self.path.stat().st_size == 0)
            self._fh = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh,
                                          fieldnames=list(row.keys()))
            if is_new:
                self._writer.writeheader()
        # Coerce non-scalar values to JSON for round-trippable storage.
        clean = {k: (json.dumps(v, default=str)
                     if isinstance(v, (dict, list, tuple)) else v)
                 for k, v in row.items()}
        self._writer.writerow(clean)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()


def _write_population_csv(path: Path,
                          gen: int,
                          objectives: np.ndarray,
                          meta: list[dict[str, Any]]) -> None:
    """Persist a single generation's population to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["generation", "individual", "fingerprint"]
                  + list(OBJECTIVE_NAMES)
                  + ["valid", "error", "candidate", "metrics"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for i, (obj, m) in enumerate(zip(objectives, meta)):
            row = {
                "generation": gen,
                "individual": i,
                "fingerprint": m.get("fingerprint"),
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
    fieldnames = (["pareto_rank", "individual_index", "fingerprint",
                   "crowding_distance"]
                  + list(OBJECTIVE_NAMES)
                  + ["valid", "error", "candidate", "metrics"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for rank, (idx, obj) in enumerate(zip(indices, front_obj)):
            m = meta[idx]
            row = {
                "pareto_rank": rank,
                "individual_index": int(idx),
                "fingerprint": m.get("fingerprint"),
                "crowding_distance": float(cd[rank]) if len(cd) else None,
                "valid": m.get("valid", True),
                "error": m.get("error"),
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
    """Persist top-K candidates from the Pareto front (by AUROC then latency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(pareto_idx) == 0:
        path.write_text(json.dumps({"candidates": []}, indent=2),
                        encoding="utf-8")
        return
    front = objectives[pareto_idx]
    # Lexicographic priority: highest AUROC (= lowest neg_auroc), then lowest latency.
    order = np.lexsort((front[:, 1], front[:, 0]))
    chosen = pareto_idx[order][:top_k]
    payload = {
        "objective_names": list(OBJECTIVE_NAMES),
        "candidates": [
            {
                "rank": r,
                "individual_index": int(idx),
                "fingerprint": meta[idx].get("fingerprint"),
                "valid": meta[idx].get("valid", True),
                "error": meta[idx].get("error"),
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

        self._rng = np.random.default_rng(cfg.seed)
        self._history_writer = HistoryWriter(
            self.cfg.results_dir / "history.csv"
        )
        # True per-evaluation trace (one row per fitness call, including
        # penalty reason, timings and normalised objectives) — fed by the
        # FitnessEvaluator callback. This is the primary input for the
        # convergence / robustness analysis.
        self._eval_writer = HistoryWriter(
            self.cfg.results_dir / "evaluations.csv"
        )
        self._global_eval_idx = 0
        self._fitness_eval_idx = 0

        # Encoder propio del pipeline (misma clase y espacio que el del
        # motor) para fingerprints y decodificación sin tocar privados.
        self._encoder: Encoder | None = None
        # Registro fingerprint -> última evaluación real del FitnessEvaluator.
        # Es la fuente de verdad de metrics/valid/error para history.csv,
        # population_gen_*.csv, pareto_front.csv y best_candidates.json
        # (antes se reconstruían con dicts vacíos y valid=True fijo).
        self._eval_registry: dict[str, dict[str, Any]] = {}
        # Rachas para el watchdog de degeneración de objetivos.
        self._degen_streaks: dict[str, int] = {n: 0 for n in OBJECTIVE_NAMES}
        self._full_front_streak: int = 0

    # ------------------------------------------------------------------
    def _fingerprint_of(self, candidate: dict) -> str | None:
        """Stable hex id of a candidate dict (None if not encodable)."""
        if not candidate or self._encoder is None:
            return None
        try:
            return self._encoder.fingerprint(self._encoder.encode(candidate))
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    def _log_evaluation(self, result: Any, candidate: dict) -> None:
        """FitnessEvaluator callback: persist one row per evaluation."""
        try:
            fp = self._fingerprint_of(candidate)
            row: dict[str, Any] = {
                "eval_id": self._fitness_eval_idx,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fingerprint": fp,
            }
            row.update(result.to_dict())
            row["candidate"] = candidate
            self._eval_writer.write(row)
            if fp is not None:
                # Última evaluación gana (si un genoma duplicado se
                # re-evalúa, el frente final usa la medición más reciente).
                self._eval_registry[fp] = {
                    "metrics": result.to_dict(),
                    "valid": not bool(getattr(result, "penalty_applied", False)),
                    "error": getattr(result, "error_message", "") or None,
                }
        except Exception:  # noqa: BLE001 — logging must never kill the search
            self.log.warning("Could not log evaluation row", exc_info=True)
        finally:
            self._fitness_eval_idx += 1

    # ------------------------------------------------------------------
    def _meta_for(self, gene_vec: np.ndarray) -> dict[str, Any]:
        """Real per-individual metadata, joined by CANONICAL fingerprint.

        The fingerprint is computed on encode(decode(gene)) — never on the
        raw gene vector — because inactive slot genes (stages beyond
        n_stages) are arbitrary: two raw genomes with identical PHENOTYPE
        can differ in inactive genes. Canonicalising guarantees the join
        with evaluations.csv (whose fingerprints come from the decoded
        candidate dict) is exact.
        """
        assert self._encoder is not None
        candidate = self._encoder.gene_to_dict(gene_vec)
        fp = self._fingerprint_of(candidate)
        reg = self._eval_registry.get(fp)
        if reg is not None:
            return {"valid": reg["valid"], "error": reg["error"],
                    "candidate": candidate, "metrics": reg["metrics"],
                    "fingerprint": fp}
        # Individuo sin registro (p. ej. población restaurada de un
        # checkpoint legacy): no inventar valid=True.
        return {"valid": None, "error": "no_evaluation_record",
                "candidate": candidate, "metrics": {}, "fingerprint": fp}

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
        # Disponible ANTES de crear el evaluador: el callback de registro
        # necesita el encoder para calcular fingerprints canónicos.
        self._encoder = encoder

        self.log.info("Building fitness evaluator (device=%s)", self.cfg.device)
        fitness_cfg_dict = {
            "splits_dir": str(self.cfg.splits_dir),
            "category": self.cfg.category,
            "device": self.cfg.device,
            # CAMBIO 8 (determinismo): la semilla de entrenamiento proxy se
            # deriva de la semilla del experimento (antes era 0 cableado en
            # fitness.py). Sigue siendo LA MISMA para todos los candidatos
            # de una corrida (comparación justa entre candidatos), pero
            # corridas con seed distinto son ahora verdaderamente
            # independientes también en la inicialización de pesos y el
            # shuffle de datos. Sobreescribible vía fitness.train_seed.
            "train_seed": int(self.cfg.seed),
            **self.cfg.fitness
        }
        # Filter kwargs to match FitnessConfig fields
        from dataclasses import fields
        valid_keys = {f.name for f in fields(FitnessConfig)}
        filtered_cfg = {k: v for k, v in fitness_cfg_dict.items() if k in valid_keys}
        fitness_config = FitnessConfig(**filtered_cfg)
        evaluator = FitnessEvaluator(
            config=fitness_config,
            extra_callbacks=[self._log_evaluation],
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
    def _write_run_metadata(self, energy_source: str) -> None:
        """Persist everything needed to reconstruct this run in 2 years.

        Pure-stdlib: the git commit is read directly from .git/ (no git
        binary required); `git diff` is attempted via subprocess but its
        absence only yields "unavailable", never an error.
        """
        import platform
        import subprocess
        import hashlib

        meta: dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "python": sys.version,
            "argv": sys.argv,
            "seed": self.cfg.seed,
            "energy_backend": energy_source,
            "config": self.cfg.to_dict(),
        }
        # --- git commit, leído del propio .git (sin binario externo) ----
        git_dir = PROJECT_ROOT / ".git"
        commit = "unavailable"
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref:"):
                ref = head.split(" ", 1)[1].strip()
                ref_file = git_dir / ref
                if ref_file.is_file():
                    commit = ref_file.read_text(encoding="utf-8").strip()
                else:  # packed refs
                    packed = git_dir / "packed-refs"
                    if packed.is_file():
                        for line in packed.read_text(encoding="utf-8").splitlines():
                            if line.endswith(ref):
                                commit = line.split(" ", 1)[0]
                                break
            else:
                commit = head  # detached HEAD
        except OSError:
            pass
        meta["git_commit"] = commit
        try:  # diff de trabajo (best-effort; requiere binario git)
            diff = subprocess.run(
                ["git", "diff", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=10,
            )
            meta["git_diff"] = diff.stdout if diff.returncode == 0 else "unavailable"
            meta["git_dirty"] = bool(meta["git_diff"].strip()) \
                if meta["git_diff"] != "unavailable" else None
        except Exception:  # noqa: BLE001
            meta["git_diff"] = "unavailable"
            meta["git_dirty"] = None
        # --- stack de cómputo -------------------------------------------
        try:
            import torch
            meta["torch"] = torch.__version__
            meta["cuda"] = torch.version.cuda
            meta["cudnn"] = torch.backends.cudnn.version()
            if torch.cuda.is_available():
                meta["gpu"] = torch.cuda.get_device_name(0)
                meta["driver_capability"] = ".".join(
                    map(str, torch.cuda.get_device_capability(0)))
        except Exception:  # noqa: BLE001
            meta["torch"] = "unavailable"
        meta["numpy"] = np.__version__
        # --- huella del dataset (manifest de splits de la categoría) ----
        split = Path(self.cfg.splits_dir) / f"{self.cfg.category}.json"
        if split.is_file():
            meta["split_manifest_sha256"] = hashlib.sha256(
                split.read_bytes()).hexdigest()
        # --- huella del espacio de búsqueda ------------------------------
        try:
            from src.nas.nsga2_engine import search_space_hash
            meta["search_space_hash"] = search_space_hash()
        except Exception:  # noqa: BLE001
            pass
        (self.cfg.results_dir / "run_metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self.log.info("run_metadata.json escrito (commit=%s, energía=%s)",
                      commit[:12], energy_source)

    # ------------------------------------------------------------------
    def _probe_energy_backend(self) -> str:
        """Run a tiny measurement to discover the active energy backend."""
        try:
            import torch
            from src.profiling.energy_meter import measure_energy
            dev = self.cfg.device if torch.cuda.is_available() else "cpu"
            probe = torch.nn.Conv2d(3, 8, 3).to(dev)
            r = measure_energy(probe, input_shape=(1, 3, 64, 64),
                               device=dev, n_warmup=2, n_iters=5)
            return str(r.get("source", "unknown"))
        except Exception as exc:  # noqa: BLE001
            self.log.warning("Energy backend probe failed: %s", exc)
            return "error"

    # ------------------------------------------------------------------
    def _on_generation_end(self,
                           gen: int,
                           result: Any,
                           engine: NSGA2Engine) -> None:
        """Callback invoked by the NSGA-II engine after each generation."""
        population = engine.population
        objectives = engine.objectives
        # Metadatos REALES por individuo (join por fingerprint canónico con
        # el registro alimentado por el FitnessEvaluator) — nunca dicts
        # vacíos con valid=True incondicional.
        meta = [self._meta_for(population[i]) for i in range(len(population))]

        # Per-generation population CSV
        gen_path = self.cfg.results_dir / f"population_gen_{gen:03d}.csv"
        _write_population_csv(gen_path, gen, objectives, meta)

        # Append every individual to the global history
        for i, (obj, m) in enumerate(zip(objectives, meta)):
            row: dict[str, Any] = {
                "eval_id": self._global_eval_idx,
                "generation": gen,
                "individual": i,
                "valid": m.get("valid", True),
                "error": m.get("error"),
            }
            for j, name in enumerate(OBJECTIVE_NAMES):
                row[name] = float(obj[j]) if obj is not None else None
            row["fingerprint"] = m.get("fingerprint")
            row["candidate"] = m.get("candidate", {})
            row["metrics"] = m.get("metrics", {})
            self._history_writer.write(row)
            self._global_eval_idx += 1

        self.log.info(
            "Gen %03d done — best AUROC=%.4f, min latency=%.2f ms, "
            "valid=%d/%d",
            gen, result.best_auroc, result.best_latency_ms,
            len(meta) - result.n_failed_evals, len(meta),
        )

        # ---- watchdog de degeneración de objetivos (CAMBIO 5) -----------
        self._degeneration_watchdog(gen, np.asarray(objectives, float),
                                    result)

        # Auto-checkpointing for easy resume
        ckpt_path = self.cfg.results_dir / "latest_checkpoint"
        engine.checkpoint(ckpt_path)

    # ------------------------------------------------------------------
    def _degeneration_watchdog(self, gen: int, obj: np.ndarray,
                               result: Any) -> None:
        """Alerta en línea si un objetivo pierde poder discriminativo.

        Firma del fallo (observada en la corrida piloto): un objetivo
        constante hace que la dominancia pierda fuerza y el frente se
        infle hasta el 100% de la población, sin ningún error visible.
        """
        n = len(obj)
        # (a) Frente = población completa
        if result.n_pareto == n:
            self._full_front_streak += 1
            lvl = self.log.error if self._full_front_streak >= 3 else self.log.warning
            lvl("DEGENERACIÓN? Frente de Pareto = %d/%d (100%% no dominados, "
                "racha=%d gen). Firma típica de objetivo constante — revisa "
                "evaluations.csv.", result.n_pareto, n, self._full_front_streak)
        else:
            self._full_front_streak = 0

        # (b) Varianza ~0 por objetivo (excluyendo individuos penalizados)
        signs = np.array([-1.0, 1.0, 1.0, 1.0])
        penalty_vec = np.asarray(self.cfg.penalty_objectives, float) * signs
        not_penalised = ~np.all(np.isclose(obj, penalty_vec), axis=1)
        ok = obj[not_penalised]
        for j, name in enumerate(OBJECTIVE_NAMES):
            col = ok[:, j] if len(ok) else np.array([])
            degenerate = (len(col) >= max(4, n // 4)
                          and float(np.std(col)) < 1e-9)
            if degenerate:
                self._degen_streaks[name] += 1
                streak = self._degen_streaks[name]
                lvl = self.log.error if streak >= 3 else self.log.warning
                lvl("OBJETIVO DEGENERADO: '%s' constante=%.6g en %d/%d "
                    "individuos válidos (gen %d, racha=%d). La búsqueda está "
                    "optimizando %d objetivos, no %d. stats: min=%.6g "
                    "p50=%.6g max=%.6g std=%.3g",
                    name, float(col[0]), len(col), n, gen, streak,
                    N_OBJECTIVES - 1, N_OBJECTIVES,
                    float(col.min()), float(np.median(col)),
                    float(col.max()), float(col.std()))
            else:
                self._degen_streaks[name] = 0

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
                "n_valid": int(valid_mask.sum())}

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute the full search loop and return a summary dict."""
        t0 = time.perf_counter()
        self.log.info("=== NSGA-II search — START ===")
        self.log.info("Config: %s", json.dumps(self.cfg.to_dict(),
                                               default=str, indent=2))

        # ---- stale-output cleanup (skip when resuming or --keep-old) ----
        if self.cfg.resume_from is not None:
            self.log.info("Cleanup skipped: resume_from is set.")
        elif self.cfg.keep_old:
            self.log.info("Cleanup skipped: --keep-old requested.")
        else:
            _clean_previous_outputs(self.cfg.results_dir, self.log)

        # ---- energy-backend probe (prevents a silently degenerate
        #      energy objective like the energy_mj=9999 run) -------------
        energy_source = self._probe_energy_backend()
        if energy_source in ("noop", "error", "unknown"):
            msg = (
                "Energy meter backend is '%s' — the energy objective would "
                "be a constant penalty (9999) and the search would degrade "
                "to 3 objectives. Install 'nvidia-ml-py' (PC) or run on "
                "Jetson (tegrastats). Use --allow-noop-energy to override."
                % energy_source
            )
            if self.cfg.allow_noop_energy:
                self.log.warning(msg)
            else:
                raise RuntimeError(msg)
        else:
            self.log.info("Energy meter backend OK: source=%s",
                          energy_source)

        # CAMBIO 6: snapshot de reproducibilidad (commit, stack, dataset).
        self._write_run_metadata(energy_source)

        # Persist the resolved config alongside the results.
        (self.cfg.results_dir / "search_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        encoder, evaluator = self._build_components()
        mutation_prob = (self.cfg.mutation_prob
                         if self.cfg.mutation_prob is not None
                         else 1.0 / max(encoder.search_space.genome_length, 1))
        self.log.info("Genome length=%d, mutation_prob=%.4f",
                      encoder.search_space.genome_length, mutation_prob)

        nsga_cfg = NSGA2Config(
            population_size=self.cfg.population_size,
            n_objectives=N_OBJECTIVES,
            crossover_prob=self.cfg.crossover_prob,
            mutation_rate=mutation_prob,
            seed=self.cfg.seed,
            hv_reference=list(self.cfg.hv_reference),
            penalty_objectives=list(self.cfg.penalty_objectives),
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
            
            result = engine.run(
                n_generations=remaining_generations,
                on_generation_end=self._on_generation_end,
            )
        finally:
            self._history_writer.close()
            self._eval_writer.close()

        final_meta = [self._meta_for(engine.population[i])
                      for i in range(len(engine.population))]

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
    p.add_argument("--keep-old", action="store_true",
                   help="Do NOT delete artifacts from a previous run.")
    p.add_argument("--allow-noop-energy", action="store_true",
                   help="Run even if the energy meter has no real backend "
                        "(objective degenerates to a constant penalty).")
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
    if args.keep_old:
        cfg.keep_old = True
    if args.allow_noop_energy:
        cfg.allow_noop_energy = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    config_found = args.config.is_file()
    if config_found:
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
    if not config_found:
        logger.warning(
            "Config file %s NOT found — running with built-in defaults. "
            "Pass --config or create the file to control the run.",
            args.config,
        )

    # Full validation (including paths) AFTER CLI overrides: an invalid
    # config must abort here, never start an expensive search.
    try:
        cfg.validate(check_paths=True)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Configuración inválida — búsqueda abortada: %s", exc)
        return 2

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

