"""
main_prepare.py
===============

Entry-point orchestration script for the data-preparation stage of the
quantized-NN / MVTec-AD research pipeline.

Responsibilities
----------------
1. Apply global reproducibility settings (seeds, deterministic flags).
2. Capture host/embedded-platform system information for the experiment record.
3. Extract the MVTec AD archive into ``data/raw/``.
4. Validate dataset structural integrity (categories, train/test layout, masks).
5. Build deterministic train/val/test splits per category into ``data/splits/``.
6. Optionally pre-process images into ``data/processed/`` (delegated to the
   extractor module if its interface supports it).
7. Persist a dataset summary and a system-info snapshot under ``results/``.

Expected module interfaces (reasonable contract for downstream implementation)
-----------------------------------------------------------------------------
``src.data.extract_dataset``
    ``extract_archive(archive_path: Path, raw_dir: Path,
                      processed_dir: Path | None = None,
                      force: bool = False) -> dict``
        Extracts the MVTec archive into ``raw_dir``. Returns a dict with at
        least ``{"raw_root": Path, "categories": list[str],
        "n_files_extracted": int, "skipped": bool}``.

``src.data.validate_dataset``
    ``validate(raw_dir: Path, expected_categories: list[str] | None = None
               ) -> dict``
        Validates layout/integrity. Returns a dict with at least
        ``{"valid": bool, "categories": list[str], "n_train": int,
        "n_test": int, "n_masks": int, "issues": list[str]}``.

``src.data.build_splits``
    ``build_splits(raw_dir: Path, splits_dir: Path, val_ratio: float,
                   seed: int, stratify_by_defect: bool = True) -> dict``
        Writes per-category split manifests (CSV/JSON) to ``splits_dir``.
        Returns ``{"splits_dir": Path, "categories": list[str],
        "summary": dict[str, dict[str, int]]}``.

``src.utils.set_seed``
    ``set_seed(seed: int, *, deterministic_torch: bool = True) -> SeedState``
        Seeds python/numpy/torch (cuda included) and returns a SeedState
        dataclass documenting the applied configuration.

``src.utils.system_info``
    ``collect_system_info() -> dict``
        Collects OS, CPU, RAM, GPU/Jetson info, library versions, CUDA, etc.

Assumptions
-----------
- The MVTec AD archive is provided as ``mvtec_anomaly_detection.tar.xz`` at
  the project root (override via ``--archive``).
- The 15 standard MVTec AD categories are expected by default; override via
  ``--categories``.
- This script is the *first* stage of the pipeline and is safe to re-run
  (idempotent unless ``--force`` is passed).
- Heavy lifting (extraction, validation, splitting) lives in the imported
  modules; this script only orchestrates, logs, and persists summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

# ---------------------------------------------------------------------------
# Project module imports (interfaces declared in the docstring above).
# ---------------------------------------------------------------------------
from src.data.extract_dataset import extract_archive
from src.data.validate_dataset import validate as validate_dataset
from src.data.build_splits import build_splits
from src.utils.set_seed import set_seed
from src.utils.system_info import collect as collect_system_info


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_ARCHIVE: Path = PROJECT_ROOT / "mvtec_anomaly_detection.tar.xz"
DEFAULT_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / "results"

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
)


# ---------------------------------------------------------------------------
# Configuration dataclass (YAML-friendly)
# ---------------------------------------------------------------------------
@dataclass
class PrepareConfig:
    """Configuration for the preparation stage."""

    archive_path: Path = DEFAULT_ARCHIVE
    raw_dir: Path = DEFAULT_RAW_DIR
    processed_dir: Path = DEFAULT_PROCESSED_DIR
    splits_dir: Path = DEFAULT_SPLITS_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR

    seed: int = 42
    deterministic: bool = True

    val_ratio: float = 0.15
    stratify_by_defect: bool = True

    expected_categories: tuple[str, ...] = MVTEC_CATEGORIES

    force: bool = False
    skip_processed: bool = False
    checksum: bool = True          # SHA-256 of the archive (cached by size+mtime)

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dictionary.

        Paths are rendered relative to the project root when possible so
        persisted artifacts stay portable across machines.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = _rel(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrepareConfig":
        """Build a PrepareConfig from a plain dict, coercing typed fields."""
        kwargs = dict(data)
        for key in ("archive_path", "raw_dir", "processed_dir",
                    "splits_dir", "results_dir"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        if "expected_categories" in kwargs:
            kwargs["expected_categories"] = tuple(kwargs["expected_categories"])
        known = set(cls.__dataclass_fields__)
        extra = {k: v for k, v in kwargs.items() if k not in known}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        cfg = cls(**kwargs)
        if extra:
            cfg.extra.update(extra)
        return cfg

    @classmethod
    def from_file(cls, path: Path) -> "PrepareConfig":
        """Load a PrepareConfig from a YAML or JSON file."""
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging(log_path: Path | None = None,
                       level: int = logging.INFO) -> logging.Logger:
    """Configure a root logger streaming to stdout and (optionally) to a file."""
    logger = logging.getLogger("prepare")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dirs(*dirs: Path) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically dump ``payload`` as pretty-printed JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)
    tmp.replace(path)


def _rel(p: Path | str | None) -> str | None:
    """Render a path relative to the project root when possible.

    Persisted artifacts must not embed machine-specific absolute paths
    (e.g. ``C:\\Users\\...``): they break portability of results between the
    development PC and the embedded target.
    """
    if p is None:
        return None
    p = Path(p)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def _sha256_file(path: Path, chunk_mb: int = 8) -> str:
    """Stream a SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
class PreparePipeline:
    """Orchestrates the data-preparation stage end-to-end."""

    def __init__(self, config: PrepareConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = config
        self.log = logger or logging.getLogger("prepare")
        self._record: dict[str, Any] = {
            "config": self.cfg.to_dict(),
            "stages": {},
        }

    # -- stage 1 -----------------------------------------------------------
    def _stage_seed(self) -> dict[str, Any]:
        self.log.info("Setting global random seed: %d (deterministic=%s)",
                      self.cfg.seed, self.cfg.deterministic)
        seed_info = set_seed(seed=self.cfg.seed,
                             deterministic_torch=self.cfg.deterministic)
        # set_seed returns a SeedState dataclass — serialise it so the
        # reproducibility record survives in JSON form.
        seed_dict = asdict(seed_info) if is_dataclass(seed_info) else dict(seed_info)
        self._record["stages"]["seed"] = seed_dict
        return seed_dict

    # -- stage 2 -----------------------------------------------------------
    def _stage_system_info(self) -> dict[str, Any]:
        self.log.info("Collecting system information")
        info = collect_system_info()
        out_path = self.cfg.results_dir / "system_info.json"
        _write_json(out_path, info)
        self.log.info("System info written to %s", out_path)
        self._record["stages"]["system_info"] = {
            "path": str(out_path),
            "summary": {
                k: info.get(k) for k in
                ("os", "python", "torch", "cuda", "gpu", "jetson")
                if k in info
            },
        }
        return info

    # -- stage 2b ----------------------------------------------------------
    def _stage_checksum(self) -> dict[str, Any]:
        """SHA-256 of the dataset archive, cached by (size, mtime).

        Dataset *identity* — not just layout — must be verifiable for the
        experiment record to be reproducible across machines.
        """
        archive = self.cfg.archive_path
        info: dict[str, Any] = {"enabled": self.cfg.checksum,
                                "archive": _rel(archive)}
        if not self.cfg.checksum:
            self.log.info("Archive checksum disabled (--no-checksum).")
            self._record["stages"]["checksum"] = info
            return info

        stat = archive.stat()
        cache_path = self.cfg.results_dir / "archive_checksum.json"
        cached: dict[str, Any] = {}
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = {}

        if (cached.get("size_bytes") == stat.st_size
                and cached.get("mtime") == int(stat.st_mtime)
                and cached.get("sha256")):
            self.log.info("Archive checksum (cached): sha256=%s",
                          cached["sha256"])
            info.update(cached)
        else:
            self.log.info("Computing SHA-256 of %s (%.1f MB) — one-time cost…",
                          archive.name, stat.st_size / 2**20)
            t0 = time.perf_counter()
            digest = _sha256_file(archive)
            info.update({
                "sha256": digest,
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
                "hash_seconds": round(time.perf_counter() - t0, 1),
            })
            _write_json(cache_path, {k: v for k, v in info.items()
                                     if k != "enabled"})
            self.log.info("Archive sha256=%s (%.1f s)",
                          digest, info["hash_seconds"])

        self._record["stages"]["checksum"] = info
        return info

    # -- stage 3 -----------------------------------------------------------
    def _stage_extract(self) -> dict[str, Any]:
        archive = self.cfg.archive_path
        if not archive.is_file():
            raise FileNotFoundError(
                f"MVTec archive not found at {archive}. "
                "Pass --archive <path> or place the .tar.xz at the project root."
            )

        self.log.info("Extracting archive %s -> %s",
                      archive, self.cfg.raw_dir)
        processed = None if self.cfg.skip_processed else self.cfg.processed_dir
        result = extract_archive(
            archive_path=archive,
            raw_dir=self.cfg.raw_dir,
            processed_dir=processed,
            force=self.cfg.force,
        )
        self._record["stages"]["extract"] = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in result.items()
        }
        self.log.info("Extraction complete: %d file(s), %d categor(ies)%s",
                      result.get("n_files_extracted", -1),
                      len(result.get("categories", [])),
                      " [skipped — already present]" if result.get("skipped") else "")
        return result

    # -- stage 4 -----------------------------------------------------------
    def _stage_validate(self) -> dict[str, Any]:
        self.log.info("Validating dataset layout under %s", self.cfg.raw_dir)
        report = validate_dataset(
            raw_dir=self.cfg.raw_dir,
            expected_categories=list(self.cfg.expected_categories),
        )
        self._record["stages"]["validate"] = report

        if not report.get("valid", False):
            issues = report.get("issues", [])
            for issue in issues:
                self.log.error("Validation issue: %s", issue)
            # Persist the failure report BEFORE raising, so the referenced
            # file actually exists when the operator goes looking for it.
            fail_path = self.cfg.results_dir / "validation_failure.json"
            _write_json(fail_path, report)
            raise RuntimeError(
                f"Dataset validation failed with {len(issues)} issue(s). "
                f"Inspect {_rel(fail_path)} for details."
            )
        self.log.info(
            "Validation OK — %d categories, train=%d, test=%d, masks=%d",
            len(report.get("categories", [])),
            report.get("n_train", 0),
            report.get("n_test", 0),
            report.get("n_masks", 0),
        )
        return report

    # -- stage 5 -----------------------------------------------------------
    def _stage_build_splits(self) -> dict[str, Any]:
        self.log.info("Building deterministic splits (val_ratio=%.3f, seed=%d)",
                      self.cfg.val_ratio, self.cfg.seed)
        result = build_splits(
            raw_dir=self.cfg.raw_dir,
            splits_dir=self.cfg.splits_dir,
            val_ratio=self.cfg.val_ratio,
            seed=self.cfg.seed,
            stratify_by_defect=self.cfg.stratify_by_defect,
        )
        self._record["stages"]["splits"] = {
            "splits_dir": str(result.get("splits_dir", self.cfg.splits_dir)),
            "categories": result.get("categories", []),
            "summary": result.get("summary", {}),
        }
        self.log.info("Splits written to %s for %d categor(ies)",
                      result.get("splits_dir"),
                      len(result.get("categories", [])))
        return result

    # -- public API --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute the full preparation pipeline and return the record dict."""
        _ensure_dirs(self.cfg.raw_dir, self.cfg.processed_dir,
                     self.cfg.splits_dir, self.cfg.results_dir)

        t0 = time.perf_counter()
        self.log.info("=== MVTec AD preparation pipeline — START ===")

        self._stage_seed()
        self._stage_system_info()
        checksum_info = self._stage_checksum()
        extract_result = self._stage_extract()
        validate_report = self._stage_validate()
        splits_result = self._stage_build_splits()

        elapsed = time.perf_counter() - t0
        self._record["elapsed_seconds"] = round(elapsed, 3)

        summary = self._build_summary(extract_result,
                                      validate_report,
                                      splits_result)
        summary["archive_sha256"] = checksum_info.get("sha256")
        out_path = self.cfg.results_dir / "dataset_summary.json"
        _write_json(out_path, summary)
        self.log.info("Dataset summary written to %s", out_path)

        # Persist the FULL stage record (seeding state included) — it was
        # previously built and then discarded.
        record_path = self.cfg.results_dir / "prepare_record.json"
        _write_json(record_path, self._record)
        self.log.info("Full preparation record written to %s", record_path)

        self.log.info("=== Pipeline finished in %.2f s ===", elapsed)
        return self._record

    # -- internals ---------------------------------------------------------
    def _build_summary(self,
                       extract_result: dict[str, Any],
                       validate_report: dict[str, Any],
                       splits_result: dict[str, Any]) -> dict[str, Any]:
        """Compose the persisted dataset_summary.json payload."""
        # NOTE: all paths are stored relative to the project root so that the
        # artifacts remain valid when results move between the development PC
        # and the embedded target.
        return {
            "config": self.cfg.to_dict(),
            "extract": {
                "raw_root": _rel(extract_result.get("raw_root",
                                                    self.cfg.raw_dir)),
                "n_files_extracted": extract_result.get("n_files_extracted", 0),
                "skipped": bool(extract_result.get("skipped", False)),
                "categories": extract_result.get("categories", []),
            },
            "validation": {
                "valid": validate_report.get("valid", False),
                "n_train": validate_report.get("n_train", 0),
                "n_test": validate_report.get("n_test", 0),
                "n_masks": validate_report.get("n_masks", 0),
                "issues": validate_report.get("issues", []),
            },
            "splits": {
                "splits_dir": _rel(splits_result.get("splits_dir",
                                                     self.cfg.splits_dir)),
                "per_category": splits_result.get("summary", {}),
            },
            "elapsed_seconds": self._record.get("elapsed_seconds"),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Prepare the MVTec AD dataset, environment, metadata, "
                     "splits, and reproducibility artifacts."),
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Optional YAML/JSON config file (flat PrepareConfig "
                        "keys). CLI flags override individual fields.")
    p.add_argument("--archive", type=Path, default=None,
                   help="Path to mvtec_anomaly_detection.tar.xz")
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--no-deterministic", action="store_true",
                   help="Disable deterministic CUDA/cuDNN flags")
    p.add_argument("--no-stratify", action="store_true",
                   help="Disable stratified split by defect type")
    p.add_argument("--force", action="store_true",
                   help="Re-extract and overwrite existing artifacts")
    p.add_argument("--skip-processed", action="store_true",
                   help="Do not generate processed-image variants")
    p.add_argument("--no-checksum", action="store_true",
                   help="Skip the SHA-256 integrity hash of the archive")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional log file path (default: results/prepare.log)")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce log verbosity to WARNING")
    return p


def _config_from_args(args: argparse.Namespace) -> PrepareConfig:
    """Build the config: file (if given) < CLI overrides < flags."""
    cfg = (PrepareConfig.from_file(args.config)
           if args.config is not None else PrepareConfig())

    overrides = {
        "archive_path": args.archive,
        "raw_dir": args.raw_dir,
        "processed_dir": args.processed_dir,
        "splits_dir": args.splits_dir,
        "results_dir": args.results_dir,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)

    if args.no_deterministic:
        cfg.deterministic = False
    if args.no_stratify:
        cfg.stratify_by_defect = False
    if args.force:
        cfg.force = True
    if args.skip_processed:
        cfg.skip_processed = True
    if args.no_checksum:
        cfg.checksum = False
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = _config_from_args(args)

    log_path = args.log_file or (cfg.results_dir / "prepare.log")
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    try:
        pipeline = PreparePipeline(cfg, logger=logger)
        pipeline.run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Pipeline failure: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during preparation")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
