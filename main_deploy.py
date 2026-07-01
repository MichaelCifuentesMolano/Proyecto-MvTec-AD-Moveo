"""
main_deploy.py
==============

Entry-point orchestration script for the *embedded deployment* stage of the
quantized-NN / MVTec-AD pipeline.

After full-budget retraining (``main_retrain.py``) produces ranked PyTorch
checkpoints, this script converts each selected model to its embedded
deployment artifact (ONNX → TensorRT engine), benchmarks it on the target
device (Jetson Orin Nano) under realistic inference conditions, and re-ranks
candidates using *on-device* metrics (latency p50/p95/p99, throughput,
peak RAM, energy per inference, retained AUROC).

Responsibilities
----------------
1. Discover candidates from ``results/retrain/model_ranked.csv`` (or
   ``final_metrics.csv``) and load the matching checkpoints.
2. For every (candidate, precision) pair:
   * Rebuild the architecture and (optionally) the QAT wrapper.
   * Load the trained state dict.
   * Export to ONNX → ``deployment/models/<id>.onnx``.
   * Build a TensorRT engine → ``deployment/models/<id>.engine``.
   * Benchmark the engine on hardware (latency / throughput / AUROC).
   * Wrap the benchmark with the energy and RAM meters to capture the
     full system-level resource footprint.
3. Stream a row per artifact to ``results/deploy/runtime_metrics.csv``.
4. Re-rank artifacts using a weighted scalar utility and persist
   ``results/deploy/final_embedded_rank.csv``.

Expected module interfaces (downstream contract)
------------------------------------------------
``src.deployment.export_onnx``
    ``export_to_onnx(model: nn.Module,
                     output_path: Path,
                     input_shape: tuple[int, ...],
                     opset: int = 17,
                     dynamic_axes: dict | None = None,
                     convert_qat: bool = True,
                     **kwargs) -> dict``
        Exports the (optionally QAT-converted) model to ONNX.
        Returns ``{"onnx_path": Path, "n_params": int,
                   "input_shape": tuple, "valid": bool, "issues": list[str]}``.

``src.deployment.export_tensorrt``
    ``build_tensorrt_engine(onnx_path: Path,
                            engine_path: Path,
                            precision: str = "fp16",   # 'fp32'|'fp16'|'int8'
                            workspace_mb: int = 1024,
                            max_batch: int = 1,
                            calibration_loader=None,
                            **kwargs) -> dict``
        Compiles a TensorRT engine from the ONNX graph.
        Returns ``{"engine_path": Path, "precision": str,
                   "build_seconds": float, "valid": bool, "log": str}``.

``src.deployment.runtime_benchmark``
    ``benchmark_engine(engine_path: Path,
                       input_shape: tuple[int, ...],
                       splits_dir: Path | None = None,
                       category: str | None = None,
                       n_warmup: int = 50,
                       n_iters: int = 500,
                       device: str = "cuda",
                       **kwargs) -> dict``
        Runs the engine end-to-end on hardware. When ``splits_dir`` and
        ``category`` are given it also computes on-device AUROC over the
        test split.
        Returns at least ``latency_ms_mean``, ``latency_ms_p50``,
        ``latency_ms_p95``, ``latency_ms_p99``, ``throughput_fps``,
        ``auroc`` (or None), and ``input_shape``.

``src.profiling.energy_meter``
    ``measure_energy(callable_or_engine, *, n_iters, device, **kwargs)``
        Returns ``{"energy_mj": float, "avg_power_w": float,
                   "duration_s": float, "source": str}``.

``src.profiling.ram_meter``
    ``measure_peak_ram(callable_or_engine, *, device, **kwargs)``
        Returns ``{"peak_ram_mb": float, "host_peak_mb": float,
                   "device_peak_mb": float | None}``.

Both meters accept either a TensorRT engine path (preferred for
deployment-time measurement) or a callable. The orchestrator uses the
engine-path form so the meters can build their own runner.

Assumptions
-----------
- ``main_retrain.py`` has been executed; ``checkpoints/final_models/<id>.pt``
  and ``results/retrain/model_ranked.csv`` exist.
- Model factory + QAT wrapper interfaces are stable (defined in
  ``main_retrain.py``'s docstring).
- The ``trtexec`` toolchain or NVIDIA TensorRT Python bindings are
  available on the deployment host. Their detection is the responsibility
  of the engine-builder module.
- ``input_shape`` is supplied via configuration. Default
  ``(1, 3, 224, 224)`` matches the standard MVTec AD pre-processing
  produced by ``main_prepare.py``.
- Failures in any sub-stage of a single candidate must not abort the batch:
  partial artifacts and partial CSV rows are always preserved.
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
# Project module imports.
# ---------------------------------------------------------------------------
from src.models.model_factory import build_model
from src.quantization.qat_wrapper import wrap_for_qat
from src.deployment.export_onnx import export_to_onnx
from src.deployment.export_tensorrt import build_tensorrt_engine
from src.deployment.runtime_benchmark import benchmark_engine
from src.profiling.energy_meter import measure_energy
from src.profiling.ram_meter import measure_peak_ram


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "configs" / "deploy.yaml"
DEFAULT_RETRAIN_RESULTS: Path = PROJECT_ROOT / "results" / "retrain"
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "deploy"
DEFAULT_DEPLOY_DIR: Path = PROJECT_ROOT / "deployment" / "models"
DEFAULT_CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints" / "final_models"
DEFAULT_SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits"

DEFAULT_INPUT_SHAPE: tuple[int, ...] = (1, 3, 224, 224)

# CSV schemas (extra fields are accepted and serialized as JSON).
RUNTIME_COLUMNS: tuple[str, ...] = (
    "candidate_id", "precision", "status",
    "onnx_path", "engine_path",
    "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99",
    "throughput_fps", "on_device_auroc",
    "energy_mj_per_inf", "avg_power_w",
    "peak_ram_mb", "device_peak_mb",
    "engine_build_seconds", "n_params",
    "retrain_test_auroc", "retrain_val_auroc",
    "input_shape",
    "extra", "candidate_config", "error",
)

RANK_COLUMNS: tuple[str, ...] = (
    "rank", "score", "candidate_id", "precision",
    "engine_path", "on_device_auroc", "retrain_test_auroc",
    "latency_ms_p95", "throughput_fps",
    "energy_mj_per_inf", "peak_ram_mb",
    "status",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class DeployConfig:
    """Configuration for the embedded-deployment stage."""

    # I/O
    retrain_results_dir: Path = DEFAULT_RETRAIN_RESULTS
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS_DIR
    deploy_dir: Path = DEFAULT_DEPLOY_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    splits_dir: Path = DEFAULT_SPLITS_DIR
    category: str = "bottle"

    # Candidate selection
    candidates_source: str = "model_ranked"  # 'model_ranked' | 'final_metrics'
    max_candidates: int | None = 5

    # Conversion / benchmarking
    input_shape: tuple[int, ...] = DEFAULT_INPUT_SHAPE
    onnx_opset: int = 17
    precisions: tuple[str, ...] = ("fp16", "int8")
    workspace_mb: int = 1024
    n_warmup: int = 50
    n_iters: int = 500

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"

    # Behavior
    skip_existing_engine: bool = True
    fail_fast: bool = False

    # Embedded ranking weights (min-max normalized inside the cohort).
    # AUROC is "higher is better"; everything else is "lower is better".
    ranking: dict[str, float] = field(default_factory=lambda: {
        "on_device_auroc":   1.00,
        "latency_ms_p95":   -0.30,
        "energy_mj_per_inf": -0.20,
        "peak_ram_mb":      -0.10,
    })

    # AUROC drop tolerance vs. retrain test AUROC (used for filtering).
    # If a TRT artifact loses more than this many AUROC points it is flagged
    # but not removed from the ranking.
    auroc_drop_threshold: float = 0.02

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path) -> "DeployConfig":
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
    def from_dict(cls, data: dict[str, Any]) -> "DeployConfig":
        kwargs = dict(data)
        for key in ("retrain_results_dir", "checkpoints_dir", "deploy_dir",
                    "results_dir", "splits_dir"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        if "input_shape" in kwargs and kwargs["input_shape"] is not None:
            kwargs["input_shape"] = tuple(kwargs["input_shape"])
        if "precisions" in kwargs and kwargs["precisions"] is not None:
            kwargs["precisions"] = tuple(kwargs["precisions"])
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging(log_path: Path | None,
                       level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("deploy")
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
# Candidate loading
# ---------------------------------------------------------------------------
def _load_candidates(cfg: DeployConfig,
                     logger: logging.Logger) -> list[dict[str, Any]]:
    """Load candidates from the retrain results directory."""
    if cfg.candidates_source == "model_ranked":
        path = cfg.retrain_results_dir / "model_ranked.csv"
    elif cfg.candidates_source == "final_metrics":
        path = cfg.retrain_results_dir / "final_metrics.csv"
    else:
        raise ValueError(f"Unknown candidates_source: {cfg.candidates_source!r}")

    if not path.is_file():
        raise FileNotFoundError(
            f"Retrain artifact not found: {path}. Run main_retrain.py first."
        )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).strip() not in {"ok", ""}:
                continue
            cid = row.get("candidate_id")
            if not cid:
                continue
            ckpt_path = cfg.checkpoints_dir / f"{cid}.pt"
            if not ckpt_path.is_file():
                logger.warning("Skipping %s — checkpoint missing at %s",
                               cid, ckpt_path)
                continue
            try:
                config = json.loads(row.get("candidate_config") or "{}")
            except json.JSONDecodeError:
                config = {}
            rows.append({
                "candidate_id": cid,
                "checkpoint_path": ckpt_path,
                "config": config,
                "retrain": {
                    "test_auroc": _maybe_float(row.get("test_auroc")),
                    "val_auroc":  _maybe_float(row.get("val_auroc")),
                    "retrain_score": _maybe_float(row.get("score")),
                },
            })

    if cfg.max_candidates is not None:
        rows = rows[: cfg.max_candidates]
    logger.info("Loaded %d candidate(s) from %s", len(rows), path.name)
    return rows


def _maybe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CSV writer (crash-safe append-only with fixed schema)
# ---------------------------------------------------------------------------
class FixedSchemaWriter:
    """CSV writer that flushes per row and serializes complex values as JSON."""

    def __init__(self, path: Path, columns: Iterable[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
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
class DeployPipeline:
    """Orchestrates ONNX → TensorRT export and on-device benchmarking."""

    def __init__(self, cfg: DeployConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("deploy")

        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        cfg.deploy_dir.mkdir(parents=True, exist_ok=True)

        self._writer = FixedSchemaWriter(
            cfg.results_dir / "runtime_metrics.csv", RUNTIME_COLUMNS,
        )
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _artifact_id(self, candidate_id: str, precision: str) -> str:
        return f"{candidate_id}__{precision}"

    # ------------------------------------------------------------------
    def _load_trained_model(self, candidate: dict[str, Any]) -> torch.nn.Module:
        """Rebuild architecture, apply QAT wrapper, and load the checkpoint."""
        config = candidate["config"]
        qconfig = (config.get("quantization", {})
                   if isinstance(config, dict) else {})
        model = build_model(config)
        model = wrap_for_qat(model, qconfig)

        state = torch.load(candidate["checkpoint_path"], map_location="cpu")
        state_dict = state.get("model_state_dict", state) \
            if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            self.log.warning(
                "[%s] state_dict mismatch — missing=%d, unexpected=%d",
                candidate["candidate_id"], len(missing), len(unexpected),
            )
        model.eval()
        return model

    # ------------------------------------------------------------------
    def _export_onnx_for(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Export the model to ONNX once per candidate (precision-agnostic)."""
        cid = candidate["candidate_id"]
        onnx_path = self.cfg.deploy_dir / f"{cid}.onnx"
        if self.cfg.skip_existing_engine and onnx_path.is_file():
            self.log.info("[%s] reusing existing ONNX %s", cid, onnx_path)
            return {"onnx_path": onnx_path, "valid": True, "reused": True}

        model = self._load_trained_model(candidate)
        result = export_to_onnx(
            model=model,
            output_path=onnx_path,
            input_shape=self.cfg.input_shape,
            opset=self.cfg.onnx_opset,
            convert_qat=True,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        result.setdefault("onnx_path", onnx_path)
        result.setdefault("valid", onnx_path.is_file())
        return result

    # ------------------------------------------------------------------
    def _build_engine_for(self,
                          candidate_id: str,
                          onnx_path: Path,
                          precision: str) -> dict[str, Any]:
        engine_path = (self.cfg.deploy_dir
                       / f"{self._artifact_id(candidate_id, precision)}.engine")
        if self.cfg.skip_existing_engine and engine_path.is_file():
            self.log.info("[%s/%s] reusing existing engine %s",
                          candidate_id, precision, engine_path)
            return {"engine_path": engine_path, "precision": precision,
                    "build_seconds": 0.0, "valid": True, "reused": True}

        return build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=precision,
            workspace_mb=self.cfg.workspace_mb,
            max_batch=int(self.cfg.input_shape[0]),
        )

    # ------------------------------------------------------------------
    def _measure_artifact(self,
                          engine_path: Path,
                          precision: str) -> dict[str, Any]:
        """Run benchmark + energy + RAM measurements for a single engine."""
        bench = benchmark_engine(
            engine_path=engine_path,
            input_shape=self.cfg.input_shape,
            splits_dir=self.cfg.splits_dir,
            category=self.cfg.category,
            n_warmup=self.cfg.n_warmup,
            n_iters=self.cfg.n_iters,
            device=self.cfg.device,
            precision=precision,
        )

        energy = measure_energy(
            engine_path,
            n_iters=self.cfg.n_iters,
            device=self.cfg.device,
            input_shape=self.cfg.input_shape,
        )
        if energy.get("energy_mj") is not None and self.cfg.n_iters > 0:
            energy["energy_mj_per_inf"] = float(energy["energy_mj"]) / float(
                self.cfg.n_iters
            )
        else:
            energy["energy_mj_per_inf"] = None

        ram = measure_peak_ram(
            engine_path,
            device=self.cfg.device,
            input_shape=self.cfg.input_shape,
        )
        return {"bench": bench, "energy": energy, "ram": ram}

    # ------------------------------------------------------------------
    def _process_candidate(self,
                           candidate: dict[str, Any]) -> list[dict[str, Any]]:
        """Export ONNX once, then build + benchmark one engine per precision."""
        cid = candidate["candidate_id"]
        retrain = candidate.get("retrain", {})

        # 1. ONNX export (shared across precisions).
        try:
            onnx_result = self._export_onnx_for(candidate)
            onnx_path = Path(onnx_result.get("onnx_path",
                                             self.cfg.deploy_dir / f"{cid}.onnx"))
            if not onnx_result.get("valid", False) or not onnx_path.is_file():
                raise RuntimeError(
                    f"ONNX export reported invalid: {onnx_result}"
                )
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] ONNX export failed", cid)
            return [self._failure_row(candidate, precision=None,
                                      stage="onnx", exc=exc)]

        rows: list[dict[str, Any]] = []
        for precision in self.cfg.precisions:
            row: dict[str, Any] = {
                "candidate_id": cid,
                "precision": precision,
                "status": "pending",
                "onnx_path": str(onnx_path),
                "input_shape": list(self.cfg.input_shape),
                "candidate_config": candidate.get("config", {}),
                "retrain_test_auroc": retrain.get("test_auroc"),
                "retrain_val_auroc":  retrain.get("val_auroc"),
                "n_params": onnx_result.get("n_params"),
            }

            # 2. TensorRT build.
            try:
                eng = self._build_engine_for(cid, onnx_path, precision)
                engine_path = Path(eng.get("engine_path",
                                           self.cfg.deploy_dir
                                           / f"{cid}__{precision}.engine"))
                if not eng.get("valid", False) or not engine_path.is_file():
                    raise RuntimeError(f"engine build invalid: {eng}")
            except Exception as exc:  # noqa: BLE001
                self.log.exception("[%s/%s] TensorRT build failed",
                                   cid, precision)
                row.update(status="failed_engine", error=str(exc))
                rows.append(row)
                continue
            row["engine_path"] = str(engine_path)
            row["engine_build_seconds"] = eng.get("build_seconds")

            # 3. Benchmark + energy + RAM.
            try:
                meas = self._measure_artifact(engine_path, precision)
            except Exception as exc:  # noqa: BLE001
                self.log.exception("[%s/%s] benchmark/profiling failed",
                                   cid, precision)
                row.update(status="failed_benchmark", error=str(exc))
                rows.append(row)
                continue

            bench = meas["bench"]
            energy = meas["energy"]
            ram = meas["ram"]
            row.update(
                latency_ms_mean=bench.get("latency_ms_mean"),
                latency_ms_p50 =bench.get("latency_ms_p50"),
                latency_ms_p95 =bench.get("latency_ms_p95"),
                latency_ms_p99 =bench.get("latency_ms_p99"),
                throughput_fps =bench.get("throughput_fps"),
                on_device_auroc=bench.get("auroc"),
                energy_mj_per_inf=energy.get("energy_mj_per_inf"),
                avg_power_w     =energy.get("avg_power_w"),
                peak_ram_mb     =ram.get("peak_ram_mb"),
                device_peak_mb  =ram.get("device_peak_mb"),
                extra={
                    "energy_total_mj": energy.get("energy_mj"),
                    "energy_source":   energy.get("source"),
                    "host_peak_mb":    ram.get("host_peak_mb"),
                    "bench_extra": {
                        k: v for k, v in bench.items()
                        if k not in {
                            "latency_ms_mean", "latency_ms_p50",
                            "latency_ms_p95",  "latency_ms_p99",
                            "throughput_fps",  "auroc",
                        }
                    },
                },
                status="ok",
            )

            # AUROC-drop sanity check vs. retrain.
            ret_a = retrain.get("test_auroc")
            on_a = row.get("on_device_auroc")
            if ret_a is not None and on_a is not None and \
                    (ret_a - on_a) > self.cfg.auroc_drop_threshold:
                self.log.warning(
                    "[%s/%s] AUROC drop %.4f exceeds threshold %.4f "
                    "(retrain=%.4f, on-device=%.4f)",
                    cid, precision, ret_a - on_a,
                    self.cfg.auroc_drop_threshold, ret_a, on_a,
                )
                row["status"] = "ok_auroc_drop"

            self.log.info(
                "[%s/%s] DONE — p95=%.2f ms, fps=%.1f, "
                "energy=%.3f mJ/inf, RAM=%.1f MB, AUROC=%.4f",
                cid, precision,
                _safe(row["latency_ms_p95"]), _safe(row["throughput_fps"]),
                _safe(row["energy_mj_per_inf"]), _safe(row["peak_ram_mb"]),
                _safe(row["on_device_auroc"]),
            )
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    def _failure_row(self, candidate: dict[str, Any],
                     precision: str | None, stage: str,
                     exc: Exception) -> dict[str, Any]:
        return {
            "candidate_id": candidate["candidate_id"],
            "precision": precision,
            "status": f"failed_{stage}",
            "candidate_config": candidate.get("config", {}),
            "input_shape": list(self.cfg.input_shape),
            "retrain_test_auroc":
                candidate.get("retrain", {}).get("test_auroc"),
            "retrain_val_auroc":
                candidate.get("retrain", {}).get("val_auroc"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    # ------------------------------------------------------------------
    def _rank_artifacts(self) -> None:
        """Compute weighted scalar score over successful artifacts."""
        rank_path = self.cfg.results_dir / "final_embedded_rank.csv"
        successful = [r for r in self._records
                      if str(r.get("status", "")).startswith("ok")]
        if not successful:
            self.log.warning("No successful artifacts — empty ranking written.")
            FixedSchemaWriter(rank_path, RANK_COLUMNS).close()
            return

        weights = self.cfg.ranking
        keys = [k for k in weights if any(r.get(k) is not None
                                          for r in successful)]
        norm: dict[str, list[float]] = {}
        for k in keys:
            values = [_maybe_float(r.get(k)) for r in successful]
            valid = [v for v in values if v is not None]
            vmin, vmax = min(valid), max(valid)
            span = vmax - vmin or 1.0
            norm[k] = [
                ((v - vmin) / span) if v is not None else 0.0
                for v in values
            ]

        scored: list[tuple[float, dict[str, Any]]] = []
        for i, r in enumerate(successful):
            score = sum(weights[k] * norm[k][i] for k in keys)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)

        writer = FixedSchemaWriter(rank_path, RANK_COLUMNS)
        try:
            for rank, (score, r) in enumerate(scored):
                writer.write({
                    "rank": rank,
                    "score": round(float(score), 6),
                    "candidate_id":      r.get("candidate_id"),
                    "precision":         r.get("precision"),
                    "engine_path":       r.get("engine_path"),
                    "on_device_auroc":   r.get("on_device_auroc"),
                    "retrain_test_auroc": r.get("retrain_test_auroc"),
                    "latency_ms_p95":    r.get("latency_ms_p95"),
                    "throughput_fps":    r.get("throughput_fps"),
                    "energy_mj_per_inf": r.get("energy_mj_per_inf"),
                    "peak_ram_mb":       r.get("peak_ram_mb"),
                    "status":            r.get("status"),
                })
        finally:
            writer.close()
        self.log.info("Wrote embedded ranking for %d artifact(s) to %s",
                      len(scored), rank_path)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute the full deployment pipeline."""
        t0 = time.perf_counter()
        self.log.info("=== Embedded deployment — START ===")
        self.log.info("Config: %s",
                      json.dumps(self.cfg.to_dict(), indent=2, default=str))

        (self.cfg.results_dir / "deploy_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        candidates = _load_candidates(self.cfg, self.log)

        try:
            for i, cand in enumerate(candidates):
                self.log.info("---- Candidate %d/%d (%s) ----",
                              i + 1, len(candidates), cand["candidate_id"])
                try:
                    rows = self._process_candidate(cand)
                except KeyboardInterrupt:
                    raise
                except Exception:  # noqa: BLE001
                    self.log.exception(
                        "Unhandled error processing %s", cand["candidate_id"]
                    )
                    rows = [{
                        "candidate_id": cand["candidate_id"],
                        "precision": None,
                        "status": "failed_unhandled",
                        "error": traceback.format_exc(limit=3),
                        "candidate_config": cand.get("config", {}),
                        "input_shape": list(self.cfg.input_shape),
                    }]

                for r in rows:
                    self._records.append(r)
                    self._writer.write(r)

                if self.cfg.fail_fast and any(
                    not str(r.get("status", "")).startswith("ok")
                    for r in rows
                ):
                    self.log.error(
                        "fail_fast=True and candidate %s produced failure(s) "
                        "— aborting batch.", cand["candidate_id"],
                    )
                    break
        finally:
            self._writer.close()

        self._rank_artifacts()

        elapsed = time.perf_counter() - t0
        n_ok = sum(1 for r in self._records
                   if str(r.get("status", "")).startswith("ok"))
        summary = {
            "n_candidates": len(candidates),
            "n_artifacts": len(self._records),
            "n_ok": n_ok,
            "n_failed": len(self._records) - n_ok,
            "elapsed_seconds": round(elapsed, 3),
            "runtime_metrics_csv":
                str(self.cfg.results_dir / "runtime_metrics.csv"),
            "final_embedded_rank_csv":
                str(self.cfg.results_dir / "final_embedded_rank.csv"),
            "deploy_dir": str(self.cfg.deploy_dir),
        }
        (self.cfg.results_dir / "deploy_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self.log.info("=== Deployment finished in %.2f s — ok=%d/%d ===",
                      elapsed, n_ok, len(self._records))
        return summary


def _safe(v: Any) -> float:
    """Format helper for log lines — converts None to NaN for printf."""
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Convert retrained candidates to ONNX+TensorRT and "
                     "benchmark them on the embedded target."),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--retrain-results-dir", type=Path, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=None)
    p.add_argument("--deploy-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--candidates-source", type=str, default=None,
                   choices=[None, "model_ranked", "final_metrics"])
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--precisions", type=str, default=None,
                   help="Comma-separated list, e.g. 'fp16,int8'")
    p.add_argument("--input-shape", type=str, default=None,
                   help="Comma-separated NCHW, e.g. '1,3,224,224'")
    p.add_argument("--n-warmup", type=int, default=None)
    p.add_argument("--n-iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   choices=[None, "cpu", "cuda"])
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-export ONNX/engines even if files already exist.")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _apply_cli_overrides(cfg: DeployConfig,
                         args: argparse.Namespace) -> DeployConfig:
    overrides: dict[str, Any] = {
        "retrain_results_dir": args.retrain_results_dir,
        "checkpoints_dir":     args.checkpoints_dir,
        "deploy_dir":          args.deploy_dir,
        "results_dir":         args.results_dir,
        "splits_dir":          args.splits_dir,
        "category":            args.category,
        "candidates_source":   args.candidates_source,
        "max_candidates":      args.max_candidates,
        "n_warmup":            args.n_warmup,
        "n_iters":             args.n_iters,
        "seed":                args.seed,
        "device":              args.device,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.precisions:
        cfg.precisions = tuple(p.strip() for p in args.precisions.split(",")
                               if p.strip())
    if args.input_shape:
        cfg.input_shape = tuple(int(x) for x in args.input_shape.split(","))
    if args.no_skip_existing:
        cfg.skip_existing_engine = False
    if args.fail_fast:
        cfg.fail_fast = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = (DeployConfig.from_file(args.config)
           if args.config.is_file() else DeployConfig())
    cfg = _apply_cli_overrides(cfg, args)

    log_path = cfg.results_dir / "deploy.log"
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    try:
        DeployPipeline(cfg, logger=logger).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Deployment failure: %s", exc)
        return 3
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during deployment")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
