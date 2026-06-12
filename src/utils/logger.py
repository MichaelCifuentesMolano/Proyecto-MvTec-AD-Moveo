"""
src/utils/logger.py
====================
Unified structured logger for the MVTec-AD pipeline.

Every log record is emitted to three destinations simultaneously:

    1. **Console**  — colourised, human-readable, level-filtered.
    2. **CSV file** — one row per record, append-safe, column-stable.
    3. **JSONL file** (optional) — one JSON object per line, full fields.

Structured fields
-----------------
Beyond the standard ``levelname / message`` pair, every record carries:

    timestamp   ISO-8601 wall-clock time (µs precision).
    epoch       Current training epoch (−1 when not in training context).
    generation  Current NSGA-II generation (−1 when not applicable).
    phase       Pipeline phase tag (e.g. "train", "eval", "nas", "deploy").
    category    MVTec-AD category name (e.g. "bottle", "cable").
    metric      Numeric metric value (NaN when not applicable).
    extra       Arbitrary JSON-serialisable dict encoded as a JSON string.

Context management
------------------
A thread-local ``LogContext`` lets any code set epoch / generation / phase /
category without passing them through every call site:

>>> with log_context(epoch=3, phase="train", category="bottle"):
...     log.info("loss", metric=0.042)

Public API
----------
>>> from utils.logger import get_logger, log_context, setup_logging
>>> setup_logging(log_dir="runs/exp01", level="INFO")
>>> log = get_logger("nas.engine")
>>> log.metric("auroc", 0.9812, phase="eval", category="carpet")
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "timestamp", "level", "logger", "epoch", "generation",
    "phase", "category", "metric", "message", "extra",
]

_SENTINEL_INT   = -1       # Unset integer context field.
_SENTINEL_FLOAT = float("nan")  # Unset metric value.

_CONSOLE_COLORS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"

# ---------------------------------------------------------------------------
# Thread-local log context
# ---------------------------------------------------------------------------

_ctx = threading.local()


def _get_ctx() -> dict:
    if not hasattr(_ctx, "stack"):
        _ctx.stack = []
    merged: dict = {
        "epoch":      _SENTINEL_INT,
        "generation": _SENTINEL_INT,
        "phase":      "",
        "category":   "",
    }
    for frame in _ctx.stack:
        merged.update({k: v for k, v in frame.items() if v is not None})
    return merged


@contextmanager
def log_context(
    *,
    epoch:      Optional[int] = None,
    generation: Optional[int] = None,
    phase:      Optional[str] = None,
    category:   Optional[str] = None,
):
    """
    Thread-local context manager that injects structured fields into every
    log record emitted within the ``with`` block.

    Contexts nest: inner values override outer ones for the same key.

    Parameters
    ----------
    epoch      : Training epoch index.
    generation : NSGA-II generation index.
    phase      : Pipeline phase label ("train", "eval", "nas", "search", …).
    category   : MVTec-AD category name ("bottle", "cable", …).

    Examples
    --------
    >>> with log_context(epoch=5, phase="train", category="carpet"):
    ...     logger.info("batch loss", metric=0.031)
    """
    if not hasattr(_ctx, "stack"):
        _ctx.stack = []
    frame = {
        k: v for k, v in {
            "epoch": epoch,
            "generation": generation,
            "phase": phase,
            "category": category,
        }.items()
        if v is not None
    }
    _ctx.stack.append(frame)
    try:
        yield
    finally:
        _ctx.stack.pop()


# ---------------------------------------------------------------------------
# Structured log record
# ---------------------------------------------------------------------------

@dataclass
class LogRecord:
    """Fully-populated structured log row written to CSV / JSONL."""
    timestamp:  str
    level:      str
    logger:     str
    epoch:      int
    generation: int
    phase:      str
    category:   str
    metric:     float
    message:    str
    extra:      str    # JSON-encoded dict or "".

    def as_csv_row(self) -> list:
        return [
            self.timestamp, self.level, self.logger,
            self.epoch, self.generation,
            self.phase, self.category,
            "" if self.metric != self.metric else round(self.metric, 8),  # NaN → ""
            self.message, self.extra,
        ]

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["metric"] != d["metric"]:   # NaN check
            d["metric"] = None
        return d


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

class _ColourConsoleHandler(logging.StreamHandler):
    """
    Stream handler that prepends ANSI colour codes in TTY environments.

    Format::

        2024-11-01 14:23:01.042  INFO  nas.engine  [nas|gen=3]  message
    """

    _USE_COLOUR: bool = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        ctx = _get_ctx()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        phase    = ctx.get("phase", "")
        gen      = ctx.get("generation", _SENTINEL_INT)
        epoch    = ctx.get("epoch", _SENTINEL_INT)
        category = ctx.get("category", "")

        # Build compact context tag.
        tags = []
        if phase:
            tags.append(phase)
        if category:
            tags.append(category)
        if epoch != _SENTINEL_INT:
            tags.append(f"ep={epoch}")
        if gen != _SENTINEL_INT:
            tags.append(f"gen={gen}")
        ctx_str = f"[{' | '.join(tags)}]  " if tags else ""

        msg = record.getMessage()
        metric_str = ""
        if hasattr(record, "metric") and record.metric == record.metric:  # not NaN
            metric_str = f"  metric={record.metric:.6g}"

        line = (
            f"{ts}  {record.levelname:<8}  "
            f"{record.name:<20}  {ctx_str}{msg}{metric_str}"
        )

        if self._USE_COLOUR:
            colour = _CONSOLE_COLORS.get(record.levelname, "")
            line = f"{colour}{_BOLD}{record.levelname:<8}{_RESET}{line[8:]}"

        return line


class _CSVHandler(logging.Handler):
    """
    Append-safe CSV handler.  Writes one row per log record.

    Thread-safe via an internal ``threading.Lock``.  Opens and closes the
    file each write to tolerate external log rotation.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists() or self._path.stat().st_size == 0:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_CSV_COLUMNS)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            lr = _build_log_record(record)
            with self._lock:
                with open(self._path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(lr.as_csv_row())
        except Exception:  # noqa: BLE001
            self.handleError(record)


class _JSONLHandler(logging.Handler):
    """
    Append-safe JSONL handler.  Each line is a self-contained JSON object.

    Easier to parse with ``pandas.read_json(..., lines=True)`` or
    ``jq`` than the CSV for exploratory analysis.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            lr = _build_log_record(record)
            line = json.dumps(lr.as_dict(), ensure_ascii=False) + "\n"
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ---------------------------------------------------------------------------
# Internal record builder
# ---------------------------------------------------------------------------

def _build_log_record(record: logging.LogRecord) -> LogRecord:
    """Combine a ``logging.LogRecord`` with thread-local context → ``LogRecord``."""
    ctx = _get_ctx()
    ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    metric = getattr(record, "metric", _SENTINEL_FLOAT)
    if metric is None:
        metric = _SENTINEL_FLOAT

    extra_raw = getattr(record, "extra", None)
    if extra_raw and isinstance(extra_raw, dict):
        try:
            extra_str = json.dumps(extra_raw, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            extra_str = str(extra_raw)
    else:
        extra_str = ""

    return LogRecord(
        timestamp  = ts,
        level      = record.levelname,
        logger     = record.name,
        epoch      = ctx.get("epoch",      _SENTINEL_INT),
        generation = ctx.get("generation", _SENTINEL_INT),
        phase      = ctx.get("phase",      ""),
        category   = ctx.get("category",   ""),
        metric     = metric,
        message    = record.getMessage(),
        extra      = extra_str,
    )


# ---------------------------------------------------------------------------
# Pipeline logger (wraps standard Logger with structured helpers)
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Thin wrapper around ``logging.Logger`` that adds structured helpers.

    Obtain an instance via ``get_logger(name)`` rather than constructing
    directly.

    Structured helpers
    ------------------
    All helpers accept ``**ctx`` keyword arguments that temporarily override
    the thread-local context for that single record:

    >>> log = get_logger("train")
    >>> log.metric("auroc", 0.9812, phase="eval", category="bottle")
    >>> log.epoch_end(epoch=5, metrics={"auroc": 0.98, "latency_ms": 4.1})
    >>> log.generation_end(gen=10, hv=0.732, pareto_size=47)
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    # ------------------------------------------------------------------
    # Standard level pass-throughs
    # ------------------------------------------------------------------

    def debug(self, msg: str, *, metric: float = _SENTINEL_FLOAT,
              extra: Optional[Dict] = None, **ctx) -> None:
        self._emit(logging.DEBUG, msg, metric, extra, ctx)

    def info(self, msg: str, *, metric: float = _SENTINEL_FLOAT,
             extra: Optional[Dict] = None, **ctx) -> None:
        self._emit(logging.INFO, msg, metric, extra, ctx)

    def warning(self, msg: str, *, metric: float = _SENTINEL_FLOAT,
                extra: Optional[Dict] = None, **ctx) -> None:
        self._emit(logging.WARNING, msg, metric, extra, ctx)

    def error(self, msg: str, *, metric: float = _SENTINEL_FLOAT,
              extra: Optional[Dict] = None, **ctx) -> None:
        self._emit(logging.ERROR, msg, metric, extra, ctx)

    def critical(self, msg: str, *, metric: float = _SENTINEL_FLOAT,
                 extra: Optional[Dict] = None, **ctx) -> None:
        self._emit(logging.CRITICAL, msg, metric, extra, ctx)

    # ------------------------------------------------------------------
    # Structured helpers
    # ------------------------------------------------------------------

    def metric(
        self,
        name: str,
        value: float,
        *,
        phase:    Optional[str] = None,
        category: Optional[str] = None,
        epoch:    Optional[int] = None,
        generation: Optional[int] = None,
        extra:    Optional[Dict] = None,
    ) -> None:
        """
        Log a named scalar metric at INFO level.

        The metric name is embedded in the message; the value is stored in
        the ``metric`` column of the CSV / JSONL files.

        Examples
        --------
        >>> log.metric("auroc", 0.9812, phase="eval", category="bottle")
        >>> log.metric("latency_ms", 4.23, phase="deploy")
        """
        ctx = {k: v for k, v in {
            "phase": phase, "category": category,
            "epoch": epoch, "generation": generation,
        }.items() if v is not None}
        self._emit(logging.INFO, f"metric/{name}={value:.6g}", value,
                   extra or {"metric_name": name}, ctx)

    def epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        *,
        phase:    str = "train",
        category: Optional[str] = None,
        extra:    Optional[Dict] = None,
    ) -> None:
        """
        Log all metrics at the end of a training epoch (one record per metric).

        Examples
        --------
        >>> log.epoch_end(5, {"loss": 0.12, "auroc": 0.97}, category="carpet")
        """
        for name, value in metrics.items():
            self.metric(
                name, value,
                phase=phase, category=category, epoch=epoch,
                extra={**(extra or {}), "epoch": epoch},
            )

    def generation_end(
        self,
        gen: int,
        *,
        hv:          Optional[float] = None,
        pareto_size: Optional[int]   = None,
        extra:       Optional[Dict]  = None,
    ) -> None:
        """
        Log the end of one NSGA-II generation.

        Examples
        --------
        >>> log.generation_end(gen=10, hv=0.732, pareto_size=47)
        """
        body = {"gen": gen}
        if hv          is not None: body["hv"]          = hv
        if pareto_size is not None: body["pareto_size"] = pareto_size
        if extra:
            body.update(extra)
        msg = (
            f"generation={gen}"
            + (f"  hv={hv:.6g}" if hv is not None else "")
            + (f"  pareto={pareto_size}" if pareto_size is not None else "")
        )
        ctx = {"generation": gen}
        self._emit(logging.INFO, msg, hv if hv is not None else _SENTINEL_FLOAT,
                   body, ctx)

    def benchmark(
        self,
        stage:          str,
        latency_ms:     float,
        throughput_fps: Optional[float] = None,
        *,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log a deployment benchmark measurement.

        Examples
        --------
        >>> log.benchmark("warm_latency", latency_ms=3.82, throughput_fps=261.5)
        """
        body: Dict[str, Any] = {
            "stage": stage, "latency_ms": latency_ms,
        }
        if throughput_fps is not None:
            body["throughput_fps"] = throughput_fps
        if extra:
            body.update(extra)
        msg = (
            f"benchmark/{stage}  latency={latency_ms:.3f} ms"
            + (f"  fps={throughput_fps:.1f}" if throughput_fps is not None else "")
        )
        self._emit(logging.INFO, msg, latency_ms, body, {"phase": "deploy"})

    def alarm(
        self,
        message:  str,
        score:    float,
        severity: str = "warning",
        *,
        category: Optional[str] = None,
        extra:    Optional[Dict] = None,
    ) -> None:
        """
        Log a detector alarm at WARNING or ERROR level.

        Examples
        --------
        >>> log.alarm("Anomaly detected", score=0.91, severity="critical")
        """
        level = logging.ERROR if severity == "critical" else logging.WARNING
        body  = {"score": score, "severity": severity, **(extra or {})}
        ctx   = {k: v for k, v in {"category": category}.items() if v is not None}
        self._emit(level, f"ALARM [{severity}] {message}", score, body, ctx)

    # ------------------------------------------------------------------
    # Standard logger pass-throughs (for compatibility with stdlib loggers)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._log.name

    def setLevel(self, level: Union[int, str]) -> None:
        self._log.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        return self._log.isEnabledFor(level)

    def getChild(self, suffix: str) -> "PipelineLogger":
        return PipelineLogger(self._log.getChild(suffix))

    # ------------------------------------------------------------------
    # Internal emit
    # ------------------------------------------------------------------

    def _emit(
        self,
        level:  int,
        msg:    str,
        metric: float,
        extra:  Optional[Dict],
        ctx_overrides: dict,
    ) -> None:
        if not self._log.isEnabledFor(level):
            return
        with log_context(**{k: v for k, v in ctx_overrides.items()
                            if k in ("epoch", "generation", "phase", "category")}):
            record = self._log.makeRecord(
                self._log.name, level,
                fn="", lno=0, msg=msg, args=(), exc_info=None,
            )
            record.metric = metric    # type: ignore[attr-defined]
            record.extra  = extra     # type: ignore[attr-defined]
            self._log.handle(record)


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

_registry: Dict[str, PipelineLogger] = {}
_registry_lock = threading.Lock()
_root_configured = False


def get_logger(name: str) -> PipelineLogger:
    """
    Return (or create) a ``PipelineLogger`` for ``name``.

    If ``setup_logging()`` has not been called yet, the logger defaults to
    console-only output at INFO level.

    Parameters
    ----------
    name : Hierarchical logger name (e.g. ``"nas.engine"``, ``"train.loop"``).

    Returns
    -------
    PipelineLogger
    """
    with _registry_lock:
        if name not in _registry:
            _registry[name] = PipelineLogger(logging.getLogger(name))
        return _registry[name]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir:      Optional[Union[str, Path]] = None,
    level:        Union[int, str]            = logging.INFO,
    console:      bool                       = True,
    csv_file:     bool                       = True,
    jsonl_file:   bool                       = False,
    csv_name:     str                        = "pipeline.csv",
    jsonl_name:   str                        = "pipeline.jsonl",
    file_level:   Union[int, str]            = logging.DEBUG,
    console_level: Union[int, str]           = logging.INFO,
    propagate:    bool                       = False,
) -> Dict[str, Path]:
    """
    Configure the root logger with console + CSV (+ optional JSONL) handlers.

    Call once at process start, before any ``get_logger()`` calls.

    Parameters
    ----------
    log_dir      : Directory for CSV / JSONL files (created if absent).
                   Required when ``csv_file=True`` or ``jsonl_file=True``.
    level        : Root logger level (default INFO).
    console      : Attach colourised console handler.
    csv_file     : Attach CSV handler.
    jsonl_file   : Attach JSONL handler.
    csv_name     : File name for the CSV log (within ``log_dir``).
    jsonl_name   : File name for the JSONL log.
    file_level   : Minimum level written to file handlers (default DEBUG).
    console_level: Minimum level written to console (default INFO).
    propagate    : Whether child loggers propagate to root.

    Returns
    -------
    dict  — ``{"csv": Path, "jsonl": Path}`` for opened file paths.

    Examples
    --------
    >>> paths = setup_logging("runs/exp01", level="DEBUG", jsonl_file=True)
    >>> log = get_logger("nas")
    >>> log.info("setup complete")
    """
    global _root_configured

    root = logging.getLogger()
    root.setLevel(_resolve_level(level))

    # Remove any pre-existing handlers to avoid duplication on re-calls.
    for h in root.handlers[:]:
        root.removeHandler(h)

    opened: Dict[str, Path] = {}

    # ── Console ───────────────────────────────────────────────────────────
    if console:
        ch = _ColourConsoleHandler(sys.stdout)
        ch.setLevel(_resolve_level(console_level))
        root.addHandler(ch)

    # ── File handlers ─────────────────────────────────────────────────────
    if csv_file or jsonl_file:
        if log_dir is None:
            raise ValueError(
                "log_dir must be specified when csv_file=True or jsonl_file=True."
            )
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if csv_file:
            csv_path = log_path / csv_name
            fh = _CSVHandler(csv_path)
            fh.setLevel(_resolve_level(file_level))
            root.addHandler(fh)
            opened["csv"] = csv_path

        if jsonl_file:
            jsonl_path = log_path / jsonl_name
            jh = _JSONLHandler(jsonl_path)
            jh.setLevel(_resolve_level(file_level))
            root.addHandler(jh)
            opened["jsonl"] = jsonl_path

    # Suppress propagation noise from third-party libraries.
    for noisy in ("PIL", "matplotlib", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _root_configured = True
    logging.getLogger(__name__).info(
        "Logging configured [level=%s, csv=%s, jsonl=%s].",
        logging.getLevelName(_resolve_level(level)),
        opened.get("csv", False),
        opened.get("jsonl", False),
    )
    return opened


def _resolve_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: '{level}'")
    return numeric


# ---------------------------------------------------------------------------
# CSV reader utility
# ---------------------------------------------------------------------------

def read_csv_log(path: Union[str, Path]) -> "list[dict]":
    """
    Read a pipeline CSV log into a list of dicts.

    Numeric columns (``epoch``, ``generation``, ``metric``) are cast to
    their proper types; empty strings become ``None``.

    Parameters
    ----------
    path : Path to the CSV file written by this module.

    Returns
    -------
    list[dict]  — one dict per log record.

    Examples
    --------
    >>> records = read_csv_log("runs/exp01/pipeline.csv")
    >>> import pandas as pd
    >>> df = pd.DataFrame(records)
    >>> df[df["phase"] == "eval"].groupby("category")["metric"].mean()
    """
    records = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Cast integer columns.
            for col in ("epoch", "generation"):
                try:
                    row[col] = int(row[col])
                except (ValueError, KeyError):
                    row[col] = _SENTINEL_INT
            # Cast metric column.
            try:
                row["metric"] = float(row["metric"]) if row.get("metric") else None
            except ValueError:
                row["metric"] = None
            # Decode extra JSON.
            extra_raw = row.get("extra", "")
            if extra_raw:
                try:
                    row["extra"] = json.loads(extra_raw)
                except json.JSONDecodeError:
                    row["extra"] = extra_raw
            else:
                row["extra"] = None
            records.append(row)
    return records


def read_jsonl_log(path: Union[str, Path]) -> "list[dict]":
    """
    Read a pipeline JSONL log into a list of dicts.

    Parameters
    ----------
    path : Path to the JSONL file written by this module.

    Returns
    -------
    list[dict]  — one dict per log record.
    """
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ---------------------------------------------------------------------------
# Convenience: filter and summarise a CSV log
# ---------------------------------------------------------------------------

def summarise_metrics(
    records: "list[dict]",
    *,
    phase:    Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metric records by (phase, category) → metric_name → mean/min/max.

    Parameters
    ----------
    records  : Output of ``read_csv_log()`` or ``read_jsonl_log()``.
    phase    : Filter to this phase (None = all phases).
    category : Filter to this category (None = all categories).

    Returns
    -------
    dict mapping metric_name → {"mean": float, "min": float, "max": float, "n": int}.

    Examples
    --------
    >>> recs = read_csv_log("runs/exp01/pipeline.csv")
    >>> stats = summarise_metrics(recs, phase="eval", category="bottle")
    >>> print(stats["auroc"])
    {"mean": 0.9812, "min": 0.972, "max": 0.991, "n": 10}
    """
    import math

    def _key_from_msg(msg: str) -> Optional[str]:
        """Extract metric name from messages like 'metric/auroc=0.98'."""
        if msg.startswith("metric/") and "=" in msg:
            return msg.split("/", 1)[1].split("=")[0]
        return None

    filtered = [
        r for r in records
        if (phase    is None or r.get("phase")    == phase)
        and (category is None or r.get("category") == category)
        and r.get("metric") is not None
    ]

    buckets: Dict[str, list] = {}
    for r in filtered:
        val = r["metric"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        name = _key_from_msg(r.get("message", "")) or r.get("message", "metric")
        buckets.setdefault(name, []).append(float(val))

    result: Dict[str, Dict[str, float]] = {}
    for name, vals in buckets.items():
        result[name] = {
            "mean": sum(vals) / len(vals),
            "min":  min(vals),
            "max":  max(vals),
            "n":    len(vals),
        }
    return result
