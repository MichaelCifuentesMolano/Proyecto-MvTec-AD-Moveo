"""
main_tracking.py
================

Entry-point orchestration script for the *robotic visual target-tracking*
validation stage of the quantized-NN / MVTec-AD pipeline.

Once a deployable TensorRT engine has been selected by ``main_deploy.py``,
this script integrates the anomaly detector into a closed-loop tracking
task. For each configured scenario (nominal + robustness perturbations) it:

* drives a camera stream (live device or recorded video),
* feeds frames through the detector,
* updates a tracker with the detection,
* derives a control command for the robotic platform,
* logs per-frame latency / energy / RAM metrics,
* optionally renders an annotated video log,
* records failure events (target lost, low confidence, control saturation),
* aggregates a session-level metrics row.

Responsibilities
----------------
1. Resolve the engine to use (CLI override, explicit candidate id, or rank 0
   from ``results/deploy/final_embedded_rank.csv``).
2. Build the tracking stack: ``CameraStream → DetectorInterface →
   TrackerCore → ControlLoop`` (each instantiated from its dedicated module).
3. Iterate scenarios from ``robustness_test`` plus the nominal pass.
4. Persist:
   * one row per session in ``results/tracking/tracking_metrics.csv``,
   * one row per failure event in ``results/tracking/failure_cases.csv``,
   * one annotated video per session in ``results/tracking/video_logs/``.

Expected module interfaces (downstream contract)
------------------------------------------------
``src.tracking.camera_stream``
    ``class CameraStream``
        ``__init__(source: str | int, fps: float | None = None,
                   resolution: tuple[int, int] | None = None,
                   max_frames: int | None = None, **kwargs)``
        Context manager. Iterating yields ``Frame`` objects with attributes
        ``index: int``, ``timestamp: float``, ``image: np.ndarray (H, W, 3)
        BGR``, ``metadata: dict``. Exposes ``.fps`` and ``.resolution``.

``src.tracking.detector_interface``
    ``class DetectorInterface``
        ``__init__(engine_path: Path, input_shape: tuple[int, ...],
                   device: str, score_threshold: float = 0.5, **kwargs)``
        ``warmup(n: int = 10) -> None``
        ``detect(image: np.ndarray) -> DetectionResult``
        ``close() -> None``
        ``DetectionResult`` exposes ``score: float``, ``is_anomaly: bool``,
        ``bbox: tuple[int,int,int,int] | None``, ``heatmap: np.ndarray|None``,
        ``latency_ms: float``, ``metadata: dict``.

``src.tracking.tracker_core``
    ``class TrackerCore``
        ``__init__(detector: DetectorInterface, algorithm: str = "kcf",
                   reinit_every: int = 10, lost_threshold: float = 0.3,
                   **kwargs)``
        ``initialize(frame: np.ndarray, bbox: tuple | None = None) -> bool``
        ``update(frame: np.ndarray) -> TrackingState``
        ``reset() -> None``
        ``is_tracking: bool``
        ``TrackingState`` exposes ``success: bool``, ``bbox``, ``confidence``,
        ``target_position_px``, ``detection: DetectionResult | None``,
        ``status: str`` (``"tracking"|"searching"|"lost"``), ``metadata``.

``src.tracking.control_loop``
    ``class ControlLoop``
        ``__init__(controller_cfg: dict, frame_size: tuple[int, int])``
        ``step(tracking: TrackingState) -> ControlCommand``
        ``reset() -> None``
        ``ControlCommand`` exposes ``pan, tilt, speed: float``,
        ``is_search_mode: bool``, ``saturated: bool``, ``metadata``.

``src.tracking.robustness_test``
    ``list_scenarios() -> list[str]``
    ``apply_scenario(name: str, image: np.ndarray, frame_index: int,
                     **kwargs) -> np.ndarray``

``src.profiling.latency_meter``
    ``class LatencyTimer``
        ``start() -> None`` / ``stop() -> float`` (returns ms)
        ``stats() -> dict`` with keys ``mean, p50, p95, p99, n``.
        ``reset() -> None``

``src.profiling.energy_meter``
    ``class EnergyMeter``
        Context manager. ``__enter__/__exit__`` start/stop sampling.
        ``stop() -> dict`` returns ``energy_mj, duration_s, avg_power_w,
        source``. Safe no-op fallback on hosts without sensors.

Assumptions
-----------
- ``main_deploy.py`` has been executed; ``results/deploy/final_embedded_rank.csv``
  and ``deployment/models/*.engine`` exist.
- The camera source can be an integer device id (e.g. ``0``), a path to a
  video file, or any URI handled by the ``CameraStream`` implementation.
- Ground-truth bounding boxes (per frame) are *optional* and supplied via a
  JSON manifest when richer metrics (IoU, pixel error) are required.
- The script is interruptible: pressing ``Ctrl+C`` cleanly closes the
  current scenario, flushes CSVs, finalizes the video file, and exits.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

import numpy as np

# OpenCV is only used for video annotation/recording; gracefully fall back.
try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Project module imports.
# ---------------------------------------------------------------------------
# NOTE: the previous imports referenced idealized names (TrackerCore,
# ControlLoop-as-step-controller, list_scenarios/apply_scenario) that do not
# exist in the implemented modules — this script could not be imported.
# Thin adapters over the *real* APIs are defined below.
from src.tracking.camera_stream import CameraStream
from src.tracking.detector_interface import DetectorInterface, DetectorConfig
from src.tracking.tracker_core import make_tracker, TrackState
from src.tracking.control_loop import PIDController
from src.profiling.latency_meter import LatencyTimer
from src.profiling.energy_meter import EnergyMeter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
# ``config/`` (singular) — ``configs/`` silently ignored config/tracking.yaml.
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config" / "tracking.yaml"
DEFAULT_DEPLOY_RESULTS: Path = PROJECT_ROOT / "results" / "deploy"
DEFAULT_RESULTS_DIR: Path = PROJECT_ROOT / "results" / "tracking"
DEFAULT_VIDEO_DIR: Path = DEFAULT_RESULTS_DIR / "video_logs"

DEFAULT_INPUT_SHAPE: tuple[int, ...] = (1, 3, 224, 224)
DEFAULT_FRAME_SIZE: tuple[int, int] = (640, 480)  # (W, H)

NOMINAL_SCENARIO: str = "nominal"


# ---------------------------------------------------------------------------
# Adapters over the real src.tracking APIs
# ---------------------------------------------------------------------------
_SCENARIOS: tuple[str, ...] = ("occlusion", "low_light", "motion_blur",
                               "gaussian_noise", "flicker")


def list_scenarios() -> list[str]:
    """Available robustness perturbations (deterministic per frame index)."""
    return list(_SCENARIOS)


def apply_scenario(name: str, image: np.ndarray,
                   frame_index: int = 0, **_kw) -> np.ndarray:
    """Apply a named perturbation. Self-contained and reproducible:
    randomness is seeded with the frame index."""
    rng = np.random.default_rng(frame_index)
    img = image.astype(np.float32)
    if name == "low_light":
        out = img * 0.35
    elif name == "gaussian_noise":
        out = img + rng.normal(0.0, 12.0, size=img.shape)
    elif name == "flicker":
        out = img * (1.0 + 0.4 * math.sin(frame_index / 3.0))
    elif name == "motion_blur":
        k = 9
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        if _HAVE_CV2:
            out = cv2.filter2D(img, -1, kernel)
        else:  # crude horizontal box blur fallback
            out = img.copy()
            for s in range(1, k // 2 + 1):
                out += np.roll(img, s, axis=1) + np.roll(img, -s, axis=1)
            out /= float(k)
    elif name == "occlusion":
        out = img.copy()
        h, w = out.shape[:2]
        ow, oh = int(w * 0.30), int(h * 0.30)
        x0 = int(rng.integers(0, max(w - ow, 1)))
        y0 = int(rng.integers(0, max(h - oh, 1)))
        out[y0:y0 + oh, x0:x0 + ow] = 0.0
    else:
        out = img
    return np.clip(out, 0, 255).astype(np.uint8)


def _primary_bbox(detection: Any) -> tuple[int, int, int, int] | None:
    """Highest-confidence ROI of a DetectionResult as (x, y, w, h)."""
    rois = getattr(detection, "rois", None) or []
    if not rois:
        return None
    x1, y1, x2, y2 = rois[0].bbox
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


@dataclass
class TrackingState:
    """Expected by the session loop; produced by :class:`TrackerCore`."""
    success: bool
    bbox: tuple[int, int, int, int] | None      # (x, y, w, h)
    confidence: float | None
    status: str                                  # tracking | searching | lost
    detection: Any = None
    metadata: dict = field(default_factory=dict)


class TrackerCore:
    """Single-target adapter over :class:`MultiObjectTracker`.

    The real tracker is multi-object and self-initialising (track birth /
    death from detections); this adapter exposes the primary (highest
    confidence) active track through the simple API the session loop uses.
    """

    def __init__(self, detector: Any, algorithm: str = "sort",
                 reinit_every: int = 10, lost_threshold: float = 0.3,
                 frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
                 **kwargs: Any) -> None:
        self._mot = make_tracker(
            algorithm=algorithm,
            frame_width=int(frame_size[0]),
            frame_height=int(frame_size[1]),
            **kwargs,
        )
        self._lost_threshold = lost_threshold
        self._last_state: TrackingState | None = None

    @property
    def is_tracking(self) -> bool:
        return bool(self._mot.active_tracks)

    def initialize(self, frame: np.ndarray,
                   bbox: tuple | None = None) -> bool:
        # MultiObjectTracker self-initialises from detections; kept for API
        # compatibility with the session loop.
        return True

    def update(self, frame: np.ndarray, detection: Any) -> TrackingState:
        self._mot.update(detection, frame=frame)
        active = self._mot.active_tracks
        if not active:
            state = TrackingState(success=False, bbox=None, confidence=None,
                                  status="searching", detection=detection)
        else:
            primary = max(active, key=lambda t: t.confidence)
            b = primary.bbox
            confirmed = primary.state == TrackState.CONFIRMED
            low_conf = primary.confidence < self._lost_threshold
            state = TrackingState(
                success=confirmed and not low_conf,
                bbox=(int(b.x1), int(b.y1),
                      int(b.x2 - b.x1), int(b.y2 - b.y1)),
                confidence=float(primary.confidence),
                status=("tracking" if confirmed and not low_conf
                        else "lost" if low_conf else "searching"),
                detection=detection,
                metadata={"track_id": primary.track_id,
                          "n_active": len(active)},
            )
        self._last_state = state
        return state

    def reset(self) -> None:
        self._mot.reset()
        self._last_state = None


@dataclass
class _StepCommand:
    """Per-frame control command produced by :class:`StepControlLoop`."""
    pan: float = 0.0
    tilt: float = 0.0
    speed: float = 0.0
    is_search_mode: bool = False
    saturated: bool = False
    metadata: dict = field(default_factory=dict)


class StepControlLoop:
    """Step-based pan/tilt controller built on :class:`PIDController`.

    (The module-level ``ControlLoop`` owns its own capture loop and is not
    usable per-frame; this adapter keeps the session loop in charge.)
    """

    def __init__(self, controller_cfg: dict[str, Any],
                 frame_size: tuple[int, int]) -> None:
        kp = float(controller_cfg.get("kp", 0.6))
        ki = float(controller_cfg.get("ki", 0.0))
        kd = float(controller_cfg.get("kd", 0.05))
        self._pan = PIDController(kp, ki, kd, output_limit=1.0)
        self._tilt = PIDController(kp, ki, kd, output_limit=1.0)
        self._deadband = float(controller_cfg.get("deadband_px", 8))
        self._max_pan = float(controller_cfg.get("max_pan_deg_s", 60.0))
        self._max_tilt = float(controller_cfg.get("max_tilt_deg_s", 60.0))
        self._cx = frame_size[0] / 2.0
        self._cy = frame_size[1] / 2.0

    def step(self, tracking: TrackingState) -> _StepCommand:
        if not tracking.success or tracking.bbox is None:
            return _StepCommand(is_search_mode=True)
        x, y, w, h = tracking.bbox
        err_x = (x + w / 2.0) - self._cx
        err_y = (y + h / 2.0) - self._cy
        if abs(err_x) < self._deadband:
            err_x = 0.0
        if abs(err_y) < self._deadband:
            err_y = 0.0
        # Normalised error -> PID -> deg/s command.
        pan = self._pan.compute(err_x / max(self._cx, 1.0)) * self._max_pan
        tilt = self._tilt.compute(err_y / max(self._cy, 1.0)) * self._max_tilt
        saturated = (abs(pan) >= 0.98 * self._max_pan
                     or abs(tilt) >= 0.98 * self._max_tilt)
        return _StepCommand(pan=pan, tilt=tilt, saturated=saturated,
                            metadata={"err_x": err_x, "err_y": err_y})

    def reset(self) -> None:
        self._pan.reset()
        self._tilt.reset()


def _backup_existing(path: Path, logger: logging.Logger) -> None:
    """Rename an existing results file instead of silently truncating it."""
    if path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
        path.rename(backup)
        logger.info("Existing %s backed up to %s", path.name, backup.name)

METRIC_COLUMNS: tuple[str, ...] = (
    "session_id", "scenario", "status",
    "candidate_id", "precision", "engine_path",
    "n_frames", "duration_s", "achieved_fps",
    "detection_latency_ms_mean", "detection_latency_ms_p95",
    "end_to_end_latency_ms_mean", "end_to_end_latency_ms_p95",
    "tracker_success_rate", "n_lost", "n_recoveries",
    "mean_iou", "mean_tracking_error_px",
    "control_saturation_rate",
    "energy_mj", "avg_power_w", "energy_mj_per_frame",
    "video_path",
    "config_snapshot", "error",
)

FAILURE_COLUMNS: tuple[str, ...] = (
    "session_id", "scenario", "frame_index", "timestamp_s",
    "failure_type", "bbox", "confidence",
    "detection_latency_ms", "tracking_status",
    "control_saturated", "tracking_error_px",
    "notes",
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrackingConfig:
    """Configuration for the closed-loop tracking validation stage."""

    # I/O
    deploy_results_dir: Path = DEFAULT_DEPLOY_RESULTS
    results_dir: Path = DEFAULT_RESULTS_DIR
    video_dir: Path = DEFAULT_VIDEO_DIR

    # Engine selection
    engine_path: Path | None = None    # explicit override
    candidate_id: str | None = None    # pick a specific candidate from rank
    precision: str | None = None       # filter by precision
    rank_index: int = 0                # default = top-ranked artifact

    # Camera / video source
    source: str | int = 0              # device id or video file path
    source_fps: float | None = None
    resolution: tuple[int, int] = DEFAULT_FRAME_SIZE
    max_frames: int | None = 1500      # ~50 s at 30 fps

    # Detector / tracker
    input_shape: tuple[int, ...] = DEFAULT_INPUT_SHAPE
    score_threshold: float = 0.5
    tracker_algorithm: str = "kcf"
    reinit_every: int = 10
    lost_threshold: float = 0.3
    warmup_iters: int = 20

    # Control loop
    controller: dict[str, Any] = field(default_factory=lambda: {
        "type": "pid",
        "kp": 0.6, "ki": 0.0, "kd": 0.05,
        "max_pan_deg_s": 60.0,
        "max_tilt_deg_s": 60.0,
        "deadband_px": 8,
    })

    # Scenarios
    scenarios: tuple[str, ...] = (
        NOMINAL_SCENARIO, "occlusion", "low_light",
        "motion_blur", "gaussian_noise",
    )

    # Failure thresholds
    min_confidence_for_track: float = 0.25
    max_consecutive_lost: int = 30

    # Repetitions per scenario (>=3 recommended so per-scenario statistics
    # and group comparisons in main_report have something to work with).
    repeats_per_scenario: int = 3

    # Ground truth (optional, for IoU / pixel error)
    ground_truth_path: Path | None = None

    # Profiling
    energy_sample_hz: float = 10.0

    # Output behavior
    record_video: bool = True
    annotate: bool = True
    fail_fast: bool = False

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path) -> "TrackingConfig":
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
    def from_dict(cls, data: dict[str, Any]) -> "TrackingConfig":
        """Build a TrackingConfig from a plain dict.

        Accepts both the flat schema (dataclass fields at top level) and the
        nested schema used by ``config/tracking.yaml`` (``camera:``,
        ``detector:``, ``tracker:``, ``control:``), which is flattened here
        so the YAML is actually honoured instead of falling into ``extra``.
        """
        kwargs = dict(data)

        # ---- Nested-schema support --------------------------------------
        cam = kwargs.pop("camera", None) or {}
        if cam:
            src = cam.get("source")
            if src == "video_file":
                vf = (cam.get("video_file") or {}).get("path")
                if vf:
                    kwargs.setdefault("source", vf)
            elif src == "usb":
                kwargs.setdefault("source",
                                  (cam.get("usb") or {}).get("device_id", 0))
            elif src is not None:
                kwargs.setdefault("source", src)
            if cam.get("width") and cam.get("height"):
                kwargs.setdefault("resolution",
                                  (cam["width"], cam["height"]))
            if cam.get("fps"):
                kwargs.setdefault("source_fps", cam["fps"])

        det = kwargs.pop("detector", None) or {}
        if det:
            if "score_threshold" in det:
                kwargs.setdefault("score_threshold", det["score_threshold"])
            if det.get("model_path"):
                kwargs.setdefault("engine_path", det["model_path"])

        trk = kwargs.pop("tracker", None) or {}
        if trk:
            if "algorithm" in trk:
                kwargs.setdefault("tracker_algorithm", trk["algorithm"])

        ctl = kwargs.pop("control", None) or {}
        pid = (ctl.get("pid") or {}) if ctl else {}
        if pid:
            controller = dict(TrackingConfig().controller)
            controller.update({k: v for k, v in pid.items()
                               if k in ("kp", "ki", "kd")})
            kwargs.setdefault("controller", controller)
        if ctl and "follow_min_score" in ctl:
            kwargs.setdefault("min_confidence_for_track",
                              ctl["follow_min_score"])

        kwargs.pop("publisher", None)     # hardware publisher: not used here
        rob = kwargs.pop("robustness", None) or {}
        if rob and "seed" in rob:
            kwargs.setdefault("seed", rob["seed"])
        # ------------------------------------------------------------------

        for key in ("deploy_results_dir", "results_dir", "video_dir",
                    "engine_path", "ground_truth_path"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        for key in ("input_shape", "resolution"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        if "scenarios" in kwargs and kwargs["scenarios"] is not None:
            kwargs["scenarios"] = tuple(kwargs["scenarios"])
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
    logger = logging.getLogger("tracking")
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
# Engine resolution
# ---------------------------------------------------------------------------
def _resolve_engine(cfg: TrackingConfig,
                    logger: logging.Logger) -> dict[str, Any]:
    """Return a dict ``{engine_path, candidate_id, precision, source}``."""
    if cfg.engine_path is not None:
        if not cfg.engine_path.is_file():
            raise FileNotFoundError(
                f"Engine file not found: {cfg.engine_path}"
            )
        logger.info("Using user-specified engine: %s", cfg.engine_path)
        return {
            "engine_path": cfg.engine_path,
            "candidate_id": cfg.candidate_id or cfg.engine_path.stem,
            "precision": cfg.precision,
            "source": "cli",
        }

    rank_path = cfg.deploy_results_dir / "final_embedded_rank.csv"
    if not rank_path.is_file():
        raise FileNotFoundError(
            f"Deploy ranking not found at {rank_path}. "
            "Run main_deploy.py first or pass --engine-path."
        )

    candidates: list[dict[str, Any]] = []
    with rank_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") and not row["status"].startswith("ok"):
                continue
            candidates.append(row)

    if cfg.candidate_id is not None:
        candidates = [r for r in candidates
                      if r.get("candidate_id") == cfg.candidate_id]
    if cfg.precision is not None:
        candidates = [r for r in candidates
                      if r.get("precision") == cfg.precision]

    if not candidates:
        raise RuntimeError(
            f"No matching engine in {rank_path} "
            f"(candidate_id={cfg.candidate_id}, precision={cfg.precision})"
        )
    pick = candidates[min(cfg.rank_index, len(candidates) - 1)]
    engine_path = Path(pick["engine_path"])
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine file missing: {engine_path}")
    logger.info(
        "Selected engine rank=%s candidate=%s precision=%s -> %s",
        pick.get("rank"), pick.get("candidate_id"),
        pick.get("precision"), engine_path,
    )
    return {
        "engine_path": engine_path,
        "candidate_id": pick.get("candidate_id"),
        "precision": pick.get("precision"),
        "source": "ranked",
    }


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def _load_ground_truth(path: Path | None) -> dict[int, tuple[int, int, int, int]]:
    """Load optional per-frame ground-truth bboxes from a JSON manifest.

    Manifest format::

        {"frames": {"0": [x, y, w, h], "12": [...], ...}}
    """
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("frames", payload)
    out: dict[int, tuple[int, int, int, int]] = {}
    for k, v in raw.items():
        if v is None:
            continue
        try:
            out[int(k)] = tuple(int(x) for x in v[:4])  # type: ignore[assignment]
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _bbox_iou(a: tuple[int, int, int, int] | None,
              b: tuple[int, int, int, int] | None) -> float | None:
    """Intersection-over-union for two ``(x, y, w, h)`` boxes."""
    if a is None or b is None:
        return None
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _bbox_center(bbox: tuple[int, int, int, int] | None
                 ) -> tuple[float, float] | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _center_distance(a: tuple[int, int, int, int] | None,
                     b: tuple[int, int, int, int] | None) -> float | None:
    ca, cb = _bbox_center(a), _bbox_center(b)
    if ca is None or cb is None:
        return None
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def _annotate_frame(frame: np.ndarray,
                    *,
                    tracking_status: str,
                    bbox: tuple[int, int, int, int] | None,
                    detection_score: float | None,
                    latency_ms: float | None,
                    fps: float | None,
                    scenario: str,
                    control_text: str | None) -> np.ndarray:
    """Draw a lightweight HUD over the frame. Falls back to a copy if no cv2."""
    if not _HAVE_CV2:
        return frame
    out = frame.copy()
    if bbox is not None:
        x, y, w, h = bbox
        color = (0, 255, 0) if tracking_status == "tracking" else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    lines = [f"scenario={scenario}", f"status={tracking_status}"]
    if detection_score is not None:
        lines.append(f"score={detection_score:.3f}")
    if latency_ms is not None:
        lines.append(f"lat={latency_ms:.1f} ms")
    if fps is not None:
        lines.append(f"fps={fps:.1f}")
    if control_text:
        lines.append(control_text)

    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Failure event recording
# ---------------------------------------------------------------------------
@dataclass
class FailureEvent:
    """In-memory record of a tracking failure event."""

    session_id: str
    scenario: str
    frame_index: int
    timestamp_s: float
    failure_type: str
    bbox: tuple[int, int, int, int] | None
    confidence: float | None
    detection_latency_ms: float | None
    tracking_status: str
    control_saturated: bool
    tracking_error_px: float | None
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        if d["bbox"] is not None:
            d["bbox"] = json.dumps(list(d["bbox"]))
        return d


# ---------------------------------------------------------------------------
# Session result
# ---------------------------------------------------------------------------
@dataclass
class SessionResult:
    """Aggregated metrics from a single tracking session."""

    session_id: str
    scenario: str
    status: str = "pending"
    n_frames: int = 0
    duration_s: float = 0.0
    achieved_fps: float = 0.0
    det_latency_mean: float | None = None
    det_latency_p95: float | None = None
    e2e_latency_mean: float | None = None
    e2e_latency_p95: float | None = None
    tracker_success_rate: float = 0.0
    n_lost: int = 0
    n_recoveries: int = 0
    mean_iou: float | None = None
    mean_tracking_error_px: float | None = None
    control_saturation_rate: float = 0.0
    energy_mj: float | None = None
    avg_power_w: float | None = None
    energy_mj_per_frame: float | None = None
    video_path: Path | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TrackingPipeline:
    """Runs the closed-loop tracking validation across all configured scenarios."""

    def __init__(self, cfg: TrackingConfig,
                 logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("tracking")

        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        cfg.video_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = cfg.results_dir / "tracking_metrics.csv"
        self._failures_path = cfg.results_dir / "failure_cases.csv"
        _backup_existing(self._metrics_path, self.log)
        _backup_existing(self._failures_path, self.log)
        self._metrics_writer = _open_csv(self._metrics_path, METRIC_COLUMNS)
        self._failures_writer = _open_csv(self._failures_path, FAILURE_COLUMNS)

        self._session_records: list[SessionResult] = []
        self._engine_info: dict[str, Any] = {}
        self._gt: dict[int, tuple[int, int, int, int]] = {}

    # ------------------------------------------------------------------
    def _record_failure(self, event: FailureEvent) -> None:
        row = event.as_row()
        self._failures_writer.writerow(row)
        self._metrics_writer.flush_handle()
        self._failures_writer.flush_handle()

    # ------------------------------------------------------------------
    def _record_metrics(self, session: SessionResult) -> None:
        row = {
            "session_id": session.session_id,
            "scenario": session.scenario,
            "status": session.status,
            "candidate_id": self._engine_info.get("candidate_id"),
            "precision": self._engine_info.get("precision"),
            "engine_path": str(self._engine_info.get("engine_path", "")),
            "n_frames": session.n_frames,
            "duration_s": round(session.duration_s, 4),
            "achieved_fps": round(session.achieved_fps, 3),
            "detection_latency_ms_mean": session.det_latency_mean,
            "detection_latency_ms_p95":  session.det_latency_p95,
            "end_to_end_latency_ms_mean": session.e2e_latency_mean,
            "end_to_end_latency_ms_p95":  session.e2e_latency_p95,
            "tracker_success_rate": round(session.tracker_success_rate, 4),
            "n_lost": session.n_lost,
            "n_recoveries": session.n_recoveries,
            "mean_iou": session.mean_iou,
            "mean_tracking_error_px": session.mean_tracking_error_px,
            "control_saturation_rate": round(session.control_saturation_rate, 4),
            "energy_mj": session.energy_mj,
            "avg_power_w": session.avg_power_w,
            "energy_mj_per_frame": session.energy_mj_per_frame,
            "video_path": str(session.video_path) if session.video_path else "",
            "config_snapshot": json.dumps(self.cfg.to_dict(), default=str),
            "error": session.error,
        }
        self._metrics_writer.writerow(row)
        self._metrics_writer.flush_handle()

    # ------------------------------------------------------------------
    def _run_session(self,
                     scenario: str,
                     detector: DetectorInterface,
                     tracker: TrackerCore,
                     controller: StepControlLoop,
                     repeat: int = 0) -> SessionResult:
        """Run one camera pass under a given scenario and return its summary."""
        session_id = f"{scenario}_r{repeat}_{int(time.time())}"
        session = SessionResult(session_id=session_id, scenario=scenario)

        # Reset stateful components for an independent pass.
        tracker.reset()
        controller.reset()

        # Open video writer if requested.
        video_writer = None
        video_path: Path | None = None
        if self.cfg.record_video and _HAVE_CV2:
            video_path = self.cfg.video_dir / f"{session_id}.mp4"
            session.video_path = video_path

        det_timer = LatencyTimer()
        e2e_timer = LatencyTimer()

        ious: list[float] = []
        track_errors: list[float] = []
        saturations: list[bool] = []
        tracking_flags: list[bool] = []
        consecutive_lost = 0
        had_lost = False
        was_lost = False

        t_start = time.perf_counter()
        try:
            with CameraStream(
                source=self.cfg.source,
                fps=self.cfg.source_fps,
                resolution=self.cfg.resolution,
                max_frames=self.cfg.max_frames,
            ) as stream, EnergyMeter(
                device=self.cfg.device,
                sample_hz=self.cfg.energy_sample_hz,
            ) as energy:

                for frame in stream:
                    e2e_timer.start()

                    # --- 1. perturbation ---------------------------------
                    image = frame.image
                    if scenario != NOMINAL_SCENARIO:
                        image = apply_scenario(scenario, image,
                                               frame_index=frame.index)

                    # --- 2. detection ------------------------------------
                    det_timer.start()
                    detection = detector.detect(image)
                    det_dt = det_timer.stop()

                    # --- 3. (re)initialize tracker if needed -------------
                    det_bbox = _primary_bbox(detection)
                    if not tracker.is_tracking and det_bbox is not None \
                            and detection.score >= self.cfg.score_threshold:
                        tracker.initialize(image, det_bbox)

                    # --- 4. tracker update -------------------------------
                    state = tracker.update(image, detection)
                    tracking_flags.append(bool(state.success))

                    # --- 5. control step ---------------------------------
                    cmd = controller.step(state)
                    saturations.append(bool(getattr(cmd, "saturated", False)))

                    # End-to-end latency covers perception + tracking +
                    # control ONLY. Video encoding/disk I/O below must not
                    # contaminate the reported loop latency.
                    e2e_timer.stop()

                    # --- 6. ground-truth metrics -------------------------
                    gt_box = self._gt.get(frame.index)
                    iou = _bbox_iou(state.bbox, gt_box)
                    if iou is not None:
                        ious.append(iou)
                    err_px = _center_distance(state.bbox, gt_box)
                    if err_px is not None:
                        track_errors.append(err_px)

                    # --- 7. lost / failure tracking ----------------------
                    is_lost = (not state.success
                               or (state.confidence is not None
                                   and state.confidence
                                   < self.cfg.min_confidence_for_track))
                    if is_lost:
                        consecutive_lost += 1
                        had_lost = True
                        if consecutive_lost == 1:
                            session.n_lost += 1
                            self._record_failure(FailureEvent(
                                session_id=session_id, scenario=scenario,
                                frame_index=frame.index,
                                timestamp_s=frame.timestamp,
                                failure_type="target_lost",
                                bbox=state.bbox,
                                confidence=state.confidence,
                                detection_latency_ms=det_dt,
                                tracking_status=getattr(state, "status",
                                                        "lost"),
                                control_saturated=saturations[-1],
                                tracking_error_px=err_px,
                                notes=f"score={detection.score:.3f}",
                            ))
                        if consecutive_lost > self.cfg.max_consecutive_lost:
                            self._record_failure(FailureEvent(
                                session_id=session_id, scenario=scenario,
                                frame_index=frame.index,
                                timestamp_s=frame.timestamp,
                                failure_type="lost_too_long",
                                bbox=state.bbox,
                                confidence=state.confidence,
                                detection_latency_ms=det_dt,
                                tracking_status="lost",
                                control_saturated=saturations[-1],
                                tracking_error_px=err_px,
                                notes=(f"consecutive_lost="
                                       f"{consecutive_lost}"),
                            ))
                            tracker.reset()
                            controller.reset()
                            consecutive_lost = 0
                    else:
                        if was_lost:
                            session.n_recoveries += 1
                        consecutive_lost = 0
                    was_lost = is_lost

                    # --- 8. annotation + video ---------------------------
                    if video_writer is None and video_path is not None \
                            and _HAVE_CV2:
                        h, w = image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        fps_out = self.cfg.source_fps or stream.fps or 30.0
                        video_writer = cv2.VideoWriter(
                            str(video_path), fourcc, fps_out, (w, h),
                        )
                    if video_writer is not None:
                        annotated = (
                            _annotate_frame(
                                image,
                                tracking_status=getattr(state, "status",
                                                        "unknown"),
                                bbox=state.bbox,
                                detection_score=detection.score,
                                latency_ms=det_dt,
                                fps=(frame.index /
                                     max(time.perf_counter() - t_start, 1e-6)),
                                scenario=scenario,
                                control_text=(f"pan={cmd.pan:.2f} "
                                              f"tilt={cmd.tilt:.2f}"),
                            )
                            if self.cfg.annotate else image
                        )
                        video_writer.write(annotated)

                    session.n_frames += 1

                # --- end of stream loop ---------------------------------
                energy_payload = energy.stop() if hasattr(energy, "stop") else {}

        except KeyboardInterrupt:
            self.log.warning("[%s] interrupted by user", session_id)
            session.status = "interrupted"
            session.error = "KeyboardInterrupt"
            energy_payload = {}
        except Exception as exc:  # noqa: BLE001
            self.log.exception("[%s] session crashed", session_id)
            session.status = "failed"
            session.error = f"{type(exc).__name__}: {exc}"
            energy_payload = {}
        else:
            session.status = "ok" if not had_lost else "ok_with_losses"
        finally:
            if video_writer is not None:
                video_writer.release()

        # --- aggregate metrics -------------------------------------------
        session.duration_s = time.perf_counter() - t_start
        session.achieved_fps = (session.n_frames / session.duration_s
                                if session.duration_s > 0 else 0.0)
        det_stats = det_timer.stats() if hasattr(det_timer, "stats") else {}
        e2e_stats = e2e_timer.stats() if hasattr(e2e_timer, "stats") else {}
        session.det_latency_mean = det_stats.get("mean")
        session.det_latency_p95 = det_stats.get("p95")
        session.e2e_latency_mean = e2e_stats.get("mean")
        session.e2e_latency_p95 = e2e_stats.get("p95")
        if tracking_flags:
            session.tracker_success_rate = (sum(tracking_flags)
                                             / len(tracking_flags))
        if saturations:
            session.control_saturation_rate = (sum(saturations)
                                                / len(saturations))
        if ious:
            session.mean_iou = float(np.mean(ious))
        if track_errors:
            session.mean_tracking_error_px = float(np.mean(track_errors))
        session.energy_mj = energy_payload.get("energy_mj")
        session.avg_power_w = energy_payload.get("avg_power_w")
        if (session.energy_mj is not None and session.n_frames > 0):
            session.energy_mj_per_frame = (session.energy_mj
                                            / session.n_frames)

        self.log.info(
            "[%s] %s — frames=%d, fps=%.1f, det_p95=%.1fms, "
            "success=%.1f%%, lost=%d, energy=%.2f mJ",
            session_id, session.status, session.n_frames,
            session.achieved_fps,
            _safe(session.det_latency_p95),
            100.0 * session.tracker_success_rate,
            session.n_lost,
            _safe(session.energy_mj),
        )
        return session

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Execute every configured scenario and persist results."""
        t0 = time.perf_counter()
        self.log.info("=== Tracking validation — START ===")
        self.log.info("Config: %s",
                      json.dumps(self.cfg.to_dict(), indent=2, default=str))

        (self.cfg.results_dir / "tracking_config.json").write_text(
            json.dumps(self.cfg.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        self._engine_info = _resolve_engine(self.cfg, self.log)
        self._gt = _load_ground_truth(self.cfg.ground_truth_path)
        if self._gt:
            self.log.info("Loaded %d ground-truth bbox(es).", len(self._gt))

        # Validate scenarios up-front so a typo doesn't blow up mid-run.
        try:
            available = set(list_scenarios()) | {NOMINAL_SCENARIO}
        except Exception:  # noqa: BLE001
            available = {NOMINAL_SCENARIO}
        unknown = [s for s in self.cfg.scenarios if s not in available]
        if unknown:
            self.log.warning("Unknown scenario(s) ignored: %s", unknown)
        scenarios = [s for s in self.cfg.scenarios if s in available]
        if not scenarios:
            scenarios = [NOMINAL_SCENARIO]

        # Build the long-lived components once; sessions reset their state.
        detector = DetectorInterface.from_trt_engine(
            self._engine_info["engine_path"],
            DetectorConfig(score_threshold=self.cfg.score_threshold),
        )
        if hasattr(detector, "warmup"):
            try:
                detector.warmup(self.cfg.warmup_iters)
            except Exception:  # noqa: BLE001
                self.log.warning("Detector warmup failed — continuing.")

        tracker = TrackerCore(
            detector=detector,
            algorithm=self.cfg.tracker_algorithm,
            reinit_every=self.cfg.reinit_every,
            lost_threshold=self.cfg.lost_threshold,
            frame_size=self.cfg.resolution,
        )
        controller = StepControlLoop(
            controller_cfg=self.cfg.controller,
            frame_size=self.cfg.resolution,
        )

        n_repeats = max(1, int(self.cfg.repeats_per_scenario))
        aborted = False
        try:
            for i, scenario in enumerate(scenarios):
                for rep in range(n_repeats):
                    self.log.info("---- Scenario %d/%d: %s (repeat %d/%d) ----",
                                  i + 1, len(scenarios), scenario,
                                  rep + 1, n_repeats)
                    try:
                        session = self._run_session(scenario, detector,
                                                    tracker, controller,
                                                    repeat=rep)
                    except KeyboardInterrupt:
                        raise
                    except Exception:  # noqa: BLE001
                        self.log.exception("Unhandled error in scenario %s",
                                           scenario)
                        session = SessionResult(
                            session_id=f"{scenario}_r{rep}_failed",
                            scenario=scenario,
                            status="failed_unhandled",
                            error=traceback.format_exc(limit=3),
                        )
                    self._session_records.append(session)
                    self._record_metrics(session)

                    if self.cfg.fail_fast and \
                            not session.status.startswith("ok"):
                        self.log.error(
                            "fail_fast=True and scenario %s failed (%s) "
                            "— abort.", scenario, session.status,
                        )
                        aborted = True
                        break
                if aborted:
                    break
        finally:
            if hasattr(detector, "close"):
                try:
                    detector.close()
                except Exception:  # noqa: BLE001
                    pass
            self._metrics_writer.close()
            self._failures_writer.close()

        elapsed = time.perf_counter() - t0
        n_ok = sum(1 for s in self._session_records
                   if s.status.startswith("ok"))
        summary = {
            "engine": {
                "path": str(self._engine_info.get("engine_path")),
                "candidate_id": self._engine_info.get("candidate_id"),
                "precision": self._engine_info.get("precision"),
            },
            "n_sessions": len(self._session_records),
            "n_ok": n_ok,
            "n_failed": len(self._session_records) - n_ok,
            "scenarios": scenarios,
            "elapsed_seconds": round(elapsed, 3),
            "tracking_metrics_csv": str(self._metrics_path),
            "failure_cases_csv": str(self._failures_path),
            "video_dir": str(self.cfg.video_dir),
        }
        (self.cfg.results_dir / "tracking_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self.log.info("=== Tracking finished in %.2f s — ok=%d/%d ===",
                      elapsed, n_ok, len(self._session_records))
        return summary


# ---------------------------------------------------------------------------
# CSV helper (small wrapper that owns its file handle)
# ---------------------------------------------------------------------------
class _CsvWriter:
    """Minimal CSV DictWriter with explicit flush/close semantics."""

    def __init__(self, path: Path, columns: Iterable[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = tuple(columns)
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(self.columns))
        self._writer.writeheader()
        self._fh.flush()

    def writerow(self, row: dict[str, Any]) -> None:
        clean = {c: (json.dumps(row[c], default=str)
                     if isinstance(row.get(c), (dict, list, tuple))
                     else row.get(c))
                 for c in self.columns}
        self._writer.writerow(clean)

    def flush_handle(self) -> None:
        if not self._fh.closed:
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _open_csv(path: Path, columns: Iterable[str]) -> _CsvWriter:
    return _CsvWriter(path, columns)


def _safe(v: Any) -> float:
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Integrate the best deployed anomaly detector into "
                     "the closed-loop visual target tracking task."),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--deploy-results-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--engine-path", type=Path, default=None,
                   help="Override engine selection (skip the rank lookup).")
    p.add_argument("--candidate-id", type=str, default=None)
    p.add_argument("--precision", type=str, default=None)
    p.add_argument("--rank-index", type=int, default=None)
    p.add_argument("--source", type=str, default=None,
                   help="Camera index or video file path.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--scenarios", type=str, default=None,
                   help="Comma-separated scenario list.")
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--no-record-video", action="store_true")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   choices=[None, "cpu", "cuda"])
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def _apply_cli_overrides(cfg: TrackingConfig,
                         args: argparse.Namespace) -> TrackingConfig:
    overrides: dict[str, Any] = {
        "deploy_results_dir": args.deploy_results_dir,
        "results_dir":        args.results_dir,
        "video_dir":          args.video_dir,
        "engine_path":        args.engine_path,
        "candidate_id":       args.candidate_id,
        "precision":          args.precision,
        "rank_index":         args.rank_index,
        "max_frames":         args.max_frames,
        "ground_truth_path":  args.ground_truth,
        "seed":               args.seed,
        "device":             args.device,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.source is not None:
        # Allow integer device ids ("0") or filesystem paths.
        try:
            cfg.source = int(args.source)
        except ValueError:
            cfg.source = args.source
    if args.scenarios:
        cfg.scenarios = tuple(s.strip() for s in args.scenarios.split(",")
                              if s.strip())
    if args.no_record_video:
        cfg.record_video = False
    if args.no_annotate:
        cfg.annotate = False
    if args.fail_fast:
        cfg.fail_fast = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = (TrackingConfig.from_file(args.config)
           if args.config.is_file() else TrackingConfig())
    cfg = _apply_cli_overrides(cfg, args)

    log_path = cfg.results_dir / "tracking.log"
    logger = _configure_logging(
        log_path=log_path,
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    if cfg.record_video and not _HAVE_CV2:
        logger.warning(
            "OpenCV not available — video logging will be disabled."
        )

    np.random.seed(cfg.seed)

    try:
        TrackingPipeline(cfg, logger=logger).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Tracking failure: %s", exc)
        return 3
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during tracking")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
