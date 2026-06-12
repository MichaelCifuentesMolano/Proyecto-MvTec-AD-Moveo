"""
src/tracking/tracker_core.py
============================
Multi-object anomaly-aware tracker for the MVTec-AD embedded pipeline.

Supported algorithms
--------------------
- CSRT        : OpenCV discriminative correlation filter (robust, ~40 FPS on Jetson).
- KCF         : OpenCV kernelised correlation filter (fast, ~120 FPS).
- SORT        : Simple Online and Realtime Tracking — Kalman filter + Hungarian
                assignment (best for many objects, no OpenCV dependency).
- LIGHTWEIGHT : IoU + exponential moving-average velocity — fastest, zero optional deps.

Integration
-----------
All algorithms share a unified ``MultiObjectTracker.update()`` interface that
accepts a ``DetectionResult`` from ``detector_interface.py`` and the raw BGR
frame.  ROI alerts embedded in the result seed new tracks; existing tracks are
updated and culled according to their hit/miss counters each frame.

Track lifecycle
---------------
TENTATIVE  →  CONFIRMED  →  LOST  →  DELETED
  (hits ≥ min_hits)          (misses > max_misses_before_lost)
                                      (misses > max_misses_before_delete)

Re-acquisition: a LOST track reverts directly to CONFIRMED when matched again.

Public API
----------
>>> cfg = TrackerConfig(algorithm=TrackerAlgorithm.SORT, min_hits_to_confirm=2)
>>> tracker = MultiObjectTracker(cfg)
>>> for frame, det in stream:
...     active = tracker.update(det, frame)
...     for obj in active:
...         print(obj.track_id, obj.bbox, obj.anomaly_score)

Or use the convenience factory:

>>> tracker = make_tracker("sort", min_hits=2, iou_threshold=0.25,
...                        frame_width=1280, frame_height=720)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies (graceful degradation)
# ---------------------------------------------------------------------------

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_lsa = None  # type: ignore
    _SCIPY_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MAX_TRACK_ID: int = 2**31 - 1   # Wrap-around guard for track IDs.
_EPS: float = 1e-6                # Numerical stability floor.

# SORT Kalman state:  [cx, cy, s, r,  v_cx, v_cy, v_s]
#   cx, cy  = bounding-box centre
#   s       = area (width × height)
#   r       = aspect ratio (width / height)
#   v_*     = first-order velocity terms
_SORT_STATE_DIM: int = 7
_SORT_MEAS_DIM: int = 4  # observed: [cx, cy, s, r]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackState(Enum):
    """Lifecycle state of a single tracked object."""
    TENTATIVE = auto()   # Awaiting consecutive hits before promotion.
    CONFIRMED = auto()   # Active, healthy track.
    LOST      = auto()   # Missed recently; kept but not output.
    DELETED   = auto()   # Scheduled for removal from the pool.


class TrackerAlgorithm(Enum):
    """Available tracking back-ends."""
    CSRT        = "csrt"
    KCF         = "kcf"
    SORT        = "sort"
    LIGHTWEIGHT = "lightweight"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    All coordinates are stored as floats (x1, y1, x2, y2).
    Convenience constructors and the ``clip`` / ``scale`` methods return
    new instances, keeping ``BBox`` effectively immutable in practice.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5

    @property
    def aspect_ratio(self) -> float:
        """width / height; guarded against zero height."""
        return self.width / (self.height + _EPS)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, w, h) — OpenCV ROI convention."""
        return self.x1, self.y1, self.width, self.height

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    # ------------------------------------------------------------------
    # Geometric operations
    # ------------------------------------------------------------------

    def iou(self, other: "BBox") -> float:
        """Intersection-over-Union with another ``BBox``."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / (union + _EPS)

    def clip(self, frame_w: int, frame_h: int) -> "BBox":
        """Return a new ``BBox`` clamped to frame boundaries."""
        return BBox(
            x1=max(0.0, self.x1),
            y1=max(0.0, self.y1),
            x2=min(float(frame_w), self.x2),
            y2=min(float(frame_h), self.y2),
        )

    def scale(self, sx: float, sy: float) -> "BBox":
        """Return a spatially scaled ``BBox`` (e.g. after resolution change)."""
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def expand(self, pad_x: float, pad_y: float) -> "BBox":
        """Expand symmetrically by ``pad_x`` / ``pad_y`` pixels."""
        return BBox(self.x1 - pad_x, self.y1 - pad_y,
                    self.x2 + pad_x, self.y2 + pad_y)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_roi_alert(cls, roi) -> "BBox":
        """
        Construct from a ``ROIAlert`` or any object with a ``.bbox``
        attribute (tuple/list x1, y1, x2, y2) or directly iterable.
        """
        if hasattr(roi, "bbox"):
            return cls(*roi.bbox)
        return cls(*roi)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }

    def __repr__(self) -> str:
        return (
            f"BBox(x1={self.x1:.1f}, y1={self.y1:.1f}, "
            f"x2={self.x2:.1f}, y2={self.y2:.1f})"
        )


@dataclass
class TrackerConfig:
    """
    Full configuration for ``MultiObjectTracker``.

    All fields have sensible defaults; override only what you need.
    """

    # --- Algorithm selection ---
    algorithm: TrackerAlgorithm = TrackerAlgorithm.SORT

    # --- Track lifecycle thresholds ---
    min_hits_to_confirm: int = 3
    """Consecutive matched frames required to promote TENTATIVE → CONFIRMED."""

    max_misses_before_lost: int = 5
    """Consecutive missed frames before CONFIRMED → LOST."""

    max_misses_before_delete: int = 10
    """Consecutive missed frames before LOST → DELETED (track removed)."""

    # --- Detection-to-track association ---
    iou_threshold: float = 0.30
    """Minimum IoU for a detection to be matched to an existing track."""

    anomaly_score_weight: float = 0.20
    """
    Additive affinity bonus: ``effective_iou += score_weight * detection_score``.
    Set to 0 to use pure IoU matching.
    """

    # --- Kalman filter tuning (SORT only) ---
    process_noise_scale: float = 1.0
    """Scale factor applied to the diagonal Q matrix."""

    measurement_noise_scale: float = 1.0
    """Scale factor applied to the diagonal R matrix."""

    # --- Per-track history ---
    max_history_len: int = 30
    """Maximum number of past bounding boxes retained per track."""

    # --- Detection filtering ---
    min_bbox_area: float = 100.0
    """Detections with pixel² area below this are discarded before matching."""

    # --- Frame geometry (for bbox clipping) ---
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class TrackedObject:
    """
    A single tracked anomaly target.

    Attributes
    ----------
    track_id      : Unique integer identifier (monotonically increasing).
    bbox          : Current bounding box (updated after each frame).
    state         : Lifecycle state (TENTATIVE / CONFIRMED / LOST / DELETED).
    anomaly_score : Latest anomaly score from the detector (0–1).
    severity      : Latest severity label from the ROI alert.
    hits          : Total frames where this track was matched.
    misses        : Consecutive frames without a match.
    age           : Total frames since track creation.
    confidence    : Latest ROI confidence from the detector.
    created_at    : Unix timestamp of track creation.
    bbox_history  : Ring buffer of recent bounding boxes.
    """

    track_id: int
    bbox: BBox
    state: TrackState
    anomaly_score: float = 0.0
    severity: str = "normal"
    hits: int = 0
    misses: int = 0
    age: int = 0
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    bbox_history: Deque[BBox] = field(default_factory=lambda: deque(maxlen=30))

    # Internal motion-model reference — opaque to callers.
    _kalman: Optional[object] = field(default=None, repr=False, compare=False)
    _cv2_tracker: Optional[object] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True for TENTATIVE and CONFIRMED tracks."""
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def velocity(self) -> Optional[Tuple[float, float]]:
        """
        Estimate (vx, vy) pixels-per-frame from the last two recorded centres.
        Returns ``None`` if fewer than two history entries exist.
        """
        if len(self.bbox_history) < 2:
            return None
        prev_cx, prev_cy = self.bbox_history[-2].center
        curr_cx, curr_cy = self.bbox_history[-1].center
        return curr_cx - prev_cx, curr_cy - prev_cy

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_dict(),
            "state": self.state.name,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "confidence": round(self.confidence, 4),
            "created_at": round(self.created_at, 3),
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObject(id={self.track_id}, state={self.state.name}, "
            f"score={self.anomaly_score:.3f}, bbox={self.bbox}, "
            f"hits={self.hits}, misses={self.misses})"
        )


# ---------------------------------------------------------------------------
# Vectorised IoU and linear assignment
# ---------------------------------------------------------------------------

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute the full (M × N) pairwise IoU matrix.

    Parameters
    ----------
    bboxes_a : (M, 4) float32  [x1, y1, x2, y2]
    bboxes_b : (N, 4) float32  [x1, y1, x2, y2]

    Returns
    -------
    iou : (M, N) float64
    """
    a = bboxes_a[:, np.newaxis, :]  # (M, 1, 4)
    b = bboxes_b[np.newaxis, :, :]  # (1, N, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / (union + _EPS)


def _linear_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear sum assignment problem (minimise total cost).

    Uses ``scipy.optimize.linear_sum_assignment`` when available; falls back
    to a greedy O(N²) approximation otherwise (acceptable for ≤ 30 tracks).

    Returns
    -------
    row_ind, col_ind : matched pairs as 1-D integer arrays.
    """
    if _SCIPY_AVAILABLE:
        return _scipy_lsa(cost_matrix)

    # Greedy fallback: repeatedly select the global minimum.
    cost = cost_matrix.copy().astype(float)
    rows, cols = [], []
    inf = np.inf
    for _ in range(min(cost.shape)):
        idx = np.argmin(cost)
        r, c = np.unravel_index(idx, cost.shape)
        if cost[r, c] >= inf:
            break
        rows.append(r)
        cols.append(c)
        cost[r, :] = inf
        cost[:, c] = inf
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def _match_detections_to_tracks(
    detections: List[BBox],
    tracks: List[TrackedObject],
    iou_threshold: float,
    score_weight: float,
    det_scores: Optional[List[float]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate each detection to at most one existing track.

    Affinity metric::

        affinity(d, t) = IoU(d, t) + score_weight * anomaly_score(d)

    The assignment maximises total affinity (equivalently minimises 1 − affinity).
    Any pair whose raw IoU is below ``iou_threshold`` is treated as unmatched
    even if selected by the assignment solver.

    Parameters
    ----------
    detections    : List of detection bounding boxes (current frame).
    tracks        : Active track list (after prediction step).
    iou_threshold : Minimum IoU required for a valid match.
    score_weight  : Anomaly-score bonus coefficient.
    det_scores    : Per-detection anomaly scores (same length as detections).

    Returns
    -------
    matches       : List of (det_idx, trk_idx) pairs.
    unmatched_det : Detection indices with no matched track.
    unmatched_trk : Track indices with no matched detection.
    """
    if not tracks or not detections:
        return [], list(range(len(detections))), list(range(len(tracks)))

    det_arr = np.array([d.as_xyxy() for d in detections], dtype=np.float32)
    trk_arr = np.array([t.bbox.as_xyxy() for t in tracks], dtype=np.float32)

    iou = _iou_batch(det_arr, trk_arr)  # (D, T)

    # Anomaly-score affinity bonus (column broadcast over tracks).
    if det_scores and score_weight > 0.0:
        bonus = np.array(det_scores, dtype=float)[:, np.newaxis] * score_weight
        iou = iou + bonus

    cost = 1.0 - iou
    row_ind, col_ind = _linear_assignment(cost)

    matches: List[Tuple[int, int]] = []
    matched_det: set = set()
    matched_trk: set = set()

    for r, c in zip(row_ind, col_ind):
        # Reject pairs whose raw IoU (without the bonus) is too low.
        raw_iou = _iou_batch(det_arr[r : r + 1], trk_arr[c : c + 1])[0, 0]
        if raw_iou < iou_threshold:
            continue
        matches.append((int(r), int(c)))
        matched_det.add(int(r))
        matched_trk.add(int(c))

    unmatched_det = [i for i in range(len(detections)) if i not in matched_det]
    unmatched_trk = [j for j in range(len(tracks)) if j not in matched_trk]
    return matches, unmatched_det, unmatched_trk


# ---------------------------------------------------------------------------
# Kalman filter for SORT
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for a single bounding box.

    State vector  x = [cx, cy, s, r, v_cx, v_cy, v_s]^T
    Measurement   z = [cx, cy, s, r]^T

    where s = area (px²), r = width/height aspect ratio.

    This mirrors the original SORT formulation (Bewley et al., 2016) with
    configurable Q / R scale factors for embedded deployment tuning.
    """

    def __init__(
        self,
        bbox: BBox,
        *,
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
    ) -> None:
        dim_x, dim_z = _SORT_STATE_DIM, _SORT_MEAS_DIM

        # State transition: F encodes constant-velocity model.
        # x_{k+1} = F x_k  =>  position += velocity * dt (dt = 1 frame)
        self.F = np.eye(dim_x, dtype=np.float64)
        for i in range(dim_z):
            self.F[i, dim_z + i] = 1.0   # position += velocity

        # Measurement matrix: observe only [cx, cy, s, r].
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.H[:dim_z, :dim_z] = np.eye(dim_z)

        # Process noise Q.
        q = np.array([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4]) * process_noise_scale
        self.Q = np.diag(q)

        # Measurement noise R.
        r = np.array([1.0, 1.0, 10.0, 10.0]) * measurement_noise_scale
        self.R = np.diag(r)

        # Initial state covariance (high uncertainty on velocity).
        p = np.array([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.P = np.diag(p)

        # State vector.
        self.x = np.zeros((dim_x, 1), dtype=np.float64)
        self._init_state(bbox)

    # ------------------------------------------------------------------

    def _init_state(self, bbox: BBox) -> None:
        cx, cy = bbox.center
        self.x[:_SORT_MEAS_DIM, 0] = [cx, cy, bbox.area, bbox.aspect_ratio]
        self.x[_SORT_MEAS_DIM:, 0] = 0.0

    @staticmethod
    def _state_to_bbox(x: np.ndarray) -> BBox:
        cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        s = max(s, _EPS)
        r = max(r, _EPS)
        w = math.sqrt(s * r)
        h = s / (w + _EPS)
        return BBox(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

    # ------------------------------------------------------------------

    def predict(self) -> BBox:
        """Advance the filter by one time step and return the predicted BBox."""
        # Prevent area from going negative through velocity drift.
        if self.x[2, 0] + self.x[6, 0] <= 0.0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_bbox(self.x.ravel())

    def update(self, bbox: BBox) -> None:
        """Correct the filter state with a new observation."""
        cx, cy = bbox.center
        z = np.array([[cx], [cy], [bbox.area], [bbox.aspect_ratio]], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R          # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    def get_bbox(self) -> BBox:
        return self._state_to_bbox(self.x.ravel())


# ---------------------------------------------------------------------------
# Lightweight constant-velocity motion model
# ---------------------------------------------------------------------------

class _LightweightMotionModel:
    """
    Predict next bbox position using exponential moving-average velocity.

    No matrix operations required — suitable for extremely constrained runtimes
    or when tracking many objects with the LIGHTWEIGHT algorithm.
    """

    _ALPHA: float = 0.5  # EMA smoothing: 0 = frozen, 1 = instantaneous.

    def __init__(self, bbox: BBox) -> None:
        self._bbox = bbox
        self._vx: float = 0.0
        self._vy: float = 0.0

    def predict(self) -> BBox:
        cx, cy = self._bbox.center
        w, h = self._bbox.width, self._bbox.height
        cx_p = cx + self._vx
        cy_p = cy + self._vy
        return BBox(cx_p - w * 0.5, cy_p - h * 0.5,
                    cx_p + w * 0.5, cy_p + h * 0.5)

    def update(self, bbox: BBox) -> None:
        prev_cx, prev_cy = self._bbox.center
        new_cx, new_cy = bbox.center
        self._vx = self._ALPHA * (new_cx - prev_cx) + (1.0 - self._ALPHA) * self._vx
        self._vy = self._ALPHA * (new_cy - prev_cy) + (1.0 - self._ALPHA) * self._vy
        self._bbox = bbox

    def get_bbox(self) -> BBox:
        return self._bbox


# ---------------------------------------------------------------------------
# OpenCV single-object tracker factory
# ---------------------------------------------------------------------------

def _create_cv2_tracker(algorithm: TrackerAlgorithm) -> Optional[object]:
    """
    Instantiate an OpenCV tracker.

    Returns ``None`` gracefully when OpenCV is unavailable or the requested
    tracker variant is absent in the current build.
    """
    if not _CV2_AVAILABLE:
        return None
    try:
        if algorithm == TrackerAlgorithm.CSRT:
            return cv2.TrackerCSRT_create()
        if algorithm == TrackerAlgorithm.KCF:
            return cv2.TrackerKCF_create()
    except AttributeError:
        log.warning(
            "cv2.Tracker%s_create not found in this OpenCV build.",
            algorithm.value.upper(),
        )
    return None


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """
    Online multi-object anomaly-aware tracker.

    Each call to ``update()`` performs:
    1. Extract detection bboxes from ``DetectionResult.rois``.
    2. Predict current track positions (Kalman / EMA / OpenCV).
    3. Associate detections ↔ tracks (IoU + optional score bonus).
    4. Update matched tracks; accumulate misses on unmatched tracks.
    5. Initialise new tracks for unmatched detections.
    6. Prune DELETED tracks.
    7. Return all TENTATIVE + CONFIRMED tracks.

    Parameters
    ----------
    config : ``TrackerConfig`` instance (uses defaults when omitted).
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config: TrackerConfig = config or TrackerConfig()
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

        # Resolve effective algorithm (may downgrade if OpenCV absent).
        self._algorithm: TrackerAlgorithm = self.config.algorithm
        if self._algorithm in (TrackerAlgorithm.CSRT, TrackerAlgorithm.KCF):
            if not _CV2_AVAILABLE:
                log.warning(
                    "OpenCV not available — downgrading %s → SORT.",
                    self._algorithm.name,
                )
                self._algorithm = TrackerAlgorithm.SORT

        log.info(
            "MultiObjectTracker initialised [algorithm=%s, iou_thr=%.2f, "
            "min_hits=%d, max_misses=%d].",
            self._algorithm.name,
            self.config.iou_threshold,
            self.config.min_hits_to_confirm,
            self.config.max_misses_before_lost,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detection_result,
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        """
        Process one frame of anomaly-detector output.

        Parameters
        ----------
        detection_result
            A ``DetectionResult`` (or duck-typed equivalent) exposing:
            - ``.rois``  : list of ``ROIAlert`` objects (may be empty).
            - ``.score`` : global frame anomaly score (float 0–1).
        frame
            Raw BGR uint8 NumPy array.  Required for CSRT / KCF; ignored
            by SORT and LIGHTWEIGHT.

        Returns
        -------
        List[TrackedObject]
            All TENTATIVE and CONFIRMED tracks after this frame.
        """
        self._frame_count += 1
        cfg = self.config

        # ── 1. Extract detections ──────────────────────────────────────
        detections: List[BBox] = []
        det_scores: List[float] = []
        det_severities: List[str] = []
        det_confidences: List[float] = []

        rois = getattr(detection_result, "rois", None) or []
        global_score = float(getattr(detection_result, "score", 0.0))

        for roi in rois:
            try:
                bbox = BBox.from_roi_alert(roi)
            except (TypeError, ValueError):
                log.debug("Skipping malformed ROI: %r", roi)
                continue
            if bbox.area < cfg.min_bbox_area:
                continue
            bbox = bbox.clip(cfg.frame_width, cfg.frame_height)
            detections.append(bbox)
            det_scores.append(float(getattr(roi, "peak_score", global_score)))
            det_severities.append(str(getattr(roi, "severity", "unknown")))
            det_confidences.append(float(getattr(roi, "confidence", 0.0)))

        # ── 2. Predict current track positions ────────────────────────
        active_tracks = [
            t for t in self._tracks.values()
            if t.state != TrackState.DELETED
        ]
        self._predict_all(active_tracks, frame)

        # ── 3. Match detections to tracks ─────────────────────────────
        matched, unmatched_det, unmatched_trk = _match_detections_to_tracks(
            detections,
            active_tracks,
            iou_threshold=cfg.iou_threshold,
            score_weight=cfg.anomaly_score_weight,
            det_scores=det_scores,
        )

        # ── 4. Update matched tracks ───────────────────────────────────
        for det_idx, trk_idx in matched:
            self._update_track(
                active_tracks[trk_idx],
                detections[det_idx],
                frame,
                det_scores[det_idx],
                det_severities[det_idx],
                det_confidences[det_idx],
            )

        # ── 5. Accumulate misses on unmatched tracks ───────────────────
        for trk_idx in unmatched_trk:
            trk = active_tracks[trk_idx]
            trk.misses += 1
            trk.age += 1
            if trk.misses > cfg.max_misses_before_delete:
                trk.state = TrackState.DELETED
                log.debug("Track %d deleted (misses=%d).", trk.track_id, trk.misses)
            elif trk.misses > cfg.max_misses_before_lost:
                if trk.state != TrackState.LOST:
                    trk.state = TrackState.LOST
                    log.debug("Track %d lost (misses=%d).", trk.track_id, trk.misses)

        # ── 6. Initialise tracks for unmatched detections ─────────────
        for det_idx in unmatched_det:
            self._create_track(
                detections[det_idx],
                frame,
                det_scores[det_idx],
                det_severities[det_idx],
                det_confidences[det_idx],
            )

        # ── 7. Prune deleted tracks ────────────────────────────────────
        self._tracks = {
            tid: t
            for tid, t in self._tracks.items()
            if t.state != TrackState.DELETED
        }

        # ── 8. Return active tracks ────────────────────────────────────
        return [
            t for t in self._tracks.values()
            if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]

    # ------------------------------------------------------------------
    # Read-only properties / inspection helpers
    # ------------------------------------------------------------------

    @property
    def active_tracks(self) -> List[TrackedObject]:
        """Snapshot of all non-deleted tracks (read-only; no update triggered)."""
        return [t for t in self._tracks.values() if t.state != TrackState.DELETED]

    @property
    def frame_count(self) -> int:
        """Total frames processed since creation or last ``reset()``."""
        return self._frame_count

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Return the ``TrackedObject`` with the given ID, or ``None``."""
        return self._tracks.get(track_id)

    def summary(self) -> dict:
        """
        Lightweight diagnostic snapshot suitable for JSON logging.

        Returns
        -------
        dict with keys: frame, algorithm, total_tracks,
                        n_tentative, n_confirmed, n_lost.
        """
        counts: Dict[str, int] = {s.name: 0 for s in TrackState}
        for t in self._tracks.values():
            counts[t.state.name] += 1
        return {
            "frame": self._frame_count,
            "algorithm": self._algorithm.name,
            "total_tracks": len(self._tracks),
            "n_tentative": counts["TENTATIVE"],
            "n_confirmed": counts["CONFIRMED"],
            "n_lost": counts["LOST"],
        }

    def reset(self) -> None:
        """Remove all tracks and reset the frame counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        log.info("MultiObjectTracker reset.")

    # ------------------------------------------------------------------
    # Internal — track lifecycle
    # ------------------------------------------------------------------

    def _next_track_id(self) -> int:
        tid = self._next_id
        self._next_id = (self._next_id % _MAX_TRACK_ID) + 1
        return tid

    def _create_track(
        self,
        bbox: BBox,
        frame: Optional[np.ndarray],
        score: float,
        severity: str,
        confidence: float,
    ) -> TrackedObject:
        """Allocate a new TENTATIVE track and attach a motion model."""
        cfg = self.config
        tid = self._next_track_id()
        history: Deque[BBox] = deque(maxlen=cfg.max_history_len)
        history.append(bbox)

        trk = TrackedObject(
            track_id=tid,
            bbox=bbox,
            state=TrackState.TENTATIVE,
            anomaly_score=score,
            severity=severity,
            confidence=confidence,
            hits=1,
            misses=0,
            age=1,
            bbox_history=history,
        )

        # Attach the appropriate motion model.
        if self._algorithm == TrackerAlgorithm.SORT:
            trk._kalman = KalmanBoxTracker(
                bbox,
                process_noise_scale=cfg.process_noise_scale,
                measurement_noise_scale=cfg.measurement_noise_scale,
            )

        elif self._algorithm == TrackerAlgorithm.LIGHTWEIGHT:
            trk._kalman = _LightweightMotionModel(bbox)

        else:  # CSRT or KCF
            cv_trk = _create_cv2_tracker(self._algorithm)
            if cv_trk is not None and frame is not None:
                x, y, w, h = bbox.as_xywh()
                cv_trk.init(frame, (int(x), int(y), int(w), int(h)))
                trk._cv2_tracker = cv_trk
            elif cv_trk is None:
                # OpenCV tracker unavailable at runtime; use lightweight fallback.
                trk._kalman = _LightweightMotionModel(bbox)

        self._tracks[tid] = trk
        log.debug(
            "Track %d created at %s (score=%.3f, severity=%s).",
            tid, bbox, score, severity,
        )
        return trk

    def _predict_all(
        self,
        active_tracks: List[TrackedObject],
        frame: Optional[np.ndarray],
    ) -> None:
        """
        Advance all active tracks one frame using their motion model.

        For OpenCV trackers this also performs the visual search step;
        the predicted bbox is updated directly on the track object so
        that ``_match_detections_to_tracks`` works with fresh positions.
        """
        cfg = self.config
        for trk in active_tracks:
            if self._algorithm in (TrackerAlgorithm.SORT, TrackerAlgorithm.LIGHTWEIGHT):
                if trk._kalman is not None:
                    predicted = trk._kalman.predict()
                    trk.bbox = predicted.clip(cfg.frame_width, cfg.frame_height)

            elif self._algorithm in (TrackerAlgorithm.CSRT, TrackerAlgorithm.KCF):
                cv_trk = trk._cv2_tracker
                if cv_trk is not None and frame is not None:
                    ok, rect = cv_trk.update(frame)
                    if ok:
                        x, y, w, h = rect
                        trk.bbox = BBox.from_xywh(x, y, w, h).clip(
                            cfg.frame_width, cfg.frame_height
                        )
                    # On failure, keep last known position; misses will mount.
                elif trk._kalman is not None:
                    # Lightweight fallback when cv2 tracker missing.
                    predicted = trk._kalman.predict()
                    trk.bbox = predicted.clip(cfg.frame_width, cfg.frame_height)

    def _update_track(
        self,
        trk: TrackedObject,
        bbox: BBox,
        frame: Optional[np.ndarray],
        score: float,
        severity: str,
        confidence: float,
    ) -> None:
        """
        Correct a matched track with a new detector observation.

        Handles TENTATIVE → CONFIRMED promotion and LOST → CONFIRMED
        re-acquisition automatically.
        """
        cfg = self.config
        trk.bbox = bbox
        trk.anomaly_score = score
        trk.severity = severity
        trk.confidence = confidence
        trk.hits += 1
        trk.misses = 0
        trk.age += 1
        trk.bbox_history.append(bbox)

        # State machine transitions.
        if trk.state == TrackState.TENTATIVE:
            if trk.hits >= cfg.min_hits_to_confirm:
                trk.state = TrackState.CONFIRMED
                log.debug("Track %d confirmed (hits=%d).", trk.track_id, trk.hits)
        elif trk.state == TrackState.LOST:
            trk.state = TrackState.CONFIRMED
            log.debug("Track %d re-acquired.", trk.track_id)

        # Correct motion model.
        if self._algorithm in (TrackerAlgorithm.SORT, TrackerAlgorithm.LIGHTWEIGHT):
            if trk._kalman is not None:
                trk._kalman.update(bbox)

        elif self._algorithm in (TrackerAlgorithm.CSRT, TrackerAlgorithm.KCF):
            # Re-initialise the OpenCV tracker at the detector-corrected bbox to
            # prevent drift accumulation between detection frames.
            if frame is not None:
                cv_trk = _create_cv2_tracker(self._algorithm)
                if cv_trk is not None:
                    x, y, w, h = bbox.as_xywh()
                    cv_trk.init(frame, (int(x), int(y), int(w), int(h)))
                    trk._cv2_tracker = cv_trk


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_tracker(
    algorithm: str = "sort",
    *,
    min_hits: int = 3,
    max_misses: int = 5,
    iou_threshold: float = 0.30,
    frame_width: int = 640,
    frame_height: int = 480,
    **kwargs,
) -> MultiObjectTracker:
    """
    Build a ``MultiObjectTracker`` from plain string / numeric arguments.

    Parameters
    ----------
    algorithm     : "sort" | "lightweight" | "csrt" | "kcf" (case-insensitive).
    min_hits      : Hits before TENTATIVE → CONFIRMED.
    max_misses    : Consecutive misses before CONFIRMED → LOST.
    iou_threshold : Minimum IoU for a valid detection-to-track match.
    frame_width   : Frame width in pixels (for bbox clipping).
    frame_height  : Frame height in pixels (for bbox clipping).
    **kwargs      : Additional ``TrackerConfig`` fields (e.g.
                    ``anomaly_score_weight``, ``process_noise_scale``).

    Returns
    -------
    MultiObjectTracker

    Raises
    ------
    ValueError : If ``algorithm`` is not a recognised value.

    Examples
    --------
    >>> tracker = make_tracker("sort", min_hits=2, iou_threshold=0.25)
    >>> tracker = make_tracker("lightweight", frame_width=1280, frame_height=720)
    """
    try:
        alg = TrackerAlgorithm(algorithm.lower())
    except ValueError:
        valid = [a.value for a in TrackerAlgorithm]
        raise ValueError(
            f"Unknown tracker algorithm '{algorithm}'. Valid options: {valid}"
        ) from None

    cfg = TrackerConfig(
        algorithm=alg,
        min_hits_to_confirm=min_hits,
        max_misses_before_lost=max_misses,
        iou_threshold=iou_threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        **kwargs,
    )
    return MultiObjectTracker(cfg)
