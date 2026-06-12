"""
src/tracking/robustness_test.py
================================
Stress-test harness for the anomaly-aware tracking pipeline.

Five degradation categories are tested:

    1. Motion blur      — increasing linear-blur kernel sizes (5 → 31 px).
    2. Occlusion        — partial/full blocking for a sustained window,
                          then measuring re-acquisition latency.
    3. Illumination     — gamma ramp, brightness ramp, and random flicker.
    4. Multiple targets — 2, 3, and 5 simultaneous anomalous objects.
    5. Dropped frames   — regular periodic drops and a burst blackout.

Design
------
All tests use a ``_MockDetector`` that injects ground-truth ROIs with
configurable position jitter and miss rate.  No GPU or trained model is
required.  The adversarial condition is applied to (a) the visual frame
(blur / occlusion / gamma), (b) the detector's miss rate, and (c) which
frames are withheld from the tracker entirely (drops / occlusion blind).

The three components under test are therefore:
    • ``MultiObjectTracker``     — does it maintain track IDs?
    • Detection-to-track pipeline — does it re-confirm after degradation?
    • Reacquisition latency       — how many frames until track recovers?

Usage
-----
>>> suite = RobustnessTestSuite()          # defaults; no model needed
>>> report = suite.run_all()
>>> print(report.summary())
>>> suite.save_report(report, Path("results/robustness"))
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

from .tracker_core import (
    BBox,
    MultiObjectTracker,
    TrackerAlgorithm,
    TrackerConfig,
    TrackState,
)

log = logging.getLogger(__name__)
_EPS: float = 1e-9
_DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Duck-typed mock types (no dependency on DetectorInterface)
# ---------------------------------------------------------------------------

@dataclass
class _MockROIAlert:
    """Minimal duck-type for ``ROIAlert`` consumed by ``MultiObjectTracker``."""
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    peak_score: float = 0.85
    confidence: float = 0.90
    severity: str = "warning"
    area_px: int = 4800
    center: Tuple[int, int] = (0, 0)


@dataclass
class _MockDetectionResult:
    """Minimal duck-type for ``DetectionResult`` consumed by the tracker."""
    score: float = 0.0
    rois: List[_MockROIAlert] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    """
    Complete configuration for ``RobustnessTestSuite``.

    All fields have sensible defaults.  Override only what you need to tune.
    """

    # ── Scene ──────────────────────────────────────────────────────────────
    frame_width: int = 640
    frame_height: int = 480
    n_frames: int = 150
    seed: int = _DEFAULT_SEED

    # ── Anomaly target size / speed ────────────────────────────────────────
    target_width: int = 80
    target_height: int = 60
    target_speed_px: float = 3.5     # px/frame; individual targets randomised ±50 %.

    # ── Mock detector ─────────────────────────────────────────────────────
    detection_jitter_px: float = 5.0
    """Gaussian std of bbox-edge noise (simulates imperfect localisation)."""

    base_miss_rate: float = 0.05
    """Baseline fraction of frames where the mock detector returns nothing."""

    anomaly_score: float = 0.85

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker_algorithm: str = "sort"
    min_hits_to_confirm: int = 3
    max_misses_before_lost: int = 8
    max_misses_before_delete: int = 15
    iou_threshold: float = 0.25

    # ── Pass/fail thresholds ───────────────────────────────────────────────
    min_detection_rate: float = 0.75
    """detection_rate ≥ this → test component passes."""

    min_track_persistence: float = 0.65
    """track_persistence ≥ this → test component passes."""

    max_id_switches: int = 5
    """id_switches ≤ this → test component passes."""

    max_reacquisition_latency: int = 15
    """Frames from track-loss to re-confirmation; ≤ this → passes."""

    # ── Motion blur ────────────────────────────────────────────────────────
    blur_kernel_sizes: List[int] = field(
        default_factory=lambda: [5, 11, 15, 21, 31]
    )
    blur_angle_deg: float = 0.0
    """Direction of the motion blur kernel (degrees from horizontal)."""

    blur_pass_max_kernel: int = 21
    """Largest kernel at which detection_rate ≥ min_detection_rate → PASS."""

    # Map kernel size → extra miss rate added on top of base_miss_rate.
    blur_extra_miss: Dict[int, float] = field(
        default_factory=lambda: {5: 0.00, 11: 0.05, 15: 0.15, 21: 0.25, 31: 0.50}
    )

    # ── Occlusion ──────────────────────────────────────────────────────────
    occlusion_start_frame: int = 40
    occlusion_duration_frames: int = 30
    occlusion_coverage: float = 0.90
    """Fraction of the target bbox that is covered by the occluder."""

    # ── Illumination ───────────────────────────────────────────────────────
    gamma_sequence: List[float] = field(
        default_factory=lambda: [
            1.0, 1.5, 2.0, 3.0, 4.0, 3.0, 2.0, 1.5,   # darkening ramp
            1.0, 0.7, 0.4, 0.2, 0.4, 0.7, 1.0,          # brightening ramp
        ]
    )
    """Gamma values cycled across ``n_frames``."""

    illumination_miss_scale: float = 0.40
    """Extra miss rate per unit |gamma − 1| (capped at 0.60)."""

    flicker_sigma: int = 60
    """Brightness offset std for random-flicker sub-test (0 = disable)."""

    # ── Multiple targets ───────────────────────────────────────────────────
    n_targets_list: List[int] = field(default_factory=lambda: [2, 3, 5])
    min_confirmed_fraction: float = 0.70
    """Fraction of expected targets that must be confirmed → PASS."""

    # ── Dropped frames ─────────────────────────────────────────────────────
    regular_drop_every: int = 5
    """Drop 1 in N frames uniformly."""

    burst_drop_start: int = 60
    burst_drop_length: int = 12
    """Consecutive dropped frames starting at ``burst_drop_start``."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Outcome of a single stress-test category."""
    name: str
    passed: bool
    metrics: Dict
    details: str
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "details": self.details,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class RobustnessReport:
    """Aggregated results from all stress-test categories."""
    results: List[TestResult]
    n_passed: int
    n_failed: int
    total_duration_s: float
    config_dict: dict

    def summary(self) -> str:
        lines = [
            f"Robustness Report  {self.n_passed}/{len(self.results)} PASSED"
            f"  ({self.total_duration_s:.2f} s total)"
        ]
        for r in self.results:
            tag = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{tag}]  {r.name:<35}  {r.details}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "total_duration_s": round(self.total_duration_s, 3),
            "results": [r.to_dict() for r in self.results],
            "config": self.config_dict,
        }


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------

@dataclass
class AnomalyTarget:
    """A coloured moving patch used as a synthetic anomaly."""
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    color: Tuple[int, int, int] = (0, 120, 255)   # BGR

    def step(self, frame_w: int, frame_h: int) -> None:
        """Advance position and bounce off frame boundaries."""
        self.x += self.vx
        self.y += self.vy
        if self.x < 0:
            self.x = 0.0
            self.vx = abs(self.vx)
        if self.x + self.w > frame_w:
            self.x = frame_w - self.w
            self.vx = -abs(self.vx)
        if self.y < 0:
            self.y = 0.0
            self.vy = abs(self.vy)
        if self.y + self.h > frame_h:
            self.y = frame_h - self.h
            self.vy = -abs(self.vy)

    @property
    def bbox(self) -> BBox:
        return BBox(self.x, self.y, self.x + self.w, self.y + self.h)


def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Gray background with subtle spatial noise."""
    base = np.full((h, w, 3), 120, dtype=np.uint8)
    noise = rng.integers(-18, 18, (h, w, 1), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _render_frame(
    background: np.ndarray,
    targets: List[AnomalyTarget],
    rng: np.random.Generator,
    *,
    texture_sigma: int = 22,
) -> Tuple[np.ndarray, List[BBox]]:
    """
    Composite ``targets`` onto ``background``.

    Returns
    -------
    (frame, gt_bboxes)  — rendered BGR frame and ground-truth bounding boxes.
    """
    frame = background.copy()
    gt_bboxes: List[BBox] = []
    for t in targets:
        x1, y1 = int(t.x), int(t.y)
        x2, y2 = int(t.x + t.w), int(t.y + t.h)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = np.full((y2 - y1, x2 - x1, 3), t.color, dtype=np.int16)
        noise = rng.integers(-texture_sigma, texture_sigma, patch.shape, dtype=np.int16)
        frame[y1:y2, x1:x2] = np.clip(patch + noise, 0, 255).astype(np.uint8)
        gt_bboxes.append(BBox(float(x1), float(y1), float(x2), float(y2)))
    return frame, gt_bboxes


# ---------------------------------------------------------------------------
# Frame augmentations
# ---------------------------------------------------------------------------

class FrameAugmenter:
    """
    Static frame augmentation methods.

    All methods accept and return ``uint8`` BGR NumPy arrays.
    OpenCV is used when available; pure-NumPy fallbacks cover all cases.
    """

    @staticmethod
    def motion_blur(
        frame: np.ndarray,
        kernel_size: int,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Apply directional motion blur.

        OpenCV path: builds a sparse line kernel and rotates it.
        NumPy path: horizontal cumulative-sum sliding-window (angle ignored).
        """
        if kernel_size <= 1:
            return frame

        if _CV2_AVAILABLE:
            k = np.zeros((kernel_size, kernel_size), np.float32)
            k[kernel_size // 2, :] = 1.0
            M = cv2.getRotationMatrix2D(
                (kernel_size / 2.0, kernel_size / 2.0), angle_deg, 1.0
            )
            k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
            s = k.sum()
            k = k / (s + _EPS)
            return cv2.filter2D(frame, -1, k)

        # Pure NumPy: horizontal 1-D average via cumulative sum.
        out = np.empty_like(frame)
        half = kernel_size // 2
        for c in range(3):
            ch = frame[:, :, c].astype(np.float32)
            padded = np.pad(ch, ((0, 0), (half, half)), mode="edge")
            cumsum = np.cumsum(padded, axis=1)
            blurred = (cumsum[:, kernel_size:] - cumsum[:, :-kernel_size]) / kernel_size
            out[:, :, c] = np.clip(blurred, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def occlude(
        frame: np.ndarray,
        bbox: BBox,
        coverage: float = 0.90,
        fill: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Draw a filled rectangle covering ``coverage`` fraction of ``bbox``.

        The occluder is centred on the target bounding box.
        """
        out = frame.copy()
        cx, cy = int(bbox.center[0]), int(bbox.center[1])
        ow = max(1, int(bbox.width * coverage))
        oh = max(1, int(bbox.height * coverage))
        x1 = max(0, cx - ow // 2)
        y1 = max(0, cy - oh // 2)
        x2 = min(frame.shape[1], cx + ow // 2)
        y2 = min(frame.shape[0], cy + oh // 2)
        out[y1:y2, x1:x2] = fill
        return out

    @staticmethod
    def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma correction: ``out = (in / 255) ^ gamma × 255``."""
        if abs(gamma - 1.0) < 1e-3:
            return frame
        table = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0,
            0, 255,
        ).astype(np.uint8)
        if _CV2_AVAILABLE:
            return cv2.LUT(frame, table)
        return table[frame]

    @staticmethod
    def adjust_brightness(frame: np.ndarray, offset: int) -> np.ndarray:
        """Add a constant brightness offset (clamped to [0, 255])."""
        if offset == 0:
            return frame
        return np.clip(frame.astype(np.int16) + offset, 0, 255).astype(np.uint8)

    @staticmethod
    def random_flicker(
        frame: np.ndarray,
        rng: np.random.Generator,
        sigma: int = 60,
    ) -> np.ndarray:
        """Apply a random per-frame brightness offset drawn from N(0, sigma)."""
        if sigma == 0:
            return frame
        offset = int(rng.normal(0, sigma))
        return FrameAugmenter.adjust_brightness(frame, offset)


# ---------------------------------------------------------------------------
# Mock detector
# ---------------------------------------------------------------------------

class _MockDetector:
    """
    Ground-truth ROI injector with configurable position jitter and miss rate.

    Call ``set_scene()`` before each ``detect()`` to update the ground truth
    for the current frame.
    """

    def __init__(self, config: RobustnessConfig, rng: np.random.Generator) -> None:
        self._cfg = config
        self._rng = rng
        self._gt_bboxes: List[BBox] = []
        self._miss_rate: float = config.base_miss_rate
        self._blind: bool = False

    def set_scene(
        self,
        gt_bboxes: List[BBox],
        *,
        miss_rate: Optional[float] = None,
        blind: bool = False,
    ) -> None:
        """
        Update ground truth for the next ``detect()`` call.

        Parameters
        ----------
        gt_bboxes  : Ground-truth bounding boxes for this frame.
        miss_rate  : Override miss rate (uses ``base_miss_rate`` if None).
        blind      : If True, detector returns nothing regardless of miss_rate
                     (simulates full occlusion or sensor blackout).
        """
        self._gt_bboxes = gt_bboxes
        self._miss_rate = miss_rate if miss_rate is not None else self._cfg.base_miss_rate
        self._blind = blind

    def detect(self, frame: np.ndarray) -> _MockDetectionResult:  # noqa: U100
        """Return a mock ``DetectionResult`` based on current ground truth."""
        if self._blind or not self._gt_bboxes:
            return _MockDetectionResult(score=0.0, rois=[])
        if self._rng.random() < self._miss_rate:
            return _MockDetectionResult(score=0.0, rois=[])

        cfg = self._cfg
        rois: List[_MockROIAlert] = []
        for bbox in self._gt_bboxes:
            j = cfg.detection_jitter_px
            x1 = bbox.x1 + float(self._rng.normal(0.0, j))
            y1 = bbox.y1 + float(self._rng.normal(0.0, j))
            x2 = bbox.x2 + float(self._rng.normal(0.0, j))
            y2 = bbox.y2 + float(self._rng.normal(0.0, j))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            rois.append(_MockROIAlert(
                bbox=(x1, y1, x2, y2),
                peak_score=cfg.anomaly_score,
                confidence=0.92,
                severity="warning",
                area_px=int(max(1.0, bbox.area)),
                center=(cx, cy),
            ))
        return _MockDetectionResult(score=cfg.anomaly_score, rois=rois)


# ---------------------------------------------------------------------------
# Per-frame snapshot and metric aggregation
# ---------------------------------------------------------------------------

@dataclass
class _FrameSnapshot:
    """Minimal per-frame tracker state for offline analysis."""
    frame_idx: int
    is_dropped: bool
    n_confirmed: int
    n_tentative: int
    primary_track_id: Optional[int]   # Highest-score confirmed track.


def _primary_id(active: list) -> Optional[int]:
    confirmed = [t for t in active if t.state == TrackState.CONFIRMED]
    if not confirmed:
        return None
    return max(confirmed, key=lambda t: t.anomaly_score).track_id


def _aggregate_metrics(
    snapshots: List[_FrameSnapshot],
    n_expected_targets: int = 1,
) -> dict:
    """
    Compute summary metrics from a list of per-frame snapshots.

    Returns
    -------
    dict with keys:
        n_frames_total, n_frames_active, n_frames_dropped,
        detection_rate, track_persistence, max_confirmed_streak,
        id_switches, avg_confirmed_tracks, reacquisition_latency_frames.
    """
    active_snaps = [s for s in snapshots if not s.is_dropped]
    n_active = len(active_snaps)
    if n_active == 0:
        return {"n_frames_total": len(snapshots), "n_frames_active": 0}

    # Detection rate: frames where ≥ 1 track confirmed.
    detection_rate = sum(1 for s in active_snaps if s.n_confirmed >= 1) / n_active

    # Track persistence: longest streak of confirmed frames.
    max_streak = cur_streak = 0
    for s in active_snaps:
        if s.n_confirmed >= 1:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    persistence = max_streak / n_active

    # ID switches: primary track ID changes (excluding first detection).
    id_switches = 0
    prev_id: Optional[int] = None
    for s in active_snaps:
        pid = s.primary_track_id
        if pid is not None and prev_id is not None and pid != prev_id:
            id_switches += 1
        if pid is not None:
            prev_id = pid

    # Average simultaneously confirmed tracks.
    avg_confirmed = sum(s.n_confirmed for s in active_snaps) / n_active

    # Reacquisition latency after first track loss.
    reacq_latency = _compute_reacquisition_latency(active_snaps)

    return {
        "n_frames_total": len(snapshots),
        "n_frames_active": n_active,
        "n_frames_dropped": len(snapshots) - n_active,
        "detection_rate": round(detection_rate, 4),
        "track_persistence": round(persistence, 4),
        "max_confirmed_streak": max_streak,
        "id_switches": id_switches,
        "avg_confirmed_tracks": round(avg_confirmed, 3),
        "reacquisition_latency_frames": reacq_latency,
    }


def _compute_reacquisition_latency(
    active_snaps: List[_FrameSnapshot],
) -> Optional[int]:
    """
    Return the number of frames from the first track loss to the first
    re-confirmation.  Returns ``None`` if the track was never lost or
    was never re-acquired.
    """
    state = "init"  # init → tracking → lost
    loss_idx: Optional[int] = None
    for s in active_snaps:
        if state == "init":
            if s.n_confirmed >= 1:
                state = "tracking"
        elif state == "tracking":
            if s.n_confirmed == 0:
                state = "lost"
                loss_idx = s.frame_idx
        elif state == "lost":
            if s.n_confirmed >= 1 and loss_idx is not None:
                return s.frame_idx - loss_idx
    return None


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class RobustnessTestSuite:
    """
    Automated stress-test suite for the tracking pipeline.

    Parameters
    ----------
    config : ``RobustnessConfig`` (defaults when omitted).

    Examples
    --------
    >>> suite = RobustnessTestSuite()
    >>> report = suite.run_all()
    >>> print(report.summary())
    >>> suite.save_report(report, Path("results/robustness"))
    """

    def __init__(self, config: Optional[RobustnessConfig] = None) -> None:
        self.config = config or RobustnessConfig()
        # Master RNG; each test derives a sub-generator to stay reproducible.
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> RobustnessReport:
        """Run all five stress tests and return the combined report."""
        t0 = time.monotonic()
        results: List[TestResult] = []
        runners = [
            self.run_motion_blur_test,
            self.run_occlusion_test,
            self.run_illumination_test,
            self.run_multiple_targets_test,
            self.run_dropped_frames_test,
        ]
        for fn in runners:
            try:
                result = fn()
                results.append(result)
                tag = "PASS" if result.passed else "FAIL"
                log.info("[%s] %s", tag, result.name)
            except Exception as exc:  # noqa: BLE001
                log.error("Test '%s' raised an exception: %s", fn.__name__, exc)
                results.append(TestResult(
                    name=fn.__name__,
                    passed=False,
                    metrics={},
                    details=f"Exception: {exc}",
                    duration_s=0.0,
                ))

        n_passed = sum(1 for r in results if r.passed)
        report = RobustnessReport(
            results=results,
            n_passed=n_passed,
            n_failed=len(results) - n_passed,
            total_duration_s=time.monotonic() - t0,
            config_dict=vars(self.config),
        )
        log.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Test 1 — Motion blur
    # ------------------------------------------------------------------

    def run_motion_blur_test(self) -> TestResult:
        """
        Evaluate tracker robustness under increasing linear motion blur.

        Procedure
        ---------
        For each kernel size in ``blur_kernel_sizes``:
          • Run ``n_frames`` with the target moving and blur applied.
          • Increase mock-detector miss rate proportional to kernel size.
        The test PASSES if ``detection_rate ≥ min_detection_rate`` at all
        kernel sizes up to ``blur_pass_max_kernel``.
        """
        cfg = self.config
        t0 = time.monotonic()
        per_kernel: Dict[int, dict] = {}
        aug = FrameAugmenter()

        for ks in cfg.blur_kernel_sizes:
            tracker = self._make_tracker()
            detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + ks))
            targets = self._make_targets(1, seed_offset=ks)
            extra_miss = cfg.blur_extra_miss.get(ks, 0.10)

            snaps = self._run_sequence(
                targets, detector, tracker,
                frame_aug=lambda f, _fi, _gt: aug.motion_blur(
                    f, ks, cfg.blur_angle_deg
                ),
                miss_rate_fn=lambda _fi: cfg.base_miss_rate + extra_miss,
                drop_fn=lambda _fi: False,
                blind_frames=set(),
            )
            per_kernel[ks] = _aggregate_metrics(snaps)
            log.debug(
                "Motion blur ks=%2d | det_rate=%.3f | persistence=%.3f",
                ks,
                per_kernel[ks].get("detection_rate", 0),
                per_kernel[ks].get("track_persistence", 0),
            )

        # Verdict: all kernels ≤ pass threshold must meet detection_rate.
        failing_kernels = [
            ks for ks in cfg.blur_kernel_sizes
            if ks <= cfg.blur_pass_max_kernel
            and per_kernel[ks].get("detection_rate", 0) < cfg.min_detection_rate
        ]
        passed = len(failing_kernels) == 0
        details = (
            f"All kernels ≤ {cfg.blur_pass_max_kernel} px meet "
            f"det_rate ≥ {cfg.min_detection_rate:.2f}"
            if passed else
            f"Kernels {failing_kernels} below det_rate threshold"
        )
        return TestResult(
            name="motion_blur",
            passed=passed,
            metrics={"per_kernel": {str(k): v for k, v in per_kernel.items()}},
            details=details,
            duration_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Test 2 — Occlusion
    # ------------------------------------------------------------------

    def run_occlusion_test(self) -> TestResult:
        """
        Evaluate re-acquisition after sustained target occlusion.

        Phases
        ------
        1. Pre-occlusion  : frames 0 … occlusion_start − 1  (target visible).
        2. Occluded       : frames occlusion_start … +duration (detector blind).
        3. Post-occlusion : remaining frames (target visible again).

        PASS criteria
        -------------
        • reacquisition_latency ≤ max_reacquisition_latency.
        • detection_rate over post-occlusion phase ≥ min_detection_rate.
        """
        cfg = self.config
        t0 = time.monotonic()
        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 200))
        targets = self._make_targets(1, seed_offset=200)
        aug = FrameAugmenter()

        occ_start = cfg.occlusion_start_frame
        occ_end = occ_start + cfg.occlusion_duration_frames
        blind_frames: Set[int] = set(range(occ_start, min(occ_end, cfg.n_frames)))

        def _occ_aug(frame: np.ndarray, fi: int, gt: List[BBox]) -> np.ndarray:
            if fi in blind_frames and gt:
                return aug.occlude(frame, gt[0], cfg.occlusion_coverage)
            return frame

        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=_occ_aug,
            miss_rate_fn=lambda _fi: cfg.base_miss_rate,
            drop_fn=lambda _fi: False,
            blind_frames=blind_frames,
        )

        # Post-occlusion metrics only.
        post_snaps = [s for s in snaps if s.frame_idx >= occ_end]
        m_all = _aggregate_metrics(snaps)
        m_post = _aggregate_metrics(post_snaps)

        reacq = m_all.get("reacquisition_latency_frames")
        post_det = m_post.get("detection_rate", 0.0)

        latency_ok = reacq is not None and reacq <= cfg.max_reacquisition_latency
        det_ok = post_det >= cfg.min_detection_rate
        passed = latency_ok and det_ok

        details = (
            f"reacq={reacq} frames (≤{cfg.max_reacquisition_latency}), "
            f"post_det_rate={post_det:.3f} (≥{cfg.min_detection_rate:.2f})"
        )
        metrics = {
            **m_all,
            "post_occlusion": m_post,
            "occlusion_frames": len(blind_frames),
        }
        return TestResult(
            name="occlusion",
            passed=passed,
            metrics=metrics,
            details=details,
            duration_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Test 3 — Illumination
    # ------------------------------------------------------------------

    def run_illumination_test(self) -> TestResult:
        """
        Evaluate tracker stability under illumination changes.

        Three sub-tests
        ---------------
        gamma_ramp    — gamma cycled through ``gamma_sequence``.
        brightness    — brightness offset ramped ±120.
        flicker       — random per-frame brightness perturbation.

        PASS if all three sub-tests achieve detection_rate ≥ min_detection_rate
        and track_persistence ≥ min_track_persistence.
        """
        cfg = self.config
        t0 = time.monotonic()
        aug = FrameAugmenter()
        sub_results: Dict[str, dict] = {}

        # ── Gamma ramp ────────────────────────────────────────────────
        gamma_seq = cfg.gamma_sequence
        n_gammas = len(gamma_seq)

        def _gamma_aug(frame: np.ndarray, fi: int, _gt: List[BBox]) -> np.ndarray:
            gamma = gamma_seq[fi % n_gammas]
            return aug.apply_gamma(frame, gamma)

        def _gamma_miss(fi: int) -> float:
            gamma = gamma_seq[fi % n_gammas]
            extra = cfg.illumination_miss_scale * abs(gamma - 1.0) / 3.0
            return min(cfg.base_miss_rate + extra, 0.60)

        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 300))
        targets = self._make_targets(1, seed_offset=300)
        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=_gamma_aug,
            miss_rate_fn=_gamma_miss,
            drop_fn=lambda _fi: False,
            blind_frames=set(),
        )
        sub_results["gamma_ramp"] = _aggregate_metrics(snaps)

        # ── Brightness ramp ───────────────────────────────────────────
        def _bright_aug(frame: np.ndarray, fi: int, _gt: List[BBox]) -> np.ndarray:
            # Ramp: 0 → +120 → 0 → -120 → 0 over n_frames.
            phase = 2.0 * np.pi * fi / max(cfg.n_frames - 1, 1)
            offset = int(120.0 * np.sin(phase))
            return aug.adjust_brightness(frame, offset)

        def _bright_miss(fi: int) -> float:
            phase = 2.0 * np.pi * fi / max(cfg.n_frames - 1, 1)
            extra = cfg.illumination_miss_scale * abs(np.sin(phase))
            return min(cfg.base_miss_rate + extra * 0.4, 0.50)

        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 310))
        targets = self._make_targets(1, seed_offset=310)
        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=_bright_aug,
            miss_rate_fn=_bright_miss,
            drop_fn=lambda _fi: False,
            blind_frames=set(),
        )
        sub_results["brightness_ramp"] = _aggregate_metrics(snaps)

        # ── Random flicker ────────────────────────────────────────────
        flicker_rng = np.random.default_rng(cfg.seed + 320)

        def _flicker_aug(frame: np.ndarray, _fi: int, _gt: List[BBox]) -> np.ndarray:
            return aug.random_flicker(frame, flicker_rng, sigma=cfg.flicker_sigma)

        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 320))
        targets = self._make_targets(1, seed_offset=320)
        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=_flicker_aug,
            miss_rate_fn=lambda _fi: cfg.base_miss_rate + 0.10,
            drop_fn=lambda _fi: False,
            blind_frames=set(),
        )
        sub_results["flicker"] = _aggregate_metrics(snaps)

        # ── Verdict ───────────────────────────────────────────────────
        failing = [
            name for name, m in sub_results.items()
            if (
                m.get("detection_rate", 0) < cfg.min_detection_rate
                or m.get("track_persistence", 0) < cfg.min_track_persistence
            )
        ]
        passed = len(failing) == 0
        details = (
            "All illumination sub-tests passed"
            if passed else
            f"Sub-tests failing: {failing}"
        )
        return TestResult(
            name="illumination",
            passed=passed,
            metrics=sub_results,
            details=details,
            duration_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Test 4 — Multiple targets
    # ------------------------------------------------------------------

    def run_multiple_targets_test(self) -> TestResult:
        """
        Evaluate tracking of multiple simultaneous anomalous objects.

        For each N in ``n_targets_list``, N targets are spawned and the suite
        measures what fraction obtain a CONFIRMED track.

        PASS if avg_confirmed_fraction ≥ min_confirmed_fraction for all N.
        """
        cfg = self.config
        t0 = time.monotonic()
        per_n: Dict[int, dict] = {}

        for n_targets in cfg.n_targets_list:
            tracker = self._make_tracker()
            detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 400 + n_targets))
            targets = self._make_targets(n_targets, seed_offset=400 + n_targets)

            snaps = self._run_sequence(
                targets, detector, tracker,
                frame_aug=lambda f, _fi, _gt: f,
                miss_rate_fn=lambda _fi: cfg.base_miss_rate,
                drop_fn=lambda _fi: False,
                blind_frames=set(),
            )
            m = _aggregate_metrics(snaps, n_expected_targets=n_targets)
            avg_conf = m.get("avg_confirmed_tracks", 0.0)
            confirmed_fraction = avg_conf / max(n_targets, 1)
            m["confirmed_fraction"] = round(confirmed_fraction, 4)
            m["n_expected_targets"] = n_targets
            per_n[n_targets] = m
            log.debug(
                "Multiple targets N=%d | confirmed_frac=%.3f | id_sw=%d",
                n_targets,
                confirmed_fraction,
                m.get("id_switches", -1),
            )

        failing = [
            n for n, m in per_n.items()
            if m.get("confirmed_fraction", 0) < cfg.min_confirmed_fraction
        ]
        passed = len(failing) == 0
        details = (
            f"All N ∈ {cfg.n_targets_list} confirmed ≥ {cfg.min_confirmed_fraction:.0%}"
            if passed else
            f"N={failing} below confirmed_fraction threshold"
        )
        return TestResult(
            name="multiple_targets",
            passed=passed,
            metrics={"per_n": {str(k): v for k, v in per_n.items()}},
            details=details,
            duration_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Test 5 — Dropped frames
    # ------------------------------------------------------------------

    def run_dropped_frames_test(self) -> TestResult:
        """
        Evaluate tracker recovery after regular and burst frame drops.

        Two sub-tests
        -------------
        regular_drops : every ``regular_drop_every``-th frame is withheld.
        burst_drops   : ``burst_drop_length`` consecutive frames dropped
                        starting at ``burst_drop_start``.

        PASS if both achieve track_persistence ≥ min_track_persistence
        and reacquisition_latency ≤ max_reacquisition_latency.
        """
        cfg = self.config
        t0 = time.monotonic()
        sub_results: Dict[str, dict] = {}

        # ── Regular drops ─────────────────────────────────────────────
        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 500))
        targets = self._make_targets(1, seed_offset=500)

        def _regular_drop(fi: int) -> bool:
            return (fi % cfg.regular_drop_every) == 0

        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=lambda f, _fi, _gt: f,
            miss_rate_fn=lambda _fi: cfg.base_miss_rate,
            drop_fn=_regular_drop,
            blind_frames=set(),
        )
        sub_results["regular_drops"] = _aggregate_metrics(snaps)

        # ── Burst drops ───────────────────────────────────────────────
        burst_start = cfg.burst_drop_start
        burst_end = burst_start + cfg.burst_drop_length
        burst_set: Set[int] = set(range(burst_start, min(burst_end, cfg.n_frames)))

        tracker = self._make_tracker()
        detector = _MockDetector(cfg, np.random.default_rng(cfg.seed + 510))
        targets = self._make_targets(1, seed_offset=510)

        snaps = self._run_sequence(
            targets, detector, tracker,
            frame_aug=lambda f, _fi, _gt: f,
            miss_rate_fn=lambda _fi: cfg.base_miss_rate,
            drop_fn=lambda fi: fi in burst_set,
            blind_frames=set(),
        )
        sub_results["burst_drops"] = _aggregate_metrics(snaps)

        # ── Verdict ───────────────────────────────────────────────────
        failing = []
        for name, m in sub_results.items():
            pers_ok = m.get("track_persistence", 0) >= cfg.min_track_persistence
            lat = m.get("reacquisition_latency_frames")
            lat_ok = lat is None or lat <= cfg.max_reacquisition_latency
            if not (pers_ok and lat_ok):
                failing.append(name)

        passed = len(failing) == 0
        details = (
            "Both drop patterns recovered within threshold"
            if passed else
            f"Sub-tests failing: {failing}"
        )
        return TestResult(
            name="dropped_frames",
            passed=passed,
            metrics=sub_results,
            details=details,
            duration_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(report: RobustnessReport, output_dir: Path) -> Path:
        """
        Write the robustness report as an indented JSON file.

        Parameters
        ----------
        report     : ``RobustnessReport`` returned by ``run_all()``.
        output_dir : Directory to write into (created if absent).

        Returns
        -------
        Path to the written JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "robustness_report.json"
        tmp_path = out_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2, default=str)
            tmp_path.replace(out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        log.info("Robustness report written to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tracker(self) -> MultiObjectTracker:
        cfg = self.config
        tc = TrackerConfig(
            algorithm=TrackerAlgorithm(cfg.tracker_algorithm),
            min_hits_to_confirm=cfg.min_hits_to_confirm,
            max_misses_before_lost=cfg.max_misses_before_lost,
            max_misses_before_delete=cfg.max_misses_before_delete,
            iou_threshold=cfg.iou_threshold,
            frame_width=cfg.frame_width,
            frame_height=cfg.frame_height,
        )
        return MultiObjectTracker(tc)

    def _make_targets(
        self,
        n: int,
        *,
        seed_offset: int = 0,
    ) -> List[AnomalyTarget]:
        """
        Spawn N targets positioned in a non-overlapping grid pattern,
        each with a random velocity direction.
        """
        cfg = self.config
        fw, fh = cfg.frame_width, cfg.frame_height
        rng = np.random.default_rng(cfg.seed + seed_offset)

        cols = max(1, int(np.ceil(np.sqrt(n))))
        rows = max(1, int(np.ceil(n / cols)))
        cell_w = fw / cols
        cell_h = fh / rows

        palette = [
            (0, 120, 255),
            (255,  50,  50),
            ( 50, 220,  50),
            (180,  50, 200),
            ( 50, 200, 200),
        ]

        targets: List[AnomalyTarget] = []
        for i in range(n):
            row, col = divmod(i, cols)
            cx = cell_w * (col + 0.5)
            cy = cell_h * (row + 0.5)
            x = float(np.clip(cx - cfg.target_width / 2, 0, fw - cfg.target_width))
            y = float(np.clip(cy - cfg.target_height / 2, 0, fh - cfg.target_height))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            speed = rng.uniform(cfg.target_speed_px * 0.5, cfg.target_speed_px * 1.5)
            targets.append(AnomalyTarget(
                x=x, y=y,
                w=float(cfg.target_width),
                h=float(cfg.target_height),
                vx=float(speed * np.cos(angle)),
                vy=float(speed * np.sin(angle)),
                color=palette[i % len(palette)],
            ))
        return targets

    def _run_sequence(
        self,
        targets: List[AnomalyTarget],
        detector: _MockDetector,
        tracker: MultiObjectTracker,
        *,
        frame_aug: Callable[[np.ndarray, int, List[BBox]], np.ndarray],
        miss_rate_fn: Callable[[int], float],
        drop_fn: Callable[[int], bool],
        blind_frames: Set[int],
    ) -> List[_FrameSnapshot]:
        """
        Core evaluation loop shared by all five stress tests.

        Parameters
        ----------
        targets     : Moving anomaly targets (mutated in place each frame).
        detector    : Mock detector for this sequence.
        tracker     : Fresh ``MultiObjectTracker``.
        frame_aug   : ``(frame, frame_idx, gt_bboxes) → augmented_frame``.
        miss_rate_fn: ``frame_idx → float`` miss rate for the mock detector.
        drop_fn     : ``frame_idx → bool``; True = skip this frame entirely.
        blind_frames: Frame indices where the detector is forced blind
                      (augmentation still applied to the visual frame).

        Returns
        -------
        List[_FrameSnapshot] — one entry per frame.
        """
        cfg = self.config
        fw, fh = cfg.frame_width, cfg.frame_height
        bg_rng = np.random.default_rng(cfg.seed + 9999)
        background = _make_background(fh, fw, bg_rng)
        snapshots: List[_FrameSnapshot] = []

        for fi in range(cfg.n_frames):
            # Advance all targets.
            for t in targets:
                t.step(fw, fh)

            is_dropped = drop_fn(fi)

            if is_dropped:
                # No detection update — targets move but detector is silent.
                active = tracker.update(
                    _MockDetectionResult(score=0.0, rois=[]), None
                )
            else:
                # Render scene.
                frame, gt_bboxes = _render_frame(
                    background, targets, bg_rng
                )
                # Apply augmentation.
                aug_frame = frame_aug(frame, fi, gt_bboxes)
                # Update detector scene.
                detector.set_scene(
                    gt_bboxes,
                    miss_rate=miss_rate_fn(fi),
                    blind=(fi in blind_frames),
                )
                det = detector.detect(aug_frame)
                active = tracker.update(det, aug_frame)

            snapshots.append(_FrameSnapshot(
                frame_idx=fi,
                is_dropped=is_dropped,
                n_confirmed=sum(
                    1 for t in active if t.state == TrackState.CONFIRMED
                ),
                n_tentative=sum(
                    1 for t in active if t.state == TrackState.TENTATIVE
                ),
                primary_track_id=_primary_id(active),
            ))

        return snapshots


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_robustness_tests(
    output_dir: Optional[Path] = None,
    *,
    config: Optional[RobustnessConfig] = None,
    save: bool = True,
) -> RobustnessReport:
    """
    Run the full robustness suite and optionally persist the report.

    Parameters
    ----------
    output_dir : Directory for the JSON report.  Defaults to
                 ``results/robustness`` relative to the current directory.
    config     : Custom ``RobustnessConfig``; uses defaults when omitted.
    save       : If True, write the report JSON to ``output_dir``.

    Returns
    -------
    RobustnessReport
    """
    logging.basicConfig(level=logging.INFO)
    suite = RobustnessTestSuite(config)
    report = suite.run_all()
    if save:
        out = Path(output_dir) if output_dir is not None else Path("results/robustness")
        RobustnessTestSuite.save_report(report, out)
    return report
