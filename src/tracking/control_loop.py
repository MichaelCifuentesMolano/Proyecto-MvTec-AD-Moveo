"""
src/tracking/control_loop.py
=============================
Closed-loop robotic behavior: follow tracked anomalous targets, recenter
the camera when the target is lost, and raise alarms on critical anomalies.

The module is intentionally decoupled from any specific hardware layer.
All actuation is expressed as normalised pan/tilt velocities and optional
mobile-base velocities.  A ``CommandPublisher`` plug-in converts those
commands to the appropriate wire format (serial bytes, ROS topics, CAN
frames, stdout, etc.).

Behaviors
---------
IDLE         No anomalous target detected; camera holds its current pose.
TRACKING     A confirmed target is followed via dual-axis PID control.
RECENTERING  Target lost; camera returns to zero pose using proportional
             control on an integrated estimated gimbal angle.
ALARMING     Anomaly score ≥ alarm threshold **or** ROI severity == "critical";
             alarm output is held for ``alarm_persist_frames`` after the last
             trigger to suppress false negatives from momentary occlusion.

Single-frame pipeline
---------------------
  CameraStream
       │
       ▼
  DetectorInterface  ──► anomaly score, heatmap, ROI alerts
       │
       ▼
  MultiObjectTracker ──► CONFIRMED / TENTATIVE tracks
       │
       ▼
  BehaviorStateMachine
       │
       ▼
  PID / recenter / alarm logic
       │
       ▼
  ControlCommand ──► CommandPublisher (serial / callback / mock)

Public API
----------
>>> from tracking.tracker_core import make_tracker
>>> tracker = make_tracker("sort", frame_width=640, frame_height=480)
>>> loop = make_control_loop(camera, detector, tracker, alarm_threshold=0.75)
>>> stats = loop.run(max_frames=300)

Or frame-by-frame for external orchestration:

>>> cmd = loop.step(frame=bgr_array)
>>> print(cmd.behavior, cmd.pan_velocity, cmd.alarm)
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np

# Optional serial publisher.
try:
    import serial as _pyserial  # type: ignore
    _SERIAL_AVAILABLE = True
except ImportError:
    _pyserial = None  # type: ignore
    _SERIAL_AVAILABLE = False

# Intra-package imports.
from .tracker_core import BBox, MultiObjectTracker, TrackedObject, TrackState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-9
_HALF_INT16: float = 32767.0     # Scale factor for 16-bit serial encoding.
_DEFAULT_FPS: float = 30.0       # Used as dt fallback before first frame.


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BehaviorState(Enum):
    """Active control behavior in the current frame."""
    IDLE        = auto()   # No target; holding pose.
    TRACKING    = auto()   # Following a confirmed anomalous target.
    RECENTERING = auto()   # Returning camera to neutral after target loss.
    ALARMING    = auto()   # High-severity anomaly; alarm output active.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ControlConfig:
    """
    Complete configuration for ``ControlLoop``.

    All fields have conservative defaults suitable for a first deployment on a
    Jetson Orin Nano with a two-axis camera gimbal.  Override only what differs
    from the defaults.
    """

    # ── Frame geometry ─────────────────────────────────────────────────────
    frame_width: int = 640
    frame_height: int = 480

    # ── Target-following eligibility ───────────────────────────────────────
    follow_min_score: float = 0.30
    """Minimum anomaly score before a track is considered a follow target."""

    follow_confirmed_only: bool = True
    """When True, only CONFIRMED (not TENTATIVE) tracks are followed."""

    target_sticky_frames: int = 10
    """Frames to remain locked onto a chosen track before reconsidering."""

    # ── PID control — pan axis and tilt axis share the same gains ──────────
    pid_kp: float = 1.20
    pid_ki: float = 0.05
    pid_kd: float = 0.10
    pid_output_limit: float = 1.0
    """Output clamp: ±1 maps to full gimbal speed."""

    pid_integral_limit: float = 0.50
    """Anti-windup saturation on the integral accumulator."""

    # ── Camera recentering ─────────────────────────────────────────────────
    kp_recenter: float = 0.80
    """Proportional gain for return-to-centre after target loss."""

    recenter_dead_zone_deg: float = 1.0
    """Estimated angle magnitude (degrees) below which the camera is centred."""

    velocity_to_angle_deg: float = 5.0
    """Degrees added to the estimated angle per unit velocity per frame.
    Tune this to match the actual gimbal response at the chosen ``target_fps``."""

    # ── Alarm ──────────────────────────────────────────────────────────────
    alarm_score_threshold: float = 0.75
    """Frame-level or ROI peak score that triggers ALARMING."""

    alarm_severity_label: str = "critical"
    """ROI severity label string that always triggers ALARMING regardless of score."""

    alarm_persist_frames: int = 30
    """Hold the alarm output active for this many frames after the last trigger."""

    alarm_freeze_tracking: bool = False
    """If True, suppress pan/tilt commands while ALARMING (camera stays put)."""

    # ── Mobile base (disabled by default) ──────────────────────────────────
    enable_base_control: bool = False

    base_target_height_px: float = 150.0
    """Desired target bounding-box height in pixels (proxy for follow distance)."""

    kp_linear: float = 0.50
    kp_angular: float = 0.80
    max_linear_velocity: float = 0.30   # m/s
    max_angular_velocity: float = 1.00  # rad/s

    # ── Loop timing ────────────────────────────────────────────────────────
    target_fps: float = 30.0
    """Desired control frequency.  ``run()`` sleeps adaptively to match this."""

    # ── Diagnostics ────────────────────────────────────────────────────────
    log_every_n_frames: int = 30
    """Emit an INFO log line every N frames."""


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------

@dataclass
class ControlCommand:
    """
    Single-frame control output produced by ``ControlLoop.step()``.

    Velocity fields are normalised to **[-1, +1]** unless noted otherwise.
    Physical scaling (deg/s, m/s) is the ``CommandPublisher``'s responsibility.
    """

    timestamp: float
    frame_idx: int
    behavior: BehaviorState

    # Camera gimbal velocities.
    pan_velocity: float = 0.0    # +1 = right, -1 = left.
    tilt_velocity: float = 0.0   # +1 = down,  -1 = up.

    # Mobile base (non-zero only when ``enable_base_control=True``).
    linear_velocity: float = 0.0   # m/s, positive = forward.
    angular_velocity: float = 0.0  # rad/s, positive = counter-clockwise.

    # Alarm state.
    alarm: bool = False
    alarm_severity: str = "none"
    alarm_score: float = 0.0

    # Active tracking target (populated when behavior == TRACKING / ALARMING).
    target_track_id: Optional[int] = None
    target_bbox: Optional[BBox] = None
    target_score: float = 0.0
    frame_error_x: float = 0.0   # Normalised target-centre offset, +1 = far right.
    frame_error_y: float = 0.0   # Normalised target-centre offset, +1 = far down.

    # Diagnostics.
    n_active_tracks: int = 0
    detector_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_idx": self.frame_idx,
            "behavior": self.behavior.name,
            "pan_velocity": round(self.pan_velocity, 4),
            "tilt_velocity": round(self.tilt_velocity, 4),
            "linear_velocity": round(self.linear_velocity, 4),
            "angular_velocity": round(self.angular_velocity, 4),
            "alarm": self.alarm,
            "alarm_severity": self.alarm_severity,
            "alarm_score": round(self.alarm_score, 4),
            "target_track_id": self.target_track_id,
            "target_score": round(self.target_score, 4),
            "frame_error_x": round(self.frame_error_x, 4),
            "frame_error_y": round(self.frame_error_y, 4),
            "n_active_tracks": self.n_active_tracks,
            "detector_ms": round(self.detector_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"ControlCommand(frame={self.frame_idx}, behavior={self.behavior.name}, "
            f"pan={self.pan_velocity:+.3f}, tilt={self.tilt_velocity:+.3f}, "
            f"alarm={self.alarm}, track={self.target_track_id})"
        )


@dataclass
class LoopStats:
    """Cumulative statistics collected across an entire ``ControlLoop.run()``."""
    frames_processed: int = 0
    frames_idle: int = 0
    frames_tracking: int = 0
    frames_recentering: int = 0
    frames_alarming: int = 0
    alarm_events: int = 0
    """Distinct alarm trigger events (rising edges only)."""
    total_track_updates: int = 0
    dropped_camera_frames: int = 0
    avg_fps: float = 0.0
    avg_detector_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "frames_idle": self.frames_idle,
            "frames_tracking": self.frames_tracking,
            "frames_recentering": self.frames_recentering,
            "frames_alarming": self.frames_alarming,
            "alarm_events": self.alarm_events,
            "total_track_updates": self.total_track_updates,
            "dropped_camera_frames": self.dropped_camera_frames,
            "avg_fps": round(self.avg_fps, 2),
            "avg_detector_ms": round(self.avg_detector_ms, 2),
        }


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

class PIDController:
    """
    Discrete-time PID with real-time ``dt`` measurement and anti-windup.

    The time step is inferred from wall-clock time between calls.  The first
    call defaults to ``1 / 30`` s.  Pass ``now`` explicitly for unit tests or
    fixed-rate simulations.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        output_limit: float = 1.0,
        integral_limit: float = 0.5,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._out_lim = abs(output_limit)
        self._int_lim = abs(integral_limit)
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        """Clear state (call when switching targets or on lost-track events)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_t = None

    def compute(self, error: float, now: Optional[float] = None) -> float:
        """
        Compute and return the PID output for ``error``.

        Parameters
        ----------
        error : Signed error signal.  Positive → output drives the actuator
                in the positive direction.
        now   : Current time (seconds).  Defaults to ``time.monotonic()``.

        Returns
        -------
        float clamped to ``[-output_limit, +output_limit]``.
        """
        t = time.monotonic() if now is None else now
        dt = (t - self._last_t) if self._last_t is not None else (1.0 / _DEFAULT_FPS)
        dt = max(dt, _EPS)
        self._last_t = t

        self._integral = float(
            np.clip(self._integral + error * dt, -self._int_lim, self._int_lim)
        )
        derivative = (error - self._prev_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return float(np.clip(output, -self._out_lim, self._out_lim))


# ---------------------------------------------------------------------------
# Command publishers
# ---------------------------------------------------------------------------

class CommandPublisher:
    """
    Abstract base for command output adapters.

    Subclass and override ``publish()`` to integrate any downstream system
    (ROS topic, CAN bus, REST endpoint, etc.).
    """

    def publish(self, cmd: ControlCommand) -> None:  # noqa: U100
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (serial port, socket, …)."""


class MockPublisher(CommandPublisher):
    """Logs each command at DEBUG level — suitable for unit tests and dry runs."""

    def publish(self, cmd: ControlCommand) -> None:
        log.debug(
            "[%s] pan=%+.3f tilt=%+.3f alarm=%s track=%s score=%.3f",
            cmd.behavior.name,
            cmd.pan_velocity,
            cmd.tilt_velocity,
            cmd.alarm,
            cmd.target_track_id,
            cmd.target_score,
        )

    def close(self) -> None:
        pass


class CallbackPublisher(CommandPublisher):
    """Calls a user-supplied callable for each command — zero serialisation overhead."""

    def __init__(self, fn: Callable[[ControlCommand], None]) -> None:
        self._fn = fn

    def publish(self, cmd: ControlCommand) -> None:
        self._fn(cmd)

    def close(self) -> None:
        pass


class SerialPublisher(CommandPublisher):
    """
    Pack a ``ControlCommand`` into a compact binary frame and write to UART.

    Wire frame (10 bytes total, big-endian):

        Offset  Size  Field
        ──────  ────  ──────────────────────────────────────────────────────
          0      1    Header sentinel: 0x43 (``'C'``)
          1      2    pan_i    int16  — pan_velocity × 32767
          3      2    tilt_i   int16  — tilt_velocity × 32767
          5      2    lin_i    int16  — linear_velocity × 1000  (mm/s)
          7      2    ang_i    int16  — angular_velocity × 1000 (mrad/s)
          9      1    alarm    uint8  — 0x00 | 0x01
         10      1    checksum uint8  — sum(bytes[1:10]) mod 256
    """

    _HEADER: bytes = b"\x43"
    _FMT: str = ">hhhhB"                          # 4 × int16 + uint8
    _PAYLOAD_LEN: int = struct.calcsize(_FMT)

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0) -> None:
        if not _SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial is not installed.  "
                "Run: pip install pyserial"
            )
        self._ser = _pyserial.Serial(port, baud, timeout=timeout)
        log.info("SerialPublisher open: %s @ %d baud.", port, baud)

    def _encode(self, cmd: ControlCommand) -> bytes:
        def i16(v: float, scale: float = _HALF_INT16) -> int:
            return int(np.clip(v * scale, -_HALF_INT16, _HALF_INT16))

        pan_i  = i16(cmd.pan_velocity)
        tilt_i = i16(cmd.tilt_velocity)
        lin_i  = i16(cmd.linear_velocity, scale=1000.0)   # mm/s
        ang_i  = i16(cmd.angular_velocity, scale=1000.0)  # mrad/s
        alarm_b = 0x01 if cmd.alarm else 0x00
        payload = struct.pack(self._FMT, pan_i, tilt_i, lin_i, ang_i, alarm_b)
        checksum = sum(payload) & 0xFF
        return self._HEADER + payload + bytes([checksum])

    def publish(self, cmd: ControlCommand) -> None:
        frame = self._encode(cmd)
        try:
            self._ser.write(frame)
        except Exception as exc:  # noqa: BLE001
            log.error("SerialPublisher write error: %s", exc)

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()
            log.info("SerialPublisher closed.")


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

class ControlLoop:
    """
    Closed-loop robotic behavior integrating camera acquisition, anomaly
    detection, and multi-object tracking into a single control cycle.

    Parameters
    ----------
    camera_stream
        A ``CameraStream`` (from ``camera_stream.py``) or ``None`` when frames
        are supplied directly to ``step(frame=...)``.
    detector
        A ``DetectorInterface`` instance (from ``detector_interface.py``).
    tracker
        A pre-built ``MultiObjectTracker`` (from ``tracker_core.py``).
        The tracker's ``frame_width`` / ``frame_height`` must match ``config``.
    config
        ``ControlConfig``; constructed with defaults when omitted.
    publisher
        ``CommandPublisher`` implementation.  Defaults to ``MockPublisher``.
    on_alarm
        Called on every frame where ``cmd.alarm is True``.
        Signature: ``(cmd: ControlCommand) -> None``.
    on_command
        Called on every frame after publishing.
        Signature: ``(cmd: ControlCommand) -> None``.
    """

    def __init__(
        self,
        camera_stream,
        detector,
        tracker: MultiObjectTracker,
        *,
        config: Optional[ControlConfig] = None,
        publisher: Optional[CommandPublisher] = None,
        on_alarm: Optional[Callable[[ControlCommand], None]] = None,
        on_command: Optional[Callable[[ControlCommand], None]] = None,
    ) -> None:
        self._camera = camera_stream
        self._detector = detector
        self._tracker = tracker
        self.config: ControlConfig = config or ControlConfig()
        self._publisher: CommandPublisher = publisher or MockPublisher()
        self._on_alarm = on_alarm
        self._on_command = on_command

        cfg = self.config

        # ── PID controllers (one per gimbal axis) ──────────────────────
        pid_kwargs = dict(
            output_limit=cfg.pid_output_limit,
            integral_limit=cfg.pid_integral_limit,
        )
        self._pid_pan = PIDController(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd, **pid_kwargs)
        self._pid_tilt = PIDController(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd, **pid_kwargs)

        # ── Behavior state ─────────────────────────────────────────────
        self._behavior: BehaviorState = BehaviorState.IDLE
        self._alarm_countdown: int = 0       # Frames remaining in alarm hold.
        self._prev_alarm_active: bool = False  # For rising-edge detection.

        # ── Estimated camera pose (integrated velocity, degrees) ───────
        self._est_pan_deg: float = 0.0    # +ve = pointing right of centre.
        self._est_tilt_deg: float = 0.0   # +ve = pointing below centre.

        # ── Target stickiness ─────────────────────────────────────────
        self._current_target_id: Optional[int] = None
        self._target_sticky_remaining: int = 0

        # ── Diagnostics ───────────────────────────────────────────────
        self._frame_idx: int = 0
        self._stats: LoopStats = LoopStats()
        self._fps_buf: List[float] = []
        self._det_ms_buf: List[float] = []
        self._last_frame_t: Optional[float] = None

        # ── Threading ─────────────────────────────────────────────────
        self._stop_event = threading.Event()

        log.info(
            "ControlLoop ready [algo=%s, fps=%.1f, alarm_thr=%.2f, "
            "follow_min=%.2f].",
            getattr(tracker, "_algorithm", "?"),
            cfg.target_fps,
            cfg.alarm_score_threshold,
            cfg.follow_min_score,
        )

    # ------------------------------------------------------------------
    # Public API — frame-level entry point
    # ------------------------------------------------------------------

    def step(self, frame: Optional[np.ndarray] = None) -> ControlCommand:
        """
        Process one frame and return the resulting ``ControlCommand``.

        When ``frame`` is ``None`` a frame is read from the attached
        ``camera_stream`` (blocking, up to 1 s).

        Parameters
        ----------
        frame : Optional BGR uint8 array (H, W, 3).

        Returns
        -------
        ControlCommand

        Raises
        ------
        RuntimeError
            If no frame is available (camera disconnected / 1 s timeout).
        """
        # ── 1. Acquire frame ──────────────────────────────────────────
        bgr = self._acquire_frame(frame)
        self._frame_idx += 1

        # ── 2. Run anomaly detector ───────────────────────────────────
        t0 = time.monotonic()
        try:
            detection_result = self._detector.detect(bgr)
        except Exception as exc:  # noqa: BLE001
            log.error("Detector error on frame %d: %s", self._frame_idx, exc)
            return self._idle_command(detector_ms=0.0)
        detector_ms = (time.monotonic() - t0) * 1e3
        self._det_ms_buf.append(detector_ms)
        if len(self._det_ms_buf) > 60:
            self._det_ms_buf.pop(0)

        # ── 3. Update tracker ─────────────────────────────────────────
        active_tracks: List[TrackedObject] = self._tracker.update(
            detection_result, bgr
        )

        # ── 4. Check alarm ────────────────────────────────────────────
        alarm_triggered, alarm_sev, alarm_score = self._check_alarm(
            detection_result, active_tracks
        )
        # Rising-edge detection for alarm event counter.
        if alarm_triggered and not self._prev_alarm_active:
            self._stats.alarm_events += 1
        self._prev_alarm_active = alarm_triggered

        if alarm_triggered:
            self._alarm_countdown = self.config.alarm_persist_frames
        elif self._alarm_countdown > 0:
            self._alarm_countdown -= 1
        alarm_out = self._alarm_countdown > 0

        # ── 5. Select follow target ───────────────────────────────────
        target = self._select_target(active_tracks)

        # ── 6. Compute command ────────────────────────────────────────
        now = time.monotonic()
        cmd = self._compute_command(
            target, active_tracks,
            alarm_out, alarm_sev, alarm_score,
            detector_ms, now,
        )

        # ── 7. Publish and fire callbacks ─────────────────────────────
        self._safe_call(self._publisher.publish, cmd, label="publisher")

        if alarm_out and self._on_alarm is not None:
            self._safe_call(self._on_alarm, cmd, label="on_alarm")

        if self._on_command is not None:
            self._safe_call(self._on_command, cmd, label="on_command")

        # ── 8. Update statistics ──────────────────────────────────────
        self._update_stats(cmd, now)

        # ── 9. Periodic diagnostic log ────────────────────────────────
        if self._frame_idx % self.config.log_every_n_frames == 0:
            log.info(
                "Frame %d | %-11s | tracks=%d | alarm=%s (%.3f) "
                "| det=%.1f ms | fps=%.1f",
                self._frame_idx,
                self._behavior.name,
                len(active_tracks),
                alarm_out,
                alarm_score,
                detector_ms,
                self._stats.avg_fps,
            )

        return cmd

    # ------------------------------------------------------------------
    # Public API — main loop
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        on_command: Optional[Callable[[ControlCommand], None]] = None,
    ) -> LoopStats:
        """
        Execute the control loop until ``stop()`` is called or
        ``max_frames`` is reached.

        The loop opens the camera stream on entry, maintains the target FPS
        via adaptive sleeping, and closes all resources on exit (including on
        ``KeyboardInterrupt``).

        Parameters
        ----------
        max_frames  : Exit after this many frames (``None`` = run forever).
        on_command  : Per-frame callback supplementing the instance-level one.

        Returns
        -------
        LoopStats  — statistics accumulated during this run.
        """
        self._stop_event.clear()
        cfg = self.config
        frame_budget = 1.0 / max(cfg.target_fps, _EPS)

        if self._camera is not None:
            self._camera.open()

        log.info(
            "ControlLoop.run() started [target_fps=%.1f, max_frames=%s].",
            cfg.target_fps,
            max_frames,
        )

        try:
            while not self._stop_event.is_set():
                if max_frames is not None and self._frame_idx >= max_frames:
                    log.info("max_frames=%d reached; stopping.", max_frames)
                    break

                t_loop = time.monotonic()

                try:
                    cmd = self.step()
                except RuntimeError as exc:
                    log.error("step() error: %s — waiting 0.5 s.", exc)
                    time.sleep(0.5)
                    continue

                if on_command is not None:
                    self._safe_call(on_command, cmd, label="run.on_command")

                # Adaptive sleep to respect target_fps.
                elapsed = time.monotonic() - t_loop
                sleep_s = frame_budget - elapsed
                if sleep_s > 1e-4:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            log.info("ControlLoop interrupted by keyboard.")
        finally:
            if self._camera is not None:
                self._camera.close()
            self._publisher.close()
            log.info("ControlLoop stopped. Stats: %s", self._stats.to_dict())

        return self._stats

    # ------------------------------------------------------------------
    # Public API — control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal ``run()`` to exit cleanly after the current frame."""
        self._stop_event.set()
        log.info("ControlLoop stop requested.")

    def reset(self) -> None:
        """Reset behavior, PIDs, pose estimate, stats, and tracker."""
        self._behavior = BehaviorState.IDLE
        self._alarm_countdown = 0
        self._prev_alarm_active = False
        self._est_pan_deg = 0.0
        self._est_tilt_deg = 0.0
        self._current_target_id = None
        self._target_sticky_remaining = 0
        self._frame_idx = 0
        self._stats = LoopStats()
        self._fps_buf.clear()
        self._det_ms_buf.clear()
        self._last_frame_t = None
        self._pid_pan.reset()
        self._pid_tilt.reset()
        self._tracker.reset()
        log.info("ControlLoop reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def behavior(self) -> BehaviorState:
        return self._behavior

    @property
    def stats(self) -> LoopStats:
        return self._stats

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def estimated_pose_deg(self) -> Tuple[float, float]:
        """Current (pan_deg, tilt_deg) estimated from integrated velocity."""
        return self._est_pan_deg, self._est_tilt_deg

    # ------------------------------------------------------------------
    # Internal — frame acquisition
    # ------------------------------------------------------------------

    def _acquire_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is not None:
            return frame
        if self._camera is None:
            raise RuntimeError(
                "No frame supplied and no camera_stream attached. "
                "Pass frame= to step() or attach a CameraStream."
            )
        # Import locally to avoid circular dependency at module load time.
        cam_frame = self._camera.read_blocking(timeout=1.0)
        if cam_frame is None:
            raise RuntimeError(
                "Camera read timeout (> 1 s) in ControlLoop.step()."
            )
        return cam_frame.bgr

    # ------------------------------------------------------------------
    # Internal — alarm detection
    # ------------------------------------------------------------------

    def _check_alarm(
        self,
        detection_result,
        active_tracks: List[TrackedObject],
    ) -> Tuple[bool, str, float]:
        """
        Determine whether an alarm condition is present this frame.

        Checks (in priority order):
        1. Global frame anomaly score ≥ threshold.
        2. Any ROI has severity == alarm_severity_label.
        3. Any ROI has peak_score ≥ threshold.

        Returns
        -------
        (alarm_active, severity_label, trigger_score)
        """
        cfg = self.config
        frame_score = float(getattr(detection_result, "score", 0.0))

        if frame_score >= cfg.alarm_score_threshold:
            return True, "score", frame_score

        rois = getattr(detection_result, "rois", None) or []
        for roi in rois:
            sev = str(getattr(roi, "severity", ""))
            peak = float(getattr(roi, "peak_score", 0.0))
            if sev == cfg.alarm_severity_label:
                return True, "critical", peak
            if peak >= cfg.alarm_score_threshold:
                return True, "peak_score", peak

        return False, "none", frame_score

    # ------------------------------------------------------------------
    # Internal — target selection
    # ------------------------------------------------------------------

    def _select_target(
        self, active_tracks: List[TrackedObject]
    ) -> Optional[TrackedObject]:
        """
        Choose the highest-priority anomalous track to follow.

        Selection rules (in order):
        1. Track must have ``anomaly_score ≥ follow_min_score``.
        2. If ``follow_confirmed_only``, only CONFIRMED tracks qualify.
        3. Prefer the current track for ``target_sticky_frames`` to avoid
           oscillation between nearby tracks of similar score.
        4. Break ties by highest ``anomaly_score``.
        """
        cfg = self.config

        eligible = [
            t for t in active_tracks
            if (
                t.anomaly_score >= cfg.follow_min_score
                and (
                    not cfg.follow_confirmed_only
                    or t.state == TrackState.CONFIRMED
                )
            )
        ]

        if not eligible:
            self._current_target_id = None
            self._target_sticky_remaining = 0
            return None

        # Stickiness: stay on the current target while the counter lasts.
        if self._current_target_id is not None and self._target_sticky_remaining > 0:
            sticky = next(
                (t for t in eligible if t.track_id == self._current_target_id),
                None,
            )
            if sticky is not None:
                self._target_sticky_remaining -= 1
                return sticky

        # Pick the best candidate by score.
        best = max(eligible, key=lambda t: t.anomaly_score)
        if best.track_id != self._current_target_id:
            log.debug(
                "Target switch: id=%s → id=%d (score=%.3f).",
                self._current_target_id,
                best.track_id,
                best.anomaly_score,
            )
        self._current_target_id = best.track_id
        self._target_sticky_remaining = cfg.target_sticky_frames
        return best

    # ------------------------------------------------------------------
    # Internal — command computation (behavior state machine)
    # ------------------------------------------------------------------

    def _compute_command(
        self,
        target: Optional[TrackedObject],
        active_tracks: List[TrackedObject],
        alarm_out: bool,
        alarm_sev: str,
        alarm_score: float,
        detector_ms: float,
        now: float,
    ) -> ControlCommand:
        """
        Apply the behavior state machine and compute the ``ControlCommand``.

        Priority: ALARMING > TRACKING > RECENTERING > IDLE.
        """
        cfg = self.config
        shared = dict(
            timestamp=now,
            frame_idx=self._frame_idx,
            alarm=alarm_out,
            alarm_severity=alarm_sev if alarm_out else "none",
            alarm_score=alarm_score if alarm_out else 0.0,
            n_active_tracks=len(active_tracks),
            detector_ms=detector_ms,
        )

        # ── ALARMING ──────────────────────────────────────────────────
        if alarm_out:
            self._behavior = BehaviorState.ALARMING
            if cfg.alarm_freeze_tracking or target is None:
                pan_v = tilt_v = err_x = err_y = 0.0
                self._pid_pan.reset()
                self._pid_tilt.reset()
            else:
                pan_v, tilt_v, err_x, err_y = self._pid_follow(target, now)
            self._integrate_pose(pan_v, tilt_v)
            lin_v, ang_v = self._base_velocity(target)
            return ControlCommand(
                behavior=BehaviorState.ALARMING,
                pan_velocity=pan_v,
                tilt_velocity=tilt_v,
                linear_velocity=lin_v,
                angular_velocity=ang_v,
                target_track_id=target.track_id if target else None,
                target_bbox=target.bbox if target else None,
                target_score=target.anomaly_score if target else 0.0,
                frame_error_x=err_x,
                frame_error_y=err_y,
                **shared,
            )

        # ── TRACKING ──────────────────────────────────────────────────
        if target is not None:
            self._behavior = BehaviorState.TRACKING
            pan_v, tilt_v, err_x, err_y = self._pid_follow(target, now)
            self._integrate_pose(pan_v, tilt_v)
            lin_v, ang_v = self._base_velocity(target)
            return ControlCommand(
                behavior=BehaviorState.TRACKING,
                pan_velocity=pan_v,
                tilt_velocity=tilt_v,
                linear_velocity=lin_v,
                angular_velocity=ang_v,
                target_track_id=target.track_id,
                target_bbox=target.bbox,
                target_score=target.anomaly_score,
                frame_error_x=err_x,
                frame_error_y=err_y,
                **shared,
            )

        # ── RECENTERING ───────────────────────────────────────────────
        pose_mag = math.hypot(self._est_pan_deg, self._est_tilt_deg)
        if pose_mag > cfg.recenter_dead_zone_deg:
            self._behavior = BehaviorState.RECENTERING
            pan_v, tilt_v = self._recenter_velocities()
            self._integrate_pose(pan_v, tilt_v)
            self._pid_pan.reset()
            self._pid_tilt.reset()
            return ControlCommand(
                behavior=BehaviorState.RECENTERING,
                pan_velocity=pan_v,
                tilt_velocity=tilt_v,
                **shared,
            )

        # ── IDLE ──────────────────────────────────────────────────────
        self._behavior = BehaviorState.IDLE
        self._pid_pan.reset()
        self._pid_tilt.reset()
        return ControlCommand(behavior=BehaviorState.IDLE, **shared)

    # ------------------------------------------------------------------
    # Internal — actuator helpers
    # ------------------------------------------------------------------

    def _pid_follow(
        self, target: TrackedObject, now: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute PID pan/tilt velocities to centre ``target`` in the frame.

        Returns
        -------
        (pan_velocity, tilt_velocity, normalised_err_x, normalised_err_y)
        """
        cfg = self.config
        cx_t, cy_t = target.bbox.center
        half_w = cfg.frame_width * 0.5
        half_h = cfg.frame_height * 0.5

        # Normalised error: +1 = target is at far right/bottom edge.
        err_x = (cx_t - half_w) / (half_w + _EPS)
        err_y = (cy_t - half_h) / (half_h + _EPS)

        pan_v  = self._pid_pan.compute(err_x, now)
        tilt_v = self._pid_tilt.compute(err_y, now)
        return pan_v, tilt_v, err_x, err_y

    def _recenter_velocities(self) -> Tuple[float, float]:
        """
        Proportional return-to-centre command based on estimated gimbal angle.

        Outputs are clamped to ±1.
        """
        kp = self.config.kp_recenter
        pan_v  = float(np.clip(-kp * self._est_pan_deg  / 90.0, -1.0, 1.0))
        tilt_v = float(np.clip(-kp * self._est_tilt_deg / 90.0, -1.0, 1.0))
        return pan_v, tilt_v

    def _base_velocity(
        self, target: Optional[TrackedObject]
    ) -> Tuple[float, float]:
        """
        Compute mobile-base (linear, angular) velocities.

        Linear velocity is proportional to the deviation of the target bbox
        height from ``base_target_height_px`` (positive = move forward).
        Angular velocity aligns the robot heading with the target centre.

        Returns (0.0, 0.0) when base control is disabled or target is None.
        """
        cfg = self.config
        if not cfg.enable_base_control or target is None:
            return 0.0, 0.0

        cx_t, _ = target.bbox.center
        half_w = cfg.frame_width * 0.5

        # Angular: align robot heading (negative = target is right → turn right).
        norm_x = (cx_t - half_w) / (half_w + _EPS)
        ang_v = float(
            np.clip(
                -cfg.kp_angular * norm_x,
                -cfg.max_angular_velocity,
                cfg.max_angular_velocity,
            )
        )

        # Linear: close-in when bbox height < target, back off when too close.
        height_err = (target.bbox.height - cfg.base_target_height_px) / (
            cfg.base_target_height_px + _EPS
        )
        lin_v = float(
            np.clip(
                cfg.kp_linear * height_err,
                -cfg.max_linear_velocity,
                cfg.max_linear_velocity,
            )
        )
        return lin_v, ang_v

    def _integrate_pose(self, pan_v: float, tilt_v: float) -> None:
        """
        Integrate velocity commands into an estimated gimbal angle (degrees).

        The scale factor converts one unit of velocity at ``target_fps`` into
        ``velocity_to_angle_deg`` degrees of camera rotation.
        """
        fps = max(self.config.target_fps, _EPS)
        scale = self.config.velocity_to_angle_deg / fps
        self._est_pan_deg  = float(np.clip(self._est_pan_deg  + pan_v  * scale, -90.0, 90.0))
        self._est_tilt_deg = float(np.clip(self._est_tilt_deg + tilt_v * scale, -90.0, 90.0))

    # ------------------------------------------------------------------
    # Internal — statistics
    # ------------------------------------------------------------------

    def _update_stats(self, cmd: ControlCommand, now: float) -> None:
        s = self._stats
        s.frames_processed += 1
        s.total_track_updates += cmd.n_active_tracks

        match cmd.behavior:
            case BehaviorState.IDLE:
                s.frames_idle += 1
            case BehaviorState.TRACKING:
                s.frames_tracking += 1
            case BehaviorState.RECENTERING:
                s.frames_recentering += 1
            case BehaviorState.ALARMING:
                s.frames_alarming += 1

        # Dropped camera frames (read from camera stats if available).
        if self._camera is not None:
            cam_stats = getattr(self._camera, "stats", None)
            if cam_stats is not None:
                s.dropped_camera_frames = int(
                    getattr(cam_stats, "dropped_frames", 0)
                )

        # Rolling FPS (last 60 frames).
        if self._last_frame_t is not None:
            fps = 1.0 / max(now - self._last_frame_t, _EPS)
            self._fps_buf.append(fps)
            if len(self._fps_buf) > 60:
                self._fps_buf.pop(0)
            s.avg_fps = float(np.mean(self._fps_buf))
        self._last_frame_t = now

        # Rolling detector latency.
        if self._det_ms_buf:
            s.avg_detector_ms = float(np.mean(self._det_ms_buf))

    # ------------------------------------------------------------------
    # Internal — error-tolerant callback invocation
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_call(fn: Callable, arg, *, label: str) -> None:
        try:
            fn(arg)
        except Exception as exc:  # noqa: BLE001
            log.error("%s callback raised an exception: %s", label, exc)

    # ------------------------------------------------------------------
    # Internal — idle command factory
    # ------------------------------------------------------------------

    def _idle_command(self, detector_ms: float = 0.0) -> ControlCommand:
        return ControlCommand(
            timestamp=time.monotonic(),
            frame_idx=self._frame_idx,
            behavior=BehaviorState.IDLE,
            detector_ms=detector_ms,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_control_loop(
    camera_stream,
    detector,
    tracker: MultiObjectTracker,
    *,
    alarm_threshold: float = 0.75,
    follow_min_score: float = 0.30,
    target_fps: float = 30.0,
    frame_width: int = 640,
    frame_height: int = 480,
    enable_serial: bool = False,
    serial_port: str = "/dev/ttyUSB0",
    serial_baud: int = 115200,
    on_alarm: Optional[Callable[[ControlCommand], None]] = None,
    on_command: Optional[Callable[[ControlCommand], None]] = None,
    **config_kwargs,
) -> ControlLoop:
    """
    Build a ``ControlLoop`` from plain scalar arguments.

    Parameters
    ----------
    camera_stream     : ``CameraStream`` or ``None``.
    detector          : ``DetectorInterface``.
    tracker           : Pre-built ``MultiObjectTracker``.
    alarm_threshold   : Anomaly score that triggers ALARMING.
    follow_min_score  : Minimum score to qualify a track for following.
    target_fps        : Desired loop frequency.
    frame_width/height: Frame dimensions in pixels.
    enable_serial     : Attach a ``SerialPublisher`` instead of ``MockPublisher``.
    serial_port       : UART device path (only used when ``enable_serial=True``).
    serial_baud       : UART baud rate.
    on_alarm          : Callback fired on every alarming frame.
    on_command        : Callback fired on every frame (post-publish).
    **config_kwargs   : Additional ``ControlConfig`` fields.

    Returns
    -------
    ControlLoop

    Examples
    --------
    >>> loop = make_control_loop(
    ...     camera, detector, tracker,
    ...     alarm_threshold=0.80,
    ...     follow_min_score=0.35,
    ...     target_fps=25.0,
    ...     frame_width=1280,
    ...     frame_height=720,
    ... )
    >>> stats = loop.run(max_frames=1000)
    """
    cfg = ControlConfig(
        alarm_score_threshold=alarm_threshold,
        follow_min_score=follow_min_score,
        target_fps=target_fps,
        frame_width=frame_width,
        frame_height=frame_height,
        **config_kwargs,
    )

    publisher: CommandPublisher
    if enable_serial:
        publisher = SerialPublisher(serial_port, serial_baud)
    else:
        publisher = MockPublisher()

    return ControlLoop(
        camera_stream,
        detector,
        tracker,
        config=cfg,
        publisher=publisher,
        on_alarm=on_alarm,
        on_command=on_command,
    )
