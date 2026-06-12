"""
src/tracking/camera_stream.py

Camera and robot-sensor frame acquisition for the visual tracking pipeline.

Supported sources
-----------------
usb         — USB / V4L2 webcam via OpenCV VideoCapture.
csi         — Jetson MIPI CSI-2 camera via GStreamer + nvarguscamerasrc.
realsense   — Intel RealSense colour stream (requires pyrealsense2).
video_file  — Pre-recorded video file for offline development / CI.
mock        — Synthetic pattern generator; zero hardware required.

Threading model
---------------
A daemon background thread reads raw frames from the capture backend as fast
as the sensor allows and appends them to a ``collections.deque(maxlen=N)``.
The consumer side (tracking loop) calls ``read()`` or ``read_blocking()`` and
always receives the *latest* available frame.  Old frames are automatically
evicted when the deque is full, keeping latency bounded regardless of
processing speed.

Preprocessing pipeline (per-frame, in the reader thread)
---------------------------------------------------------
1. Optional ``cv2.resize`` to the configured target resolution.
2. BGR → RGB colour conversion (unless ``convert_rgb=False``).
3. Optional ImageNet normalisation to a ``(1, 3, H, W)`` float32 tensor,
   ready to be passed directly into the TensorRT engine runner.

Jetson CSI pipeline
-------------------
``build_csi_pipeline()`` generates the full nvarguscamerasrc GStreamer string.
Pass the returned string to ``CameraConfig(gstreamer_pipeline=…)`` or set
``source=CameraSource.CSI`` and the config will build it automatically.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator, Iterator

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import pyrealsense2 as rs  # type: ignore[import]
    _RS_AVAILABLE = True
except ImportError:
    rs = None  # type: ignore[assignment]
    _RS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 30
_MAX_RECONNECT = 5          # attempts before giving up on a failing camera
_RECONNECT_DELAY_S = 1.0    # seconds between reconnect attempts


# ---------------------------------------------------------------------------
# Enums and exceptions
# ---------------------------------------------------------------------------


class CameraSource(str, Enum):
    """Supported acquisition backends."""
    USB = "usb"
    CSI = "csi"
    REALSENSE = "realsense"
    VIDEO_FILE = "video_file"
    MOCK = "mock"


class CameraOpenError(RuntimeError):
    """Raised when a camera cannot be opened."""


class CameraReadError(RuntimeError):
    """Raised after exhausting reconnect attempts."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CameraConfig:
    """
    Full configuration for a :class:`CameraStream`.

    Parameters
    ----------
    source : CameraSource
        Acquisition backend.
    device_id : int
        USB device index (``/dev/videoN``) or CSI sensor ID.
    width, height : int
        Requested capture resolution.  Actual resolution is reported in
        :attr:`CameraFrame.width` / ``height``.
    fps : int
        Requested frame rate.  Actual FPS may differ; see
        :attr:`CameraStream.actual_fps`.
    video_path : str or None
        Path to a video file; required when ``source=VIDEO_FILE``.
    loop_video : bool
        Restart the video file from the beginning when it ends.
    target_size : (H, W) or None
        If given, each frame is resized to this resolution after capture.
        Useful for aligning the sensor resolution with the model input size.
    convert_rgb : bool
        Convert BGR (OpenCV default) to RGB before delivering frames.
    normalize : bool
        Apply ImageNet normalisation and package the frame as a
        ``(1, 3, H, W)`` float32 PyTorch tensor in ``CameraFrame.tensor``.
        Requires PyTorch.  Has no effect when ``convert_rgb=False``.
    buffer_size : int
        Maximum number of frames kept in the internal ring buffer.  The
        oldest frame is evicted when the buffer is full (real-time drop
        policy).
    gstreamer_pipeline : str or None
        Custom GStreamer pipeline string passed verbatim to OpenCV.
        Overrides automatic pipeline generation for CSI and USB sources.
    flip_method : int
        Rotation / flip applied by nvvidconv for CSI sources.
        0=none, 1=ccw-90, 2=180°, 3=cw-90, 4=h-flip, 6=v-flip.
    """

    source: CameraSource = CameraSource.USB
    device_id: int = 0
    width: int = _DEFAULT_WIDTH
    height: int = _DEFAULT_HEIGHT
    fps: int = _DEFAULT_FPS
    video_path: str | None = None
    loop_video: bool = False
    target_size: tuple[int, int] | None = None   # (H, W)
    convert_rgb: bool = True
    normalize: bool = True
    buffer_size: int = 2
    gstreamer_pipeline: str | None = None
    flip_method: int = 0


# ---------------------------------------------------------------------------
# Frame container
# ---------------------------------------------------------------------------


@dataclass
class CameraFrame:
    """
    One captured and pre-processed frame.

    Attributes
    ----------
    image : np.ndarray
        ``(H, W, 3)`` uint8 image in RGB (or BGR if ``convert_rgb=False``).
    tensor : torch.Tensor or None
        ``(1, 3, H, W)`` float32 ImageNet-normalised tensor.  None when
        ``config.normalize=False`` or PyTorch is unavailable.
    timestamp : float
        ``time.perf_counter()`` at the moment the raw frame was captured.
    frame_idx : int
        Monotonically increasing frame counter (across reconnects).
    source : str
        Backend name (e.g. "usb", "csi").
    width, height : int
        Spatial dimensions of ``image`` (after any resize).
    latency_ms : float
        Elapsed time from ``timestamp`` to the moment preprocessing finished.
    """

    image: np.ndarray
    tensor: "torch.Tensor | None"
    timestamp: float
    frame_idx: int
    source: str
    width: int
    height: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Stream statistics
# ---------------------------------------------------------------------------


@dataclass
class StreamStats:
    """Live statistics snapshot from a running :class:`CameraStream`."""

    frame_count: int         # total frames captured since open()
    drop_count: int          # frames evicted from the ring buffer before read
    actual_fps: float        # measured capture rate (rolling 30-frame window)
    buffer_depth: int        # current number of frames waiting in the buffer
    is_running: bool
    uptime_s: float


# ---------------------------------------------------------------------------
# Internal capture backends
# ---------------------------------------------------------------------------


class _MockCapture:
    """
    Synthetic frame generator — no hardware required.

    Produces a simple pattern (grey ramp + red square) that changes each
    frame so downstream detectors see a plausible moving signal.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self._w = width
        self._h = height
        self._delay = 1.0 / max(fps, 1)
        self._idx = 0
        self._opened = True

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(self._delay)
        frame = np.full((self._h, self._w, 3), 80, dtype=np.uint8)
        # Moving red square: position cycles with frame index
        offset = int((self._idx * 3) % (self._w // 2))
        x0 = self._w // 4 + offset
        x1 = x0 + self._w // 8
        y0 = self._h // 4
        y1 = 3 * self._h // 4
        frame[y0:y1, x0:min(x1, self._w), 2] = 220   # red channel
        self._idx += 1
        return True, frame

    def get(self, prop: int) -> float:
        _map = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self._w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._h),
            cv2.CAP_PROP_FPS: float(1.0 / self._delay),
        } if _CV2_AVAILABLE else {}
        return _map.get(prop, 0.0)

    def set(self, _prop: int, _val: float) -> bool:
        return True

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self._opened = False


class _RealSenseCapture:
    """
    Wraps a ``pyrealsense2`` colour stream as an OpenCV-compatible capture.

    Only the colour channel is used; depth data is ignored.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        if not _RS_AVAILABLE:
            raise CameraOpenError(
                "pyrealsense2 is not installed.  "
                "Install with: pip install pyrealsense2"
            )
        self._w = width
        self._h = height
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )
        try:
            self._pipeline.start(cfg)
        except Exception as exc:
            raise CameraOpenError(f"RealSense pipeline failed to start: {exc}") from exc
        self._opened = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2_000)
            colour = frames.get_color_frame()
            if not colour:
                return False, None
            return True, np.asanyarray(colour.get_data())
        except Exception as exc:
            log.warning("RealSense read error: %s", exc)
            return False, None

    def get(self, prop: int) -> float:
        _map = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self._w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self._h),
            cv2.CAP_PROP_FPS: 30.0,
        } if _CV2_AVAILABLE else {}
        return _map.get(prop, 0.0)

    def set(self, _prop: int, _val: float) -> bool:
        return False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass
        self._opened = False


# ---------------------------------------------------------------------------
# GStreamer pipeline builders
# ---------------------------------------------------------------------------


def build_csi_pipeline(
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
    *,
    sensor_id: int = 0,
    flip_method: int = 0,
) -> str:
    """
    Build a GStreamer pipeline string for Jetson MIPI CSI-2 cameras.

    Uses ``nvarguscamerasrc`` + ``nvvidconv`` for zero-copy HW decoding.
    Pass the result to ``CameraConfig(gstreamer_pipeline=…)``.

    Parameters
    ----------
    flip_method : 0=none, 1=ccw-90°, 2=180°, 3=cw-90°, 4=h-flip, 6=v-flip.

    Returns
    -------
    GStreamer pipeline string accepted by ``cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)``.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,width={width},height={height},format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=2"
    )


def build_usb_gstreamer_pipeline(
    device_id: int = 0,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
) -> str:
    """
    Build a GStreamer pipeline for a USB camera on Jetson (V4L2 source).

    Useful when OpenCV's default V4L2 backend lacks HW acceleration.
    """
    return (
        f"v4l2src device=/dev/video{device_id} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=2"
    )


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------


def _resize(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)


def _to_tensor(rgb: np.ndarray) -> "torch.Tensor":
    """Convert ``(H, W, 3)`` uint8 RGB → ``(1, 3, H, W)`` float32 tensor."""
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    t = torch.from_numpy(arr.transpose(2, 0, 1))   # CHW
    return t.unsqueeze(0).contiguous()              # 1CHW


# ---------------------------------------------------------------------------
# Main stream class
# ---------------------------------------------------------------------------


class CameraStream:
    """
    Thread-safe camera frame source for the visual tracking pipeline.

    Usage
    -----
    ::

        cfg = CameraConfig(source=CameraSource.USB, device_id=0,
                           target_size=(224, 224), normalize=True)

        with CameraStream(cfg) as stream:
            for frame in stream.frames(max_frames=500):
                anomaly_scores = model(frame.tensor.cuda())

    Parameters
    ----------
    config : CameraConfig or None
        Stream configuration; uses defaults (USB, 1280×720, 30 fps) if None.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self._cfg = config or CameraConfig()
        self._cap: object = None          # any backend with .read() / .release()
        self._buffer: deque[CameraFrame] = deque(
            maxlen=max(self._cfg.buffer_size, 1)
        )
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_count = 0
        self._drop_count = 0
        self._lock = threading.Lock()
        self._fps_window: deque[float] = deque(maxlen=30)
        self._t_open: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open the capture backend and start the background reader thread.

        Raises
        ------
        CameraOpenError
            When the requested backend cannot be initialised.
        RuntimeError
            When called on an already-open stream.
        """
        if self._running:
            raise RuntimeError("CameraStream is already open.")

        if not _CV2_AVAILABLE and self._cfg.source not in (
            CameraSource.MOCK, CameraSource.REALSENSE
        ):
            raise CameraOpenError(
                "opencv-python is not installed.  "
                "Install with: pip install opencv-python-headless"
            )

        self._cap = self._open_capture()
        self._stop.clear()
        self._running = True
        self._t_open = time.perf_counter()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="CameraStream-reader",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "CameraStream opened | source=%s | device=%s",
            self._cfg.source.value,
            self._cfg.device_id
            if self._cfg.source != CameraSource.VIDEO_FILE
            else self._cfg.video_path,
        )

    def close(self) -> None:
        """Stop the reader thread and release the capture backend."""
        if not self._running:
            return
        self._stop.set()
        with self._cond:
            self._cond.notify_all()   # unblock any waiting consumers
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            try:
                self._cap.release()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._running = False
        log.info(
            "CameraStream closed | frames=%d | drops=%d",
            self._frame_count, self._drop_count,
        )

    def is_open(self) -> bool:
        """Return True while the reader thread is running."""
        return self._running and not self._stop.is_set()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> CameraFrame | None:
        """
        Non-blocking read of the latest available frame.

        Returns
        -------
        The most recent :class:`CameraFrame`, or None when the buffer is empty.
        """
        with self._cond:
            return self._buffer[-1] if self._buffer else None

    def read_blocking(self, timeout: float = 1.0) -> CameraFrame | None:
        """
        Block until a new frame is available or *timeout* seconds elapse.

        Parameters
        ----------
        timeout : float
            Maximum wait in seconds.

        Returns
        -------
        :class:`CameraFrame` or None on timeout / stream closed.
        """
        deadline = time.perf_counter() + timeout
        with self._cond:
            while not self._buffer and self._running:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return self._buffer[-1] if self._buffer else None

    def frames(
        self,
        max_frames: int | None = None,
        timeout: float = 2.0,
    ) -> Generator[CameraFrame, None, None]:
        """
        Generator that yields frames until the stream closes or *max_frames*
        is reached.

        Parameters
        ----------
        max_frames : int or None
            Stop after this many frames.  None = unlimited.
        timeout : float
            Per-frame blocking timeout.

        Yields
        ------
        CameraFrame
        """
        yielded = 0
        while self.is_open():
            if max_frames is not None and yielded >= max_frames:
                break
            frame = self.read_blocking(timeout=timeout)
            if frame is None:
                break
            yield frame
            yielded += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def actual_fps(self) -> float:
        """Measured capture rate computed over the last 30 frames."""
        with self._lock:
            ts = list(self._fps_window)
        if len(ts) < 2:
            return 0.0
        return len(ts) / max(ts[-1] - ts[0], 1e-6)

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def drop_count(self) -> int:
        with self._lock:
            return self._drop_count

    @property
    def stats(self) -> StreamStats:
        """Return a :class:`StreamStats` snapshot."""
        with self._lock:
            fc = self._frame_count
            dc = self._drop_count
        with self._cond:
            bd = len(self._buffer)
        return StreamStats(
            frame_count=fc,
            drop_count=dc,
            actual_fps=round(self.actual_fps, 2),
            buffer_depth=bd,
            is_running=self._running,
            uptime_s=round(time.perf_counter() - self._t_open, 2)
            if self._t_open > 0
            else 0.0,
        )

    # ------------------------------------------------------------------
    # Context manager + iterator
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __iter__(self) -> Iterator[CameraFrame]:
        return self.frames()

    # ------------------------------------------------------------------
    # Internal — capture backend factory
    # ------------------------------------------------------------------

    def _open_capture(self) -> object:
        cfg = self._cfg

        if cfg.source == CameraSource.MOCK:
            return _MockCapture(cfg.width, cfg.height, cfg.fps)

        if cfg.source == CameraSource.REALSENSE:
            return _RealSenseCapture(cfg.width, cfg.height, cfg.fps)

        # All remaining sources use OpenCV.
        cap = self._open_cv2_capture(cfg)
        if not cap.isOpened():
            raise CameraOpenError(
                f"Failed to open {cfg.source.value} source "
                f"(device={cfg.device_id}, pipeline={cfg.gstreamer_pipeline})"
            )

        # Apply resolution / FPS hints (USB / video-file).
        if cfg.source in (CameraSource.USB, CameraSource.VIDEO_FILE):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            # Reduce internal OpenCV buffer to minimise capture latency.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    @staticmethod
    def _open_cv2_capture(cfg: CameraConfig) -> "cv2.VideoCapture":
        if cfg.source == CameraSource.VIDEO_FILE:
            if cfg.video_path is None:
                raise CameraOpenError(
                    "source=VIDEO_FILE requires video_path to be set."
                )
            return cv2.VideoCapture(str(cfg.video_path))

        if cfg.source == CameraSource.CSI:
            pipeline = cfg.gstreamer_pipeline or build_csi_pipeline(
                cfg.width, cfg.height, cfg.fps,
                sensor_id=cfg.device_id,
                flip_method=cfg.flip_method,
            )
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if cfg.source == CameraSource.USB:
            if cfg.gstreamer_pipeline:
                return cv2.VideoCapture(
                    cfg.gstreamer_pipeline, cv2.CAP_GSTREAMER
                )
            return cv2.VideoCapture(cfg.device_id, cv2.CAP_V4L2)

        raise CameraOpenError(f"Unhandled source: {cfg.source}")

    # ------------------------------------------------------------------
    # Internal — reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: read → preprocess → push to ring buffer."""
        consecutive_failures = 0

        while not self._stop.is_set():
            ret, raw = self._cap.read()  # type: ignore[attr-defined]

            if not ret or raw is None:
                # ── Video-file end ──────────────────────────────────────
                if self._cfg.source == CameraSource.VIDEO_FILE:
                    if self._cfg.loop_video:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[attr-defined]
                        consecutive_failures = 0
                        continue
                    else:
                        log.info("Video file ended; closing stream.")
                        break

                # ── Transient read failure — attempt reconnect ──────────
                consecutive_failures += 1
                log.warning(
                    "Frame read failed (%d / %d); retrying …",
                    consecutive_failures, _MAX_RECONNECT,
                )
                if consecutive_failures >= _MAX_RECONNECT:
                    log.error(
                        "Exhausted %d reconnect attempts; stopping reader.",
                        _MAX_RECONNECT,
                    )
                    break
                time.sleep(_RECONNECT_DELAY_S)
                continue

            consecutive_failures = 0
            t_capture = time.perf_counter()

            with self._lock:
                self._frame_count += 1
                idx = self._frame_count
                self._fps_window.append(t_capture)

            frame = self._preprocess(raw, t_capture, idx)

            with self._cond:
                # deque(maxlen=N) silently evicts the oldest entry when full —
                # count that as a drop.
                was_full = len(self._buffer) == self._buffer.maxlen
                self._buffer.append(frame)
                if was_full:
                    with self._lock:
                        self._drop_count += 1
                self._cond.notify_all()

        # Signal close to any blocked consumers.
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._running = False

    # ------------------------------------------------------------------
    # Internal — preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        raw: np.ndarray,
        t_capture: float,
        idx: int,
    ) -> CameraFrame:
        """Convert a raw BGR frame into a :class:`CameraFrame`."""
        cfg = self._cfg

        # Resize
        if cfg.target_size is not None:
            raw = _resize(raw, cfg.target_size)

        # Colour conversion
        if cfg.convert_rgb:
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            image = raw

        # Tensor
        tensor = None
        if cfg.normalize and _TORCH_AVAILABLE and cfg.convert_rgb:
            tensor = _to_tensor(image)

        h, w = image.shape[:2]
        latency_ms = (time.perf_counter() - t_capture) * 1_000.0

        return CameraFrame(
            image=image,
            tensor=tensor,
            timestamp=t_capture,
            frame_idx=idx,
            source=self._cfg.source.value,
            width=w,
            height=h,
            latency_ms=round(latency_ms, 3),
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


def open_usb(
    device_id: int = 0,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> CameraStream:
    """
    Open a USB webcam and return a started :class:`CameraStream`.

    The caller is responsible for calling ``stream.close()`` or using it as a
    context manager.
    """
    cfg = CameraConfig(
        source=CameraSource.USB,
        device_id=device_id,
        width=width,
        height=height,
        fps=fps,
        target_size=target_size,
        normalize=normalize,
    )
    stream = CameraStream(cfg)
    stream.open()
    return stream


def open_csi(
    sensor_id: int = 0,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    fps: int = _DEFAULT_FPS,
    flip_method: int = 0,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> CameraStream:
    """
    Open a Jetson CSI camera and return a started :class:`CameraStream`.

    Builds the nvarguscamerasrc GStreamer pipeline automatically.
    """
    cfg = CameraConfig(
        source=CameraSource.CSI,
        device_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        flip_method=flip_method,
        target_size=target_size,
        normalize=normalize,
    )
    stream = CameraStream(cfg)
    stream.open()
    return stream


def open_video(
    path: str | Path,
    *,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
    loop: bool = False,
) -> CameraStream:
    """
    Open a pre-recorded video file and return a started :class:`CameraStream`.

    Useful for offline development and CI pipelines without a physical camera.
    """
    cfg = CameraConfig(
        source=CameraSource.VIDEO_FILE,
        video_path=str(path),
        target_size=target_size,
        normalize=normalize,
        loop_video=loop,
    )
    stream = CameraStream(cfg)
    stream.open()
    return stream


def open_mock(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> CameraStream:
    """
    Open a synthetic mock source — no hardware or files required.

    Ideal for unit testing and CI environments.
    """
    cfg = CameraConfig(
        source=CameraSource.MOCK,
        width=width,
        height=height,
        fps=fps,
        target_size=target_size,
        normalize=normalize,
    )
    stream = CameraStream(cfg)
    stream.open()
    return stream
