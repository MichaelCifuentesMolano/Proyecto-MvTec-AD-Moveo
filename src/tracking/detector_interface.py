"""
src/tracking/detector_interface.py

Live anomaly-detection inference interface for the visual tracking pipeline.

Responsibilities
----------------
* Accept a raw camera frame (:class:`~tracking.camera_stream.CameraFrame`),
  a pre-normalized tensor, or a NumPy image and produce a fully processed
  :class:`DetectionResult` in a single ``detect()`` call.
* Support three inference backends interchangeably:
    TRT  — TensorRT engine (primary; Jetson deployment).
    Torch — plain PyTorch ``nn.Module`` (development / FP16 fallback).
    ONNX  — ONNXRuntime (cross-platform validation).
* Post-process the raw model output into:
    score          — image-level anomaly probability in [0, 1].
    heatmap        — (H, W) float32 spatial anomaly map, normalised to [0, 1].
    heatmap_overlay — BGR image with the heatmap blended over the camera frame.
    rois           — list of :class:`ROIAlert` bounding-boxes extracted from the
                     thresholded heatmap via connected-component analysis.
* Manage score thresholds: fixed config values, percentile calibration from
  normal frames, or loaded from a training-output JSON file.

Assumed model output schema
----------------------------
The detector expects the model's ``forward()`` to return a dict (or the TRT /
ONNX engine to expose named outputs) with any subset of:

    "anomaly_map"  — (1, 1, H, W) or (1, H, W) spatial reconstruction-error map.
    "score"        — (1,) or (1, 1) image-level score (higher = more anomalous).
    "recon"        — (1, C, H, W) reconstructed input (used to derive MSE map).

When a key is absent, the interface falls back gracefully (score derived from
heatmap max; heatmap tiled from scalar score, etc.).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .camera_stream import CameraFrame

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
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter as _scipy_gauss
    from scipy.ndimage import label as _scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_gauss = None  # type: ignore[assignment]
    _scipy_label = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_EPS = 1e-6

# Key priority order when inspecting model outputs.
_HEATMAP_KEYS = ("anomaly_map", "recon_error", "score_map", "output")
_SCORE_KEYS   = ("score", "logits")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """
    Tunable parameters for the anomaly detector interface.

    Parameters
    ----------
    score_threshold : float
        Image-level anomaly decision boundary in [0, 1].  Frames with
        ``score >= score_threshold`` are flagged as anomalous.
    roi_threshold : float
        Heatmap value (normalised) above which a pixel is considered part of an
        anomalous region.  Should be ≤ ``score_threshold``.
    roi_min_area_frac : float
        Minimum connected-component area expressed as a fraction of the total
        frame area.  Tiny noise blobs below this size are suppressed.
    max_rois : int
        Maximum number of ROI alerts returned per frame (ranked by confidence).
    severity_thresholds : (low, high) floats
        Peak-score boundaries for severity classification:
        < low → "low";  low ≤ x < high → "medium";  ≥ high → "high".
    score_min, score_max : float
        Linear normalisation bounds applied to the raw model score before
        comparing against ``score_threshold``.
    heatmap_blur_sigma : float
        Standard deviation of the Gaussian smoothing kernel applied to the
        upsampled heatmap.  0 disables smoothing.
    overlay_alpha : float
        Blend weight of the colourised heatmap over the camera frame (0 = no
        overlay, 1 = full heatmap).
    colormap : int
        OpenCV colourmap ID.  Default ``cv2.COLORMAP_JET`` (2).
    upsample_mode : str
        Interpolation used when upsampling the heatmap to the frame
        resolution.  "bilinear" (default) or "nearest".
    device_id : int
        CUDA device index used by the TRT and Torch backends.
    half_precision : bool
        Run the Torch backend in FP16.
    """

    score_threshold: float = 0.5
    roi_threshold: float = 0.40
    roi_min_area_frac: float = 0.005
    max_rois: int = 10
    severity_thresholds: tuple[float, float] = (0.40, 0.70)
    score_min: float = 0.0
    score_max: float = 1.0
    heatmap_blur_sigma: float = 2.0
    overlay_alpha: float = 0.45
    colormap: int = 2          # cv2.COLORMAP_JET
    upsample_mode: str = "bilinear"
    device_id: int = 0
    half_precision: bool = False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ROIAlert:
    """
    One detected anomalous region extracted from the heatmap.

    Attributes
    ----------
    bbox : (x1, y1, x2, y2) pixel coordinates in the original frame.
    confidence : float — mean normalised heatmap value inside the region.
    peak_score : float — maximum normalised heatmap value inside the region.
    area_px : int — number of pixels belonging to the region.
    center : (cx, cy) centroid of the region.
    severity : "low" | "medium" | "high" — classified from peak_score.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    peak_score: float
    area_px: int
    center: tuple[int, int]
    severity: str


@dataclass
class DetectionResult:
    """
    Complete output produced by one call to :meth:`DetectorInterface.detect`.

    Attributes
    ----------
    score : float
        Normalised image-level anomaly probability in [0, 1].
    is_anomaly : bool
        ``score >= config.score_threshold``.
    heatmap : np.ndarray
        ``(H, W)`` float32 spatial anomaly map, normalised to [0, 1],
        registered to the original camera frame.
    heatmap_overlay : np.ndarray or None
        ``(H, W, 3)`` uint8 BGR image: heatmap colourised and blended over
        the camera frame.  None when OpenCV is unavailable.
    rois : list[ROIAlert]
        Detected anomalous regions, sorted by confidence descending.
    timestamp : float
        Capture timestamp of the source frame (``time.perf_counter()``).
    frame_idx : int
        Monotonic frame index from the camera stream.
    inference_ms : float
        Backend inference time in milliseconds.
    postproc_ms : float
        Heatmap post-processing + ROI extraction time in milliseconds.
    total_ms : float
        Combined ``inference_ms + postproc_ms``.
    """

    score: float
    is_anomaly: bool
    heatmap: np.ndarray
    heatmap_overlay: np.ndarray | None
    rois: list[ROIAlert]
    timestamp: float
    frame_idx: int
    inference_ms: float
    postproc_ms: float
    total_ms: float

    def as_dict(self) -> dict:
        """JSON-serialisable summary (arrays excluded)."""
        return {
            "score": self.score,
            "is_anomaly": self.is_anomaly,
            "rois": [asdict(r) for r in self.rois],
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "inference_ms": self.inference_ms,
            "postproc_ms": self.postproc_ms,
            "total_ms": self.total_ms,
        }


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------


class _TRTBackend:
    """TensorRT ICudaEngine inference with pre-allocated CUDA buffers."""

    def __init__(self, engine_path: str | Path, device_id: int = 0) -> None:
        from .export_tensorrt import (          # type: ignore[import]
            load_engine,
            _get_engine_io_names,
            _set_input_shape_ctx,
            _get_output_shape_ctx,
            _execute_ctx,
        )
        self._load_engine = load_engine
        self._set_shape = _set_input_shape_ctx
        self._get_shape = _get_output_shape_ctx
        self._execute = _execute_ctx
        self._get_names = _get_engine_io_names

        self._engine = load_engine(Path(engine_path), device_id=device_id)
        self._ctx = self._engine.create_execution_context()
        self._device_id = device_id

        inp_names, out_names = _get_engine_io_names(self._engine)
        self._input_name: str = inp_names[0]
        self._output_names: list[str] = out_names

        # Buffers allocated on first call (shape-aware).
        self._input_buf: "torch.Tensor | None" = None
        self._output_bufs: "dict[str, torch.Tensor] | None" = None
        self._last_shape: tuple = ()

    @property
    def name(self) -> str:
        return "trt"

    def _setup_buffers(self, shape: tuple) -> None:
        dev = f"cuda:{self._device_id}"
        self._input_buf = torch.empty(*shape, dtype=torch.float32, device=dev)
        self._set_shape(self._ctx, self._input_name, shape)
        self._output_bufs = {
            n: torch.empty(
                self._get_shape(self._ctx, n),
                dtype=torch.float32, device=dev,
            )
            for n in self._output_names
        }
        self._last_shape = shape

    def infer(self, tensor: "torch.Tensor") -> "dict[str, torch.Tensor]":
        tensor = tensor.float().cuda(self._device_id).contiguous()
        if tensor.shape != self._last_shape:
            self._setup_buffers(tuple(tensor.shape))
        self._input_buf.copy_(tensor)  # type: ignore[union-attr]
        stream = torch.cuda.current_stream(self._device_id)
        self._execute(
            self._ctx,
            {self._input_name: self._input_buf.data_ptr()},  # type: ignore[union-attr]
            {n: b.data_ptr() for n, b in self._output_bufs.items()},  # type: ignore[union-attr]
            stream.cuda_stream,
        )
        torch.cuda.synchronize(self._device_id)
        return {n: b.clone().cpu() for n, b in self._output_bufs.items()}  # type: ignore[union-attr]


class _TorchBackend:
    """Plain PyTorch nn.Module inference."""

    def __init__(
        self,
        model: "torch.nn.Module",
        device: str | "torch.device" = "cpu",
        half: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._half = half
        self._model = model.eval().to(self._device)
        if half:
            self._model = self._model.half()

    @property
    def name(self) -> str:
        return "torch"

    def infer(self, tensor: "torch.Tensor") -> "dict[str, torch.Tensor]":
        tensor = tensor.to(self._device)
        if self._half:
            tensor = tensor.half()
        with torch.no_grad():
            out = self._model(tensor)
        if isinstance(out, torch.Tensor):
            return {"output": out.float().cpu()}
        return {k: v.float().cpu() for k, v in out.items() if v is not None}


class _ONNXBackend:
    """ONNXRuntime inference (CPU or CUDA execution provider)."""

    def __init__(self, onnx_path: str | Path, device_id: int = 0) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime is not installed.")
        providers = (
            [("CUDAExecutionProvider", {"device_id": device_id}),
             "CPUExecutionProvider"]
            if torch is not None and torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_names: list[str] = [
            o.name for o in self._session.get_outputs()
        ]

    @property
    def name(self) -> str:
        return "onnx"

    def infer(self, tensor: "torch.Tensor") -> "dict[str, torch.Tensor]":
        np_in = tensor.cpu().numpy().astype(np.float32)
        outs = self._session.run(None, {self._input_name: np_in})
        return {
            n: torch.from_numpy(o)
            for n, o in zip(self._output_names, outs)
        }


# ---------------------------------------------------------------------------
# Pure post-processing helpers
# ---------------------------------------------------------------------------


def _image_to_tensor(rgb: np.ndarray) -> "torch.Tensor":
    """Convert ``(H, W, 3)`` uint8 RGB → ``(1, 3, H, W)`` float32 tensor."""
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _tensor_to_rgb(tensor: "torch.Tensor") -> np.ndarray:
    """Reverse ImageNet normalisation; return ``(H, W, 3)`` uint8 RGB."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def _extract_outputs(
    outputs: "dict[str, torch.Tensor]",
) -> tuple[float, np.ndarray]:
    """
    Extract ``(raw_score, raw_heatmap)`` from inference output dict.

    Priority for heatmap: anomaly_map → recon_error → first spatial output.
    Priority for score:   score/logits → max(anomaly_map) → max(output).

    Returns ``raw_heatmap`` as a 2-D ``(H, W)`` float32 NumPy array
    (any scale — normalisation happens downstream).
    """
    def _squeeze_2d(t: "torch.Tensor") -> np.ndarray:
        t = t.float()
        while t.dim() > 2:
            t = t.squeeze(0)
        return t.cpu().numpy()

    # ── Heatmap ──────────────────────────────────────────────────────────
    heatmap: np.ndarray | None = None
    for key in _HEATMAP_KEYS:
        if key in outputs:
            heatmap = _squeeze_2d(outputs[key])
            break
    if heatmap is None:
        # Fall back to first output that is spatial.
        for t in outputs.values():
            if t.dim() >= 3:
                heatmap = _squeeze_2d(t)
                break
    if heatmap is None:
        heatmap = np.zeros((7, 7), dtype=np.float32)

    # ── Score ─────────────────────────────────────────────────────────────
    raw_score: float = 0.0
    for key in _SCORE_KEYS:
        if key in outputs:
            raw_score = float(outputs[key].float().mean())
            break
    else:
        # Derive from heatmap: use 99th-percentile for robustness to outliers.
        raw_score = float(np.percentile(heatmap, 99))

    return raw_score, heatmap.astype(np.float32)


def _upsample_heatmap(
    heatmap: np.ndarray,
    target_hw: tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """Resize ``(H_m, W_m)`` → ``(H, W)`` heatmap using PyTorch or OpenCV."""
    h, w = heatmap.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return heatmap

    if _TORCH_AVAILABLE:
        t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
        align = mode == "bilinear"
        t = F.interpolate(
            t, size=(th, tw), mode=mode,
            align_corners=align if mode == "bilinear" else None,
        )
        return t.squeeze().numpy()

    if _CV2_AVAILABLE:
        interp = cv2.INTER_LINEAR if mode == "bilinear" else cv2.INTER_NEAREST
        return cv2.resize(heatmap, (tw, th), interpolation=interp)

    # Pure NumPy nearest-neighbour fallback.
    y_idx = (np.arange(th) * h / th).astype(int)
    x_idx = (np.arange(tw) * w / tw).astype(int)
    return heatmap[np.ix_(y_idx, x_idx)]


def _smooth_heatmap(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing using scipy or OpenCV."""
    if sigma <= 0:
        return heatmap
    if _SCIPY_AVAILABLE:
        return _scipy_gauss(heatmap, sigma=sigma).astype(np.float32)
    if _CV2_AVAILABLE:
        k = int(sigma * 6) | 1          # make kernel size odd
        return cv2.GaussianBlur(heatmap, (k, k), sigmaX=sigma).astype(np.float32)
    return heatmap


def _normalise_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns uniform 0.5 on flat input."""
    lo, hi = float(heatmap.min()), float(heatmap.max())
    if hi - lo < _EPS:
        return np.full_like(heatmap, 0.0)
    return ((heatmap - lo) / (hi - lo)).astype(np.float32)


def _normalise_score(
    raw: float,
    score_min: float,
    score_max: float,
) -> float:
    rng = score_max - score_min
    if rng < _EPS:
        return 0.0
    return float(np.clip((raw - score_min) / rng, 0.0, 1.0))


def _make_overlay(
    rgb: np.ndarray,
    heatmap_norm: np.ndarray,
    alpha: float,
    colormap: int = 2,
) -> np.ndarray | None:
    """
    Blend the colourised heatmap over the RGB camera frame.

    Returns a ``(H, W, 3)`` uint8 BGR image, or None when OpenCV is absent.
    """
    if not _CV2_AVAILABLE:
        return None
    hm_uint8 = (heatmap_norm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(hm_uint8, colormap)          # BGR
    bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr_frame, 1.0 - alpha, coloured, alpha, 0)


def _classify_severity(
    peak: float,
    thresholds: tuple[float, float],
) -> str:
    lo, hi = thresholds
    if peak >= hi:
        return "high"
    if peak >= lo:
        return "medium"
    return "low"


def _extract_rois(
    heatmap: np.ndarray,
    threshold: float,
    min_area_frac: float,
    max_rois: int,
    severity_thresholds: tuple[float, float],
) -> list[ROIAlert]:
    """
    Extract anomalous ROIs via connected-component analysis on the heatmap.

    Preferred backend: OpenCV ``connectedComponentsWithStats``.
    Fallback:          SciPy ``ndimage.label``.
    Last resort:       single whole-frame ROI when the heatmap is anomalous.

    Parameters
    ----------
    heatmap    : ``(H, W)`` float32, normalised to [0, 1].
    threshold  : pixel-level detection threshold.
    min_area_frac : minimum component area as fraction of total frame area.
    max_rois   : cap on returned ROIs.
    severity_thresholds : (low, high) for classification.

    Returns
    -------
    List of :class:`ROIAlert` sorted by confidence descending.
    """
    h, w = heatmap.shape
    min_area_px = max(1, int(min_area_frac * h * w))
    binary = (heatmap >= threshold).astype(np.uint8)

    if not binary.any():
        return []

    rois: list[ROIAlert] = []

    # ── OpenCV path ───────────────────────────────────────────────────────
    if _CV2_AVAILABLE:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            x1 = int(stats[i, cv2.CC_STAT_LEFT])
            y1 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            x2, y2 = x1 + bw, y1 + bh
            mask = labels[y1:y2, x1:x2] == i
            vals = heatmap[y1:y2, x1:x2][mask]
            mean_v = float(vals.mean())
            peak_v = float(vals.max())
            cx = int(round(centroids[i][0]))
            cy = int(round(centroids[i][1]))
            rois.append(ROIAlert(
                bbox=(x1, y1, x2, y2),
                confidence=round(mean_v, 4),
                peak_score=round(peak_v, 4),
                area_px=area,
                center=(cx, cy),
                severity=_classify_severity(peak_v, severity_thresholds),
            ))

    # ── SciPy fallback ────────────────────────────────────────────────────
    elif _SCIPY_AVAILABLE:
        labeled, n_comp = _scipy_label(binary)
        for i in range(1, n_comp + 1):
            mask = labeled == i
            area = int(mask.sum())
            if area < min_area_px:
                continue
            rows, cols = np.where(mask)
            y1, y2 = int(rows.min()), int(rows.max())
            x1, x2 = int(cols.min()), int(cols.max())
            vals = heatmap[mask]
            mean_v, peak_v = float(vals.mean()), float(vals.max())
            rois.append(ROIAlert(
                bbox=(x1, y1, x2 + 1, y2 + 1),
                confidence=round(mean_v, 4),
                peak_score=round(peak_v, 4),
                area_px=area,
                center=(int(cols.mean()), int(rows.mean())),
                severity=_classify_severity(peak_v, severity_thresholds),
            ))

    # ── Whole-frame fallback ──────────────────────────────────────────────
    else:
        peak_v = float(heatmap.max())
        mean_v = float(heatmap[binary.astype(bool)].mean())
        rois.append(ROIAlert(
            bbox=(0, 0, w, h),
            confidence=round(mean_v, 4),
            peak_score=round(peak_v, 4),
            area_px=int(binary.sum()),
            center=(w // 2, h // 2),
            severity=_classify_severity(peak_v, severity_thresholds),
        ))

    rois.sort(key=lambda r: r.confidence, reverse=True)
    return rois[:max_rois]


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------


class DetectorInterface:
    """
    Unified anomaly-detection inference interface.

    Accepts :class:`~tracking.camera_stream.CameraFrame`, a ``torch.Tensor``,
    or a NumPy array and returns a :class:`DetectionResult` with score,
    normalised heatmap, colourised overlay, and ROI alerts.

    Construct via the class-method factories rather than ``__init__`` directly:

    ::

        # TensorRT (primary for Jetson)
        det = DetectorInterface.from_trt_engine("model.trt")

        # PyTorch (development)
        det = DetectorInterface.from_torch_model(model, device="cuda")

        # ONNX (cross-platform)
        det = DetectorInterface.from_onnx("model.onnx")

    Parameters
    ----------
    backend : one of the internal ``_*Backend`` objects.
    config  : :class:`DetectorConfig`; defaults used if None.
    """

    def __init__(
        self,
        backend: _TRTBackend | _TorchBackend | _ONNXBackend,
        config: DetectorConfig | None = None,
    ) -> None:
        self._backend = backend
        self._cfg = config or DetectorConfig()

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_trt_engine(
        cls,
        engine_path: str | Path,
        config: DetectorConfig | None = None,
    ) -> "DetectorInterface":
        """
        Build a detector backed by a TensorRT engine file.

        Parameters
        ----------
        engine_path : Path to a ``.trt`` engine compiled by ``export_tensorrt``.
        config      : Detection configuration.
        """
        cfg = config or DetectorConfig()
        backend = _TRTBackend(engine_path, device_id=cfg.device_id)
        log.info("TRT detector ready | engine=%s", Path(engine_path).name)
        return cls(backend, cfg)

    @classmethod
    def from_torch_model(
        cls,
        model: "torch.nn.Module",
        config: DetectorConfig | None = None,
        *,
        device: str | "torch.device" = "cpu",
    ) -> "DetectorInterface":
        """
        Build a detector backed by a PyTorch ``nn.Module``.

        Parameters
        ----------
        model  : Trained model (QAT-aware or plain) in any device state.
        config : Detection configuration.
        device : Target inference device (e.g. "cpu", "cuda:0").
        """
        cfg = config or DetectorConfig()
        backend = _TorchBackend(model, device=device, half=cfg.half_precision)
        log.info("Torch detector ready | device=%s", device)
        return cls(backend, cfg)

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        config: DetectorConfig | None = None,
    ) -> "DetectorInterface":
        """
        Build a detector backed by ONNXRuntime.

        Parameters
        ----------
        onnx_path : Path to a ``.onnx`` model exported by ``export_onnx``.
        config    : Detection configuration.
        """
        cfg = config or DetectorConfig()
        backend = _ONNXBackend(onnx_path, device_id=cfg.device_id)
        log.info("ONNX detector ready | model=%s", Path(onnx_path).name)
        return cls(backend, cfg)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: "CameraFrame | torch.Tensor | np.ndarray",
        *,
        frame_size: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """
        Run the full detection pipeline on one frame.

        Parameters
        ----------
        frame : One of:

            * :class:`~tracking.camera_stream.CameraFrame` — primary type;
              provides both ``image`` (for the overlay) and ``tensor``
              (for inference).
            * ``torch.Tensor`` of shape ``(1, 3, H, W)`` — assumed
              ImageNet-normalised; pass ``frame_size=(H, W)`` for a
              correctly sized overlay.
            * ``np.ndarray`` of shape ``(H, W, 3)`` uint8 — interpreted
              as BGR (OpenCV convention); converted internally to RGB and
              normalised.

        frame_size : ``(H, W)`` override used when *frame* is a tensor and
            the heatmap must be upsampled to a specific resolution.

        Returns
        -------
        DetectionResult
        """
        tensor, rgb, ts, fidx = self._prepare_input(frame, frame_size)

        # ── Inference ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        outputs = self._backend.infer(tensor)
        inference_ms = (time.perf_counter() - t0) * 1_000.0

        # ── Post-processing ───────────────────────────────────────────────
        t1 = time.perf_counter()
        raw_score, raw_hm = _extract_outputs(outputs)

        H, W = rgb.shape[:2]
        hm = _upsample_heatmap(raw_hm, (H, W), self._cfg.upsample_mode)
        hm = _smooth_heatmap(hm, self._cfg.heatmap_blur_sigma)
        hm_norm = _normalise_heatmap(hm)

        score = _normalise_score(
            raw_score, self._cfg.score_min, self._cfg.score_max
        )
        is_anomaly = score >= self._cfg.score_threshold

        rois = _extract_rois(
            hm_norm,
            self._cfg.roi_threshold,
            self._cfg.roi_min_area_frac,
            self._cfg.max_rois,
            self._cfg.severity_thresholds,
        )

        overlay = _make_overlay(
            rgb, hm_norm, self._cfg.overlay_alpha, self._cfg.colormap
        )

        postproc_ms = (time.perf_counter() - t1) * 1_000.0

        return DetectionResult(
            score=round(score, 5),
            is_anomaly=is_anomaly,
            heatmap=hm_norm,
            heatmap_overlay=overlay,
            rois=rois,
            timestamp=ts,
            frame_idx=fidx,
            inference_ms=round(inference_ms, 3),
            postproc_ms=round(postproc_ms, 3),
            total_ms=round(inference_ms + postproc_ms, 3),
        )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        frames: "list[CameraFrame | torch.Tensor | np.ndarray]",
        *,
        percentile: float = 95.0,
        update_normalization: bool = True,
    ) -> float:
        """
        Set the score threshold from a calibration set of *normal* frames.

        Runs inference on every frame, collects the score distribution, and
        sets ``config.score_threshold`` to the given percentile.

        Parameters
        ----------
        frames      : List of normal (non-anomalous) frames.
        percentile  : Score percentile used as the decision boundary.
        update_normalization : When True, also updates ``score_min`` /
            ``score_max`` from the observed distribution.

        Returns
        -------
        The computed threshold value.
        """
        if not frames:
            raise ValueError("frames must not be empty.")

        scores = [self.detect(f).score for f in frames]
        arr = np.array(scores, dtype=np.float64)
        threshold = float(np.percentile(arr, percentile))

        if update_normalization:
            self._cfg.score_min = float(arr.min())
            self._cfg.score_max = float(arr.max()) + 0.2 * float(arr.std() + _EPS)

        self._cfg.score_threshold = threshold
        log.info(
            "Threshold calibrated | n=%d | p%.0f=%.4f | min=%.4f | max=%.4f",
            len(scores), percentile, threshold,
            self._cfg.score_min, self._cfg.score_max,
        )
        return threshold

    def load_threshold(
        self,
        path: str | Path,
        *,
        category: str | None = None,
    ) -> None:
        """
        Load threshold settings from a JSON file produced during training.

        The JSON may be a flat dict or a category-keyed dict::

            {"bottle": {"score_threshold": 0.47, "score_min": 0.0, "score_max": 0.91}}

        Parameters
        ----------
        path     : JSON file path.
        category : Optional key to look up within a category-keyed file.
        """
        data: dict = json.loads(Path(path).read_text())
        if category and category in data:
            data = data[category]
        for key in ("score_threshold", "score_min", "score_max",
                    "roi_threshold", "roi_min_area_frac"):
            if key in data:
                setattr(self._cfg, key, float(data[key]))
        log.info(
            "Threshold loaded | threshold=%.4f | min=%.4f | max=%.4f",
            self._cfg.score_threshold, self._cfg.score_min, self._cfg.score_max,
        )

    def save_threshold(self, path: str | Path, *, category: str | None = None) -> None:
        """
        Persist current threshold settings to a JSON file.

        Parameters
        ----------
        path     : Destination JSON file path (created atomically).
        category : When provided, the settings are nested under this key.
        """
        payload = {
            "score_threshold": self._cfg.score_threshold,
            "score_min": self._cfg.score_min,
            "score_max": self._cfg.score_max,
            "roi_threshold": self._cfg.roi_threshold,
            "roi_min_area_frac": self._cfg.roi_min_area_frac,
        }
        out: dict = {category: payload} if category else payload
        p = Path(path)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(p)
        log.info("Threshold saved → %s", p)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Name of the active inference backend: "trt", "torch", or "onnx"."""
        return self._backend.name

    @property
    def config(self) -> DetectorConfig:
        """Live reference to the detector configuration (mutable)."""
        return self._cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        frame: "CameraFrame | torch.Tensor | np.ndarray",
        frame_size: tuple[int, int] | None,
    ) -> "tuple[torch.Tensor, np.ndarray, float, int]":
        """
        Normalise the input frame into ``(tensor, rgb_image, timestamp, idx)``.

        Returns
        -------
        tensor    : ``(1, 3, H, W)`` float32 ImageNet-normalised.
        rgb_image : ``(H, W, 3)`` uint8 RGB for overlay generation.
        timestamp : capture time (``time.perf_counter()`` if unavailable).
        frame_idx : monotonic frame index (-1 if unavailable).
        """
        from .camera_stream import CameraFrame  # local import to avoid cycle

        ts = time.perf_counter()
        fidx = -1

        if isinstance(frame, CameraFrame):
            ts = frame.timestamp
            fidx = frame.frame_idx
            rgb = frame.image    # already RGB (camera_stream converts)
            tensor = frame.tensor if frame.tensor is not None else _image_to_tensor(rgb)
            return tensor, rgb, ts, fidx

        if _TORCH_AVAILABLE and isinstance(frame, torch.Tensor):
            tensor = frame.float()
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            # De-normalise for display if frame_size provides context.
            if frame_size is not None:
                rgb = _tensor_to_rgb(tensor)
                rgb = _upsample_heatmap(rgb.astype(np.float32), frame_size)
                rgb = rgb.astype(np.uint8)
            else:
                rgb = _tensor_to_rgb(tensor)
            return tensor, rgb, ts, fidx

        if isinstance(frame, np.ndarray):
            # Accept either BGR (H,W,3) or RGB; if BGR convert.
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if _CV2_AVAILABLE:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame[..., ::-1].copy()    # assume BGR
            tensor = _image_to_tensor(rgb)
            return tensor, rgb, ts, fidx

        raise TypeError(
            f"Unsupported frame type: {type(frame).__name__}.  "
            "Expected CameraFrame, torch.Tensor, or np.ndarray."
        )
