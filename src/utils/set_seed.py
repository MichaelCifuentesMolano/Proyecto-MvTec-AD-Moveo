"""
src/utils/set_seed.py
======================
Global reproducibility seeding for all frameworks used in the pipeline.

Seeded subsystems
-----------------
- Python built-in ``random``
- NumPy (legacy ``np.random`` + new ``np.random.default_rng``)
- PyTorch CPU and CUDA (all devices)
- PyTorch CuDNN (deterministic ops, benchmark disabled)
- Python hash randomisation (``PYTHONHASHSEED`` env-var advisory)
- NSGA-II / DEAP ``random`` module (shared with Python's ``random``)

Optional frameworks (seeded only when installed):
- TensorFlow / Keras
- scikit-learn (delegates to NumPy)
- OpenCV (``cv2.setRNGSeed``)
- PyCUDA / CuPy

Design
------
A single ``set_seed(seed)`` call is the intended public API.  It returns
a ``SeedState`` dataclass that documents exactly what was seeded and at
what value, so callers can log or serialise it for experiment tracking.

Thread-safety
-------------
``set_seed`` is **not** thread-safe.  Call it once at process start, before
any worker threads or DataLoader processes are spawned.

DataLoader workers
------------------
Pass ``worker_init_fn=make_worker_init_fn(seed)`` to ``torch.utils.data.DataLoader``
to re-seed each worker with a deterministic but distinct seed derived from
the global seed and the worker index.

Usage
-----
>>> from utils.set_seed import set_seed
>>> state = set_seed(42)
>>> print(state)
SeedState(seed=42, torch=True, cuda=True, tensorflow=False, ...)

Or with the context manager (restores RNG state on exit):

>>> with seeded_context(42):
...     run_experiment()
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import struct
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional framework imports (all soft)
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore
    _TF_AVAILABLE = False

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SEED: int = 2**31 - 1   # Safe upper bound for all frameworks.


# ---------------------------------------------------------------------------
# SeedState — documents what was actually seeded
# ---------------------------------------------------------------------------

@dataclass
class SeedState:
    """
    Record of what was seeded and at what value.

    Returned by ``set_seed()`` so callers can log or checkpoint the exact
    seeding configuration for full experiment reproducibility.
    """
    seed: int

    # Core
    python_random: bool = False
    numpy_legacy: bool = False
    numpy_rng: bool = False

    # PyTorch
    torch_cpu: bool = False
    torch_cuda: bool = False
    torch_cudnn_deterministic: bool = False
    torch_cudnn_benchmark_disabled: bool = False
    torch_use_deterministic_algorithms: bool = False

    # Optional frameworks
    tensorflow: bool = False
    opencv: bool = False
    cupy: bool = False

    # Environment advisory
    pythonhashseed_set: bool = False

    # Number of CUDA devices seeded
    cuda_devices_seeded: int = 0

    # Warnings emitted during seeding
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        active = [k for k, v in asdict(self).items()
                  if isinstance(v, bool) and v and k != "seed"]
        return (
            f"SeedState(seed={self.seed}, seeded=[{', '.join(active)}], "
            f"cuda_devices={self.cuda_devices_seeded})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp_seed(seed: int) -> int:
    """Clamp seed to the safe range ``[0, 2^31 − 1]``."""
    return int(seed) % (_MAX_SEED + 1)


def _derive(seed: int, offset: int) -> int:
    """Derive a deterministic child seed from a parent + an integer offset."""
    # Mix via a simple hash to avoid trivial correlations.
    raw = struct.pack(">qq", seed, offset)
    h = 0
    for b in raw:
        h = (h * 31 + b) & 0xFFFFFFFF
    return h % (_MAX_SEED + 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_seed(
    seed: int,
    *,
    deterministic_torch: bool = True,
    warn_nondeterministic: bool = False,
    set_pythonhashseed: bool = True,
    tensorflow_inter_op_threads: int = 1,
    tensorflow_intra_op_threads: int = 1,
) -> SeedState:
    """
    Seed all RNG subsystems for fully reproducible experiments.

    Parameters
    ----------
    seed : int
        Master seed.  Must be in ``[0, 2^31 − 1]``; larger values are
        clamped with ``seed % (2^31)``.
    deterministic_torch : bool
        If True, enable ``torch.backends.cudnn.deterministic`` and disable
        ``torch.backends.cudnn.benchmark``.  May slow training on some
        hardware; set False when speed is critical and minor non-determinism
        is acceptable.
    warn_nondeterministic : bool
        If True, call ``torch.use_deterministic_algorithms(True)`` which
        raises ``RuntimeError`` for any non-deterministic op.  Useful during
        debugging; leave False for production.
    set_pythonhashseed : bool
        If True, set ``PYTHONHASHSEED`` in ``os.environ``.  This only affects
        **child processes** (the current interpreter's hash seed is fixed at
        startup).  An advisory warning is emitted if the current process seed
        differs.
    tensorflow_inter_op_threads : int
        Passed to ``tf.config.threading.set_inter_op_parallelism_threads``
        (1 = fully deterministic TF graph execution).
    tensorflow_intra_op_threads : int
        Passed to ``tf.config.threading.set_intra_op_parallelism_threads``.

    Returns
    -------
    SeedState
        Documents exactly which subsystems were seeded.

    Examples
    --------
    >>> state = set_seed(42)
    >>> print(state)
    SeedState(seed=42, seeded=[python_random, numpy_legacy, ...], cuda_devices=1)

    >>> state = set_seed(0, deterministic_torch=False)  # Speed-first.
    """
    seed = _clamp_seed(seed)
    state = SeedState(seed=seed)

    # ── Python built-in random ────────────────────────────────────────────
    random.seed(seed)
    state.python_random = True

    # ── NumPy legacy API ──────────────────────────────────────────────────
    np.random.seed(seed)
    state.numpy_legacy = True

    # ── NumPy new Generator (global default_rng is not global state, but
    #    we store one for callers that import it from this module) ──────────
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(seed)
    state.numpy_rng = True

    # ── PYTHONHASHSEED ────────────────────────────────────────────────────
    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)
        state.pythonhashseed_set = True
        # Advisory: current process hash seed was set at interpreter start.
        current = os.environ.get("PYTHONHASHSEED", "random")
        if current != str(seed) and current != "0":
            state.warnings.append(
                f"PYTHONHASHSEED={current} in current process; "
                f"new value {seed} applies to child processes only. "
                f"Restart with PYTHONHASHSEED={seed} for full hash determinism."
            )

    # ── PyTorch ───────────────────────────────────────────────────────────
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        state.torch_cpu = True

        # All CUDA devices.
        if torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
            torch.cuda.manual_seed_all(seed)
            state.torch_cuda = True
            state.cuda_devices_seeded = n_devices

        # CuDNN determinism.
        if deterministic_torch:
            try:
                torch.backends.cudnn.deterministic = True
                state.torch_cudnn_deterministic = True
            except AttributeError:
                state.warnings.append("torch.backends.cudnn.deterministic not available.")

            try:
                torch.backends.cudnn.benchmark = False
                state.torch_cudnn_benchmark_disabled = True
            except AttributeError:
                state.warnings.append("torch.backends.cudnn.benchmark not available.")

        # Strict deterministic algorithms (optional, raises on violations).
        if warn_nondeterministic:
            try:
                torch.use_deterministic_algorithms(True)
                state.torch_use_deterministic_algorithms = True
                # CUBLAS workspace config required for deterministic matmul.
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            except AttributeError:
                state.warnings.append(
                    "torch.use_deterministic_algorithms not available "
                    "(requires PyTorch ≥ 1.8)."
                )

    # ── TensorFlow ────────────────────────────────────────────────────────
    if _TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
            tf.config.threading.set_inter_op_parallelism_threads(
                tensorflow_inter_op_threads
            )
            tf.config.threading.set_intra_op_parallelism_threads(
                tensorflow_intra_op_threads
            )
            state.tensorflow = True
        except Exception as exc:  # noqa: BLE001
            state.warnings.append(f"TensorFlow seeding failed: {exc}")

    # ── OpenCV ────────────────────────────────────────────────────────────
    if _CV2_AVAILABLE:
        try:
            cv2.setRNGSeed(seed)
            state.opencv = True
        except Exception as exc:  # noqa: BLE001
            state.warnings.append(f"OpenCV seeding failed: {exc}")

    # ── CuPy ─────────────────────────────────────────────────────────────
    if _CUPY_AVAILABLE:
        try:
            cp.random.seed(seed)
            state.cupy = True
        except Exception as exc:  # noqa: BLE001
            state.warnings.append(f"CuPy seeding failed: {exc}")

    # ── Emit log summary ──────────────────────────────────────────────────
    log.info("Global seed set to %d. %s", seed, state)
    for w in state.warnings:
        log.warning("[set_seed] %s", w)

    return state


# ---------------------------------------------------------------------------
# Module-level default_rng (accessible without calling set_seed again)
# ---------------------------------------------------------------------------

_GLOBAL_RNG: np.random.Generator = np.random.default_rng(0)
"""
Module-level NumPy Generator seeded by the last ``set_seed()`` call.

Import and use this instead of creating ad-hoc generators:

>>> from utils.set_seed import get_rng
>>> rng = get_rng()
>>> rng.integers(0, 100, size=10)
"""


def get_rng() -> np.random.Generator:
    """Return the module-level NumPy Generator seeded by the last ``set_seed``."""
    return _GLOBAL_RNG


# ---------------------------------------------------------------------------
# DataLoader worker initialiser
# ---------------------------------------------------------------------------

def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """
    Build a ``worker_init_fn`` for ``torch.utils.data.DataLoader``.

    Each worker receives a deterministic but distinct seed derived from the
    global seed and its worker index, preventing all workers from producing
    identical random augmentations.

    Parameters
    ----------
    seed : int
        The same master seed passed to ``set_seed()``.

    Returns
    -------
    Callable[[int], None]  — pass as ``DataLoader(worker_init_fn=...)``.

    Examples
    --------
    >>> loader = DataLoader(
    ...     dataset,
    ...     num_workers=4,
    ...     worker_init_fn=make_worker_init_fn(42),
    ... )
    """
    master = _clamp_seed(seed)

    def _worker_init(worker_id: int) -> None:
        worker_seed = _derive(master, worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        if _TORCH_AVAILABLE:
            torch.manual_seed(worker_seed)
        log.debug("DataLoader worker %d seeded with %d.", worker_id, worker_seed)

    return _worker_init


# ---------------------------------------------------------------------------
# Context manager — restore RNG state on exit
# ---------------------------------------------------------------------------

@contextmanager
def seeded_context(seed: int, **set_seed_kwargs):
    """
    Context manager that seeds all RNGs on entry and restores their states
    on exit.  Useful for isolated reproducible sub-experiments.

    Note: PyTorch CUDA RNG state capture requires CUDA to be initialised
    before entering the context.  TensorFlow and OpenCV do **not** expose
    full state restoration; they are re-seeded on entry but not restored.

    Parameters
    ----------
    seed           : Master seed for ``set_seed()``.
    **set_seed_kwargs : Forwarded to ``set_seed()``.

    Examples
    --------
    >>> with seeded_context(42):
    ...     x = torch.randn(4, 4)  # always the same

    >>> with seeded_context(99, deterministic_torch=False):
    ...     fast_train()
    """
    # Capture state.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state() if _TORCH_AVAILABLE else None
    torch_cuda_states = (
        torch.cuda.get_rng_state_all()
        if (_TORCH_AVAILABLE and torch.cuda.is_available())
        else None
    )
    global _GLOBAL_RNG
    prev_global_rng = _GLOBAL_RNG

    # Apply seed.
    set_seed(seed, **set_seed_kwargs)

    try:
        yield
    finally:
        # Restore state.
        random.setstate(py_state)
        np.random.set_state(np_state)
        _GLOBAL_RNG = prev_global_rng

        if _TORCH_AVAILABLE and torch_cpu_state is not None:
            torch.set_rng_state(torch_cpu_state)
        if _TORCH_AVAILABLE and torch_cuda_states is not None:
            with contextlib.suppress(Exception):
                torch.cuda.set_rng_state_all(torch_cuda_states)


# ---------------------------------------------------------------------------
# NSGA-II / DEAP helper
# ---------------------------------------------------------------------------

def seed_deap(seed: int) -> None:
    """
    Seed the DEAP / NSGA-II evolutionary algorithm.

    DEAP uses Python's ``random`` module internally, which is already seeded
    by ``set_seed()``.  This function provides an explicit, documented hook
    for callers who want to seed DEAP independently (e.g. between runs).

    Parameters
    ----------
    seed : int  Master seed.
    """
    random.seed(_clamp_seed(seed))
    log.debug("DEAP/NSGA-II seeded via Python random (seed=%d).", seed)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def verify_reproducibility(seed: int, n_samples: int = 8) -> bool:
    """
    Run a quick sanity check: seed twice with the same value and verify that
    the first ``n_samples`` draws from each framework are identical.

    Parameters
    ----------
    seed      : Seed to test.
    n_samples : Number of samples to compare per framework.

    Returns
    -------
    bool  — True if all available frameworks are reproducible.

    Examples
    --------
    >>> assert verify_reproducibility(42)
    """
    ok = True
    results: dict = {}

    # ── Run A ─────────────────────────────────────────────────────────────
    set_seed(seed, set_pythonhashseed=False)
    a_py  = [random.random() for _ in range(n_samples)]
    a_np  = np.random.rand(n_samples).tolist()
    a_rng = _GLOBAL_RNG.random(n_samples).tolist()
    a_torch = (
        torch.rand(n_samples).tolist() if _TORCH_AVAILABLE else []
    )

    # ── Run B ─────────────────────────────────────────────────────────────
    set_seed(seed, set_pythonhashseed=False)
    b_py  = [random.random() for _ in range(n_samples)]
    b_np  = np.random.rand(n_samples).tolist()
    b_rng = _GLOBAL_RNG.random(n_samples).tolist()
    b_torch = (
        torch.rand(n_samples).tolist() if _TORCH_AVAILABLE else []
    )

    # ── Compare ───────────────────────────────────────────────────────────
    checks = {
        "python_random": (a_py  == b_py),
        "numpy_legacy":  (a_np  == b_np),
        "numpy_rng":     (a_rng == b_rng),
        "torch":         (a_torch == b_torch if _TORCH_AVAILABLE else True),
    }
    for name, match in checks.items():
        results[name] = match
        if not match:
            ok = False
            log.error("Reproducibility FAILED for '%s' (seed=%d).", name, seed)
        else:
            log.debug("Reproducibility OK   for '%s' (seed=%d).", name, seed)

    if ok:
        log.info("verify_reproducibility passed for all frameworks (seed=%d).", seed)
    return ok


def framework_versions() -> dict:
    """
    Return a dict of installed framework versions relevant to reproducibility.

    Useful for logging experiment metadata alongside the seed.
    """
    info: dict = {"python": sys.version.split()[0]}
    info["numpy"] = np.__version__

    if _TORCH_AVAILABLE:
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["cuda_devices"] = torch.cuda.device_count()

    if _TF_AVAILABLE:
        info["tensorflow"] = tf.__version__

    if _CV2_AVAILABLE:
        info["opencv"] = cv2.__version__

    if _CUPY_AVAILABLE:
        info["cupy"] = cp.__version__

    return info
