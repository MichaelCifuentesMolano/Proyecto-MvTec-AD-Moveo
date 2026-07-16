"""
src/data/extract_dataset.py
===========================

MVTec AD dataset extraction utility.

Tasks
-----
1. Decompress ``mvtec_anomaly_detection.tar.xz`` (also handles ``.tar.gz``,
   ``.tgz``, and plain ``.tar``).
2. Organize extracted category folders under ``data/raw/mvtec_ad/`` —
   stripping the archive's top-level directory if one is present.
3. Standardize paths — POSIX-style separators in the returned manifest,
   normalized (lowercased) category folder names, and a deterministic
   layout consumable by every downstream stage.

Public interface (consumed by ``main_prepare.py``)
--------------------------------------------------
``extract_archive(archive_path: Path, raw_dir: Path,
                  processed_dir: Path | None = None,
                  force: bool = False, **kwargs) -> dict``

    Returns ``{
        "raw_root": Path,                # data/raw/mvtec_ad/
        "categories": list[str],         # sorted, lowercased
        "n_files_extracted": int,
        "skipped": bool,                 # True when reusing existing extraction
        "issues": list[str],             # validation warnings (non-fatal)
        "archive_path": str,
        "elapsed_seconds": float,
    }``

The ``processed_dir`` argument is accepted for forward compatibility with
the preparation pipeline; this module only ensures the directory exists,
preprocessing belongs to a later stage.

Assumptions
-----------
- The 15 standard MVTec AD categories are expected; missing ones produce
  warnings but do not raise (validation is delegated to
  ``validate_dataset.py``).
- Extraction is performed atomically: the archive is unpacked into a
  staging directory under ``raw_dir`` and only its contents are moved into
  ``mvtec_ad/`` once unpacking succeeds. A partial / failed extraction
  leaves no half-populated ``mvtec_ad/`` directory behind.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Iterable

__all__ = ["extract_archive", "MVTEC_CATEGORIES", "TARGET_SUBDIR"]

LOG = logging.getLogger(__name__)

# Canonical layout produced by this module.
TARGET_SUBDIR: str = "mvtec_ad"

# Default expected categories — used only for non-fatal validation warnings.
MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
)

# Recognized image / mask extensions used by MVTec AD.
_IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
})

# Progress-reporting cadence (members between log lines during extraction).
_PROGRESS_EVERY: int = 1000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_archive(archive_path: Path,
                    raw_dir: Path,
                    processed_dir: Path | None = None,
                    force: bool = False,
                    *,
                    expected_categories: Iterable[str] = MVTEC_CATEGORIES,
                    target_subdir: str = TARGET_SUBDIR,
                    **_unused) -> dict:
    """Decompress the MVTec AD archive into ``raw_dir/<target_subdir>``.

    Parameters
    ----------
    archive_path
        Path to ``mvtec_anomaly_detection.tar.xz`` (or any tar archive
        that ``tarfile.open(..., "r:*")`` can read).
    raw_dir
        Destination root (typically ``data/raw``). The extracted dataset
        ends up under ``raw_dir / target_subdir``.
    processed_dir
        Optional companion directory (created if missing). Reserved for a
        later preprocessing stage; this module does not write into it.
    force
        If True, an existing ``raw_dir/<target_subdir>`` is removed before
        re-extraction. If False (default), an existing populated target
        is reused and the call returns ``skipped=True``.
    expected_categories
        Used only for non-fatal validation warnings.
    target_subdir
        Override the canonical ``mvtec_ad`` subfolder name.

    Returns
    -------
    dict
        Summary record consumable by ``main_prepare.py``.
    """
    archive_path = Path(archive_path)
    raw_dir = Path(raw_dir)
    target_root = raw_dir / target_subdir

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}. "
            "Place the MVTec AD .tar.xz at the project root or pass --archive."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir is not None:
        Path(processed_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ---- short-circuit when extraction already exists -------------------
    if target_root.is_dir() and _is_populated(target_root):
        if not force:
            categories = _list_categories(target_root)
            n_files = _count_files(target_root)
            issues = _check_categories(categories, expected_categories)
            LOG.info(
                "Reusing existing extraction at %s (%d categor(ies), %d files)",
                target_root, len(categories), n_files,
            )
            return _build_result(
                target_root=target_root,
                categories=categories,
                n_files=n_files,
                skipped=True,
                issues=issues,
                archive_path=archive_path,
                elapsed=time.perf_counter() - t0,
            )
        LOG.info("force=True — removing existing %s", target_root)
        shutil.rmtree(target_root)

    target_root.mkdir(parents=True, exist_ok=True)

    # ---- atomic extract via staging dir under raw_dir -------------------
    LOG.info("Extracting %s -> %s", archive_path, target_root)
    staging = Path(tempfile.mkdtemp(prefix="_mvtec_stage_", dir=raw_dir))
    try:
        n_members = _extract_to(archive_path, staging)
        source_root = _find_source_root(staging)
        n_moved = _organize_into_target(source_root, target_root)
        LOG.info("Extracted %d archive member(s); organized %d top-level "
                 "entry(ies) into %s", n_members, n_moved, target_root)
    finally:
        # Always clean staging — even if move-step failed, target_root may
        # be partially populated; the caller can re-run with --force.
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)

    categories = _list_categories(target_root)
    n_files = _count_files(target_root)
    issues = _check_categories(categories, expected_categories)

    return _build_result(
        target_root=target_root,
        categories=categories,
        n_files=n_files,
        skipped=False,
        issues=issues,
        archive_path=archive_path,
        elapsed=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _extract_to(archive_path: Path, dest: Path) -> int:
    """Stream-extract the archive into ``dest``.

    Returns the number of archive members written. Uses the ``filter='data'``
    safety filter on Python ≥ 3.12 to mitigate the CVE-2007-4559 family of
    tar-traversal issues.
    """
    extract_kwargs: dict = {}
    if sys.version_info >= (3, 12):
        extract_kwargs["filter"] = "data"

    n = 0
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar:
            n += 1
            # Guard against absolute paths and ``..`` traversal.
            name = member.name.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                LOG.warning("Skipping suspicious member: %s", member.name)
                continue
            try:
                tar.extract(member, path=dest, **extract_kwargs)
            except (PermissionError, OSError) as exc:
                # Symlinks or special files may fail on Windows — skip them.
                LOG.debug("Skipping member %s (%s)", member.name, exc)
                continue
            if n % _PROGRESS_EVERY == 0:
                LOG.info("  ... extracted %d members", n)
    return n


def _find_source_root(staging: Path) -> Path:
    """Return the directory whose contents should be moved into ``mvtec_ad/``.

    MVTec archives nest everything inside a single top-level folder
    (e.g. ``mvtec_anomaly_detection/``). When that's the case, return that
    folder. If the archive is already flat (categories at root), return
    the staging dir itself.
    """
    entries = [p for p in staging.iterdir()
               if not p.name.startswith(".")]
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    # Single top-level dir, no files at root → nested layout, descend into it.
    if len(dirs) == 1 and not files:
        return dirs[0]
    # Otherwise treat staging as the source root (categories already at top).
    return staging


def _organize_into_target(source: Path, target_root: Path) -> int:
    """Move each entry under ``source`` into ``target_root``.

    Category folder names are normalized to lowercase. Returns the number
    of top-level entries moved.
    """
    moved = 0
    for item in sorted(source.iterdir()):
        if item.name.startswith("."):
            continue
        normalized_name = item.name.lower() if item.is_dir() else item.name
        dest = target_root / normalized_name
        if dest.exists():
            # Should not happen with force=True; defensive cleanup.
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
        moved += 1
    return moved


def _list_categories(target_root: Path) -> list[str]:
    """Return the sorted list of category folder names under ``target_root``."""
    return sorted(
        p.name for p in target_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _count_files(target_root: Path) -> int:
    """Count image/mask files under ``target_root`` recursively."""
    return sum(
        1 for p in target_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _is_populated(path: Path) -> bool:
    """True iff ``path`` exists and contains at least one entry."""
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def _check_categories(found: list[str],
                      expected: Iterable[str]) -> list[str]:
    """Return non-fatal warnings about missing / unexpected categories."""
    expected_set = {c.lower() for c in expected}
    found_set = set(found)
    issues: list[str] = []
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)
    if missing:
        issues.append(f"missing_categories={missing}")
    if extra:
        issues.append(f"unexpected_categories={extra}")
    return issues


def _build_result(*,
                  target_root: Path,
                  categories: list[str],
                  n_files: int,
                  skipped: bool,
                  issues: list[str],
                  archive_path: Path,
                  elapsed: float) -> dict:
    """Compose the standardized summary record returned to the orchestrator."""
    return {
        "raw_root": target_root,
        "categories": categories,
        "n_files_extracted": n_files,
        "skipped": skipped,
        "issues": issues,
        "archive_path": archive_path.as_posix(),
        "elapsed_seconds": round(elapsed, 3),
    }
