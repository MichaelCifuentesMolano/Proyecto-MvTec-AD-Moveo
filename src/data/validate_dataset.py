"""
src/data/validate_dataset.py
============================

Strict integrity validator for the extracted MVTec AD dataset.

Tasks
-----
1. Verify missing files / directories per the canonical MVTec layout.
2. Verify image integrity (header parse + decode), flagging corrupt files.
3. Count samples per class/category (train good, test good, test defect,
   ground-truth masks).
4. Generate a detailed integrity report and persist it as JSON.

Public interface (consumed by ``main_prepare.py``)
--------------------------------------------------
``validate(raw_dir: Path,
           expected_categories: list[str] | None = None,
           *,
           report_path: Path | None = None,
           check_masks: bool = True,
           verify_images: bool = True,
           max_workers: int = 8) -> dict``

    Returns the integrity report dict, with at least::

        {
            "valid":              bool,
            "categories":         list[str],
            "n_train":            int,
            "n_test":             int,
            "n_masks":            int,
            "issues":             list[str],
            ...
        }

Output artifact
---------------
``results/dataset_validation.json`` (under the project root, inferred from
``raw_dir`` unless ``report_path`` is provided explicitly).

Canonical MVTec AD layout (per category)
----------------------------------------
::

    mvtec_ad/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/<image_stem>_mask.<ext>

Assumptions
-----------
- ``raw_dir`` points at ``data/raw`` (or wherever ``extract_dataset`` was
  asked to place the categories). The validator looks for the categories
  directly under ``raw_dir`` first, and transparently descends into a
  single ``mvtec_ad/`` subfolder if present.
- A ``test/<defect>/`` sample with no corresponding mask under
  ``ground_truth/<defect>/`` is reported but does not by itself flip
  ``valid`` to ``False`` — only structural absences (missing
  ``train/good`` or empty ``test``) are fatal. Mask coverage is reported
  via the issue ``mask_coverage_below_1.0`` per category.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image  # type: ignore
    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    _HAVE_PIL = False

__all__ = ["validate", "MVTEC_CATEGORIES", "IMAGE_EXTS", "MASK_EXTS"]

LOG = logging.getLogger(__name__)

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
)

IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
})
MASK_EXTS: frozenset[str] = frozenset({
    ".png", ".bmp", ".tif", ".tiff",
})

GOOD_LABEL: str = "good"
TRAIN_DIR: str = "train"
TEST_DIR: str = "test"
MASK_DIR: str = "ground_truth"
MASK_SUFFIX: str = "_mask"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate(raw_dir: Path,
             expected_categories: Iterable[str] | None = None,
             *,
             report_path: Path | None = None,
             check_masks: bool = True,
             verify_images: bool = True,
             max_workers: int = 8,
             **_unused) -> dict:
    """Validate the integrity of an extracted MVTec AD dataset.

    See module docstring for the exact return schema. The report is also
    written to ``report_path`` (or the default ``<project>/results/
    dataset_validation.json`` if not provided).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    expected = tuple(c.lower() for c in (expected_categories
                                          or MVTEC_CATEGORIES))

    dataset_root = _resolve_dataset_root(raw_dir)
    LOG.info("Validating dataset rooted at %s", dataset_root)

    if not _HAVE_PIL and verify_images:
        LOG.warning("PIL/Pillow not available — disabling image-decode checks.")
        verify_images = False

    t0 = time.perf_counter()
    found_categories = _list_categories(dataset_root)
    per_category: dict[str, dict] = {}

    for category in found_categories:
        per_category[category] = _validate_category(
            category_root=dataset_root / category,
            check_masks=check_masks,
            verify_images=verify_images,
            max_workers=max_workers,
        )

    report = _aggregate(
        dataset_root=dataset_root,
        expected=expected,
        found_categories=found_categories,
        per_category=per_category,
    )
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    out_path = report_path or _default_report_path(raw_dir)
    _write_json_atomic(out_path, report)
    report["report_path"] = str(out_path)
    LOG.info(
        "Validation %s — %d cat(s), %d train, %d test, %d masks, "
        "%d corrupt; report -> %s",
        "OK" if report["valid"] else "FAILED",
        len(report["categories"]),
        report["n_train"], report["n_test"], report["n_masks"],
        report.get("n_corrupt", 0), out_path,
    )
    return report


# ---------------------------------------------------------------------------
# Per-category validation
# ---------------------------------------------------------------------------
def _validate_category(category_root: Path,
                       *,
                       check_masks: bool,
                       verify_images: bool,
                       max_workers: int) -> dict:
    """Run all integrity checks for a single category folder."""
    issues: list[str] = []
    missing_paths: list[str] = []

    train_good = category_root / TRAIN_DIR / GOOD_LABEL
    test_root = category_root / TEST_DIR
    masks_root = category_root / MASK_DIR

    # ---- structural checks ----
    if not train_good.is_dir():
        missing_paths.append(_rel(train_good, category_root))
        issues.append("missing_train_good")
    if not test_root.is_dir():
        missing_paths.append(_rel(test_root, category_root))
        issues.append("missing_test_dir")

    # ---- enumerate samples ----
    train_good_files = _list_images(train_good)
    test_subdirs = (sorted(p for p in test_root.iterdir()
                           if p.is_dir())
                    if test_root.is_dir() else [])
    defect_types = [p.name for p in test_subdirs if p.name != GOOD_LABEL]

    test_good_files: list[Path] = []
    test_defect_files: dict[str, list[Path]] = {}
    for sub in test_subdirs:
        files = _list_images(sub)
        if sub.name == GOOD_LABEL:
            test_good_files = files
        else:
            test_defect_files[sub.name] = files

    if test_root.is_dir() and not (test_good_files or test_defect_files):
        issues.append("empty_test_dir")

    # ---- mask coverage ----
    masks_per_defect: dict[str, list[Path]] = {}
    missing_masks: list[str] = []
    if check_masks and defect_types:
        if not masks_root.is_dir():
            missing_paths.append(_rel(masks_root, category_root))
            issues.append("missing_ground_truth_dir")
        for defect in defect_types:
            mask_dir = masks_root / defect if masks_root.is_dir() else None
            if mask_dir is None or not mask_dir.is_dir():
                missing_paths.append(
                    _rel(masks_root / defect, category_root)
                )
                masks_per_defect[defect] = []
                continue
            masks_per_defect[defect] = _list_images(mask_dir, MASK_EXTS)
            for img in test_defect_files.get(defect, []):
                if _expected_mask_for(img, mask_dir) is None:
                    missing_masks.append(_rel(img, category_root))
        n_defect_imgs = sum(len(v) for v in test_defect_files.values())
        n_defect_masks = sum(len(v) for v in masks_per_defect.values())
        if n_defect_imgs > 0:
            coverage = (n_defect_imgs - len(missing_masks)) / n_defect_imgs
            if coverage < 1.0:
                issues.append(
                    f"mask_coverage_below_1.0[{coverage:.3f}]"
                )

    # ---- image integrity ----
    all_images: list[Path] = list(train_good_files) + list(test_good_files)
    for files in test_defect_files.values():
        all_images.extend(files)
    for files in masks_per_defect.values():
        all_images.extend(files)

    corrupt_files: list[str] = []
    if verify_images and all_images:
        corrupt = _verify_images_concurrent(
            all_images, max_workers=max_workers,
        )
        corrupt_files = [_rel(p, category_root) for p in corrupt]
        if corrupt_files:
            issues.append(f"corrupt_images={len(corrupt_files)}")

    n_train = len(train_good_files)
    n_test = len(test_good_files) + sum(
        len(v) for v in test_defect_files.values()
    )
    n_masks = sum(len(v) for v in masks_per_defect.values())

    return {
        "category_root": str(category_root),
        "n_train_good": n_train,
        "n_test_good": len(test_good_files),
        "n_test_defect": sum(
            len(v) for v in test_defect_files.values()
        ),
        "n_test": n_test,
        "n_masks": n_masks,
        "defect_types": defect_types,
        "n_per_defect": {k: len(v) for k, v in test_defect_files.items()},
        "missing_paths": missing_paths,
        "missing_masks": missing_masks,
        "corrupt_files": corrupt_files,
        "issues": issues,
        "valid": not any(
            i.startswith(("missing_train_good", "missing_test_dir",
                          "empty_test_dir"))
            or i.startswith("corrupt_images")
            for i in issues
        ),
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _resolve_dataset_root(raw_dir: Path) -> Path:
    """Detect whether categories sit at ``raw_dir`` or under ``mvtec_ad/``."""
    nested = raw_dir / "mvtec_ad"
    if any((raw_dir / c).is_dir() for c in MVTEC_CATEGORIES):
        return raw_dir
    if nested.is_dir() and any((nested / c).is_dir()
                                for c in MVTEC_CATEGORIES):
        return nested
    # Fall back to whichever exists; downstream check will catch issues.
    return nested if nested.is_dir() else raw_dir


def _list_categories(dataset_root: Path) -> list[str]:
    return sorted(
        p.name for p in dataset_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _list_images(directory: Path,
                 exts: frozenset[str] = IMAGE_EXTS) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _expected_mask_for(image_path: Path,
                       mask_dir: Path) -> Path | None:
    """Return the mask path matching ``image_path`` if it exists."""
    stem = image_path.stem
    for ext in MASK_EXTS:
        candidate = mask_dir / f"{stem}{MASK_SUFFIX}{ext}"
        if candidate.is_file():
            return candidate
        # Some MVTec mirrors omit the ``_mask`` suffix; tolerate it.
        candidate_alt = mask_dir / f"{stem}{ext}"
        if candidate_alt.is_file():
            return candidate_alt
    return None


def _rel(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` as a POSIX string."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Image verification
# ---------------------------------------------------------------------------
def _verify_images_concurrent(paths: list[Path],
                              *,
                              max_workers: int) -> list[Path]:
    """Return the subset of ``paths`` that fail to decode."""
    corrupt: list[Path] = []
    if not paths or not _HAVE_PIL:
        return corrupt
    workers = max(1, min(max_workers, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(_verify_image, p): p for p in paths}
        for fut in as_completed(future_to_path):
            ok, err = fut.result()
            if not ok:
                bad = future_to_path[fut]
                corrupt.append(bad)
                LOG.debug("Corrupt image: %s (%s)", bad, err)
    return corrupt


def _verify_image(path: Path) -> tuple[bool, str | None]:
    """Open + verify + decode a single image. Returns ``(ok, error)``."""
    try:
        # verify() catches header-level corruption; load() forces full decode
        # and is required to detect truncated payloads.
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Aggregation + persistence
# ---------------------------------------------------------------------------
def _aggregate(*,
               dataset_root: Path,
               expected: tuple[str, ...],
               found_categories: list[str],
               per_category: dict[str, dict]) -> dict:
    """Combine per-category records into the top-level integrity report."""
    expected_set = set(expected)
    found_set = set(found_categories)
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)

    n_train = sum(c["n_train_good"] for c in per_category.values())
    n_test = sum(c["n_test"] for c in per_category.values())
    n_masks = sum(c["n_masks"] for c in per_category.values())
    n_corrupt = sum(len(c["corrupt_files"]) for c in per_category.values())
    n_missing_masks = sum(
        len(c["missing_masks"]) for c in per_category.values()
    )

    issues: list[str] = []
    if missing:
        issues.append(f"missing_categories={missing}")
    if extra:
        issues.append(f"unexpected_categories={extra}")
    if n_corrupt:
        issues.append(f"corrupt_images_total={n_corrupt}")
    if n_missing_masks:
        issues.append(f"missing_masks_total={n_missing_masks}")
    for cat, rec in per_category.items():
        for issue in rec["issues"]:
            issues.append(f"{cat}:{issue}")

    fatal = (
        bool(missing) or
        any(not rec["valid"] for rec in per_category.values()) or
        n_corrupt > 0
    )
    return {
        "valid": not fatal,
        "raw_dir": str(dataset_root),
        "expected_categories": list(expected),
        "categories": found_categories,
        "missing_categories": missing,
        "unexpected_categories": extra,
        "n_train": n_train,
        "n_test": n_test,
        "n_masks": n_masks,
        "n_corrupt": n_corrupt,
        "n_missing_masks": n_missing_masks,
        "per_category": per_category,
        "issues": issues,
    }


def _default_report_path(raw_dir: Path) -> Path:
    """Compute ``<project>/results/dataset_validation.json`` from ``raw_dir``.

    Convention: ``raw_dir`` is ``<project>/data/raw`` so the project root
    is two parents up. Falls back to the parent of ``raw_dir`` if that
    layout doesn't apply.
    """
    candidates = [raw_dir.parent.parent, raw_dir.parent, raw_dir]
    for root in candidates:
        if root.is_dir():
            return root / "results" / "dataset_validation.json"
    return Path("results") / "dataset_validation.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)
