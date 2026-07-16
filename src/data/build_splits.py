"""
src/data/build_splits.py
========================

Deterministic train / validation / test split builder for the MVTec AD
dataset. Writes one JSON manifest per category plus cross-category
experimental bundles under ``data/splits/``.

Tasks
-----
1. Build train / validation / test splits per category. The validation
   set is carved out of ``train/good`` (the only labelled-as-normal pool
   available before training); test is taken verbatim from the official
   ``test/`` and ``ground_truth/`` directories.
2. Preserve the official anomaly test data — every image under
   ``test/good`` and ``test/<defect>`` and every mask under
   ``ground_truth/<defect>`` is kept intact and lands in the test split
   without resampling, so reported AUROC remains comparable to the
   published MVTec AD benchmark.
3. Cross-category experimental splits — produce two extra bundles:
   * ``_cross_category/all_categories.json`` — pooled split combining
     train/val/test from every category, stratified by category.
   * ``_cross_category/leave_one_out_<category>.json`` — one bundle per
     category where train+val pools all *other* categories and test is
     the held-out category's official test partition.

Public interface (consumed by ``main_prepare.py``)
--------------------------------------------------
``build_splits(raw_dir: Path, splits_dir: Path, val_ratio: float,
               seed: int, stratify_by_defect: bool = True,
               *, write_cross_category: bool = True) -> dict``

    Returns::

        {
            "splits_dir": Path,
            "categories": list[str],
            "summary": {
                "<category>": {
                    "train": int, "val": int,
                    "test_good": int, "test_defect": int, "test": int,
                    "manifest": str,                 # relative path
                },
                ...,
                "_cross_category": {
                    "all_categories": {...counts..., "manifest": "..."},
                    "leave_one_out": {
                        "<category>": {...counts..., "manifest": "..."},
                        ...
                    },
                },
            },
            "n_categories": int,
            "elapsed_seconds": float,
        }

Manifest schema (per-category)
------------------------------
::

    {
        "category": "bottle",
        "raw_root": "<absolute path>",
        "seed": 42,
        "val_ratio": 0.15,
        "stratify_by_defect": true,
        "splits": {
            "train": [SampleRecord, ...],
            "val":   [SampleRecord, ...],
            "test":  [SampleRecord, ...]
        },
        "counts": {...}
    }

``SampleRecord`` fields::

    {
        "path":          "test/good/000.png",     # POSIX, relative to raw_root
        "abs_path":      "<absolute>",
        "label":         0 | 1,                   # 0 = normal, 1 = anomaly
        "defect":        "good" | "<defect_name>",
        "mask":          "ground_truth/<defect>/000_mask.png" | null,
        "split_origin":  "train_good" | "test_good" | "test_defect"
    }

Cross-category records add a ``"category"`` key.

Assumptions
-----------
- The dataset has been extracted by ``extract_dataset.py`` and (optionally)
  validated by ``validate_dataset.py``. Categories sit either directly
  under ``raw_dir`` or under ``raw_dir/mvtec_ad/``.
- Splits are reproducible: identical ``(raw_dir, val_ratio, seed)`` always
  produce identical manifests on the same machine, regardless of
  filesystem listing order.
- ``stratify_by_defect`` only meaningfully applies to mixed-pool settings
  (cross-category bundles stratify by category; per-category splits draw
  the val partition randomly from ``train/good`` since no defect labels
  exist there).
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

__all__ = ["build_splits"]

LOG = logging.getLogger(__name__)

IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
})
MASK_EXTS: frozenset[str] = frozenset({".png", ".bmp", ".tif", ".tiff"})

GOOD_LABEL: str = "good"
TRAIN_DIR: str = "train"
TEST_DIR: str = "test"
MASK_DIR: str = "ground_truth"
MASK_SUFFIX: str = "_mask"

LABEL_NORMAL: int = 0
LABEL_ANOMALY: int = 1

CROSS_CATEGORY_DIR: str = "_cross_category"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_splits(raw_dir: Path,
                 splits_dir: Path,
                 val_ratio: float,
                 seed: int,
                 stratify_by_defect: bool = True,
                 *,
                 write_cross_category: bool = True,
                 **_unused) -> dict:
    """Generate per-category and cross-category split manifests.

    See module docstring for the contract.
    """
    raw_dir = Path(raw_dir)
    splits_dir = Path(splits_dir)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must lie in [0, 1); got {val_ratio!r}"
        )

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    splits_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = _resolve_dataset_root(raw_dir)
    LOG.info("Building splits from %s -> %s", dataset_root, splits_dir)

    t0 = time.perf_counter()
    categories = _list_categories(dataset_root)
    if not categories:
        raise RuntimeError(
            f"No category folders found under {dataset_root}. "
            "Run extract_dataset.py first."
        )

    summary: dict[str, dict] = OrderedDict()
    per_category_records: dict[str, dict] = {}

    for category in categories:
        manifest = _build_category_manifest(
            category=category,
            category_root=dataset_root / category,
            val_ratio=val_ratio,
            seed=seed,
            stratify_by_defect=stratify_by_defect,
        )
        path = splits_dir / f"{category}.json"
        _write_json_atomic(path, manifest)
        per_category_records[category] = manifest
        summary[category] = {
            "train":       manifest["counts"]["train"],
            "val":         manifest["counts"]["val"],
            "test_good":   manifest["counts"]["test_good"],
            "test_defect": manifest["counts"]["test_defect"],
            "test":        manifest["counts"]["test"],
            "manifest":    str(path.relative_to(splits_dir).as_posix()),
        }
        LOG.info(
            "[%s] train=%d, val=%d, test=%d (good=%d, defect=%d) -> %s",
            category, manifest["counts"]["train"], manifest["counts"]["val"],
            manifest["counts"]["test"], manifest["counts"]["test_good"],
            manifest["counts"]["test_defect"], path.name,
        )

    if write_cross_category and len(categories) > 1:
        cross_summary = _write_cross_category_bundles(
            splits_dir=splits_dir,
            per_category=per_category_records,
            seed=seed,
        )
        summary[CROSS_CATEGORY_DIR] = cross_summary

    elapsed = time.perf_counter() - t0
    return {
        "splits_dir": splits_dir,
        "categories": categories,
        "summary": dict(summary),
        "n_categories": len(categories),
        "elapsed_seconds": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Per-category split
# ---------------------------------------------------------------------------
def _build_category_manifest(*,
                             category: str,
                             category_root: Path,
                             val_ratio: float,
                             seed: int,
                             stratify_by_defect: bool) -> dict:
    """Construct a deterministic train/val/test manifest for one category."""
    train_good_dir = category_root / TRAIN_DIR / GOOD_LABEL
    test_root = category_root / TEST_DIR
    masks_root = category_root / MASK_DIR

    if not train_good_dir.is_dir():
        raise FileNotFoundError(
            f"[{category}] missing train/good at {train_good_dir}"
        )
    if not test_root.is_dir():
        raise FileNotFoundError(
            f"[{category}] missing test/ at {test_root}"
        )

    # ---- train/good → train + val ---------------------------------------
    train_good_files = _list_images(train_good_dir)
    rng = random.Random(_combine_seed(seed, category))
    rng.shuffle(train_good_files)

    n_total = len(train_good_files)
    n_val = int(round(val_ratio * n_total))
    n_train = n_total - n_val
    train_files = train_good_files[:n_train]
    val_files = train_good_files[n_train:]

    train_records = [
        _make_record(
            path=p, raw_root=category_root,
            label=LABEL_NORMAL, defect=GOOD_LABEL,
            mask_path=None, split_origin="train_good",
        )
        for p in train_files
    ]
    val_records = [
        _make_record(
            path=p, raw_root=category_root,
            label=LABEL_NORMAL, defect=GOOD_LABEL,
            mask_path=None, split_origin="train_good",
        )
        for p in val_files
    ]

    # ---- official test set (preserved verbatim) -------------------------
    test_records: list[dict] = []
    test_good_dir = test_root / GOOD_LABEL
    n_test_good = 0
    for p in _list_images(test_good_dir):
        test_records.append(_make_record(
            path=p, raw_root=category_root,
            label=LABEL_NORMAL, defect=GOOD_LABEL,
            mask_path=None, split_origin="test_good",
        ))
        n_test_good += 1

    n_test_defect = 0
    n_test_defect_per_type: dict[str, int] = {}
    for defect_dir in sorted(p for p in test_root.iterdir()
                              if p.is_dir() and p.name != GOOD_LABEL):
        defect_name = defect_dir.name
        mask_dir = masks_root / defect_name
        defect_count = 0
        for img in _list_images(defect_dir):
            mask_path = _find_mask(img, mask_dir)
            test_records.append(_make_record(
                path=img, raw_root=category_root,
                label=LABEL_ANOMALY, defect=defect_name,
                mask_path=mask_path, split_origin="test_defect",
            ))
            defect_count += 1
        n_test_defect_per_type[defect_name] = defect_count
        n_test_defect += defect_count

    counts = {
        "train":           len(train_records),
        "val":             len(val_records),
        "test":            len(test_records),
        "test_good":       n_test_good,
        "test_defect":     n_test_defect,
        "test_per_defect": n_test_defect_per_type,
    }
    return {
        "category": category,
        "raw_root": str(category_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "stratify_by_defect": stratify_by_defect,
        "splits": {
            "train": train_records,
            "val":   val_records,
            "test":  test_records,
        },
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Cross-category bundles
# ---------------------------------------------------------------------------
def _write_cross_category_bundles(*,
                                  splits_dir: Path,
                                  per_category: dict[str, dict],
                                  seed: int) -> dict:
    """Pool / leave-one-out manifests across categories."""
    target_dir = splits_dir / CROSS_CATEGORY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    # ---- 1. all_categories (pooled, stratified by category) -------------
    pooled = _pool_manifests(per_category)
    all_path = target_dir / "all_categories.json"
    _write_json_atomic(all_path, pooled)
    summary["all_categories"] = {
        "manifest": str(all_path.relative_to(splits_dir).as_posix()),
        **pooled["counts"],
    }
    LOG.info(
        "[cross/all_categories] train=%d val=%d test=%d -> %s",
        pooled["counts"]["train"], pooled["counts"]["val"],
        pooled["counts"]["test"], all_path.name,
    )

    # ---- 2. leave-one-out -----------------------------------------------
    loo_summary: dict[str, dict] = {}
    categories = list(per_category.keys())
    for held_out in categories:
        bundle = _leave_one_out(per_category, held_out=held_out, seed=seed)
        path = target_dir / f"leave_one_out_{held_out}.json"
        _write_json_atomic(path, bundle)
        loo_summary[held_out] = {
            "manifest": str(path.relative_to(splits_dir).as_posix()),
            **bundle["counts"],
        }
        LOG.info(
            "[cross/leave_one_out:%s] train=%d val=%d test=%d -> %s",
            held_out, bundle["counts"]["train"], bundle["counts"]["val"],
            bundle["counts"]["test"], path.name,
        )

    summary["leave_one_out"] = loo_summary
    return summary


def _pool_manifests(per_category: dict[str, dict]) -> dict:
    """Concatenate per-category splits into a single pooled manifest.

    By construction (each category contributes its own train/val/test
    proportionally), the result is naturally stratified by category.
    """
    train, val, test = [], [], []
    for category, manifest in per_category.items():
        for r in manifest["splits"]["train"]:
            train.append({**r, "category": category})
        for r in manifest["splits"]["val"]:
            val.append({**r, "category": category})
        for r in manifest["splits"]["test"]:
            test.append({**r, "category": category})

    return {
        "name": "all_categories",
        "categories": list(per_category.keys()),
        "splits": {"train": train, "val": val, "test": test},
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "per_category": {
                cat: {
                    "train": m["counts"]["train"],
                    "val":   m["counts"]["val"],
                    "test":  m["counts"]["test"],
                }
                for cat, m in per_category.items()
            },
        },
    }


def _leave_one_out(per_category: dict[str, dict],
                   *,
                   held_out: str,
                   seed: int) -> dict:
    """Build a leave-one-out manifest where ``held_out`` is the test set."""
    train, val = [], []
    for category, manifest in per_category.items():
        if category == held_out:
            continue
        for r in manifest["splits"]["train"]:
            train.append({**r, "category": category})
        for r in manifest["splits"]["val"]:
            val.append({**r, "category": category})

    test = [
        {**r, "category": held_out}
        for r in per_category[held_out]["splits"]["test"]
    ]

    return {
        "name": f"leave_one_out_{held_out}",
        "held_out_category": held_out,
        "train_categories": [c for c in per_category if c != held_out],
        "seed": seed,
        "splits": {"train": train, "val": val, "test": test},
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
        },
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _resolve_dataset_root(raw_dir: Path) -> Path:
    """Find the directory holding category folders (handles ``mvtec_ad/``)."""
    nested = raw_dir / "mvtec_ad"
    if nested.is_dir() and any((nested / d).is_dir() for d in nested.iterdir()
                                if d.is_dir()):
        return nested
    return raw_dir


def _list_categories(dataset_root: Path) -> list[str]:
    return sorted(
        p.name for p in dataset_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
        and (p / TRAIN_DIR).is_dir()
    )


def _list_images(directory: Path,
                 exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    if not directory.is_dir():
        return []
    exts_l = {e.lower() for e in exts}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts_l
    )


def _find_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Locate the mask for ``image_path`` under ``mask_dir`` (MVTec naming)."""
    if not mask_dir.is_dir():
        return None
    stem = image_path.stem
    for ext in MASK_EXTS:
        candidate = mask_dir / f"{stem}{MASK_SUFFIX}{ext}"
        if candidate.is_file():
            return candidate
        candidate_alt = mask_dir / f"{stem}{ext}"
        if candidate_alt.is_file():
            return candidate_alt
    return None


# ---------------------------------------------------------------------------
# Record / I/O helpers
# ---------------------------------------------------------------------------
def _make_record(*,
                 path: Path,
                 raw_root: Path,
                 label: int,
                 defect: str,
                 mask_path: Path | None,
                 split_origin: str) -> dict:
    """Build a portable per-sample record (POSIX relative + absolute paths)."""
    return {
        "path":         _rel(path, raw_root),
        "abs_path":     str(path),
        "label":        int(label),
        "defect":       defect,
        "mask":         _rel(mask_path, raw_root) if mask_path else None,
        "abs_mask":     str(mask_path) if mask_path else None,
        "split_origin": split_origin,
    }


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _combine_seed(seed: int, key: str) -> int:
    """Combine the global seed with a per-category key for independent RNGs."""
    return (seed * 1_000_003 + (hash(key) & 0xFFFFFFFF)) & 0x7FFFFFFF


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(path)
