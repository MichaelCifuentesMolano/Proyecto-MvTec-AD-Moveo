"""
main_hierarchical.py
====================

Hierarchical multi-objective optimisation of architecture and numeric
representation / quantization (doctoral objective 2), on a pluggable task.

Modes (``mode`` in the YAML; same task, objectives and evaluator so the
results are directly comparable):

``hierarchical``
    **Level 1** — NSGA-II over architectures trained in floating point
    (objectives e.g. AUROC, MACs, latency, peak memory). The ``top_k``
    Level-1 solutions (best performance + knee points) go down.
    **Quantization sensitivity** — per architecture, task-level performance
    drop when one segment of the network is quantized to low precision
    (all others at 8 bit). It seeds Level 2 and is saved for analysis.
    **Level 2** — per architecture, NSGA-II over representation (int / pot /
    fp16), symmetry, per-channel and per-segment weight / activation bits,
    fine-tuning (QAT) from the Level-1 weights; objectives e.g. AUROC, BOPs,
    weight memory.
    **Level 3 hand-off** — merged Level-2 front, ``deployable`` flag
    (TensorRT runs fp16 and symmetric INT8 only) and ``best_candidates.json``
    for ``main_retrain.py`` / ``main_deploy.py`` (on-device latency + energy).
``flat``
    Ablation: one NSGA-II over Level-1 + Level-2 genes, trained from scratch.
``random``
    Baseline: random sampling of the flat space with the same evaluator.

Outputs (``<results_dir>/<mode>_seed<seed>/``): ``run_config.json``
(config, git commit, environment), per level ``evaluations.csv``,
``history.csv`` (hypervolume per generation, normalised with the level's
global ideal / nadir), ``pareto.csv``; ``sensitivity.json`` per Level-2 arch;
``final_front.csv``, ``best_candidates.json`` and ``run_summary.json``.

Usage::

    python main_hierarchical.py --config config/hierarchical.yaml
    python main_hierarchical.py --config config/hierarchical.yaml --mode flat --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.nas.fitness import FitnessConfig, FitnessEvaluator, candidate_id
from src.nas.nsga2_engine import NSGA2Config, NSGA2Engine
from src.nas.pareto import find_knee_points, hypervolume, pareto_front
from src.nas.space import arch_space, flat_space, quant_space
from src.quantization.qat_wrapper import _quantizable_layers
from src.utils.set_seed import set_seed

LOG = logging.getLogger("hierarchical")
HW_KEYS = {"latency_ms", "latency_ms_p95", "peak_ram_mb", "energy_mj"}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def make_task(cfg: dict[str, Any]):
    t = dict(cfg["task"])
    name = t.pop("name")
    if name == "mvtec":
        from src.tasks.mvtec import MVTecTask
        return MVTecTask(device=cfg["device"], seed=cfg["seed"], **t)
    raise ValueError(f"unknown task {name!r}")


def make_evaluator(cfg: dict[str, Any], task, level: dict[str, Any],
                   save_states_dir: Path | None = None) -> FitnessEvaluator:
    return FitnessEvaluator(FitnessConfig(
        device=cfg["device"], n_epochs=level["n_epochs"],
        profile_hardware=bool(HW_KEYS & set(level["objectives"])),
        energy_backend=cfg["energy_backend"], max_params_m=cfg["max_params_m"],
        save_states_dir=str(save_states_dir) if save_states_dir else None,
    ), task)


def provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """What is needed to reproduce the run: config, code version, environment."""
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001
            return ""
    return {
        "config": cfg,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# One NSGA-II level + its artefacts
# ---------------------------------------------------------------------------

def _minimised(metrics: dict[str, Any], keys: list[str], maximize: list[str]) -> list[float]:
    return [-float(metrics[k]) if k in maximize else float(metrics[k]) for k in keys]


def write_evaluations(path: Path, evaluations: list[dict[str, Any]]) -> None:
    metric_keys = sorted({k for e in evaluations for k in e["metrics"]})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generation", "fingerprint", "valid", *metric_keys,
                    "arch_spec", "quant_spec"])
        for e in evaluations:
            c = e["candidate"]
            w.writerow([e["generation"], e["fingerprint"], e["valid"],
                        *[e["metrics"].get(k, "") for k in metric_keys],
                        json.dumps(c.get("arch_spec")), json.dumps(c.get("quant_spec"))])


def run_level(name: str, space, level: dict[str, Any], evaluator: FitnessEvaluator,
              out_dir: Path, seed: int, seed_population: np.ndarray | None = None
              ) -> list[dict[str, Any]]:
    """Run NSGA-II for one level; return the valid evaluations (all generations)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys, maximize = list(level["objectives"]), list(level.get("maximize", ["performance"]))
    engine = NSGA2Engine(NSGA2Config(
        population_size=level["population_size"], objectives=keys,
        maximize=maximize, seed=seed), space, evaluator)
    LOG.info("[%s] genome=%d genes, N=%d, G=%d, objectives=%s", name,
             space.genome_length, level["population_size"],
             level.get("n_generations", 0), keys)

    snapshots: list[np.ndarray] = []

    def on_generation(_res, eng: NSGA2Engine) -> None:
        snapshots.append(eng.objectives[[i for i in range(len(eng.population))
                                         if eng.is_valid(i)]].copy())
        write_evaluations(out_dir / "evaluations.csv", eng.evaluations)

    t0 = time.perf_counter()
    engine.initialize(seed_population)
    on_generation(None, engine)
    engine.run(int(level.get("n_generations", 0)), on_generation)
    elapsed = time.perf_counter() - t0

    valid = [e for e in engine.evaluations if e["valid"]]
    if not valid:
        raise RuntimeError(f"[{name}] no valid evaluation")

    # Hypervolume per generation, normalised with this level's global
    # ideal / nadir so generations are comparable (reference = 1.1 * nadir).
    all_obj = np.array([_minimised(e["metrics"], keys, maximize) for e in valid])
    ideal, nadir = all_obj.min(axis=0), all_obj.max(axis=0)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generation", "n_valid", "n_pareto", "hypervolume"])
        for g, obj in enumerate(snapshots):
            if len(obj) == 0:
                w.writerow([g, 0, 0, 0.0])
                continue
            norm = (obj - ideal) / span
            front = norm[pareto_front(norm)]
            hv = hypervolume(front, np.full(len(keys), 1.1), seed=seed)
            w.writerow([g, len(obj), len(front), round(float(hv), 6)])

    front_idx = pareto_front(all_obj)
    write_evaluations(out_dir / "pareto.csv", [valid[i] for i in front_idx])
    LOG.info("[%s] done: %d evaluations (%d valid), front=%d, %.0f s", name,
             len(engine.evaluations), len(valid), len(front_idx), elapsed)
    level_stats[name] = {"evaluations": len(engine.evaluations), "valid": len(valid),
                         "pareto": len(front_idx), "seconds": round(elapsed, 1)}
    return valid


level_stats: dict[str, dict[str, Any]] = {}


def select_top(evals: list[dict[str, Any]], keys: list[str], maximize: list[str],
               k: int) -> list[dict[str, Any]]:
    """Best-performance solution + Chebyshev knee points of the front."""
    obj = np.array([_minimised(e["metrics"], keys, maximize) for e in evals])
    front = [int(i) for i in pareto_front(obj)]
    chosen = [min(front, key=lambda i: obj[i][keys.index("performance")])]
    if len(front) > 1:
        knees = find_knee_points(obj[front], n_knees=min(k, len(front)))
        chosen += [front[int(i)] for i in knees if front[int(i)] not in chosen]
    return [evals[i] for i in chosen[:k]]


# ---------------------------------------------------------------------------
# Quantization sensitivity (Level 1 -> Level 2)
# ---------------------------------------------------------------------------

def segment_sensitivity(evaluator: FitnessEvaluator, task, arch_spec: dict[str, Any],
                        state: str, n_seg: int, low_bits: int) -> dict[str, Any]:
    """Task performance drop when one segment alone drops to ``low_bits``
    (weights and activations; all other segments at 8 bit, PTQ, no training)."""
    def perf(w_bits: list[int], a_bits: list[int]) -> float:
        cand = {"arch_spec": arch_spec, "init_state": state,
                "quant_spec": {"representation": "int", "symmetric": True,
                               "per_channel": True, "weight_bits": w_bits,
                               "act_bits": a_bits}}
        model = evaluator.build(cand)
        task.calibrate(model, cand)
        return task.evaluate(model, cand)

    w8a8 = perf([8] * n_seg, [8] * n_seg)
    drops = []
    for s in range(n_seg):
        bits = [8] * n_seg
        bits[s] = low_bits
        drops.append(round(w8a8 - perf(bits, bits), 6))
    return {"n_segments": n_seg, "low_bits": low_bits, "w8a8_performance": w8a8,
            "performance_drop": drops}


def seed_population(space, sens: dict[str, Any], q: dict[str, list]) -> np.ndarray:
    """W8A8 / fp16 baselines + sensitivity-guided mixed precision per representation:
    the more sensitive half of the segments keeps the highest bits."""
    n = sens["n_segments"]
    order = np.argsort(sens["performance_drop"])[::-1]   # most sensitive first
    hi_w, hi_a = max(q["weight_bits"]), max(q["act_bits"])
    lo_w = min(b for b in q["weight_bits"] if b > 2) if any(b > 2 for b in q["weight_bits"]) else min(q["weight_bits"])
    lo_a = min(q["act_bits"])
    sym, perch = q["symmetric"][0], q["per_channel"][0]

    def genome(rep: str, w: list[int], a: list[int]) -> np.ndarray:
        v = {"representation": rep, "symmetric": sym, "per_channel": perch}
        v.update({f"w_bits_{s}": w[s] for s in range(n)})
        v.update({f"a_bits_{s}": a[s] for s in range(n)})
        return space.encode(v)

    guided_w, guided_a = [lo_w] * n, [lo_a] * n
    for s in order[: (n + 1) // 2]:
        guided_w[s], guided_a[s] = hi_w, hi_a
    seeds = [genome(r, guided_w, guided_a) for r in q["representations"] if r != "fp16"]
    if "int" in q["representations"]:
        seeds.insert(0, genome("int", [hi_w] * n, [hi_a] * n))
    if "fp16" in q["representations"]:
        seeds.append(genome("fp16", [hi_w] * n, [hi_a] * n))
    return np.vstack(seeds)


def deployable(quant_spec: dict[str, Any]) -> bool:
    """TensorRT on Jetson executes fp16 and symmetric INT8 convolutions only."""
    if quant_spec.get("representation") == "fp16":
        return True
    return (quant_spec.get("representation") == "int" and quant_spec.get("symmetric", False)
            and set(quant_spec["weight_bits"]) | set(quant_spec["act_bits"]) == {8})


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_hierarchical(cfg, task, out: Path) -> tuple[list[dict], dict[str, Any]]:
    l1, l2 = cfg["level1"], cfg["level2"]
    states = out / "l1" / "states"
    l1_evals = run_level("L1", arch_space(cfg["arch_choices"]), l1,
                         make_evaluator(cfg, task, l1, states), out / "l1", cfg["seed"])
    finalists = select_top(l1_evals, l1["objectives"], l1.get("maximize", ["performance"]),
                           l1["top_k"])

    ev2 = make_evaluator(cfg, task, l2)
    l2_evals: list[dict[str, Any]] = []
    for e in finalists:
        arch = e["candidate"]["arch_spec"]
        aid = candidate_id(e["candidate"])
        n_seg = min(cfg["max_segments"],
                    len(_quantizable_layers(task.build(e["candidate"]))))
        space = quant_space(arch, cfg["quant_choices"], n_seg, e["metrics"]["state_path"])
        low = min(b for b in cfg["quant_choices"]["weight_bits"] if b > 2)
        sens = segment_sensitivity(ev2, task, arch, e["metrics"]["state_path"], n_seg, low)
        sens["float_performance"] = e["metrics"]["performance"]
        (out / f"l2_{aid}").mkdir(parents=True, exist_ok=True)
        (out / f"l2_{aid}" / "sensitivity.json").write_text(json.dumps(sens, indent=2))
        LOG.info("[L2 %s] sensitivity: %s", aid, sens["performance_drop"])
        l2_evals += run_level(f"L2_{aid}", space, l2, ev2, out / f"l2_{aid}",
                              cfg["seed"], seed_population(space, sens, cfg["quant_choices"]))
    return l2_evals, l2


def run_flat(cfg, task, out: Path) -> tuple[list[dict], dict[str, Any]]:
    lv = cfg["flat"]
    space = flat_space(cfg["arch_choices"], cfg["quant_choices"], cfg["max_segments"])
    return run_level("flat", space, lv, make_evaluator(cfg, task, lv), out / "flat",
                     cfg["seed"]), lv


def run_random(cfg, task, out: Path) -> tuple[list[dict], dict[str, Any]]:
    lv = dict(cfg["random"], population_size=cfg["random"]["n_evaluations"], n_generations=0)
    space = flat_space(cfg["arch_choices"], cfg["quant_choices"], cfg["max_segments"])
    return run_level("random", space, lv, make_evaluator(cfg, task, lv), out / "random",
                     cfg["seed"]), lv


def finalize(evals: list[dict], level: dict[str, Any], top_k: int, out: Path) -> None:
    """Merged final front + Level-3 hand-off (best_candidates.json)."""
    keys, maximize = level["objectives"], level.get("maximize", ["performance"])
    obj = np.array([_minimised(e["metrics"], keys, maximize) for e in evals])
    front = [evals[int(i)] for i in pareto_front(obj)]
    for e in front:
        e["metrics"]["deployable"] = deployable(e["candidate"]["quant_spec"])
    write_evaluations(out / "final_front.csv", front)

    chosen = select_top(front, keys, maximize, top_k)
    deploy = [e for e in front if e["metrics"]["deployable"]]
    if deploy:  # always hand at least one model that TensorRT can run
        best_dep = max(deploy, key=lambda e: e["metrics"]["performance"])
        if best_dep not in chosen:
            chosen.append(best_dep)
    payload = {"objective_names": keys, "candidates": [{
        "candidate_id": f"rank{r:03d}_{candidate_id(e['candidate'])}",
        "candidate": {k: e["candidate"][k] for k in ("arch_spec", "quant_spec")},
        "objectives": {k: e["metrics"].get(k) for k in keys},
        "metrics": e["metrics"],
    } for r, e in enumerate(chosen)]}
    (out / "best_candidates.json").write_text(json.dumps(payload, indent=2, default=str))
    LOG.info("Final front: %d solutions (%d deployable); %d candidates handed to L3",
             len(front), len(deploy), len(chosen))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config/hierarchical.yaml")
    ap.add_argument("--mode", choices=["hierarchical", "flat", "random"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--category")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for k in ("mode", "seed"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.category:
        cfg["task"]["category"] = args.category

    out = Path(cfg["results_dir"]) / f"{cfg['task'].get('category', cfg['task']['name'])}" \
        / f"{cfg['mode']}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    (out / "run_config.json").write_text(json.dumps(provenance(cfg), indent=2))
    set_seed(cfg["seed"], warn_nondeterministic=True)

    task = make_task(cfg)
    t0 = time.perf_counter()
    runner = {"hierarchical": run_hierarchical, "flat": run_flat, "random": run_random}
    evals, final_level = runner[cfg["mode"]](cfg, task, out)
    finalize(evals, final_level, cfg["final_top_k"], out)
    (out / "run_summary.json").write_text(json.dumps({
        "mode": cfg["mode"], "seed": cfg["seed"],
        "total_evaluations": sum(s["evaluations"] for s in level_stats.values()),
        "total_seconds": round(time.perf_counter() - t0, 1),
        "levels": level_stats}, indent=2))
    LOG.info("Done -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
