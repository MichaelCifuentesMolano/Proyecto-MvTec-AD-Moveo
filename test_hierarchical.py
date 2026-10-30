"""Self-checks for the hierarchical pipeline (no dataset / GPU needed).

    python test_hierarchical.py

Each check guards a bug found in the May 2026 pipeline or a new mechanism.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from src.data.build_splits import _combine_seed, _split_stratified
from src.models.model_factory import build_model
from src.nas.nsga2_engine import NSGA2Config, NSGA2Engine
from src.nas.space import DiscreteSpace, quant_space
from src.profiling.complexity import measure_complexity
from src.quantization.qat_wrapper import FakeQuantize, QConv2d, wrap_for_qat


def test_build_model_honours_arch_spec():
    # May 2026: every candidate silently built the default autoencoder.
    m = build_model({"arch_spec": {"family": "patch_cnn", "depth": 3,
                                   "base_width": 8, "input_size": 128}})
    assert (m.arch_spec.family, m.arch_spec.depth, m.arch_spec.base_width) == ("patch_cnn", 3, 8)


def test_quant_spec_is_applied_per_segment():
    # May 2026: the genome's bits never reached the model (always W8A8 / fp32).
    model = nn.Sequential(*[nn.Conv2d(4, 4, 3, padding=1) for _ in range(4)])
    wrap_for_qat(model, {"representation": "int", "weight_bits": [2, 8],
                         "act_bits": [4, 8], "symmetric": True, "per_channel": False})
    bits = [(m.weight_fq.cfg.bits, m.act_fq.cfg.bits) for m in model if isinstance(m, QConv2d)]
    assert bits == [(2, 4), (2, 4), (8, 8), (8, 8)], bits
    fp16 = nn.Sequential(nn.Conv2d(4, 4, 3))
    wrap_for_qat(fp16, {"representation": "fp16"})   # Level-1 spec: no bits at all
    assert not any(isinstance(m, FakeQuantize) for m in fp16.modules())


def test_ternary_and_power_of_two_levels():
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3)
    for rep, bits in (("int", [2]), ("pot", [4])):
        model = nn.Sequential(nn.Conv2d(4, 8, 3))
        model[0].weight.data.copy_(w)
        wrap_for_qat(model, {"representation": rep, "weight_bits": bits, "act_bits": [8],
                             "symmetric": True, "per_channel": False})
        q = model[0]
        q.weight_fq._observe(q.conv.weight)
        wq = q.weight_fq(q.conv.weight).detach()
        if rep == "int":   # ternary: exactly three levels {-s, 0, +s}
            assert len(torch.unique(wq)) == 3
        else:              # pot: |w| = max|w| * 2^-k, k in 0..6, or 0
            mags = wq.abs()[wq != 0] / w.abs().max()
            k = -torch.log2(mags)
            assert torch.allclose(k, k.round(), atol=1e-4) and k.max() <= 6


def test_bops_follow_bit_widths():
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1))
    float_cost = measure_complexity(model, (1, 3, 16, 16))
    wrap_for_qat(model, {"representation": "int", "weight_bits": [4], "act_bits": [8]})
    q_cost = measure_complexity(model, (1, 3, 16, 16))
    assert float_cost["macs_m"] == q_cost["macs_m"] == 16 * 16 * 8 * 27 / 1e6
    assert abs(q_cost["bops_g"] / float_cost["bops_g"] - 4 * 8 / (32 * 32)) < 1e-9
    assert abs(q_cost["weight_kb"] * 8 * 1024 - 8 * 27 * 4) < 1e-6


def test_space_roundtrip_and_fp16_dedup():
    space = quant_space({"family": "autoencoder"},
                        {"representations": ["int", "fp16"], "symmetric": [True],
                         "per_channel": [False], "weight_bits": [4, 8], "act_bits": [8]},
                        n_segments=2)
    rng = np.random.default_rng(0)
    v = space.sample_population(1, rng)[0]
    assert np.array_equal(space.encode(space.decode(v)), v)
    a = space.encode({"representation": "fp16", "symmetric": True, "per_channel": False,
                      "w_bits_0": 4, "w_bits_1": 4, "a_bits_0": 8, "a_bits_1": 8})
    b = a.copy()
    b[space.names.index("w_bits_0")] = 1          # different bits, same fp16 model
    assert space.fingerprint(a) == space.fingerprint(b)


def test_engine_finds_known_front_and_penalises_failures():
    # Two objectives on x in 0..9: maximise x, minimise x^2 -> every x is
    # Pareto-optimal except failures (x == 3 raises).
    calls = []

    def fitness(c):
        calls.append(c["x"])
        if c["x"] == 3:
            raise RuntimeError("boom")
        return {"performance": c["x"], "cost": c["x"] ** 2}

    space = DiscreteSpace({"x": list(range(10))}, lambda d: dict(d))
    eng = NSGA2Engine(NSGA2Config(population_size=10, objectives=["performance", "cost"],
                                  seed=0), space, fitness)
    eng.initialize(np.arange(10).reshape(-1, 1))
    eng.run(3)
    front = {int(eng.population[i][0]) for i in eng.pareto_indices()}
    assert front == set(range(10)) - {3}, front
    assert len(calls) == len(set(calls)) == 10   # each genome evaluated once (cache)


def test_splits_are_reproducible_and_disjoint():
    # May 2026: hash() is salted per process, so splits changed between runs.
    assert _combine_seed(42, "bottle") == 791316563   # same in every process
    recs = [{"defect": d, "path": f"{d}{i}"} for d in ("good", "crack") for i in range(10)]
    a, b = _split_stratified(recs, 0.3, random.Random(1))
    assert {r["path"] for r in a}.isdisjoint(r["path"] for r in b)
    assert sorted(r["defect"] for r in a) == ["crack"] * 3 + ["good"] * 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
