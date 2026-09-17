"""COMET + EPLB physical-slot mapping — CPU tier (no GPU, no collectives).

The arm (flux.testing.comet_eplb; driver test_moe_l0l1_traffic.py
--placement eplb) executes an EPLB plan on the UNMODIFIED fused COMET ops by
routing over P = W*nlp physical slots. What must hold for that to be the
plan, checked here over a case battery + fuzz:

  * homing: the fused ops' destination rule (slot // nlp) sends every entry
    to the rank EPLB assigned that slot to (p2l consistent, no empty slot),
  * fidelity: p2l[phys] == the logical routing, entry for entry (a token's
    K entries keep their k-order, so routing weights need no permutation),
  * replica rule: per source rank, each expert's entries split over its
    instances by largest remainder (`local_spread`), i.e. exactly the
    eplb arm's fused-wire counts; `local_static` = src mod C,
  * rows: per-rank physical rows == plan.physical_rows_per_rank() (the
    eplb arm's own row accounting) — the two arms GEMM the same rows,
  * the setup reference (derive_physical_routing_all) equals each rank's
    own in-window derive_fused shard bitwise,
  * canonical slot weights: replicas of one expert are identical, the
    moved-bytes book-keeping counts exactly the non-home slots.

Run: pytest test/python/moe_combined/test_comet_eplb_map.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_ag_scatter")
)
from eplb_oracle import rebalance_experts  # noqa: E402
from flux.testing.comet_eplb import (  # noqa: E402
    build_comet_eplb_plan,
    comet_eplb_stats,
    derive_physical_routing_all,
    fill_canonical_slot_weights,
)
from flux.testing.eplb_semantics import EplbIterPlanner  # noqa: E402
from flux.testing.ep_gpu_plan import largest_remainder_split  # noqa: E402

CASES = [
    # (W, L, G, S, K, red, policy, seed)
    (4, 4, 32, 64, 4, 2, "global", 1),
    (8, 4, 64, 48, 8, 2, "global", 2),
    (8, 4, 128, 40, 8, 2, "global", 3),
    (8, 4, 64, 48, 8, 2, "hier", 4),
    (16, 4, 128, 32, 8, 1, "global", 5),
]


def _routing(W, S, K, G, seed, skew=3.0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(G, generator=g) ** skew + 1e-3
    out = torch.empty(W * S, K, dtype=torch.int32)
    for t in range(W * S):
        out[t] = torch.multinomial(w, K, replacement=False, generator=g).int()
    return out


def _build(W, L, G, S, K, red, policy, seed, replica_select="local_spread", pool=None):
    topk_all = _routing(W, S, K, G, seed).reshape(W, S, K)
    cfg, plan, tpe = build_comet_eplb_plan(
        G, W, S, K, 64, L, topk_all, pool, policy, rebalance_experts,
        redundant_per_rank=red, replica_select=replica_select,
    )
    planner = EplbIterPlanner(plan, 0, torch.device("cpu"), topk_all,
                              replica_select=replica_select)
    phys = derive_physical_routing_all(planner, W)
    return cfg, plan, tpe, topk_all, planner, phys


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("replica_select", ["local_spread", "local_static"])
def test_physical_routing_is_the_plan(case, replica_select):
    W, L, G, S, K, red, policy, seed = case
    cfg, plan, tpe, topk_all, planner, phys = _build(*case, replica_select=replica_select)
    assert cfg.P == W * cfg.nlp and cfg.nlp == G // W + red
    assert phys.shape == (W * S, K) and phys.dtype == torch.int32
    assert int(phys.min()) >= 0 and int(phys.max()) < cfg.P
    # fidelity, entry for entry (k-order preserved)
    logical = topk_all.reshape(W * S, K).long()
    assert torch.equal(plan.p2l.long()[phys.long()], logical)
    # rows per rank == the eplb arm's own accounting
    rows = torch.bincount((phys.long() // cfg.nlp).reshape(-1), minlength=W).tolist()
    assert rows == plan.physical_rows_per_rank()
    assert sum(rows) == W * S * K
    # setup reference == each rank's in-window shard
    for r in range(W):
        planner.rank = r
        assert torch.equal(planner.derive_fused(), phys[r * S:(r + 1) * S])


@pytest.mark.parametrize("case", CASES)
def test_local_spread_counts(case):
    W, L, G, S, K, red, policy, seed = case
    cfg, plan, tpe, topk_all, planner, phys = _build(*case)
    l2p, lcnts = plan.l2p.long(), plan.lcnts.long()
    for src in range(W):
        p = phys[src * S:(src + 1) * S].long().reshape(-1)
        for l in range(G):
            C = int(lcnts[l])
            n = int(tpe[src, l])
            got = [int((p == int(l2p[l, j])).sum()) for j in range(C)]
            base, rem = divmod(n, C)
            assert got == [base + (1 if j < rem else 0) for j in range(C)], (src, l, got)


@pytest.mark.parametrize("case", CASES[:3])
def test_local_static_is_src_mod_c(case):
    W, L, G, S, K, red, policy, seed = case
    cfg, plan, tpe, topk_all, planner, phys = _build(*case, replica_select="local_static")
    l2p, lcnts = plan.l2p.long(), plan.lcnts.long()
    for src in range(W):
        p = phys[src * S:(src + 1) * S].long().reshape(-1)
        for l in range(G):
            C = int(lcnts[l])
            n = int(tpe[src, l])
            j = src % C
            assert int((p == int(l2p[l, j])).sum()) == n


def test_placement_depends_on_pool_only():
    W, L, G, S, K, red, policy, seed = CASES[1]
    pool = (torch.rand(G, generator=torch.Generator().manual_seed(9)) * 100).tolist()
    _, plan_a, _, _, _, _ = _build(W, L, G, S, K, red, policy, 11, pool=pool)
    _, plan_b, _, _, _, _ = _build(W, L, G, S, K, red, policy, 12, pool=pool)
    assert torch.equal(plan_a.p2l, plan_b.p2l) and torch.equal(plan_a.l2p, plan_b.l2p)
    _, plan_c, _, _, _, _ = _build(W, L, G, S, K, red, policy, 11, pool=None)
    # self-oracle (batch load) generally differs from the pool placement
    assert plan_c.plan_hash() != plan_a.plan_hash()


def test_canonical_slot_weights():
    W, L, G, S, K, red, policy, seed = CASES[0]
    cfg, plan, tpe, topk_all, planner, phys = _build(*CASES[0])
    ffn, H = 16, 8
    fc1 = [torch.zeros(cfg.nlp, ffn, H, dtype=torch.bfloat16) for _ in range(W)]
    fc2 = [torch.zeros(cfg.nlp, H, ffn, dtype=torch.bfloat16) for _ in range(W)]
    moved = [fill_canonical_slot_weights(plan, r, fc1[r], fc2[r]) for r in range(W)]
    by_logical = {}
    n_nonhome = 0
    for r in range(W):
        for j in range(cfg.nlp):
            l = int(plan.p2l[r * cfg.nlp + j])
            if l in by_logical:
                a, b = by_logical[l]
                assert torch.equal(a, fc1[r][j]) and torch.equal(b, fc2[r][j])
            else:
                by_logical[l] = (fc1[r][j].clone(), fc2[r][j].clone())
            if not (r * cfg.epn <= l < (r + 1) * cfg.epn):
                n_nonhome += 1
    assert len(by_logical) == G  # every logical expert hosted somewhere
    per_slot = (ffn * H + H * ffn) * 2
    assert sum(moved) == n_nonhome * per_slot
    # distinct experts have distinct weights
    ws = torch.stack([v[0].float().reshape(-1) for v in by_logical.values()])
    assert torch.unique(ws, dim=0).shape[0] == G


def test_stats_columns():
    cfg, plan, tpe, topk_all, planner, phys = _build(*CASES[2])
    st = comet_eplb_stats(cfg, plan, tpe, tpe.long().sum(0).tolist(), phys)
    assert st["eplb_physical_experts"] == cfg.P
    assert st["eplb_imbalance_after"] >= 1.0 and st["eplb_imbalance_before"] >= 1.0
    # self-oracle placement balances the batch it was built from
    assert st["eplb_imbalance_after"] <= st["eplb_imbalance_before"] + 1e-9
    assert len(st["gemm_rows_per_rank_home"]) == cfg.R


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_fidelity(seed):
    g = torch.Generator().manual_seed(100 + seed)
    W = int([4, 8, 16][int(torch.randint(0, 3, (1,), generator=g))])
    G = W * int(torch.randint(2, 9, (1,), generator=g))
    K = int(torch.randint(2, 5, (1,), generator=g))
    S = int(torch.randint(8, 40, (1,), generator=g)) * 2
    red = int(torch.randint(1, 3, (1,), generator=g))
    cfg, plan, tpe, topk_all, planner, phys = _build(W, 4, G, S, K, red, "global", 200 + seed)
    logical = topk_all.reshape(W * S, K).long()
    assert torch.equal(plan.p2l.long()[phys.long()], logical)
    rows = torch.bincount((phys.long() // cfg.nlp).reshape(-1), minlength=W).tolist()
    assert rows == plan.physical_rows_per_rank()


# ---- fused router kernel parity (GPU tier; skipped without CUDA) ----------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("replica_select", ["local_spread", "local_static"])
@pytest.mark.parametrize("interleave", [True, False])
def test_kernel_matches_torch_router(case, replica_select, interleave):
    from flux.testing.comet_eplb_ext import MODE, load_ext
    ext = load_ext()
    W, L, G, S, K, red, policy, seed = case
    topk_all = _routing(W, S, K, G, seed).reshape(W, S, K)
    cfg, plan, tpe = build_comet_eplb_plan(
        G, W, S, K, 64, L, topk_all, None, policy, rebalance_experts,
        redundant_per_rank=red, interleave=interleave, replica_select=replica_select,
    )
    planner = EplbIterPlanner(plan, 0, torch.device("cuda"), topk_all,
                              replica_select=replica_select)
    l2p = planner.l2p.to(torch.int32).contiguous()
    lcnts = planner.lcnts.to(torch.int32).contiguous()
    for r in range(W):
        planner.rank = r
        ref = planner.derive_fused()
        got = ext.route_local(planner.topk_all[r].to(torch.int32).contiguous(),
                              l2p, lcnts, r, MODE[replica_select], interleave)
        torch.cuda.synchronize()
        assert torch.equal(got, ref), (case, replica_select, interleave, r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seed", range(4))
def test_kernel_fuzz_large(seed):
    """K2/K3-class sizes: G up to 896, K 16, S up to 4096."""
    from flux.testing.comet_eplb_ext import MODE, load_ext
    ext = load_ext()
    g = torch.Generator().manual_seed(500 + seed)
    W = int([8, 16, 32][seed % 3])
    G, K = [(384, 8), (896, 16), (128, 8), (384, 8)][seed]
    S = int(torch.randint(256, 4097, (1,), generator=g))
    topk_all = _routing(W, S, K, G, 700 + seed).reshape(W, S, K)
    cfg, plan, tpe = build_comet_eplb_plan(
        G, W, S, K, 64, 4, topk_all, None, "global", rebalance_experts,
        redundant_per_rank=2, replica_select="local_spread")
    planner = EplbIterPlanner(plan, 0, torch.device("cuda"), topk_all,
                              replica_select="local_spread")
    l2p = planner.l2p.to(torch.int32).contiguous()
    lcnts = planner.lcnts.to(torch.int32).contiguous()
    for r in (0, W // 2, W - 1):
        planner.rank = r
        ref = planner.derive_fused()
        got = ext.route_local(planner.topk_all[r].to(torch.int32).contiguous(),
                              l2p, lcnts, r, 0, True)
        torch.cuda.synchronize()
        assert torch.equal(got, ref), (seed, r)
    # timing: kernel vs torch router on the last rank (informational)
    topk32 = planner.topk_all[W - 1].to(torch.int32).contiguous()
    for _ in range(3):
        ext.route_local(topk32, l2p, lcnts, W - 1, 0, True); planner.derive_fused()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(20):
        ext.route_local(topk32, l2p, lcnts, W - 1, 0, True)
    torch.cuda.synchronize(); tk = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    for _ in range(20):
        planner.derive_fused()
    torch.cuda.synchronize(); tt = (time.perf_counter() - t0) / 20 * 1e3
    print(f"\n[route timing] W={W} G={G} K={K} S={S}: kernel {tk:.3f} ms, torch {tt:.3f} ms")
