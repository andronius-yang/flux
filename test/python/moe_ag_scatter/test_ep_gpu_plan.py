"""CPU parity tier for flux.testing.ep_gpu_plan (SCHEMA rule 5).

Every vectorized planner primitive must be BITWISE-identical to its scalar
reference on device="cpu" — same tensors, same dtypes, same tie-breaks.
The battery reuses the eplb/ultraep case shapes (skews, no-interleave,
R_red=0 => C=1, pool 'zeros' => zero-load experts).

Run: pytest test/python/moe_ag_scatter/test_ep_gpu_plan.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eplb_oracle import rebalance_experts
from flux.testing.eplb_semantics import build_eplb_plan
from flux.testing.epic_semantics import epic_rank_quota_prefix
from flux.testing.ep_gpu_plan import (
    comb_dst_slot_from_topk,
    d6_rank_quota_prefix,
    interleave_params_batched,
    largest_remainder_split,
    local_spread_rank_quota_prefix,
    place_slots_from_locals,
    rank_quota_prefix_nonlocal,
    reroute_expand_all_gpu,
    reroute_expand_gpu,
)
from flux.testing.ultraep_semantics import (
    _interleave_params,
    build_comm_layout,
    loads_from_topk,
    reroute_expand,
)

from test_ultraep_planner import Case, make_cfg, make_topk

CASES = [
    Case("gp_uniform", alpha=0.0),
    Case("gp_skewed", alpha=1.2),
    Case("gp_hot_expert", alpha=2.0, seed=5),
    Case("gp_no_interleave", alpha=0.8, interleave=False),
    Case("gp_no_redundant", alpha=0.8, R_red=0),   # C == 1 everywhere
    Case("gp_rred4", alpha=1.2, R_red=4),
    Case("tiny", S=64, K=2, G=16, R=4, D=4, alpha=1.5),
    Case("gp_fuzz0", alpha=0.55, seed=101),
    Case("gp_fuzz1", alpha=1.25, seed=102),
]

POOL_MODES = ("batch", "zeros")


def _pool_load(case, cfg, tpe, mode):
    if mode == "batch":
        return tpe.long().sum(0).double()
    gen = torch.Generator().manual_seed(case.seed + 7777)
    w = (torch.arange(cfg.G, dtype=torch.float64) + 1.0) ** (-1.1)
    w = w[torch.randperm(cfg.G, generator=gen)] * 10000.0
    w[torch.randperm(cfg.G, generator=gen)[:cfg.G // 4]] = 0.0
    return w


def build(case, pool_mode):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = build_eplb_plan(cfg, tpe, _pool_load(case, cfg, tpe, pool_mode),
                           "global", 1, rebalance_experts)
    return cfg, topk_all, tpe, plan


@pytest.mark.parametrize("pool_mode", POOL_MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_quota_and_rank_quota_parity(case, pool_mode):
    cfg, topk_all, tpe, plan = build(case, pool_mode)
    loads_g = tpe.long().sum(0)
    quota, prefix = largest_remainder_split(loads_g, plan.lcnts,
                                            cfg.max_replicas_dim)
    assert torch.equal(quota, plan.quota)
    assert torch.equal(prefix, plan.quota_prefix)

    rqp = rank_quota_prefix_nonlocal(tpe, plan.quota, plan.lcnts)
    assert torch.equal(rqp, plan.rank_quota_prefix)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_local_spread_prefix(case):
    """local_spread producer: (a) bitwise == largest_remainder_split
    applied per source row; (b) conservation prefix[C-1] == tpe[src,l];
    (c) count-equivalence to token-ordinal round-robin replica selection."""
    cfg, topk_all, tpe, plan = build(case, "batch")
    Cmax = cfg.max_replicas_dim
    rqp = local_spread_rank_quota_prefix(tpe, plan.lcnts, Cmax)
    for src in range(cfg.R):
        _, ref = largest_remainder_split(tpe[src].long(), plan.lcnts, Cmax)
        assert torch.equal(rqp[src], ref), f"src {src} per-source parity"
    # conservation: prefix at the last valid instance equals the load
    C = plan.lcnts.long()
    last = rqp.long().gather(
        2, (C - 1).clamp(min=0).view(1, cfg.G, 1).expand(cfg.R, cfg.G, 1)
    ).squeeze(-1)
    assert torch.equal(last, tpe.long())
    # count-equivalence vs true token round-robin: residue j of `o % C`
    # over m ordinals receives floor(m/C) + (j < m % C) — ELEMENTWISE
    # identical to the largest-remainder equal split.
    alloc = rqp.long().clone()
    alloc[:, :, 1:] -= rqp.long()[:, :, :-1]
    for src in range(min(cfg.R, 2)):
        for l in range(cfg.G):
            m, Cl = int(tpe[src, l]), int(plan.lcnts[l])
            rr = torch.bincount(
                torch.arange(m, dtype=torch.int64) % Cl, minlength=Cl)
            assert torch.equal(alloc[src, l, :Cl], rr), (src, l)


@pytest.mark.parametrize("mode", ["local_spread", "local_static"])
@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: c.name)
def test_eplb_iter_planner_replica_modes(case, mode):
    """End-to-end drift-guard consistency in the two campaign-2 replica
    modes: a plan built with replica_select == the planner's mode must
    reproduce build_comm_layout bitwise on every rank."""
    from flux.testing.eplb_semantics import EplbIterPlanner, build_eplb_plan
    from flux.testing.ultraep_semantics import loads_from_topk

    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = build_eplb_plan(cfg, tpe, tpe.long().sum(0).double(), "global",
                           1, rebalance_experts, replica_select=mode)
    loads_gather = tpe.reshape(-1).to(torch.int32)
    for rank in range(min(cfg.R, 4)):
        lay = build_comm_layout(plan, rank, topk_all, pinned_masters=False)
        planner = EplbIterPlanner(plan, rank, torch.device("cpu"),
                                  topk_all, want_comb=True,
                                  replica_select=mode)
        ip = planner.derive(loads_gather)
        planner.check_against(ip, lay)


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: c.name)
def test_d6_prefix_parity(case):
    cfg, topk_all, tpe, plan = build(case, "batch")
    ref = epic_rank_quota_prefix(cfg, tpe, plan.lcnts)
    got = d6_rank_quota_prefix(tpe, plan.lcnts, cfg.max_replicas_dim)
    assert torch.equal(got, ref)


def test_interleave_params_parity():
    totals = list(range(1, 512)) + [1000, 4095, 4096, 8191, 8192]
    for expert_id in (0, 1, 7, 63, 127):
        t = torch.tensor(totals, dtype=torch.int64)
        e = torch.full_like(t, expert_id)
        stride, offset = interleave_params_batched(t, e)
        for i, total in enumerate(totals):
            s_ref, o_ref = _interleave_params(total, expert_id)
            assert int(stride[i]) == s_ref, (total, expert_id)
            assert int(offset[i]) == o_ref, (total, expert_id)


@pytest.mark.parametrize("pool_mode", POOL_MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reroute_expand_parity(case, pool_mode):
    cfg, topk_all, tpe, plan = build(case, pool_mode)
    tok_all, phys_all = reroute_expand_all_gpu(
        plan.rank_quota_prefix, plan.l2p, plan.lcnts, topk_all.long(),
        cfg.interleave)
    for src in range(cfg.R):
        t_ref, p_ref = reroute_expand(cfg, plan, src, topk_all[src])
        t_gpu, p_gpu = reroute_expand_gpu(
            plan.rank_quota_prefix[src], plan.l2p, plan.lcnts,
            topk_all[src].long(), cfg.interleave)
        assert torch.equal(t_gpu, t_ref), f"src {src} tokens"
        assert torch.equal(p_gpu, p_ref), f"src {src} phys"
        assert torch.equal(tok_all[src], t_ref), f"src {src} batched tokens"
        assert torch.equal(phys_all[src], p_ref), f"src {src} batched phys"


@pytest.mark.parametrize("case", CASES[:6], ids=lambda c: c.name)
def test_recv_layout_parity(case):
    """place_slots_from_locals + the surrounding send/recv derivation
    reproduce build_comm_layout's fields for every rank."""
    cfg, topk_all, tpe, plan = build(case, "batch")
    R, nlp, S = cfg.R, cfg.nlp, cfg.S
    tok_all, phys_all = reroute_expand_all_gpu(
        plan.rank_quota_prefix, plan.l2p, plan.lcnts, topk_all.long(),
        cfg.interleave)
    # canonical (phys, token) order per source, as build_comm_layout does
    ent_tok, ent_phys = [], []
    for src in range(R):
        order = torch.argsort(phys_all[src] * (S + 1) + tok_all[src],
                              stable=True)
        ent_tok.append(tok_all[src][order])
        ent_phys.append(phys_all[src][order])

    for rank in range(R):
        lay = build_comm_layout(plan, rank, topk_all, pinned_masters=False)
        assert torch.equal(ent_tok[rank], lay.send_row_index)
        assert torch.equal(plan.p2l.long()[ent_phys[rank]],
                           lay.send_entry_logical)
        send_counts = torch.bincount(ent_phys[rank] // nlp, minlength=R)
        assert send_counts.tolist() == lay.send_counts
        locals_per_src = [
            ent_phys[src][(ent_phys[src] // nlp) == rank] - rank * nlp
            for src in range(R)
        ]
        assert [int(x.numel()) for x in locals_per_src] == lay.recv_counts
        all_local = (torch.cat(locals_per_src) if locals_per_src
                     else torch.zeros(0, dtype=torch.int64))
        place_slots, seg_rows, seg_start = place_slots_from_locals(
            all_local, nlp)
        assert torch.equal(place_slots, lay.place_slots)
        assert seg_rows.tolist() == lay.seg_rows
        assert seg_start.tolist() == lay.seg_start


@pytest.mark.parametrize("pool_mode", POOL_MODES)
@pytest.mark.parametrize("case", CASES[:6], ids=lambda c: c.name)
def test_eplb_iter_planner_matches_layout(case, pool_mode):
    """End-to-end: EplbIterPlanner.derive on device='cpu' reproduces the
    setup build_comm_layout bitwise on every rank (the driver's loud
    setup-time guard), including zero-row slots ('zeros' pool mode) and
    the l01 comb_dst_slot permutation."""
    from flux.testing.eplb_semantics import EplbIterPlanner

    cfg, topk_all, tpe, plan = build(case, pool_mode)
    loads_gather = tpe.reshape(-1).to(torch.int32)
    for rank in range(cfg.R):
        lay = build_comm_layout(plan, rank, topk_all, pinned_masters=False)
        planner = EplbIterPlanner(plan, rank, torch.device("cpu"),
                                  topk_all, want_comb=True)
        assert torch.equal(planner.local_loads(), tpe[rank].to(torch.int32))
        ip = planner.derive(loads_gather)
        planner.check_against(ip, lay)
        assert torch.equal(
            torch.sort(ip.comb_dst_slot).values,
            torch.arange(cfg.S * cfg.K, dtype=torch.int64))


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: c.name)
def test_comb_dst_slot(case):
    cfg, topk_all, tpe, plan = build(case, "batch")
    for src in range(cfg.R):
        tok, phys = reroute_expand(cfg, plan, src, topk_all[src])
        logical = plan.p2l.long()[phys]
        comb = comb_dst_slot_from_topk(topk_all[src].long(), tok, logical,
                                       cfg.G)
        # permutation of [0, S*K)
        assert torch.equal(torch.sort(comb).values,
                           torch.arange(cfg.S * cfg.K, dtype=torch.int64))
        # j really is the k-index of the entry's expert in its token row
        j = comb - tok * cfg.K
        assert torch.equal(topk_all[src].long()[tok, j], logical)
