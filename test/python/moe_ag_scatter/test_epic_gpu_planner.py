"""CPU parity tier for the EPIC per-iteration GPU planner (SCHEMA rule 5).

Pins, on device="cpu":
  * plan_migration_swaps_gpu == plan_migration_swaps (list equality across
    migration rounds — tie-breaks, tau gate, one-swap-per-rank included);
  * slot_loads_from_rqp == slot_batch_loads;
  * the device-agnostic a2av combine/compress builders == the
    pre-2026-08-20 scalar-loop bodies (vendored below as the frozen
    reference, since the shipped originals now delegate to the _dev twins);
  * EpicIterPlanner.derive reproduces build_epic_group_layouts (m=1),
    build_epic_hc_bundles, and the combine-entry indices bitwise
    (planner.check_against on a stub runner).

Run: pytest test/python/moe_ag_scatter/test_epic_gpu_planner.py -q
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.testing.a2av_combine_indices import (
    build_a2av_combine_indices_dev,
    build_a2av_compress_indices_dev,
    build_a2av_unique_counts_dev,
)
from flux.testing.epic_semantics import (
    EpicIterPlanner,
    apply_swaps,
    build_epic_group_layouts,
    build_epic_hc_bundles,
    plan_migration_swaps,
    plan_migration_swaps_gpu,
    slot_batch_loads,
    slot_loads_from_rqp,
)
from flux.testing.ep_gpu_plan import d6_rank_quota_prefix

from test_epic_planner import NAMED_CASES, build

CASES = NAMED_CASES[:6]


# ---------------------------------------------------------------------------
# Frozen pre-2026-08-20 reference bodies (minus the .cuda() coercions) for
# the _dev builder parity pins.
# ---------------------------------------------------------------------------


def _legacy_combine_indices(routing_idx, split_cpu, rank, world_size, topk):
    routing_idx = routing_idx.long().cpu()
    m_full = routing_idx.numel()
    cpr = m_full // world_size
    splits = split_cpu.long().cpu()
    n_experts_per_rank = splits.numel() // world_size
    ep_m_start = int(splits[: rank * n_experts_per_rank].sum())
    m_this_ep = int(splits[rank * n_experts_per_rank:
                           (rank + 1) * n_experts_per_rank].sum())
    iota_m = torch.arange(m_full, dtype=torch.long)
    copy_of_row = torch.empty(m_full, dtype=torch.long).scatter_(
        0, routing_idx, iota_m)
    copy_of_row = copy_of_row[ep_m_start:ep_m_start + m_this_ep]
    home = copy_of_row // cpr
    pack_index = (home * m_this_ep
                  + torch.arange(m_this_ep, dtype=torch.long)).argsort()
    splits_cum = splits.cumsum(0)
    my_copies = routing_idx[rank * cpr:(rank + 1) * cpr]
    e_of = torch.searchsorted(splits_cum, my_copies, right=True)
    iota_c = torch.arange(cpr, dtype=torch.long)
    perm = (e_of * cpr + iota_c).argsort()
    reduce_index = torch.empty(cpr, dtype=torch.long).scatter_(
        0, perm, iota_c)
    return pack_index.int(), reduce_index.int()


def _legacy_compress_indices(routing_idx, split_cpu, unique_counts, rank,
                             world_size, nnodes, topk):
    routing_idx = routing_idx.long().cpu()
    splits = split_cpu.long().cpu()
    U = unique_counts.long().cpu()
    m_full = routing_idx.numel()
    W, NN = world_size, nnodes
    L = W // NN
    cpr = m_full // W
    ntok_local = cpr // topk
    ntokens = m_full // topk
    nex = splits.numel()
    E_loc = nex // W
    my_node, my_lr = rank // L, rank % L
    iota_m = torch.arange(m_full, dtype=torch.long)
    splits_cum = splits.cumsum(0)
    e_of_copy = torch.searchsorted(
        splits_cum, routing_idx, right=True).clamp_max_(nex - 1)
    owner = e_of_copy // E_loc
    home = iota_m // cpr
    kmax = torch.iinfo(torch.long).max
    C = torch.zeros(W, W, dtype=torch.long)
    C.index_put_((owner, home), torch.ones(m_full, dtype=torch.long),
                 accumulate=True)
    recv_off_C = torch.cat([torch.zeros(1, dtype=torch.long),
                            C[:, rank].cumsum(0)[:-1]])
    Cp_col = torch.zeros(W, dtype=torch.long)
    for s in range(W):
        if s // L == my_node:
            Cp_col[s] = C[s, rank]
        elif s % L == my_lr:
            Cp_col[s] = U[rank, s // L]
    recv_off_Cp = torch.cat([torch.zeros(1, dtype=torch.long),
                             Cp_col.cumsum(0)[:-1]])
    owner_node, home_node = owner // L, home // L
    conv_mask = ((owner_node == my_node) & (home_node != my_node)
                 & (home % L == my_lr))
    conv_total = int(conv_mask.sum())
    if conv_total > 0:
        seg = home_node - (home_node > my_node).long()
        ls = owner % L
        conv_key = ((((seg * L + ls) * nex + e_of_copy) * m_full
                     + iota_m).masked_fill(~conv_mask, kmax))
        conv_copy = conv_key.argsort()[:conv_total]
        wkey = seg[conv_copy] * ntokens + conv_copy // topk
        worder = wkey.argsort(stable=True)
        wire_copy = worder
        _, counts = torch.unique_consecutive(wkey[worder],
                                             return_counts=True)
        wire_ptr = torch.cat([torch.zeros(1, dtype=torch.long),
                              counts.cumsum(0)])
    else:
        wire_ptr = torch.zeros(1, dtype=torch.long)
        wire_copy = torch.empty(0, dtype=torch.long)
    iota_c = torch.arange(cpr, dtype=torch.long)
    e_my = e_of_copy[rank * cpr:(rank + 1) * cpr]
    owner_my = owner[rank * cpr:(rank + 1) * cpr]
    perm = (e_my * cpr + iota_c).argsort()
    rows_C = torch.empty(cpr, dtype=torch.long).scatter_(0, perm, iota_c)
    rows_Cp = rows_C - recv_off_C[owner_my] + recv_off_Cp[owner_my]
    K = topk + NN + 1
    tl = iota_c // topk
    own_mask = owner_my // L == my_node
    own_total = int(own_mask.sum())
    key_own = (tl * K + (iota_c - tl * topk)).masked_fill(~own_mask, kmax)
    ord_own = key_own.argsort()
    own_rows = rows_Cp[ord_own][:own_total]
    own_keys = key_own[ord_own][:own_total]
    onode = owner_my // L
    flags = torch.zeros(ntok_local * NN, dtype=torch.long)
    flags.scatter_(0, tl * NN + onode, 1)
    flags = flags.view(ntok_local, NN)
    flags[:, my_node] = 0
    rem_total = int(flags.sum())
    pos = flags.cumsum(0) - flags
    rem_base = torch.zeros(NN, dtype=torch.long)
    for m in range(NN):
        if m != my_node:
            rem_base[m] = recv_off_Cp[m * L + my_lr]
    rem_rows2d = pos + rem_base.view(1, NN)
    tl_col = torch.arange(ntok_local, dtype=torch.long).view(-1, 1)
    m_row = torch.arange(NN, dtype=torch.long).view(1, -1)
    key_rem = ((tl_col * K + topk + m_row)
               .masked_fill(flags.eq(0), kmax).reshape(-1))
    ord_rem = key_rem.argsort()
    rem_rows = rem_rows2d.reshape(-1)[ord_rem][:rem_total]
    rem_keys = key_rem[ord_rem][:rem_total]
    keys_all = torch.cat([own_keys, rem_keys])
    vals_all = torch.cat([own_rows, rem_rows])
    order = keys_all.argsort()
    red_row = vals_all[order]
    red_ptr = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.bincount(keys_all[order] // K,
                       minlength=ntok_local).cumsum(0)])
    return wire_ptr.int(), wire_copy.int(), red_ptr.int(), red_row.int()


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 5.0])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_migration_decision_parity(case, tau):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    D = cfg.R // nn
    for _round in range(10):
        ref = plan_migration_swaps(plan, tau, D)
        got = plan_migration_swaps_gpu(
            plan.p2l, slot_batch_loads(plan), tau, D, cfg.nlp, cfg.G)
        assert got == ref, f"round {_round}"
        if not ref:
            break
        apply_swaps(plan, ref)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_slot_loads_parity(case):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    rqp = d6_rank_quota_prefix(tpe, plan.lcnts, cfg.max_replicas_dim)
    assert torch.equal(rqp, plan.rank_quota_prefix)
    got = slot_loads_from_rqp(rqp, plan.l2p, plan.lcnts, cfg.P)
    assert torch.equal(got, slot_batch_loads(plan))


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: c.name)
def test_dev_builders_vs_legacy(case):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    L = cfg.R // nn
    bundles = build_epic_hc_bundles(plan, topk_all, 1, L)
    b = bundles[0]
    routing = b.meta.scatter_index.flatten()
    for rank in range(cfg.R):
        p_ref, r_ref = _legacy_combine_indices(
            routing, b.meta.splits, rank, cfg.R, b.K_g)
        p_dev, r_dev = build_a2av_combine_indices_dev(
            routing.long(), b.meta.splits, rank, cfg.R, b.K_g)
        assert torch.equal(p_dev, p_ref) and torch.equal(r_dev, r_ref)
    uc_dev = build_a2av_unique_counts_dev(
        b.virtual_choosed, cfg.R, nn, b.gpe)
    if nn > 1:
        for rank in range(cfg.R):
            ref = _legacy_compress_indices(
                routing, b.meta.splits, uc_dev, rank, cfg.R, nn, b.K_g)
            got = build_a2av_compress_indices_dev(
                routing.long(), b.meta.splits, uc_dev, rank, cfg.R, nn,
                b.K_g)
            for g, r in zip(got, ref):
                assert torch.equal(g, r)


@pytest.mark.parametrize("l01", [False, True], ids=["l0", "l01"])
@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: c.name)
def test_epic_iter_planner_matches_reference(case, l01):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case)
    L = cfg.R // nn
    loads_gather = tpe.reshape(-1).to(torch.int32)
    bundles = build_epic_hc_bundles(plan, topk_all, 1, L)
    for rank in range(min(cfg.R, 6)):
        lay_ref = build_epic_group_layouts(plan, rank, topk_all, 1,
                                           ranks_per_node=L)
        hcc = l01
        entry = {}
        if hcc:
            b = bundles[0]
            routing = b.meta.scatter_index.flatten()
            pk, rd = _legacy_combine_indices(
                routing, b.meta.splits, rank, cfg.R, b.K_g)
            entry = dict(pack=pk, red=rd, uc=None, wire=None, redcsr=None,
                         inbuf=torch.zeros(
                             max(int(b.meta.m_per_rank[rank]), 1), 1))
            if nn > 1:
                uc = build_a2av_unique_counts_dev(
                    b.virtual_choosed, cfg.R, nn, b.gpe)
                wp, wc, rp, rr = _legacy_compress_indices(
                    routing, b.meta.splits, uc, rank, cfg.R, nn, b.K_g)
                entry.update(uc=uc, wire=[wp, wc], redcsr=[rp, rr])
        stub = SimpleNamespace(elay=lay_ref, _hc_bundles=bundles,
                               _hcc=[entry])
        planner = EpicIterPlanner(
            plan, rank, torch.device("cpu"), topk_all, L, l01=l01,
            hc=True, hcc=hcc, kg_frozen=bundles[0].K_g)
        assert torch.equal(planner.local_loads(), tpe[rank].to(torch.int32))
        ip = planner.derive(loads_gather)
        planner.check_against(ip, stub)
