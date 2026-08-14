################################################################################
#
# Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""CPU acceptance tests for the MoonEP -> fused-op virtual-expert mapping
(python/flux/testing/moonep_fused_map.py).

What is proven here (single process, no GPU, no torchrun):
  1. Homing rule: the virtual expert's home rank IS the plan's destination.
  2. scatter_index is a permutation; splits[v] == alloc[e, d] exactly.
  3. Delivered-set/dedup equivalence: compress's unique-token wire count
     (u_mat, owner-bitmap) equals the plan's Part-3 representative count
     bit-for-bit -- MoonEP rank-level dedup == compress intra-node dedup.
  4. The op's collective FLUX_CHECK preflight passes in-process.
  5. Order theorem: within each virtual expert the flatten-order entries have
     seg_pos == arange(cnt) and contiguous stable scatter positions, so
     fused_row_map's fused-row <-> padded-plan-slot correspondence is exact.
  6. Independent re-derivation of destinations from alloc cumsums matches the
     dst decode.
  7. Capacity-knob formulas are internally consistent and bound the layout.
  8. B < epn plans fail loudly; gateway assignment is deterministic, covers
     exactly the prefetch pairs, and only ever forwards within the dest node.

Run: pytest test/python/moe_ag_scatter/test_moonep_fused_map.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_moonep_planner import (  # noqa: E402
    CASES,
    R16_CASES,
    TEST_RS,
    PortCase,
    _case_supported,
    make_topk_all,
)

from flux.testing.moonep_semantics import MoonEPConfig, compute_moonep_plan  # noqa: E402
from flux.testing.moonep_fused_map import (  # noqa: E402
    assign_gateways,
    build_fused_metadata,
    build_virtual_map,
    fused_row_map,
    preflight_metadata_checks,
    required_a2av_knobs,
)

# Extra cases the planner grid does not stress for the mapping specifically.
EXTRA_CASES = [
    PortCase("dup_topk_r4", S=16, K=4, epn=2, min_R=4, max_R=4, routing="duplicate_topk"),
    PortCase("single_expert_r4", S=16, K=4, epn=2, min_R=4, max_R=4, routing="single_expert"),
]

GRID = [
    (case, R)
    for case in list(CASES) + list(R16_CASES) + EXTRA_CASES
    for R in TEST_RS
    if _case_supported(case, R)
]


def _L(R: int) -> int:
    return 4 if R % 4 == 0 else (2 if R % 2 == 0 else 1)


def _build(case: PortCase, R: int):
    topk_all = make_topk_all(case, R)
    cfg = MoonEPConfig(
        S=case.S,
        K=case.K,
        E=R * case.epn,
        R=R,
        B=case.B,
        token_padding=case.token_padding,
    )
    plan = compute_moonep_plan(cfg, topk_all)
    vmap = build_virtual_map(plan, topk_all)
    meta = build_fused_metadata(vmap, _L(R))
    return cfg, topk_all, plan, vmap, meta


def _decoded(plan):
    enc = plan.dst.long()
    raw = torch.where(enc < 0, -enc - 1, enc)
    dest = torch.div(raw, plan.cfg.NvS, rounding_mode="floor")
    loff = raw % plan.cfg.NvS
    return raw, dest, loff, enc >= 0


@pytest.mark.parametrize("case,R", GRID, ids=[f"{c.name}-R{r}" for c, r in GRID])
def test_mapping_invariants(case, R):
    cfg, topk_all, plan, vmap, meta = _build(case, R)
    gpe, E_virt = vmap.gpe, vmap.E_virt
    W, L = R, _L(R)
    _, dest, _, rep = _decoded(plan)

    # 1. homing rule, every entry (dedup'd included)
    vce = vmap.virtual_choosed.long().view(R, cfg.N)
    assert torch.equal(torch.div(vce, gpe, rounding_mode="floor"), dest)

    # 2. scatter_index is a permutation of [0, R*S*K)
    flat = meta.scatter_index.long().flatten()
    assert torch.equal(torch.sort(flat).values, torch.arange(flat.numel()))

    # splits[v] == alloc[e, d]; empty slots exactly zero
    expected = torch.zeros(E_virt, dtype=torch.int64)
    for d in range(R):
        for g in range(gpe):
            e = d * cfg.epn + g if g < cfg.epn else int(plan.experts_to_copy[d, g - cfg.epn])
            if g >= cfg.epn and e < 0:
                continue
            # local experts may still be migrated away; alloc is the authority
            expected[d * gpe + g] = plan.alloc[e, d]
    # a local expert whose tokens all migrated elsewhere contributes 0 rows
    # here even though alloc[e, d] may be 0 anyway -- alloc IS what lands.
    assert torch.equal(meta.splits.long(), expected)
    assert torch.equal(meta.m_per_rank, plan.alloc.sum(dim=0).long())

    # 4. op preflight (the collective FLUX_CHECK replica) passes
    preflight_metadata_checks(meta, W, L)

    # 3. compress unique-token counts == plan representative counts, bitwise
    u_mat = meta.a2av_unique_counts[:, :W].long()
    rep_cnt = torch.zeros(W, W, dtype=torch.int64)
    for s in range(W):
        rep_cnt[s] = torch.bincount(dest[s][rep[s]], minlength=W)
    assert torch.equal(u_mat, rep_cnt)


@pytest.mark.parametrize(
    "case,R",
    [(c, r) for c, r in GRID if r <= 8 or c.name.startswith("pm4n_g128")],
    ids=[f"{c.name}-R{r}" for c, r in GRID if r <= 8 or c.name.startswith("pm4n_g128")],
)
def test_order_theorem_and_row_map(case, R):
    cfg, topk_all, plan, vmap, meta = _build(case, R)
    gpe = vmap.gpe
    raw, dest, loff, _ = _decoded(plan)
    vce_flat = vmap.virtual_choosed.long().flatten()  # global flatten (token, k) order
    loff_flat = loff.flatten()
    scatter_flat = meta.scatter_index.long().flatten()
    splits = meta.splits.long()
    global_base = torch.cumsum(splits, 0) - splits

    for d in range(R):
        fused_rows, plan_slots = fused_row_map(vmap, d)
        base_local = 0
        for g in range(gpe):
            v = d * gpe + g
            cnt = int(splits[v])
            if cnt == 0:
                continue
            e = d * cfg.epn + g if g < cfg.epn else int(plan.experts_to_copy[d, g - cfg.epn])
            idx = (vce_flat == v).nonzero(as_tuple=True)[0]  # ascending flatten order
            assert idx.numel() == cnt
            # order theorem: seg_pos ascending 0..cnt-1 in flatten order
            seg_pos = loff_flat[idx] - int(plan.expert_off[d, e])
            assert torch.equal(seg_pos, torch.arange(cnt))
            # stable scatter positions are the contiguous window of v
            assert torch.equal(scatter_flat[idx], global_base[v] + torch.arange(cnt))
            # fused_row_map agrees with the dst-derived slots
            sel = slice(base_local, base_local + cnt)
            assert torch.equal(fused_rows[sel], torch.arange(base_local, base_local + cnt))
            assert torch.equal(plan_slots[sel], loff_flat[idx])
            base_local += cnt
        assert base_local == int(meta.m_per_rank[d])
        assert fused_rows.numel() == base_local


@pytest.mark.parametrize(
    "case,R",
    [(c, r) for c, r in GRID if r in (2, 4)],
    ids=[f"{c.name}-R{r}" for c, r in GRID if r in (2, 4)],
)
def test_independent_dest_rederivation(case, R):
    """Re-derive destinations from alloc cumsums (plan Part-2 math,
    independent implementation) and compare with the dst decode."""
    cfg, topk_all, plan, vmap, _ = _build(case, R)
    _, dest, _, _ = _decoded(plan)
    tpe_cumsum = plan.tpe.long().cumsum(0)  # [R, E]
    alloc_cumsum = plan.alloc.long().cumsum(1)  # [E, R]
    for r in range(R):
        e_flat = topk_all[r].reshape(-1).long()
        # occurrence index of each entry among same-expert entries of rank r
        local_cnt = torch.zeros_like(e_flat)
        seen: dict = {}
        for i, e in enumerate(e_flat.tolist()):
            local_cnt[i] = seen.get(e, 0)
            seen[e] = seen.get(e, 0) + 1
        prev = tpe_cumsum[r - 1] if r > 0 else torch.zeros(cfg.E, dtype=torch.int64)
        g = prev[e_flat] + local_cnt
        d_re = torch.searchsorted(alloc_cumsum[e_flat], g.unsqueeze(1), right=True).squeeze(1)
        assert torch.equal(d_re, dest[r])


@pytest.mark.parametrize(
    "case,R",
    [(c, r) for c, r in GRID if r >= 4 and _L(r) > 1],
    ids=[f"{c.name}-R{r}" for c, r in GRID if r >= 4 and _L(r) > 1],
)
def test_knob_formulas(case, R):
    cfg, topk_all, plan, vmap, meta = _build(case, R)
    W, L = R, _L(R)
    nn = W // L
    knobs = required_a2av_knobs(meta, W, L)
    recv = int(knobs["FLUX_A2AV_MAX_RECV_NTOKENS"])
    stage = int(knobs["FLUX_A2AV_MAX_STAGE_NTOKENS"])
    relay = int(knobs["FLUX_A2AV_MAX_RELAY_NTOKENS"])
    u = meta.a2av_unique_counts[:, :W].long()
    U = meta.a2av_unique_counts[:, W:].long()

    # recv covers both the GEMM row count and the worst union-region column
    assert recv >= int(meta.m_per_rank.max())
    for d in range(W):
        col = 0
        for s in range(W):
            col += int(U[s, d // L]) if (nn > 1 and s // L != d // L) else int(u[s, d])
        assert recv >= col

    if nn > 1:
        # chunk cuts partition each canonical stream exactly
        from flux.testing.moonep_fused_map import _chunk_bound

        for n in range(nn):
            for m in range(nn):
                if n == m:
                    continue
                total = int(U[n * L : (n + 1) * L, m].sum())
                assert _chunk_bound(U, L, n, m, L) == total
                rows = [
                    _chunk_bound(U, L, n, m, k + 1) - _chunk_bound(U, L, n, m, k)
                    for k in range(L)
                ]
                assert sum(rows) == total and min(rows) >= 0
        # stage covers the worst gateway inbound; relay the worst outbound
        for n in range(nn):
            for k in range(L):
                srows = sum(
                    _chunk_bound(U, L, ns, n, k + 1) - _chunk_bound(U, L, ns, n, k)
                    for ns in range(nn)
                    if ns != n
                )
                assert stage >= srows
                rrows = sum(
                    _chunk_bound(U, L, n, (n - dn + nn) % nn, k + 1)
                    - _chunk_bound(U, L, n, (n - dn + nn) % nn, k)
                    for dn in range(1, nn)
                )
                assert relay >= rrows


def test_b_less_than_epn_rejected():
    """Craft a plan where one destination must host >B distinct migrated
    experts: an empty receiver group gets quota S*K = 16, and every hot
    expert has < 16 tokens, so the greedy shed necessarily migrates >= 2
    distinct experts to that receiver; with B = 1 the mapping must refuse.
    (Note pm4n_all_remote is NOT such a case: its routing is remote but the
    rebalance migrates every token home, so no prefetch slot is ever
    needed.)"""
    R, epn, S, K = 4, 3, 8, 2  # E = 12, CAP = S*K = 16, 64 entries total
    ids = (
        [0] * 14 + [1] * 14 + [2] * 12  # group 0: 40 entries (every expert < 16)
        + [6] * 3 + [7] * 3 + [8] * 2  # group 2: 8
        + [9] * 6 + [10] * 5 + [11] * 5  # group 3: 16; group 1 empty (quota 16)
    )
    topk_all = torch.tensor(ids, dtype=torch.int32).reshape(R, S, K)
    cfg = MoonEPConfig(S=S, K=K, E=R * epn, R=R, B=1, token_padding=4)
    plan = compute_moonep_plan(cfg, topk_all)
    with pytest.raises(AssertionError, match="B = E/R"):
        build_virtual_map(plan, topk_all)


@pytest.mark.parametrize(
    "case,R",
    [(c, r) for c, r in GRID if r >= 4],
    ids=[f"{c.name}-R{r}" for c, r in GRID if r >= 4],
)
def test_gateway_assignment(case, R):
    cfg, topk_all, plan, vmap, _ = _build(case, R)
    L = _L(R)
    pairs = assign_gateways(plan, L)
    # determinism
    assert torch.equal(pairs, assign_gateways(plan, L))
    # coverage: exactly the valid prefetch pairs
    expected = {
        (d, b, int(plan.experts_to_copy[d, b]) // cfg.epn, int(plan.experts_to_copy[d, b]) % cfg.epn)
        for d in range(R)
        for b in range(cfg.B)
        if int(plan.experts_to_copy[d, b]) >= 0
    }
    got = {(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in pairs}
    assert got == expected
    # gateway structure
    direct = {(int(p[0]), int(p[1])) for p in pairs if int(p[4]) < 0}
    for p in pairs.tolist():
        d, b, home, src_row, gw, gws = p
        e = int(plan.experts_to_copy[d, b])
        assert e >= 0 and e // cfg.epn == home and e % cfg.epn == src_row
        if gw >= 0:
            assert gw // L == d // L and gw != d, "gateway must be a same-node peer"
            assert int(plan.experts_to_copy[gw, gws]) == e, "gateway slot must hold the expert"
            assert (gw, gws) in direct, "gateway itself must receive the inter-node leg"
        elif home // L != d // L:
            # cross-node direct member: it is either a singleton group or the
            # group's gateway; verify no OTHER same-node member of the same
            # expert is also direct (one inter-node leg per (e, node))
            others = [
                q
                for q in pairs.tolist()
                if q[2] == home
                and int(plan.experts_to_copy[q[0], q[1]]) == e
                and q[0] // L == d // L
                and q[0] != d
            ]
            assert all(int(q[4]) == d for q in others), (
                "cross-node group must have exactly one direct member (the gateway)"
            )


@pytest.mark.parametrize(
    "case,R",
    [(c, r) for c, r in GRID if r >= 4],
    ids=[f"{c.name}-R{r}" for c, r in GRID if r >= 4],
)
def test_push_plan_stats(case, R):
    from flux.testing.moonep_fused_map import push_plan_stats

    cfg, topk_all, plan, vmap, _ = _build(case, R)
    L = _L(R)
    pairs = assign_gateways(plan, L)
    stats = push_plan_stats(pairs, L)
    assert stats == push_plan_stats(pairs, L)  # deterministic
    # independent recomputation from the plan itself
    cross = [
        (d, int(plan.experts_to_copy[d, b]))
        for d in range(R)
        for b in range(cfg.B)
        if int(plan.experts_to_copy[d, b]) >= 0
        and (int(plan.experts_to_copy[d, b]) // cfg.epn) // L != d // L
    ]
    groups = {}
    for d, e in cross:
        groups[(e, d // L)] = groups.get((e, d // L), 0) + 1
    assert stats["n_cross_legs"] == len(cross)
    assert stats["n_cross_groups"] == len(groups)
    assert stats["n_multi_groups"] == sum(1 for v in groups.values() if v > 1)
    assert stats["max_fan"] == (max(groups.values()) if groups else 0)
    # gateway legs exist exactly when a multi group exists (the F-C auto rule)
    has_gw = any(int(p[4]) >= 0 for p in pairs.tolist())
    assert has_gw == (stats["n_multi_groups"] > 0)
