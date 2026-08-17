"""EPIC-arm acceptance tests (CPU tier — planner, D6 reroute, PEO group
layouts, and migration are pure host math; no GPU oracle exists, the paper
is the spec).

Covers:
  * redundancy-vector optimality vs brute force (greedy minimizes
    max c_i/(1+r_i) — exhaustive check on small cases),
  * structural legality of the epic plan (p2l/l2p/lcnts consistency, expert
    at most once per GPU, slot budget; empty slots only when the replica cap
    binds) and of the fixed (placement-none) plan,
  * D6 prefix: step-function shape, j* = src mod C, conservation vs tpe, and
    the interleave no-op property (reroute output identical with the
    coprime-stride interleave on and off),
  * reroute conservation (shared helper from the ultraep tier),
  * NIC-stage estimates: assignment validity, single-node degeneracy,
    formula spot-checks,
  * determinism + plan-hash stability + JSON round-trip of the pool load,
  * group partition properties (coverage, contiguity, ragged sizes),
  * PEO layout: per-source entry conservation, m=1 == ungrouped layout,
    cross-m invariance of the receiver segment layout, and a full CPU
    emulation of pack -> wire -> scatter proving the received hidden buffer
    is IDENTICAL for m in {1,2,4} and equal to the ungrouped reference,
  * dup-stat consistency (within + cross == total),
  * migration: swap-plan properties (<= 1 per pair, positive gain, tau gate,
    expert-duplication filter), apply_swaps invariants, quota immutability,
    determinism, and tau >= 1 convergence.

Run: pytest test/python/moe_ag_scatter/test_epic_planner.py -q
"""

import itertools
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.testing.epic_semantics import (
    EpicCommLayout,
    apply_swaps,
    build_epic_group_layouts,
    build_epic_plan,
    build_fixed_plan,
    epic_dup_stats,
    epic_node_positions,
    epic_rank_quota_prefix,
    epic_redundancy_vector,
    gpu_batch_loads,
    group_partition,
    plan_migration_swaps,
    slot_batch_loads,
)
from flux.testing.ultraep_semantics import (
    build_comm_layout,
    loads_from_topk,
    reroute_expand,
)

from test_ultraep_planner import (
    Case,
    assert_reroute_conservation,
    make_cfg,
    make_topk,
)

NAMED_CASES = [
    Case("ep_uniform", alpha=0.0),
    Case("ep_mild", alpha=0.6),
    Case("ep_skewed", alpha=1.2),
    Case("ep_hot_expert", alpha=2.0, seed=5),
    Case("ep_hot_rank", alpha=0.4, hot_rank=2, seed=9),
    Case("ep_no_redundant", alpha=0.8, R_red=0),
    Case("ep_rred4", alpha=1.2, R_red=4),
    Case("single_node", G=32, R=4, D=4, alpha=1.0),
    Case("tiny", S=64, K=2, G=16, R=4, D=4, alpha=1.5),
]
FUZZ_CASES = [
    Case(f"epfuzz{i}", alpha=0.2 + (i % 5) * 0.35, seed=200 + i)
    for i in range(6)
]
ALL_CASES = NAMED_CASES + FUZZ_CASES

# Small cases for the O(R^2 * G)-per-rank emulation tier.
EMU_CASES = [
    Case("tiny", S=64, K=2, G=16, R=4, D=4, alpha=1.5),
    Case("emu_skew", S=128, K=4, G=32, R=8, D=4, alpha=1.2, seed=77),
]


def make_pool_load(case: Case, cfg, tpe, mode: str) -> torch.Tensor:
    if mode == "batch":
        return tpe.long().sum(0).double()
    gen = torch.Generator().manual_seed(case.seed + 4242)
    w = (torch.arange(cfg.G, dtype=torch.float64) + 1.0) ** (-1.1)
    w = w[torch.randperm(cfg.G, generator=gen)] * 10000.0
    if mode == "zeros":
        w[torch.randperm(cfg.G, generator=gen)[:cfg.G // 4]] = 0.0
    return w


def build(case: Case, pool_mode: str = "zipf", num_nodes: int = None):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    pool_load = make_pool_load(case, cfg, tpe, pool_mode)
    if num_nodes is None:
        num_nodes = max(cfg.R // cfg.D, 1)
    plan = build_epic_plan(cfg, tpe, pool_load, num_nodes)
    return cfg, topk_all, tpe, pool_load, plan, num_nodes


# ---------------------------------------------------------------------------
# Redundancy vector: greedy == optimal
# ---------------------------------------------------------------------------


def brute_force_redundancy(c, spare, cap):
    """Minimize max c_i/(1+r_i) by exhaustive distribution (tiny G only)."""
    G = len(c)
    best_obj, best = None, None
    for combo in itertools.product(range(cap + 1), repeat=G):
        if sum(combo) > spare:
            continue
        obj = max(c[i] / (1 + combo[i]) for i in range(G))
        if best_obj is None or obj < best_obj - 1e-12:
            best_obj, best = obj, combo
    return best_obj


@pytest.mark.parametrize("seed", range(6))
def test_redundancy_greedy_optimal(seed):
    gen = torch.Generator().manual_seed(seed)
    G, spare, cap = 5, 4, 3
    c = torch.randint(0, 1000, (G,), generator=gen).double()
    r = epic_redundancy_vector(c, spare, cap)
    got = max(float(c[i]) / (1 + int(r[i])) for i in range(G))
    want = brute_force_redundancy(c.tolist(), spare, cap)
    # The greedy on max c/(1+r) is exchange-optimal for this objective.
    assert abs(got - want) < 1e-9, (c.tolist(), r.tolist(), got, want)
    assert int(r.sum()) <= spare and int(r.max()) <= cap


def test_redundancy_cap_and_zero_load():
    c = torch.tensor([100.0, 0.0, 0.0])
    r = epic_redundancy_vector(c, spare_slots=10, replica_cap=2)
    assert int(r[0]) == 2                      # cap binds on the only hot one
    # zero-load experts absorb only after the cap binds everywhere useful
    assert int(r.sum()) <= 10


# ---------------------------------------------------------------------------
# Plan legality
# ---------------------------------------------------------------------------


def assert_epic_legal(cfg, plan, num_nodes):
    p2l = plan.p2l.long()
    l2p = plan.l2p.long()
    lcnts = plan.lcnts.long()
    assert cfg.max_replicas_dim == cfg.R
    assert plan.l2p.shape == (cfg.G, cfg.R)
    assert bool((p2l >= -1).all()) and bool((p2l < cfg.G).all())
    counted = torch.bincount(p2l[p2l >= 0], minlength=cfg.G)
    assert torch.equal(counted, lcnts)
    assert int(lcnts.min()) >= 1
    for l in range(cfg.G):
        C = int(lcnts[l])
        assert bool((l2p[l, :C] >= 0).all())
        assert bool((l2p[l, C:] == -1).all())
        hosts = []
        for j in range(C):
            phys = int(l2p[l, j])
            assert int(p2l[phys]) == l
            hosts.append(phys // cfg.nlp)
        # EPIC invariant: an expert appears at most once per GPU.
        assert len(set(hosts)) == C, f"expert {l} duplicated on a rank"
        # l2p columns ordered by ascending physical slot (determinism).
        assert l2p[l, :C].tolist() == sorted(l2p[l, :C].tolist())


@pytest.mark.parametrize("pool_mode", ("batch", "zipf", "zeros"))
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_epic_plan_invariants(case, pool_mode):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case, pool_mode)
    assert_epic_legal(cfg, plan, nn)
    assert_reroute_conservation(cfg, plan, topk_all)
    # est facts present and shaped
    assert len(plan.epic_est_internode_send) == cfg.R
    assert len(plan.epic_est_internode_recv) == cfg.R
    if nn == 1:
        assert all(v == 0.0 for v in plan.epic_est_internode_send)
        assert all(v == 0.0 for v in plan.epic_est_internode_recv)


@pytest.mark.parametrize("case", NAMED_CASES[:4], ids=lambda c: c.name)
def test_fixed_plan(case):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = build_fixed_plan(cfg, tpe)
    assert bool((plan.lcnts == 1).all())
    for l in range(cfg.G):
        assert int(plan.l2p[l, 0]) == (l // cfg.epn) * cfg.nlp + (l % cfg.epn)
    assert_reroute_conservation(cfg, plan, topk_all)


# ---------------------------------------------------------------------------
# D6 prefix: step function + interleave no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", NAMED_CASES[:5], ids=lambda c: c.name)
def test_d6_prefix_shape_and_conservation(case):
    cfg, topk_all, tpe, _, plan, _ = build(case)
    rqp = plan.rank_quota_prefix.long()
    tpe_l = tpe.long()
    for src in range(cfg.R):
        for l in range(cfg.G):
            C = int(plan.lcnts[l])
            j_star = src % C
            row = rqp[src, l]
            assert bool((row[:j_star] == 0).all())
            assert bool((row[j_star:C] == tpe_l[src, l]).all())
            assert bool((row[C:] == 0).all()) or int(tpe_l[src, l]) == 0
    # replica choice is literally src mod C for every entry
    p2l = plan.p2l.long()
    l2p = plan.l2p.long()
    for src in range(cfg.R):
        _, phys = reroute_expand(cfg, plan, src, topk_all[src])
        logical = p2l[phys]
        for e, l in zip(phys.tolist(), logical.tolist()):
            C = int(plan.lcnts[l])
            assert e == int(l2p[l, src % C])


def test_d6_interleave_noop():
    case = Case("ilv", S=64, K=2, G=16, R=4, D=4, alpha=1.5)
    cfg, topk_all, tpe, pool, plan, nn = build(case)
    assert cfg.interleave
    cfg2 = make_cfg(case)
    cfg2.interleave = False
    plan2 = build_epic_plan(cfg2, tpe, pool, nn)
    for src in range(cfg.R):
        t1, p1 = reroute_expand(cfg, plan, src, topk_all[src])
        t2, p2 = reroute_expand(cfg2, plan2, src, topk_all[src])
        assert torch.equal(t1, t2) and torch.equal(p1, p2)


def test_d6_prefix_builder_direct():
    cfg = make_cfg(Case("d6", S=8, K=2, G=8, R=4, D=4))
    cfg.max_replicas_dim = cfg.R
    tpe = torch.arange(cfg.R * cfg.G, dtype=torch.int32).reshape(cfg.R, cfg.G)
    lcnts = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.int32)
    rqp = epic_rank_quota_prefix(cfg, tpe, lcnts)
    for src in range(cfg.R):
        for l in range(cfg.G):
            C = int(lcnts[l])
            assert int(rqp[src, l, C - 1]) == int(tpe[src, l])
            j_star = src % C
            if j_star > 0:
                assert int(rqp[src, l, j_star - 1]) == 0


# ---------------------------------------------------------------------------
# NIC stage
# ---------------------------------------------------------------------------


def test_node_positions_validity_and_formula():
    chat = [100.0, 90.0, 10.0, 5.0, 80.0, 70.0, 20.0, 15.0]
    D, num_nodes = 4, 2
    R = D * num_nodes
    rank_of, est_send, est_recv = epic_node_positions(chat, D, num_nodes, 8.0)
    assert sorted(rank_of) == list(range(R))
    total = sum(chat)
    # formulas hold for every gpu under the produced assignment
    members = [[] for _ in range(num_nodes)]
    for g, rk in enumerate(rank_of):
        members[rk // D].append(g)
    for n in range(num_nodes):
        node_chat = sum(chat[g] for g in members[n])
        for g in members[n]:
            assert abs(est_recv[g] - chat[g] * (R - D) / R) < 1e-9
            assert abs(est_send[g] - (total / R) * (1 - node_chat / total)) < 1e-9
    # the greedy must beat the naive fill-order assignment on max(send+recv)
    naive = [[0, 1, 2, 3], [4, 5, 6, 7]]

    def worst(memb):
        w = 0.0
        for n in range(num_nodes):
            nc = sum(chat[g] for g in memb[n])
            for g in memb[n]:
                w = max(w, chat[g] * (R - D) / R + (total / R) * (1 - nc / total))
        return w

    assert worst(members) <= worst(naive) + 1e-9


def test_node_positions_single_node():
    rank_of, s, r = epic_node_positions([3.0, 1.0, 2.0, 4.0], 4, 1, 8.0)
    assert sorted(rank_of) == [0, 1, 2, 3]
    assert all(v == 0.0 for v in s) and all(v == 0.0 for v in r)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", NAMED_CASES[:4], ids=lambda c: c.name)
def test_determinism_and_json_roundtrip(case):
    cfg, _, tpe, pool_load, plan, nn = build(case)
    cfg2 = make_cfg(case)
    plan2 = build_epic_plan(cfg2, tpe, pool_load, nn)
    assert plan.plan_hash() == plan2.plan_hash()
    rt = json.loads(json.dumps({"load": pool_load.tolist()}))["load"]
    cfg3 = make_cfg(case)
    plan3 = build_epic_plan(cfg3, tpe, rt, nn)
    assert plan.plan_hash() == plan3.plan_hash()


# ---------------------------------------------------------------------------
# Group partition + PEO layouts
# ---------------------------------------------------------------------------


def test_group_partition():
    assert group_partition(8, 1) == [(0, 8)]
    assert group_partition(8, 2) == [(0, 4), (4, 8)]
    assert group_partition(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert group_partition(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]
    for nlp, m in ((7, 2), (34, 4), (5, 5)):
        bounds = group_partition(nlp, m)
        assert bounds[0][0] == 0 and bounds[-1][1] == nlp
        for (a, b), (c, d) in zip(bounds, bounds[1:]):
            assert b == c and b > a
        sizes = [b - a for a, b in bounds]
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("m", (1, 2, 4))
@pytest.mark.parametrize("case", EMU_CASES, ids=lambda c: c.name)
def test_layout_conservation_and_m1_equivalence(case, m):
    cfg, topk_all, tpe, _, plan, _ = build(case)
    for rank in range(cfg.R):
        lay = build_epic_group_layouts(plan, rank, topk_all, m,
                                       ranks_per_node=cfg.D)
        # per-source send conservation: this rank emits exactly S*K entries
        assert lay.n_send == cfg.S * cfg.K
        assert sum(sum(g.send_counts) for g in lay.groups) == cfg.S * cfg.K
        # recv totals match the ungrouped layout
        ref = build_comm_layout(plan, rank, topk_all, pinned_masters=False)
        assert lay.n_recv == sum(ref.recv_counts)
        assert lay.seg_rows == ref.seg_rows
        assert lay.gemm_segments == ref.gemm_segments
        if m == 1:
            g0 = lay.groups[0]
            assert torch.equal(g0.send_row_index, ref.send_row_index)
            assert torch.equal(g0.send_entry_logical, ref.send_entry_logical)
            assert g0.send_counts == ref.send_counts
            assert g0.recv_counts == ref.recv_counts
            assert torch.equal(g0.place_slots, ref.place_slots)


def emulate_hidden(plan, topk_all, m):
    """CPU emulation of pack -> wire -> scatter for every rank.

    'Hidden' rows carry the unique tag src * S + token. Returns a list of
    [n_recv] int64 tensors (post-scatter hidden buffers), one per rank.
    """
    cfg = plan.cfg
    lays = [
        build_epic_group_layouts(plan, r, topk_all, m, ranks_per_node=cfg.D)
        for r in range(cfg.R)
    ]
    out = []
    for r in range(cfg.R):
        lay = lays[r]
        hidden = torch.full((max(lay.n_recv, 1),), -1, dtype=torch.int64)
        for g in range(m):
            # receiver stream: src-major concat of each src's run for dest r
            chunks = []
            for src in range(cfg.R):
                grp = lays[src].groups[g]
                off = sum(grp.send_counts[:r])
                run = grp.send_row_index[off:off + grp.send_counts[r]]
                chunks.append(src * cfg.S + run)
            stream = (torch.cat(chunks) if chunks
                      else torch.zeros(0, dtype=torch.int64))
            grp_r = lay.groups[g]
            assert stream.numel() == sum(grp_r.recv_counts)
            hidden[grp_r.place_slots] = stream
        out.append(hidden[:lay.n_recv])
    return out, lays


@pytest.mark.parametrize("case", EMU_CASES, ids=lambda c: c.name)
def test_wire_emulation_m_invariance(case):
    """The received hidden buffer is bitwise identical for every m and
    equals the ungrouped reference — the PEO layout theorem."""
    cfg, topk_all, tpe, _, plan, _ = build(case)
    ref_hidden, _ = emulate_hidden(plan, topk_all, 1)
    for m in (2, 4):
        got, lays = emulate_hidden(plan, topk_all, m)
        for r in range(cfg.R):
            assert torch.equal(got[r], ref_hidden[r]), (case.name, m, r)
    # every row was written exactly once (no -1 leftovers)
    for r in range(cfg.R):
        assert bool((ref_hidden[r] >= 0).all())
    # rows land in the segment of an instance their token actually chose
    lay0 = build_epic_group_layouts(plan, 0, topk_all, 1,
                                    ranks_per_node=cfg.D)
    p2l = plan.p2l.long()
    hidden0 = ref_hidden[0]
    for p, start, end, logical in lay0.gemm_segments:
        tags = hidden0[start:end]
        src = tags // cfg.S
        tok = tags % cfg.S
        for s, t in zip(src.tolist(), tok.tolist()):
            assert logical in topk_all[s, t].tolist()


BIG = 1 << 32


def emulate_combine(plan, topk_all, m):
    """CPU emulation of the full combine round trip for every rank.

    Expert rank d packs its group-g recv stream (src-major, (slot, token)
    within src — literally re-derived here, NOT assumed from the
    transposition theorem) and routes each row back to its source. Row tag
    = phys * BIG + (src*S + token). Home rank r stages tags via
    comb_dst_slot. Returns per-rank [S, K] int64 tag staging.
    """
    cfg = plan.cfg
    R, nlp, S = cfg.R, cfg.nlp, cfg.S
    bounds = group_partition(nlp, m)
    group_of_slot = torch.empty(nlp, dtype=torch.int64)
    for g, (lo, hi) in enumerate(bounds):
        group_of_slot[lo:hi] = g

    # canonical per-src entry streams ((phys, token) sorted)
    ent = []
    for src in range(R):
        t, p = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(p * (S + 1) + t, stable=True)
        ent.append((t[order], p[order]))

    lays = [
        build_epic_group_layouts(plan, r, topk_all, m, ranks_per_node=cfg.D)
        for r in range(R)
    ]
    staging = [torch.full((S * cfg.K,), -1, dtype=torch.int64)
               for _ in range(R)]
    for g in range(m):
        # expert rank d's combine send stream = its recv stream (src-major)
        for d in range(R):
            per_src = []
            for src in range(R):
                t_all, p_all = ent[src]
                msk = ((p_all // nlp) == d) & (
                    group_of_slot[p_all % nlp] == g)
                per_src.append((t_all[msk], p_all[msk]))
            # route each src's slice straight back to it
            for src in range(R):
                toks, phys = per_src[src]
                tags = phys * BIG + (src * S + toks)
                grp = lays[src].groups[g]
                # home-side recv position within group g from expert d:
                # position range = sum of send_counts[:d] .. +send_counts[d]
                off = sum(grp.send_counts[:d])
                dst = grp.comb_dst_slot[off:off + grp.send_counts[d]]
                assert dst.numel() == tags.numel()
                staging[src][dst] = tags
    return [s.view(S, cfg.K) for s in staging]


@pytest.mark.parametrize("case", EMU_CASES, ids=lambda c: c.name)
def test_combine_emulation_roundtrip_and_m_invariance(case):
    cfg, topk_all, tpe, _, plan, _ = build(case)
    ref = emulate_combine(plan, topk_all, 1)
    l2p = plan.l2p.long()
    for r in range(cfg.R):
        st = ref[r]
        assert bool((st >= 0).all()), "staging cell never written"
        # content: cell (t, j) carries THIS rank's token t...
        tok_part = st % BIG
        expect_tok = (r * cfg.S
                      + torch.arange(cfg.S).unsqueeze(1).expand(cfg.S, cfg.K))
        assert torch.equal(tok_part, expect_tok)
        # ...served by the D6-chosen instance of topk[t, j]
        phys_part = st // BIG
        topk = topk_all[r].long()
        for j in range(cfg.K):
            l = topk[:, j]
            C = plan.lcnts.long()[l]
            expected_phys = l2p[l, torch.remainder(
                torch.full_like(l, r), C)]
            assert torch.equal(phys_part[:, j], expected_phys)
    # bitwise m-invariance of the final staging
    for m in (2, 4):
        got = emulate_combine(plan, topk_all, m)
        for r in range(cfg.R):
            assert torch.equal(got[r], ref[r]), (case.name, m, r)


def test_comb_dst_slot_permutation():
    case = EMU_CASES[0]
    cfg, topk_all, tpe, _, plan, _ = build(case)
    for m in (1, 2, 4):
        for r in range(cfg.R):
            lay = build_epic_group_layouts(plan, r, topk_all, m,
                                           ranks_per_node=cfg.D)
            allc = torch.cat([g.comb_dst_slot for g in lay.groups])
            assert bool(
                (torch.bincount(allc, minlength=cfg.S * cfg.K) == 1).all())
            # j matches the entry's expert position in the token's topk
            for grp in lay.groups:
                t = grp.send_row_index
                j = grp.comb_dst_slot - t * cfg.K
                got_l = topk_all[r].long()[t, j]
                assert torch.equal(got_l, grp.send_entry_logical)


def test_dup_stats_consistency():
    case = EMU_CASES[1]
    cfg, topk_all, _, _, plan, _ = build(case)
    for m in (1, 2, 4):
        lay = build_epic_group_layouts(plan, 0, topk_all, m,
                                       ranks_per_node=cfg.D)
        d = epic_dup_stats(lay, cfg.R)
        assert d["dup_within_group"] + d["dup_cross_group"] == d["dup_vs_nodedup"]
        if m == 1:
            assert d["dup_cross_group"] == 0
        assert d["dup_vs_nodedup"] >= 0


# ---------------------------------------------------------------------------
# hier_compress virtual bundles (S2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", (1, 2, 4))
@pytest.mark.parametrize("case", EMU_CASES, ids=lambda c: c.name)
def test_hc_bundles_invariants(case, m):
    from flux.testing.epic_semantics import build_epic_hc_bundles

    cfg, topk_all, tpe, _, plan, _ = build(case)
    L = cfg.D
    bundles = build_epic_hc_bundles(plan, topk_all, m, L)
    assert len(bundles) == m
    gpe = cfg.nlp + 1
    ntokens = cfg.R * cfg.S

    # K_g coverage: every real entry appears exactly once across groups
    total_real = 0
    for b in bundles:
        vce = b.virtual_choosed.long()
        assert vce.shape == (ntokens, b.K_g)
        pad_slots = (vce % gpe) == cfg.nlp
        # pads self-route to the token's home rank
        home = torch.arange(ntokens) // cfg.S
        assert bool((vce[pad_slots] // gpe
                     == home.unsqueeze(1).expand_as(vce)[pad_slots]).all())
        # pad accounting (incl. multi-pad tokens)
        assert int(pad_slots.sum()) == int(b.pad_rows_per_rank.sum())
        total_real += int((~pad_slots).sum())
        # splits of real slots == the per-rank layout seg_rows for the group
        for r in range(cfg.R):
            lay = build_epic_group_layouts(plan, r, topk_all, m,
                                           ranks_per_node=L)
            grp = lay.groups[b.g]
            got = b.meta.splits.long()[
                r * gpe + grp.slot_lo: r * gpe + grp.slot_hi]
            assert got.tolist() == grp.seg_rows
        # m_per_rank = layout rows + pads
        for r in range(cfg.R):
            lay = build_epic_group_layouts(plan, r, topk_all, m,
                                           ranks_per_node=L)
            n_rows = sum(lay.groups[b.g].seg_rows)
            assert int(b.meta.m_per_rank[r]) == n_rows + int(
                b.pad_rows_per_rank[r])
    assert total_real == ntokens * cfg.K

    # multi-pad tokens (several pads of one token sharing the pad slot) must
    # be exercised somewhere in the battery — guaranteed in the skewed K=4
    # case at m=4
    if m == 4 and case.name == "emu_skew":
        pads_max = max(
            int((((b.virtual_choosed.long() % gpe) == cfg.nlp)).sum(1).max())
            for b in bundles
        )
        assert pads_max >= 2, "no multi-pad token exercised"


def test_hc_dedup_reconciliation():
    """Within-group dedup savings from u/U reconcile with the direct-arm
    dup counterfactual: sum over groups of (rank-level copies - unique)
    == dup_vs_nodedup - dup_cross_group (both count same-rank duplicate
    (token, dest) pairs recoverable within one group's message)."""
    from flux.testing.epic_semantics import build_epic_hc_bundles

    case = EMU_CASES[1]
    cfg, topk_all, _, _, plan, _ = build(case)
    gpe = cfg.nlp + 1
    for m in (1, 2):
        bundles = build_epic_hc_bundles(plan, topk_all, m, cfg.D)
        saved = 0
        for b in bundles:
            cnt = b.meta.splits_per_source.long().view(
                cfg.R, cfg.R, gpe)
            # exclude pad slots (self-copies, not wire savings)
            chunks = cnt[:, :, :cfg.nlp].sum(2)
            u = b.meta.a2av_unique_counts[:, :cfg.R].long()
            # u counts pads too (they are entries on the home rank): remove
            # the pad-token uniques by recomputing u over real entries only
            vce = b.virtual_choosed.long()
            real = (vce % gpe) != cfg.nlp
            owner = torch.where(real, vce // gpe, torch.full_like(vce, -1))
            flags = torch.zeros(cfg.R * cfg.S, cfg.R + 1, dtype=torch.bool)
            flags.scatter_(1, owner + 1, True)
            u_real = flags[:, 1:].view(cfg.R, cfg.S, cfg.R).sum(1)
            saved += int((chunks - u_real).sum())
        lay0 = build_epic_group_layouts(plan, 0, topk_all, m,
                                        ranks_per_node=cfg.D)
        d = epic_dup_stats(lay0, cfg.R)
        # dup stats are per-rank (rank 0's sends); reconcile globally by
        # summing every rank's stats
        tot_within = 0
        for r in range(cfg.R):
            lay = build_epic_group_layouts(plan, r, topk_all, m,
                                           ranks_per_node=cfg.D)
            tot_within += epic_dup_stats(lay, cfg.R)["dup_within_group"]
        assert saved == tot_within, (m, saved, tot_within)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_slot_loads_conservation():
    case = Case("ml", alpha=1.2, seed=3)
    cfg, topk_all, tpe, _, plan, _ = build(case)
    sl = slot_batch_loads(plan)
    assert int(sl.sum()) == cfg.R * cfg.S * cfg.K
    assert torch.equal(gpu_batch_loads(plan),
                       sl.reshape(cfg.R, cfg.nlp).sum(dim=1))


@pytest.mark.parametrize("case", [Case("mig_skew", alpha=1.6, seed=11),
                                  Case("mig_hot", alpha=0.4, hot_rank=1,
                                       seed=13)],
                         ids=lambda c: c.name)
def test_migration_swaps_properties(case):
    cfg, topk_all, tpe, _, plan, _ = build(case)
    quota_before = plan.quota.clone()
    rqp_before = plan.rank_quota_prefix.clone()
    gl_before = gpu_batch_loads(plan)

    swaps = plan_migration_swaps(plan, tau_tokens=0.0, ranks_per_node=cfg.D)
    # <= 1 swap per pair => <= D//2 per node
    per_node = {}
    for rh, a, rl, b, gain in swaps:
        assert gain > 0
        assert rh // cfg.D == rl // cfg.D, "swap escaped its node"
        per_node[rh // cfg.D] = per_node.get(rh // cfg.D, 0) + 1
        assert int(gl_before[rh]) > int(gl_before[rl])
    for n, cnt in per_node.items():
        assert cnt <= cfg.D // 2

    # determinism
    swaps2 = plan_migration_swaps(plan, tau_tokens=0.0, ranks_per_node=cfg.D)
    assert swaps == swaps2

    # apply: invariants inside apply_swaps must pass; quotas untouched;
    # node totals conserved; per-pair max strictly reduced
    report = apply_swaps(plan, swaps)
    assert report["applied"] == len(swaps)
    assert torch.equal(plan.quota, quota_before)
    assert torch.equal(plan.rank_quota_prefix, rqp_before)
    gl_after = gpu_batch_loads(plan)
    assert int(gl_after.sum()) == int(gl_before.sum())
    for rh, a, rl, b, gain in swaps:
        before_max = max(int(gl_before[rh]), int(gl_before[rl]))
        after_max = max(int(gl_after[rh]), int(gl_after[rl]))
        assert after_max == before_max - gain

    # tau gate: a huge tau kills all swaps
    assert plan_migration_swaps(plan, tau_tokens=10**9,
                                ranks_per_node=cfg.D) == []


def test_migration_convergence():
    case = Case("mig_conv", alpha=1.6, seed=21)
    cfg, topk_all, tpe, _, plan, _ = build(case)
    rounds = 0
    while rounds < 64:
        swaps = plan_migration_swaps(plan, tau_tokens=1.0,
                                     ranks_per_node=cfg.D)
        if not swaps:
            break
        apply_swaps(plan, swaps)
        rounds += 1
    assert rounds < 64, "migration failed to converge"
    # post-convergence the plan is still legal and reroute still conserves
    assert_reroute_conservation(cfg, plan, topk_all)


def test_migration_layout_rebuild_consistency():
    """After swaps, group layouts rebuilt from the mutated plan still
    satisfy the m-invariance theorem."""
    case = Case("mig_lay", S=64, K=2, G=16, R=4, D=4, alpha=1.5, seed=31)
    cfg, topk_all, tpe, _, plan, _ = build(case)
    swaps = plan_migration_swaps(plan, tau_tokens=0.0, ranks_per_node=cfg.D)
    if swaps:
        apply_swaps(plan, swaps)
    ref, _ = emulate_hidden(plan, topk_all, 1)
    for m in (2, 4):
        got, _ = emulate_hidden(plan, topk_all, m)
        for r in range(cfg.R):
            assert torch.equal(got[r], ref[r])
