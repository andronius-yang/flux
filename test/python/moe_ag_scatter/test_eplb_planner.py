"""EPLB-arm acceptance tests (CPU tier only — the arm has no GPU oracle:
the vendored eplb.py IS the production algorithm).

Covers, over a named case battery + fuzz:
  * vendored-file pinning: the README example of deepseek-ai/EPLB reproduces
    DeepSeek's own documented phy2log (guards the verbatim copy and the
    torch.sort behavior the algorithm depends on),
  * structural legality of the mapped plan (p2l/l2p/lcnts consistency, slot
    budget, replica-rank column order; deliberately NO master-pinning or
    one-copy-per-rank asserts — EPLB does full re-placement and may
    co-locate instances),
  * hier-policy node confinement,
  * quota conservation + the equal-split (largest-remainder, extras at the
    lowest instance index) property, and rank-quota per-source conservation,
  * reroute conservation (reusing the ultraep-port helper verbatim — the
    reroute machinery is shared),
  * placement/quota separation: placement depends only on pool_load, quotas
    only on the batch tpe,
  * determinism (double-build and JSON round-trip of the load vector give
    identical plan hashes),
  * weight_placement_pairs / predicted_rows_per_rank sanity.

Run: pytest test/python/moe_ag_scatter/test_eplb_planner.py -q
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eplb_oracle import rebalance_experts
from flux.testing.eplb_semantics import (
    EPLB_POLICIES,
    build_eplb_plan,
    predicted_rows_per_rank,
    weight_placement_pairs,
)
from flux.testing.ultraep_semantics import loads_from_topk

from test_ultraep_planner import (
    Case,
    assert_reroute_conservation,
    make_cfg,
    make_topk,
)

# Case battery: reuse the ultraep Case shapes (solver knobs are ignored by
# the EPLB builder). alpha skews the BATCH routing; the pool load is derived
# per pool_mode below.
NAMED_CASES = [
    Case("pm_uniform", alpha=0.0),
    Case("pm_mild", alpha=0.6),
    Case("pm_skewed", alpha=1.2),
    Case("pm_hot_expert", alpha=2.0, seed=5),
    Case("pm_hot_rank", alpha=0.4, hot_rank=2, seed=9),
    Case("pm_no_interleave", alpha=0.8, interleave=False),
    Case("pm_no_redundant", alpha=0.8, R_red=0),
    Case("pm_rred4", alpha=1.2, R_red=4),
    Case("single_node", G=32, R=4, D=4, alpha=1.0),
    Case("tiny", S=64, K=2, G=16, R=4, D=4, alpha=1.5),
]
FUZZ_CASES = [
    Case(f"fuzz{i}", alpha=0.2 + (i % 5) * 0.35, seed=100 + i)
    for i in range(8)
]
ALL_CASES = NAMED_CASES + FUZZ_CASES

POOL_MODES = ("batch", "zipf", "zeros")


def make_pool_load(case: Case, cfg, tpe, mode: str) -> torch.Tensor:
    """Pool-predicted load: equal to the batch ('batch' = the self-oracle
    ceiling), an independent skew ('zipf' = pool != batch mismatch), or a
    vector with hard zero-load experts ('zeros')."""
    if mode == "batch":
        return tpe.long().sum(0).double()
    gen = torch.Generator().manual_seed(case.seed + 7777)
    w = (torch.arange(cfg.G, dtype=torch.float64) + 1.0) ** (-1.1)
    w = w[torch.randperm(cfg.G, generator=gen)] * 10000.0
    if mode == "zeros":
        w[torch.randperm(cfg.G, generator=gen)[:cfg.G // 4]] = 0.0
    return w


def build(case: Case, policy: str, pool_mode: str, num_nodes: int = None):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    pool_load = make_pool_load(case, cfg, tpe, pool_mode)
    if num_nodes is None:
        num_nodes = max(cfg.R // cfg.D, 1)
    plan = build_eplb_plan(cfg, tpe, pool_load, policy, num_nodes,
                           rebalance_experts)
    return cfg, topk_all, tpe, pool_load, plan, num_nodes


# ---------------------------------------------------------------------------
# Vendored-file pinning: DeepSeek's own README example, documented output
# ---------------------------------------------------------------------------


def test_vendored_readme_example():
    weight = torch.tensor([
        [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
        [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
    ])
    phy2log, log2phy, logcnt = rebalance_experts(
        weight, 16, num_groups=4, num_nodes=2, num_gpus=8
    )
    expected = torch.tensor([
        [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
        [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1],
    ])
    # DeepSeek's documented output (EPLB README). A mismatch means the
    # vendored copy drifted or torch.sort semantics changed on this stack.
    assert torch.equal(phy2log, expected), (
        "vendored eplb.py no longer reproduces the README example "
        f"(torch {torch.__version__})"
    )
    assert torch.equal(logcnt.sum(dim=-1), torch.tensor([16, 16]))
    # log2phy inverts phy2log with column index == replica rank
    for layer in range(2):
        for l in range(12):
            C = int(logcnt[layer, l])
            phys = log2phy[layer, l, :C]
            assert bool((phys >= 0).all())
            assert bool((phy2log[layer][phys.long()] == l).all())
            assert bool((log2phy[layer, l, C:] < 0).all())


# ---------------------------------------------------------------------------
# Structural legality of the mapped plan
# ---------------------------------------------------------------------------


def assert_eplb_legal(cfg, plan, policy, num_nodes):
    p2l = plan.p2l.long()
    l2p = plan.l2p.long()
    lcnts = plan.lcnts.long()
    assert plan.l2p.shape == (cfg.G, cfg.max_replicas_dim)
    assert cfg.max_replicas_dim >= 1 + cfg.R * cfg.R_red or cfg.R_red == 0
    # every physical slot is used (EPLB fills all P slots)
    assert bool((p2l >= 0).all()) and bool((p2l < cfg.G).all())
    assert int(lcnts.sum()) == cfg.P
    counted = torch.bincount(p2l, minlength=cfg.G)
    assert torch.equal(counted, lcnts)
    ranks_per_node = cfg.R // num_nodes
    for l in range(cfg.G):
        C = int(lcnts[l])
        assert C >= 1
        assert bool((l2p[l, :C] >= 0).all())
        assert bool((l2p[l, C:] == -1).all())
        hosts = set()
        for j in range(C):
            phys = int(l2p[l, j])
            assert int(p2l[phys]) == l
            hosts.add(phys // cfg.nlp)
        # NOTE: no one-copy-per-rank assert (co-location is legal EPLB
        # output) and no master-slot pinning (full re-placement).
        if policy == "hier":
            nodes = {h // ranks_per_node for h in hosts}
            assert len(nodes) == 1, (
                f"hier policy: expert {l} instances span nodes {nodes}"
            )


@pytest.mark.parametrize("pool_mode", POOL_MODES)
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_global_policy_invariants(case, pool_mode):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case, "global", pool_mode)
    assert_eplb_legal(cfg, plan, "global", nn)
    assert_equal_split_quota(cfg, plan, tpe)
    assert_reroute_conservation(cfg, plan, topk_all)
    assert_movement_facts(cfg, plan, pool_load)


@pytest.mark.parametrize("case",
                         [c for c in NAMED_CASES if c.R % 4 == 0 and c.R > 4],
                         ids=lambda c: c.name)
def test_hier_policy_invariants(case):
    cfg, topk_all, tpe, pool_load, plan, nn = build(case, "hier", "zipf",
                                                    num_nodes=4)
    assert_eplb_legal(cfg, plan, "hier", 4)
    assert_equal_split_quota(cfg, plan, tpe)
    assert_reroute_conservation(cfg, plan, topk_all)


# ---------------------------------------------------------------------------
# Quota: equal split of the batch load, largest-remainder
# ---------------------------------------------------------------------------


def assert_equal_split_quota(cfg, plan, tpe):
    loads = tpe.long().sum(0)
    for l in range(cfg.G):
        C = int(plan.lcnts[l])
        q = plan.quota[l, :C].long()
        assert int(q.sum()) == int(loads[l])
        assert bool((plan.quota[l, C:] == 0).all())
        assert int(plan.quota_prefix[l, max(C - 1, 0)]) == int(loads[l])
        # equal split: entries differ by <= 1, extras at the lowest indices
        if C > 1:
            assert int(q.max()) - int(q.min()) <= 1
            assert bool((q[:-1] >= q[1:]).all())
        for src in range(cfg.R):
            assert (int(plan.rank_quota_prefix[src, l, max(C - 1, 0)])
                    == int(tpe[src, l]))


def test_placement_from_pool_quota_from_batch():
    """Same pool -> identical placement even for different batches; the
    quotas track the batch."""
    case_a = Case("sep_a", alpha=0.9, seed=41)
    case_b = Case("sep_b", alpha=0.9, seed=42)
    cfg_a = make_cfg(case_a)
    cfg_b = make_cfg(case_b)
    pool = make_pool_load(case_a, cfg_a, None, "zipf")
    tpe_a = loads_from_topk(cfg_a, make_topk(case_a))
    tpe_b = loads_from_topk(cfg_b, make_topk(case_b))
    plan_a = build_eplb_plan(cfg_a, tpe_a, pool, "global", 4,
                             rebalance_experts)
    plan_b = build_eplb_plan(cfg_b, tpe_b, pool, "global", 4,
                             rebalance_experts)
    for name in ("p2l", "l2p", "lcnts"):
        assert torch.equal(getattr(plan_a, name), getattr(plan_b, name))
    assert not torch.equal(plan_a.quota, plan_b.quota)


# ---------------------------------------------------------------------------
# Determinism (replicated-planning contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", NAMED_CASES[:4], ids=lambda c: c.name)
def test_determinism_and_json_roundtrip(case):
    cfg, _, tpe, pool_load, plan, nn = build(case, "global", "zipf")
    cfg2 = make_cfg(case)
    plan2 = build_eplb_plan(cfg2, tpe, pool_load, "global", nn,
                            rebalance_experts)
    assert plan.plan_hash() == plan2.plan_hash()
    # the load vector travels as JSON floats (the .eplb_load.json sidecar)
    rt = json.loads(json.dumps({"load": pool_load.tolist()}))["load"]
    cfg3 = make_cfg(case)
    plan3 = build_eplb_plan(cfg3, tpe, rt, "global", nn, rebalance_experts)
    assert plan.plan_hash() == plan3.plan_hash()


# ---------------------------------------------------------------------------
# One-time movement + predicted-balance facts
# ---------------------------------------------------------------------------


def assert_movement_facts(cfg, plan, pool_load):
    pairs = weight_placement_pairs(plan)
    p2l = plan.p2l.long()
    seen_slots = []
    for host, b, l, home in pairs:
        assert 0 <= host < cfg.R and 0 <= b < cfg.nlp
        assert int(p2l[host * cfg.nlp + b]) == l
        assert home == l // cfg.epn
        assert host != home
        seen_slots.append(host * cfg.nlp + b)
    # globally ordered by physical slot (batched-P2P matching requirement)
    assert seen_slots == sorted(seen_slots)
    moved = {host * cfg.nlp + b for host, b, l, home in pairs}
    for p in range(cfg.P):
        stayed = (p // cfg.nlp) == (int(p2l[p]) // cfg.epn)
        assert (p not in moved) == stayed

    rows = predicted_rows_per_rank(plan, pool_load)
    assert len(rows) == cfg.R
    assert abs(sum(rows) - float(torch.as_tensor(pool_load,
                                                 dtype=torch.float64).sum())
               ) < 1e-6


def test_producer_weighting():
    """sweeps/gen_trace_routing.combine_pool_loads: the load vector mirrors
    the sampler's expected batch mix per sem."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.join(repo_root, "sweeps"))
    from gen_trace_routing import combine_pool_loads, pool_histogram

    rows_a = [(0, 1), (0, 2), (1, 2)]            # 3 rows
    rows_b = [(3, 0)] * 6                        # 6 rows, different skew
    G = 4
    ca, cb = pool_histogram(rows_a, G), pool_histogram(rows_b, G)
    assert ca == [2, 2, 2, 0] and cb == [6, 0, 0, 6]

    load, w = combine_pool_loads([ca], [3], "homog")
    assert w == "homog" and load == [2.0, 2.0, 2.0, 0.0]

    load, w = combine_pool_loads([ca, cb], [3, 6], "pernode")
    assert w == "pernode_equal_mix"
    # each pool normalized by its row count, then averaged: pool sizes must
    # NOT leak into the mix (every node contributes the same token count)
    expect = [(2 / 3 + 6 / 6) / 2, (2 / 3) / 2, (2 / 3) / 2, (6 / 6) / 2]
    assert all(abs(x - y) < 1e-12 for x, y in zip(load, expect))

    load, w = combine_pool_loads([ca, cb], [3, 6], "mixed")
    assert w == "mixed_concat" and load == [8.0, 2.0, 2.0, 6.0]


def test_pool_oracle_balances_prediction():
    """With pool == batch and redundancy, the PREDICTED per-rank load under
    the EPLB placement must beat the fixed-placement imbalance (this is the
    algorithm's whole objective)."""
    case = Case("balance", alpha=1.2, seed=3)
    cfg, _, tpe, pool_load, plan, _ = build(case, "global", "batch")
    pred = predicted_rows_per_rank(plan, pool_load)
    mean = sum(pred) / len(pred)
    before = tpe.long().sum(0).reshape(cfg.R, cfg.epn).sum(1)
    before_ratio = float(before.max()) / float(before.double().mean())
    after_ratio = max(pred) / mean
    assert after_ratio <= before_ratio + 1e-9
    assert after_ratio >= 1.0 - 1e-9
