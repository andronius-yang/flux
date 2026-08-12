"""UltraEP-port acceptance tests, two tiers.

CPU tier (login node, no GPU): structural invariants of the PyTorch port
(flux.testing.ultraep_semantics) over a named case battery + fuzz — the
assert_legal contract from UltraEP tests/test_solving.py (masters at the
canonical slot, replicas confined to their master's NVLink domain, at most
one copy of a logical expert per rank, replicas only in redundant slots,
per-rank slot budgets), quota conservation (sum of instance quotas == global
expert load; final rank-quota prefix column == that source rank's load),
reroute conservation (every token expands to exactly K physical entries,
each targeting an instance of the right logical expert, per-instance counts
== the rank-quota allocations), and determinism (identical plan twice).

GPU tier (skipped without CUDA or without the ultra_ep SM80 oracle build):
BIT-EQUALITY of all 6 solver output tensors per rank_quota_source_rank and
of the reroute expansion, against UltraEP's real kernels via
ultra_ep._C.solve_placement_for_test / dense_reroute_for_test.

Fidelity is defined against UltraEP v1.0.0 @ 94cab09 (sm80-oracle branch
d6b983d, which changes no solver semantics — only TMA shims and build fixes).
A future upstream change to e.g. QUOTA_SOLVER_WARPS or the fast-eps retry
policy moves the probe ladder and breaks bit-equality by design.

Run: pytest test/python/moe_ag_scatter/test_ultraep_planner.py -q
"""

import os
import sys
from dataclasses import dataclass, field

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.testing.ultraep_semantics import (
    UltraEPConfig,
    build_rank_quota_prefix,
    loads_from_topk,
    nvl_domain_lower_bound,
    reroute_expand,
    reroute_expand_dense,
    solve_placement,
)

try:
    import ultra_ep._C as _C

    HAS_ULTRAEP = hasattr(_C, "solve_placement_for_test")
except Exception:
    HAS_ULTRAEP = False

HAS_CUDA = torch.cuda.is_available()

SOLVER_OUTPUT_NAMES = ["p2l", "l2p", "lcnts", "quota", "quota_prefix",
                       "rank_quota_prefix"]


@dataclass(frozen=True)
class Case:
    name: str
    S: int = 1024
    K: int = 8
    G: int = 128
    R: int = 16
    D: int = 4
    R_red: int = 2
    min_tokens_per_replica: int = 128
    allow_zero_master_quota: bool = False
    locality_aware: bool = True
    balance_threshold: float = 1.0
    oracle_eps: float = 0.01
    interleave: bool = True
    alpha: float = 0.8       # zipf-ish skew exponent (0 = uniform)
    seed: int = 33
    hot_rank: int = -1       # if >= 0, concentrate extra mass on that rank's experts


def make_cfg(c: Case) -> UltraEPConfig:
    return UltraEPConfig(
        S=c.S, K=c.K, G=c.G, R=c.R, H=0, D=c.D, R_red=c.R_red,
        balance_threshold=c.balance_threshold,
        min_tokens_per_replica=c.min_tokens_per_replica,
        allow_zero_master_quota=c.allow_zero_master_quota,
        locality_aware=c.locality_aware,
        oracle_eps=c.oracle_eps,
        interleave=c.interleave,
    )


def make_topk(c: Case) -> torch.Tensor:
    """Zipf-ish top-K-without-replacement routing, deterministic per case."""
    gen = torch.Generator().manual_seed(c.seed)
    if c.alpha == 0.0:
        w = torch.ones(c.G, dtype=torch.float64)
    else:
        w = (torch.arange(c.G, dtype=torch.float64) + 1.0) ** (-c.alpha)
        perm = torch.randperm(c.G, generator=gen)
        w = w[perm]
    if c.hot_rank >= 0:
        epn = c.G // c.R
        w[c.hot_rank * epn:(c.hot_rank + 1) * epn] *= 50.0
    return torch.multinomial(
        w.expand(c.R * c.S, -1), c.K, replacement=False, generator=gen
    ).reshape(c.R, c.S, c.K).int()


NAMED_CASES = [
    Case("pm_uniform", alpha=0.0),
    Case("pm_mild", alpha=0.6),
    Case("pm_skewed", alpha=1.2),
    Case("pm_hot_expert", alpha=2.0, seed=5),
    Case("pm_hot_rank", alpha=0.4, hot_rank=2, seed=9),
    Case("pm_mtpr_runtime_default", alpha=1.0, min_tokens_per_replica=1024),
    Case("pm_no_locality", alpha=0.8, locality_aware=False),
    Case("pm_zero_master", alpha=0.8, allow_zero_master_quota=True),
    Case("pm_no_interleave", alpha=0.8, interleave=False),
    Case("pm_bt_eps", alpha=0.8, balance_threshold=1.2, oracle_eps=0.05),
    Case("pm_no_redundant", alpha=0.8, R_red=0),
    Case("pm_rred4", alpha=1.2, R_red=4),
    Case("domain16", alpha=1.2, D=16, R_red=4, min_tokens_per_replica=64),
    Case("domain16_deep_replicas", alpha=2.5, D=16, R_red=8,
         min_tokens_per_replica=16, seed=17),   # drives C > 8 slow path
    Case("single_domain", G=32, R=4, D=4, alpha=1.0),
    Case("tiny", S=64, K=2, G=16, R=4, D=4, min_tokens_per_replica=4,
         alpha=1.5),
]

FUZZ_CASES = [
    Case(f"fuzz{i}", alpha=0.2 + (i % 5) * 0.35, seed=100 + i,
         min_tokens_per_replica=(64 if i % 3 else 1024))
    for i in range(12)
]

ALL_CASES = NAMED_CASES + FUZZ_CASES


# ---------------------------------------------------------------------------
# CPU tier
# ---------------------------------------------------------------------------


def assert_legal(cfg, plan):
    """Port of UltraEP tests/test_solving.py assert_legal."""
    l2p = plan.l2p.long()
    p2l = plan.p2l.long()
    lcnts = plan.lcnts.long()
    for l in range(cfg.G):
        C = int(lcnts[l])
        assert C >= 1
        # master pinned at the canonical physical slot
        master = int(l2p[l, 0])
        assert master == (l // cfg.epn) * cfg.nlp + l % cfg.epn
        assert int(p2l[master]) == l
        # lcnts consistent with valid entries; -1 padding after C
        assert bool((l2p[l, :C] >= 0).all())
        assert bool((l2p[l, C:] == -1).all())
        hosts = set()
        for j in range(C):
            phys = int(l2p[l, j])
            assert int(p2l[phys]) == l
            host = phys // cfg.nlp
            # at most one copy of a logical expert per rank
            assert host not in hosts
            hosts.add(host)
            if j > 0:
                # replicas only in redundant slots, inside the master domain
                assert phys % cfg.nlp >= cfg.epn
                assert host // cfg.D == (l // cfg.epn) // cfg.D
    # redundant slots not oversubscribed
    for r in range(cfg.R):
        used = sum(
            1 for p in range(r * cfg.nlp + cfg.epn, (r + 1) * cfg.nlp)
            if int(p2l[p]) >= 0
        )
        assert used <= cfg.R_red


def assert_quota_conservation(cfg, plan, tpe):
    loads = tpe.long().sum(0)
    for l in range(cfg.G):
        C = int(plan.lcnts[l])
        assert int(plan.quota[l, :C].sum()) == int(loads[l])
        assert int(plan.quota_prefix[l, max(C - 1, 0)]) == int(loads[l])
        assert bool((plan.quota[l, C:] == 0).all())
        for src in range(cfg.R):
            # decomposition is conservative per source rank
            assert (int(plan.rank_quota_prefix[src, l, max(C - 1, 0)])
                    == int(tpe[src, l]))
        if not cfg.allow_zero_master_quota and int(loads[l]) > 0:
            assert int(plan.quota[l, 0]) >= 1


def assert_reroute_conservation(cfg, plan, topk_all):
    p2l = plan.p2l.long()
    for src in range(cfg.R):
        tok, phys = reroute_expand(cfg, plan, src, topk_all[src])
        assert tok.numel() == cfg.S * cfg.K
        # every entry targets an instance of the logical expert the token chose
        logical = p2l[phys]
        routing = torch.zeros(cfg.S, cfg.G, dtype=torch.bool)
        routing[torch.arange(cfg.S).unsqueeze(1), topk_all[src].long()] = True
        assert bool(routing[tok, logical].all())
        # per token: exactly K entries
        assert bool((torch.bincount(tok, minlength=cfg.S) == cfg.K).all())
        # per instance: counts == rank-quota allocations exactly (the
        # interleave permutation is a bijection over [0, total))
        alloc = plan.rank_quota_prefix[src].long().clone()
        alloc[:, 1:] -= plan.rank_quota_prefix[src].long()[:, :-1]
        got = torch.zeros(cfg.P, dtype=torch.int64)
        got.scatter_add_(0, phys, torch.ones_like(phys))
        for l in range(cfg.G):
            C = int(plan.lcnts[l])
            for j in range(C):
                assert int(got[int(plan.l2p[l, j])]) == int(alloc[l, j])


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_cpu_invariants(case):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = solve_placement(cfg, tpe)

    assert_legal(cfg, plan)
    assert_quota_conservation(cfg, plan, tpe)
    assert_reroute_conservation(cfg, plan, topk_all)

    # determinism: identical plan on a second run (replicated-planning
    # contract — every rank computes the same thing)
    plan2 = solve_placement(cfg, tpe)
    assert plan.plan_hash() == plan2.plan_hash()

    # balance sanity: with redundancy and low mtpr, final imbalance must not
    # exceed initial and the floor must be respected
    rows = plan.physical_rows_per_rank()
    loads = tpe.long().sum(0)
    before = loads.reshape(cfg.R, cfg.epn).sum(1)
    mean = float(before.double().mean())
    if mean > 0:
        lb = nvl_domain_lower_bound(cfg, tpe)
        after_ratio = max(rows) / mean
        before_ratio = float(before.max()) / mean
        assert after_ratio <= before_ratio + 1e-9
        assert after_ratio >= lb - 1e-9


def test_cpu_locality_reduces_remote_traffic():
    case = Case("locality_ab", alpha=1.0)
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = solve_placement(cfg, tpe)

    def remote_tokens(locality):
        total = 0
        for src in range(cfg.R):
            rqp = build_rank_quota_prefix(
                cfg, tpe, plan.l2p, plan.lcnts, plan.quota, src,
                locality_aware=locality,
            ).long()
            alloc = rqp.clone()
            alloc[:, 1:] -= rqp[:, :-1]
            for l in range(cfg.G):
                for j in range(int(plan.lcnts[l])):
                    host = int(plan.l2p[l, j]) // cfg.nlp
                    if host != src:
                        total += int(alloc[l, j])
        return total

    with_loc = remote_tokens(True)
    without_loc = remote_tokens(False)
    assert with_loc <= without_loc


# ---------------------------------------------------------------------------
# Golden tier (CPU, no ultra_ep build needed): bit-equality vs frozen kernel
# outputs vendored under ultraep_oracle/goldens/ (see dump_goldens.py)
# ---------------------------------------------------------------------------

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ultraep_oracle", "goldens")


def _golden_cases():
    return [c for c in NAMED_CASES
            if os.path.exists(os.path.join(GOLDEN_DIR, f"{c.name}.pt"))]


@pytest.mark.skipif(not os.path.isdir(GOLDEN_DIR),
                    reason="no vendored goldens")
@pytest.mark.parametrize("case", _golden_cases(), ids=lambda c: c.name)
def test_golden_bit_equality(case):
    blob = torch.load(os.path.join(GOLDEN_DIR, f"{case.name}.pt"),
                      weights_only=False)
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = solve_placement(cfg, tpe)

    expected = {
        "p2l": plan.p2l, "l2p": plan.l2p, "lcnts": plan.lcnts,
        "quota": plan.quota, "quota_prefix": plan.quota_prefix,
        "rank_quota_prefix": plan.rank_quota_prefix,
    }
    for name, exp in expected.items():
        assert torch.equal(blob[name], exp), (
            f"{case.name}: '{name}' differs from frozen kernel golden"
        )

    for src in (0, 1):
        if src >= cfg.R:
            continue
        for interleave in (True, False):
            key = f"src{src}_il{int(interleave)}"
            if key not in blob["reroute"]:
                continue
            saved_interleave = cfg.interleave
            cfg.interleave = interleave
            exp_map = reroute_expand_dense(cfg, plan, src, topk_all[src])
            cfg.interleave = saved_interleave
            golden = blob["reroute"][key]
            got_map = torch.zeros(cfg.S, cfg.P, dtype=torch.bool)
            got_map[golden["entry_token"].long(),
                    golden["entry_phys"].long()] = True
            assert torch.equal(got_map, exp_map), (
                f"{case.name} {key}: reroute expansion differs from golden"
            )


# ---------------------------------------------------------------------------
# GPU tier: bit-equality vs the real kernels (SM80 oracle build)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (HAS_CUDA and HAS_ULTRAEP),
                    reason="needs CUDA + ultra_ep SM80 oracle build")
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_gpu_bit_equality(case):
    cfg = make_cfg(case)
    topk_all = make_topk(case)
    tpe = loads_from_topk(cfg, topk_all)
    plan = solve_placement(cfg, tpe)

    loads_gpu = tpe.long().sum(0).to(torch.int32).cuda()
    tpe_gpu = tpe.cuda()
    for src in range(cfg.R):
        out = _C.solve_placement_for_test(
            loads_gpu, tpe_gpu,
            num_ranks=cfg.R,
            num_local_master_experts=cfg.epn,
            num_local_redundant_experts=cfg.R_red,
            num_nvl_ranks=cfg.D,
            legacy_placement=False,
            balance_threshold=cfg.balance_threshold,
            min_tokens_per_replica=cfg.min_tokens_per_replica,
            # NB: the pybind default is True but the runtime default is
            # False — always pass explicitly.
            allow_zero_master_quota=cfg.allow_zero_master_quota,
            locality_aware=cfg.locality_aware,
            oracle_eps=cfg.oracle_eps,
            kernel_stage=1,
            rank_quota_source_rank=src,
        )
        expected = [plan.p2l, plan.l2p, plan.lcnts, plan.quota,
                    plan.quota_prefix, plan.rank_quota_prefix[src]]
        for name, got, exp in zip(SOLVER_OUTPUT_NAMES, out, expected):
            assert torch.equal(got.cpu(), exp), (
                f"{case.name} src={src}: solver output '{name}' differs"
            )

        if src < 2:   # reroute spot-check on two source ranks per case
            routing = torch.zeros(cfg.S, cfg.G, dtype=torch.bool)
            routing[torch.arange(cfg.S).unsqueeze(1),
                    topk_all[src].long()] = True
            gen = torch.Generator().manual_seed(1234 + src)
            probs = torch.rand(cfg.S, cfg.G, generator=gen) * routing
            exp_map, exp_probs = reroute_expand_dense(
                cfg, plan, src, topk_all[src], probs
            )
            got_probs, got_map = _C.dense_reroute_for_test(
                routing.cuda(), probs.cuda(), out[1], out[2], out[5],
                cfg.P, quota_reroute=True,
                interleave_by_rank_quota=cfg.interleave,
            )
            assert torch.equal(got_map.cpu(), exp_map), (
                f"{case.name} src={src}: expanded_routing_map differs"
            )
            assert torch.equal(got_probs.cpu(), exp_probs), (
                f"{case.name} src={src}: expanded_probs differs"
            )


@pytest.mark.skipif(not (HAS_CUDA and HAS_ULTRAEP),
                    reason="needs CUDA + ultra_ep SM80 oracle build")
def test_gpu_kernel_determinism():
    """Kernel-side determinism gate (no atomics in the solver / dense
    reroute): identical outputs on back-to-back runs."""
    case = Case("determinism", alpha=1.2)
    cfg = make_cfg(case)
    tpe = loads_from_topk(cfg, make_topk(case))
    loads_gpu = tpe.long().sum(0).to(torch.int32).cuda()
    tpe_gpu = tpe.cuda()
    kwargs = dict(
        num_ranks=cfg.R, num_local_master_experts=cfg.epn,
        num_local_redundant_experts=cfg.R_red, num_nvl_ranks=cfg.D,
        legacy_placement=False, balance_threshold=cfg.balance_threshold,
        min_tokens_per_replica=cfg.min_tokens_per_replica,
        allow_zero_master_quota=cfg.allow_zero_master_quota,
        locality_aware=cfg.locality_aware, oracle_eps=cfg.oracle_eps,
        kernel_stage=1, rank_quota_source_rank=0,
    )
    a = _C.solve_placement_for_test(loads_gpu, tpe_gpu, **kwargs)
    b = _C.solve_placement_for_test(loads_gpu, tpe_gpu, **kwargs)
    for name, x, y in zip(SOLVER_OUTPUT_NAMES, a, b):
        assert torch.equal(x, y), f"kernel nondeterminism in {name}"
