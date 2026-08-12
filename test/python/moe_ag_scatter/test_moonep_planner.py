"""M0 acceptance tests: the vectorized MoonEP planner is bit-identical to
MoonEP's own oracle (moonep_oracle/planning_reference.py, vendored).

Runs single-process (no torchrun, no GPU required): with 3-D topk + 2-D tpe
inputs the oracle's only collective is the Part-3 all_gather of each rank's
canonical dst, which run_oracle_all_ranks satisfies with a two-pass shim
(pass 1 captures each rank's own dst and aborts, pass 2 serves the captured
table), so every oracle output stays self-computed. Case list = MoonEP's own
18 PLANNING_CASES (imported from the MoonEP checkout when present, else a
vendored fallback subset) + R=16 Perlmutter-shaped cases MoonEP itself never
ran + lognormal fuzz.

Run: pytest test/python/moe_ag_scatter/test_moonep_planner.py
"""

import os
import sys
from dataclasses import dataclass
from unittest import mock

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moonep_oracle.dedup_check import (
    dedup_plan_semantic_errors,
    planning_invariant_errors_lite,
)
from moonep_oracle.generate_topk_routing import generate_topk_routing
from moonep_oracle.planning_reference import launch_planning_torch_reference

from flux.testing.moonep_semantics import MoonEPConfig, compute_moonep_plan

MOONEP_ROOT = os.environ.get(
    "MOONEP_ROOT",
    "/global/u1/y/yufeid/workspace/changchen/andrewy/MoonEP",
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class PortCase:
    """Mirror of MoonEP tests/kernel_test_utils.KernelCase (planner fields)."""

    name: str
    S: int
    K: int
    epn: int
    B: int | None = None
    token_padding: int = 128
    routing: str = "balanced"
    bias_ratio: float = 0.0
    seed: int = 42
    min_R: int = 1
    max_R: int | None = None


def _load_moonep_cases():
    """MoonEP's PLANNING_CASES from the sibling checkout, as PortCases."""
    if not os.path.isdir(os.path.join(MOONEP_ROOT, "tests")):
        return None
    sys.path.insert(0, MOONEP_ROOT)
    try:
        from tests.test_planning import PLANNING_CASES
    except Exception:
        return None
    finally:
        sys.path.remove(MOONEP_ROOT)
    return [
        PortCase(
            name=c.name,
            S=c.S,
            K=c.K,
            epn=c.epn,
            B=c.B,
            token_padding=c.token_padding,
            routing=c.routing,
            bias_ratio=c.bias_ratio,
            seed=c.seed,
            min_R=c.min_R,
            max_R=c.max_R,
        )
        for c in PLANNING_CASES
    ]


# Fallback if the MoonEP checkout is absent (e.g. capsule reruns): a
# representative subset of MoonEP's PLANNING_CASES, copied verbatim.
FALLBACK_CASES = [
    PortCase("balanced_epn16", S=256, K=8, epn=16),
    PortCase("tiny_s1_k1", S=1, K=1, epn=1, token_padding=8),
    PortCase("tiny_biased_with_prefetch", S=3, K=5, epn=3, B=1, token_padding=32,
             routing="biased", bias_ratio=1.0, seed=9002, min_R=2),
    PortCase("no_padding", S=33, K=2, epn=2, B=1, token_padding=1,
             routing="biased", bias_ratio=1.0, seed=10001),
    PortCase("near_degenerate_bias", S=64, K=2, epn=5, B=3,
             routing="biased", bias_ratio=5.0, seed=12001),
    PortCase("typical_bias", S=32, K=4, epn=4, B=2, token_padding=64,
             routing="biased", bias_ratio=1.0, seed=13001),
    PortCase("all_remote_with_prefetch_slots", S=32, K=4, epn=8, B=3,
             token_padding=16, routing="all_remote", min_R=2),
    PortCase("duplicate_topk", S=16, K=4, epn=4, B=2, token_padding=8,
             routing="duplicate_topk", min_R=2),
    PortCase("single_expert_padding_tail", S=9, K=2, epn=4, B=1,
             token_padding=8, routing="single_expert"),
]

MOONEP_CASES = _load_moonep_cases()
CASES = MOONEP_CASES if MOONEP_CASES is not None else FALLBACK_CASES

# Perlmutter-shaped cases MoonEP never ran: R=16 (4 nodes x 4 ranks), the
# sweep config G=128/topk=8, and MoonEP-native epn=56 (E=896 at R=16).
R16_CASES = [
    PortCase("pm4n_g128_balanced", S=128, K=8, epn=8, min_R=16, max_R=16),
    PortCase("pm4n_g128_typical", S=128, K=8, epn=8, min_R=16, max_R=16,
             routing="biased", bias_ratio=1.0, seed=21001),
    PortCase("pm4n_g128_heavy", S=128, K=8, epn=8, min_R=16, max_R=16,
             routing="biased", bias_ratio=2.0, seed=21002),
    PortCase("pm4n_g128_degenerate", S=64, K=8, epn=8, min_R=16, max_R=16,
             routing="biased", bias_ratio=5.0, seed=21003),
    PortCase("pm4n_e896_typical", S=64, K=8, epn=56, min_R=16, max_R=16,
             routing="biased", bias_ratio=1.0, seed=21004),
    PortCase("pm4n_all_remote", S=32, K=4, epn=8, B=3, token_padding=16,
             routing="all_remote", min_R=16, max_R=16),
]

TEST_RS = (1, 2, 4, 8, 16)


def _case_supported(case: PortCase, R: int) -> bool:
    if R < case.min_R or (case.max_R is not None and R > case.max_R):
        return False
    E = R * case.epn
    if case.routing in {"balanced", "biased"} and case.K > E:
        return False
    if case.routing == "biased" and case.K > E:
        return False
    return True


def make_topk_all(case: PortCase, R: int) -> torch.Tensor:
    """All ranks' routing, replicating MoonEP tests/kernel_test_utils.make_topk."""
    E = R * case.epn
    per_rank = []
    for rank in range(R):
        if case.routing in {"balanced", "biased"}:
            bias = case.bias_ratio if case.routing == "biased" else 0.0
            topk, _ = generate_topk_routing(
                case.S, case.K, E, R, bias, DEV, case.seed, rank=rank
            )
        else:
            s = torch.arange(case.S, device=DEV)[:, None]
            k = torch.arange(case.K, device=DEV)[None, :]
            epn = case.epn
            if case.routing == "all_local":
                topk = rank * epn + ((s + k) % epn)
            elif case.routing == "all_remote":
                remote_rank = (rank + 1) % R
                topk = remote_rank * epn + ((s + k) % epn)
            elif case.routing == "single_expert":
                topk = torch.zeros((case.S, case.K), dtype=torch.long, device=DEV)
            elif case.routing == "duplicate_topk":
                expert = ((rank + 1) % R) * epn
                topk = torch.full((case.S, case.K), expert, dtype=torch.long, device=DEV)
            else:
                raise ValueError(f"unknown routing pattern: {case.routing}")
            topk = topk.to(torch.int32)
        per_rank.append(topk.cpu())
    return torch.stack(per_rank)


class _AbortOracle(Exception):
    """Raised by the pass-1 shim right after capturing a rank's dst."""


def run_oracle_all_ranks(cfg: MoonEPConfig, topk_all: torch.Tensor, tpe_2d):
    """Run the vendored oracle for every rank in one process.

    2-D tpe skips the oracle's tpe gather; the remaining collective is the
    Part-3 all_gather of canonical dst (only reached when R > 1). Pass 1
    runs the oracle per rank with an all_gather shim that captures the
    passed dst — each rank's own, oracle-computed — and aborts; pass 2
    serves the captured table so dedup/encoding complete normally.
    """
    R = cfg.R
    canonical = {}

    if R > 1:
        state = {"rank": None}

        def capture(gathered, tensor, group=None):
            canonical[state["rank"]] = tensor.clone().cpu()
            raise _AbortOracle

        with mock.patch.object(dist, "all_gather", capture):
            for rank in range(R):
                state["rank"] = rank
                try:
                    launch_planning_torch_reference(
                        cfg.oracle_ctx(rank), topk_all, tpe_2d
                    )
                except _AbortOracle:
                    pass
        assert len(canonical) == R, "oracle never reached the dst all_gather"

    def serve(gathered, tensor, group=None):
        for i in range(R):
            gathered[i].copy_(canonical[i].to(gathered[i].device))

    results = []
    with mock.patch.object(dist, "all_gather", serve):
        for rank in range(R):
            results.append(
                launch_planning_torch_reference(cfg.oracle_ctx(rank), topk_all, tpe_2d)
            )
    return results


def assert_plan_matches_oracle(cfg: MoonEPConfig, topk_all: torch.Tensor):
    plan = compute_moonep_plan(cfg, topk_all)
    R = cfg.R

    # S2: exact S*K balance on every rank, any skew.
    received = plan.alloc.sum(dim=0)
    assert torch.equal(
        received, torch.full((R,), cfg.NvS_capacity, dtype=received.dtype)
    ), f"balance invariant violated: per-rank rows {received.tolist()}"

    tpe_2d = plan.tpe  # [R, E]; 2-D tpe skips the oracle's tpe gather
    oracle_out = run_oracle_all_ranks(cfg, topk_all, tpe_2d)
    for rank in range(R):
        (
            ref_dst,
            ref_cu_seqlens,
            ref_experts_to_copy,
            ref_remote_stats,
            ref_zero_fill,
            ref_dedup,
        ) = oracle_out[rank]
        assert torch.equal(plan.dst[rank], ref_dst), (
            f"rank {rank} dst differs: "
            f"{(plan.dst[rank] != ref_dst).nonzero()[:5].flatten().tolist()}"
        )
        assert torch.equal(plan.cu_seqlens[rank], ref_cu_seqlens)
        assert torch.equal(plan.experts_to_copy, ref_experts_to_copy)
        assert torch.equal(plan.remote_stats[rank], ref_remote_stats)
        assert torch.equal(plan.zero_fill_ranges[rank], ref_zero_fill)

        errors = dedup_plan_semantic_errors(
            f"rank{rank} dedup", plan.rank_dedup_plan(rank), ref_dedup
        )
        assert not errors, "; ".join(errors[:5])

        errors = planning_invariant_errors_lite(
            cfg, plan.dst[rank], plan.cu_seqlens[rank], plan.experts_to_copy
        )
        assert not errors, "; ".join(errors[:5])

    # Determinism: same inputs -> bit-identical plan (replicated-planning
    # requirement; the distributed cross-rank hash check lives in M1).
    assert compute_moonep_plan(cfg, topk_all).plan_hash() == plan.plan_hash()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("R", TEST_RS)
def test_moonep_cases_match_oracle(case, R):
    if not _case_supported(case, R):
        pytest.skip(f"{case.name} unsupported at R={R}")
    cfg = MoonEPConfig(
        S=case.S, K=case.K, E=R * case.epn, R=R,
        B=case.B, token_padding=case.token_padding,
    )
    assert_plan_matches_oracle(cfg, make_topk_all(case, R))


@pytest.mark.parametrize("case", R16_CASES, ids=lambda c: c.name)
def test_pm4n_r16_cases_match_oracle(case):
    R = 16
    assert _case_supported(case, R)
    cfg = MoonEPConfig(
        S=case.S, K=case.K, E=R * case.epn, R=R,
        B=case.B, token_padding=case.token_padding,
    )
    assert_plan_matches_oracle(cfg, make_topk_all(case, R))


@pytest.mark.parametrize("bias_ratio", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("seed", [1234, 5678, 91011])
def test_lognormal_fuzz_r16(bias_ratio, seed):
    R, epn, S, K = 16, 8, 96, 8
    cfg = MoonEPConfig(S=S, K=K, E=R * epn, R=R)
    case = PortCase(
        f"fuzz_b{bias_ratio}_s{seed}", S=S, K=K, epn=epn,
        routing="biased", bias_ratio=bias_ratio, seed=seed,
    )
    assert_plan_matches_oracle(cfg, make_topk_all(case, R))


def test_moonep_cases_loaded_from_checkout():
    """Informational: prefer the real MoonEP case list when available."""
    if MOONEP_CASES is None:
        pytest.skip(f"MoonEP checkout not found at {MOONEP_ROOT}; used fallback cases")
    assert len(MOONEP_CASES) >= 18
