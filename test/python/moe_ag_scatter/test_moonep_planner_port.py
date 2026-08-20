"""Parity tier for the replicated MoonEP planner port (SCHEMA rule 5).

Two tiers:
  * CPU (always runs): derive_moonep_layout_gpu on device="cpu" reproduces
    build_comm_layout bitwise on every rank (dup pairs as a set — order is
    documented-unstable upstream) from the CPU reference plan's tensors.
  * GPU (single process, single GPU — the port has NO cross-rank sync, so
    one device covers every rank's outputs at any R): the ported CuTe
    kernel's dst/cu_seqlens/experts_to_copy/zero_fill/remote_stats are
    BITWISE-equal to compute_moonep_plan for R in {2, 4, 16}, and the
    end-to-end kernel -> derive_moonep_layout_gpu chain matches the CPU
    layouts. A mismatch here triggers the fallback ladder (raw-CUDA port,
    then torch-GPU planner) — never debug this at 4n.

Run: pytest test/python/moe_ag_scatter/test_moonep_planner_port.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.testing.moonep_semantics import (
    MoonEPConfig,
    build_comm_layout,
    check_moonep_iter_plan,
    compute_moonep_plan,
    derive_moonep_layout_gpu,
)

from test_moonep_planner import (
    PortCase,
    R16_CASES,
    _case_supported,
    make_topk_all,
)

GPU_CASES = [
    (PortCase("r2_typical", S=128, K=8, epn=8, routing="biased",
              bias_ratio=1.0, seed=31001), 2),
    (PortCase("r4_heavy", S=128, K=8, epn=8, routing="biased",
              bias_ratio=2.0, seed=31002), 4),
    (PortCase("r4_balanced", S=64, K=4, epn=8), 4),
] + [(c, 16) for c in R16_CASES[:4]]

CPU_CASES = GPU_CASES


def _build(case, R):
    cfg = MoonEPConfig(S=case.S, K=case.K, E=R * case.epn, R=R,
                       B=case.B, token_padding=case.token_padding)
    topk_all = make_topk_all(case, R)
    plan = compute_moonep_plan(cfg, topk_all.long())
    return cfg, topk_all, plan


@pytest.mark.parametrize("case,R", CPU_CASES,
                         ids=lambda p: getattr(p, "name", str(p)))
def test_layout_derivation_cpu_parity(case, R):
    if not _case_supported(case, R):
        pytest.skip("case unsupported at this R")
    cfg, topk_all, plan = _build(case, R)
    for rank in range(min(R, 6)):
        lay = build_comm_layout(plan, rank)
        ip = derive_moonep_layout_gpu(
            cfg, rank, plan.dst, plan.zero_fill_ranges, plan.cu_seqlens,
            plan.experts_to_copy)
        check_moonep_iter_plan(ip, lay, plan, rank)
        assert ip.total_rows == int(plan.cu_seqlens[rank, -1])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="ported planner kernel needs a GPU")
@pytest.mark.parametrize("case,R", GPU_CASES,
                         ids=lambda p: getattr(p, "name", str(p)))
def test_ported_kernel_matches_reference(case, R):
    if not _case_supported(case, R):
        pytest.skip("case unsupported at this R")
    from moonep_oracle.planning_port import ReplicatedPlannerWorkspace

    cfg, topk_all, plan = _build(case, R)
    dev = torch.device("cuda")
    ws = ReplicatedPlannerWorkspace(cfg, dev, num_sms=32)
    topk_flat = topk_all.reshape(R, cfg.N).to(torch.int32).to(dev)
    tpe_all = plan.tpe.to(torch.int32).to(dev)
    dst_all, cu_all, etc_all, zfr_all, stats_all = ws.launch(
        topk_flat.contiguous(), tpe_all.contiguous())
    torch.cuda.synchronize()

    assert torch.equal(dst_all.cpu(), plan.dst.to(torch.int32)), "dst"
    assert torch.equal(cu_all.cpu(), plan.cu_seqlens.to(torch.int32)), (
        "cu_seqlens")
    assert torch.equal(etc_all.cpu(),
                       plan.experts_to_copy.to(torch.int32)), (
        "experts_to_copy")
    assert torch.equal(zfr_all.cpu(),
                       plan.zero_fill_ranges.to(torch.int32)), (
        "zero_fill_ranges")
    assert torch.equal(stats_all.cpu(),
                       plan.remote_stats.to(torch.int32)), "remote_stats"

    # end-to-end: kernel outputs -> per-iteration layout == CPU layouts
    for rank in range(min(R, 4)):
        lay = build_comm_layout(plan, rank)
        ip = derive_moonep_layout_gpu(cfg, rank, dst_all, zfr_all, cu_all,
                                      etc_all)
        check_moonep_iter_plan(ip, lay, plan, rank)

    # determinism: a second launch is bitwise-identical (self-resetting
    # grid barrier + no cross-launch state)
    dst2 = dst_all.clone()
    ws.launch(topk_flat.contiguous(), tpe_all.contiguous())
    torch.cuda.synchronize()
    assert torch.equal(ws.dst_all, dst2)
