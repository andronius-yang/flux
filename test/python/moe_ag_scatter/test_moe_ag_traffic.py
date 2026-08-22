################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
"""COMET MoE layer0 (all-gather + scatter + grouped GEMM) driven by a traffic matrix.

Routing is built so that tokens homed on rank s choose experts owned by rank d for
exactly M[s][d] / chunk_bytes (token, topk-slot) copies, where chunk_bytes =
H * dtype_size. EP is fixed to ep_size == world_size so each expert's full FFN
weight resides on one rank.

NOTE on physical traffic: with --comm_pattern allgather (default), layer0
performs a dense all-gather of all token shards (fixed wire bytes independent of
the matrix), then consumes chunks[s][d] token copies per (s, d) via the local
scatter feeding the grouped GEMM — the matrix governs only the logical dispatch
and per-rank GEMM load. With --comm_pattern a2av (sm80/V2 only), each (token,
topk-slot) copy is sent directly producer -> expert-owner rank via NVSHMEM
putmem_signal, so the wire bytes s->d equal exactly M[s][d], and the grouped
GEMM claims tiles dynamically in signal-arrival order. --comm_pattern a2av_ring
moves the same M[s][d] wire bytes, but sends follow the reverse hierarchical
ring (the mirror of the allgather stage order), so the grouped GEMM keeps the
dense path's static ring-order tile schedule. --comm_pattern a2av_hier is the
hierarchical alltoallv: intra-node traffic goes direct, while all data for a
remote node travels as ONE aggregated message to the same-local-rank "gateway"
rank there, which then forwards each destination's sub-chunk intra-node
(inter-node wire = per-node column sums of M; static ring schedule).
"""

import argparse
import os
import time
from functools import partial
from typing import Any, List, Optional

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    RING_MODE_MAP,
    MoeAgScatterWithTorch,
    MoeMlp1Ctx,
    gen_moe_gating_args,
    choosed_experts_to_matrix_chunks,
    load_routing_file,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds
from flux.testing.payload_probe import PayloadProbe, payload_probe_enabled
from flux.testing.recorder import RECORDER

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def init_ep_group(ep_size: int):
    assert DIST_ENV.WORLD_SIZE % ep_size == 0, f"{DIST_ENV.WORLD_SIZE} % {ep_size} != 0"
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP already initialized"

    assert TP_GROUP.size() % ep_size == 0, f"{TP_GROUP.size()} % {ep_size} != 0"
    ffn_tp_size = TP_GROUP.size() // ep_size

    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)

    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])

    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group


class PerfResult:
    def __init__(
        self,
        name: str,
        outputs: List[torch.Tensor],
        gathered_input: torch.Tensor,
        gemm_time_ms: float,
        scatter_time_ms: float,
        comm_time_ms: float,
    ) -> None:
        self.name = name
        self.outputs = outputs
        self.gathered_input = gathered_input
        self.gemm_time_ms = gemm_time_ms
        self.scatter_time_ms = scatter_time_ms
        self.comm_time_ms = comm_time_ms
        self.total_ms = self.gemm_time_ms + self.scatter_time_ms + self.comm_time_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"
            f", scatter {self.scatter_time_ms:.3f} ms"
            f", comm {self.comm_time_ms:.3f} ms"
        )


def take_first_or_none(x: Optional[List[Any]]):
    return x[0] if x is not None else None


@torch.no_grad()
def perf_torch(
    ctx: MoeMlp1Ctx,
    warmup_iters: int,
    iters: int,
    gather_input: bool = True,
    meta_op=None,
    topk_shard: torch.Tensor = None,
    topk_gather_buf: torch.Tensor = None,
):
    gemm_only_op = flux.GemmOnly(
        ctx.inputs.dtype,
        ctx.inputs.dtype,
        ctx.outputs[0].dtype,
        use_fp8_gemm=flux.is_fp8_dtype(ctx.inputs.dtype),
    )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    plan_comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    plan_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    scatter_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    # rule-5 in-window planning scratch (allocation is setup scope, CONTENTS
    # are re-derived per iteration): the reconstructed gather_index and the
    # static token-id ramp it is scattered from.
    n_copies = topk_gather_buf.numel()
    topk = topk_gather_buf.size(1)
    gather_index_dev = torch.empty(n_copies, dtype=torch.int32, device="cuda")
    token_of_copy = (
        torch.arange(n_copies, dtype=torch.int32, device="cuda") // topk
    )
    # wire-ordering audit (CLAUDE.md invariant 5): per-iteration payload change.
    # SAME seed as perf_flux's probe (both loops run total_iters steps) so both
    # loops end on the identical final payload and the flux-vs-torch output
    # comparison below stays valid. No-op unless FLUX_RANDOM_PAYLOAD=1.
    probe = PayloadProbe(ctx.inputs_shard, TP_GROUP.rank(), keep_ledger=False)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    for i in range(total_iters):
        probe.step(i)
        start_events[i].record()
        # rule 5 (SCHEMA protocol): routing exchange + ALL routing-derived
        # metadata inside the timed window, every iteration.
        torch.distributed.all_gather_into_tensor(topk_gather_buf, topk_shard, group=TP_GROUP)
        plan_comm_events[i].record()
        sd, scd, _, _ = meta_op.derive_routed_meta(topk_gather_buf)
        # gather_index[p] = token of the copy at sorted position p; splits_cpu
        # is the D2H the host-side per-expert gemm loop genuinely requires.
        gather_index_dev.scatter_(0, scd.view(-1).long(), token_of_copy)
        ctx.gather_index = gather_index_dev
        ctx.splits_cpu = sd.cpu()
        plan_events[i].record()
        MoeAgScatterWithTorch.comm_impl(ctx, TP_GROUP)
        comm_end_events[i].record()
        MoeAgScatterWithTorch.scatter_impl(ctx)
        scatter_end_events[i].record()
        MoeAgScatterWithTorch.gemm_impl(ctx, gemm_only_op)
        gemm_end_events[i].record()
    plan_comm_times = []
    plan_times = []
    comm_times = []
    scatter_times = []
    gemm_times = []
    total_times = []
    for i in range(total_iters):
        gemm_end_events[i].synchronize()
        if i >= warmup_iters:
            plan_comm_times.append(start_events[i].elapsed_time(plan_comm_events[i]))
            plan_times.append(plan_comm_events[i].elapsed_time(plan_events[i]))
            comm_times.append(plan_events[i].elapsed_time(comm_end_events[i]))
            scatter_times.append(comm_end_events[i].elapsed_time(scatter_end_events[i]))
            gemm_times.append(scatter_end_events[i].elapsed_time(gemm_end_events[i]))
            total_times.append(start_events[i].elapsed_time(gemm_end_events[i]))
    comm_time = sum(comm_times) / iters
    scatter_time = sum(scatter_times) / iters
    gemm_time = sum(gemm_times) / iters

    result = PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=flux.testing.clone_with_fp8(ctx.inputs),
        gemm_time_ms=gemm_time,
        scatter_time_ms=scatter_time,
        comm_time_ms=comm_time,
    )
    result.iter_times = {
        "plan_comm_ms": plan_comm_times,
        "plan_ms": plan_times,
        "comm_ms": comm_times,
        "scatter_ms": scatter_times,
        "gemm_ms": gemm_times,
        "total_ms": total_times,
    }
    return result


@torch.no_grad()
def make_flux_op(ctx: MoeMlp1Ctx, comm_pattern: str):
    """Construct the layer0 op once (setup scope: allocation only — every
    routing-derived quantity is re-derived in-window, per iteration)."""
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )
    use_a2av = comm_pattern in ("a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress")
    if flux.util.get_arch() >= 90:
        assert not use_a2av, "--comm_pattern a2av is only implemented for the sm80/V2 op"
        return flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    return flux.GemmGroupedV2AGScatterOp(
        tp_env=tp_env,
        moe_args=moe_args,
        a2av_dispatch=use_a2av,
        a2av_ring=(comm_pattern == "a2av_ring"),
        a2av_hier=(comm_pattern == "a2av_hier"),
        a2av_hier_compress=(comm_pattern == "a2av_hier_compress"),
    )


@torch.no_grad()
def perf_flux(
    ctx: MoeMlp1Ctx,
    warmup_iters: int,
    iters: int,
    gather_input: bool = True,
    ag_option: flux.AllGatherOption = flux.AllGatherOption(),
    comm_pattern: str = "allgather",
    op=None,
    topk_shard: torch.Tensor = None,
    topk_gather_buf: torch.Tensor = None,
):
    use_a2av = comm_pattern in ("a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress")
    extra_args = {}
    if flux.util.get_arch() < 90:
        extra_args = {
            "ag_option": ag_option,
            "bias": take_first_or_none(ctx.bias),
            "input_scale": take_first_or_none(ctx.input_scale),
            "weight_scale": take_first_or_none(ctx.weight_scale),
        }
    if use_a2av:
        assert not gather_input, "--gather_input has no dense gathered buffer in a2av mode"

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    plan_comm_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    plan_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    gathered_input = torch.empty_like(ctx.inputs) if gather_input else None
    # isolated mode (sweeps SCHEMA.md): drain the device and align all ranks
    # before EVERY timed window, so each iteration measures one isolated layer
    # execution (inference semantics — routing changes per activation) with no
    # cross-iteration pipelining. iso_sync_ms (host wall time of the pair) is
    # a per-rank straggler indicator: it is the wait for the slowest rank.
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    # wire-ordering audit (CLAUDE.md invariant 5): the op re-reads
    # ctx.inputs_shard every forward, so an in-place per-iteration
    # randomization (outside the timed window) exercises the wire with
    # changing bytes; a static payload hides a one-epoch-stale delivery.
    probe = PayloadProbe(ctx.inputs_shard, TP_GROUP.rank(), keep_ledger=True)
    for i in range(total_iters):
        ctx.clear_outputs()
        op.clear_buffers()
        probe.step(i)
        # provenance (audit): the payload BEFORE this iteration's = ledger[-2]
        ctx._probe_prev = probe.ledger[-2] if len(probe.ledger) >= 2 else None
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        # NVTX iteration markers: ns-scale, inert without a profiler; lets an
        # nsys-mode capture segment warmup vs timed without device syncs
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        start_events[i].record()
        with torch.cuda.nvtx.range(nvtx_tag):
            # rule 5 (SCHEMA protocol): the routing exchange (plan_comm) and
            # ALL routing-derived metadata (plan) are timed, per iteration.
            torch.distributed.all_gather_into_tensor(
                topk_gather_buf, topk_shard, group=TP_GROUP
            )
            plan_comm_events[i].record()
            # fused C++/CUDA derivation (FLUX_A2AV_INWINDOW_META_TAG). The
            # returned tensors alias op-internal buffers and are overwritten
            # by the next derive call — safe here because forward consumes
            # them before the next iteration derives again. The internal
            # pinned-D2H event sync is the honest host sync the op's
            # host-planned a2av genuinely requires.
            sd, scd, sps_c, uc_c = op.derive_routed_meta(topk_gather_buf)
            plan_events[i].record()
            iter_extra = dict(extra_args)
            if not args.no_metadata_cnt:
                iter_extra["splits_per_source"] = sps_c
            if comm_pattern == "a2av_hier_compress":
                iter_extra["a2av_unique_counts"] = uc_c
            op.forward(
                inputs_shard=ctx.inputs_shard,
                weights=ctx.weights[0],
                splits_gpu=sd,
                scatter_index=scd,
                output_scale=take_first_or_none(ctx.output_scale),
                outputs_buf=take_first_or_none(ctx.outputs),
                fast_accum=ctx.fast_accum,
                sm_margin=args.sm_margin,
                allgather_output=gathered_input,
                **iter_extra,
            )
        end_events[i].record()

    plan_comm_times = []
    plan_times = []
    gemm_times = []
    total_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            plan_comm_times.append(start_events[i].elapsed_time(plan_comm_events[i]))
            plan_times.append(plan_comm_events[i].elapsed_time(plan_events[i]))
            gemm_times.append(plan_events[i].elapsed_time(end_events[i]))
            total_times.append(start_events[i].elapsed_time(end_events[i]))

    gemm_time_ms = sum(gemm_times) / iters

    result = PerfResult(
        name=f"flux #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=gathered_input,
        gemm_time_ms=gemm_time_ms,
        scatter_time_ms=0.0,
        comm_time_ms=0.0,
    )
    # e2e_ms keeps its historical meaning (the op.forward window; comm-start
    # anchor unchanged); plan_comm_ms/plan_ms sit inside total_ms but OUTSIDE
    # e2e_ms (SCHEMA plan_ms contract). Quote total_ms for isolated latency.
    result.iter_times = {
        "plan_comm_ms": plan_comm_times,
        "plan_ms": plan_times,
        "e2e_ms": gemm_times,
        "total_ms": total_times,
    }
    if isolated:
        result.iter_times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return result


@torch.no_grad()
def tune_flux(ctx: MoeMlp1Ctx) -> flux.ProfilingContext:
    name = f"config_ag_scatter_sm{flux.get_arch()}"
    prof_ctx = flux.ProfilingContext(name)
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )

    if flux.util.get_arch() >= 90:
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    else:
        op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)

    op.profiling(
        inputs_shard=ctx.inputs_shard,
        weights=ctx.weights,
        splits_gpu=ctx.splits_gpu,
        scatter_index=ctx.scatter_index,
        output_scale=ctx.output_scale,
        outputs_buf=ctx.outputs,
        fast_accum=ctx.fast_accum,
        prof_ctx=prof_ctx,
    )
    torch.cuda.synchronize()
    return prof_ctx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True, help="traffic matrix file")
    parser.add_argument(
        "--routing_file",
        type=str,
        default=None,
        help="per-token expert-id sidecar (<matrix>.routing.txt, trace family):"
        " use the REAL token-overlap structure instead of synthesizing the"
        " max-dedup dealer assignment from the byte matrix; must realize"
        " exactly --traffic_matrix",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=int,
        default=8192,
        help="bytes of one routed token copy in the traffic matrix (H * dtype size)",
    )
    parser.add_argument("--H", type=int, default=4096, help="token hidden dim")
    parser.add_argument("--ffn_hidden_size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--G", type=int, default=32, help="number of experts")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=10, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=["bfloat16", "float16"]
    )
    parser.add_argument(
        "--profile", default=False, action="store_true", help="dump torch.profiler.profile"
    )
    parser.add_argument("--tune", default=False, action="store_true", help="find best GemmHParams")
    parser.add_argument(
        "--gather_input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="gather input",
    )
    parser.add_argument(
        "--ring_mode",
        default="auto",
        choices=["auto", "all2all", "ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    parser.add_argument(
        "--use_cuda_core_local",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl local copy, auto select if not specified",
    )
    parser.add_argument(
        "--use_cuda_core_ag",
        action=argparse.BooleanOptionalAction,
        help="use cuda core to impl all gather, auto select if not specified",
    )
    parser.add_argument(
        "--comm_pattern",
        default="allgather",
        choices=["allgather", "a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress"],
        help="layer0 comm pattern: dense allgather (default), raw alltoallv"
        " dispatch whose wire bytes equal the traffic matrix (dynamic tile"
        " schedule), a2av_ring (same wire bytes, static ring schedule),"
        " a2av_hier (hierarchical: one aggregated inter-node message per peer"
        " node to the same-local-rank gateway, which forwards intra-node;"
        " static ring schedule), or a2av_hier_compress (hierarchical with"
        " token-dedup wire semantics: each token crosses the wire at most once"
        " per destination rank / node; requires metadata inputs and, multi-node,"
        " --sm_margin >= 1)",
    )
    parser.add_argument(
        "--skip_correctness",
        default=False,
        action="store_true",
        help="skip the torch reference (perf_torch + result checks) and shrink"
        " its (ntokens * topk, H) scatter staging buffer to one row — for"
        " large-budget perf sweeps where the reference dominates wall time or"
        " OOMs; correctness is then NOT verified",
    )
    parser.add_argument(
        "--no_metadata_cnt",
        default=False,
        action="store_true",
        help="ABLATION ONLY (never in campaign specs): derive cnt[s][e]"
        " in-window as usual but do not pass it to forward, so the op falls"
        " back to its internal re-derivation from splits/scatter_index"
        " (pre-metadata-input behavior, for A/B comparison)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # each expert's full ffn weight resides on one rank
    init_ep_group(DIST_ENV.WORLD_SIZE)

    print("before flux_shm initialization")
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    print("after flux_shm initialization")

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.H * input_dtype.itemsize == args.chunk_bytes, (
        f"H ({args.H}) * dtype size ({input_dtype.itemsize}) must equal the traffic matrix"
        f" chunk granularity ({args.chunk_bytes} bytes)"
    )
    assert args.G % DIST_ENV.WORLD_SIZE == 0, f"{args.G} % {DIST_ENV.WORLD_SIZE} != 0"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == DIST_ENV.WORLD_SIZE, (
        f"traffic matrix is for {matrix.shape[0]} ranks but world size is" f" {DIST_ENV.WORLD_SIZE}"
    )
    if args.routing_file:
        choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
        assert choosed_experts.shape[0] % DIST_ENV.WORLD_SIZE == 0
        got = choosed_experts_to_matrix_chunks(
            choosed_experts, DIST_ENV.WORLD_SIZE, args.G // DIST_ENV.WORLD_SIZE
        )
        assert torch.equal(got * args.chunk_bytes, matrix), (
            f"routing file {args.routing_file} does not realize --traffic_matrix"
            f" {args.traffic_matrix}"
        )
        if torch.cuda.is_available():
            choosed_experts = choosed_experts.cuda()
        if TP_GROUP.rank() == 0:
            print(f"routing: REAL trace file {args.routing_file}")
    else:
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
    ntokens = choosed_experts.shape[0]
    gating_args = gen_moe_gating_args(args.G, args.topk, ntokens, choosed_experts=choosed_experts)

    moe_ctx = MoeMlp1Ctx(
        TP_GROUP,
        EP_GROUP,
        b=1,
        s=ntokens,
        h=args.H,
        ffn_size=args.ffn_hidden_size,
        nexperts=args.G,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        dist="uniform",
        fast_accum=False,
        weight_groups=1,
        drop_token=False,
        gating_args=gating_args,
        skip_reference=args.skip_correctness,
        local_scatter=True,
    )

    # cnt[s][e] reference build (untimed setup, drift guard ONLY since the
    # 2026-08-21 rule-5 conversion): cnt[s][e] = copies source rank s sends to
    # expert e; splits is its column sum. The timed loops re-derive all of this
    # per iteration via op.derive_routed_meta; this python build exists to
    # bitwise-check that derivation once, at setup.
    W = DIST_ENV.WORLD_SIZE
    tokens_per_rank = ntokens // W
    src_of_copy = (torch.arange(ntokens, dtype=torch.long) // tokens_per_rank).repeat_interleave(
        args.topk
    )
    e_of_copy = choosed_experts.reshape(-1).long().cpu()
    splits_per_source_cpu = (
        torch.bincount(src_of_copy * args.G + e_of_copy, minlength=W * args.G).view(W, args.G).int()
    )
    assert torch.equal(
        splits_per_source_cpu.sum(0), moe_ctx.splits_cpu[: args.G].cpu().int()
    ), "splits_per_source column sums must equal splits"

    # compress dedup counts reference build (untimed setup, drift guard ONLY —
    # same rule-5 contract as splits_per_source above): u[s][d] = UNIQUE
    # tokens source s must deliver to rank d (any of d's experts), U[s][n] =
    # unique tokens s must deliver to the node-n union. NOT derivable from
    # cnt[s][e] — depends on which tokens overlap across experts/ranks.
    a2av_unique_counts_cpu = None
    if args.comm_pattern == "a2av_hier_compress":
        assert (
            not args.no_metadata_cnt
        ), "a2av_hier_compress requires the metadata inputs (drop --no_metadata_cnt)"
        experts_per_rank = args.G // W
        L = DIST_ENV.LOCAL_WORLD_SIZE
        nn = W // L
        owner = choosed_experts.long().cpu() // experts_per_rank  # [ntokens, topk] dest rank
        flags = torch.zeros(ntokens, W, dtype=torch.bool)
        flags.scatter_(1, owner, True)  # token t needed by rank d (any expert)
        u_mat = flags.view(W, tokens_per_rank, W).sum(1)  # [W, W]
        U_mat = flags.view(ntokens, nn, L).any(dim=2).view(W, tokens_per_rank, nn).sum(1)  # [W, nn]
        a2av_unique_counts_cpu = torch.cat([u_mat, U_mat], dim=1).int().contiguous()

    # ---- rule-5 apparatus (SCHEMA protocol rule 5, 2026-08-21) --------------
    # The op is constructed ONCE here (allocation is setup scope); the routing
    # exchange + every routing-derived quantity is re-derived per iteration
    # inside the timed windows of perf_flux/perf_torch. The only untimed
    # routing work below is the bitwise drift guard.
    assert flux.util.get_arch() < 90, (
        "rule-5 per-iteration derivation uses the sm80/V2 op's"
        " derive_routed_meta; the V3 path is not deployed here"
    )
    flux_op = make_flux_op(moe_ctx, args.comm_pattern)
    topk_shard = choosed_experts[
        TP_GROUP.rank() * tokens_per_rank : (TP_GROUP.rank() + 1) * tokens_per_rank
    ].contiguous()
    topk_gather_buf = torch.zeros(ntokens, args.topk, dtype=torch.int32, device="cuda")

    # Drift guard (untimed, once): the in-window derivation must reproduce the
    # replicated routing and the python reference metadata bitwise.
    torch.distributed.all_gather_into_tensor(topk_gather_buf, topk_shard, group=TP_GROUP)
    assert torch.equal(topk_gather_buf, choosed_experts), (
        "allgathered routing != replicated harness routing"
    )
    g_sd, g_scd, g_sps, g_uc = flux_op.derive_routed_meta(topk_gather_buf)
    assert torch.equal(
        g_sd.cpu(), moe_ctx.splits_cpu[: args.G].cpu().int()
    ), "derive_routed_meta splits drift vs harness reference"
    assert torch.equal(g_scd, moe_ctx.scatter_index.int()), (
        "derive_routed_meta stable scatter_index drift vs harness reference"
    )
    assert torch.equal(g_sps, splits_per_source_cpu), (
        "derive_routed_meta splits_per_source drift vs harness reference"
    )
    if a2av_unique_counts_cpu is not None:
        assert torch.equal(g_uc, a2av_unique_counts_cpu), (
            "derive_routed_meta a2av_unique_counts drift vs harness reference"
        )
    # perf_torch's in-window gather_index reconstruction, guarded the same way
    _g_gather = torch.empty(ntokens * args.topk, dtype=torch.int32, device="cuda")
    _g_gather.scatter_(
        0,
        g_scd.view(-1).long(),
        torch.arange(ntokens * args.topk, dtype=torch.int32, device="cuda") // args.topk,
    )
    assert torch.equal(_g_gather, moe_ctx.gather_index.int()), (
        "in-window gather_index reconstruction drift vs harness reference"
    )
    del _g_gather

    if TP_GROUP.rank() == 0:
        experts_per_rank = args.G // DIST_ENV.WORLD_SIZE
        rows_per_rank = moe_ctx.splits_cpu.view(DIST_ENV.WORLD_SIZE, experts_per_rank).sum(dim=1)
        print(f"ntokens: {ntokens} ({ntokens // DIST_ENV.WORLD_SIZE} per rank), topk: {args.topk}")
        print(f"Splits: {moe_ctx.splits_cpu.tolist()}, Sum: {sum(moe_ctx.splits_cpu.tolist())}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(f"comm_pattern: {args.comm_pattern}")
        RECORDER.emit_info(
            ntokens=ntokens,
            tokens_per_rank=ntokens // DIST_ENV.WORLD_SIZE,
            gemm_rows_per_rank=rows_per_rank.tolist(),
            # SCHEMA protocol rule 5 (flux driver converted 2026-08-21): the
            # routing allgather + all plan derivation timed per iteration.
            timing_accounting="per_iter_gpu",
            # 2026-08-21 torch-reference fix, attributable per capsule: the
            # reference scatters only the local EP slice (W-fold less memory
            # and scatter work); torch rows never compare across this flip.
            torch_ref_impl="local_slice_scatter",
        )
        if args.comm_pattern in ("a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress"):
            send_bytes = (matrix.sum(dim=1) - matrix.diag()).tolist()
            recv_bytes = (matrix.sum(dim=0) - matrix.diag()).tolist()
            print(f"a2av wire bytes per rank (send): {send_bytes}")
            print(f"a2av wire bytes per rank (recv): {recv_bytes}")
        if args.comm_pattern == "a2av_hier":
            # actual inter-node wire in hier mode: per source rank, bytes summed
            # over destination-node columns, own node excluded (they travel as
            # one aggregated message per remote node); intra-node forwarding is
            # extra NVLink traffic on top
            L = DIST_ENV.LOCAL_WORLD_SIZE
            nn = DIST_ENV.WORLD_SIZE // L
            per_node = matrix.view(DIST_ENV.WORLD_SIZE, nn, L).sum(dim=2)
            src_node = torch.arange(DIST_ENV.WORLD_SIZE) // L
            inter_bytes = per_node.sum(dim=1) - per_node.gather(1, src_node.view(-1, 1)).squeeze(1)
            print(f"a2av_hier inter-node wire bytes per rank (send): {inter_bytes.tolist()}")
        if args.comm_pattern == "a2av_hier_compress":
            # actual wire in compress mode: intra-node dedup puts (own rank
            # excluded) + one union aggregate per remote node; the matrix above
            # stays the LOGICAL traffic. Gateway forwarding is extra NVLink
            # traffic on top, exactly u[s][d] rows per (source, local dest).
            L = DIST_ENV.LOCAL_WORLD_SIZE
            nn = DIST_ENV.WORLD_SIZE // L
            src_node = torch.arange(W) // L
            intra_rows = u_mat.view(W, nn, L)[torch.arange(W), src_node].sum(1) - u_mat.diag()
            inter_rows = U_mat.sum(1) - U_mat[torch.arange(W), src_node]
            comp_bytes = (intra_rows + inter_rows) * args.chunk_bytes
            inter_bytes_c = inter_rows * args.chunk_bytes
            logical_bytes = matrix.sum(dim=1) - matrix.diag()
            ratio = comp_bytes.sum().item() / max(logical_bytes.sum().item(), 1)
            print(f"a2av_hier_compress wire bytes per rank (send): {comp_bytes.tolist()}")
            print(
                "a2av_hier_compress inter-node wire bytes per rank (send):"
                f" {inter_bytes_c.tolist()}"
            )
            print(f"a2av_hier_compress dedup wire/logical send-byte ratio: {ratio:.3f}")
            RECORDER.emit_info(wire_ratio=ratio)
            # gateway forward cost on NVLink, exact-subset gathers vs the
            # FLUX_A2AV_UNION_BCAST=1 whole-union broadcast ((L-1) * U rows;
            # the gateway's own copy is a local D2D either way)
            if nn > 1:
                gw_gather = torch.zeros(W, dtype=torch.long)
                gw_bcast = torch.zeros(W, dtype=torch.long)
                for n in range(nn):
                    for lr in range(L):
                        g = n * L + lr
                        for m in range(nn):
                            if m == n:
                                continue
                            s = m * L + lr
                            gw_gather[g] += int(u_mat[s, n * L : (n + 1) * L].sum()) - int(
                                u_mat[s, g]
                            )
                            gw_bcast[g] += (L - 1) * int(U_mat[s, n])
                print(
                    "a2av gateway intra-node forward bytes per gateway"
                    f" (gather): {(gw_gather * args.chunk_bytes).tolist()}"
                )
                print(
                    "a2av gateway intra-node forward bytes per gateway"
                    f" (bcast):  {(gw_bcast * args.chunk_bytes).tolist()}"
                )
                # FLUX_A2AV_LB_UNION=1: each gateway forwards only its balanced
                # window, chunk lr of every inbound round's union stream — same
                # aggregate NVLink bytes as bcast, but the per-gateway max drops
                # to ~(L-1)/L of the round totals
                gw_lb = torch.zeros(W, dtype=torch.long)
                for n in range(nn):
                    for lr in range(L):
                        g = n * L + lr
                        for m in range(nn):
                            if m == n:
                                continue
                            total = int(U_mat[m * L : (m + 1) * L, n].sum())
                            lo = (total // L) * lr + min(lr, total % L)
                            hi = (total // L) * (lr + 1) + min(lr + 1, total % L)
                            gw_lb[g] += (L - 1) * (hi - lo)
                print(
                    "a2av gateway intra-node forward bytes per gateway"
                    f" (lb_bcast): {(gw_lb * args.chunk_bytes).tolist()}"
                )
                RECORDER.emit_info(gw_lb_bcast_bytes=int(gw_lb.max()) * args.chunk_bytes)
            # balanced-relay effect (FLUX_A2AV_RELAY_IDENTITY=0, the default):
            # per (node, round) the wire pace is ceil(total / L) instead of the
            # hottest rank's U — report the per-round worst sender both ways
            rounds = []
            for n in range(nn):
                for dn in range(1, nn):
                    tn = (n - dn + nn) % nn
                    seg = [int(U_mat[n * L + sl, tn]) for sl in range(L)]
                    total = sum(seg)
                    rounds.append((max(seg), (total + L - 1) // L))
            if rounds:
                ident = sum(m for m, _ in rounds) * args.chunk_bytes
                balanced = sum(b for _, b in rounds) * args.chunk_bytes
                print(
                    "a2av relay balance: sum of per-round max sender bytes"
                    f" identity {ident} -> balanced {balanced}"
                    f" ({balanced / max(ident, 1):.3f}x)"
                )
                RECORDER.emit_info(relay_ident_bytes=ident, relay_balanced_bytes=balanced)

    if args.tune:
        prof_ctx = tune_flux(moe_ctx)

        if DIST_ENV.RANK == 0:
            print("====== Profiling Results =======")
            print("\n".join(prof_ctx.get_all_prof_results()))
            print("====== Generated Config Code =======")
            print(prof_ctx.get_code())

        flux.load_tuning_record(prof_ctx.get_latest_record())

    ag_option = flux.AllGatherOption()
    ag_option.use_cuda_core_local = args.use_cuda_core_local
    ag_option.use_cuda_core_ag = args.use_cuda_core_ag
    ag_option.mode = RING_MODE_MAP[args.ring_mode]

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_ag_scatter_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        perf_result_flux = perf_flux(
            moe_ctx,
            args.warmup_iters,
            args.iters,
            args.gather_input,
            ag_option,
            args.comm_pattern,
            op=flux_op,
            topk_shard=topk_shard,
            topk_gather_buf=topk_gather_buf,
        )
        perf_result_torch = (
            None
            if args.skip_correctness
            else perf_torch(
                moe_ctx,
                args.warmup_iters,
                args.iters,
                args.gather_input,
                meta_op=flux_op,
                topk_shard=topk_shard,
                topk_gather_buf=topk_gather_buf,
            )
        )

    if TP_GROUP.rank() == 0:
        flux.testing.print_grouped_gemm_sol_time_ms(
            moe_ctx.ntokens * moe_ctx.topk // moe_ctx.ep_size,
            moe_ctx.ffn_size_shard,
            moe_ctx.h,
            args.G // moe_ctx.ep_size,  # E
            input_dtype=input_dtype,
        )
    if should_log_to_rds():
        set_global_args("moe_ag_scatter_traffic", args)
    if perf_result_torch is not None:
        flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
        RECORDER.emit_iters("torch", perf_result_torch.iter_times)
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))
    RECORDER.emit_iters("flux", perf_result_flux.iter_times)

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    elif input_dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1.5e-2
    else:
        raise ValueError(f"Unsupported dtype {input_dtype}")

    def check_result(perf_out_x, perf_out_y, name_x: str, name_y: str):
        print(f"Checking RANK #{TP_GROUP.rank()}...")
        if args.gather_input:
            assert flux.testing.bitwise_eq(perf_out_x.gathered_input, perf_out_y.gathered_input)
        bitwise_all = True
        for x, y in zip(perf_out_x.outputs, perf_out_y.outputs):
            print("output shape", x.size())
            if flux.testing.bitwise_eq(x, y):
                print(f"✅ {name_x} and torch bitwise match")
            else:
                bitwise_all = False
                print(f"❌ {name_x} and torch not bitwise match")
                if payload_probe_enabled():
                    bad = (x != y).any(dim=1)
                    print(f"  probe rank {TP_GROUP.rank()}: {int(bad.sum())}/{x.shape[0]}"
                          f" output rows differ (GEMM rows of the local EP block)")
            try:
                flux.torch_allclose(x, y, atol=atol, rtol=rtol)
            except Exception as e:
                # audit detail: how many ROWS violate tolerance (a stale row
                # under the sign-alternating probe violates it in every elem;
                # a tolerance artifact shows as scattered elements)
                viol = ((x.float() - y.float()).abs() > (atol + rtol * y.float().abs()))
                rows_bad = int(viol.any(dim=1).sum())
                rows_all = int(viol.all(dim=1).sum())
                idx = viol.any(dim=1).nonzero(as_tuple=True)[0][:8].tolist()
                per = [int(viol[i].sum()) for i in idx]
                print(f"  probe rank {TP_GROUP.rank()}: allclose violation rows "
                      f"{rows_bad}/{x.shape[0]} (rows violating in EVERY element: {rows_all}; "
                      f"max|x-y| {float((x.float()-y.float()).abs().max()):.4g}; "
                      f"first rows {idx} violating elems/row {per}; "
                      f"|y| scale {float(y.float().abs().mean()):.4g})", flush=True)
                if perf_result_torch_prev is not None:
                    yp = perf_result_torch_prev.outputs[perf_out_x.outputs.index(x)]
                    bad_rows = viol.any(dim=1).nonzero(as_tuple=True)[0]
                    close_prev = ((x.float()[bad_rows] - yp.float()[bad_rows]).abs()
                                  <= (atol + rtol * yp.float()[bad_rows].abs())).all(dim=1)
                    print(f"  probe rank {TP_GROUP.rank()}: of {int(bad_rows.numel())} bad rows, "
                          f"{int(close_prev.sum())} match the PREVIOUS payload's torch output "
                          f"(stale-by-one), {int((~close_prev).sum())} match neither", flush=True)
                dump_dir = os.environ.get("FLUX_DEBUG_DUMP_DIR", "/tmp")
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(x, os.path.join(dump_dir, f"{name_x}_{TP_GROUP.rank()}.pt"))
                torch.save(y, os.path.join(dump_dir, f"{name_y}_{TP_GROUP.rank()}.pt"))
                torch.save(moe_ctx, os.path.join(dump_dir, f"moe_ctx_{TP_GROUP.rank()}.pt"))
                print(f"❌ {name_x} check failed, debug tensors dumped to {dump_dir}")
                RECORDER.emit_correctness(bitwise=bitwise_all, allclose=False)
                raise e
            else:
                print(f"✅ {name_x} check passed")
        RECORDER.emit_correctness(bitwise=bitwise_all, allclose=True)

    perf_result_torch_prev = None
    if (perf_result_torch is not None and payload_probe_enabled()
            and getattr(moe_ctx, "_probe_prev", None) is not None):
        # PROVENANCE (collective): torch reference on the PREVIOUS payload so a
        # wrong flux row can be classified as "stale = previous iteration".
        _saved = moe_ctx.inputs_shard.clone()
        moe_ctx.inputs_shard.copy_(moe_ctx._probe_prev)
        _envs = {k: os.environ.get(k) for k in ("FLUX_RANDOM_PAYLOAD", "FLUX_PLL_RANDOM_PAYLOAD")}
        os.environ["FLUX_RANDOM_PAYLOAD"] = "0"
        os.environ["FLUX_PLL_RANDOM_PAYLOAD"] = "0"
        try:
            perf_result_torch_prev = perf_torch(
                moe_ctx, 0, 1, args.gather_input, meta_op=flux_op,
                topk_shard=topk_shard, topk_gather_buf=topk_gather_buf)
        finally:
            for k, v in _envs.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            moe_ctx.inputs_shard.copy_(_saved)
        torch.cuda.synchronize()

    if perf_result_torch is not None:
        _ok = [True]

        def _check_no_raise():
            try:
                check_result(perf_result_flux, perf_result_torch, "flux", "torch")
            except Exception as e:  # per-rank raise would wedge the other ranks
                _ok[0] = False
                print(f"❌ rank {TP_GROUP.rank()}: check raised {type(e).__name__}", flush=True)

        flux.exec_in_rank_order(TP_GROUP, _check_no_raise)
        _flag = torch.tensor([int(_ok[0])], device="cuda")
        torch.distributed.all_reduce(_flag, op=torch.distributed.ReduceOp.MIN, group=TP_GROUP)
        assert int(_flag) == 1, "correctness failed on at least one rank"

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
