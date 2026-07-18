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
dense path's static ring-order tile schedule.
"""

import argparse
import os
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
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds

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
def perf_torch(ctx: MoeMlp1Ctx, warmup_iters: int, iters: int, gather_input: bool = True):
    gemm_only_op = flux.GemmOnly(
        ctx.inputs.dtype,
        ctx.inputs.dtype,
        ctx.outputs[0].dtype,
        use_fp8_gemm=flux.is_fp8_dtype(ctx.inputs.dtype),
    )

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    comm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    scatter_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    gemm_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    for i in range(total_iters):
        start_events[i].record()
        MoeAgScatterWithTorch.comm_impl(ctx, TP_GROUP)
        comm_end_events[i].record()
        MoeAgScatterWithTorch.scatter_impl(ctx)
        scatter_end_events[i].record()
        MoeAgScatterWithTorch.gemm_impl(ctx, gemm_only_op)
        gemm_end_events[i].record()
    comm_times = []
    scatter_times = []
    gemm_times = []
    for i in range(total_iters):
        comm_end_events[i].synchronize()
        scatter_end_events[i].synchronize()
        gemm_end_events[i].synchronize()
        if i >= warmup_iters:
            comm_times.append(start_events[i].elapsed_time(comm_end_events[i]))
            scatter_times.append(comm_end_events[i].elapsed_time(scatter_end_events[i]))
            gemm_times.append(scatter_end_events[i].elapsed_time(gemm_end_events[i]))
    comm_time = sum(comm_times) / iters
    scatter_time = sum(scatter_times) / iters
    gemm_time = sum(gemm_times) / iters

    return PerfResult(
        name=f"torch #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=flux.testing.clone_with_fp8(ctx.inputs),
        gemm_time_ms=gemm_time,
        scatter_time_ms=scatter_time,
        comm_time_ms=comm_time,
    )


@torch.no_grad()
def perf_flux(
    ctx: MoeMlp1Ctx,
    warmup_iters: int,
    iters: int,
    gather_input: bool = True,
    ag_option: flux.AllGatherOption = flux.AllGatherOption(),
    comm_pattern: str = "allgather",
    splits_per_source: torch.Tensor = None,
):
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

    use_a2av = comm_pattern in ("a2av", "a2av_ring")
    extra_args = {}
    if flux.util.get_arch() >= 90:
        assert not use_a2av, "--comm_pattern a2av is only implemented for the sm80/V2 op"
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    else:
        op = flux.GemmGroupedV2AGScatterOp(
            tp_env=tp_env,
            moe_args=moe_args,
            a2av_dispatch=use_a2av,
            a2av_ring=(comm_pattern == "a2av_ring"),
        )
        extra_args = {
            "ag_option": ag_option,
            "bias": take_first_or_none(ctx.bias),
            "input_scale": take_first_or_none(ctx.input_scale),
            "weight_scale": take_first_or_none(ctx.weight_scale),
        }
        if splits_per_source is not None:
            # metadata-exchange result (untimed setup, like splits/scatter_index):
            # int32 CPU [W, nexperts]; the real exchange is a ~W*nexperts-int
            # allgather (~10-20 us), declared out of scope by the harness contract
            extra_args["splits_per_source"] = splits_per_source
    if use_a2av:
        assert not gather_input, "--gather_input has no dense gathered buffer in a2av mode"

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    ctx.clear_outputs()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    gathered_input = torch.empty_like(ctx.inputs) if gather_input else None
    for i in range(total_iters):
        ctx.clear_outputs()
        op.clear_buffers()
        start_events[i].record()
        op.forward(
            inputs_shard=ctx.inputs_shard,
            weights=ctx.weights[0],
            splits_gpu=ctx.splits_gpu,
            scatter_index=ctx.scatter_index,
            output_scale=take_first_or_none(ctx.output_scale),
            outputs_buf=take_first_or_none(ctx.outputs),
            fast_accum=ctx.fast_accum,
            sm_margin=args.sm_margin,
            allgather_output=gathered_input,
            **extra_args,
        )
        end_events[i].record()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]))

    gemm_time_ms = sum(gemm_times) / iters

    return PerfResult(
        name=f"flux #{TP_GROUP.rank()}",
        outputs=ctx.get_outputs_clone(),
        gathered_input=gathered_input,
        gemm_time_ms=gemm_time_ms,
        scatter_time_ms=0.0,
        comm_time_ms=0.0,
    )


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
        choices=["allgather", "a2av", "a2av_ring"],
        help="layer0 comm pattern: dense allgather (default), raw alltoallv"
        " dispatch whose wire bytes equal the traffic matrix (dynamic tile"
        " schedule), or a2av_ring (same wire bytes, static ring schedule)",
    )
    parser.add_argument(
        "--no_metadata_cnt",
        default=False,
        action="store_true",
        help="do not pass splits_per_source (cnt[s][e]) to forward; each rank"
        " re-derives all metadata from splits/scatter_index inside the timed"
        " region (pre-metadata-input behavior, for A/B comparison)",
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
        f"traffic matrix is for {matrix.shape[0]} ranks but world size is"
        f" {DIST_ENV.WORLD_SIZE}"
    )
    choosed_experts = traffic_matrix_to_choosed_experts(
        matrix, args.G, args.topk, args.chunk_bytes
    )
    ntokens = choosed_experts.shape[0]
    gating_args = gen_moe_gating_args(
        args.G, args.topk, ntokens, choosed_experts=choosed_experts
    )

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
    )

    # metadata-exchange result (untimed setup): cnt[s][e] = copies source rank s
    # sends to expert e; splits is its column sum. In a real system this is a
    # W x nexperts int allgather (~10-20 us) done right after gating.
    W = DIST_ENV.WORLD_SIZE
    tokens_per_rank = ntokens // W
    src_of_copy = (
        torch.arange(ntokens, dtype=torch.long) // tokens_per_rank
    ).repeat_interleave(args.topk)
    e_of_copy = choosed_experts.reshape(-1).long().cpu()
    splits_per_source_cpu = (
        torch.bincount(src_of_copy * args.G + e_of_copy, minlength=W * args.G)
        .view(W, args.G)
        .int()
    )
    assert torch.equal(
        splits_per_source_cpu.sum(0), moe_ctx.splits_cpu[: args.G].cpu().int()
    ), "splits_per_source column sums must equal splits"

    if TP_GROUP.rank() == 0:
        experts_per_rank = args.G // DIST_ENV.WORLD_SIZE
        rows_per_rank = moe_ctx.splits_cpu.view(DIST_ENV.WORLD_SIZE, experts_per_rank).sum(dim=1)
        print(f"ntokens: {ntokens} ({ntokens // DIST_ENV.WORLD_SIZE} per rank), topk: {args.topk}")
        print(f"Splits: {moe_ctx.splits_cpu.tolist()}, Sum: {sum(moe_ctx.splits_cpu.tolist())}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(f"comm_pattern: {args.comm_pattern}")
        if args.comm_pattern in ("a2av", "a2av_ring"):
            send_bytes = (matrix.sum(dim=1) - matrix.diag()).tolist()
            recv_bytes = (matrix.sum(dim=0) - matrix.diag()).tolist()
            print(f"a2av wire bytes per rank (send): {send_bytes}")
            print(f"a2av wire bytes per rank (recv): {recv_bytes}")

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
            splits_per_source=None if args.no_metadata_cnt else splits_per_source_cpu,
        )
        perf_result_torch = perf_torch(moe_ctx, args.warmup_iters, args.iters, args.gather_input)

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
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))

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
        for x, y in zip(perf_out_x.outputs, perf_out_y.outputs):
            print("output shape", x.size())
            if flux.testing.bitwise_eq(x, y):
                print(f"✅ {name_x} and torch bitwise match")
            else:
                print(f"❌ {name_x} and torch not bitwise match")
            try:
                flux.torch_allclose(x, y, atol=atol, rtol=rtol)
            except Exception as e:
                torch.save(x, f"{name_x}_{TP_GROUP.rank()}.pt")
                torch.save(y, f"{name_y}_{TP_GROUP.rank()}.pt")
                torch.save(moe_ctx, f"moe_ctx_{TP_GROUP.rank()}.pt")
                print(f"❌ {name_x} check failed")
                raise e
            else:
                print(f"✅ {name_x} check passed")

    flux.exec_in_rank_order(
        TP_GROUP, lambda: check_result(perf_result_flux, perf_result_torch, "flux", "torch")
    )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
