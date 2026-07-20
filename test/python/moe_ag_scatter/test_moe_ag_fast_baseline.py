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
"""COMET MoE layer0 un-overlapped baseline: FAST alltoallv + separate grouped GEMM.

The load-balancing alltoallv (FAST, 3rdparty/FAST: BvN decomposition of the
demand matrix into balanced inter-node permutation steps + intra-node
load-balance/redistribute over NVLink) moves exactly M[s][d] wire bytes source
rank -> expert-owner rank; a separate, comm-free grouped GEMM
(flux.GemmGroupedV2) then consumes the landed tokens. This replaces the dense
all-gather + scatter of the torch baseline and is the un-overlapped counterpart
of the fused --comm_pattern a2av* modes in test_moe_ag_traffic.py.

PRIMARY METRIC: one end-to-end window per iteration, communication start ->
computation finish (schedule + fill + wire + unpack + gemm), directly comparable
to the fused op.forward numbers of test_moe_ag_traffic.py. The BvN schedule is
recomputed inside the window every iteration (one-shot methodology; never
amortized). The send-side pack (send-row index_select) is reported separately,
outside the window. Phase breakdown is diagnostic only.

NVSHMEM: this test never calls flux.init_flux_shm. FAST performs the only
NVSHMEM initialization in the process (uid broadcast over torch.distributed);
flux contributes only comm-free ops and testing utilities.

Constraints: multi-node only (FAST asserts server_n > 1), 4 GPUs/node,
node-major ranks (torchrun layout), ep_size == world_size.

Launch (one per node):
    srun --nodes=N --ntasks-per-node=1 ./launch_fast.sh \
        test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py \
        --traffic_matrix $PSCRATCH/.../a2av_4n_16r_dist_001.txt
"""

import argparse
import os
import sys
import time
from functools import partial
from typing import List

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    MoeAgScatterWithTorch,
    MoeMlp1Ctx,
    gen_moe_gating_args,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_baseline_utils import build_pack_index, build_unpack_index

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
    ep_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        for j in range(0, len(ranks), ep_size):
            ep_groups.append(ranks[j : j + ep_size])
    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group


def load_fast(fast_dir: str):
    """Load libflash.so (must come after `import flux`; FAST self-inits NVSHMEM)."""
    sys.path.insert(0, fast_dir)
    import flash_utils  # noqa: F401  (loads libflash.so)

    return flash_utils


def broadcast_uid(flash_utils) -> torch.Tensor:
    # uid is a CPU byte tensor; the global PG is NCCL, so bounce it via GPU
    if TP_GROUP.rank() == 0:
        uid = flash_utils.get_nvshmem_init_id()
    else:
        uid = torch.zeros((128,), dtype=torch.uint8, device="cpu")
    uid_gpu = uid.cuda()
    torch.distributed.broadcast(uid_gpu, src=0, group=TP_GROUP)
    return uid_gpu.cpu()


class FastPerfResult:
    def __init__(self, name, e2e_ms, pack_ms, schedule_ms, fill_ms, wire_ms, unpack_ms, gemm_ms, host_e2e_ms):
        self.name = name
        self.e2e_ms = e2e_ms  # PRIMARY: comm start -> gemm finish (CUDA events)
        self.pack_ms = pack_ms  # outside the window
        self.schedule_ms = schedule_ms
        self.fill_ms = fill_ms
        self.wire_ms = wire_ms
        self.unpack_ms = unpack_ms
        self.gemm_ms = gemm_ms
        self.host_e2e_ms = host_e2e_ms  # host wall cross-check of e2e_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: e2e {self.e2e_ms:.3f} ms (comm start -> gemm finish;"
            f" host {self.host_e2e_ms:.3f})"
            f" | pack {self.pack_ms:.3f} (outside window)"
            f" | schedule {self.schedule_ms:.3f} + fill {self.fill_ms:.3f}"
            f" + wire {self.wire_ms:.3f} + unpack {self.unpack_ms:.3f}"
            f" + gemm {self.gemm_ms:.3f}"
        )


@torch.no_grad()
def perf_fast(
    ctx: MoeMlp1Ctx,
    comm,
    gemm_op,
    matrix_cpu: torch.Tensor,
    pack_index_gpu: torch.Tensor,
    unpack_index_gpu: torch.Tensor,
    split_cpu: torch.Tensor,
    warmup_iters: int,
    iters: int,
    sm_margin: int = 0,
):
    input_dtype = ctx.inputs_shard.dtype
    H = ctx.h

    total_iters = warmup_iters + iters
    ev = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    pack_start, pack_end = ev(), ev()
    e2e_start, comm_end, unpack_end, e2e_end = ev(), ev(), ev(), ev()
    host_e2e = [0.0] * total_iters
    schedule_us = [0.0] * total_iters
    fill_us = [0.0] * total_iters
    wire_us = [0.0] * total_iters

    out = None
    torch.distributed.barrier()
    torch.cuda.synchronize()
    for i in range(total_iters):
        pack_start[i].record()
        send_rows = torch.index_select(ctx.inputs_shard, dim=0, index=pack_index_gpu)
        pack_end[i].record()
        torch.cuda.synchronize()  # pack complete before the e2e window opens

        t0 = time.perf_counter()
        e2e_start[i].record()
        # host-synchronous: schedule (BvN recompute) + fill + wire to completion
        recv_u8, out_sz, timings = comm.alltoallv(send_rows.view(torch.uint8), matrix_cpu)
        comm_end[i].record()
        gemm_input = torch.index_select(
            recv_u8.view(input_dtype).view(-1, H), dim=0, index=unpack_index_gpu
        )
        unpack_end[i].record()
        out = gemm_op.forward(gemm_input, split_cpu, sm_margin=sm_margin)
        e2e_end[i].record()
        e2e_end[i].synchronize()
        host_e2e[i] = (time.perf_counter() - t0) * 1e3
        schedule_us[i], fill_us[i], wire_us[i] = timings.tolist()

    def mean_ms(starts, ends):
        return sum(starts[i].elapsed_time(ends[i]) for i in range(warmup_iters, total_iters)) / iters

    def mean_host_ms(vals_us):
        return sum(vals_us[warmup_iters:]) / iters / 1e3

    return (
        FastPerfResult(
            name=f"fast #{TP_GROUP.rank()}",
            e2e_ms=mean_ms(e2e_start, e2e_end),
            pack_ms=mean_ms(pack_start, pack_end),
            schedule_ms=mean_host_ms(schedule_us),
            fill_ms=mean_host_ms(fill_us),
            wire_ms=mean_host_ms(wire_us),
            unpack_ms=mean_ms(comm_end, unpack_end),
            gemm_ms=mean_ms(unpack_end, e2e_end),
            host_e2e_ms=sum(host_e2e[warmup_iters:]) / iters,
        ),
        out,
        gemm_input,
    )


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
        "--capacity_mib",
        type=int,
        default=0,
        help="FAST buffer capacity per buffer in MiB; 0 = auto"
        " (4 x max(row sum, col sum) of the matrix)",
    )
    parser.add_argument(
        "--fast_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "3rdparty", "FAST", "nvidia"
        ),
        help="directory containing libflash.so + flash_utils.py",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)  # each expert's full ffn weight on one rank

    W = DIST_ENV.WORLD_SIZE
    L = DIST_ENV.LOCAL_WORLD_SIZE
    assert L == 4, f"FAST Perlmutter baseline expects 4 GPUs/node; got {L}"
    assert W > L, "FAST requires at least 2 nodes (server_n > 1)"

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.H * input_dtype.itemsize == args.chunk_bytes
    assert args.G % W == 0, f"{args.G} % {W} != 0"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W, f"matrix is for {matrix.shape[0]} ranks, world size {W}"
    choosed_experts = traffic_matrix_to_choosed_experts(matrix, args.G, args.topk, args.chunk_bytes)
    ntokens = choosed_experts.shape[0]
    tokens_per_rank = ntokens // W
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
    )

    # metadata-exchange result (untimed setup, same contract as the traffic test):
    # cnt[s][e] = copies source rank s sends to expert e
    src_of_copy = (
        torch.arange(ntokens, dtype=torch.long) // tokens_per_rank
    ).repeat_interleave(args.topk)
    e_of_copy = choosed_experts.reshape(-1).long().cpu()
    splits_per_source_cpu = (
        torch.bincount(src_of_copy * args.G + e_of_copy, minlength=W * args.G)
        .view(W, args.G)
        .int()
    )
    assert torch.equal(splits_per_source_cpu.sum(0), moe_ctx.splits_cpu[: args.G].cpu().int())

    rank = TP_GROUP.rank()
    epr = args.G // W

    # ---- index math (host, untimed metadata like splits/scatter_index) ----
    ce_local = choosed_experts[rank * tokens_per_rank : (rank + 1) * tokens_per_rank]
    pack_index_gpu = build_pack_index(ce_local, args.topk).cuda()
    unpack_index, split_cpu = build_unpack_index(splits_per_source_cpu, rank, args.G, W)
    unpack_index_gpu = unpack_index.cuda()
    assert torch.equal(split_cpu, moe_ctx.splits_cpu[rank * epr : (rank + 1) * epr].cpu().int())
    # wire-byte invariants
    assert int(pack_index_gpu.numel()) * args.chunk_bytes == int(matrix[rank].sum())
    assert int(unpack_index.numel()) * args.chunk_bytes == int(matrix[:, rank].sum())

    # ---- FAST bring-up (the only NVSHMEM init in this process) ----
    flash_utils = load_fast(os.path.abspath(args.fast_dir))
    uid = broadcast_uid(flash_utils)
    comm = flash_utils.flash_comm_t(rank, L, W, uid)
    if args.capacity_mib > 0:
        capacity_bytes = args.capacity_mib << 20
    else:
        capacity_bytes = 4 * int(max(matrix.sum(dim=1).max(), matrix.sum(dim=0).max()))
    capacity_bytes = (capacity_bytes + 15) // 16 * 16
    comm.alltoallv_setup(capacity_bytes)

    if rank == 0:
        rows_per_rank = moe_ctx.splits_cpu.view(W, epr).sum(dim=1)
        print(f"ntokens: {ntokens} ({tokens_per_rank} per rank), topk: {args.topk}")
        print(f"Splits: {moe_ctx.splits_cpu.tolist()}, Sum: {sum(moe_ctx.splits_cpu.tolist())}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(f"FAST wire bytes per rank (send): {matrix.sum(dim=1).tolist()}")
        print(f"FAST wire bytes per rank (recv): {matrix.sum(dim=0).tolist()}")
        print(f"FAST buffer capacity: {capacity_bytes >> 20} MiB per buffer")

    # ---- torch reference (untimed): populates ctx.inputs / scatter_inputs / outputs ----
    gemm_only_op = flux.GemmOnly(
        moe_ctx.inputs.dtype,
        moe_ctx.inputs.dtype,
        moe_ctx.outputs[0].dtype,
        use_fp8_gemm=flux.is_fp8_dtype(moe_ctx.inputs.dtype),
    )
    moe_ctx.clear_outputs()
    MoeAgScatterWithTorch.comm_impl(moe_ctx, TP_GROUP)
    MoeAgScatterWithTorch.scatter_impl(moe_ctx)
    MoeAgScatterWithTorch.gemm_impl(moe_ctx, gemm_only_op)
    torch_outputs = moe_ctx.get_outputs_clone()
    torch.cuda.synchronize()

    # ---- timed FAST baseline ----
    gemm_op = flux.GemmGroupedV2(
        moe_ctx.weights[0], epr, moe_ctx.inputs_shard.dtype, moe_ctx.outputs[0].dtype
    )
    perf_result, fast_out, fast_gemm_input = perf_fast(
        moe_ctx,
        comm,
        gemm_op,
        matrix,
        pack_index_gpu,
        unpack_index_gpu,
        split_cpu,
        args.warmup_iters,
        args.iters,
        args.sm_margin,
    )

    flux.exec_in_rank_order(TP_GROUP, lambda: print(perf_result))

    # ---- correctness ----
    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-2, 1.5e-2

    def check_result():
        print(f"Checking RANK #{rank}...")
        # data movement: FAST wire + unpack must reproduce the reference
        # scatter block bit-for-bit (strongest possible ordering check)
        input_offset = int(moe_ctx.splits_cpu[: rank * epr].sum())
        ref_block = moe_ctx.scatter_inputs[input_offset : input_offset + moe_ctx.nrows_ep]
        if flux.testing.bitwise_eq(fast_gemm_input, ref_block):
            print("✅ FAST wire + unpack bitwise-match the reference scatter block")
        else:
            raise AssertionError("❌ FAST gemm input does not match the reference scatter block")
        # same-op check: GemmGroupedV2 on the reference block must reproduce the
        # FAST-path output bit-for-bit (pure data-movement equivalence)
        ref_gg_out = gemm_op.forward(ref_block, split_cpu, sm_margin=args.sm_margin)
        if flux.testing.bitwise_eq(fast_out, ref_gg_out):
            print("✅ GemmGroupedV2(FAST input) bitwise-matches GemmGroupedV2(reference block)")
        else:
            raise AssertionError("❌ same-op outputs differ: data movement is broken")
        # numerics vs the per-expert torch.matmul loop: standard elementwise
        # allclose (|x-y| <= atol + rtol*|y|) — CUTLASS vs cuBLAS bf16 outputs
        # legitimately differ by ~1 ulp (0.8% relative), which flux.torch_allclose's
        # rtol*min(|y|) formulation would misreport as a failure
        if not torch.allclose(fast_out, torch_outputs[0], atol=atol, rtol=rtol):
            bad = (fast_out - torch_outputs[0]).abs() > atol + rtol * torch_outputs[0].abs()
            raise AssertionError(
                f"❌ allclose vs torch reference failed: {int(bad.sum())} elements"
                f" (max diff {(fast_out - torch_outputs[0]).abs().max().item():.6f})"
            )
        print("✅ FAST baseline output allclose vs torch reference")

    flux.exec_in_rank_order(TP_GROUP, check_result)

    comm.alltoallv_teardown()
    del comm
    TP_GROUP.barrier()
    torch.cuda.synchronize()
