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
"""COMET MoE layer1 (combine) un-overlapped baseline: grouped GEMM + FAST alltoallv.

compute -> communicate -> sum. Each rank owns G/W expert segments and holds the
GEMM inputs (M_this_ep, K); the comm-free grouped GEMM (flux.GemmGroupedV2)
produces (M_this_ep, N) partial rows — one per (token, topk-slot) copy homed
elsewhere — which then travel owner -> home over the load-balancing alltoallv
(FAST, 3rdparty/FAST: BvN decomposition into balanced inter-node permutation
steps + intra-node redistribute over NVLink), and each home rank sums its topk
partials per token into the RS-sharded output. This is the un-overlapped
counterpart of --comm_pattern a2av_hier in test_moe_gather_rs_traffic.py, the
layer1 port of test_moe_ag_scatter/test_moe_ag_fast_baseline.py.

WIRE ORIENTATION: --traffic_matrix is the layer0 DISPATCH matrix (M[s][d] =
bytes token-home rank s routes to expert-owner rank d). In the combine
direction the sender is the expert owner, so the matrix handed to FAST is the
TRANSPOSE: rank r sends column r (matrix.t()[r]) and receives row r.

PRIMARY METRIC: one end-to-end window per iteration, gemm start -> reduce
finish (gemm + pack + schedule + fill + wire + unpack/topk-sum). It is
STRUCTURALLY ISOLATED: comm.alltoallv is host-blocking and every iteration
syncs on its tail event, so no cross-iteration pipelining exists and e2e here
compares against the fused tests' `isolated` mode numbers, never their `e2e`
mode. The BvN schedule is recomputed inside the window every iteration
(one-shot methodology, never amortized); it is a flat host cost (~4.4 ms on
the 2-node Perlmutter reference) and dominates small budgets. FAST's
inter-iteration signal/credit reset runs OUTSIDE the window (reported
separately) — benchmark hygiene with no analogue in the other baselines; its
tail barrier doubles as the per-iteration rank aligner. FLUX_SWEEP_ISOLATED_ITERS
is still honored for sweep-runner uniformity (extra device-drain + rank
barrier before each window, host cost reported as iso_sync_ms).

Scales: the torch reference (moe_gather_rs_forward_torch) applies
scale_v = weight_scale[e] * input_scale[0] to each expert's GEMM output and
output_vec_scale per row; the standalone comm-free GemmGroupedV2 applies
neither for bf16/fp16, so the combined per-row scale is folded into the pack
step's multiply (in-window — the fused op pays it in its epilogue).

NVSHMEM: this test never calls flux.init_flux_shm. FAST performs the only
NVSHMEM initialization in the process (uid broadcast over torch.distributed);
flux contributes only comm-free ops and testing utilities.

Constraints: multi-node only (FAST asserts server_n > 1), 4 or 8 GPUs/node,
node-major ranks (torchrun layout), T=1 / E=world_size (each expert's full FFN
weight on one rank), N * dtype_size == chunk_bytes.

Launch (one per node):
    srun --nodes=N --ntasks-per-node=1 ./launch_fast.sh \
        test/python/moe_gather_rs/test_moe_gather_rs_fast_baseline.py \
        --traffic_matrix $PSCRATCH/.../a2av_2n_8r_dist_001.txt
"""

import argparse
import os
import sys
import time
from functools import partial

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    gen_moe_gating_args,
    generate_data,
    moe_gather_rs_forward_torch,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.recorder import RECORDER

# reuse the layer1 traffic bench's combine index math (pack_index orders this
# rank's gemm rows home-major with (expert, copy)-ordered blocks == FAST's
# destination-major send contract; reduce_index inverts the recv-panel layout)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_moe_gather_rs_traffic import build_a2av_combine_indices

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


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
    def __init__(
        self,
        name,
        e2e_ms,
        gemm_ms,
        pack_ms,
        schedule_ms,
        fill_ms,
        wire_ms,
        unpack_reduce_ms,
        reset_ms,
        host_e2e_ms,
    ):
        self.name = name
        self.e2e_ms = e2e_ms  # PRIMARY: gemm start -> topk-reduce finish (CUDA events)
        self.gemm_ms = gemm_ms
        self.pack_ms = pack_ms
        self.schedule_ms = schedule_ms
        self.fill_ms = fill_ms
        self.wire_ms = wire_ms
        self.unpack_reduce_ms = unpack_reduce_ms
        # inter-iteration signal/credit reset incl. 2x nvshmem_barrier_all —
        # OUTSIDE the window (benchmark hygiene / rank aligner; see docstring)
        self.reset_ms = reset_ms
        self.host_e2e_ms = host_e2e_ms  # host wall cross-check of e2e_ms

    def __repr__(self) -> str:
        return (
            f"{self.name}: e2e {self.e2e_ms:.3f} ms (gemm start -> reduce finish;"
            f" host {self.host_e2e_ms:.3f})"
            f" | gemm {self.gemm_ms:.3f} + pack {self.pack_ms:.3f}"
            f" + schedule {self.schedule_ms:.3f} + fill {self.fill_ms:.3f}"
            f" + wire {self.wire_ms:.3f} + unpack_reduce {self.unpack_reduce_ms:.3f}"
            f" | reset {self.reset_ms:.3f} (outside window, inter-iteration)"
        )


@torch.no_grad()
def perf_fast(
    inputs: torch.Tensor,
    gemm_op,
    comm,
    matrix_t_cpu: torch.Tensor,
    split_local_cpu: torch.Tensor,
    pack_index_gpu: torch.Tensor,
    reduce_index_gpu: torch.Tensor,
    scale_packed_gpu: torch.Tensor,
    ntok_local: int,
    topk: int,
    n_dim: int,
    warmup_iters: int,
    iters: int,
    sm_margin: int = 0,
):
    input_dtype = inputs.dtype
    total_iters = warmup_iters + iters
    ev = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    e2e_start, gemm_end, pack_end, comm_end, e2e_end = ev(), ev(), ev(), ev(), ev()
    host_e2e = [0.0] * total_iters
    reset_ms = [0.0] * total_iters
    iso_sync_ms = [0.0] * total_iters
    schedule_us = [0.0] * total_iters
    fill_us = [0.0] * total_iters
    wire_us = [0.0] * total_iters
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))

    out = packed = recv_u8 = None
    torch.distributed.barrier()
    torch.cuda.synchronize()
    for i in range(total_iters):
        # inter-iteration hygiene, OUTSIDE the window; not needed before the
        # first call but harmless — its tail barrier aligns the ranks
        t_r0 = time.perf_counter()
        comm.alltoallv_reset()
        reset_ms[i] = (time.perf_counter() - t_r0) * 1e3
        torch.cuda.synchronize()
        if isolated:
            # sweep-runner uniformity: drain + realign right before the window
            # (mostly redundant here — this baseline is structurally isolated —
            # so iso_sync_ms doubles as a residual-straggler indicator)
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_ms[i] = (time.perf_counter() - t_iso) * 1e3

        t0 = time.perf_counter()
        e2e_start[i].record()
        # 1. un-overlapped grouped GEMM over the local expert segments
        gemm_out = gemm_op.forward(inputs, split_local_cpu, sm_margin=sm_margin)
        gemm_end[i].record()
        # 2. pack home-major + fold the reference's scale_v * output_vec_scale
        packed = torch.index_select(gemm_out, dim=0, index=pack_index_gpu)
        packed.mul_(scale_packed_gpu)
        pack_end[i].record()
        # 3. host-synchronous: schedule (BvN recompute) + fill + wire to completion
        recv_u8, out_sz, timings = comm.alltoallv(packed.view(torch.uint8), matrix_t_cpu)
        comm_end[i].record()
        # 4. unpack to (token, slot) order and topk-sum; same op/dtype as the
        # reference's full_output.view(ntok, topk, N).sum(1) (bf16 tensor sum,
        # fp32 op-math accumulation)
        out = (
            recv_u8.view(input_dtype)
            .view(-1, n_dim)
            .index_select(0, reduce_index_gpu)
            .view(ntok_local, topk, n_dim)
            .sum(1)
        )
        e2e_end[i].record()
        e2e_end[i].synchronize()
        host_e2e[i] = (time.perf_counter() - t0) * 1e3
        schedule_us[i], fill_us[i], wire_us[i] = timings.tolist()

    def mean_ms(starts, ends):
        return (
            sum(starts[i].elapsed_time(ends[i]) for i in range(warmup_iters, total_iters)) / iters
        )

    def mean_host_ms(vals_us):
        return sum(vals_us[warmup_iters:]) / iters / 1e3

    def per_iter_ms(starts, ends):
        return [starts[i].elapsed_time(ends[i]) for i in range(warmup_iters, total_iters)]

    result = FastPerfResult(
        name=f"fast #{TP_GROUP.rank()}",
        e2e_ms=mean_ms(e2e_start, e2e_end),
        gemm_ms=mean_ms(e2e_start, gemm_end),
        pack_ms=mean_ms(gemm_end, pack_end),
        schedule_ms=mean_host_ms(schedule_us),
        fill_ms=mean_host_ms(fill_us),
        wire_ms=mean_host_ms(wire_us),
        unpack_reduce_ms=mean_ms(comm_end, e2e_end),
        reset_ms=sum(reset_ms[warmup_iters:]) / iters,
        host_e2e_ms=sum(host_e2e[warmup_iters:]) / iters,
    )
    result.iter_times = {
        "e2e_ms": per_iter_ms(e2e_start, e2e_end),
        "gemm_ms": per_iter_ms(e2e_start, gemm_end),
        "pack_ms": per_iter_ms(gemm_end, pack_end),
        "unpack_reduce_ms": per_iter_ms(comm_end, e2e_end),
        "schedule_ms": [v / 1e3 for v in schedule_us[warmup_iters:]],
        "fill_ms": [v / 1e3 for v in fill_us[warmup_iters:]],
        "wire_ms": [v / 1e3 for v in wire_us[warmup_iters:]],
        "reset_ms": reset_ms[warmup_iters:],
        "host_e2e_ms": host_e2e[warmup_iters:],
    }
    if isolated:
        result.iter_times["iso_sync_ms"] = iso_sync_ms[warmup_iters:]
    return result, out, packed, recv_u8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traffic_matrix",
        type=str,
        required=True,
        help="layer0 DISPATCH traffic matrix file; the combine wire is its transpose",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=int,
        default=8192,
        help="bytes of one routed token copy in the traffic matrix (N * dtype size)",
    )
    parser.add_argument("-N", type=int, default=4096, help="model hidden dim (gemm output)")
    parser.add_argument("-K", type=int, default=4096, help="ffn hidden dim (gemm input)")
    parser.add_argument("-G", type=int, default=32, help="number of experts")
    parser.add_argument("--topk", type=int, default=4)
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
        " (4 x max(row sum, col sum) of the matrix — transpose-symmetric)",
    )
    parser.add_argument(
        "--fast_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "3rdparty",
            "FAST",
            "nvidia",
        ),
        help="directory containing libflash.so + flash_utils.py",
    )
    parser.add_argument(
        "--skip_correctness",
        default=False,
        action="store_true",
        help="skip the untimed torch reference and all result checks — the"
        " reference materializes two dense (ntokens * topk, N) buffers and OOMs"
        " at large budgets; correctness is then NOT verified",
    )
    return parser.parse_args()


ABSOLUTE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

RELATIVE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


if __name__ == "__main__":
    args = parse_args()
    torch.use_deterministic_algorithms(False)

    W = DIST_ENV.WORLD_SIZE
    L = DIST_ENV.LOCAL_WORLD_SIZE
    rank = TP_GROUP.rank()
    assert L in (4, 8), f"FAST baseline expects 4 (Perlmutter) or 8 (p4d) GPUs/node; got {L}"
    assert W > L, "FAST requires at least 2 nodes (server_n > 1)"

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.N * input_dtype.itemsize == args.chunk_bytes, (
        f"N ({args.N}) * dtype size ({input_dtype.itemsize}) must equal the traffic matrix"
        f" chunk granularity ({args.chunk_bytes} bytes)"
    )
    assert args.G % W == 0, f"{args.G} % {W} != 0"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W, f"matrix is for {matrix.shape[0]} ranks, world size {W}"
    choosed_experts = traffic_matrix_to_choosed_experts(matrix, args.G, args.topk, args.chunk_bytes)
    ntokens = choosed_experts.shape[0]
    assert ntokens % W == 0
    tokens_per_rank = ntokens // W
    cpr = tokens_per_rank * args.topk  # copies (== recv rows) per home rank
    M = ntokens * args.topk

    gating_args = gen_moe_gating_args(args.G, args.topk, ntokens, choosed_experts=choosed_experts)
    split_cpu = gating_args.splits_gpu.to("cpu")
    token_index = gating_args.gather_index
    topk_index = gating_args.topk_index
    routing_idx = gating_args.scatter_index.flatten()

    epr = args.G // W
    eid_start = rank * epr
    eid_end = eid_start + epr
    ep_rank_m_start = int(split_cpu[:eid_start].sum())
    m_this_ep = int(split_cpu[eid_start:eid_end].sum())
    ep_rank_m_end = ep_rank_m_start + m_this_ep
    split_local_cpu = split_cpu[eid_start:eid_end]

    # ---- index math (host, untimed metadata like splits/scatter_index; the
    # same convention as the layer0 FAST baseline — a fused pipeline inherits
    # this plan from layer0, and FAST's in-window BvN schedule already charges
    # the one-shot scheduling cost) ----
    pack_index_gpu, reduce_index_gpu = build_a2av_combine_indices(
        routing_idx, split_cpu, rank, W, args.topk
    )
    # combine wire = transpose of the dispatch matrix: owner r sends column r
    matrix_t = matrix.t().contiguous()
    # wire-byte invariants: send bytes = my dispatch column, recv = my row
    assert int(pack_index_gpu.numel()) == m_this_ep
    assert m_this_ep * args.chunk_bytes == int(matrix[:, rank].sum())
    assert int(reduce_index_gpu.numel()) == cpr
    assert cpr * args.chunk_bytes == int(matrix[rank].sum())

    # ---- inputs/weights/scales: identical conventions to the layer1 traffic
    # bench (T=1, E=W), so the torch reference comparison is apples-to-apples
    data_config = [
        ((m_this_ep, args.K), input_dtype, (0.1, 0.0)),  # input
        ((epr, args.N, args.K), input_dtype, (0.1, 0.0)),  # weight
        ((epr,), torch.float32, (1, 0)),  # weight_scale
        ((1,), torch.float32, (1, 0)),  # input_scale
        ((m_this_ep,), torch.float32, (1, 0)),  # output_vec_scale
    ]
    inputs, weights, weight_scales, input_scales, output_vec_scales = next(
        generate_data(data_config)
    )

    # per-row combined scale in packed order (see docstring "Scales"):
    # scale_v[expert_of_row] * output_vec_scale[row], permuted by pack_index
    e_of_row_local = torch.repeat_interleave(
        torch.arange(epr, dtype=torch.long), split_local_cpu.long()
    ).cuda()
    row_scale = weight_scales.index_select(0, e_of_row_local) * input_scales[0]
    row_scale = row_scale * output_vec_scales
    scale_packed_gpu = row_scale.index_select(0, pack_index_gpu.long()).unsqueeze(1)

    # ---- FAST bring-up (the only NVSHMEM init in this process) ----
    flash_utils = load_fast(os.path.abspath(args.fast_dir))
    uid = broadcast_uid(flash_utils)
    comm = flash_utils.flash_comm_t(rank, L, W, uid)
    if args.capacity_mib > 0:
        capacity_bytes = args.capacity_mib << 20
    else:
        # identical to the layer0 baseline: 4 x max(row sum, col sum) is
        # invariant under transposition, so the same matrix yields the same
        # capacity in both directions
        capacity_bytes = 4 * int(max(matrix.sum(dim=1).max(), matrix.sum(dim=0).max()))
    capacity_bytes = (capacity_bytes + 15) // 16 * 16
    comm.alltoallv_setup(capacity_bytes)

    if rank == 0:
        rows_per_rank = split_cpu.view(W, epr).sum(dim=1)
        print(f"ntokens: {ntokens} ({tokens_per_rank} per rank), topk: {args.topk}, M: {M}")
        print(f"Splits: {split_cpu.tolist()}, Sum: {sum(split_cpu.tolist())}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(f"FAST wire bytes per rank (send): {matrix.sum(dim=0).tolist()}")
        print(f"FAST wire bytes per rank (recv): {matrix.sum(dim=1).tolist()}")
        print(f"FAST buffer capacity: {capacity_bytes >> 20} MiB per buffer")
        RECORDER.emit_info(
            ntokens=ntokens,
            tokens_per_rank=tokens_per_rank,
            gemm_rows_per_rank=rows_per_rank.tolist(),
            fast_send_bytes=matrix.sum(dim=0).tolist(),
            fast_recv_bytes=matrix.sum(dim=1).tolist(),
            capacity_bytes=capacity_bytes,
        )

    # ---- timed FAST baseline ----
    gemm_op = flux.GemmGroupedV2(weights, epr, input_dtype, output_dtype)
    perf_result, fast_out, last_packed, last_recv_u8 = perf_fast(
        inputs,
        gemm_op,
        comm,
        matrix_t,
        split_local_cpu,
        pack_index_gpu,
        reduce_index_gpu,
        scale_packed_gpu,
        tokens_per_rank,
        args.topk,
        args.N,
        args.warmup_iters,
        args.iters,
        args.sm_margin,
    )

    flux.exec_in_rank_order(TP_GROUP, lambda: print(perf_result))
    RECORDER.emit_iters("fast", perf_result.iter_times)

    # ---- correctness ----
    if not args.skip_correctness:
        # tier 1 (collective part): torch alltoallv on the identical packed rows
        # must reproduce FAST's recv buffer bit-for-bit — both contracts are
        # source-major segments in global rank order, each in the sender's
        # send order restricted to me
        in_splits = (matrix_t[rank] // args.chunk_bytes).tolist()
        out_splits = (matrix_t[:, rank] // args.chunk_bytes).tolist()
        a2a_out = torch.empty(cpr, args.chunk_bytes, dtype=torch.uint8, device="cuda")
        torch.distributed.all_to_all_single(
            a2a_out,
            last_packed.view(torch.uint8),
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=TP_GROUP,
        )
        fast_recv_panel = last_recv_u8.view(-1)[: cpr * args.chunk_bytes].view(
            cpr, args.chunk_bytes
        )
        # tier 2 (collective part): untimed torch reference (per-expert matmul
        # + scatter + topk-sum + ring reduce-scatter)
        ref_out = moe_gather_rs_forward_torch(
            TP_GROUP,
            M,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
            inputs,
            weights,
            split_cpu,
            token_index,
            topk_index,
            args.topk,
            input_scales,
            weight_scales,
            output_vec_scales,
            do_all_reduce=False,
            fast_acc=False,
        )
        torch.cuda.synchronize()

        # Numerics tolerance (same maps as test_moe_gather_rs_traffic.py):
        # GemmGroupedV2 (CUTLASS) vs the per-expert torch.matmul loop (cuBLAS)
        # differ in accumulation order, the reference folds its scales in two
        # rounding steps vs our one, and it sums cross-rank partials via a bf16
        # NCCL reduce-scatter while we topk-sum all partials in one fp32-opmath
        # reduction. Exact data movement is covered by the bitwise wire check.
        atol = ABSOLUTE_THRESHOLD_MAP[input_dtype]
        rtol = RELATIVE_THRESHOLD_MAP[input_dtype]

        def check_result():
            print(f"Checking RANK #{rank}...")
            if flux.testing.bitwise_eq(fast_recv_panel, a2a_out):
                print("✅ FAST wire bitwise-matches torch all_to_all_single")
            else:
                RECORDER.emit_correctness(bitwise=False, allclose=False)
                raise AssertionError("❌ FAST recv buffer does not match torch alltoallv")
            diff = (fast_out.float() - ref_out.float()).abs()
            print(f"   max |diff|: {diff.max().item():.6f}")
            try:
                flux.torch_allclose(fast_out, ref_out, atol=atol, rtol=rtol)
            except Exception:
                RECORDER.emit_correctness(bitwise=True, allclose=False)
                print("❌ FAST baseline output does not match torch reference")
                raise
            print("✅ FAST baseline output allclose vs torch reference")
            RECORDER.emit_correctness(bitwise=True, allclose=True)

        flux.exec_in_rank_order(TP_GROUP, check_result)

    comm.alltoallv_teardown()
    del comm
    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
