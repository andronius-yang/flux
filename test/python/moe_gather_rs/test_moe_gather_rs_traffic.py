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
"""COMET MoE layer1 (grouped GEMM + gather + topk-reduce + reduce-scatter) driven
by a traffic matrix.

Routing is built so that tokens homed on rank s (rank s owns output rows
[s * ntokens_per_rank, (s+1) * ntokens_per_rank) after reduce-scatter) choose
experts owned by rank d for exactly M[s][d] / chunk_bytes (token, topk-slot)
copies, where chunk_bytes = N * dtype_size (N is the GEMM output dim = model
hidden). TP/EP is fixed to T=1, E=world_size so each expert's full FFN weight
resides on one rank.

The gather-RS payload delivered from expert-owner rank d to token-home rank s is
therefore exactly M[s][d] bytes (the transpose of the matrix); the hierarchical
multi-node implementation may combine partial sums en route, so inter-node wire
bytes can fall below that logical payload.

--comm_pattern a2av_hier runs the hierarchical a2av combine instead of the dense
ring reduce-scatter: each generated [1, hidden] copy travels once from its
expert-owner rank back to its token-home rank (wire bytes = the matrix
transpose), with at most one inter-node relay via the same-local-rank gateway,
and the topk reduction happens per split at the destination. Requires
splits_per_source (built here in untimed setup, mirroring the layer0 harness).
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Union

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
    gen_moe_gating_args,
    moe_gather_rs_forward_torch,
    generate_data,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.perf_db_helper import log_perf, set_global_args, should_log_to_rds
from flux.testing.recorder import RECORDER
from flux.testing.traffic_matrix import choosed_experts_to_matrix_chunks, load_routing_file
from flux.util import get_arch


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms
        self.total_ms = self.gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"


def perf_gemm(iters: int, warmup_iters: int, name: str, fn: callable):
    """Per-iteration CUDA-event timing, mirroring the layer0 harness: warmup
    iterations run inside the same loop and are filtered at collection; NVTX
    tags segment warmup vs timed for nsys captures without device syncs; the
    sweeps isolated mode (FLUX_SWEEP_ISOLATED_ITERS) drains the device and
    aligns all ranks before EVERY timed window, reporting the host wall time of
    the sync pair separately (per-rank straggler indicator)."""
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    for i in range(total_iters):
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        start_events[i].record()
        with torch.cuda.nvtx.range(nvtx_tag):
            output = fn()
        end_events[i].record()
    times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            times.append(start_events[i].elapsed_time(end_events[i]))
    result = PerfResult(name=name, output=output, gemm_time_ms=sum(times) / iters)
    result.iter_times = {"e2e_ms": times}
    if isolated:
        result.iter_times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return result


def perf_torch(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    split_cpu: torch.Tensor,
    iters: int,
    warmup_iters: int,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk: int,
    input_scales: Union[torch.Tensor, None],
    weight_scales: Union[torch.Tensor, None],
    output_vec_scales: Union[torch.Tensor, None],
    do_all_reduce: bool = False,
):
    return perf_gemm(
        iters,
        warmup_iters,
        f"torch #{TP_GROUP.rank()}",
        lambda: moe_gather_rs_forward_torch(
            TP_GROUP,
            args.M,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
            inputs,
            weights,
            split_cpu,
            token_index,
            topk_index,
            topk,
            input_scales,
            weight_scales,
            output_vec_scales,
            do_all_reduce,
            fast_acc=args.fastacc,
        ),
    )


# Combine index builders live in flux.testing.a2av_combine_indices (moved
# 2026-08-17 for the EPIC baseline's combine-only protocol); re-imported here
# so this driver and its downstream importers (test_moe_l0l1_traffic.py)
# keep their historical import surface.
from flux.testing.a2av_combine_indices import (  # noqa: E402,F401
    build_a2av_combine_indices,
    build_a2av_unique_counts,
)


from flux.testing.a2av_combine_indices import (  # noqa: E402,F401
    build_a2av_compress_indices,
)

def perf_flux(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    iters: int,
    warmup_iters: int,
    max_m: int,
    topk: int,
    routing_idx: torch.Tensor,
    input_scale: Union[torch.Tensor, None],
    weight_scale: Union[torch.Tensor, None],
    output_vec_scale: Union[torch.Tensor, None],
    do_all_reduce: bool = False,
    use_read_mode: bool = False,
    splits_per_source: Union[torch.Tensor, None] = None,
    unique_counts: Union[torch.Tensor, None] = None,
):
    n_dim = args.N
    assert weight.size(1) == n_dim
    use_compress = args.comm_pattern == "a2av_hier_compress"
    use_a2av = args.comm_pattern == "a2av_hier" or use_compress

    input_dtype = input.dtype
    output_dtype = input_dtype
    if flux.util.get_arch() >= 90:
        assert not use_a2av, "a2av_hier is a V2/sm80 mode"
        op = flux.GemmGroupedV3GatherRS(
            args.G,
            max_m,
            n_dim,
            topk,
            RANK,
            WORLD_SIZE,
            args.T,
            args.E,
            1,
            False,
        )
    else:
        op = flux.GemmGroupedV2GatherRSOp(
            TP_GROUP,
            args.G,
            max_m,
            n_dim,
            topk,
            output_dtype,
            args.T,
            args.E,
            1,
            nnodes=NNODES,
            n_split=args.n_split,
            do_all_reduce=do_all_reduce,
            use_read_mode=use_read_mode,
            a2av_hier=use_a2av and not use_compress,
            a2av_hier_compress=use_compress,
        )

    a2av_kwargs = {}
    if use_a2av:
        assert splits_per_source is not None
        a2av_kwargs["splits_per_source"] = splits_per_source
        if use_compress and NNODES > 1:
            # untimed host metadata (like splits_per_source) in BOTH timing modes
            assert unique_counts is not None
            a2av_kwargs["a2av_unique_counts"] = unique_counts
        if args.timing_mode == "amortized":
            # layer0-amortized mode: hand the whole routing plan in precomputed,
            # as a fused layer0+layer1 pipeline would hand over layer0's tensors;
            # isolated mode leaves the index math on the timed critical path
            pack_index, reduce_index = build_a2av_combine_indices(
                routing_idx, split_cpu, RANK, WORLD_SIZE, topk
            )
            a2av_kwargs["a2av_pack_index"] = pack_index
            a2av_kwargs["a2av_reduce_index"] = reduce_index
            if use_compress and NNODES > 1:
                wire_ptr, wire_copy, red_ptr, red_row = build_a2av_compress_indices(
                    routing_idx, split_cpu, unique_counts, RANK, WORLD_SIZE, NNODES, topk
                )
                a2av_kwargs["a2av_wire_csr"] = [wire_ptr, wire_copy]
                a2av_kwargs["a2av_reduce_csr"] = [red_ptr, red_row]

    def fn():
        is_v2 = get_arch() < 90
        extra_args = {"bias": None} if is_v2 else {"n_tokens_per_rank": None}
        return op.forward_gather_rs(
            input,
            weight,
            split_cpu,
            routing_idx,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_vec_scale=output_vec_scale,
            fast_accum=args.fastacc,
            sm_margin=args.sm_margin,
            **extra_args,
            **a2av_kwargs,
        )

    return perf_gemm(iters, warmup_iters, f"flux #{TP_GROUP.rank()}", fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True, help="traffic matrix file")
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
    parser.add_argument("--warmup_iters", default=5, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=["bfloat16", "float16"]
    )
    parser.add_argument(
        "--fastacc", default=False, action="store_true", help="whether to enbale fast accumulate"
    )
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument(
        "--all_reduce",
        default=False,
        action="store_true",
        help="whether to use all_reduce (single-node only)",
    )
    parser.add_argument("--use_read_mode", default=False, action="store_true")
    parser.add_argument(
        "--comm_pattern",
        default="dense",
        choices=["dense", "a2av_hier", "a2av_hier_compress"],
        help="dense: ring reduce-scatter (default). a2av_hier: hierarchical alltoallv"
        " combine -- every copy travels owner->home once (wire bytes = matrix transpose),"
        " one inter-node relay via same-local-rank gateways, per-split topk reduce at the"
        " destination. a2av_hier_compress: token-dedup combine -- the source node's"
        " copies converge on the same-lr gateway, are pre-reduced per token, and ONE"
        " partial per (token, source node) crosses the wire (degrades to a2av_hier on a"
        " single node)",
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=4,
        help="split-N pipeline depth (N/n_split must be a multiple of 1024)",
    )
    parser.add_argument(
        "--timing_mode",
        default="isolated",
        choices=["isolated", "amortized"],
        help="isolated: the op builds the routing plan (indices + compress CSRs)"
        " in-forward, so schedule computation is on the timed critical path (layer1"
        " alone must derive it). amortized: the harness precomputes everything a fused"
        " layer0+layer1 pipeline would inherit from layer0 and passes it in untimed.",
    )
    parser.add_argument(
        "--precomputed_indices",
        default=False,
        action="store_true",
        help="deprecated alias for --timing_mode amortized",
    )
    parser.add_argument(
        "--routing_file",
        type=str,
        default=None,
        help="real routing trace sidecar (ntokens topk G header + one token per line);"
        " must realize --traffic_matrix exactly, mirroring the layer0 harness",
    )
    parser.add_argument(
        "--skip_correctness",
        default=False,
        action="store_true",
        help="skip the torch reference (perf_torch + result checks); needed at large"
        " budgets where the reference materializes full gathered buffers and OOMs"
        " (mirrors the layer0 harness flag). Correctness columns in the sweep capsule"
        " stay empty for the cell.",
    )
    args = parser.parse_args()
    if args.precomputed_indices:
        args.timing_mode = "amortized"
    return args


ABSOLUTE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

RELATIVE_THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    torch.use_deterministic_algorithms(False)
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    args = parse_args()
    # each expert's full ffn weight resides on one rank
    args.T = 1
    args.E = WORLD_SIZE

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.N * input_dtype.itemsize == args.chunk_bytes, (
        f"N ({args.N}) * dtype size ({input_dtype.itemsize}) must equal the traffic matrix"
        f" chunk granularity ({args.chunk_bytes} bytes)"
    )
    assert args.G % WORLD_SIZE == 0, f"{args.G} % {WORLD_SIZE} != 0"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == WORLD_SIZE, (
        f"traffic matrix is for {matrix.shape[0]} ranks but world size is {WORLD_SIZE}"
    )
    if args.routing_file:
        # real routing trace (must realize the matrix), mirroring the layer0 harness
        choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
        assert choosed_experts.shape[0] % WORLD_SIZE == 0
        got = choosed_experts_to_matrix_chunks(
            choosed_experts, WORLD_SIZE, args.G // WORLD_SIZE
        )
        assert torch.equal(got * args.chunk_bytes, matrix), (
            f"routing file {args.routing_file} does not realize --traffic_matrix"
            f" {args.traffic_matrix}"
        )
        if torch.cuda.is_available():
            choosed_experts = choosed_experts.cuda()
        if RANK == 0:
            print(f"routing: REAL trace file {args.routing_file}")
    else:
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
    total_token_num = choosed_experts.shape[0]
    assert total_token_num % WORLD_SIZE == 0
    if NNODES > 1:
        assert total_token_num % (WORLD_SIZE * args.topk) == 0, (
            f"multi-node requires token count ({total_token_num}) divisible by"
            f" world_size * topk ({WORLD_SIZE * args.topk}): per-source row budget in chunks"
            f" must be divisible by topk^2"
        )

    gating_args = gen_moe_gating_args(
        args.G, args.topk, total_token_num, choosed_experts=choosed_experts
    )
    split_cpu = gating_args.splits_gpu.to("cpu")
    token_index = gating_args.gather_index
    topk_index = gating_args.topk_index
    routing_idx = gating_args.scatter_index.flatten()
    args.M = total_token_num * args.topk

    local_K = args.K // args.T
    n_experts_per_rank = args.G // args.E
    ep_rank = TP_GROUP.rank() // args.T
    eid_start = ep_rank * n_experts_per_rank
    eid_end = eid_start + n_experts_per_rank
    ep_rank_m_start = 0
    for i in range(eid_start):
        ep_rank_m_start += split_cpu[i]
    M_cur_ep_rank = torch.sum(split_cpu[eid_start:eid_end]).item()
    ep_rank_m_end = ep_rank_m_start + M_cur_ep_rank

    use_compress = args.comm_pattern == "a2av_hier_compress"
    use_a2av = args.comm_pattern == "a2av_hier" or use_compress
    if use_a2av:
        assert not args.all_reduce, "a2av_hier does not support all_reduce"
        assert not args.use_read_mode, "a2av_hier does not support use_read_mode"
    if use_compress and NNODES == 1 and RANK == 0:
        print("a2av_hier_compress on a single node degrades to plain a2av_hier")
    # combine tile is 1024 or 512 (2026-08-21, K3 H=3584)
    assert args.N % args.n_split == 0 and (args.N // args.n_split) % 512 == 0, (
        f"N ({args.N}) / n_split ({args.n_split}) must be a multiple of 512"
    )

    # metadata-exchange result (untimed setup), mirroring the layer0 harness:
    # cnt[s][e] = copies token-home rank s routed to expert e. The combine op
    # derives its chunk matrix as the transpose-aggregate of the same input.
    tokens_per_rank = total_token_num // WORLD_SIZE
    src_of_copy = (
        torch.arange(total_token_num, dtype=torch.long) // tokens_per_rank
    ).repeat_interleave(args.topk)
    e_of_copy = choosed_experts.reshape(-1).long().cpu()
    splits_per_source_cpu = (
        torch.bincount(src_of_copy * args.G + e_of_copy, minlength=WORLD_SIZE * args.G)
        .view(WORLD_SIZE, args.G)
        .int()
    )
    assert torch.equal(
        splits_per_source_cpu.sum(0), split_cpu[: args.G].int()
    ), "splits_per_source column sums must equal splits"
    # compress dedup counts (transposed U), untimed host metadata like
    # splits_per_source; on a single node the degrade path never consumes them
    unique_counts_cpu = None
    if use_compress and NNODES > 1:
        unique_counts_cpu = build_a2av_unique_counts(
            choosed_experts, WORLD_SIZE, NNODES, args.G // args.E
        )

    if RANK == 0:
        rows_per_rank = split_cpu.view(WORLD_SIZE, n_experts_per_rank).sum(dim=1)
        print(
            f"ntokens: {total_token_num} ({total_token_num // WORLD_SIZE} per rank),"
            f" topk: {args.topk}, M: {args.M}"
        )
        print(f"split_cpu: {split_cpu.tolist()}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(f"comm_pattern: {args.comm_pattern}, n_split: {args.n_split}")
        if use_a2av:
            # combine direction: the wire sender is the expert owner, so per-rank
            # wire bytes are the TRANSPOSE of the dispatch matrix (owner r sends
            # column r of M, receives row r)
            send_bytes = (matrix.sum(dim=0) - matrix.diag()).tolist()
            recv_bytes = (matrix.sum(dim=1) - matrix.diag()).tolist()
            print(f"a2av combine wire bytes per rank (send): {send_bytes}")
            print(f"a2av combine wire bytes per rank (recv): {recv_bytes}")
            # inter-node wire in hier mode: per owner rank, bytes for remote home
            # NODES travel as one aggregated message per node (matrix.T viewed by
            # destination-node blocks, own node excluded); gateway forwarding is
            # extra NVLink traffic on top
            L = WORLD_SIZE // NNODES
            mt = matrix.t().contiguous()
            per_node = mt.view(WORLD_SIZE, NNODES, L).sum(dim=2)
            src_node = torch.arange(WORLD_SIZE) // L
            inter_bytes = per_node.sum(dim=1) - per_node.gather(
                1, src_node.view(-1, 1)
            ).squeeze(1)
            print(f"a2av_hier inter-node wire bytes per rank (send): {inter_bytes.tolist()}")

    data_config = [
        ((M_cur_ep_rank, local_K), input_dtype, (0.1, 0.0)),  # input
        ((n_experts_per_rank, args.N, local_K), input_dtype, (0.1, 0.0)),  # weight
        ((n_experts_per_rank,), torch.float32, (1, 0)),  # weight_scale
        ((1,), torch.float32, (1, 0)),  # input_scale
        ((M_cur_ep_rank,), torch.float32, (1, 0)),  # output_scale
    ]

    inputs, weights, weight_scales, input_scales, output_vec_scales = next(
        generate_data(data_config)
    )
    torch.distributed.barrier()

    with flux.group_profile(
        name="moe_gather_rs_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        perf_result_flux = perf_flux(
            inputs,
            weights,
            split_cpu,
            args.iters,
            args.warmup_iters,
            args.M,
            args.topk,
            routing_idx,
            input_scales,
            weight_scales,
            output_vec_scales,
            args.all_reduce,
            args.use_read_mode,
            splits_per_source_cpu if use_a2av else None,
            unique_counts_cpu,
        )
        perf_result_torch = (
            None
            if args.skip_correctness
            else perf_torch(
                inputs,
                weights,
                split_cpu,
                args.iters,
                args.warmup_iters,
                token_index,
                topk_index,
                args.topk,
                input_scales,
                weight_scales,
                output_vec_scales,
                args.all_reduce,
            )
        )

    if TP_GROUP.rank() == 0:
        flux.testing.print_grouped_gemm_sol_time_ms(
            args.M // args.E,
            args.N,
            local_K,
            n_experts_per_rank,
            input_dtype,
        )
    TP_GROUP.barrier()
    if should_log_to_rds():
        set_global_args("moe_gather_rs_traffic", args)
    if perf_result_torch is not None:
        flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))
    if RANK == 0:
        RECORDER.emit_info(
            ntokens=int(total_token_num),
            topk=int(args.topk),
            comm_pattern=args.comm_pattern,
            timing_mode=args.timing_mode,
            n_split=int(args.n_split),
        )
    if perf_result_torch is not None:
        RECORDER.emit_iters("torch", getattr(perf_result_torch, "iter_times", {}))
    RECORDER.emit_iters("flux", getattr(perf_result_flux, "iter_times", {}))
    TP_GROUP.barrier()

    if args.skip_correctness:
        # No torch reference ran: emit no correctness record so the sweep
        # capsule's correct_* columns stay empty (layer0 harness behavior).
        RECORDER.flush()
        sys.exit(0)

    atol, rtol = ABSOLUTE_THRESHOLD_MAP[input_dtype], RELATIVE_THRESHOLD_MAP[input_dtype]

    def check_result():
        print(f"#{TP_GROUP.rank()} Threshold = Atol:{atol}  Rtol:{rtol}")
        print(f"flux  output shape: {flux_output.size()}")
        print(f"torch output shape: {torch_output.size()}")

        try:
            flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            torch.save(flux_output, f"flux_output_{TP_GROUP.rank()}.pt")
            torch.save(torch_output, f"torch_output_{TP_GROUP.rank()}.pt")
            print("❌ flux and torch not matches")
            RECORDER.emit_correctness(bitwise=False, allclose=False)
            RECORDER.flush()
            raise e
        else:
            print("✅ flux and torch matches")

    torch_output = perf_result_torch.output
    flux_output = perf_result_flux.output
    flux.exec_in_rank_order(TP_GROUP, check_result)
    RECORDER.emit_correctness(bitwise=False, allclose=True)
    RECORDER.flush()
