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
"""

import argparse
import os
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
    for _ in range(warmup_iters):
        output = fn()
    torch.cuda.synchronize()
    total_time = 0
    start = time.time()
    for _ in range(iters):
        output = fn()
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    return PerfResult(name=name, output=output, gemm_time_ms=total_time / iters * 1000)


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
):
    n_dim = args.N
    assert weight.size(1) == n_dim

    input_dtype = input.dtype
    output_dtype = input_dtype
    if flux.util.get_arch() >= 90:
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
            do_all_reduce=do_all_reduce,
            use_read_mode=use_read_mode,
        )

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

    if RANK == 0:
        rows_per_rank = split_cpu.view(WORLD_SIZE, n_experts_per_rank).sum(dim=1)
        print(
            f"ntokens: {total_token_num} ({total_token_num // WORLD_SIZE} per rank),"
            f" topk: {args.topk}, M: {args.M}"
        )
        print(f"split_cpu: {split_cpu.tolist()}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")

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
        )
        perf_result_torch = perf_torch(
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
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_torch))
    flux.exec_in_rank_order(TP_GROUP, lambda: log_perf(perf_result_flux))
    atol, rtol = ABSOLUTE_THRESHOLD_MAP[input_dtype], RELATIVE_THRESHOLD_MAP[input_dtype]
    TP_GROUP.barrier()

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
            raise e
        else:
            print("✅ flux and torch matches")

    torch_output = perf_result_torch.output
    flux_output = perf_result_flux.output
    flux.exec_in_rank_order(TP_GROUP, check_result)
