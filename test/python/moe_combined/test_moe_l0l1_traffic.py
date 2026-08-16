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
"""COMET MoE combined layer0+layer1 continuous benchmark, driven by a traffic
matrix.

One timed window per iteration contains a FULL MoE layer pass:

    layer0 (all-gather / a2av dispatch + scatter + grouped GEMM0,
            GemmGroupedV2AGScatterOp)
      -> GELU on the [gemm_rows_this_ep, ffn] intermediate
      -> layer1 (grouped GEMM1 + gather + topk-reduce + reduce-scatter,
                 GemmGroupedV2GatherRSOp)

Window-accounting contract (the reason this bench exists — the l01 arm of the
layer-axis campaign, cells.csv layer=l01):

- Routing/scheduling is computed ONCE per window, inside layer0's forward
  (its in-window stage2 metadata work). Nothing routing-derived is cached
  across iterations (one-shot inference semantics).
- Layer1 consumes PRECOMPUTED inverse indices (pack/reduce indices + compress
  CSRs, the l1 bench's `--timing_mode amortized` contract): a real fused
  pipeline inherits them ~free from layer0's in-window stage2 work, so the
  python builders (build_a2av_combine_indices / build_a2av_compress_indices)
  run OUTSIDE the timed window here. Their one-shot wall time is reported via
  rank-0 cell_info `l1_index_build_ms` — never silently dropped.
  `--l1_index_in_window` moves those same python builders INSIDE the window
  (into l1_ms) for sensitivity runs; note they contain GPU->CPU copies of
  routing_idx, so in-window builds serialize the host on the l0+GELU tail by
  construction.
- splits_per_source (cnt[s][e]) and the dedup unique-count matrices stay
  untimed host metadata in BOTH modes, exactly like the per-layer benches
  (a real system computes them locally from the gating allgather).

Per-iteration CUDA events: e2e_start / l0_end / act_end / e2e_end, emitted as
e2e_ms / l0_ms / act_ms / l1_ms (+ iso_sync_ms under FLUX_SWEEP_ISOLATED_ITERS,
same drain+barrier-before-every-window discipline as the per-layer benches).

Validation identity (checked offline by sweeps/check_l01_identity.py against a
capsule holding an l0 isolated cell, an l1 tmamo isolated cell, and an l01
isolated cell on the SAME matrix/build):

    mean_iters(max_ranks e2e_ms[l01])
      ~= mean_iters(max_ranks e2e_ms[l0])
       + mean_iters(max_ranks act_ms[l01])
       + mean_iters(max_ranks e2e_ms[l1 tmamo])

within tolerance — the continuous pass should cost its parts; a residual is a
finding (lost overlap, sync amortization), not noise to hide.

Traffic-matrix orientation: --traffic_matrix is the layer0 DISPATCH matrix
(M[s][d] = bytes token-home rank s routes to expert-owner rank d,
chunk_bytes = H * dtype_size). The layer1 combine direction interprets the SAME
matrix as its transpose (owner d sends column d back to home s), exactly as
test_moe_gather_rs_traffic.py documents. TP/EP fixed to T=1, E=world_size.

Arms (--impl):
- flux : the fused pairing above.
- torch: fully unfused two-layer reference (MoeAgScatterWithTorch comm/scatter/
  gemm -> GELU -> moe_gather_rs_forward_torch), same window discipline and
  event names, emitted impl="torch". It materializes the full (ntokens, H)
  gathered input, the (ntokens*topk, H) scatter staging buffer and a
  (ntokens*topk, H) layer1 staging output — watch memory at large budgets.
- fast : NOT IMPLEMENTED yet (stub below).

Correctness (skippable via --skip_correctness): the flux arm's final
RS-sharded output is compared against ONE untimed run of the torch two-layer
reference, thresholds from the l1 bench maps.

2n bring-up (one launch.sh per node; matrices live on $PSCRATCH, logs in /tmp):

    salloc -A m5350_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
    srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 ./launch.sh \\
        test/python/moe_combined/test_moe_l0l1_traffic.py \\
        --traffic_matrix $PSCRATCH/a2av_test_matrices/<matrix>.txt \\
        --l0_comm_pattern a2av_hier --l1_comm_pattern a2av_hier
"""

import argparse
import os
import sys
import time
from functools import partial
from typing import Any, List, Optional

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    MoeAgScatterWithTorch,
    MoeMlp1Ctx,
    gen_moe_gating_args,
    generate_data,
    choosed_experts_to_matrix_chunks,
    load_routing_file,
    moe_gather_rs_forward_torch,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.recorder import RECORDER

# reuse the layer1 traffic bench's inherited-index math and correctness
# thresholds (import via sys.path like the fast baselines do): the builders are
# the executable spec of what a fused layer0+layer1 pipeline hands layer1
# precomputed, and the thresholds keep the l01 verdict comparable to l1 cells
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_gather_rs")
)
from test_moe_gather_rs_traffic import (  # noqa: E402
    ABSOLUTE_THRESHOLD_MAP,
    RELATIVE_THRESHOLD_MAP,
    build_a2av_combine_indices,
    build_a2av_compress_indices,
    build_a2av_unique_counts,
)

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def init_ep_group(ep_size: int):
    # verbatim from test_moe_ag_scatter/test_moe_ag_traffic.py: EP groups over
    # the world; with ep_size == world_size this yields one group of all ranks
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


def take_first_or_none(x: Optional[List[Any]]):
    return x[0] if x is not None else None


class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, e2e_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.e2e_time_ms = e2e_time_ms
        self.total_ms = e2e_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: e2e {self.e2e_time_ms:.3f} ms"


@torch.no_grad()
def perf_combined(name: str, iters: int, warmup_iters: int, prep_fn, l0_fn, l1_fn):
    """One timed window per iteration: l0_fn() -> GELU -> l1_fn(intermediate),
    with events e2e_start / l0_end / act_end / e2e_end. Mirrors the per-layer
    benches' perf loops: warmup runs in the same loop and is filtered at
    collection; NVTX tags segment warmup vs timed for nsys captures; the sweeps
    isolated mode (FLUX_SWEEP_ISOLATED_ITERS) drains the device and aligns all
    ranks before EVERY timed window, host wall of the pair -> iso_sync_ms."""
    total_iters = warmup_iters + iters
    e2e_start = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    l0_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    act_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    e2e_end = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    output = None
    for i in range(total_iters):
        prep_fn()
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        e2e_start[i].record()
        with torch.cuda.nvtx.range(nvtx_tag):
            l0_out = l0_fn()
            l0_end[i].record()
            # activation on the [gemm_rows_this_ep, ffn] intermediate — a fresh
            # allocation per window (gelu has no out=); cheap post-warmup via
            # the caching allocator, and part of the pass by definition
            intermediate = torch.nn.functional.gelu(l0_out)
            act_end[i].record()
            output = l1_fn(intermediate)
        e2e_end[i].record()

    iter_times = {"e2e_ms": [], "l0_ms": [], "act_ms": [], "l1_ms": []}
    for i in range(total_iters):
        e2e_end[i].synchronize()
        if i >= warmup_iters:
            iter_times["e2e_ms"].append(e2e_start[i].elapsed_time(e2e_end[i]))
            iter_times["l0_ms"].append(e2e_start[i].elapsed_time(l0_end[i]))
            iter_times["act_ms"].append(l0_end[i].elapsed_time(act_end[i]))
            iter_times["l1_ms"].append(act_end[i].elapsed_time(e2e_end[i]))
    if isolated:
        iter_times["iso_sync_ms"] = iso_sync_times[warmup_iters:]

    result = PerfResult(
        name=name, output=output, e2e_time_ms=sum(iter_times["e2e_ms"]) / iters
    )
    result.iter_times = iter_times
    return result


def perf_fast(*_args, **_kwargs):
    # --impl fast (the FAST+FAST combined arm: comm-free grouped GEMM0 + FAST
    # alltoallv dispatch -> GELU -> grouped GEMM1 + FAST alltoallv combine)
    # lands only after its own bring-up — see
    # test/python/moe_gather_rs/test_moe_gather_rs_fast_baseline.py and
    # test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py for the halves.
    # Open question blocking it: FAST's signal/credit reset currently runs once
    # per window OUTSIDE it; a combined pass issues TWO alltoallv calls per
    # window and it is unverified whether one tail reset is sufficient or each
    # call needs its own in-window reset (which would change the accounting).
    raise NotImplementedError(
        "--impl fast is not implemented yet; run the per-layer FAST baselines"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True, help="traffic matrix file")
    parser.add_argument(
        "--chunk_bytes",
        type=int,
        default=8192,
        help="bytes of one routed token copy in the traffic matrix (H * dtype size;"
        " H doubles as layer1's GEMM output dim N, so the combine chunk is identical)",
    )
    parser.add_argument(
        "-N", type=int, default=4096, help="model hidden dim H (layer0 input, layer1 output)"
    )
    parser.add_argument(
        "-K", type=int, default=4096, help="ffn intermediate dim (layer0 output, layer1 input)"
    )
    parser.add_argument("-G", type=int, default=32, help="number of experts")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--warmup_iters", default=5, type=int, help="warmup iterations")
    parser.add_argument("--sm_margin", default=0, type=int, help="sm margin (both layers)")
    parser.add_argument(
        "--dtype", default="bfloat16", help="data type", choices=["bfloat16", "float16"]
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=4,
        help="layer1 split-N pipeline depth (N/n_split must be a multiple of 1024)",
    )
    parser.add_argument(
        "--l0_comm_pattern",
        default="allgather",
        choices=["allgather", "a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress"],
        help="layer0 dispatch pattern, exactly test_moe_ag_traffic.py --comm_pattern"
        " (a2av_hier_compress multi-node requires --sm_margin >= 1)",
    )
    parser.add_argument(
        "--l1_comm_pattern",
        default="dense",
        choices=["dense", "a2av_hier", "a2av_hier_compress"],
        help="layer1 combine pattern, exactly test_moe_gather_rs_traffic.py"
        " --comm_pattern (env knobs FLUX_A2AV_RS_* — no collision with layer0's"
        " FLUX_A2AV_*)",
    )
    parser.add_argument(
        "--impl",
        default="flux",
        choices=["torch", "flux", "fast"],
        help="timed arm: flux (fused pairing), torch (unfused two-layer reference,"
        " same window discipline), fast (NOT IMPLEMENTED — stub)",
    )
    parser.add_argument(
        "--routing_file",
        type=str,
        default=None,
        help="per-token expert-id sidecar (<matrix>.routing.txt, trace family);"
        " must realize exactly --traffic_matrix, mirroring the per-layer benches",
    )
    parser.add_argument(
        "--skip_correctness",
        default=False,
        action="store_true",
        help="skip the untimed torch two-layer reference and its result check"
        " (and shrink the reference scatter staging buffer); correctness columns"
        " in the sweep capsule stay empty for the cell",
    )
    parser.add_argument(
        "--l1_index_in_window",
        default=False,
        action="store_true",
        help="sensitivity mode: run the layer1 python index builders"
        " (pack/reduce + compress CSRs) INSIDE the timed window (counted in"
        " l1_ms) instead of inheriting them untimed; contains GPU->CPU copies,"
        " so it serializes the host on the l0+GELU tail by construction",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # combined pairing is defined over the sm80/V2 ops only (a2av dispatch +
    # gather-rs a2av combine both live there); Hopper's V3 pair has neither
    assert flux.util.get_arch() < 90, "test_moe_l0l1_traffic.py targets the sm80/V2 ops"
    if args.impl == "fast":
        perf_fast()

    # each expert's full ffn weight resides on one rank: T=1, E=world_size
    init_ep_group(DIST_ENV.WORLD_SIZE)
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = DIST_ENV.LOCAL_WORLD_SIZE

    print("before flux_shm initialization")
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    print("after flux_shm initialization")

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    H, FFN = args.N, args.K
    assert H * input_dtype.itemsize == args.chunk_bytes, (
        f"N/H ({H}) * dtype size ({input_dtype.itemsize}) must equal the traffic matrix"
        f" chunk granularity ({args.chunk_bytes} bytes)"
    )
    assert args.G % WORLD_SIZE == 0, f"{args.G} % {WORLD_SIZE} != 0"
    assert args.N % args.n_split == 0 and (args.N // args.n_split) % 1024 == 0, (
        f"N ({args.N}) / n_split ({args.n_split}) must be a multiple of 1024"
    )
    if args.l0_comm_pattern == "a2av_hier_compress" and NNODES > 1:
        assert args.sm_margin >= 1, "multi-node l0 a2av_hier_compress requires --sm_margin >= 1"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == WORLD_SIZE, (
        f"traffic matrix is for {matrix.shape[0]} ranks but world size is {WORLD_SIZE}"
    )
    if args.routing_file:
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
    ntokens = choosed_experts.shape[0]
    assert ntokens % WORLD_SIZE == 0
    if NNODES > 1:
        # layer1 multi-node constraint (see test_moe_gather_rs_traffic.py)
        assert ntokens % (WORLD_SIZE * args.topk) == 0, (
            f"multi-node requires token count ({ntokens}) divisible by"
            f" world_size * topk ({WORLD_SIZE * args.topk})"
        )

    gating_args = gen_moe_gating_args(args.G, args.topk, ntokens, choosed_experts=choosed_experts)

    # torch reference buffers are needed whenever the torch path runs: as the
    # timed arm (--impl torch) or as the untimed correctness reference
    need_torch_path = args.impl == "torch" or not args.skip_correctness
    moe_ctx = MoeMlp1Ctx(
        TP_GROUP,
        EP_GROUP,
        b=1,
        s=ntokens,
        h=H,
        ffn_size=FFN,
        nexperts=args.G,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        dist="uniform",
        fast_accum=False,
        weight_groups=1,
        drop_token=False,
        gating_args=gating_args,
        skip_reference=not need_torch_path,
    )

    # ---- layer1 identities (T=1, E=W; each expert's rows are contiguous) ----
    T, E = 1, WORLD_SIZE
    n_experts_per_rank = args.G // E
    local_K = FFN // T
    split_cpu = moe_ctx.splits_cpu
    routing_idx = gating_args.scatter_index.flatten()
    token_index = gating_args.gather_index
    topk_index = gating_args.topk_index
    M = ntokens * args.topk
    eid_start = RANK * n_experts_per_rank
    eid_end = eid_start + n_experts_per_rank
    ep_rank_m_start = int(torch.sum(split_cpu[:eid_start]))
    ep_rank_m_end = ep_rank_m_start + int(torch.sum(split_cpu[eid_start:eid_end]))
    assert ep_rank_m_end - ep_rank_m_start == moe_ctx.nrows_ep
    # MoeMlp1Ctx with ffn_tp_size == 1 gives ffn_size_shard == FFN, so layer0's
    # grouped-GEMM output (nrows_ep, FFN) IS layer1's GEMM input after GELU
    assert moe_ctx.ffn_size_shard == local_K

    # ---- untimed host metadata (metadata-exchange contract, both layers) ----
    # cnt[s][e] = copies token-home rank s routed to expert e; layer0 consumes
    # it as-is, the layer1 combine op derives its transpose-aggregate itself
    W = WORLD_SIZE
    tokens_per_rank = ntokens // W
    src_of_copy = (torch.arange(ntokens, dtype=torch.long) // tokens_per_rank).repeat_interleave(
        args.topk
    )
    e_of_copy = choosed_experts.reshape(-1).long().cpu()
    splits_per_source_cpu = (
        torch.bincount(src_of_copy * args.G + e_of_copy, minlength=W * args.G).view(W, args.G).int()
    )
    assert torch.equal(
        splits_per_source_cpu.sum(0), split_cpu[: args.G].cpu().int()
    ), "splits_per_source column sums must equal splits"

    l0_use_a2av = args.l0_comm_pattern in ("a2av", "a2av_ring", "a2av_hier", "a2av_hier_compress")
    l1_use_compress = args.l1_comm_pattern == "a2av_hier_compress"
    l1_use_a2av = args.l1_comm_pattern == "a2av_hier" or l1_use_compress

    # layer0 compress dedup counts [W, W + nn] (layer0 orientation): u[s][d] =
    # unique tokens source s must deliver to rank d, U[s][n] = unique tokens s
    # must deliver to the node-n union (verbatim from test_moe_ag_traffic.py)
    l0_unique_counts_cpu = None
    if args.l0_comm_pattern == "a2av_hier_compress":
        experts_per_rank = args.G // W
        L = LOCAL_WORLD_SIZE
        nn = W // L
        owner = choosed_experts.long().cpu() // experts_per_rank  # [ntokens, topk] dest rank
        flags = torch.zeros(ntokens, W, dtype=torch.bool)
        flags.scatter_(1, owner, True)  # token t needed by rank d (any expert)
        u_mat = flags.view(W, tokens_per_rank, W).sum(1)  # [W, W]
        U_mat = flags.view(ntokens, nn, L).any(dim=2).view(W, tokens_per_rank, nn).sum(1)
        l0_unique_counts_cpu = torch.cat([u_mat, U_mat], dim=1).int().contiguous()

    # layer1 compress dedup counts [W, NN] (transposed-U recipe from the l1 bench)
    l1_unique_counts_cpu = None
    if l1_use_compress and NNODES > 1:
        l1_unique_counts_cpu = build_a2av_unique_counts(
            choosed_experts, WORLD_SIZE, NNODES, n_experts_per_rank
        )

    # ---- layer1 inherited indices (DECIDED accounting: OUTSIDE the window;
    # --l1_index_in_window moves exactly these builder calls inside) ----
    def build_l1_indices():
        kwargs = {}
        pack_index, reduce_index = build_a2av_combine_indices(
            routing_idx, split_cpu, RANK, WORLD_SIZE, args.topk
        )
        kwargs["a2av_pack_index"] = pack_index
        kwargs["a2av_reduce_index"] = reduce_index
        if l1_use_compress and NNODES > 1:
            wire_ptr, wire_copy, red_ptr, red_row = build_a2av_compress_indices(
                routing_idx, split_cpu, l1_unique_counts_cpu, RANK, WORLD_SIZE, NNODES, args.topk
            )
            kwargs["a2av_wire_csr"] = [wire_ptr, wire_copy]
            kwargs["a2av_reduce_csr"] = [red_ptr, red_row]
        return kwargs

    l1_index_build_ms = 0.0
    l1_static_kwargs = {}
    if l1_use_a2av and args.impl == "flux":  # flux-only inputs; torch arm ignores them
        l1_static_kwargs["splits_per_source"] = splits_per_source_cpu
        if l1_use_compress and NNODES > 1:
            l1_static_kwargs["a2av_unique_counts"] = l1_unique_counts_cpu
        if not args.l1_index_in_window:
            t0 = time.perf_counter()
            l1_static_kwargs.update(build_l1_indices())
            torch.cuda.synchronize()  # builders end in .cuda() copies
            l1_index_build_ms = (time.perf_counter() - t0) * 1e3
    elif args.l1_index_in_window and RANK == 0:
        print("NOTE: --l1_index_in_window is a no-op with --l1_comm_pattern dense")

    # ---- layer1 weights/scales (l1 traffic bench data_config; the input slot
    # is layer0's live output, so it is not generated here) ----
    data_config = [
        ((n_experts_per_rank, args.N, local_K), input_dtype, (0.1, 0.0)),  # weight
        ((n_experts_per_rank,), torch.float32, (1, 0)),  # weight_scale
        ((1,), torch.float32, (1, 0)),  # input_scale
        ((moe_ctx.nrows_ep,), torch.float32, (1, 0)),  # output_vec_scale
    ]
    l1_weight, l1_weight_scale, l1_input_scale, l1_output_vec_scale = next(
        generate_data(data_config)
    )

    if RANK == 0:
        rows_per_rank = split_cpu.view(WORLD_SIZE, n_experts_per_rank).sum(dim=1)
        print(f"ntokens: {ntokens} ({tokens_per_rank} per rank), topk: {args.topk}, M: {M}")
        print(f"Splits: {split_cpu.tolist()}, Sum: {sum(split_cpu.tolist())}")
        print(f"Per-rank gemm rows: {rows_per_rank.tolist()}")
        print(
            f"impl: {args.impl}, l0_comm_pattern: {args.l0_comm_pattern},"
            f" l1_comm_pattern: {args.l1_comm_pattern}, n_split: {args.n_split},"
            f" l1_index_in_window: {args.l1_index_in_window}"
        )
        if l0_use_a2av:
            send_bytes = (matrix.sum(dim=1) - matrix.diag()).tolist()
            print(f"l0 a2av wire bytes per rank (send): {send_bytes}")
        if l1_use_a2av:
            # combine wire sender is the expert owner: transpose of the matrix
            send_bytes = (matrix.sum(dim=0) - matrix.diag()).tolist()
            print(f"l1 a2av combine wire bytes per rank (send): {send_bytes}")
        RECORDER.emit_info(
            ntokens=int(ntokens),
            tokens_per_rank=int(tokens_per_rank),
            topk=int(args.topk),
            gemm_rows_per_rank=rows_per_rank.tolist(),
            l0_comm_pattern=args.l0_comm_pattern,
            l1_comm_pattern=args.l1_comm_pattern,
            n_split=int(args.n_split),
            l1_index_in_window=bool(args.l1_index_in_window),
            l1_index_build_ms=round(l1_index_build_ms, 6),
        )

    # ---- build the ops (flux arm; ctor mirrors the per-layer benches) ----
    gemm_only_op = None
    if need_torch_path:
        gemm_only_op = flux.GemmOnly(
            moe_ctx.inputs.dtype,
            moe_ctx.inputs.dtype,
            moe_ctx.outputs[0].dtype,
            use_fp8_gemm=flux.is_fp8_dtype(moe_ctx.inputs.dtype),
        )

    def run_torch_two_layer():
        """One unfused two-layer pass over the shared ctx; returns the final
        RS-sharded output. Used per-window by --impl torch and once untimed as
        the correctness reference for --impl flux."""
        moe_ctx.clear_outputs()
        MoeAgScatterWithTorch.comm_impl(moe_ctx, TP_GROUP)
        MoeAgScatterWithTorch.scatter_impl(moe_ctx)
        MoeAgScatterWithTorch.gemm_impl(moe_ctx, gemm_only_op)
        intermediate = torch.nn.functional.gelu(moe_ctx.outputs[0])
        return moe_gather_rs_forward_torch(
            TP_GROUP,
            M,
            eid_start,
            ep_rank_m_start,
            ep_rank_m_end,
            intermediate,
            l1_weight,
            split_cpu,
            token_index,
            topk_index,
            args.topk,
            l1_input_scale,
            l1_weight_scale,
            l1_output_vec_scale,
            do_all_reduce=False,
        )

    if args.impl == "flux":
        tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
        moe_args = flux.MoeArguments(
            max_ntokens=ntokens,
            hidden=H,
            ffn_hidden=FFN,
            nexperts=args.G,
            topk=args.topk,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        l0_op = flux.GemmGroupedV2AGScatterOp(
            tp_env=tp_env,
            moe_args=moe_args,
            a2av_dispatch=l0_use_a2av,
            a2av_ring=(args.l0_comm_pattern == "a2av_ring"),
            a2av_hier=(args.l0_comm_pattern == "a2av_hier"),
            a2av_hier_compress=(args.l0_comm_pattern == "a2av_hier_compress"),
        )
        l1_op = flux.GemmGroupedV2GatherRSOp(
            TP_GROUP,
            args.G,
            M,
            args.N,
            args.topk,
            output_dtype,
            T,
            E,
            1,
            nnodes=NNODES,
            n_split=args.n_split,
            do_all_reduce=False,
            use_read_mode=False,
            a2av_hier=l1_use_a2av and not l1_use_compress,
            a2av_hier_compress=l1_use_compress,
        )

        l0_extra = {
            "ag_option": flux.AllGatherOption(),
            "bias": take_first_or_none(moe_ctx.bias),
            "input_scale": take_first_or_none(moe_ctx.input_scale),
            "weight_scale": take_first_or_none(moe_ctx.weight_scale),
            "splits_per_source": splits_per_source_cpu,
        }
        if l0_unique_counts_cpu is not None:
            l0_extra["a2av_unique_counts"] = l0_unique_counts_cpu

        def flux_prep():
            moe_ctx.clear_outputs()
            l0_op.clear_buffers()

        def flux_l0():
            # routing/scheduling for the whole pass is computed HERE, in-window
            # (layer0's stage2 metadata work) — never cached across iterations
            l0_op.forward(
                inputs_shard=moe_ctx.inputs_shard,
                weights=moe_ctx.weights[0],
                splits_gpu=moe_ctx.splits_gpu,
                scatter_index=moe_ctx.scatter_index,
                output_scale=take_first_or_none(moe_ctx.output_scale),
                outputs_buf=moe_ctx.outputs[0],
                fast_accum=moe_ctx.fast_accum,
                sm_margin=args.sm_margin,
                allgather_output=None,
                **l0_extra,
            )
            return moe_ctx.outputs[0]

        def flux_l1(intermediate):
            l1_kwargs = dict(l1_static_kwargs)
            if l1_use_a2av and args.l1_index_in_window:
                l1_kwargs.update(build_l1_indices())
            return l1_op.forward_gather_rs(
                intermediate,
                l1_weight,
                split_cpu,
                routing_idx,
                input_scale=l1_input_scale,
                weight_scale=l1_weight_scale,
                output_vec_scale=l1_output_vec_scale,
                fast_accum=False,
                sm_margin=args.sm_margin,
                bias=None,
                **l1_kwargs,
            )

        perf_result = perf_combined(
            f"flux #{RANK}", args.iters, args.warmup_iters, flux_prep, flux_l0, flux_l1
        )
    else:  # torch arm: same window discipline, unfused impls, impl="torch"

        def torch_prep():
            moe_ctx.clear_outputs()

        def torch_l0():
            MoeAgScatterWithTorch.comm_impl(moe_ctx, TP_GROUP)
            MoeAgScatterWithTorch.scatter_impl(moe_ctx)
            MoeAgScatterWithTorch.gemm_impl(moe_ctx, gemm_only_op)
            return moe_ctx.outputs[0]

        def torch_l1(intermediate):
            return moe_gather_rs_forward_torch(
                TP_GROUP,
                M,
                eid_start,
                ep_rank_m_start,
                ep_rank_m_end,
                intermediate,
                l1_weight,
                split_cpu,
                token_index,
                topk_index,
                args.topk,
                l1_input_scale,
                l1_weight_scale,
                l1_output_vec_scale,
                do_all_reduce=False,
            )

        perf_result = perf_combined(
            f"torch #{RANK}", args.iters, args.warmup_iters, torch_prep, torch_l0, torch_l1
        )

    flux.exec_in_rank_order(TP_GROUP, lambda: print(perf_result))
    RECORDER.emit_iters(args.impl, perf_result.iter_times)

    # ---- correctness: flux final RS-sharded output vs ONE untimed torch
    # two-layer reference (thresholds inherited from the l1 bench maps; if the
    # GELU-amplified GEMM0 spread forces loosening at bring-up, do it HERE
    # explicitly and record the observed max abs/rel error as evidence) ----
    if args.impl == "flux" and not args.skip_correctness:
        torch_output = run_torch_two_layer()
        flux_output = perf_result.output
        atol = ABSOLUTE_THRESHOLD_MAP[input_dtype]
        rtol = RELATIVE_THRESHOLD_MAP[input_dtype]

        def check_result():
            print(f"#{RANK} Threshold = Atol:{atol}  Rtol:{rtol}")
            print(f"flux  output shape: {flux_output.size()}")
            print(f"torch output shape: {torch_output.size()}")
            try:
                flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)
            except Exception as e:
                dump_dir = os.environ.get("FLUX_DEBUG_DUMP_DIR", "/tmp")
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(flux_output, os.path.join(dump_dir, f"l01_flux_output_{RANK}.pt"))
                torch.save(torch_output, os.path.join(dump_dir, f"l01_torch_output_{RANK}.pt"))
                print(f"❌ flux and torch not matches, debug tensors dumped to {dump_dir}")
                RECORDER.emit_correctness(bitwise=False, allclose=False)
                RECORDER.flush()
                raise e
            else:
                print("✅ flux and torch matches")

        flux.exec_in_rank_order(TP_GROUP, check_result)
        RECORDER.emit_correctness(bitwise=False, allclose=True)

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
