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
layer-axis campaign, cells.csv layer=l01; RULE-5 CONVERTED 2026-08-21,
timing_accounting=per_iter_gpu):

- Per iteration, INSIDE the timed bracket: the routing allgather
  (plan_comm_ms), then ALL routing-derived metadata for BOTH layers
  (plan_ms) — splits / stable scatter_index / splits_per_source / unique
  counts via the l0 op's fused derive_routed_meta, and the layer1
  pack/reduce indices + compress CSRs via the vectorized _dev builder
  twins seeded from that derive (the l1 unique-counts host kwarg is the
  U-slice of the l0 derive; bit-parity with the CPU builders is guarded at
  setup). Layer0's stage2 scheduling still runs in-window inside forward.
  Nothing routing-derived is cached across iterations (one-shot inference
  semantics). total_ms = plan_comm + plan + e2e; e2e/l0/act/l1 anchors are
  unchanged vs the pre-conversion bench (never compare totals across the
  accounting boundary).
- The CPU builders survive only as untimed setup drift-guard references;
  their wall time is still reported as rank-0 cell_info
  `l1_index_build_ms` (guard cost, not timed work). The old
  `--l1_index_in_window` sensitivity flag is retired.

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
from flux.testing.a2av_combine_indices import (
    build_a2av_combine_indices_dev,
    build_a2av_compress_indices_dev,
    build_a2av_unique_counts_dev,
)
from flux.testing.recorder import RECORDER

# reuse the layer1 traffic bench's inherited-index math and correctness
# thresholds (import via sys.path like the fast baselines do): the CPU
# builders are now the untimed DRIFT-GUARD references (rule 5, 2026-08-21 —
# the timed path re-derives everything per iteration via the _dev twins),
# and the thresholds keep the l01 verdict comparable to l1 cells
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
def perf_combined(name: str, iters: int, warmup_iters: int, prep_fn, plan_comm_fn, plan_fn, l0_fn, l1_fn):
    """One timed window per iteration (SCHEMA rule 5, converted 2026-08-21):
    plan_comm_fn() [routing allgather] -> plan_fn() [ALL routing-derived
    metadata for BOTH layers, on GPU] -> l0_fn(plan) -> GELU ->
    l1_fn(plan, intermediate). Events iter_start / plan_comm_end / plan_end /
    e2e_start / l0_end / act_end / e2e_end; e2e/l0/act/l1 anchors unchanged
    vs the pre-rule-5 bench (check_l01_identity compatibility), plan brackets
    sit inside total_ms but OUTSIDE e2e_ms. One plan bracket for the whole
    pass: routing is fully known after one allgather, which is the fused
    pipeline the combined arm models. Warmup runs in the same loop and is
    filtered at collection; NVTX tags segment warmup vs timed for nsys; the
    sweeps isolated mode (FLUX_SWEEP_ISOLATED_ITERS) drains the device and
    aligns all ranks before EVERY timed window (host wall -> iso_sync_ms)."""
    total_iters = warmup_iters + iters
    ev = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    iter_start, plan_comm_end, plan_end = ev(), ev(), ev()
    e2e_start, l0_end, act_end, e2e_end = ev(), ev(), ev(), ev()
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
        with torch.cuda.nvtx.range(nvtx_tag):
            iter_start[i].record()
            plan_comm_fn()
            plan_comm_end[i].record()
            plan = plan_fn()
            plan_end[i].record()
            e2e_start[i].record()
            l0_out = l0_fn(plan)
            l0_end[i].record()
            # activation on the [gemm_rows_this_ep, ffn] intermediate — a fresh
            # allocation per window (gelu has no out=); cheap post-warmup via
            # the caching allocator, and part of the pass by definition
            intermediate = torch.nn.functional.gelu(l0_out)
            act_end[i].record()
            output = l1_fn(plan, intermediate)
            e2e_end[i].record()
        if TP_GROUP.rank() == 0:
            # per-window heartbeat for the sweep runner's idle-kill: the torch
            # arm is otherwise silent for its whole loop (minutes/window at
            # K3 b32 — every-5 was not enough, 2026-08-21)
            print(f"[hb] window {i + 1}/{total_iters}")

    iter_times = {
        "e2e_ms": [],
        "l0_ms": [],
        "act_ms": [],
        "l1_ms": [],
        "plan_comm_ms": [],
        "plan_ms": [],
        "total_ms": [],
    }
    for i in range(total_iters):
        e2e_end[i].synchronize()
        if i >= warmup_iters:
            iter_times["plan_comm_ms"].append(iter_start[i].elapsed_time(plan_comm_end[i]))
            iter_times["plan_ms"].append(plan_comm_end[i].elapsed_time(plan_end[i]))
            iter_times["e2e_ms"].append(e2e_start[i].elapsed_time(e2e_end[i]))
            iter_times["l0_ms"].append(e2e_start[i].elapsed_time(l0_end[i]))
            iter_times["act_ms"].append(l0_end[i].elapsed_time(act_end[i]))
            iter_times["l1_ms"].append(act_end[i].elapsed_time(e2e_end[i]))
            iter_times["total_ms"].append(iter_start[i].elapsed_time(e2e_end[i]))
    if isolated:
        iter_times["iso_sync_ms"] = iso_sync_times[warmup_iters:]

    result = PerfResult(
        name=name, output=output, e2e_time_ms=sum(iter_times["e2e_ms"]) / iters
    )
    result.iter_times = iter_times
    return result


# ---- FAST combined arm helpers (2026-08-21; mirrors the layer0 fast test) --
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
    # NOTE (2026-08-21, rule-5 conversion): --l1_index_in_window is RETIRED —
    # per-iteration in-window GPU derivation (plan bracket) is the only mode;
    # the old flag's python-builders-inside-l1_ms accounting would be a third
    # never-mix regime.
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="wrap the timed loop in flux.group_profile (sweep torchprof mode)",
    )
    parser.add_argument(
        "--capacity_mib",
        type=int,
        default=0,
        help="FAST buffer capacity per buffer in MiB (--impl fast); 0 = auto"
        " (4 x max(row sum, col sum) of the matrix; transpose-invariant, so"
        " both wire directions get the same capacity)",
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
        help="directory containing libflash.so + flash_utils.py (--impl fast)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # combined pairing is defined over the sm80/V2 ops only (a2av dispatch +
    # gather-rs a2av combine both live there); Hopper's V3 pair has neither
    assert flux.util.get_arch() < 90, "test_moe_l0l1_traffic.py targets the sm80/V2 ops"

    # each expert's full ffn weight resides on one rank: T=1, E=world_size
    init_ep_group(DIST_ENV.WORLD_SIZE)
    RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
    LOCAL_WORLD_SIZE = DIST_ENV.LOCAL_WORLD_SIZE

    if args.impl != "fast":
        # --impl fast skips flux shm: FAST owns the only NVSHMEM init in the
        # process (same contract as the layer0 fast test)
        print("before flux_shm initialization")
        flux.init_flux_shm(TP_GROUP)
        torch.cuda.synchronize()
        print("after flux_shm initialization")
    else:
        assert LOCAL_WORLD_SIZE in (4, 8), (
            f"FAST expects 4 (Perlmutter) or 8 (p4d) GPUs/node; got {LOCAL_WORLD_SIZE}"
        )
        assert WORLD_SIZE > LOCAL_WORLD_SIZE, "FAST requires at least 2 nodes (server_n > 1)"

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    H, FFN = args.N, args.K
    assert H * input_dtype.itemsize == args.chunk_bytes, (
        f"N/H ({H}) * dtype size ({input_dtype.itemsize}) must equal the traffic matrix"
        f" chunk granularity ({args.chunk_bytes} bytes)"
    )
    assert args.G % WORLD_SIZE == 0, f"{args.G} % {WORLD_SIZE} != 0"
    # combine tile is 1024 or 512 (2026-08-21, K3 H=3584): 512-alignment is
    # the dense-path requirement; a2av-only arms would tolerate %8 but every
    # planned cell satisfies %512 (K3: n_split=7 -> n_per=512)
    assert args.N % args.n_split == 0 and (args.N // args.n_split) % 512 == 0, (
        f"N ({args.N}) / n_split ({args.n_split}) must be a multiple of 512"
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
        local_scatter=True,
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

    # ---- layer1 index REFERENCE builds (untimed, rule-5 drift guards ONLY
    # since the 2026-08-21 conversion: the timed path re-derives everything
    # per iteration on GPU inside the plan bracket) ----
    def build_l1_indices_reference():
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

    l1_index_build_ms = 0.0  # wall of the reference build (drift-guard cost, untimed)
    l1_reference_kwargs = {}
    if l1_use_a2av and args.impl == "flux":
        t0 = time.perf_counter()
        l1_reference_kwargs = build_l1_indices_reference()
        torch.cuda.synchronize()  # builders end in .cuda() copies
        l1_index_build_ms = (time.perf_counter() - t0) * 1e3

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
            f" l1_comm_pattern: {args.l1_comm_pattern}, n_split: {args.n_split}"
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
            # SCHEMA protocol rule 5 (l01 driver converted 2026-08-21): the
            # routing allgather + ALL plan derivation for BOTH layers timed
            # per iteration; l1_index_build_ms is now the untimed drift-guard
            # reference build wall, kept for column continuity.
            timing_accounting="per_iter_gpu",
            torch_ref_impl="local_slice_scatter",
            l1_index_build_ms=round(l1_index_build_ms, 6),
        )

    # ---- build the ops (ctor mirrors the per-layer benches) ----
    gemm_only_op = None
    if need_torch_path:
        gemm_only_op = flux.GemmOnly(
            moe_ctx.inputs.dtype,
            moe_ctx.inputs.dtype,
            moe_ctx.outputs[0].dtype,
            use_fp8_gemm=flux.is_fp8_dtype(moe_ctx.inputs.dtype),
        )

    # ---- rule-5 apparatus (SCHEMA protocol rule 5, l01 converted 2026-08-21)
    # Buffers/ops are setup scope (allocation only); the routing exchange and
    # EVERY routing-derived quantity for BOTH layers re-derive per iteration
    # inside the plan bracket. Untimed work below is the bitwise drift guard.
    topk_shard = choosed_experts[
        RANK * tokens_per_rank : (RANK + 1) * tokens_per_rank
    ].contiguous()
    topk_gather_buf = torch.zeros(ntokens, args.topk, dtype=torch.int32, device="cuda")

    def plan_comm_fn():
        torch.distributed.all_gather_into_tensor(topk_gather_buf, topk_shard, group=TP_GROUP)

    plan_comm_fn()
    assert torch.equal(topk_gather_buf, choosed_experts), (
        "allgathered routing != replicated harness routing"
    )

    # The l0 op doubles as the rule-5 meta engine (derive_routed_meta): the
    # flux arm builds it with its comm pattern; the torch arm builds the plain
    # allgather-mode op purely for the per-iteration derivation (minimal
    # heap). The fast arm builds NO flux op (FAST owns NVSHMEM) — it derives
    # via derive_fast_l01_meta_gpu, guarded in its own branch below.
    is_flux = args.impl == "flux"
    l0_op = None
    if args.impl != "fast":
        tp_env = flux.DistEnvTPWithEP(
            tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP
        )
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
            a2av_dispatch=l0_use_a2av and is_flux,
            a2av_ring=is_flux and (args.l0_comm_pattern == "a2av_ring"),
            a2av_hier=is_flux and (args.l0_comm_pattern == "a2av_hier"),
            a2av_hier_compress=is_flux and (args.l0_comm_pattern == "a2av_hier_compress"),
        )
    if args.impl == "fast":
        g_sd = g_scd = g_sps = g_uc = None
    else:
        g_sd, g_scd, g_sps, g_uc = l0_op.derive_routed_meta(topk_gather_buf)
    if args.impl != "fast":
        assert torch.equal(g_sd.cpu(), split_cpu[: args.G].cpu().int()), "derive splits drift"
        assert torch.equal(g_scd, moe_ctx.scatter_index.int()), "derive scatter_index drift"
        assert torch.equal(g_sps, splits_per_source_cpu), "derive splits_per_source drift"
        if l0_unique_counts_cpu is not None:
            assert torch.equal(g_uc, l0_unique_counts_cpu), "derive l0 unique_counts drift"
    if l1_use_a2av and is_flux:
        d_pack, d_red = build_a2av_combine_indices_dev(
            g_scd.view(-1), g_sd, RANK, WORLD_SIZE, args.topk
        )
        assert torch.equal(d_pack, l1_reference_kwargs["a2av_pack_index"].int()), (
            "in-window pack_index drift vs CPU reference builder"
        )
        assert torch.equal(d_red, l1_reference_kwargs["a2av_reduce_index"].int()), (
            "in-window reduce_index drift vs CPU reference builder"
        )
        if l1_use_compress and NNODES > 1:
            d_uc = build_a2av_unique_counts_dev(
                topk_gather_buf, WORLD_SIZE, NNODES, n_experts_per_rank
            )
            assert torch.equal(d_uc.cpu(), l1_unique_counts_cpu), "l1 unique_counts drift"
            assert torch.equal(
                g_uc[:, WORLD_SIZE:].contiguous(), l1_unique_counts_cpu
            ), "l0-derive U slice != l1 unique_counts (transposed-U identity)"
            d_wp, d_wc, d_rp, d_rr = build_a2av_compress_indices_dev(
                g_scd.view(-1), g_sd, d_uc, RANK, WORLD_SIZE, NNODES, args.topk
            )
            r_wp, r_wc = l1_reference_kwargs["a2av_wire_csr"]
            r_rp, r_rr = l1_reference_kwargs["a2av_reduce_csr"]
            for got, ref, nm in (
                (d_wp, r_wp, "wire_ptr"),
                (d_wc, r_wc, "wire_copy"),
                (d_rp, r_rp, "red_ptr"),
                (d_rr, r_rr, "red_row"),
            ):
                assert torch.equal(got, ref.int()), f"in-window compress {nm} drift"
    if need_torch_path and args.impl != "fast":
        # torch-arm in-window reconstructions, guarded once here (the fast
        # arm's untimed reference uses the setup gating values directly)
        _iota_m = torch.arange(M, dtype=torch.int32, device="cuda")
        _copy_buf = torch.empty(M, dtype=torch.int32, device="cuda")
        _copy_buf.scatter_(0, g_scd.view(-1).long(), _iota_m)
        assert torch.equal(_copy_buf // args.topk, token_index.int()), "token_index drift"
        assert torch.equal(_copy_buf % args.topk, topk_index.int()), "topk_index drift"

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

        # drift guard (untimed, once): the C++ derive_combine_meta must
        # reproduce the _dev twins bitwise (which the setup guard already
        # pinned against the CPU builders — closing the equality chain)
        if l1_use_a2av:
            _uc_l1 = (
                l1_unique_counts_cpu if (l1_use_compress and NNODES > 1) else None
            )
            _g_meta = l1_op.derive_combine_meta(
                g_sd, g_scd.view(-1), g_sps, a2av_unique_counts=_uc_l1
            )
            assert torch.equal(_g_meta[0], d_pack), "C++ pack_index drift vs _dev"
            assert torch.equal(_g_meta[1], d_red), "C++ reduce_index drift vs _dev"
            if _uc_l1 is not None:
                for got, ref, nm in (
                    (_g_meta[2], d_wp, "wire_ptr"),
                    (_g_meta[3], d_wc, "wire_copy"),
                    (_g_meta[4], d_rp, "red_ptr"),
                    (_g_meta[5], d_rr, "red_row"),
                ):
                    assert torch.equal(got, ref), f"C++ compress {nm} drift vs _dev"

        # allocation-scope extras; the routing-derived entries
        # (splits_per_source / unique counts) join per-iteration in flux_plan
        l0_extra_base = {
            "ag_option": flux.AllGatherOption(),
            "bias": take_first_or_none(moe_ctx.bias),
            "input_scale": take_first_or_none(moe_ctx.input_scale),
            "weight_scale": take_first_or_none(moe_ctx.weight_scale),
        }

        def flux_prep():
            moe_ctx.clear_outputs()
            l0_op.clear_buffers()

        def flux_plan():
            # rule 5: ALL routing-derived metadata for BOTH layers, per
            # iteration, on GPU — seeded from ONE fused derive (the l1
            # unique-counts host kwarg is the U-slice of the l0 derive; the
            # derive's internal pinned-D2H event sync is the honest host
            # sync; derive outputs alias op buffers, consumed before the
            # next iteration derives again)
            sd, scd, sps_c, uc_c = l0_op.derive_routed_meta(topk_gather_buf)
            l0x = dict(l0_extra_base)
            l0x["splits_per_source"] = sps_c
            if args.l0_comm_pattern == "a2av_hier_compress":
                l0x["a2av_unique_counts"] = uc_c
            l1k = {}
            if l1_use_a2av:
                # C++ single-call derivation (2026-08-21,
                # FLUX_A2AV_RS_DERIVE_COMBINE_TAG): the op's internal
                # arithmetic-identity builders — host offset tables come free
                # from the pinned cnt, no mid-chain D2H syncs, one launch
                # chain (replaced the python _dev twins whose ~40 launches +
                # 5 scalar syncs cost 3.9-6.7 ms/iter at K3)
                uc_l1 = (
                    uc_c[:, WORLD_SIZE:].contiguous()
                    if (l1_use_compress and NNODES > 1)
                    else None
                )
                meta = l1_op.derive_combine_meta(
                    sd, scd.view(-1), sps_c, a2av_unique_counts=uc_l1
                )
                l1k = {
                    "splits_per_source": sps_c,
                    "a2av_pack_index": meta[0],
                    "a2av_reduce_index": meta[1],
                }
                if uc_l1 is not None:
                    l1k["a2av_wire_csr"] = [meta[2], meta[3]]
                    l1k["a2av_reduce_csr"] = [meta[4], meta[5]]
                    l1k["a2av_unique_counts"] = uc_l1
            return {"sd": sd, "scd": scd, "l0_extra": l0x, "l1_kwargs": l1k}

        def flux_l0(plan):
            # layer0 stage2 scheduling still runs in-window inside forward
            l0_op.forward(
                inputs_shard=moe_ctx.inputs_shard,
                weights=moe_ctx.weights[0],
                splits_gpu=plan["sd"],
                scatter_index=plan["scd"],
                output_scale=take_first_or_none(moe_ctx.output_scale),
                outputs_buf=moe_ctx.outputs[0],
                fast_accum=moe_ctx.fast_accum,
                sm_margin=args.sm_margin,
                allgather_output=None,
                **plan["l0_extra"],
            )
            return moe_ctx.outputs[0]

        def flux_l1(plan, intermediate):
            return l1_op.forward_gather_rs(
                intermediate,
                l1_weight,
                plan["sd"],  # CUDA splits accepted directly by the op
                plan["scd"].view(-1),
                input_scale=l1_input_scale,
                weight_scale=l1_weight_scale,
                output_vec_scale=l1_output_vec_scale,
                fast_accum=False,
                sm_margin=args.sm_margin,
                bias=None,
                **plan["l1_kwargs"],
            )

        with flux.group_profile(
            name="moe_ag_scatter_traffic_" + os.environ.get("TORCHELASTIC_RUN_ID", "l01"),
            do_prof=args.profile,
            group=TP_GROUP,
        ):
            perf_result = perf_combined(
                f"flux #{RANK}",
                args.iters,
                args.warmup_iters,
                flux_prep,
                plan_comm_fn,
                flux_plan,
                flux_l0,
                flux_l1,
            )
    elif args.impl == "torch":  # torch arm: same window discipline, unfused impls
        # preallocated derivation scratch (contents re-derived per iteration)
        iota_m = torch.arange(M, dtype=torch.int32, device="cuda")
        copy_buf = torch.empty(M, dtype=torch.int32, device="cuda")

        def torch_prep():
            moe_ctx.clear_outputs()

        def torch_plan():
            # rule 5: derive splits/indices on GPU + the honest D2H the
            # host-side per-expert gemm loop and layer1 segment sums require
            sd, scd, _, _ = l0_op.derive_routed_meta(topk_gather_buf)
            copy_buf.scatter_(0, scd.view(-1).long(), iota_m)
            tok_idx = copy_buf // args.topk
            tpk_idx = copy_buf % args.topk
            splits_cpu_iter = sd.cpu()
            moe_ctx.gather_index = tok_idx
            moe_ctx.splits_cpu = splits_cpu_iter
            return {
                "split_cpu": splits_cpu_iter,
                "token_index": tok_idx,
                "topk_index": tpk_idx,
                "ep_m_start": int(splits_cpu_iter[:eid_start].sum()),
                "ep_m_end": int(splits_cpu_iter[:eid_end].sum()),
            }

        def torch_l0(plan):
            MoeAgScatterWithTorch.comm_impl(moe_ctx, TP_GROUP)
            MoeAgScatterWithTorch.scatter_impl(moe_ctx)
            MoeAgScatterWithTorch.gemm_impl(moe_ctx, gemm_only_op)
            return moe_ctx.outputs[0]

        def torch_l1(plan, intermediate):
            return moe_gather_rs_forward_torch(
                TP_GROUP,
                M,
                eid_start,
                plan["ep_m_start"],
                plan["ep_m_end"],
                intermediate,
                l1_weight,
                plan["split_cpu"],
                plan["token_index"],
                plan["topk_index"],
                args.topk,
                l1_input_scale,
                l1_weight_scale,
                l1_output_vec_scale,
                do_all_reduce=False,
            )

        with flux.group_profile(
            name="moe_ag_scatter_traffic_" + os.environ.get("TORCHELASTIC_RUN_ID", "l01"),
            do_prof=args.profile,
            group=TP_GROUP,
        ):
            perf_result = perf_combined(
                f"torch #{RANK}",
                args.iters,
                args.warmup_iters,
                torch_prep,
                plan_comm_fn,
                torch_plan,
                torch_l0,
                torch_l1,
            )
    else:  # --impl fast (2026-08-21): FAST+FAST combined — the authoritative
        # unfused paper baseline on REAL routing: trace-driven dispatch
        # alltoallv -> grouped GEMM0 -> GELU -> grouped GEMM1 -> combine
        # alltoallv (transposed matrix) -> home-side topk-reduce. TWO
        # flash_comm_t instances (one per wire direction; vendored refcount
        # patch scripts/fast_two_instance.patch) so both credit resets stay
        # OUTSIDE the window — the FAST contract requires a reset between
        # calls on one instance, and a timed mid-window reset would insert a
        # full-world barrier between the layers. e2e mode only (host-blocking
        # alltoallv, like the layer0 fast arm).
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_ag_scatter")
        )
        from fast_baseline_utils import (  # noqa: E402
            build_pack_index,
            build_unpack_index,
            derive_fast_l01_meta_gpu,
        )

        def derive_fn(buf):
            return derive_fast_l01_meta_gpu(
                buf, RANK, args.G, WORLD_SIZE, tokens_per_rank, args.chunk_bytes
            )

        # drift guard (untimed, once): the in-window derivation must reproduce
        # the host reference index math + the combine-direction extensions
        gm = derive_fn(topk_gather_buf)
        ce_local = choosed_experts[RANK * tokens_per_rank : (RANK + 1) * tokens_per_rank]
        pack_ref = build_pack_index(ce_local, args.topk).cuda()
        unpack_ref, split_ref = build_unpack_index(
            splits_per_source_cpu, RANK, args.G, WORLD_SIZE
        )
        assert torch.equal(gm["pack_index"], pack_ref), "derived pack_index drift"
        assert torch.equal(gm["unpack_index"], unpack_ref.cuda()), "derived unpack_index drift"
        assert torch.equal(gm["split_cpu"], split_ref), "derived gemm splits drift"
        assert torch.equal(gm["matrix_cpu"], matrix.long()), "derived BvN matrix drift"
        assert torch.equal(gm["matrix_T_cpu"], matrix.long().t().contiguous()), (
            "derived combine matrix drift"
        )
        assert torch.equal(gm["pack_order"] // args.topk, gm["pack_index"]), (
            "pack_order/pack_index identity drift"
        )
        assert torch.equal(
            gm["inv_unpack"][gm["unpack_index"]],
            torch.arange(gm["unpack_index"].numel(), device="cuda"),
        ), "inv_unpack is not the inverse of unpack_index"

        # the reference gemm loop multiplies by output_scale, which the
        # standalone GemmGroupedV2 does not (layer0 fast test contract) —
        # pin to 1 so both sides compute the pure GEMM
        moe_ctx.output_scale[0].fill_(1.0)

        # ---- FAST bring-up (the only NVSHMEM init in this process; two
        # instances, one per wire direction) ----
        flash_utils = load_fast(os.path.abspath(args.fast_dir))
        uid = broadcast_uid(flash_utils)
        dispatch_comm = flash_utils.flash_comm_t(RANK, LOCAL_WORLD_SIZE, WORLD_SIZE, uid)
        combine_comm = flash_utils.flash_comm_t(RANK, LOCAL_WORLD_SIZE, WORLD_SIZE, uid)
        if args.capacity_mib > 0:
            capacity_bytes = args.capacity_mib << 20
        else:
            capacity_bytes = 4 * int(max(matrix.sum(dim=1).max(), matrix.sum(dim=0).max()))
        capacity_bytes = (capacity_bytes + 15) // 16 * 16
        dispatch_comm.alltoallv_setup(capacity_bytes)
        combine_comm.alltoallv_setup(capacity_bytes)
        if RANK == 0:
            print(f"FAST combined: capacity {capacity_bytes >> 20} MiB per buffer x2 comms")

        gemm0_op = flux.GemmGroupedV2(
            moe_ctx.weights[0], n_experts_per_rank, input_dtype, output_dtype
        )
        gemm1_op = flux.GemmGroupedV2(l1_weight, n_experts_per_rank, input_dtype, output_dtype)
        # the reference (and the flux op) scale gemm1 by weight_scale[e] *
        # input_scale; static per-expert constants (setup scope) expanded to
        # A-order rows per iteration from the in-window splits
        l1_scale_const = (l1_weight_scale * l1_input_scale).float()  # [epr]

        @torch.no_grad()
        def perf_fast_combined(iters, warmup_iters):
            total_iters = warmup_iters + iters
            ev = lambda: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
            iter_start, plan_comm_end, plan_end = ev(), ev(), ev()
            e2e_start, pack_end, disp_end, l0_end, act_end = ev(), ev(), ev(), ev(), ev()
            gemm2_end, cpack_end, comb_end, e2e_end = ev(), ev(), ev(), ev()
            reset_ms = [0.0] * total_iters
            host_e2e = [0.0] * total_iters
            d_sched = [0.0] * total_iters
            d_fill = [0.0] * total_iters
            d_wire = [0.0] * total_iters
            c_sched = [0.0] * total_iters
            c_fill = [0.0] * total_iters
            c_wire = [0.0] * total_iters
            out = out1 = gemm_in0 = None
            full = torch.empty(M // WORLD_SIZE, H, dtype=input_dtype, device="cuda")
            torch.distributed.barrier()
            torch.cuda.synchronize()
            for i in range(total_iters):
                # inter-iteration hygiene, OUTSIDE the window (both directions)
                t_r0 = time.perf_counter()
                dispatch_comm.alltoallv_reset()
                combine_comm.alltoallv_reset()
                reset_ms[i] = (time.perf_counter() - t_r0) * 1e3
                torch.cuda.synchronize()

                nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
                with torch.cuda.nvtx.range(nvtx_tag):
                    # rule 5: routing exchange + ALL metadata (both wire
                    # directions + gemm splits + both BvN matrices) in-window
                    iter_start[i].record()
                    plan_comm_fn()
                    plan_comm_end[i].record()
                    m = derive_fn(topk_gather_buf)
                    plan_end[i].record()

                    t0 = time.perf_counter()
                    e2e_start[i].record()
                    send0 = torch.index_select(
                        moe_ctx.inputs_shard, dim=0, index=m["pack_index"]
                    )
                    pack_end[i].record()
                    recv0, _, t_d = dispatch_comm.alltoallv(
                        send0.view(torch.uint8), m["matrix_cpu"]
                    )
                    disp_end[i].record()
                    gemm_in0 = torch.index_select(
                        recv0.view(input_dtype).view(-1, H), dim=0, index=m["unpack_index"]
                    )
                    out0 = gemm0_op.forward(gemm_in0, m["split_cpu"], sm_margin=args.sm_margin)
                    l0_end[i].record()
                    intermediate = torch.nn.functional.gelu(out0)
                    act_end[i].record()
                    out1 = gemm1_op.forward(
                        intermediate, m["split_cpu"], sm_margin=args.sm_margin
                    )
                    row_scale = torch.repeat_interleave(
                        l1_scale_const, m["split_cpu"].to("cuda", non_blocking=True).long()
                    )
                    out1.mul_(row_scale.unsqueeze(1))
                    out1.mul_(l1_output_vec_scale.unsqueeze(1))
                    gemm2_end[i].record()
                    send1 = torch.index_select(out1, dim=0, index=m["inv_unpack"])
                    cpack_end[i].record()
                    recv1, _, t_c = combine_comm.alltoallv(
                        send1.view(torch.uint8), m["matrix_T_cpu"]
                    )
                    comb_end[i].record()
                    # home side: recv arrives in this rank's own pack order
                    full.index_copy_(
                        0, m["pack_order"], recv1.view(input_dtype).view(-1, H)
                    )
                    out = (
                        full.view(tokens_per_rank, args.topk, H).float().sum(1).to(input_dtype)
                    )
                    e2e_end[i].record()
                e2e_end[i].synchronize()
                host_e2e[i] = (time.perf_counter() - t0) * 1e3
                d_sched[i], d_fill[i], d_wire[i] = t_d.tolist()
                c_sched[i], c_fill[i], c_wire[i] = t_c.tolist()
                if RANK == 0:
                    print(f"[hb] window {i + 1}/{total_iters}")

            def per_iter_ms(starts, ends):
                return [
                    starts[i].elapsed_time(ends[i]) for i in range(warmup_iters, total_iters)
                ]

            iter_times = {
                "plan_comm_ms": per_iter_ms(iter_start, plan_comm_end),
                "plan_ms": per_iter_ms(plan_comm_end, plan_end),
                "total_ms": per_iter_ms(iter_start, e2e_end),
                "e2e_ms": per_iter_ms(e2e_start, e2e_end),
                "l0_ms": per_iter_ms(e2e_start, l0_end),
                "act_ms": per_iter_ms(l0_end, act_end),
                "l1_ms": per_iter_ms(act_end, e2e_end),
                "pack_ms": per_iter_ms(e2e_start, pack_end),
                "gemm_ms": per_iter_ms(disp_end, l0_end),
                "gemm2_ms": per_iter_ms(act_end, gemm2_end),
                "cpack_ms": per_iter_ms(gemm2_end, cpack_end),
                "comb_ms": per_iter_ms(cpack_end, comb_end),
                "acc_ms": per_iter_ms(comb_end, e2e_end),
                "schedule_ms": [v / 1e3 for v in d_sched[warmup_iters:]],
                "fill_ms": [v / 1e3 for v in d_fill[warmup_iters:]],
                "wire_ms": [v / 1e3 for v in d_wire[warmup_iters:]],
                "comb_schedule_ms": [v / 1e3 for v in c_sched[warmup_iters:]],
                "comb_fill_ms": [v / 1e3 for v in c_fill[warmup_iters:]],
                "comb_wire_ms": [v / 1e3 for v in c_wire[warmup_iters:]],
                "reset_ms": reset_ms[warmup_iters:],
                "host_e2e_ms": host_e2e[warmup_iters:],
            }
            result = PerfResult(
                name=f"fast #{RANK}",
                output=out,
                e2e_time_ms=sum(iter_times["e2e_ms"]) / iters,
            )
            result.iter_times = iter_times
            return result, out1, gemm_in0

        perf_result, fast_out1, fast_gemm_in0 = perf_fast_combined(
            args.iters, args.warmup_iters
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

    if args.impl == "fast" and not args.skip_correctness:
        torch_output = run_torch_two_layer()
        # (a) bitwise: FAST dispatch wire + unpack must reproduce the reference
        # scatter block (local_scatter: scatter_inputs IS the local EP block)
        assert flux.testing.bitwise_eq(
            fast_gemm_in0, moe_ctx.scatter_inputs[: moe_ctx.nrows_ep]
        ), "❌ FAST dispatch+unpack does not match the reference scatter block"
        print(f"✅ #{RANK} FAST dispatch+unpack bitwise-match the reference scatter block")
        # (b) bitwise same-op chain on the reference block — isolates the
        # combine wire as the only movement the final allclose still covers
        ref0 = gemm0_op.forward(
            moe_ctx.scatter_inputs[: moe_ctx.nrows_ep],
            gm["split_cpu"],
            sm_margin=args.sm_margin,
        )
        ref1 = gemm1_op.forward(
            torch.nn.functional.gelu(ref0), gm["split_cpu"], sm_margin=args.sm_margin
        )
        ref_row_scale = torch.repeat_interleave(
            l1_scale_const, gm["split_cpu"].to("cuda").long()
        )
        ref1.mul_(ref_row_scale.unsqueeze(1))
        ref1.mul_(l1_output_vec_scale.unsqueeze(1))
        assert flux.testing.bitwise_eq(fast_out1, ref1), (
            "❌ same-op gemm0->gelu->gemm1 chain mismatch: data movement is broken"
        )
        print(f"✅ #{RANK} same-op gemm chain bitwise-matches on the reference block")
        # (c) final RS-sharded output vs the unfused two-layer reference
        atol = ABSOLUTE_THRESHOLD_MAP[input_dtype]
        rtol = RELATIVE_THRESHOLD_MAP[input_dtype]

        def check_fast():
            print(f"#{RANK} Threshold = Atol:{atol}  Rtol:{rtol}")
            try:
                flux.torch_allclose(perf_result.output, torch_output, atol=atol, rtol=rtol)
            except Exception as e:
                dump_dir = os.environ.get("FLUX_DEBUG_DUMP_DIR", "/tmp")
                os.makedirs(dump_dir, exist_ok=True)
                torch.save(perf_result.output, os.path.join(dump_dir, f"l01_fast_output_{RANK}.pt"))
                torch.save(torch_output, os.path.join(dump_dir, f"l01_torch_output_{RANK}.pt"))
                print(f"❌ fast and torch not matches, debug tensors dumped to {dump_dir}")
                RECORDER.emit_correctness(bitwise=True, allclose=False)
                RECORDER.flush()
                raise e
            else:
                print("✅ fast and torch matches")

        flux.exec_in_rank_order(TP_GROUP, check_fast)
        RECORDER.emit_correctness(bitwise=True, allclose=True)

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
