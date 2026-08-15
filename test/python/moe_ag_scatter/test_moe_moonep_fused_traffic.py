################################################################################
#
# Copyright 2026 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
"""MoonEP plan driving the FUSED GemmGroupedV2AGScatterOp (a2av_hier_compress /
lb_union) via the virtual expert space — the merged moonep x flux arm.

The replicated MoonEP plan (identical alloc / experts_to_copy to the staged
test_moe_moonep_traffic.py arms) is re-encoded as routing over R*(epn+B)
virtual experts (flux.testing.moonep_fused_map), so the unmodified fused op
executes the plan's placement with flux's compress wire and tile-level
comm/GEMM overlap. Weights ride the one-sided getmem pull
(WeightPrefetchGetmem, contiguous [epn+B, ffn_shard, H] layout = the op's
single weight group), event-joined BEFORE forward — scenario-1 invariant:
the fused GEMM reads weights ungated.

Declared deviations vs the staged arms (the PLAN is bit-identical; these are
execution-layer differences): no token_padding / zero-fill / NvS layout (the
grouped GEMM takes arbitrary segment sizes; padded rows are not computed),
and per-entry route weights are not moved on the wire (deterministically
replicated; layer0's GEMM does not consume them).

Untimed-metadata contract (walkthrough ep_semantics §6.3): plan + virtual
metadata are untimed setup, like the flux arms' splits/scatter_index; the
staged arms time plan_comm (the topk allgather) — do not compare that phase
across drivers. Reported as moonep_plan_host_ms.
"""

import argparse
import os
import time
from functools import partial

import torch
import torch.distributed

import flux
import flux.testing
from flux.testing import (
    DTYPE_MAP,
    MoeMlp1Ctx,
    choosed_experts_to_matrix_chunks,
    gen_moe_gating_args,
    load_routing_file,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.moonep_fused_map import (
    assign_gateways,
    build_fused_metadata,
    build_virtual_map,
    fused_row_map,
    preflight_metadata_checks,
    push_plan_stats,
    required_a2av_knobs,
)
from flux.testing.moonep_semantics import MoonEPConfig, compute_moonep_plan
from flux.testing.recorder import RECORDER

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def init_ep_group(ep_size: int):
    assert DIST_ENV.WORLD_SIZE % ep_size == 0
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP already initialized"
    assert TP_GROUP.size() % ep_size == 0
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True)
    parser.add_argument("--routing_file", type=str, default="",
                        help="real per-token routing sidecar; validated"
                        " against the matrix")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--G", type=int, default=32,
                        help="ORIGINAL global expert count (virtual space is"
                        " derived)")
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--chunk_bytes", type=int, default=8192)
    parser.add_argument("--ffn_hidden_size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--sm_margin", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--token_padding", type=int, default=128,
                        help="plan constant (kept identical to the staged"
                        " arms so plan hashes match; the fused execution has"
                        " no padded rows)")
    parser.add_argument("--num_comm_sm", type=int, default=8,
                        help="SMs for the getmem prefetch kernel")
    parser.add_argument("--prefetch_chunk_bytes", type=int, default=4 << 20)
    parser.add_argument("--prefetch_impl", default="kernel",
                        choices=["kernel", "stream"])
    parser.add_argument("--weight_path", default="getmem",
                        choices=["getmem", "push"],
                        help="weight-movement transport. getmem (scenario"
                        " 1): one-sided pull, joined before forward. push"
                        " (scenario 2): one-sided CE putmem_signal from the"
                        " home ranks (WeightPushMulticast), the concurrent-"
                        "flow arm")
    parser.add_argument("--weight_push_mode", default="direct",
                        choices=["direct", "mcast", "auto"],
                        help="push wire shape: direct = one put per"
                        " (home, dest) pair; mcast = one inter-node put per"
                        " (expert, dest node) + NVLink gateway fan-out (M4);"
                        " auto (F-C) = mcast iff the plan census finds a"
                        " real fan-out group, else direct (NR-13 fact 1:"
                        " MoonEP-planner plans are usually fan-out-free)")
    parser.add_argument("--weight_gate", default="join",
                        choices=["join", "tiles"],
                        help="how the GEMM observes weight landing. join:"
                        " zero-SM stream waits on my slots before forward"
                        " (serialized wire, the ungated A/B baseline)."
                        " tiles: per-tile signal spin on prefetch-slot"
                        " problems only — dispatch, weights, and GEMM all"
                        " concurrent (M5)")
    parser.add_argument("--weight_issue_order", default="weights_first",
                        choices=["weights_first", "tokens_first"],
                        help="host enqueue order of the two wire flows."
                        " weights_first (legacy): weight movement issued on"
                        " w_stream first, forward launch waits pref_end"
                        " (issue serialization). tokens_first (E1, NR-14):"
                        " the fused forward (token a2av + GEMM) is enqueued"
                        " FIRST and the weight push after -- pure"
                        " fire-ordering, no completion waits between the"
                        " flows; requires push + tiles gate")
    parser.add_argument("--check_staged", default=False, action="store_true",
                        help="also run one staged MoonEPLayer0Runner"
                        " iteration on the same plan and compare outputs"
                        " through the order-theorem row map")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


@torch.no_grad()
def perf_moonep_fused(args, op, op_w, moe_ctx, wfull, splits_gpu, scatter_gpu,
                      meta, out_buf, epn):
    total_iters = args.warmup_iters + args.iters
    ev = {
        name: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for name in ("start", "pref_start", "pref_end", "gw_end", "gate", "end")
    }
    w_stream = torch.cuda.Stream(priority=-1)
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    push = args.weight_path == "push"
    mcast = args.weight_push_mode == "mcast"
    tiles = push and args.weight_gate == "tiles"
    tokens_first = args.weight_issue_order == "tokens_first"  # implies push+tiles
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for i in range(total_iters):
        op.clear_buffers()
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < args.warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            if tokens_first:
                # E1 (NR-14): tokens-first issue order — the fused forward's
                # token puts are host-issued before any weight leg, so the
                # latency-critical token windows own the NIC first and the
                # bulk weight legs ride behind them. Pure FIRE-ordering: the
                # epoch is peeked (epoch()+1), the weight side is enqueued
                # after op.forward returns, and NO stream waits are added
                # between the flows.
                ev["start"][i].record()
                epoch = op_w.epoch() + 1
                ev["gate"][i].record()  # gate_ms is 0 by definition here
                op.forward(
                    inputs_shard=moe_ctx.inputs_shard,
                    weights=wfull,
                    splits_gpu=splits_gpu,
                    scatter_index=scatter_gpu,
                    outputs_buf=out_buf,
                    fast_accum=False,
                    sm_margin=args.sm_margin,
                    splits_per_source=meta.splits_per_source,
                    a2av_unique_counts=meta.a2av_unique_counts,
                    weight_signal=op_w.signals(),
                    weight_signal_epoch=epoch,
                    weight_gate_group_start=epn,
                )
                # DEADLOCK RULE: w_stream may wait ONLY on ev["start"] — any
                # event recorded after op.forward would order the weight
                # ISSUE after completion of the GEMM, whose prefetch-slot
                # tiles spin on these very signals (cycle).
                # QUIET DEFERRAL: iteration i's weight nbi tail is now only
                # quieted by iteration i+1's a2av barrier_all. Benign in this
                # benchmark (immutable weight_home rows, monotonic GEQ epoch
                # signals, same-value slot rewrites, main stream ends behind
                # gw_end) but NOT free for a real model whose weights change
                # between layers — see the NR-14 ledger caveat.
                with torch.cuda.stream(w_stream):
                    w_stream.wait_event(ev["start"][i])
                    ev["pref_start"][i].record()
                    got = op_w.forward(multicast=mcast)
                    assert got == epoch == i + 1, \
                        f"epoch skew: {got} vs peeked {epoch} vs iter {i + 1}"
                    ev["pref_end"][i].record()
                    op_w.forward_gateway()
                    ev["gw_end"][i].record()
                torch.cuda.current_stream().wait_event(ev["gw_end"][i])
                ev["end"][i].record()
                continue
            ev["start"][i].record()
            # weight movement on a side stream: getmem = destination pull
            # (source ranks passive, quiet join); push = home-rank CE
            # putmem_signal into destination slots (nbi issue -- pref events
            # bracket the ISSUE, the landing shows up in gate_ms)
            epoch = 0
            with torch.cuda.stream(w_stream):
                w_stream.wait_event(ev["start"][i])
                ev["pref_start"][i].record()
                if push:
                    epoch = op_w.forward(multicast=mcast)
                else:
                    op_w.forward(
                        op_w.prefetch_slots(),
                        args.num_comm_sm,
                        args.prefetch_chunk_bytes,
                        args.prefetch_impl == "kernel",
                    )
                ev["pref_end"][i].record()
                if push:
                    # NR-13 F-B: the gateway's slot-arrival wait + NVLink
                    # fan-out run AFTER pref_end, so the forward launch never
                    # waits on weight ARRIVAL — pref_end now means "home puts
                    # issued" on every rank. The fan-out overlaps the fused
                    # forward; gw_end is joined back in after it (below).
                    op_w.forward_gateway()
                    ev["gw_end"][i].record()
            # nbi-issue ordering: the fused forward's end-of-iteration
            # barrier_all only quiets puts already issued in stream order
            torch.cuda.current_stream().wait_event(ev["pref_end"][i])
            if push:
                assert epoch == i + 1, f"epoch skew: {epoch} != {i + 1}"
            if push and args.weight_gate == "join":
                # destination landing gate: zero-SM waits on my slots
                op_w.join()
            # getmem path: quiet ran on w_stream; the event join above IS the
            # landing proof (completion is locally observable)
            ev["gate"][i].record()
            gate_kwargs = {}
            if tiles:
                # M5 concurrency: no destination-side join — only prefetch-
                # slot tiles spin on their slot's weight epoch signal, so
                # dispatch wire, weight wire, and GEMM are all in flight
                gate_kwargs = dict(
                    weight_signal=op_w.signals(),
                    weight_signal_epoch=epoch,
                    weight_gate_group_start=epn,
                )
            op.forward(
                inputs_shard=moe_ctx.inputs_shard,
                weights=wfull,
                splits_gpu=splits_gpu,
                scatter_index=scatter_gpu,
                outputs_buf=out_buf,
                fast_accum=False,
                sm_margin=args.sm_margin,
                splits_per_source=meta.splits_per_source,
                a2av_unique_counts=meta.a2av_unique_counts,
                **gate_kwargs,
            )
            if push:
                # drain the concurrent gateway fan-out inside the iteration
                # (keeps next iteration's slot rewrites happens-after, and
                # the timing bracket honest)
                torch.cuda.current_stream().wait_event(ev["gw_end"][i])
            ev["end"][i].record()

    times = {"e2e_ms": [], "prefetch_ms": [], "gate_ms": [], "fused_ms": []}
    for i in range(total_iters):
        ev["end"][i].synchronize()
        if i < args.warmup_iters:
            continue
        # bracket semantics under tokens_first (E1): prefetch_ms = the weight
        # ISSUE window, now concurrent with the fused window; gate_ms = 0 by
        # definition (no serialization point exists); fused_ms spans the whole
        # forward+drain. Only e2e_ms is comparable across issue orders.
        times["e2e_ms"].append(ev["start"][i].elapsed_time(ev["end"][i]))
        times["prefetch_ms"].append(ev["pref_start"][i].elapsed_time(ev["pref_end"][i]))
        times["gate_ms"].append(
            0.0 if tokens_first else ev["pref_end"][i].elapsed_time(ev["gate"][i])
        )
        times["fused_ms"].append(ev["gate"][i].elapsed_time(ev["end"][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[args.warmup_iters :]
    return times


@torch.no_grad()
def check_correctness(args, plan, vmap, meta, moe_ctx, op_w, wfull, out_buf):
    """Three gates: (V3a) prefetch slots bitwise vs an NCCL broadcast from
    home; (V3b) per-virtual-expert fp32 matmul reference, allclose; optional
    (--check_staged) staged-runner A/B through the order-theorem row map."""
    rank = TP_GROUP.rank()
    cfg = plan.cfg
    gpe, epn = vmap.gpe, cfg.epn
    ffn_shard = moe_ctx.ffn_size_shard
    ok_bitwise = True

    # V3a: every prefetch pair, independent NCCL-broadcast code path
    pairs_all = [
        (d, b, int(plan.experts_to_copy[d, b]))
        for d in range(cfg.R)
        for b in range(cfg.B)
        if int(plan.experts_to_copy[d, b]) >= 0
    ]
    tmp = torch.zeros(ffn_shard, args.H, dtype=moe_ctx.inputs.dtype, device="cuda")
    for d, b, e in pairs_all:
        home = e // epn
        if rank == home:
            tmp.copy_(moe_ctx.weights[0][e % epn])
        else:
            tmp.zero_()
        torch.distributed.broadcast(tmp, src=home, group=TP_GROUP)
        if d == rank and not torch.equal(wfull[epn + b], tmp):
            ok_bitwise = False
            print(f"rank {rank}: V3a slot {b} (expert {e}) mismatch")

    # V3b: fp32 reference per virtual expert on the fused output rows
    torch.distributed.all_gather_into_tensor(
        moe_ctx.inputs, moe_ctx.inputs_shard, group=TP_GROUP
    )
    vce_flat = vmap.virtual_choosed.long().flatten()
    splits = meta.splits.long()
    ok_allclose = True
    base = 0
    for g in range(gpe):
        v = rank * gpe + g
        cnt = int(splits[v])
        if cnt == 0:
            continue
        idx = (vce_flat == v).nonzero(as_tuple=True)[0]
        tokens = torch.div(idx, cfg.K, rounding_mode="floor").cuda()
        ref = torch.matmul(
            moe_ctx.inputs.index_select(0, tokens).float(), wfull[g].float().t()
        )
        got = out_buf[base : base + cnt].float()
        try:
            flux.torch_allclose(got, ref, atol=1e-2, rtol=1.5e-2, verbose=False)
        except RuntimeError:
            ok_allclose = False
            print(f"rank {rank}: V3b virtual expert {v} (group {g}) mismatch")
        base += cnt

    if args.check_staged:
        from flux.testing.moonep_semantics import MoonEPLayer0Runner

        runner = MoonEPLayer0Runner(
            plan, rank, TP_GROUP, torch.cuda.current_device(),
            dtype=moe_ctx.inputs.dtype, ffn_size_shard=ffn_shard,
        )
        gen = torch.Generator().manual_seed(777)
        w_all = torch.rand(cfg.R, cfg.S, cfg.K, dtype=torch.float32, generator=gen)
        route_weights = w_all[rank].cuda()
        gemm_only_op = flux.GemmOnly(
            moe_ctx.inputs.dtype, moe_ctx.inputs.dtype,
            moe_ctx.outputs[0].dtype, use_fp8_gemm=False,
        )
        runner.pack(moe_ctx.inputs_shard, route_weights)
        runner.a2av()
        runner.place_and_epilogue()
        runner.prefetch(moe_ctx.weights[0])
        runner.gemm(gemm_only_op, moe_ctx.weights[0])
        torch.cuda.synchronize()
        fused_rows, plan_slots = fused_row_map(vmap, rank)
        got = out_buf.index_select(0, fused_rows.cuda()).float()
        ref = runner.out_buf.index_select(0, plan_slots.cuda()).float()
        # different GEMM kernels (grouped CUTLASS vs GemmOnly): allclose bar
        try:
            flux.torch_allclose(got, ref, atol=1e-2, rtol=1.5e-2, verbose=False)
        except RuntimeError:
            ok_allclose = False
            print(f"rank {rank}: staged-vs-fused row-map mismatch")

    RECORDER.emit_correctness(bitwise=ok_bitwise, allclose=ok_allclose)
    assert ok_bitwise and ok_allclose, "correctness check failed"


if __name__ == "__main__":
    args = parse_args()
    W = DIST_ENV.WORLD_SIZE
    rank = TP_GROUP.rank()
    init_ep_group(W)
    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.H * input_dtype.itemsize == args.chunk_bytes
    assert args.G % W == 0
    assert W <= 64, "fused gating segment masks cap world_size at 64"

    if args.weight_issue_order == "tokens_first":
        # join's destination gate and getmem's landing join must precede
        # op.forward on-stream — tokens-first is definitionally impossible
        # for them; only the tile-gated push path can start compute first.
        assert args.weight_path == "push" and args.weight_gate == "tiles", (
            "--weight_issue_order tokens_first requires --weight_path push "
            "--weight_gate tiles"
        )

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W
    if args.routing_file:
        choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
        assert choosed_experts.shape[0] % W == 0
        got = choosed_experts_to_matrix_chunks(choosed_experts, W, args.G // W)
        assert torch.equal(got * args.chunk_bytes, matrix)
    else:
        if rank == 0:
            print("WARNING: synthetic dealer routing (max-dedup); use a"
                  " --routing_file for real token-overlap semantics")
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
    ntokens = choosed_experts.shape[0]
    S = ntokens // W

    cfg = MoonEPConfig(
        S=S, K=args.topk, E=args.G, R=W, H=args.H,
        token_padding=args.token_padding,
    )
    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    t0 = time.perf_counter()
    plan = compute_moonep_plan(cfg, topk_all)
    vmap = build_virtual_map(plan, topk_all)
    meta = build_fused_metadata(vmap, DIST_ENV.LOCAL_WORLD_SIZE)
    plan_host_ms = (time.perf_counter() - t0) * 1e3
    preflight_metadata_checks(meta, W, DIST_ENV.LOCAL_WORLD_SIZE)

    # cross-rank determinism guard (plan + derived metadata are replicated)
    h = torch.tensor([plan.plan_hash()], dtype=torch.int64, device="cuda")
    h_all = torch.zeros(W, dtype=torch.int64, device="cuda")
    torch.distributed.all_gather_into_tensor(h_all, h, group=TP_GROUP)
    assert bool((h_all == h_all[0]).all()), "plan hash differs across ranks"

    # exact capacity knobs, set BEFORE op construction (ctor-read); explicit
    # env wins so a cell can still override deliberately
    knobs = required_a2av_knobs(meta, W, DIST_ENV.LOCAL_WORLD_SIZE)
    for k, v in knobs.items():
        os.environ.setdefault(k, v)

    flux.init_flux_shm(TP_GROUP)

    gating_args = gen_moe_gating_args(
        args.G, args.topk, ntokens, choosed_experts=choosed_experts
    )
    moe_ctx = MoeMlp1Ctx(
        TP_GROUP, EP_GROUP,
        b=1, s=ntokens, h=args.H,
        ffn_size=args.ffn_hidden_size,
        nexperts=args.G, topk=args.topk,
        input_dtype=input_dtype, output_dtype=output_dtype,
        dist="uniform", fast_accum=False, weight_groups=1, drop_token=False,
        gating_args=gating_args, skip_reference=True,
    )
    ffn_shard = moe_ctx.ffn_size_shard
    epn, B, gpe = cfg.epn, cfg.B, vmap.gpe

    # contiguous [epn+B] weight tensor on the symmetric heap (home rows +
    # prefetch slots) -- the op's single weight group AND the weight-movement
    # source/destination. Exactly one weight op is live per run.
    push_mode_requested = args.weight_push_mode
    if args.weight_path == "push":
        op_w = flux.WeightPushMulticast(
            TP_GROUP, epn, B, ffn_shard, args.H, input_dtype,
        )
        push_pairs = assign_gateways(plan, DIST_ENV.LOCAL_WORLD_SIZE)
        op_w.set_plan(push_pairs)
        pstats = push_plan_stats(push_pairs, DIST_ENV.LOCAL_WORLD_SIZE)
        if args.weight_push_mode == "auto":
            # F-C: engage the gateway machinery only when the plan actually
            # has something to multicast (replicated census -> identical
            # resolution on every rank)
            args.weight_push_mode = "mcast" if pstats["n_multi_groups"] > 0 else "direct"
        if rank == 0:
            print(f"push plan census: {pstats} -> mode {args.weight_push_mode}"
                  f" (requested {push_mode_requested})")
            RECORDER.emit_info(
                wpush_mode_requested=push_mode_requested,
                wpush_mode_resolved=args.weight_push_mode,
                **{f"wpush_{k}": v for k, v in pstats.items()},
            )
        if rank == 0:
            # wire accounting in bytes (derived from the census)
            ebytes = ffn_shard * args.H * input_dtype.itemsize
            RECORDER.emit_info(
                wpush_internode_bytes_direct=pstats["n_cross_legs"] * ebytes,
                wpush_internode_bytes_mcast=pstats["n_cross_groups"] * ebytes,
            )
    else:
        assert args.weight_push_mode == "direct", "--weight_push_mode needs --weight_path push"
        op_w = flux.WeightPrefetchGetmem(
            TP_GROUP, epn, B, ffn_shard, args.H, input_dtype,
            contiguous_layout=True,
        )
        my_pairs = [
            (b, int(plan.experts_to_copy[rank, b]) // epn,
             int(plan.experts_to_copy[rank, b]) % epn)
            for b in range(B)
            if int(plan.experts_to_copy[rank, b]) >= 0
        ]
        op_w.set_pairs(torch.tensor(my_pairs, dtype=torch.int32).reshape(-1, 3))
    op_w.weight_home().copy_(moe_ctx.weights[0])
    wfull = op_w.weight_full()

    tp_env = flux.DistEnvTPWithEP(
        tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP
    )
    moe_args = flux.MoeArguments(
        max_ntokens=ntokens,
        hidden=args.H,
        ffn_hidden=args.ffn_hidden_size,
        nexperts=vmap.E_virt,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    op = flux.GemmGroupedV2AGScatterOp(
        tp_env=tp_env,
        moe_args=moe_args,
        a2av_dispatch=True,
        a2av_hier_compress=True,
    )

    splits_gpu = meta.splits.cuda()
    scatter_gpu = meta.scatter_index.cuda()
    out_buf = torch.zeros(
        int(meta.m_per_rank[rank]), ffn_shard, dtype=output_dtype, device="cuda"
    )

    if rank == 0:
        wire_rows = [
            [int(((plan.dst[s] >= 0)
                  & (torch.div(plan.dst[s].long(), cfg.NvS,
                               rounding_mode="floor") == d)).sum())
             for d in range(W)]
            for s in range(W)
        ]
        print(f"moonep_fused: E_virt={vmap.E_virt} gpe={gpe} plan_host_ms={plan_host_ms:.1f}")
        print(f"gemm rows per rank: {meta.m_per_rank.tolist()}")
        print(f"knobs: {knobs}")
        RECORDER.emit_info(
            moonep_plan_host_ms=plan_host_ms,
            moonep_e_virt=vmap.E_virt,
            moonep_gemm_rows=meta.m_per_rank.tolist(),
            moonep_wire_bytes=int(sum(sum(r) for r in wire_rows)) * args.chunk_bytes,
            moonep_z_matrix=plan.z.tolist(),
            weight_path=args.weight_path,
            weight_push_mode=args.weight_push_mode,
            weight_gate=args.weight_gate,
            weight_issue_order=args.weight_issue_order,
            **{f"knob_{k}": v for k, v in knobs.items()},
        )

    times = perf_moonep_fused(
        args, op, op_w, moe_ctx, wfull, splits_gpu, scatter_gpu, meta, out_buf, epn
    )
    RECORDER.emit_iters("flux", times)
    e2e = sum(times["e2e_ms"]) / len(times["e2e_ms"])
    pref = sum(times["prefetch_ms"]) / len(times["prefetch_ms"])
    fused = sum(times["fused_ms"]) / len(times["fused_ms"])
    print(f"rank {rank}: e2e {e2e:.3f} ms (prefetch {pref:.3f} + fused {fused:.3f})")

    if not args.skip_correctness:
        check_correctness(args, plan, vmap, meta, moe_ctx, op_w, wfull, out_buf)
        if rank == 0:
            print("correctness OK (V3a bitwise slots, V3b allclose gemm"
                  + (", staged row-map A/B" if args.check_staged else "") + ")")
