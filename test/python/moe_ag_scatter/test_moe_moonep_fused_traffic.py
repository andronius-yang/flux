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
    egress_byte_stats,
    fused_row_map,
    plan_weight_shards,
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
    parser.add_argument("--weight_shard", default="off",
                        choices=["off", "auto", "on"],
                        help="egress NIC-sharding of cross-node weight legs"
                        " (push only): byte-split each wire leg across the"
                        " home node's same-local-rank wires with dest-side"
                        " NVLink reassembly, so all L NICs carry it on both"
                        " ends. auto = shard iff expert_bytes >="
                        " --weight_shard_min_bytes and cross legs exist;"
                        " on = shard every cross leg regardless of size")
    parser.add_argument("--weight_shard_min_bytes", type=int, default=8 << 20,
                        help="auto-mode threshold (microbench-calibrated;"
                        " below it a leg keeps the single-NIC path)")
    parser.add_argument("--weight_shard_chunk_bytes", type=int, default=0,
                        help="per-chunk staging signals pipeline the NVLink"
                        " stage against the NIC push; 0 = whole-shard chunks")
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
    parser.add_argument("--layers", default="l0", choices=["l0", "l01"],
                        help="l01 appends the optimized layer1: gelu + the"
                        " fused gather-rs op (gemm2+combine) over the SAME"
                        " virtual expert space with INHERITED combine"
                        " metadata (built once at setup — the no-recalc l01"
                        " contract). w2 rides a second weight op issued in"
                        " the same pref bracket as w1 (both matrices upfront"
                        " — upstream one-pass prefetch, api.py:158-173); an"
                        " explicit op_w2.join() gates gemm2 (v1: no gemm2"
                        " tile gate — named follow-up). The l1 push op's"
                        " weight_full feeds gather_rs with zero copies.")
    parser.add_argument("--l1_comm_pattern", default="a2av_hier_compress",
                        choices=["a2av_hier", "a2av_hier_compress"],
                        help="combine transport for --layers l01 (compress ="
                        " the W16 combined winner; degrades to hier on 1"
                        " node)")
    parser.add_argument("--l1_n_split", type=int, default=4)
    parser.add_argument("--check_staged", default=False, action="store_true",
                        help="also run one staged MoonEPLayer0Runner"
                        " iteration on the same plan and compare outputs"
                        " through the order-theorem row map")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


@torch.no_grad()
def perf_moonep_fused(args, op, op_w, moe_ctx, wfull, splits_gpu, scatter_gpu,
                      meta, out_buf, epn, sharded=False, op_w2=None, l1=None):
    total_iters = args.warmup_iters + args.iters
    names = ["start", "pref_start", "pref_end", "gw_end", "gate", "end"]
    if l1 is not None:
        names += ["l0_end", "act_end", "l1_join"]
    ev = {
        name: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for name in names
    }
    w_stream = torch.cuda.Stream(priority=-1)
    # NR-13 F-B extended to sharding: every shard-machinery wait (egress
    # chunk waits, ingress chunk waits, the dest finalize) lives on its own
    # late-drained stream, never in any issue window. The gateway fan-out
    # rides AFTER the finalize when sharding is on, since gateway slots may
    # themselves be shard-fed (its GEQ-epoch wait is satisfied by the
    # finalize SET).
    shard_stream = torch.cuda.Stream(priority=-1) if sharded else None
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    push = args.weight_path == "push"
    mcast = args.weight_push_mode == "mcast"
    tiles = push and args.weight_gate == "tiles"
    tokens_first = args.weight_issue_order == "tokens_first"  # implies push+tiles

    w_ops = [op_w] + ([op_w2] if op_w2 is not None else [])

    def emit_shard_chain(i):
        # DEADLOCK RULE compliance under tokens_first: shard_stream waits
        # only on pref_end, which w_stream records after its own wait-free
        # issue window — nothing here is ordered after op.forward's
        # completion, and every CUStreamWaitValue's satisfying writer is a
        # remote rank's wait-free issue (or an earlier op on this stream).
        with torch.cuda.stream(shard_stream):
            shard_stream.wait_event(ev["pref_end"][i])
            for w in w_ops:
                w.forward_egress()
                w.forward_ingress()
                w.forward_shard_join()
                w.forward_gateway()
            ev["gw_end"][i].record()

    def emit_l1_chain(i):
        # optimized layer1 on the main stream: gelu -> explicit w2 landing
        # gate (v1: zero-SM join, no gemm2 tile gate — named follow-up) ->
        # fused gather-rs (gemm2 + combine) with INHERITED metadata. The
        # join sits after the whole l0 window, so it is almost always
        # already satisfied by the time the stream reaches it.
        ev["l0_end"][i].record()
        l1["act_buf"].copy_(torch.nn.functional.gelu(out_buf))
        ev["act_end"][i].record()
        if push:
            op_w2.join()  # w2 landing gate (getmem: pref_end join proved it)
        ev["l1_join"][i].record()
        l1["out"] = l1["op"].forward_gather_rs(
            l1["act_buf"], l1["wfull2"], l1["split"], l1["routing"],
            input_scale=l1["input_scales"],
            weight_scale=l1["weight_scales"],
            output_vec_scale=l1["vec"],
            fast_accum=False,
            sm_margin=args.sm_margin,
            bias=None,
            **l1["kwargs"],
        )
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
                    if op_w2 is not None:
                        # both matrices in ONE issue window (upstream
                        # one-pass prefetch); w2 has no tile gate, its
                        # landing gate is the join inside emit_l1_chain
                        op_w2.forward(multicast=mcast)
                    ev["pref_end"][i].record()
                    if not sharded:
                        for w in w_ops:
                            w.forward_gateway()
                        ev["gw_end"][i].record()
                if sharded:
                    emit_shard_chain(i)
                if l1 is not None:
                    emit_l1_chain(i)
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
                    if op_w2 is not None:
                        op_w2.forward(multicast=mcast)
                else:
                    op_w.forward(
                        op_w.prefetch_slots(),
                        args.num_comm_sm,
                        args.prefetch_chunk_bytes,
                        args.prefetch_impl == "kernel",
                    )
                    if op_w2 is not None:
                        op_w2.forward(
                            op_w2.prefetch_slots(),
                            args.num_comm_sm,
                            args.prefetch_chunk_bytes,
                            args.prefetch_impl == "kernel",
                        )
                ev["pref_end"][i].record()
                if push and not sharded:
                    # NR-13 F-B: the gateway's slot-arrival wait + NVLink
                    # fan-out run AFTER pref_end, so the forward launch never
                    # waits on weight ARRIVAL — pref_end now means "home puts
                    # issued" on every rank. The fan-out overlaps the fused
                    # forward; gw_end is joined back in after it (below).
                    for w in w_ops:
                        w.forward_gateway()
                    ev["gw_end"][i].record()
            if push and sharded:
                emit_shard_chain(i)
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
            if l1 is not None:
                emit_l1_chain(i)
            if push:
                # drain the concurrent gateway fan-out inside the iteration
                # (keeps next iteration's slot rewrites happens-after, and
                # the timing bracket honest)
                torch.cuda.current_stream().wait_event(ev["gw_end"][i])
            ev["end"][i].record()

    times = {"e2e_ms": [], "prefetch_ms": [], "gate_ms": [], "fused_ms": []}
    if l1 is not None:
        times.update({"act_ms": [], "l1_join_ms": [], "l1_ms": []})
    if sharded:
        # the shard machinery window: pref_end (home issue done) -> gw_end
        # (egress + reassembly + finalize + gateway drained). Overlaps the
        # fused window under the tiles gate; serial ahead of gate under join.
        times["shard_ms"] = []
    for i in range(total_iters):
        ev["end"][i].synchronize()
        if i < args.warmup_iters:
            continue
        if sharded:
            times["shard_ms"].append(ev["pref_end"][i].elapsed_time(ev["gw_end"][i]))
        # bracket semantics under tokens_first (E1): prefetch_ms = the weight
        # ISSUE window, now concurrent with the fused window; gate_ms = 0 by
        # definition (no serialization point exists); fused_ms spans the whole
        # forward+drain. Only e2e_ms is comparable across issue orders.
        times["e2e_ms"].append(ev["start"][i].elapsed_time(ev["end"][i]))
        times["prefetch_ms"].append(ev["pref_start"][i].elapsed_time(ev["pref_end"][i]))
        times["gate_ms"].append(
            0.0 if tokens_first else ev["pref_end"][i].elapsed_time(ev["gate"][i])
        )
        if l1 is not None:
            # l01 brackets: fused_ms narrows to the l0 window; prefetch_ms
            # stays its own bracket (the future persistent-experts baseline)
            times["fused_ms"].append(ev["gate"][i].elapsed_time(ev["l0_end"][i]))
            times["act_ms"].append(ev["l0_end"][i].elapsed_time(ev["act_end"][i]))
            times["l1_join_ms"].append(ev["act_end"][i].elapsed_time(ev["l1_join"][i]))
            times["l1_ms"].append(ev["l1_join"][i].elapsed_time(ev["end"][i]))
        else:
            times["fused_ms"].append(ev["gate"][i].elapsed_time(ev["end"][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[args.warmup_iters :]
    return times


@torch.no_grad()
def check_correctness_l01(args, moe_ctx, choosed_experts, w_tok, full_w2,
                          l1_out, epn, atol, rtol):
    """Independent two-layer reference for the optimized l01 journey: for
    each of MY tokens, sum over its top-k entries of
    route_w * gelu(x @ w1_e^T) @ w2_e^T — from the RAW ORIGINAL routing and
    inputs (never the virtual space or the ops' buffers). w1 arrives per
    expert via NCCL broadcast; w2 is replicated by construction. Rounding
    points mirror the pipeline (bf16 casts after each GEMM); the reference
    accumulates in fp32 — absorbed by the tolerances."""
    rank = TP_GROUP.rank()
    W = TP_GROUP.size()
    ntokens = choosed_experts.shape[0]
    S = ntokens // W
    x = moe_ctx.inputs_shard  # [S, H] my tokens
    my_topk = choosed_experts[rank * S:(rank + 1) * S].to(x.device)
    ref = torch.zeros(S, args.H, dtype=torch.float32, device=x.device)
    ffn_shard = moe_ctx.ffn_size_shard
    tmp_w1 = torch.empty(ffn_shard, args.H, dtype=x.dtype, device=x.device)
    for e in range(args.G):
        home = e // epn
        if home == rank:
            tmp_w1.copy_(moe_ctx.weights[0][e % epn])
        torch.distributed.broadcast(tmp_w1, src=home, group=TP_GROUP)
        mask = my_topk == e
        if not bool(mask.any()):
            continue
        s_idx, k_idx = mask.nonzero(as_tuple=True)
        h1 = torch.matmul(x[s_idx].float(), tmp_w1.float().t()).to(x.dtype)
        a = torch.nn.functional.gelu(h1)
        w2e = full_w2[e].to(x.device)
        h2 = torch.matmul(a.float(), w2e.float().t()).to(x.dtype)
        ref[s_idx] += h2.float() * w_tok[rank * S + s_idx.cpu(), k_idx.cpu()].to(
            x.device
        ).unsqueeze(1)
    try:
        flux.torch_allclose(l1_out, ref.to(l1_out.dtype), atol=atol, rtol=rtol)
    except Exception as e:  # noqa: BLE001
        print(f"❌ rank {rank}: l01 combined output vs two-layer reference"
              " MISMATCH")
        RECORDER.emit_correctness(bitwise=False, allclose=False)
        RECORDER.flush()
        raise e
    print(f"✅ rank {rank}: l01 combined output matches the two-layer"
          " reference (allclose)")


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
    assert W <= 128, "progress buckets cap world_size at 128 (gate itself is W-unbounded)"

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
    weight_sharded = False
    if args.weight_shard != "off":
        assert args.weight_path == "push", "--weight_shard needs --weight_path push"
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
        if args.weight_shard != "off":
            # egress NIC-sharding: byte-split cross-node wire legs of the
            # RESOLVED mode across same-local-rank wires (replicated table,
            # collective set_shard_plan). auto respects the size threshold;
            # on shards every cross leg regardless.
            ebytes = ffn_shard * args.H * input_dtype.itemsize
            shard_min = args.weight_shard_min_bytes if args.weight_shard == "auto" else 1
            shard_table = plan_weight_shards(
                push_pairs, DIST_ENV.LOCAL_WORLD_SIZE, ebytes,
                mode=args.weight_push_mode, min_bytes=shard_min,
            )
            weight_sharded = shard_table.shape[0] > 0
            if weight_sharded:
                op_w.set_shard_plan(
                    shard_table, args.weight_shard_chunk_bytes,
                    DIST_ENV.LOCAL_WORLD_SIZE,
                )
            bstats = egress_byte_stats(
                push_pairs, DIST_ENV.LOCAL_WORLD_SIZE, W, ebytes,
                mode=args.weight_push_mode,
                shards=shard_table if weight_sharded else None,
            )
            if rank == 0:
                print(f"weight shard: requested {args.weight_shard} ->"
                      f" {'on' if weight_sharded else 'off'}"
                      f" ({shard_table.shape[0]} shard legs); egress census:"
                      f" {bstats}")
                RECORDER.emit_info(
                    wshard_requested=args.weight_shard,
                    wshard_resolved="on" if weight_sharded else "off",
                    wshard_n_legs=int(shard_table.shape[0]),
                    wshard_max_rank_egress_bytes=bstats["max_rank_egress_bytes"],
                    wshard_sharded_max_rank_egress_bytes=bstats.get(
                        "sharded_max_rank_egress_bytes",
                        bstats["max_rank_egress_bytes"],
                    ),
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

    # ---- optimized layer1 (--layers l01): second weight op + fused
    # gather-rs over the SAME virtual space with INHERITED metadata ----
    op_w2 = None
    l1_bundle = None
    full_w2 = None
    w_tok = None
    if args.layers == "l01":
        from flux.testing.a2av_combine_indices import (
            build_a2av_combine_indices,
            build_a2av_compress_indices,
            build_a2av_unique_counts,
        )
        from flux.testing.moonep_fused_map import required_a2av_rs_knobs

        gen2 = torch.Generator().manual_seed(1234)
        full_w2 = (
            torch.rand((args.G, args.H, ffn_shard), generator=gen2) * 0.02 - 0.01
        ).to(input_dtype)
        w2_home = full_w2[rank * epn:(rank + 1) * epn].cuda().contiguous()
        if args.weight_path == "push":
            op_w2 = flux.WeightPushMulticast(
                TP_GROUP, epn, B, args.H, ffn_shard, input_dtype,
            )
            op_w2.set_plan(push_pairs)
            if weight_sharded:
                # same table: w2's expert_bytes equal w1's (H*ffn == ffn*H)
                op_w2.set_shard_plan(
                    shard_table, args.weight_shard_chunk_bytes,
                    DIST_ENV.LOCAL_WORLD_SIZE,
                )
        else:
            op_w2 = flux.WeightPrefetchGetmem(
                TP_GROUP, epn, B, args.H, ffn_shard, input_dtype,
                contiguous_layout=True,
            )
            op_w2.set_pairs(torch.tensor(my_pairs, dtype=torch.int32).reshape(-1, 3))
        op_w2.weight_home().copy_(w2_home)

        # exact RS knobs BEFORE the gather_rs ctor (parity-tested formulas)
        for k, v in required_a2av_rs_knobs(meta, W, DIST_ENV.LOCAL_WORLD_SIZE).items():
            os.environ.setdefault(k, v)
        use_l1_compress = args.l1_comm_pattern == "a2av_hier_compress"
        M_all = ntokens * args.topk
        l1_op = flux.GemmGroupedV2GatherRSOp(
            TP_GROUP, vmap.E_virt, M_all, args.H, args.topk, output_dtype,
            1, W, 1,
            nnodes=DIST_ENV.NNODES, n_split=args.l1_n_split,
            do_all_reduce=False, use_read_mode=False,
            a2av_hier=not use_l1_compress,
            a2av_hier_compress=use_l1_compress,
        )
        # INHERITED combine metadata, built ONCE (the no-recalc contract):
        # split/routing/dedup come from the same plan metadata the l0 op
        # consumes; the amortized index/CSR set is what a fused pipeline
        # hands over.
        split_cpu_v = meta.splits
        routing_v = scatter_gpu.flatten()
        l1_kwargs = {"splits_per_source": meta.splits_per_source}
        uc = None
        if use_l1_compress and DIST_ENV.NNODES > 1:
            uc = build_a2av_unique_counts(
                vmap.virtual_choosed.long(), W, DIST_ENV.NNODES, gpe
            )
            l1_kwargs["a2av_unique_counts"] = uc
        pack_index, reduce_index = build_a2av_combine_indices(
            routing_v, split_cpu_v, rank, W, args.topk
        )
        l1_kwargs["a2av_pack_index"] = pack_index
        l1_kwargs["a2av_reduce_index"] = reduce_index
        if uc is not None:
            wire_ptr, wire_copy, red_ptr, red_row = build_a2av_compress_indices(
                routing_v, split_cpu_v, uc, rank, W, DIST_ENV.NNODES, args.topk
            )
            l1_kwargs["a2av_wire_csr"] = [wire_ptr, wire_copy]
            l1_kwargs["a2av_reduce_csr"] = [red_ptr, red_row]

        # per-copy route weights (replicated seed) -> this rank's gemm-row
        # vec scale via the inverse of the virtual scatter index
        gen3 = torch.Generator().manual_seed(777)
        w_tok = torch.rand((ntokens, args.topk), generator=gen3) + 0.5
        m_start = int(meta.splits[: rank * gpe].sum())
        M_cur = int(meta.m_per_rank[rank])
        rows = meta.scatter_index.flatten().long()
        mask = (rows >= m_start) & (rows < m_start + M_cur)
        t_idx = (torch.arange(ntokens * args.topk) // args.topk)[mask]
        k_idx = (torch.arange(ntokens * args.topk) % args.topk)[mask]
        vec_v = torch.zeros(max(M_cur, 1), dtype=torch.float32)
        vec_v[rows[mask] - m_start] = w_tok[t_idx, k_idx]
        l1_bundle = {
            "op": l1_op,
            "wfull2": op_w2.weight_full(),
            "split": split_cpu_v,
            "routing": routing_v,
            "vec": vec_v[:M_cur].cuda(),
            "kwargs": l1_kwargs,
            "input_scales": torch.ones((1,), dtype=torch.float32, device="cuda"),
            "weight_scales": torch.ones((gpe,), dtype=torch.float32, device="cuda"),
            "act_buf": torch.zeros(M_cur, ffn_shard, dtype=output_dtype, device="cuda"),
            "out": None,
        }

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
            weight_shard=args.weight_shard,
            weight_layers=args.layers,
            **{f"knob_{k}": v for k, v in knobs.items()},
        )

    times = perf_moonep_fused(
        args, op, op_w, moe_ctx, wfull, splits_gpu, scatter_gpu, meta, out_buf, epn,
        sharded=weight_sharded, op_w2=op_w2, l1=l1_bundle,
    )
    RECORDER.emit_iters("flux", times)
    e2e = sum(times["e2e_ms"]) / len(times["e2e_ms"])
    pref = sum(times["prefetch_ms"]) / len(times["prefetch_ms"])
    fused = sum(times["fused_ms"]) / len(times["fused_ms"])
    print(f"rank {rank}: e2e {e2e:.3f} ms (prefetch {pref:.3f} + fused {fused:.3f})")

    if not args.skip_correctness:
        check_correctness(args, plan, vmap, meta, moe_ctx, op_w, wfull, out_buf)
        if args.layers == "l01":
            if input_dtype == torch.float16:
                l01_atol, l01_rtol = 1e-2, 1e-3
            else:
                l01_atol, l01_rtol = 1e-2, 1.5e-2
            check_correctness_l01(
                args, moe_ctx, choosed_experts.cpu(), w_tok, full_w2,
                l1_bundle["out"], epn, l01_atol, l01_rtol,
            )
        if rank == 0:
            print("correctness OK (V3a bitwise slots, V3b allclose gemm"
                  + (", staged row-map A/B" if args.check_staged else "")
                  + (", l01 two-layer reference" if args.layers == "l01" else "")
                  + ")")
