################################################################################
#
# MoonEP-semantics layer0 arm for the sweep harness.
# Structural clone of test_moe_ag_traffic.py; algorithm semantics ported from
# MoonshotAI/MoonEP (see python/flux/testing/moonep_semantics.py and the
# vendored oracle under moonep_oracle/).
#
################################################################################
"""COMET-style layer0 benchmark under MoonEP's redundant-expert dispatch.

Pipeline per iteration (each phase bracketed with CUDA events):
  plan_comm  all_gather of the [S, K] topk routing (replicated planning wire;
             stands in for MoonEP's tpe-push + plan multicast + src_info)
  pack       representative-row gather into the dest-sorted send buffer
             (port-added local copy, kept out of comm_ms)
  comm       all_to_all_single of dedup'd representative rows + per-entry
             fp32 route weights (wire rows/bytes == MoonEP dedup semantics)
  scatter    placement scatter to plan slots + zero-fill + local duplicate
             expansion (MoonEP fused-permute + zero warp + dispatch_epilogue)
  prefetch   redundant expert weights home rank -> prefetch slots
             (reported separately: weight traffic, not token traffic)
  gemm       per-segment GEMM over cu_seqlens[E+B] (padded rows computed)

Timing accounting = per_iter_gpu (SCHEMA protocol rule 5, 2026-08-20): the
AUTHENTIC upstream planner — MoonEP's single fused cooperative planning
kernel, ported to the replicated multi-node setting in
moonep_oracle/planning_port.py (only its cross-rank-sync sites replaced) —
runs PER ITERATION on device inside the `plan` event bracket, followed by
the on-device layout derivation (derive_moonep_layout_gpu). Nothing
routing-derived is cached across iterations; upstream fuses planning into
one kernel, so plan_ms is one bracket, un-separated by design. The CPU
compute_moonep_plan survives only as buffer sizing, the correctness
reference, the cross-rank plan-hash guard, and the loud setup-time
bitwise drift check against the kernel outputs (reported as
plan_host_ms — the setup reference build, NOT the timed planner).
plan_comm is measured every iteration as before: it is the recurring
per-layer wire cost of replicated planning.
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
    MoeMlp1Ctx,
    gen_moe_gating_args,
    choosed_experts_to_matrix_chunks,
    load_routing_file,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.moonep_semantics import (
    MoonEPConfig,
    MoonEPLayer0Runner,
    check_moonep_iter_plan,
    compute_moonep_plan,
    derive_moonep_layout_gpu,
)
from flux.testing.recorder import RECORDER
from flux.testing.payload_probe import PayloadProbe, payload_probe_enabled

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moonep_oracle.planning_port import (  # noqa: E402  (ported planner)
    ReplicatedPlannerWorkspace,
)

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


def plan_iteration(runner, plan_ws, topk_gather_buf):
    """Rule-5 timed planning: per-iteration [R, E] histogram of the
    gathered routing, the authentic ported planner kernel, the on-device
    layout derivation, and the bind — all inside the `plan` bracket."""
    cfg = runner.cfg
    R, E, N = cfg.R, cfg.E, cfg.N
    topk_flat = topk_gather_buf.view(R, N)
    src_base = torch.arange(R, device=topk_flat.device,
                            dtype=torch.int64).unsqueeze(1) * E
    ids = (src_base + topk_flat.long()).reshape(-1)
    if int(os.environ.get("FLUX_PLL_FAST_TAIL", "1")):
        # sync-free histogram twin (bincount hides an output-sizing D2H)
        tpe_all = torch.zeros(R * E, dtype=torch.int64, device=ids.device)
        tpe_all.index_add_(0, ids, torch.ones_like(ids))
        tpe_all = tpe_all.view(R, E).to(torch.int32)
    else:
        tpe_all = torch.bincount(
            ids, minlength=R * E).view(R, E).to(torch.int32)
    dst_all, cu_all, etc_all, zfr_all, _stats = plan_ws.launch(
        topk_flat.contiguous(), tpe_all.contiguous())
    ip = derive_moonep_layout_gpu(cfg, runner.rank, dst_all, zfr_all,
                                  cu_all, etc_all)
    runner.bind_iter_plan(ip)
    return ip


@torch.no_grad()
def perf_moonep(
    runner: MoonEPLayer0Runner,
    plan_ws: ReplicatedPlannerWorkspace,
    ctx: MoeMlp1Ctx,
    route_weights: torch.Tensor,
    topk_shard: torch.Tensor,
    topk_gather_buf: torch.Tensor,
    gemm_only_op,
    warmup_iters: int,
    iters: int,
    do_prefetch: bool = True,
    overlap_prefetch: bool = False,
    shared_comm: bool = False,
    prefetch_group=None,
    prefetch_stream=None,
    layers: str = "l0",
    w2_local=None,
):
    l01 = layers == "l01"
    total_iters = warmup_iters + iters
    names = ["start", "plan_comm", "plan", "pack", "comm", "scatter",
             "prefetch", "gemm"]
    if l01:
        names += ["act", "gemm2", "cpack", "comb", "acc"]
    if overlap_prefetch:
        # prefetch rides its own high-priority stream + its own NCCL
        # communicator (MoonEP async_finish semantics: dedicated comm stream,
        # event-joined by the consumer); "join" marks the main stream having
        # absorbed the prefetch dependency before GEMM.
        names += ["pref_end", "join"]
    ev = {
        name: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for name in names
    }
    torch.distributed.barrier()
    torch.cuda.synchronize()

    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    # wire-ordering audit (CLAUDE.md invariant 5): per-iteration payload
    # randomization; runner.pack reads ctx.inputs_shard every iteration
    probe = PayloadProbe(ctx.inputs_shard, runner.rank)
    for i in range(total_iters):
        if isolated:
            # sweeps SCHEMA.md isolated mode; also the lawful stand-in for
            # MoonEP's inter_rank_sync entry barrier.
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        probe.step(i)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            ev["start"][i].record()
            torch.distributed.all_gather_into_tensor(
                topk_gather_buf, topk_shard, group=TP_GROUP
            )
            ev["plan_comm"][i].record()
            # Per-iteration authentic planning (rule 5): the ported fused
            # planner kernel + on-device layout derivation + bind. NOTHING
            # routing-derived crosses iterations.
            plan_iteration(runner, plan_ws, topk_gather_buf)
            ev["plan"][i].record()
            if overlap_prefetch and do_prefetch and not shared_comm:
                # fork right after plan_comm: weight movement overlaps
                # pack + wire + scatter on the main stream (FINER than
                # upstream, which shares one comm stream — NR-12 fact 7)
                with torch.cuda.stream(prefetch_stream):
                    prefetch_stream.wait_event(ev["plan_comm"][i])
                    runner.prefetch(ctx.weights[0], group=prefetch_group)
                    if l01:
                        # both projections in ONE phase, one join (upstream
                        # api.py:158-173: per-matrix launches, single sync)
                        runner.prefetch2(w2_local, group=prefetch_group)
                    ev["pref_end"][i].record()
            runner.pack(ctx.inputs_shard, route_weights)
            ev["pack"][i].record()
            runner.a2av()
            ev["comm"][i].record()
            if overlap_prefetch and do_prefetch and shared_comm:
                # AUTHENTIC serialization (MoonEP api.py:487-499): dispatch
                # and prefetch share ONE comm path, so prefetch is enqueued
                # after the a2av on the SAME communicator (every rank
                # enqueues a2av then prefetch — consistent order); overlap
                # window = scatter only
                with torch.cuda.stream(prefetch_stream):
                    prefetch_stream.wait_event(ev["comm"][i])
                    runner.prefetch(ctx.weights[0])
                    if l01:
                        runner.prefetch2(w2_local)
                    ev["pref_end"][i].record()
            runner.place_and_epilogue()
            ev["scatter"][i].record()
            if overlap_prefetch:
                if do_prefetch:
                    torch.cuda.current_stream().wait_event(ev["pref_end"][i])
                ev["join"][i].record()
            else:
                if do_prefetch:
                    runner.prefetch(ctx.weights[0])
                    if l01:
                        runner.prefetch2(w2_local)
                ev["prefetch"][i].record()
            runner.gemm(gemm_only_op, ctx.weights[0])
            ev["gemm"][i].record()
            if l01:
                runner.act()
                ev["act"][i].record()
                runner.gemm2(gemm_only_op, w2_local)
                ev["gemm2"][i].record()
                runner.combine_pack()
                ev["cpack"][i].record()
                runner.combine_a2av()
                ev["comb"][i].record()
                runner.combine_place_reduce()
                ev["acc"][i].record()

    keys = ["plan_comm_ms", "plan_ms", "pack_ms", "comm_ms", "scatter_ms",
            "prefetch_ms", "gemm_ms", "total_ms"]
    if l01:
        keys += ["act_ms", "gemm2_ms", "cpack_ms", "comb_ms", "acc_ms"]
    if overlap_prefetch:
        keys.append("prefetch_wait_ms")
    times = {k: [] for k in keys}
    last = "acc" if l01 else "gemm"
    for i in range(total_iters):
        ev[last][i].synchronize()
        if i < warmup_iters:
            continue
        seq = ["plan_comm", "plan", "pack", "comm", "scatter"]
        prev = ev["start"][i]
        for name in seq:
            times[f"{name}_ms"].append(prev.elapsed_time(ev[name][i]))
            prev = ev[name][i]
        if overlap_prefetch:
            # prefetch stream duration (fork -> done) and the exposed stall
            # the main stream paid waiting for it (scatter -> join); the
            # fork event is comm in shared_comm mode (authentic upstream
            # serialization enqueues prefetch after dispatch)
            fork_ev = ev["comm"][i] if shared_comm else ev["plan_comm"][i]
            times["prefetch_ms"].append(
                fork_ev.elapsed_time(ev["pref_end"][i])
                if do_prefetch else 0.0
            )
            times["prefetch_wait_ms"].append(
                ev["scatter"][i].elapsed_time(ev["join"][i])
            )
            times["gemm_ms"].append(ev["join"][i].elapsed_time(ev["gemm"][i]))
        else:
            times["prefetch_ms"].append(
                ev["scatter"][i].elapsed_time(ev["prefetch"][i])
            )
            times["gemm_ms"].append(
                ev["prefetch"][i].elapsed_time(ev["gemm"][i])
            )
        if l01:
            prev = ev["gemm"][i]
            for name in ("act", "gemm2", "cpack", "comb", "acc"):
                times[f"{name}_ms"].append(prev.elapsed_time(ev[name][i]))
                prev = ev[name][i]
        times["total_ms"].append(ev["start"][i].elapsed_time(ev[last][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return times


def check_correctness(runner, ctx, plan, w_all, gemm_only_op, atol, rtol,
                      do_prefetch=True):
    """V2 content + V3 numeric checks (see plan fidelity contract)."""
    cfg = runner.cfg
    rank, R, S, K, NvS = runner.rank, cfg.R, cfg.S, cfg.K, cfg.NvS
    dev = runner.device
    ok_bitwise = True

    # Full inputs via the harness's own allgather (untimed reference path).
    torch.distributed.all_gather_into_tensor(
        ctx.inputs, ctx.inputs_shard, group=TP_GROUP
    )

    # Expected hidden/weights buffers from the replicated plan: every entry
    # (dedup'd included) maps its raw slot to its token's row / its weight.
    enc = plan.dst.long()
    raw = torch.where(enc < 0, -enc - 1, enc)
    dest = torch.div(raw, NvS, rounding_mode="floor")
    loff = raw % NvS
    expected_hidden = torch.zeros_like(runner.hidden_buf)
    expected_weights = torch.zeros_like(runner.weights_buf)
    covered = torch.zeros(NvS, dtype=torch.bool, device=dev)
    n_rows_check = 0
    for src in range(R):
        m = dest[src] == rank
        slots = loff[src][m].to(dev)
        entries = m.nonzero(as_tuple=True)[0]
        tokens = (src * S + torch.div(entries, K, rounding_mode="floor")).to(dev)
        expected_hidden[slots] = ctx.inputs[tokens]
        expected_weights[slots] = w_all[src].reshape(-1)[entries].to(dev)
        covered[slots] = True
        n_rows_check += int(m.sum())

    # S2 balance fingerprint: every rank holds exactly S*K real rows.
    assert n_rows_check == S * K, f"rank {rank} holds {n_rows_check} != S*K rows"

    if not torch.equal(runner.hidden_buf[covered], expected_hidden[covered]):
        ok_bitwise = False
        print(f"❌ rank {rank}: dispatched rows differ from plan prediction")
        if payload_probe_enabled():
            _bad = (runner.hidden_buf[covered]
                    != expected_hidden[covered]).any(dim=1)
            print(f"  probe rank {rank}: bad {int(_bad.sum())}/{int(covered.sum())} "
                  "rows (payload randomized per iteration)", flush=True)
    if not torch.equal(runner.weights_buf[covered], expected_weights[covered]):
        ok_bitwise = False
        print(f"❌ rank {rank}: route weights differ from plan prediction")

    # S9: pad rows exactly zero (only the plan's zero_fill ranges).
    if runner.zero_rows.numel():
        if not bool((runner.hidden_buf[runner.zero_rows] == 0).all()):
            ok_bitwise = False
            print(f"❌ rank {rank}: zero-fill pad rows are not zero")

    # V3a: prefetched weights match an independent broadcast copy (NCCL
    # broadcast — a different code path than the batched P2P prefetch).
    if do_prefetch:
        tmp = torch.empty_like(runner.prefetch_w[0])
        for d, b, e, home in runner.prefetch_pairs:
            if home == rank:
                tmp.copy_(ctx.weights[0][e % cfg.epn])
            else:
                tmp.zero_()
            torch.distributed.broadcast(tmp, src=home, group=TP_GROUP)
            if d == rank and not torch.equal(runner.prefetch_w[b], tmp):
                ok_bitwise = False
                print(f"❌ rank {rank}: prefetch slot {b} (expert {e}) mismatch")

    # V3b: per-segment GEMM vs torch.matmul.
    ok_allclose = True
    lo = cfg.epn * rank
    for g, start, end, expert_id in runner.lay.gemm_segments:
        w = (
            ctx.weights[0][expert_id - lo]
            if g < cfg.E
            else runner.prefetch_w[g - cfg.E]
        )
        ref = torch.matmul(
            runner.hidden_buf[start:end].float(), w.float().t()
        ).to(runner.out_buf.dtype)
        try:
            flux.torch_allclose(runner.out_buf[start:end], ref, atol=atol, rtol=rtol)
        except Exception:
            ok_allclose = False
            print(f"❌ rank {rank}: gemm segment g={g} expert={expert_id} mismatch")

    status = "✅" if (ok_bitwise and ok_allclose) else "❌"
    print(f"{status} rank {rank}: moonep dispatch content "
          f"{'bitwise-exact' if ok_bitwise else 'MISMATCH'}, "
          f"gemm {'allclose' if ok_allclose else 'MISMATCH'}")
    RECORDER.emit_correctness(bitwise=ok_bitwise, allclose=ok_allclose)
    # collective verdict (wire-ordering audit hygiene): a per-rank assert
    # wedges the surviving ranks in the next collective and holds the srun
    # step; fail together instead.
    _flag = torch.tensor([int(ok_bitwise and ok_allclose)], device="cuda")
    torch.distributed.all_reduce(_flag, op=torch.distributed.ReduceOp.MIN,
                                 group=TP_GROUP)
    assert int(_flag) == 1, "correctness failed on at least one rank"
    return ok_bitwise and ok_allclose


def check_correctness_l01(runner, ctx, w_all, full_w2, topk_shard, atol, rtol):
    """Independent two-layer reference for the staged l01 journey: for each
    of MY tokens, sum over its top-k entries of
    route_w * gelu(x @ w1_e^T) @ w2_e^T — built from the RAW routing and
    inputs, never from the runner's buffers. w1 arrives per expert via NCCL
    broadcast (a different code path than any prefetch transport); w2 is
    replicated by construction. Rounding points mirror the pipeline (bf16
    casts after each GEMM) so the standard thresholds hold; the reference
    accumulates in fp32 while the pipeline accumulates in bf16 — absorbed by
    the tolerances at these magnitudes."""
    cfg = runner.cfg
    rank, S, H = runner.rank, cfg.S, cfg.H
    dev = runner.device
    x = ctx.inputs_shard  # [S, H] my tokens
    ref = torch.zeros(S, H, dtype=torch.float32, device=dev)
    tmp_w1 = torch.empty(runner.ffn_size_shard, H, dtype=x.dtype, device=dev)
    home_w1 = runner.weight_home if runner.weight_home is not None else ctx.weights[0]
    for e in range(cfg.E):
        home = e // cfg.epn
        if home == rank:
            tmp_w1.copy_(home_w1[e % cfg.epn])
        torch.distributed.broadcast(tmp_w1, src=home, group=TP_GROUP)
        mask = topk_shard == e  # [S, K]
        if not bool(mask.any()):
            continue
        s_idx, k_idx = mask.nonzero(as_tuple=True)
        h1 = torch.matmul(x[s_idx].float(), tmp_w1.float().t()).to(x.dtype)
        a = torch.nn.functional.gelu(h1)
        w2e = full_w2[e].to(dev)
        h2 = torch.matmul(a.float(), w2e.float().t()).to(x.dtype)
        ref[s_idx] += h2.float() * w_all[rank][s_idx.cpu(), k_idx.cpu()].to(dev).unsqueeze(1)
    ok = True
    try:
        flux.torch_allclose(runner.final_out, ref.to(runner.final_out.dtype),
                            atol=atol, rtol=rtol)
    except Exception:  # noqa: BLE001
        ok = False
        print(f"❌ rank {rank}: l01 combined output vs two-layer reference"
              " MISMATCH")
        RECORDER.emit_correctness(bitwise=False, allclose=False)
        RECORDER.flush()
    else:
        print(f"✅ rank {rank}: l01 combined output matches the two-layer"
              " reference (allclose)")
    # collective verdict (wire-ordering audit hygiene): a per-rank assert
    # wedges the surviving ranks in the next collective and holds the srun
    # step; fail together instead.
    _flag = torch.tensor([int(ok)], device="cuda")
    torch.distributed.all_reduce(_flag, op=torch.distributed.ReduceOp.MIN,
                                 group=TP_GROUP)
    assert int(_flag) == 1, "correctness failed on at least one rank"
    return ok


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True)
    parser.add_argument("--routing_file", type=str, default=None,
                        help="per-token expert-id sidecar (trace family)")
    parser.add_argument("--chunk_bytes", type=int, default=8192)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--G", type=int, default=32, help="number of experts")
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup_iters", default=10, type=int)
    parser.add_argument("--sm_margin", default=0, type=int,
                        help="accepted for sweep-runner CLI parity; unused")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--token_padding", type=int, default=128,
                        help="MoonEP segment padding (segments pad to this)")
    parser.add_argument("--transport", default="nvshmem",
                        choices=["nccl", "nvshmem"],
                        help="dispatch a2av transport. Default nvshmem: flux's"
                        " one-sided NVSHMEM All2AllSingle (M4a — the authentic"
                        " port of MoonEP's one-sided writes; the port's goal is"
                        " semantic fidelity, not speed). nccl: two-sided grouped"
                        " P2P — faster on Slingshot but least like upstream;"
                        " the historical sweep arms pin it explicitly")
    parser.add_argument("--num_comm_sm", type=int, default=8,
                        help="SMs for the NVSHMEM a2av kernel and the getmem"
                        " prefetch kernel (nvshmem/getmem paths only)")
    parser.add_argument("--prefetch_transport", default="getmem",
                        choices=["nccl", "getmem"],
                        help="weight-movement transport. Default getmem"
                        " (faithful-baseline flip 2026-08-12): one-sided"
                        " destination-initiated pull from the symmetric"
                        " weight home (flux.WeightPrefetchGetmem — the"
                        " authentic analog of MoonEP's prefetch; source"
                        " ranks passive, zero signaling, no communicator)."
                        " nccl: two-sided batch_isend_irecv (declared port"
                        " artifact — wrong initiator direction AND protocol"
                        " class, NR-12 fact 6); the historical sweep arms"
                        " pin it explicitly")
    parser.add_argument("--prefetch_chunk_bytes", type=int, default=4 << 20,
                        help="getmem pull chunk size per block (getmem only;"
                        " tuned by the a2av_comm_bench prefetch mode)")
    parser.add_argument("--prefetch_impl", default="kernel",
                        choices=["kernel", "stream"],
                        help="getmem issue path: SM device kernel (authentic"
                        " — upstream burns SMs via TMA) or host"
                        " getmem_nbi_on_stream (A/B fallback)")
    parser.add_argument("--overlap_prefetch", default=False, action="store_true",
                        help="run weight prefetch on a dedicated high-priority"
                        " stream + separate NCCL communicator, event-joined"
                        " before GEMM (M4c — MoonEP async_finish semantics;"
                        " serialized prefetch is the deliberately-pessimistic"
                        " default). NOTE (NR-12 fact 7): this is FINER than"
                        " upstream, which serializes dispatch+prefetch on one"
                        " shared comm stream — see --shared_comm_stream")
    parser.add_argument("--shared_comm_stream", default=False,
                        action="store_true",
                        help="with --overlap_prefetch: reproduce upstream's"
                        " AUTHENTIC single-comm-path serialization (MoonEP"
                        " api.py:487-499) — prefetch enqueued after the"
                        " dispatch a2av on the SAME communicator; overlap"
                        " window = scatter only")
    parser.add_argument("--no_prefetch", default=False, action="store_true",
                        help="skip per-iteration redundant-weight prefetch")
    parser.add_argument("--layers", default="l0", choices=["l0", "l01"],
                        help="l01 runs the full staged journey: dispatch +"
                        " gemm1 + gelu + gemm2 + combine (gemm2 weights"
                        " prefetched IN THE SAME phase as gemm1's — upstream"
                        " moves every projection in one pass under one sync,"
                        " api.py:158-173; combine = the dispatch mirror:"
                        " scale, reverse-dedup partial sums at the expert"
                        " side, direct a2av transpose, index_add at home)")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    if args.transport == "nvshmem" or args.prefetch_transport == "getmem":
        # the one-sided All2AllSingle / getmem weight home need the flux
        # shm / NVSHMEM heap
        flux.init_flux_shm(TP_GROUP)
    # (all-nccl configuration needs no NVSHMEM symmetric heap or flux shm)
    torch.cuda.synchronize()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.H * input_dtype.itemsize == args.chunk_bytes
    W = DIST_ENV.WORLD_SIZE
    assert args.G % W == 0, f"{args.G} % {W} != 0"

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W
    if args.routing_file:
        choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
        assert choosed_experts.shape[0] % W == 0
        got = choosed_experts_to_matrix_chunks(choosed_experts, W, args.G // W)
        assert torch.equal(got * args.chunk_bytes, matrix)
        choosed_experts = choosed_experts.cuda()
        if TP_GROUP.rank() == 0:
            print(f"routing: REAL trace file {args.routing_file}")
    else:
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
        if TP_GROUP.rank() == 0:
            print("routing: synthetic dealer assignment (max-dedup construction;"
                  " prefer trace-family cells for headline numbers)")
    ntokens = choosed_experts.shape[0]
    S = ntokens // W
    rank = TP_GROUP.rank()

    cfg = MoonEPConfig(
        S=S, K=args.topk, E=args.G, R=W, H=args.H,
        token_padding=args.token_padding,
    )

    # Setup reference plan (CPU): buffer sizing, correctness reference,
    # plan-hash guard, and the drift check against the ported kernel. The
    # TIMED planner is the per-iteration plan_iteration() (rule 5); wall
    # time here is book-kept as plan_host_ms only.
    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    t0 = time.perf_counter()
    plan = compute_moonep_plan(cfg, topk_all)
    plan_host_ms = (time.perf_counter() - t0) * 1e3

    # Cross-rank determinism guard: identical plan hash everywhere.
    h = torch.tensor([plan.plan_hash()], dtype=torch.int64, device="cuda")
    h_all = torch.zeros(W, dtype=torch.int64, device="cuda")
    torch.distributed.all_gather_into_tensor(h_all, h, group=TP_GROUP)
    assert bool((h_all == h_all[0]).all()), "plan hash differs across ranks"

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

    runner = MoonEPLayer0Runner(
        plan, rank, TP_GROUP, torch.cuda.current_device(),
        dtype=input_dtype, ffn_size_shard=moe_ctx.ffn_size_shard,
    )
    if args.transport == "nvshmem":
        runner.enable_nvshmem(DIST_ENV.LOCAL_WORLD_SIZE, args.num_comm_sm)
    if args.prefetch_transport == "getmem":
        # collective (symmetric weight-home alloc); runs after enable_nvshmem
        # so all ranks perform the same symmetric allocations in the same
        # order. Copies this rank's weights onto the heap once (one-shot
        # deployment scope, rule-5 boundary).
        runner.enable_getmem_prefetch(
            moe_ctx.weights[0],
            num_comm_sm=args.num_comm_sm,
            chunk_bytes=args.prefetch_chunk_bytes,
            device_kernel=args.prefetch_impl == "kernel",
        )

    w2_local = None
    full_w2 = None
    if args.layers == "l01":
        assert not args.no_prefetch, (
            "--layers l01 needs the prefetch (slot experts' gemm2 weights"
            " arrive with it)"
        )
        # replicated full down-projection set (deterministic seed): each
        # rank slices its home shard; the reference check reads any expert's
        # w2 without communication. Magnitudes match the l1 bench scheme.
        gen2 = torch.Generator().manual_seed(1234)
        full_w2 = (
            torch.rand((args.G, args.H, moe_ctx.ffn_size_shard), generator=gen2)
            * 0.02 - 0.01
        ).to(input_dtype)
        w2_local = (
            full_w2[rank * cfg.epn:(rank + 1) * cfg.epn].cuda().contiguous()
        )
        runner.enable_layer1()
        if args.prefetch_transport == "getmem":
            # collective (second symmetric weight home); same ordering rules
            # as enable_getmem_prefetch above
            runner.enable_getmem_prefetch_w2(
                w2_local,
                num_comm_sm=args.num_comm_sm,
                chunk_bytes=args.prefetch_chunk_bytes,
                device_kernel=args.prefetch_impl == "kernel",
            )

    # Ported authentic planner (rule 5): JIT-compile at SETUP (legal
    # toolchain cost — minutes-scale on first use, lru-cached after) so
    # warmup iterations only pay kernel launches; then ONE untimed drift
    # check — the kernel outputs must be bitwise-equal to the CPU
    # reference plan, and the derived layout must match runner.lay.
    if rank == 0:
        print("planner port: compiling the replicated MoonEP planning "
              "kernel (CuTe DSL JIT; first-ever compile can take a while)"
              " ...", flush=True)
    plan_ws = ReplicatedPlannerWorkspace(
        cfg, torch.device(torch.cuda.current_device()))
    setup_lay = runner.lay          # snapshot BEFORE the bind swaps it
    topk_gather_buf_setup = topk_all.reshape(W * S, args.topk).to(
        torch.int32).cuda()
    ip0 = plan_iteration(runner, plan_ws, topk_gather_buf_setup)
    torch.cuda.synchronize()
    assert torch.equal(plan_ws.dst_all.cpu(), plan.dst.to(torch.int32)), (
        "ported planner dst != CPU reference plan (see planning_port.py "
        "fallback ladder)")
    check_moonep_iter_plan(ip0, setup_lay, plan, rank)
    del topk_gather_buf_setup

    prefetch_group = None
    prefetch_stream = None
    if args.overlap_prefetch:
        prefetch_stream = torch.cuda.Stream(priority=-1)
        if not args.shared_comm_stream and args.prefetch_transport == "nccl":
            # separate NCCL communicator: prefetch P2Ps must not serialize on
            # the dispatch collectives' communicator; every rank enqueues
            # prefetch first (right after plan_comm), so cross-communicator
            # order is consistent and deadlock-free
            prefetch_group = torch.distributed.new_group(
                ranks=list(range(W)), backend="nccl"
            )
            # eager-init the communicator (untimed): NCCL init is collective
            # and lazy — a rank with zero prefetch pairs would never call
            # into prefetch_group and hang the others in init (this bit the
            # ultraep ws_group first; moonep plans have so far always had
            # every rank participating, but nothing guarantees it)
            torch.distributed.barrier(group=prefetch_group)
        # shared_comm_stream: no second communicator — authentic to MoonEP's
        # single comm path; safe because every rank enqueues a2av before
        # prefetch on the shared communicator. getmem prefetch also needs no
        # communicator in ANY mode (one-sided, no rendezvous — the second
        # NCCL communicator was itself port machinery, NR-12 fact 3), so
        # prefetch_group stays None there too.

    # Per-entry route weights, replicated deterministically so receivers can
    # verify without an extra exchange (values still travel the wire).
    gen = torch.Generator().manual_seed(777)
    w_all = torch.rand(W, S, args.topk, dtype=torch.float32, generator=gen)
    route_weights = w_all[rank].cuda()

    topk_shard = topk_all[rank].cuda().contiguous()
    topk_gather_buf = torch.zeros(W * S, args.topk, dtype=torch.int32, device="cuda")

    gemm_only_op = flux.GemmOnly(
        moe_ctx.inputs.dtype,
        moe_ctx.inputs.dtype,
        moe_ctx.outputs[0].dtype,
        use_fp8_gemm=False,
    )

    if rank == 0:
        gemm_rows = plan.cu_seqlens[:, -1].tolist()
        wire_rows = [
            [int(((plan.dst[s] >= 0)
                  & (torch.div(plan.dst[s].long(), cfg.NvS,
                               rounding_mode="floor") == d)).sum())
             for d in range(W)]
            for s in range(W)
        ]
        wire_bytes = [[r * args.chunk_bytes for r in row] for row in wire_rows]
        logical_send = (matrix.sum(dim=1) - matrix.diag()).tolist()
        realized_send = [
            sum(row) - row[i] for i, row in enumerate(wire_bytes)
        ]
        print(f"ntokens: {ntokens} ({S} per rank), topk: {args.topk}, "
              f"E: {args.G}, NvS: {cfg.NvS}, token_padding: {args.token_padding}")
        print(f"moonep gemm rows per rank (padded): {gemm_rows}  <- constant "
              f"S*K+pad is the balance fingerprint")
        print(f"moonep z (home-group -> dest migrations):\n{plan.z}")
        print(f"logical send bytes per rank:  {logical_send}")
        print(f"realized wire send bytes per rank (dedup'd reps): {realized_send}")
        print(f"plan_host_ms (setup reference build; the timed planner is "
              f"the per-iteration ported kernel, plan_ms — SCHEMA rule 5): "
              f"{plan_host_ms:.1f}")
        RECORDER.emit_info(
            timing_accounting="per_iter_gpu",
            moonep_planner_impl="cute_port",
            ntokens=ntokens,
            tokens_per_rank=S,
            gemm_rows_per_rank=gemm_rows,
            moonep_nvs=cfg.NvS,
            moonep_token_padding=args.token_padding,
            moonep_plan_host_ms=plan_host_ms,
            moonep_z_matrix=plan.z.tolist(),
            moonep_wire_bytes=wire_bytes,
            moonep_prefetch=not args.no_prefetch,
            moonep_transport=args.transport,
            moonep_prefetch_transport=args.prefetch_transport,
            moonep_prefetch_chunk_bytes=args.prefetch_chunk_bytes,
            moonep_prefetch_impl=args.prefetch_impl,
            moonep_overlap_prefetch=bool(args.overlap_prefetch),
            moonep_shared_comm_stream=bool(args.shared_comm_stream),
        )
    RECORDER.emit_info(moonep_prefetch_recv_bytes=runner.prefetch_recv_bytes())
    if args.layers == "l01":
        # w2 shares the pair list, so the same byte count again (2-of-3
        # matrices vs upstream's 3 — see the walkthrough deviation note)
        RECORDER.emit_info(
            moonep_layers=args.layers,
            moonep_prefetch_recv_bytes_w2=runner.prefetch_recv_bytes(),
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_moonep_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        iter_times = perf_moonep(
            runner, plan_ws, moe_ctx, route_weights, topk_shard,
            topk_gather_buf, gemm_only_op, args.warmup_iters, args.iters,
            do_prefetch=not args.no_prefetch,
            overlap_prefetch=args.overlap_prefetch,
            shared_comm=args.shared_comm_stream,
            prefetch_group=prefetch_group,
            prefetch_stream=prefetch_stream,
            layers=args.layers,
            w2_local=w2_local,
        )

    def fmt(times):
        return ", ".join(
            f"{k[:-3]} {sum(v) / max(len(v), 1):.3f} ms"
            for k, v in times.items() if k != "iso_sync_ms"
        )

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"moonep #{rank}: {fmt(iter_times)}")
    )
    RECORDER.emit_iters("moonep", iter_times)

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-2, 1.5e-2

    if not args.skip_correctness:
        # NOT under exec_in_rank_order: the check runs collectives (inputs
        # allgather, prefetch-verification broadcasts) on every rank.
        check_correctness(
            runner, moe_ctx, plan, w_all, gemm_only_op, atol, rtol,
            do_prefetch=not args.no_prefetch,
        )
        if args.layers == "l01":
            check_correctness_l01(
                runner, moe_ctx, w_all, full_w2, topk_shard, atol, rtol
            )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
