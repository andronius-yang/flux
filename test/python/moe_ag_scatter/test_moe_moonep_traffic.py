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

The plan itself is deterministic integer math computed identically on every
rank at setup (untimed-metadata contract, like splits_per_source in
test_moe_ag_traffic.py; reported as plan_host_ms): routing is static per
cell, so per-iteration planning would time redundant host work, not the
algorithm. plan_comm is still measured every iteration because it is the
recurring per-layer wire cost of replicated planning.
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
    gen_moe_gating_args,
    choosed_experts_to_matrix_chunks,
    load_routing_file,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.moonep_semantics import (
    MoonEPConfig,
    MoonEPLayer0Runner,
    compute_moonep_plan,
)
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


@torch.no_grad()
def perf_moonep(
    runner: MoonEPLayer0Runner,
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
):
    total_iters = warmup_iters + iters
    names = ["start", "plan_comm", "pack", "comm", "scatter", "prefetch", "gemm"]
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
    for i in range(total_iters):
        if isolated:
            # sweeps SCHEMA.md isolated mode; also the lawful stand-in for
            # MoonEP's inter_rank_sync entry barrier.
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            ev["start"][i].record()
            torch.distributed.all_gather_into_tensor(
                topk_gather_buf, topk_shard, group=TP_GROUP
            )
            ev["plan_comm"][i].record()
            if overlap_prefetch and do_prefetch and not shared_comm:
                # fork right after plan_comm: weight movement overlaps
                # pack + wire + scatter on the main stream (FINER than
                # upstream, which shares one comm stream — NR-12 fact 7)
                with torch.cuda.stream(prefetch_stream):
                    prefetch_stream.wait_event(ev["plan_comm"][i])
                    runner.prefetch(ctx.weights[0], group=prefetch_group)
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
                ev["prefetch"][i].record()
            runner.gemm(gemm_only_op, ctx.weights[0])
            ev["gemm"][i].record()

    keys = ["plan_comm_ms", "pack_ms", "comm_ms", "scatter_ms",
            "prefetch_ms", "gemm_ms", "total_ms"]
    if overlap_prefetch:
        keys.append("prefetch_wait_ms")
    times = {k: [] for k in keys}
    for i in range(total_iters):
        ev["gemm"][i].synchronize()
        if i < warmup_iters:
            continue
        seq = ["plan_comm", "pack", "comm", "scatter"]
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
        times["total_ms"].append(ev["start"][i].elapsed_time(ev["gemm"][i]))
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
    assert ok_bitwise and ok_allclose
    return ok_bitwise and ok_allclose


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
                        help="SMs for the NVSHMEM a2av kernel (nvshmem only)")
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
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    if args.transport == "nvshmem":
        # the one-sided All2AllSingle needs the flux shm / NVSHMEM heap
        flux.init_flux_shm(TP_GROUP)
    # (nccl transport needs no NVSHMEM symmetric heap or flux shm groups)
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

    # Replicated planning: identical integer plan on every rank from the
    # (globally known) routing. Wall time reported as plan_host_ms under the
    # untimed-metadata contract.
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

    prefetch_group = None
    prefetch_stream = None
    if args.overlap_prefetch:
        prefetch_stream = torch.cuda.Stream(priority=-1)
        if not args.shared_comm_stream:
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
        # prefetch on the shared communicator

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
        print(f"plan_host_ms (untimed-metadata contract): {plan_host_ms:.1f}")
        RECORDER.emit_info(
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
            moonep_overlap_prefetch=bool(args.overlap_prefetch),
            moonep_shared_comm_stream=bool(args.shared_comm_stream),
        )
    RECORDER.emit_info(moonep_prefetch_recv_bytes=runner.prefetch_recv_bytes())

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_moonep_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        iter_times = perf_moonep(
            runner, moe_ctx, route_weights, topk_shard, topk_gather_buf,
            gemm_only_op, args.warmup_iters, args.iters,
            do_prefetch=not args.no_prefetch,
            overlap_prefetch=args.overlap_prefetch,
            shared_comm=args.shared_comm_stream,
            prefetch_group=prefetch_group,
            prefetch_stream=prefetch_stream,
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

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
