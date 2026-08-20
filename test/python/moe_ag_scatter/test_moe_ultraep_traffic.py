################################################################################
#
# UltraEP-semantics layer0 arm for the sweep harness.
# Structural clone of test_moe_moonep_traffic.py; algorithm semantics ported
# from Dots-Infra/UltraEP (see python/flux/testing/ultraep_semantics.py,
# bit-equality-tested against the real kernels by test_ultraep_planner.py).
#
################################################################################
"""COMET-style layer0 benchmark under UltraEP's replicated-expert balancing.

Pipeline per iteration (each phase bracketed with CUDA events):
  plan_comm  all_gather of the [W, G] int32 per-rank expert loads (UltraEP's
             metadata fcollect before planning — loads only, ~W*G*4 bytes,
             NOT MoonEP's [S, K] topk allgather; the byte asymmetry is
             recorded as ultraep_plan_comm_bytes)
  pack       rerouted-row gather into the (physical expert, token)-sorted
             send buffer (port-added local copy, kept out of comm_ms)
  comm       all_to_all_single of token rows + per-entry fp32 route probs
             (NO dedup: one wire row per (token, physical expert) entry,
             faithful to UltraEP/Megatron dispatch; ultraep_dup_rows counts
             what MoonEP-style dedup would have saved)
  scatter    placement into per-physical-expert segments
  prefetch   UltraEP weight_sync, `direct` plan: master expert weights ->
             replica slots, all pairs intra-NVLink-domain by construction
             (at NVL domain <= 8 direct IS UltraEP's algorithm; adaptive
             relay needs >= 4 replicas of one expert — impossible with 3
             peers). fc1 feeds the GEMM; fc2 rides along (bitwise-verified)
             so sync bytes match UltraEP's full-expert sync. Serialized on
             the main stream by default (deliberately pessimistic);
             --overlap_ws restores upstream's async weight_sync on a
             dedicated high-priority comm stream (ultra_ep.cpp:253,
             async_finish=True), with the join point exposed via --ws_join:
             `dispatch` (default) joins before the token a2a — the
             AUTHENTIC placement, and upstream's publication mechanism (its
             direct mode has no receiver-side signaling; the dispatch
             collective is the cross-rank happens-before, see NR-12);
             `gemm` joins before the GEMM — a labeled COUNTERFACTUAL, sound
             in this NCCL port only because two-sided irecv completion
             rides the ws-stream event; never quote it as UltraEP behavior.
  gemm       per-segment GEMM over the received rows, NO padding: the
             residual physical imbalance IS the measurement
             (gemm_rows_per_rank is per-rank, unlike moonep's constant S*K)

The plan is deterministic integer math computed identically on every rank at
setup (pre-rule-5 legacy_untimed_plan accounting; see SCHEMA protocol rule 5, reported as ultraep_plan_host_ms): routing
is static per cell, so per-iteration planning would time redundant host
work. plan_comm is still measured every iteration as the recurring per-layer
wire cost of replicated planning.

Semantics caveat (by design, reported not fixed): one independent solve per
NVLink domain means cross-node imbalance is untouched; the reachable floor
is ultraep_lb_floor = max over domains of domain-mean/global-mean rank load.
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
from flux.testing.ultraep_semantics import (
    UltraEPConfig,
    UltraEPLayer0Runner,
    loads_from_topk,
    nvl_domain_lower_bound,
    remote_token_fraction,
    solve_placement,
    wire_matrix,
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


def assert_node_major_ranks():
    """UltraEP topology math assumes node-major contiguous ranks (rank k on
    node k // local_world). torchrun w/ --nproc_per_node gives this; assert
    it instead of silently producing a wrong domain split."""
    import socket

    host = socket.gethostname()
    hosts = [None] * DIST_ENV.WORLD_SIZE
    torch.distributed.all_gather_object(hosts, host, group=TP_GROUP)
    lw = DIST_ENV.LOCAL_WORLD_SIZE
    for node in range(DIST_ENV.WORLD_SIZE // lw):
        block = hosts[node * lw:(node + 1) * lw]
        assert len(set(block)) == 1, f"ranks not node-major: {hosts}"


@torch.no_grad()
def perf_ultraep(
    runner: UltraEPLayer0Runner,
    ctx: MoeMlp1Ctx,
    probs_shard: torch.Tensor,
    loads_shard: torch.Tensor,
    loads_gather_buf: torch.Tensor,
    fc2_home,
    gemm_only_op,
    warmup_iters: int,
    iters: int,
    do_weight_sync: bool = True,
    overlap_ws: bool = False,
    ws_join: str = "dispatch",
    ws_group=None,
    ws_stream=None,
):
    total_iters = warmup_iters + iters
    names = ["start", "plan_comm", "pack", "comm", "scatter", "prefetch", "gemm"]
    if overlap_ws:
        # weight_sync rides its own high-priority stream + its own NCCL
        # communicator (UltraEP async_finish semantics: dedicated comm
        # stream, event-joined by the consumer); "join" marks the main
        # stream having absorbed the ws dependency at the --ws_join point.
        names += ["ws_end", "join"]
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
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            ev["start"][i].record()
            # UltraEP's pre-plan metadata fcollect: per-rank loads only.
            torch.distributed.all_gather_into_tensor(
                loads_gather_buf, loads_shard, group=TP_GROUP
            )
            ev["plan_comm"][i].record()
            if overlap_ws and do_weight_sync:
                # fork right after plan_comm (upstream launches weight_sync
                # right after update_placement); EVERY rank enqueues ws
                # first so cross-communicator order is consistent
                with torch.cuda.stream(ws_stream):
                    ws_stream.wait_event(ev["plan_comm"][i])
                    runner.weight_sync(ctx.weights[0], fc2_home, group=ws_group)
                    ev["ws_end"][i].record()
            runner.pack(ctx.inputs_shard, probs_shard)
            ev["pack"][i].record()
            if overlap_ws and ws_join == "dispatch":
                # AUTHENTIC join point: upstream publishes its bare peer-VA
                # pushes through the dispatch collective (NR-12) — window =
                # pack only, weight_sync mostly exposed, by upstream design
                if do_weight_sync:
                    torch.cuda.current_stream().wait_event(ev["ws_end"][i])
                ev["join"][i].record()
            runner.a2av()
            ev["comm"][i].record()
            runner.place()
            ev["scatter"][i].record()
            if overlap_ws:
                if ws_join == "gemm":
                    # COUNTERFACTUAL join: legal under the data dependency
                    # (weights consumed only by the GEMM) and sound here
                    # because NCCL irecv completion rides ws_end
                    if do_weight_sync:
                        torch.cuda.current_stream().wait_event(ev["ws_end"][i])
                    ev["join"][i].record()
            else:
                if do_weight_sync:
                    runner.weight_sync(ctx.weights[0], fc2_home)
                ev["prefetch"][i].record()
            runner.gemm(gemm_only_op, ctx.weights[0])
            ev["gemm"][i].record()

    keys = ["plan_comm_ms", "pack_ms", "comm_ms", "scatter_ms",
            "prefetch_ms", "gemm_ms", "total_ms"]
    if overlap_ws:
        keys.append("prefetch_wait_ms")
    times = {k: [] for k in keys}
    for i in range(total_iters):
        ev["gemm"][i].synchronize()
        if i < warmup_iters:
            continue
        if overlap_ws:
            # prefetch_ms = ws stream duration (fork -> done), off-chain in
            # both join modes; prefetch_wait_ms = exposed stall the main
            # stream paid at the join. The phase AFTER the join measures
            # from the join event, so the six phase columns partition
            # [start, gemm] in every mode (SCHEMA overlap metric rule).
            times["plan_comm_ms"].append(
                ev["start"][i].elapsed_time(ev["plan_comm"][i])
            )
            times["pack_ms"].append(
                ev["plan_comm"][i].elapsed_time(ev["pack"][i])
            )
            times["prefetch_ms"].append(
                ev["plan_comm"][i].elapsed_time(ev["ws_end"][i])
                if do_weight_sync else 0.0
            )
            if ws_join == "dispatch":
                times["prefetch_wait_ms"].append(
                    ev["pack"][i].elapsed_time(ev["join"][i])
                )
                times["comm_ms"].append(
                    ev["join"][i].elapsed_time(ev["comm"][i])
                )
                times["scatter_ms"].append(
                    ev["comm"][i].elapsed_time(ev["scatter"][i])
                )
                times["gemm_ms"].append(
                    ev["scatter"][i].elapsed_time(ev["gemm"][i])
                )
            else:  # gemm join
                times["comm_ms"].append(
                    ev["pack"][i].elapsed_time(ev["comm"][i])
                )
                times["scatter_ms"].append(
                    ev["comm"][i].elapsed_time(ev["scatter"][i])
                )
                times["prefetch_wait_ms"].append(
                    ev["scatter"][i].elapsed_time(ev["join"][i])
                )
                times["gemm_ms"].append(
                    ev["join"][i].elapsed_time(ev["gemm"][i])
                )
        else:
            seq = ["plan_comm", "pack", "comm", "scatter", "prefetch", "gemm"]
            prev = ev["start"][i]
            for name in seq:
                times[f"{name}_ms"].append(prev.elapsed_time(ev[name][i]))
                prev = ev[name][i]
        times["total_ms"].append(ev["start"][i].elapsed_time(ev["gemm"][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return times


def check_correctness(runner, ctx, plan, topk_all, w_all, fc2_home,
                      gemm_only_op, atol, rtol, do_weight_sync=True):
    """Content bitwise + wire-quota audit + weight-sync broadcast check +
    per-segment GEMM allclose (collective: run on every rank)."""
    from flux.testing.ultraep_semantics import reroute_expand

    cfg = runner.cfg
    rank, R, S = runner.rank, cfg.R, cfg.S
    dev = runner.device
    ok_bitwise = True

    torch.distributed.all_gather_into_tensor(
        ctx.inputs, ctx.inputs_shard, group=TP_GROUP
    )

    # Independent recomputation of the receive layout: expand every source,
    # walk entries destined here in (src; phys, token) order, fill segments.
    expected_hidden = torch.zeros_like(runner.hidden_buf)
    expected_probs = torch.zeros_like(runner.weights_buf)
    seg_fill = list(runner.lay.seg_start)
    n_rows_check = 0
    per_instance_rows = [0] * cfg.nlp
    for src in range(R):
        tok, phys = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(phys * (S + 1) + tok, stable=True)
        tok, phys = tok[order], phys[order]
        m = (phys // cfg.nlp) == rank
        tok, phys = tok[m], phys[m]
        for t, p in zip(tok.tolist(), phys.tolist()):
            p_local = p - rank * cfg.nlp
            slot = seg_fill[p_local]
            seg_fill[p_local] += 1
            expected_hidden[slot] = ctx.inputs[src * S + t]
            logical = int(plan.p2l[p])
            expected_probs[slot] = float(w_all[src][t, logical])
            per_instance_rows[p_local] += 1
            n_rows_check += 1

    assert n_rows_check == runner.n_recv, (
        f"rank {rank}: recomputed {n_rows_check} rows != runner {runner.n_recv}"
    )

    # Wire-quota audit: per-instance received rows == sum over sources of the
    # rank-quota allocations to instances hosted here.
    alloc = plan.rank_quota_prefix.long().clone()
    alloc[:, :, 1:] -= plan.rank_quota_prefix.long()[:, :, :-1]
    for l in range(cfg.G):
        for j in range(int(plan.lcnts[l])):
            p = int(plan.l2p[l, j])
            if p // cfg.nlp == rank:
                expect = int(alloc[:, l, j].sum())
                got = per_instance_rows[p % cfg.nlp]
                if got != expect:
                    ok_bitwise = False
                    print(f"❌ rank {rank}: instance ({l},{j}) rows {got} != "
                          f"rank-quota total {expect}")

    if not torch.equal(runner.hidden_buf[:runner.n_recv],
                       expected_hidden[:runner.n_recv]):
        ok_bitwise = False
        print(f"❌ rank {rank}: dispatched rows differ from plan prediction")
    if not torch.equal(runner.weights_buf[:runner.n_recv],
                       expected_probs[:runner.n_recv]):
        ok_bitwise = False
        print(f"❌ rank {rank}: route probs differ from plan prediction")

    # Weight sync vs an independent NCCL broadcast (different code path than
    # the batched P2P sync).
    if do_weight_sync:
        tmp1 = torch.empty_like(runner.replica_fc1[0])
        tmp2 = torch.empty_like(runner.replica_fc2[0]) if runner.sync_fc2 else None
        for dest, b, l, home in runner.lay.weight_sync_pairs:
            e_local = l % cfg.epn
            tmp1.copy_(ctx.weights[0][e_local]) if home == rank else tmp1.zero_()
            torch.distributed.broadcast(tmp1, src=home, group=TP_GROUP)
            if dest == rank and not torch.equal(runner.replica_fc1[b], tmp1):
                ok_bitwise = False
                print(f"❌ rank {rank}: fc1 replica slot {b} (expert {l}) mismatch")
            if runner.sync_fc2:
                tmp2.copy_(fc2_home[e_local]) if home == rank else tmp2.zero_()
                torch.distributed.broadcast(tmp2, src=home, group=TP_GROUP)
                if dest == rank and not torch.equal(runner.replica_fc2[b], tmp2):
                    ok_bitwise = False
                    print(f"❌ rank {rank}: fc2 replica slot {b} (expert {l}) "
                          f"mismatch")

    # Per-segment GEMM vs torch.matmul.
    ok_allclose = True
    for p, start, end, logical in runner.lay.gemm_segments:
        w = (
            ctx.weights[0][logical % cfg.epn] if p < cfg.epn
            else runner.replica_fc1[p - cfg.epn]
        )
        ref = torch.matmul(
            runner.hidden_buf[start:end].float(), w.float().t()
        ).to(runner.out_buf.dtype)
        try:
            flux.torch_allclose(runner.out_buf[start:end], ref, atol=atol, rtol=rtol)
        except Exception:
            ok_allclose = False
            print(f"❌ rank {rank}: gemm segment slot={p} expert={logical} mismatch")

    status = "✅" if (ok_bitwise and ok_allclose) else "❌"
    print(f"{status} rank {rank}: ultraep dispatch content "
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
    # --- UltraEP algorithm knobs (defaults = UltraEP runtime defaults) ---
    parser.add_argument("--nvl_domain_size", type=int, default=0,
                        help="NVLink domain size D; 0 = LOCAL_WORLD_SIZE "
                        "(faithful). Set 16 for the rack-scale-single-node "
                        "counterfactual (weight sync then crosses nodes).")
    parser.add_argument("--redundant_per_rank", type=int, default=2,
                        help="redundant expert slots per rank (UltraEP R)")
    parser.add_argument("--min_tokens_per_replica", type=int, default=1024,
                        help="ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA")
    parser.add_argument("--balance_threshold", type=float, default=1.0)
    parser.add_argument("--oracle_eps", type=float, default=0.01)
    parser.add_argument("--no_locality", default=False, action="store_true",
                        help="disable locality-aware quota decomposition")
    parser.add_argument("--no_interleave", default=False, action="store_true",
                        help="disable coprime-stride reroute interleaving")
    parser.add_argument("--allow_zero_master_quota", default=False,
                        action="store_true")
    parser.add_argument("--weight_sync", default="fc1fc2",
                        choices=["fc1fc2", "fc1", "none"],
                        help="per-iteration replica weight sync scope: "
                        "fc1fc2 = faithful full-expert bytes (default), "
                        "fc1 = only what the layer0 GEMM consumes "
                        "(moonep-comparable prefetch bytes), none = sync "
                        "once untimed at setup")
    parser.add_argument("--transport", default="nccl",
                        choices=["nccl", "nvshmem"],
                        help="dispatch a2av transport: NCCL alltoallv or"
                        " flux's one-sided NVSHMEM All2AllSingle"
                        " (putmem_nbi into symmetric staging + 2 team"
                        " barriers per call — put-then-barrier, the"
                        " transport class of UltraEP's own external"
                        " dispatchers; BF16/FP32 only)")
    parser.add_argument("--num_comm_sm", type=int, default=8,
                        help="SMs for the NVSHMEM a2av kernel (nvshmem only)")
    parser.add_argument("--overlap_ws", default=False, action="store_true",
                        help="run weight_sync on a dedicated high-priority"
                        " stream + separate NCCL communicator, forked right"
                        " after plan_comm (restores upstream's async"
                        " comm-stream weight_sync; serialized is the"
                        " deliberately-pessimistic default)")
    parser.add_argument("--ws_join", default="dispatch",
                        choices=["dispatch", "gemm"],
                        help="where the main stream joins weight_sync:"
                        " 'dispatch' = before the token a2a — AUTHENTIC"
                        " (upstream's reference integration joins there,"
                        " and it is the publication mechanism for its"
                        " unsignaled peer-VA pushes, NR-12); 'gemm' ="
                        " before the GEMM — labeled COUNTERFACTUAL, sound"
                        " in this NCCL port only because two-sided irecv"
                        " completion rides the ws event")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    if args.transport == "nvshmem":
        # the one-sided All2AllSingle needs the flux shm / NVSHMEM heap;
        # world group so pg ranks == NVSHMEM PEs (moonep precedent)
        assert DTYPE_MAP[args.dtype] != torch.float16, (
            "All2AllSingle instantiates BF16/FP32 only; fp16 aborts in-kernel"
        )
        flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    assert args.H * input_dtype.itemsize == args.chunk_bytes
    W = DIST_ENV.WORLD_SIZE
    assert args.G % W == 0, f"{args.G} % {W} != 0"
    D = args.nvl_domain_size or DIST_ENV.LOCAL_WORLD_SIZE
    assert W % D == 0, f"W ({W}) % D ({D}) != 0"
    if D == DIST_ENV.LOCAL_WORLD_SIZE:
        assert_node_major_ranks()

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

    cfg = UltraEPConfig(
        S=S, K=args.topk, G=args.G, R=W, H=args.H, D=D,
        R_red=args.redundant_per_rank,
        balance_threshold=args.balance_threshold,
        min_tokens_per_replica=args.min_tokens_per_replica,
        allow_zero_master_quota=args.allow_zero_master_quota,
        locality_aware=not args.no_locality,
        oracle_eps=args.oracle_eps,
        interleave=not args.no_interleave,
    )

    # Replicated planning: identical integer plan on every rank from the
    # (globally known) routing. Wall time reported as plan_host_ms under the
    # pre-rule-5 legacy_untimed_plan accounting; see SCHEMA protocol rule 5.
    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    t0 = time.perf_counter()
    tpe = loads_from_topk(cfg, topk_all)
    plan = solve_placement(cfg, tpe)
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

    runner = UltraEPLayer0Runner(
        plan, rank, TP_GROUP, torch.cuda.current_device(), topk_all,
        dtype=input_dtype, ffn_size_shard=moe_ctx.ffn_size_shard,
        sync_fc2=(args.weight_sync == "fc1fc2"),
    )
    if args.transport == "nvshmem":
        # collective + device-syncing ctor: setup only, never timed
        runner.enable_nvshmem(DIST_ENV.LOCAL_WORLD_SIZE, args.num_comm_sm)
    fc2_home = runner.make_fc2_home() if args.weight_sync == "fc1fc2" else None
    do_weight_sync = args.weight_sync != "none"
    if not do_weight_sync:
        # replicas still need weights for the GEMM: one untimed sync at setup
        runner.weight_sync(moe_ctx.weights[0], fc2_home)

    ws_group = None
    ws_stream = None
    if args.overlap_ws:
        # separate NCCL communicator: ws P2Ps must not serialize on the
        # dispatch collectives' communicator; every rank enqueues weight_sync
        # first (right after plan_comm), so cross-communicator order is
        # consistent and deadlock-free
        ws_group = torch.distributed.new_group(ranks=list(range(W)),
                                               backend="nccl")
        # eager-init the communicator (untimed): NCCL comm init is collective
        # and otherwise LAZY on first use — a rank with zero weight_sync
        # pairs (e.g. rank 11 under the layer-92 trace plan) would never
        # call into ws_group and the other ranks would hang forever in init
        torch.distributed.barrier(group=ws_group)
        ws_stream = torch.cuda.Stream(priority=-1)

    # Per-entry route probs, replicated deterministically so receivers can
    # verify without an extra exchange (values still travel the wire).
    gen = torch.Generator().manual_seed(777)
    w_all = torch.rand(W, S, args.G, dtype=torch.float32, generator=gen)
    probs_shard = w_all[rank].cuda()

    # UltraEP's plan_comm payload: this rank's [G] load histogram.
    loads_shard = tpe[rank].cuda().contiguous()
    loads_gather_buf = torch.zeros(W * args.G, dtype=torch.int32, device="cuda")

    gemm_only_op = flux.GemmOnly(
        moe_ctx.inputs.dtype,
        moe_ctx.inputs.dtype,
        moe_ctx.outputs[0].dtype,
        use_fp8_gemm=False,
    )

    if rank == 0:
        gemm_rows = plan.physical_rows_per_rank()
        loads_g = tpe.long().sum(0)
        before = loads_g.reshape(W, cfg.epn).sum(1)
        mean = float(before.double().mean())
        imb_before = float(before.max()) / mean if mean else 1.0
        imb_after = max(gemm_rows) / mean if mean else 1.0
        lb = nvl_domain_lower_bound(cfg, tpe)
        wire_rows = wire_matrix(cfg, plan, topk_all)
        wire_bytes = [[r * args.chunk_bytes for r in row] for row in wire_rows]
        rep = plan.replica_summary()
        remote_with = remote_token_fraction(cfg, plan, True)
        remote_without = remote_token_fraction(cfg, plan, False)
        print(f"ntokens: {ntokens} ({S} per rank), topk: {args.topk}, "
              f"G: {args.G}, D: {D}, redundant/rank: {args.redundant_per_rank}, "
              f"mtpr: {args.min_tokens_per_replica}")
        print(f"ultraep gemm rows per rank: {gemm_rows}  <- residual physical "
              f"imbalance is the balance fingerprint")
        print(f"imbalance max/mean: before {imb_before:.3f} -> after "
              f"{imb_after:.3f} (LB {lb:.3f}, NVL-domain floor)")
        print(f"solver: T={[d.threshold for d in plan.domain_solutions]} "
              f"paths={[d.path for d in plan.domain_solutions]}")
        print(f"replicas: {rep}")
        print(f"remote token fraction: locality {remote_with:.4f} vs "
              f"no-locality {remote_without:.4f}")
        print(f"plan_host_ms (pre-rule-5 legacy_untimed_plan accounting; see SCHEMA protocol rule 5): {plan_host_ms:.1f}")
        RECORDER.emit_info(
            timing_accounting="legacy_untimed_plan",
            ntokens=ntokens,
            tokens_per_rank=S,
            gemm_rows_per_rank=gemm_rows,
            ultraep_nvl_domain_size=D,
            ultraep_redundant_per_rank=args.redundant_per_rank,
            ultraep_min_tokens_per_replica=args.min_tokens_per_replica,
            ultraep_imbalance_before=imb_before,
            ultraep_imbalance_after=imb_after,
            ultraep_lb_floor=lb,
            ultraep_threshold_T=[d.threshold for d in plan.domain_solutions],
            ultraep_solver_path=[d.path for d in plan.domain_solutions],
            ultraep_replicas_total=rep["total_replicas"],
            ultraep_replicas_max_per_expert=rep["max_replicas_per_expert"],
            ultraep_slots_total=rep["slots_total"],
            ultraep_remote_frac_with_locality=remote_with,
            ultraep_remote_frac_without_locality=remote_without,
            ultraep_plan_host_ms=plan_host_ms,
            ultraep_plan_comm_bytes=W * args.G * 4,
            ultraep_wire_bytes=wire_bytes,
            ultraep_locality=not args.no_locality,
            ultraep_interleave=not args.no_interleave,
            ultraep_weight_sync=args.weight_sync,
            ultraep_overlap_ws=bool(args.overlap_ws),
            ultraep_ws_join=(args.ws_join if args.overlap_ws else "serialized"),
            ultraep_transport=args.transport,
        )
    RECORDER.emit_info(
        ultraep_weight_sync_recv_bytes=runner.weight_sync_recv_bytes(),
        ultraep_dup_rows=runner.dup_rows(),
    )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_ultraep_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        iter_times = perf_ultraep(
            runner, moe_ctx, probs_shard, loads_shard, loads_gather_buf,
            fc2_home, gemm_only_op, args.warmup_iters, args.iters,
            do_weight_sync=do_weight_sync,
            overlap_ws=args.overlap_ws,
            ws_join=args.ws_join,
            ws_group=ws_group,
            ws_stream=ws_stream,
        )

    def fmt(times):
        return ", ".join(
            f"{k[:-3]} {sum(v) / max(len(v), 1):.3f} ms"
            for k, v in times.items() if k != "iso_sync_ms"
        )

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"ultraep #{rank}: {fmt(iter_times)}")
    )
    RECORDER.emit_iters("ultraep", iter_times)

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-2, 1.5e-2

    if not args.skip_correctness:
        # NOT under exec_in_rank_order: the check runs collectives (inputs
        # allgather, weight-sync-verification broadcasts) on every rank.
        check_correctness(
            runner, moe_ctx, plan, topk_all, w_all, fc2_home, gemm_only_op,
            atol, rtol, do_weight_sync=do_weight_sync,
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
