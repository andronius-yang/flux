################################################################################
#
# EPLB static-placement layer0 arm for the sweep harness.
# Structural clone of test_moe_ultraep_traffic.py; placement algorithm is the
# VENDORED deepseek-ai/EPLB (eplb_oracle/eplb.py @ d52c72d), mapped onto the
# shared UltraEPPlan machinery by python/flux/testing/eplb_semantics.py
# (invariant-tested by test_eplb_planner.py).
#
################################################################################
"""COMET-style layer0 benchmark under EPLB's static predicted-load placement.

EPLB computes ONE placement per cell — a full re-placement (masters move too)
plus replicas in the same nlp = G/W + R_red slot budget as the ultraep arm —
from a PREDICTED per-expert load vector (--eplb_load_file: the full trace
pool histogram; absent = the batch's own load, the degenerate self-oracle).
Per iteration there is NO solver and NO weight movement; the one-time weight
placement runs untimed at setup and is book-kept as
eplb_weight_place_bytes / eplb_weight_place_ms_oneshot.

Pipeline per iteration (each phase bracketed with CUDA events):
  plan_comm  all_gather of the [W, G] int32 per-rank expert loads. KEPT per
             iteration by design: the two-sided a2av needs recv splits, which
             a real deployment derives from exactly this exchange — EPLB
             removes the solver and the weight movement from the critical
             path, not the counts exchange. Never claim plan_comm = 0.
  pack       rerouted-row gather, (physical expert, token)-sorted
  comm       a2av of token rows + per-entry fp32 route probs (NO dedup,
             ultraep-identical wire semantics; eplb_dup_rows audits)
  scatter    placement into per-physical-expert segments
  prefetch   EMPTY (~0 by construction; the event is kept so the six phase
             names stay capsule-comparable across the EP arms — this
             near-zero column vs moonep prefetch / ultraep weight_sync IS
             EPLB's headline recurring-cost advantage)
  gemm       per-segment GEMM over the received rows, no padding: the
             residual imbalance of the static placement on THIS batch is the
             measurement (vs moonep's constant S*K and ultraep's per-batch
             re-solve)

Token -> physical instance split: equal split of each expert's batch load
across its instances (largest-remainder), decomposed per source by the
shared non-locality rank-quota path with the same coprime-stride interleave
as the ultraep arm. Deterministic and communication-free given the static
placement + the loads all-gather. The locality-aware path is structurally
FORBIDDEN on EPLB plans (replicas are not NVL-domain-confined).
"""

import argparse
import hashlib
import json
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
from flux.testing.eplb_semantics import (
    EPLB_POLICIES,
    EPLBLayer0Runner,
    build_eplb_plan,
    predicted_rows_per_rank,
    weight_placement_pairs,
)
from flux.testing.ultraep_semantics import (
    UltraEPConfig,
    loads_from_topk,
    remote_token_fraction,
    wire_matrix,
)
from flux.testing.recorder import RECORDER

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eplb_oracle import rebalance_experts  # noqa: E402  (vendored algorithm)

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
    """EPLB's hier policy and the one-time placement accounting assume
    node-major contiguous ranks (rank k on node k // local_world)."""
    import socket

    host = socket.gethostname()
    hosts = [None] * DIST_ENV.WORLD_SIZE
    torch.distributed.all_gather_object(hosts, host, group=TP_GROUP)
    lw = DIST_ENV.LOCAL_WORLD_SIZE
    for node in range(DIST_ENV.WORLD_SIZE // lw):
        block = hosts[node * lw:(node + 1) * lw]
        assert len(set(block)) == 1, f"ranks not node-major: {hosts}"


@torch.no_grad()
def perf_eplb(
    runner: EPLBLayer0Runner,
    ctx: MoeMlp1Ctx,
    probs_shard: torch.Tensor,
    loads_shard: torch.Tensor,
    loads_gather_buf: torch.Tensor,
    gemm_only_op,
    warmup_iters: int,
    iters: int,
):
    total_iters = warmup_iters + iters
    names = ["start", "plan_comm", "pack", "comm", "scatter", "prefetch", "gemm"]
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
            # Recurring counts exchange: recv splits are derived from this
            # (the placement itself is static — no solver runs here).
            torch.distributed.all_gather_into_tensor(
                loads_gather_buf, loads_shard, group=TP_GROUP
            )
            ev["plan_comm"][i].record()
            runner.pack(ctx.inputs_shard, probs_shard)
            ev["pack"][i].record()
            runner.a2av()
            ev["comm"][i].record()
            runner.place()
            ev["scatter"][i].record()
            # No per-iteration weight movement: prefetch is an empty phase,
            # recorded for six-name parity (reads ~0 by construction).
            ev["prefetch"][i].record()
            runner.gemm(gemm_only_op)
            ev["gemm"][i].record()

    keys = ["plan_comm_ms", "pack_ms", "comm_ms", "scatter_ms",
            "prefetch_ms", "gemm_ms", "total_ms"]
    times = {k: [] for k in keys}
    for i in range(total_iters):
        ev["gemm"][i].synchronize()
        if i < warmup_iters:
            continue
        seq = ["plan_comm", "pack", "comm", "scatter", "prefetch", "gemm"]
        prev = ev["start"][i]
        for name in seq:
            times[f"{name}_ms"].append(prev.elapsed_time(ev[name][i]))
            prev = ev[name][i]
        times["total_ms"].append(ev["start"][i].elapsed_time(ev["gemm"][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    return times


def check_correctness(runner, ctx, plan, topk_all, w_all, gemm_only_op,
                      atol, rtol):
    """Content bitwise + wire-quota audit + placed-weight bitwise check +
    per-segment GEMM allclose (collective: run on every rank)."""
    from flux.testing.ultraep_semantics import reroute_expand

    cfg = runner.cfg
    rank, R, S = runner.rank, cfg.R, cfg.S
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

    # Placed weights: every local slot must hold the canonical weights of its
    # logical expert, bitwise. Non-locally-homed slots were zeroed at setup
    # and only filled by the one-time P2P irecv, so equality proves the wire
    # content without any broadcast.
    p2l = plan.p2l.long()
    for b in range(cfg.nlp):
        l = int(p2l[rank * cfg.nlp + b])
        exp1 = runner.make_canonical_fc1(l).to(runner.device)
        if not torch.equal(runner.slot_fc1[b], exp1):
            ok_bitwise = False
            print(f"❌ rank {rank}: fc1 slot {b} (expert {l}) mismatch")
        if runner.place_fc2:
            exp2 = runner.make_canonical_fc2(l).to(runner.device)
            if not torch.equal(runner.slot_fc2[b], exp2):
                ok_bitwise = False
                print(f"❌ rank {rank}: fc2 slot {b} (expert {l}) mismatch")

    # Per-segment GEMM vs torch.matmul (weights indexed by physical slot).
    ok_allclose = True
    for p, start, end, logical in runner.lay.gemm_segments:
        w = runner.slot_fc1[p]
        ref = torch.matmul(
            runner.hidden_buf[start:end].float(), w.float().t()
        ).to(runner.out_buf.dtype)
        try:
            flux.torch_allclose(runner.out_buf[start:end], ref, atol=atol, rtol=rtol)
        except Exception:
            ok_allclose = False
            print(f"❌ rank {rank}: gemm segment slot={p} expert={logical} mismatch")

    status = "✅" if (ok_bitwise and ok_allclose) else "❌"
    print(f"{status} rank {rank}: eplb dispatch content "
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
    # --- EPLB knobs ---
    parser.add_argument("--eplb_load_file", type=str, default=None,
                        help="predicted per-expert load JSON (the "
                        "<mid>.eplb_load.json sidecar: full trace pool "
                        "histogram). Absent = the batch's own aggregate load "
                        "(degenerate self-oracle; fine for smoke tests)")
    parser.add_argument("--eplb_policy", default="global",
                        choices=list(EPLB_POLICIES),
                        help="global = DeepSeek's decode deployment "
                        "(canonical; replicas unconfined), hier = per-expert "
                        "node packing with node-confined replicas")
    parser.add_argument("--redundant_per_rank", type=int, default=2,
                        help="redundant expert slots per rank (slot-budget "
                        "parity with the ultraep arm)")
    parser.add_argument("--no_interleave", default=False, action="store_true",
                        help="disable coprime-stride reroute interleaving")
    parser.add_argument("--weight_place", default="fc1fc2",
                        choices=["fc1fc2", "fc1"],
                        help="one-time placement scope: fc1fc2 = faithful "
                        "full-expert bytes (default), fc1 = only what the "
                        "layer0 GEMM consumes (halves setup memory)")
    parser.add_argument("--transport", default="nccl",
                        choices=["nccl", "nvshmem"],
                        help="dispatch a2av transport: NCCL alltoallv or"
                        " flux's one-sided NVSHMEM All2AllSingle"
                        " (putmem_nbi into symmetric staging + 2 team"
                        " barriers per call; BF16/FP32 only)")
    parser.add_argument("--num_comm_sm", type=int, default=8,
                        help="SMs for the NVSHMEM a2av kernel (nvshmem only)")
    parser.add_argument("--skip_correctness", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    if args.transport == "nvshmem":
        # the one-sided All2AllSingle needs the flux shm / NVSHMEM heap;
        # world group so pg ranks == NVSHMEM PEs (moonep/ultraep precedent)
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
    assert_node_major_ranks()
    num_nodes = W // DIST_ENV.LOCAL_WORLD_SIZE

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

    # D is only divisibility bookkeeping here (the locality path is never
    # used on EPLB plans); solver knobs are irrelevant to the EPLB builder.
    cfg = UltraEPConfig(
        S=S, K=args.topk, G=args.G, R=W, H=args.H,
        D=DIST_ENV.LOCAL_WORLD_SIZE,
        R_red=args.redundant_per_rank,
        locality_aware=False,
        interleave=not args.no_interleave,
    )

    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    tpe = loads_from_topk(cfg, topk_all)

    # Predicted load: the full-pool histogram sidecar, or the batch's own
    # aggregate load as the degenerate fallback.
    if args.eplb_load_file:
        with open(args.eplb_load_file, "rb") as f:
            raw = f.read()
        blob = json.loads(raw)
        assert blob.get("version") == 1, f"unknown load-file version: {blob.get('version')}"
        assert int(blob["G"]) == args.G, (
            f"load file G={blob['G']} != --G {args.G}"
        )
        pool_load = blob["load"]
        assert len(pool_load) == args.G
        load_source = "pool"
        load_sha = hashlib.sha256(raw).hexdigest()[:16]
    else:
        pool_load = tpe.long().sum(0).tolist()
        load_source = "batch"
        load_sha = ""
        if rank == 0:
            print("eplb: NO --eplb_load_file; placement from the batch's own "
                  "load (self-oracle — fine for smoke, not a headline cell)")

    # Static placement, computed identically on every rank at setup
    # (untimed-metadata contract; the vendored algorithm is deterministic
    # per software stack, and the plan-hash all-gather below is the guard).
    t0 = time.perf_counter()
    plan = build_eplb_plan(cfg, tpe, pool_load, args.eplb_policy, num_nodes,
                           rebalance_experts)
    plan_host_ms = (time.perf_counter() - t0) * 1e3

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

    runner = EPLBLayer0Runner(
        plan, rank, TP_GROUP, torch.cuda.current_device(), topk_all,
        dtype=input_dtype, ffn_size_shard=moe_ctx.ffn_size_shard,
        place_fc2=(args.weight_place == "fc1fc2"),
    )
    if args.transport == "nvshmem":
        # collective + device-syncing ctor: setup only, never timed
        runner.enable_nvshmem(DIST_ENV.LOCAL_WORLD_SIZE, args.num_comm_sm)

    # ONE-TIME weight placement (the whole point of the arm): untimed
    # w.r.t. the phase loop, wall-clocked once and book-kept.
    place_bytes, place_ms = runner.place_weights(TP_GROUP)

    # Per-entry route probs, replicated deterministically so receivers can
    # verify without an extra exchange (values still travel the wire).
    gen = torch.Generator().manual_seed(777)
    w_all = torch.rand(W, S, args.G, dtype=torch.float32, generator=gen)
    probs_shard = w_all[rank].cuda()

    # Recurring counts exchange payload: this rank's [G] load histogram.
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
        pred = predicted_rows_per_rank(plan, pool_load)
        pred_mean = sum(pred) / len(pred)
        imb_pred = max(pred) / pred_mean if pred_mean else 1.0
        wire_rows = wire_matrix(cfg, plan, topk_all)
        wire_bytes = [[r * args.chunk_bytes for r in row] for row in wire_rows]
        rep = plan.replica_summary()
        # locality-aware fraction is FORBIDDEN on EPLB plans (would assert
        # NVL-domain membership the placement doesn't satisfy)
        remote_frac = remote_token_fraction(cfg, plan, False)
        n_moved = len(weight_placement_pairs(plan))
        print(f"ntokens: {ntokens} ({S} per rank), topk: {args.topk}, "
              f"G: {args.G}, policy: {args.eplb_policy}, "
              f"redundant/rank: {args.redundant_per_rank}, "
              f"load source: {load_source}")
        print(f"eplb gemm rows per rank: {gemm_rows}  <- residual imbalance "
              f"of the STATIC placement on this batch is the measurement")
        print(f"imbalance max/mean: before {imb_before:.3f} -> after "
              f"{imb_after:.3f} (predicted, on the pool load: {imb_pred:.3f})")
        print(f"replicas: {rep}; re-homed physical slots: {n_moved}")
        print(f"remote token fraction (no-locality split): {remote_frac:.4f}")
        print(f"one-time weight placement: {place_ms:.1f} ms wall "
              f"(recv {place_bytes} B on rank 0)")
        print(f"plan_host_ms (untimed-metadata contract): {plan_host_ms:.1f}")
        RECORDER.emit_info(
            ntokens=ntokens,
            tokens_per_rank=S,
            gemm_rows_per_rank=gemm_rows,
            eplb_policy=args.eplb_policy,
            eplb_redundant_per_rank=args.redundant_per_rank,
            eplb_imbalance_before=imb_before,
            eplb_imbalance_after=imb_after,
            eplb_pred_imbalance=imb_pred,
            eplb_replicas_total=rep["total_replicas"],
            eplb_replicas_max_per_expert=rep["max_replicas_per_expert"],
            eplb_slots_total=rep["slots_total"],
            eplb_rehomed_slots=n_moved,
            eplb_remote_frac=remote_frac,
            eplb_plan_host_ms=plan_host_ms,
            eplb_plan_comm_bytes=W * args.G * 4,
            eplb_wire_bytes=wire_bytes,
            eplb_load_source=load_source,
            eplb_load_file=args.eplb_load_file or "",
            eplb_load_sha=load_sha,
            eplb_interleave=not args.no_interleave,
            eplb_weight_place=args.weight_place,
            eplb_weight_place_ms_oneshot=place_ms,
            eplb_transport=args.transport,
        )
    RECORDER.emit_info(
        eplb_weight_place_bytes=place_bytes,
        eplb_dup_rows=runner.dup_rows(),
    )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_eplb_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        iter_times = perf_eplb(
            runner, moe_ctx, probs_shard, loads_shard, loads_gather_buf,
            gemm_only_op, args.warmup_iters, args.iters,
        )

    def fmt(times):
        return ", ".join(
            f"{k[:-3]} {sum(v) / max(len(v), 1):.3f} ms"
            for k, v in times.items() if k != "iso_sync_ms"
        )

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"eplb #{rank}: {fmt(iter_times)}")
    )
    RECORDER.emit_iters("eplb", iter_times)

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-2, 1.5e-2

    if not args.skip_correctness:
        # NOT under exec_in_rank_order: the check runs the inputs allgather
        # collectively on every rank.
        check_correctness(
            runner, moe_ctx, plan, topk_all, w_all, gemm_only_op, atol, rtol,
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
