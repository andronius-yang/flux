################################################################################
#
# OURS — the fused PLACE-lambda + LocCap + Slipstream-v2 l0+l1 arm.
#
# Fresh integrated algorithm (2026-08-24 fusion campaign): the LLC trio's
# placement/routing feeding the Slipstream v2 overlapped dispatch+combine
# through the virtual-slot space, ONE op pair, no staged wire, no python
# permutes, no probs side-wire. See python/flux/testing/ours.py for the
# algorithm module and the timing contract.
#
################################################################################
"""Scenario-1 driver: placement solved from the pre-batch oracle (untimed,
reported), LocCap sender-local kernel routes per iteration (relaxed
contract), fused dispatch+GEMM0 (LB_UNION Tier-B + wave-pack) and fused
GEMM1+combine (msplit + fused-pack + bucket, ns1) overlap comm with compute.

Per-iteration timed window (rule-5 anchors match the epic/l01 drivers):
  iter_start | d[R,G] allgather | plan_comm_end
  | route kernel -> fused phys+probs allgather -> vce -> derive_routed_meta
  [-> derive_combine_meta when --plan_overlap 0] | plan_end
  | e2e_start | fused l0 (side stream: combine meta + scale when
  --plan_overlap 1) | l0_end | GELU | act_end | fused l1 | e2e_end

Correctness: relaxed-kernel timed iterations under FLUX_RANDOM_PAYLOAD; one
FINAL DETERMINISTIC iteration on the setup torch loccap_route_sl routing,
validated against a local logical-space two-layer torch reference
(replicas share logical weights, so the logical reference is exact)."""

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
    choosed_experts_to_matrix_chunks,
    load_routing_file,
    parse_traffic_matrix,
    traffic_matrix_to_choosed_experts,
)
from flux.testing.epic_semantics import (
    build_nodeaware_plan,
    python_meta_from_vce,
)
from flux.testing.placelambda_gpu import (
    loccap_route_sl,
    loccap_sl_bounds,
)
from flux.testing import placelambda_fast as plfast
from flux.testing.ultraep_semantics import (
    UltraEPConfig,
    loads_from_topk,
)
from flux.testing.ours import OursIterPlanner, OursRunner
from flux.testing.payload_probe import PayloadProbe, payload_probe_enabled
from flux.testing.recorder import RECORDER

DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)
print = partial(print, flush=True)


def init_ep_group(ep_size: int):
    global EP_GROUP
    assert DIST_ENV.WORLD_SIZE % ep_size == 0
    assert EP_GROUP is None
    ffn_tp_size = TP_GROUP.size() // ep_size
    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)
    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i:i + ep_size])
    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group


def assert_node_major_ranks():
    import socket

    host = socket.gethostname()
    hosts = [None] * DIST_ENV.WORLD_SIZE
    torch.distributed.all_gather_object(hosts, host, group=TP_GROUP)
    lw = DIST_ENV.LOCAL_WORLD_SIZE
    for node in range(DIST_ENV.WORLD_SIZE // lw):
        block = hosts[node * lw:(node + 1) * lw]
        assert len(set(block)) == 1, f"ranks not node-major: {hosts}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traffic_matrix", type=str, required=True)
    p.add_argument("--routing_file", type=str, required=True,
                   help="real trace routing (OURS is defined for trace cells)")
    p.add_argument("--oracle_routing_file", type=str, default="",
                   help="scenario-1 oracle window rows (placement basis)")
    p.add_argument("--G", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--ffn_hidden_size", type=int, required=True)
    p.add_argument("--chunk_bytes", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup_iters", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.0625)
    p.add_argument("--pll_f_cap", type=int, default=-1,
                   help="-1 = auto from the reference tables")
    p.add_argument("--sizing", choices=["demand", "capacity"],
                   default="demand",
                   help="demand: realized reference + drift cushions "
                        "(default); capacity: provable caps everywhere")
    p.add_argument("--plan_overlap", type=int, default=0,
                   help="1: combine-meta derive + scale build overlap the "
                        "fused l0 on a side stream; 0 (default): inside the "
                        "plan bracket. Flip only after the 4n+16n A/B.")
    p.add_argument("--sm_margin", type=int, default=1)
    p.add_argument("--skip_correctness", default=False, action="store_true")
    p.add_argument("--check_iters", type=int, default=0,
                   help="GATE MODE: validate EVERY iteration's output "
                        "against the logical local reference (relaxed "
                        "kernel routing + random payload — catches "
                        "first-call-freeze bugs in the fused path). "
                        "Perturbs timing; never a perf configuration.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def gen_expert_w1(e: int, ffn: int, H: int, dtype):
    g = torch.Generator().manual_seed(10007 + int(e))
    return ((torch.rand((ffn, H), generator=g) * 0.02) - 0.01).to(dtype)


def gen_expert_w2(e: int, ffn: int, H: int, dtype):
    g = torch.Generator().manual_seed(20011 + int(e))
    return ((torch.rand((H, ffn), generator=g) * 0.02) - 0.01).to(dtype)


def build_slot_weights(p2l, rank, nlp, gpe, ffn, H, dtype):
    """w1v [gpe, ffn, H], w2v [gpe, H, ffn]; PAD-FIRST convention: local
    index 0 is the (zero-row) pad slot, real slot i lives at index 1+i."""
    w1 = torch.zeros(gpe, ffn, H, dtype=dtype)
    w2 = torch.zeros(gpe, H, ffn, dtype=dtype)
    for i in range(nlp):
        e = int(p2l[rank * nlp + i])
        w1[1 + i] = gen_expert_w1(e, ffn, H, dtype)
        w2[1 + i] = gen_expert_w2(e, ffn, H, dtype)
    return w1.cuda(), w2.cuda()


@torch.no_grad()
def torch_reference_local(inputs_shard, topk_own, probs_own, ffn, H, dtype):
    """Exact logical-space two-layer reference for THIS rank's tokens:
    y[t] = sum_k probs[t,k] * gelu(x[t] @ W1[e]^T) @ W2[e]^T, e = topk[t,k].
    GEMMs in the op dtype, accumulation fp32 (mirrors the fused numerics)."""
    S = inputs_shard.shape[0]
    K = topk_own.shape[1]
    y = torch.zeros(S, H, dtype=torch.float32, device="cuda")
    e_flat = topk_own.reshape(-1).long()
    t_flat = (torch.arange(S, device=e_flat.device, dtype=torch.long)
              .repeat_interleave(K))
    p_flat = probs_own.reshape(-1).float().cuda()
    for e in torch.unique(e_flat).tolist():
        sel = (e_flat == e).nonzero(as_tuple=True)[0]
        rows = t_flat[sel].cuda()
        w1 = gen_expert_w1(e, ffn, H, dtype).cuda()
        w2 = gen_expert_w2(e, ffn, H, dtype).cuda()
        part = torch.nn.functional.gelu(inputs_shard[rows] @ w1.t()) @ w2.t()
        y.index_add_(0, rows, part.float() * p_flat[sel.cuda()].unsqueeze(1))
    return y.to(dtype)


def l0_union_recv_demand(uc_ref, rank, W, L, S, K):
    """Realized l0 recv demand for THIS rank under LB_UNION recv regions:
    intra-node sources deliver per-rank u[s][me]; remote nodes deliver the
    node union U[s][my_node] (every local rank holds the window). Mirrors
    sweep.py exact_scale_knobs recv_union."""
    my_node = rank // L
    u = uc_ref[:, :W]
    U = uc_ref[:, W:]
    total = 0
    for s in range(W):
        if s // L == my_node:
            total += int(u[s, rank])
        else:
            total += int(U[s, my_node])
    return total


def main():
    args = parse_args()
    input_dtype = DTYPE_MAP[args.dtype]
    assert input_dtype == torch.bfloat16, "OURS arm instantiates BF16"
    assert args.H * input_dtype.itemsize == args.chunk_bytes
    W = DIST_ENV.WORLD_SIZE
    L = DIST_ENV.LOCAL_WORLD_SIZE
    nn = W // L
    rank = TP_GROUP.rank()
    assert args.G % W == 0
    assert args.warmup_iters >= 1, (
        "warmup >= 1 is mandatory (LAZY module loads must happen before "
        "timing; see critique M3)")
    assert_node_major_ranks()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()

    matrix = parse_traffic_matrix(args.traffic_matrix)
    assert matrix.shape[0] == W
    choosed_experts = load_routing_file(args.routing_file, args.G, args.topk)
    assert choosed_experts.shape[0] % W == 0
    got = choosed_experts_to_matrix_chunks(choosed_experts, W, args.G // W)
    assert torch.equal(got * args.chunk_bytes, matrix), (
        "routing file does not realize --traffic_matrix")
    ntokens = choosed_experts.shape[0]
    S = ntokens // W
    assert ntokens % (W * args.topk) == 0, "l01 needs ntokens % (W*topk) == 0"

    cfg = UltraEPConfig(
        S=S, K=args.topk, G=args.G, R=W, H=args.H, D=L,
        R_red=0, locality_aware=False, interleave=True,
    )
    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    tpe = loads_from_topk(cfg, topk_all)

    # gate weights (harness gating output; replicated generation)
    gen_p = torch.Generator().manual_seed(777)
    probs_all_setup = torch.rand((ntokens, args.topk), generator=gen_p) + 0.5

    # ---- placement: PLACE-lambda FAST from the s1 oracle (untimed) ----
    tk_dev = topk_all.long().cuda()
    tk_solve = tk_dev
    oracle_basis = "self"
    if args.oracle_routing_file:
        _oc = load_routing_file(args.oracle_routing_file, args.G, args.topk)
        assert _oc.shape[0] % W == 0
        tk_solve = _oc.view(W, -1, args.topk).long().cuda()
        oracle_basis = "prev_batch"
    pf_cfg = dict(
        passes_a=int(os.environ.get("FLUX_PLACE_FAST_PA", "4")),
        passes_b=int(os.environ.get("FLUX_PLACE_FAST_PB", "3")),
        repair_passes=int(os.environ.get("FLUX_PLACE_FAST_REPAIR", "2")),
        seed=os.environ.get("FLUX_PLACE_FAST_SEED", "affinity"),
    )
    torch.cuda.synchronize()
    t_ps = time.perf_counter()
    pf_solve = plfast.build_placement_fast(tk_solve, L, cfg.nlp, args.G,
                                          **pf_cfg)
    hosts_pll = plfast.finalize_hosts(pf_solve, W, L, cfg.nlp,
                                      method="snake")
    torch.cuda.synchronize()
    place_solver_ms = (time.perf_counter() - t_ps) * 1e3
    pblob = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
             "hosts": hosts_pll, "planner": plfast.stats_host(pf_solve)}
    plan = build_nodeaware_plan(cfg, tpe, pblob)
    _drift = plfast.drift_ppm(
        plfast.demand_hist(tk_dev, L, args.G),
        plfast.demand_hist(tk_solve, L, args.G))

    # ---- setup reference route (deterministic torch; sizes buffers,
    #      binds the final correctness iteration) ----
    phys_ref, pll_aux = loccap_route_sl(
        topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
        cfg.nlp, L, args.eps, return_tables=True)
    plan.phys_override = phys_ref
    pll_bounds = loccap_sl_bounds(pll_aux, W, args.pll_f_cap)
    args.pll_f_cap = pll_bounds["f_cap"]

    gpe = cfg.nlp + 1
    E_virt = W * gpe
    _pr = phys_ref.view(ntokens, args.topk).long()
    vce_ref = (_pr // cfg.nlp) * gpe + 1 + _pr % cfg.nlp  # pad-FIRST
    sp_ref, sc_ref, sps_ref, uc_ref = python_meta_from_vce(
        vce_ref.cuda(), W, S, gpe, nn, L)

    # kernel-drift cushion (handoff 17 / llc demand-sizing lineage)
    fp_slack = int(pll_aux["forced_pair"].sum(0).max())
    cushion = fp_slack + 8 * W

    # recv rows this rank computes (dispatch recv == combine send)
    recv_real = int(sps_ref[:, rank * gpe:(rank + 1) * gpe].sum())
    if args.sizing == "demand":
        recv_cap = min(pll_bounds["recv_cap"], recv_real + cushion)
    else:
        recv_cap = pll_bounds["recv_cap"]

    # l0 recv buffer must ALSO hold the LB_UNION recv regions
    union_recv_real = l0_union_recv_demand(uc_ref.cpu(), rank, W, L, S,
                                           args.topk)
    l0_recv_rows = max(recv_cap, union_recv_real + cushion)
    # collective max: the l0 recv gate is per-rank — never let one rank
    # under-size relative to a peer's view of the same expression
    t_red = torch.tensor([l0_recv_rows, recv_cap], dtype=torch.int64,
                         device="cuda")
    torch.distributed.all_reduce(t_red,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=TP_GROUP)
    l0_recv_rows = int(t_red[0])
    recv_cap = int(t_red[1])

    # relay/stage under lb_union: balanced chunks ~ per-round union mass
    U_ref = uc_ref[:, W:].cpu()
    relay_rows = int(U_ref.sum(0).max()) + cushion
    stage_rows = relay_rows

    os.environ["FLUX_A2AV_MAX_RECV_NTOKENS"] = str(int(l0_recv_rows))
    os.environ["FLUX_A2AV_MAX_STAGE_NTOKENS"] = str(int(stage_rows))
    os.environ["FLUX_A2AV_MAX_RELAY_NTOKENS"] = str(int(relay_rows))

    # l1 (combine) capacity: EXACT demands on the reference virtual meta
    # (parity-tested formulas from the moonep virtual space) + drift cushion
    from types import SimpleNamespace
    from flux.testing.moonep_fused_map import required_a2av_rs_knobs
    ref_meta = SimpleNamespace(
        splits=sp_ref.cpu(),
        splits_per_source=sps_ref.cpu(),
        a2av_unique_counts=uc_ref.cpu(),
    )
    rs_exact = required_a2av_rs_knobs(ref_meta, W, L)
    for k, v in rs_exact.items():
        os.environ[k] = str(int(v) + cushion)
    # send panel additionally bounds the fused-pack capacity check: it must
    # cover the RELAXED per-iteration recv (== combine send) worst case
    os.environ["FLUX_A2AV_RS_MAX_SEND_ROWS"] = str(
        max(int(rs_exact["FLUX_A2AV_RS_MAX_SEND_ROWS"]) + cushion, recv_cap))

    if rank == 0:
        print(f"OURS sizing({args.sizing}): recv_cap {recv_cap} "
              f"(real {recv_real}), l0_recv {l0_recv_rows} "
              f"(union {union_recv_real}), relay {relay_rows}, "
              f"cushion {cushion} (fp_slack {fp_slack})")

    # ---- runner + planner ----
    runner = OursRunner(
        TP_GROUP, EP_GROUP, DIST_ENV.NNODES, L, cfg, args.ffn_hidden_size,
        input_dtype, l0_recv_rows, sm_margin=args.sm_margin,
        plan_overlap=bool(args.plan_overlap))
    w1v, w2v = build_slot_weights(plan.p2l, rank, cfg.nlp, gpe,
                                 args.ffn_hidden_size, args.H, input_dtype)
    runner.set_weights(w1v, w2v)
    planner = OursIterPlanner(plan, rank, torch.device("cuda"), topk_all,
                              probs_all_setup, L, args.eps, args.pll_f_cap,
                              TP_GROUP)

    d_gather_buf = torch.zeros(W, args.G, dtype=torch.int32, device="cuda")

    gen_x = torch.Generator(device="cuda").manual_seed(4242 + rank)
    inputs_shard = ((torch.rand((S, args.H), device="cuda",
                                generator=gen_x) * 0.02) - 0.01
                    ).to(input_dtype)
    probe = PayloadProbe(inputs_shard, rank)

    # ---- setup audit: reference vce through the op derive must equal the
    #      python recipe bitwise (v2b in-window guard, fused-arm edition) ----
    ip_ref = planner.derive_reference()
    assert torch.equal(ip_ref.vce.long(), vce_ref.cuda()), "vce recipe drift"
    runner.plan_meta(ip_ref)
    assert torch.equal(runner._sd.long().cpu(), sp_ref.long().cpu()), (
        "derive splits != python_meta_from_vce")
    assert torch.equal(runner._sps.cpu(), sps_ref.cpu().int()), (
        "derive sps != python_meta_from_vce")
    assert torch.equal(runner._uc.cpu(), uc_ref.cpu().int()), (
        "derive uc != python_meta_from_vce")
    if rank == 0:
        print(f"setup audit OK: E_virt {E_virt} gpe {gpe} m_ref "
              f"{runner._m_this}; placement basis {oracle_basis} "
              f"drift {_drift} ppm; solver {place_solver_ms:.1f} ms")

    RECORDER.emit_info(
        ours_fusion="slipstream_v2",
        ours_plan_overlap=int(bool(args.plan_overlap)),
        ours_sizing=args.sizing,
        ours_recv_cap=int(recv_cap),
        ours_l0_recv_rows=int(l0_recv_rows),
        epic_route_relaxed=1,
        epic_pll_f_cap=int(args.pll_f_cap),
        epic_pll_recv_cap=int(pll_bounds["recv_cap"]),
        epic_pll_oracle_file=args.oracle_routing_file or "",
        epic_pll_oracle_basis=oracle_basis,
        epic_pll_oracle_drift_ppm=_drift,
        epic_place_solver_ms=round(place_solver_ms, 3),
        timing_accounting="per_iter_gpu",
        ours_E_virt=int(E_virt),
    )

    # ---- timed loop ----
    total_iters = args.warmup_iters + args.iters
    ev = lambda: [torch.cuda.Event(enable_timing=True)
                  for _ in range(total_iters)]
    iter_start, plan_comm_end, plan_end = ev(), ev(), ev()
    e2e_start, l0_end, act_end, e2e_end = ev(), ev(), ev(), ev()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    out = None
    for i in range(total_iters):
        runner.prep()
        probe.step(i)
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < args.warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            iter_start[i].record()
            torch.distributed.all_gather_into_tensor(
                d_gather_buf, planner.local_loads(), group=TP_GROUP)
            plan_comm_end[i].record()
            ip = planner.derive(d_gather_buf)
            runner.plan_meta(ip)
            plan_end[i].record()
            e2e_start[i].record()
            runner.issue_combine_meta(ip)
            l0_out = runner.l0_forward(inputs_shard)
            l0_end[i].record()
            intermediate = torch.nn.functional.gelu(l0_out)
            act_end[i].record()
            out = runner.l1_forward(intermediate)
            e2e_end[i].record()
        if args.check_iters:
            # GATE MODE (critique H1): validate THIS iteration's output —
            # relaxed kernel routing + this iteration's random payload —
            # against the routing-independent logical reference. Catches
            # first-call-freeze / stale-metadata bugs in the fused path.
            torch.cuda.synchronize()
            ref_i = torch_reference_local(
                inputs_shard, topk_all[rank],
                probs_all_setup[rank * S:(rank + 1) * S],
                args.ffn_hidden_size, args.H, input_dtype)
            bad_i = int((~torch.isclose(out.float(), ref_i.float(),
                                        atol=1e-2, rtol=1.5e-2))
                        .any(dim=1).sum())
            print(f"ours #{rank}: iter {i} gate "
                  f"{'OK' if bad_i == 0 else 'BAD'} ({bad_i} bad rows)")
            assert bad_i == 0, (
                f"rank {rank} iter {i}: {bad_i} bad rows under changing "
                f"routing — fused-path per-iteration metadata bug")
        if rank == 0:
            print(f"[hb] window {i + 1}/{total_iters}")

    iter_times = {k: [] for k in ("e2e_ms", "l0_ms", "act_ms", "l1_ms",
                                  "plan_comm_ms", "plan_ms", "total_ms")}
    for i in range(total_iters):
        e2e_end[i].synchronize()
        if i >= args.warmup_iters:
            iter_times["plan_comm_ms"].append(
                iter_start[i].elapsed_time(plan_comm_end[i]))
            iter_times["plan_ms"].append(
                plan_comm_end[i].elapsed_time(plan_end[i]))
            iter_times["e2e_ms"].append(
                e2e_start[i].elapsed_time(e2e_end[i]))
            iter_times["l0_ms"].append(e2e_start[i].elapsed_time(l0_end[i]))
            iter_times["act_ms"].append(l0_end[i].elapsed_time(act_end[i]))
            iter_times["l1_ms"].append(act_end[i].elapsed_time(e2e_end[i]))
            iter_times["total_ms"].append(
                iter_start[i].elapsed_time(e2e_end[i]))
    if isolated:
        iter_times["iso_sync_ms"] = iso_sync_times[args.warmup_iters:]

    def fmt(times):
        return ", ".join(f"{k[:-3]} {sum(v) / max(len(v), 1):.3f} ms"
                         for k, v in times.items())

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"ours #{rank}: {fmt(iter_times)}"))
    RECORDER.emit_iters("ours", iter_times)

    # ---- final deterministic iteration + correctness ----
    ip_ref = planner.derive_reference()
    runner.prep()
    runner.plan_meta(ip_ref)
    runner.issue_combine_meta(ip_ref)
    l0_out = runner.l0_forward(inputs_shard)
    intermediate = torch.nn.functional.gelu(l0_out)
    out = runner.l1_forward(intermediate)
    torch.cuda.synchronize()

    import hashlib
    sha = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()[:16]
    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"ours #{rank}: out_sha {sha}"))
    RECORDER.emit_info(ours_out_sha=sha, epic_layers="l01")

    if not args.skip_correctness:
        ref = torch_reference_local(
            inputs_shard, topk_all[rank], probs_all_setup[rank * S:(rank + 1) * S],
            args.ffn_hidden_size, args.H, input_dtype)
        atol, rtol = 1e-2, 1.5e-2
        ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
        n_bad = int((~torch.isclose(out.float(), ref.float(), atol=atol,
                                    rtol=rtol)).any(dim=1).sum())
        flux.exec_in_rank_order(
            TP_GROUP,
            lambda: print(f"ours #{rank}: correctness "
                          f"{'PASS' if ok else 'FAIL'} (bad rows {n_bad}"
                          f"/{out.shape[0]}); {probe.describe()}"))
        RECORDER.emit_info(ours_allclose=int(ok), ours_bad_rows=n_bad)
        assert ok, f"rank {rank}: OURS correctness FAILED ({n_bad} bad rows)"

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()


if __name__ == "__main__":
    main()
