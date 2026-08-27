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
    p.add_argument("--redundant_per_rank", type=int, default=0,
                   help="replica slot headroom per rank (nlp = G/W + this). "
                        "OURS canon = 0; EPIC/EPLB/llc default 2 -- the r2 "
                        "arms are the parity probe (2026-08-25)")
    p.add_argument("--check_iters", type=int, default=0,
                   help="GATE MODE: validate EVERY iteration's output "
                        "against the logical local reference (relaxed "
                        "kernel routing + random payload — catches "
                        "first-call-freeze bugs in the fused path). "
                        "Perturbs timing; never a perf configuration.")
    p.add_argument("--scenario", choices=["s1", "s2"], default="s1",
                   help="s1: static oracle placement (no weight movement); "
                        "s2: per-iteration live re-placement with "
                        "OVERLAPPED weight movement (WPM multicast + "
                        "NIC-shard + per-slot weight-gated tiles)")
    p.add_argument("--place_gain_threshold_ppm", type=int, default=50000)
    p.add_argument("--place_keep_bonus", type=int, default=-1,
                   help="warm-solve resident stickiness; -1 = auto "
                        "(0 in the always regime, 90090 otherwise). The "
                        "kb ladder probes the drift-demanded-moves band: "
                        "kb=0 re-derives ~freely, 90090 freezes.")
    p.add_argument("--place_drift_prefilter_ppm", type=int, default=10000,
                   help="skip the warm solve when the observed demand "
                        "drift vs the resident placement's basis is below "
                        "this (place_ms then ≈ the drift check)")
    p.add_argument("--weight_shard", choices=["off", "on"], default="on")
    p.add_argument("--s2_join", choices=["tiles", "join"], default="tiles",
                   help="tiles: per-slot weight-epoch gated GEMM tiles "
                        "(movement overlaps dispatch+GEMM); join: one "
                        "zero-SM landing gate before the l0 forward "
                        "(prices the tile gate)")
    p.add_argument("--s2_stale", choices=["0", "oracle", "rot"],
                   default="0",
                   help="PROBE: reset the resident placement every "
                        "iteration — 'oracle' (the setup solve; near-"
                        "optimal, trigger rarely fires) or 'rot' (rank-"
                        "rolled hosts: structurally suboptimal, adds "
                        "guaranteed) — so movement fires in EVERY timed "
                        "iteration (worst-case overlap; with the weight "
                        "probe this wire-audits WPM per rule 6c)")
    p.add_argument("--s2_force_trigger", type=int, default=0,
                   help="GATE aid: trigger whenever the solve yields any "
                        "adds, ignoring the gain threshold")
    p.add_argument("--s2_wprobe", type=int, default=0,
                   help="re-randomize moved experts' home weights per "
                        "trigger epoch (deterministic per (epoch, "
                        "expert)); the reference follows — a stale slot "
                        "fails allclose on exactly the moved rows")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


_W_CACHE = {}


def gen_expert_w1(e: int, ffn: int, H: int, dtype):
    key = ("w1", int(e))
    if key not in _W_CACHE:
        g = torch.Generator().manual_seed(10007 + int(e))
        _W_CACHE[key] = ((torch.rand((ffn, H), generator=g) * 0.02)
                         - 0.01).to(dtype)
    return _W_CACHE[key]


def gen_expert_w2(e: int, ffn: int, H: int, dtype):
    key = ("w2", int(e))
    if key not in _W_CACHE:
        g = torch.Generator().manual_seed(20011 + int(e))
        _W_CACHE[key] = ((torch.rand((H, ffn), generator=g) * 0.02)
                         - 0.01).to(dtype)
    return _W_CACHE[key]


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
    # setup trace (2026-08-26 r2 16n hang RCA): the hung gate cell died
    # with ZERO bytes on every rank log — nothing localizes a pre-print
    # wedge (NCCL rendezvous / nvshmem init / first collectives).
    # Rank-tagged milestone prints, default ON for gate cells
    # (--check_iters — the validation lane, where perturbed timing is
    # already accepted) and via FLUX_OURS_SETUP_TRACE=1 elsewhere. Perf
    # cells are byte-unchanged by default.
    _strace_on = bool(args.check_iters) or bool(int(os.environ.get(
        "FLUX_OURS_SETUP_TRACE", "0")))

    def _st(tag):
        if _strace_on:
            print(f"[setup-trace] r{rank} {tag}", flush=True)
    _st("enter (dist env up)")
    assert_node_major_ranks()
    _st("node-major ok")
    init_ep_group(DIST_ENV.WORLD_SIZE)
    flux.init_flux_shm(TP_GROUP)
    torch.cuda.synchronize()
    _st("flux shm init ok")

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
        R_red=args.redundant_per_rank, locality_aware=False, interleave=True,
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
    _st("placement solved")
    pblob = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
             "hosts": hosts_pll, "planner": plfast.stats_host(pf_solve)}
    plan = build_nodeaware_plan(cfg, tpe, pblob)
    _drift = plfast.drift_ppm(
        plfast.demand_hist(tk_dev, L, args.G),
        plfast.demand_hist(tk_solve, L, args.G))

    # ---- scenario 2: ALSO solve the batch (adoption-target) placement at
    #      setup — s2 sizing must cover BOTH the resident (oracle) and the
    #      adopted placements (the stale probe oscillates between exactly
    #      these two; the runtime warm solve lands near the cold batch
    #      solve, covered by the drift cushions) ----
    plan_batch = None
    if args.scenario == "s2":
        pf_solve_b = plfast.build_placement_fast(tk_dev, L, cfg.nlp,
                                                 args.G, **pf_cfg)
        hosts_b = plfast.finalize_hosts(pf_solve_b, W, L, cfg.nlp,
                                        method="snake")
        pblob_b = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
                   "hosts": hosts_b,
                   "planner": plfast.stats_host(pf_solve_b)}
        plan_batch = build_nodeaware_plan(cfg, tpe, pblob_b)

    # ---- setup reference route (deterministic torch; sizes buffers,
    #      binds the final correctness iteration) ----
    phys_ref, pll_aux = loccap_route_sl(
        topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
        cfg.nlp, L, args.eps, return_tables=True)
    plan.phys_override = phys_ref
    pll_bounds = loccap_sl_bounds(pll_aux, W, args.pll_f_cap)
    args.pll_f_cap = pll_bounds["f_cap"]
    _st("reference route + bounds ok")
    if plan_batch is not None:
        phys_ref_b, pll_aux_b = loccap_route_sl(
            topk_all.long().cpu(), plan_batch.p2l, plan_batch.l2p,
            plan_batch.lcnts, cfg.nlp, L, args.eps, return_tables=True)
        # r2 fix (2026-08-26): the batch/adopted placement's forced
        # geometry can exceed the resident-derived f_cap (kstats[2]
        # ticket overflow at i0 under capacity sizing + replica headroom;
        # exact no-op whenever the resident f_cap already dominates) —
        # auto-derive the batch bounds' own f_cap and take the max.
        bounds_b = loccap_sl_bounds(pll_aux_b, W, -1)
        args.pll_f_cap = max(args.pll_f_cap, bounds_b["f_cap"])
        for k in ("recv_cap", "pair_cap"):
            pll_bounds[k] = max(pll_bounds[k], bounds_b[k])
        # s2 provable recv ceiling (2026-08-26, layer-2 of the r2 RCA —
        # K2 4n gate: adopted-placement recv 7086 > envelope 6212, l1
        # send-panel FLUX_CHECK): reference-derived recv bounds only cover
        # the resident/batch placements, but the runtime warm solve adopts
        # OTHER placements and (with the f_cap escalate-and-reroute)
        # forced admission is uncapped. Placement-INDEPENDENT bound: a
        # rank hosts <= nlp slots, and no expert can route more entries
        # to one rank than its total demand — so per-rank recv <= sum of
        # the nlp hottest experts' demands. Exact-cheap from the batch
        # histogram; recv-scaled caps (l0_recv, RS send panel) already
        # take max(..., recv_cap) downstream.
        _d_glob = torch.bincount(topk_all.reshape(-1).long(),
                                 minlength=args.G)
        provable_recv = int(_d_glob.topk(min(cfg.nlp, args.G))
                            .values.sum())
        pll_bounds["recv_cap"] = max(pll_bounds["recv_cap"],
                                     provable_recv)

    gpe = cfg.nlp + 1
    E_virt = W * gpe
    _pr = phys_ref.view(ntokens, args.topk).long()
    vce_ref = (_pr // cfg.nlp) * gpe + 1 + _pr % cfg.nlp  # pad-FIRST
    sp_ref, sc_ref, sps_ref, uc_ref = python_meta_from_vce(
        vce_ref.cuda(), W, S, gpe, nn, L)
    # s2: the batch-placement twin meta joins every sizing max below
    sps_all, uc_all = [sps_ref], [uc_ref]
    if plan_batch is not None:
        _prb = phys_ref_b.view(ntokens, args.topk).long()
        vce_ref_b = (_prb // cfg.nlp) * gpe + 1 + _prb % cfg.nlp
        _, _, sps_b, uc_b = python_meta_from_vce(
            vce_ref_b.cuda(), W, S, gpe, nn, L)
        sps_all.append(sps_b)
        uc_all.append(uc_b)

    # kernel-drift cushion (handoff 17 / llc demand-sizing lineage)
    fp_slack = int(pll_aux["forced_pair"].sum(0).max())
    if plan_batch is not None:
        fp_slack = max(fp_slack, int(pll_aux_b["forced_pair"].sum(0).max()))
    cushion = fp_slack + 8 * W

    # recv rows this rank computes (dispatch recv == combine send)
    recv_real = max(int(s[:, rank * gpe:(rank + 1) * gpe].sum())
                    for s in sps_all)
    if args.sizing == "demand" and args.scenario == "s1":
        recv_cap = min(pll_bounds["recv_cap"], recv_real + cushion)
    else:
        # s2 always sizes at the provable caps (placement varies at
        # runtime; demand sizing derives from one realized placement)
        recv_cap = pll_bounds["recv_cap"]

    # l0 caps: EXACT lb_union demands (union recv regions + chunk-bound
    # stage/relay — parity-tested formulas, moonep virtual-space twin) on
    # the reference routing + drift cushions. The old hand-rolled bound
    # (U.sum total for relay/stage) over-sized by ~L and still
    # under-covered nothing — 8n b64 heap failure class, fixed 2026-08-25.
    from types import SimpleNamespace
    from flux.testing.moonep_fused_map import (
        required_a2av_knobs, required_a2av_rs_knobs)
    def _exact_knobs(sps_x, uc_x):
        m = SimpleNamespace(
            scatter_index=sc_ref.cpu().int(),
            splits=sps_x.cpu().sum(0).int(),
            splits_per_source=sps_x.cpu(),
            a2av_unique_counts=uc_x.cpu(),
            m_per_rank=sps_x.cpu().long().view(W, W, gpe).sum(2).sum(0),
        )
        return (required_a2av_knobs(m, W, L),
                required_a2av_rs_knobs(m, W, L))

    knob_pairs = [_exact_knobs(s, u) for s, u in zip(sps_all, uc_all)]
    l0_exact = {k: max(int(kp[0][k]) for kp in knob_pairs)
                for k in knob_pairs[0][0]}
    rs_exact = {k: max(int(kp[1][k]) for kp in knob_pairs)
                for k in knob_pairs[0][1]}
    # r2 hardening (2026-08-26 RCA): with replica headroom the relaxed
    # kernel's forced traffic redistributes across instance ranks in ways
    # the reference's per-L split can understate; 16n r2 is the untested
    # forced-heavy regime (forced_frac 0.10-0.13 at qwen b8). Use the
    # UNDIVIDED fp_slack for the stage/relay cushions when R_red > 0
    # (tens of MB; exact no-op at --redundant_per_rank 0).
    cushion_sr = ((fp_slack + 8 * W) if args.redundant_per_rank > 0
                  else (fp_slack + L - 1) // L + 8 * W)
    l0_recv_rows = max(
        int(l0_exact["FLUX_A2AV_MAX_RECV_NTOKENS"]) + cushion, recv_cap)
    stage_rows = int(l0_exact["FLUX_A2AV_MAX_STAGE_NTOKENS"]) + cushion_sr
    relay_rows = int(l0_exact["FLUX_A2AV_MAX_RELAY_NTOKENS"]) + cushion_sr
    # collective max: never let one rank under-size vs a peer's view
    t_red = torch.tensor([l0_recv_rows, recv_cap, stage_rows, relay_rows],
                         dtype=torch.int64, device="cuda")
    torch.distributed.all_reduce(t_red,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=TP_GROUP)
    l0_recv_rows, recv_cap = int(t_red[0]), int(t_red[1])
    stage_rows, relay_rows = int(t_red[2]), int(t_red[3])

    os.environ["FLUX_A2AV_MAX_RECV_NTOKENS"] = str(int(l0_recv_rows))
    os.environ["FLUX_A2AV_MAX_STAGE_NTOKENS"] = str(int(stage_rows))
    os.environ["FLUX_A2AV_MAX_RELAY_NTOKENS"] = str(int(relay_rows))
    _st("sizing collective ok")

    # l1 (combine) capacity: EXACT demands (placement-maxed under s2) +
    # drift cushion
    if args.scenario == "s2" and nn > 1:
        # provable per-dest-population ceilings (2026-08-26 r2 RCA layer
        # 3, K2 4n gate: conv 4641 > 4199): reference-derived RS panels
        # only cover the resident/batch placements; adopted placements
        # redistribute combine sends arbitrarily. Column sums are FIXED
        # (every owner receives exactly cpr = S*K rows), so per panel:
        #   conv(n2, dl)  = node n2's rows for remote lane-dl owners
        #                 <= (nn-1) * S * K   (those owners' whole cpr)
        #   stage(gn, gl) = gateway staging of its node's remote sends
        #                 <= (nn-1) * S * K   (same population)
        #   wire(n2, dl)  = UNIQUE rows per (owner, source-node)
        #                 <= (nn-1) * S       (<= 1 per owned token)
        # send already inherits recv_cap (provable top-nlp ceiling).
        _rs_ceil = {
            "FLUX_A2AV_RS_MAX_CONV_ROWS": (nn - 1) * S * args.topk,
            "FLUX_A2AV_RS_MAX_STAGE_ROWS": (nn - 1) * S * args.topk,
            "FLUX_A2AV_RS_MAX_WIRE_ROWS": (nn - 1) * S,
        }
        for k, v in _rs_ceil.items():
            rs_exact[k] = max(int(rs_exact[k]), int(v))
    for k, v in rs_exact.items():
        os.environ[k] = str(int(v) + cushion)
    # send panel additionally bounds the fused-pack capacity check: it must
    # cover the RELAXED per-iteration recv (== combine send) worst case
    os.environ["FLUX_A2AV_RS_MAX_SEND_ROWS"] = str(
        max(int(rs_exact["FLUX_A2AV_RS_MAX_SEND_ROWS"]) + cushion, recv_cap))

    if rank == 0:
        print(f"OURS sizing({args.sizing},{args.scenario}): recv_cap "
              f"{recv_cap} (real {recv_real}), l0_recv {l0_recv_rows}, "
              f"stage {stage_rows}, relay {relay_rows}, "
              f"cushion {cushion} (fp_slack {fp_slack})"
              + (f", provable_recv {pll_bounds['recv_cap']}"
                 if args.scenario == "s2" else ""))

    # ---- runner + planner (+ s2 movement lane) ----
    runner = OursRunner(
        TP_GROUP, EP_GROUP, DIST_ENV.NNODES, L, cfg, args.ffn_hidden_size,
        input_dtype, l0_recv_rows, sm_margin=args.sm_margin,
        plan_overlap=bool(args.plan_overlap))
    _st("runner (fused ops) constructed")
    lane = None
    store = None
    if args.scenario == "s1":
        w1v, w2v = build_slot_weights(plan.p2l, rank, cfg.nlp, gpe,
                                     args.ffn_hidden_size, args.H,
                                     input_dtype)
        runner.set_weights(w1v, w2v)
    else:
        from flux.testing.ours_s2 import OursMovementLane
        store = plfast.PlacementStore(pf_solve, W, L, cfg.nlp,
                                      hosts=hosts_pll)
        lane = OursMovementLane(
            TP_GROUP, rank, W, L, cfg, args.ffn_hidden_size, args.H,
            input_dtype,
            gen_w1=lambda e: gen_expert_w1(e, args.ffn_hidden_size,
                                           args.H, input_dtype),
            gen_w2=lambda e: gen_expert_w2(e, args.ffn_hidden_size,
                                           args.H, input_dtype),
            gain_threshold_ppm=args.place_gain_threshold_ppm,
            weight_shard=("on" if args.weight_shard == "on" else "off"))
        lane.fill_slots_local(plan.p2l)
        w1v, w2v = lane.gemm_weights()
        runner.set_weights(w1v, w2v)   # views SHARE the WPM symmetric buf
    planner = OursIterPlanner(plan, rank, torch.device("cuda"), topk_all,
                              probs_all_setup, L, args.eps, args.pll_f_cap,
                              TP_GROUP)
    # r2/s2 f_cap contract fix (handoff 22 §4): runtime-adopted placements
    # can exceed any setup-derived forced budget — enable the planner's
    # local escalate-and-reroute (kstats[2] breach -> 4x then uncapped).
    planner.f_cap_retry = (args.scenario == "s2")


    # s2 machinery: oracle snapshots for the stale probe + the weight probe
    if args.scenario == "s2":
        from flux.testing.loccap_semantics import plan_tensors_from_hosts
        if args.s2_stale == "rot":
            # rank-rolled hosts: every expert's instances shift one rank —
            # crosses node boundaries, structurally suboptimal, so the
            # warm solve always finds adds (movement fires per iteration)
            hosts_rot = [sorted((r + 1) % W for r in hs)
                         for hs in hosts_pll]
            p2l_r, l2p_r, lcnts_r = plan_tensors_from_hosts(
                hosts_rot, W, cfg.nlp)
            ion_r = plfast.hosts_to_ion(hosts_rot, W, L,
                                        device=store.ion.device)
            primary_r = ion_r.long().argmax(dim=1)
            oracle_snap = {
                "p2l": p2l_r, "l2p": l2p_r, "lcnts": lcnts_r,
                "primary": primary_r.to(store.primary.dtype),
                "ion": ion_r,
                "hist": store.hist.clone(),
                "load_e": store.load_e.clone(),
            }
        else:
            oracle_snap = {
                "p2l": plan.p2l.clone(), "l2p": plan.l2p.clone(),
                "lcnts": plan.lcnts.clone(),
                "primary": store.primary.clone(),
                "ion": store.ion.clone(),
                "hist": store.hist.clone(),
                "load_e": store.load_e.clone(),
            }
        tk_dev_solve = tk_dev  # runtime warm solves observe the BATCH
        wprobe_version = [0]

        def wprobe_cb(changed_experts, changed_slots):
            # rule-6c weight payload probe: re-randomize home rows of every
            # moved expert ALL of whose instances move this trigger; the
            # reference cache follows (replicated determinism).
            if not args.s2_wprobe:
                return
            wprobe_version[0] += 1
            ver = wprobe_version[0]
            new_p2l = plan.p2l.long()
            # probe-scope cap: 8 experts/trigger — the wire audit needs
            # path coverage, not volume; unbounded regen OOM-killed K2
            # hosts (4 ranks x ~200 experts x 56 MB churn per iteration)
            budget = 8
            for e in changed_experts:
                if budget <= 0:
                    break
                inst = (new_p2l == e).nonzero(as_tuple=True)[0]
                # skip experts with unmoved live instances (their old slots
                # would legitimately hold pre-probe bytes)
                if not all(int(i) in changed_slots for i in inst.tolist()):
                    continue
                g1 = torch.Generator().manual_seed(
                    10007 + e + 1000003 * ver)
                _W_CACHE[("w1", e)] = ((torch.rand(
                    (args.ffn_hidden_size, args.H), generator=g1) * 0.02)
                    - 0.01).to(input_dtype)
                g2 = torch.Generator().manual_seed(
                    20011 + e + 1000003 * ver)
                _W_CACHE[("w2", e)] = ((torch.rand(
                    (args.H, args.ffn_hidden_size), generator=g2) * 0.02)
                    - 0.01).to(input_dtype)
                if e // lane.epn == rank:
                    lane.op_w1.weight_home()[e % lane.epn].copy_(
                        _W_CACHE[("w1", e)].cuda())
                    lane.op_w2.weight_home()[e % lane.epn].copy_(
                        _W_CACHE[("w2", e)].cuda())
                budget -= 1
            torch.cuda.synchronize()

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
    _st("setup audit ok")

    RECORDER.emit_info(
        ours_fusion="slipstream_v2",
        ours_plan_overlap=int(bool(args.plan_overlap)),
        # plan-lane cost knobs (ours.py module header; all default OFF)
        ours_plan_xchg_narrow=planner.xchg_narrow,
        ours_plan_prealloc=int(planner.plan_prealloc),
        ours_plan_graph=int(planner.plan_graph),
        ours_plan_scale_graph=int(runner.scale_graph),
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

    # ---- place-solve CUDA-graph capture (handoff 12 containment,
    #      recovered 2026-08-25): the warm solve is ~200 small kernel
    #      launches; the placefast dynamic lane graphed it (default ON)
    #      and that is how place stayed few-ms. Capture once on the
    #      persistent seed buffers (store.primary/store.ion — solver
    #      clones internally, replay re-reads current contents); output
    #      tensors are reused per replay and consumed synchronously by
    #      adopt before the next replay. Eager fallback on any capture
    #      failure, epic-lane style. ----
    _solve_graph = None
    _solve_out = None
    if lane is not None:
        _always0 = args.place_gain_threshold_ppm == 0
        _solve_kw = dict(passes_a=2, passes_b=1, repair_passes=1,
                         seed="warm", seed_primary=store.primary,
                         seed_inst_nodes=store.ion,
                         keep_bonus=(args.place_keep_bonus
                                     if args.place_keep_bonus >= 0
                                     else (0 if _always0 else 90090)))

        def _solve_eager():
            return plfast.build_placement_fast(
                tk_dev_solve, L, cfg.nlp, args.G, **_solve_kw)
        if bool(int(os.environ.get("FLUX_OURS_PLACE_GRAPH", "1"))):
            try:
                for _ in range(2):
                    _solve_eager()
                torch.cuda.synchronize()
                _solve_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(_solve_graph):
                    _solve_out = _solve_eager()
                _solve_graph.replay()
                torch.cuda.synchronize()
                if rank == 0:
                    print("[s2] place-solve graph captured", flush=True)
            except Exception as _e:  # noqa: BLE001 — eager fallback
                _solve_graph = None
                if rank == 0:
                    print(f"[s2] place-solve graph capture failed "
                          f"({type(_e).__name__}: {_e}); eager", flush=True)

    # ---- timed loop ----
    total_iters = args.warmup_iters + args.iters
    ev = lambda: [torch.cuda.Event(enable_timing=True)
                  for _ in range(total_iters)]
    iter_start, plan_comm_end, place_end, plan_end = ev(), ev(), ev(), ev()
    e2e_start, l0_end, act_end, e2e_end = ev(), ev(), ev(), ev()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    move_stats = []   # (trigger, moves, bytes, movement_ms, gain_ppm)
    out = None
    for i in range(total_iters):
        runner.prep()
        probe.step(i)
        if lane is not None and args.s2_stale != "0":
            # PROBE: reset the resident placement/tables to the oracle
            # solve OUTSIDE the window — trigger+movement re-fire every
            # timed iteration (worst-case movement overlap)
            plan.p2l = oracle_snap["p2l"].clone()
            plan.l2p = oracle_snap["l2p"].clone()
            plan.lcnts = oracle_snap["lcnts"].clone()
            store.primary.copy_(oracle_snap["primary"])
            store.ion.copy_(oracle_snap["ion"])
            store.hist.copy_(oracle_snap["hist"])
            store.load_e.copy_(oracle_snap["load_e"])
            lane.resident_p2l = oracle_snap["p2l"].long().clone()
            planner.refresh_placement()
            torch.cuda.synchronize()
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
            if lane is not None:
                # PLACE lane (timed): drift prefilter -> warm solve ->
                # decision -> trigger -> adoption + movement issue. The
                # movement itself runs on lane.w_stream/side_stream and
                # overlaps everything up to the weight-gated tiles.
                _pt0 = time.perf_counter()
                hist_now = d_gather_buf.long().view(nn, L, args.G).sum(1)
                drift = plfast.drift_ppm(hist_now, store.hist)
                _pt1 = time.perf_counter()
                always = args.place_gain_threshold_ppm == 0
                if always or drift >= args.place_drift_prefilter_ppm:
                    # USER TEST REGIME (threshold 0 => `always`): EVERY
                    # iteration migrates. The solve stays the CHEAP warm
                    # path but with keep_bonus=0 — no resident-replica
                    # pull, so Stage B re-derives replicas fresh and the
                    # diff vs the (stale-reset) resident is real movement
                    # (4n repro: rot->warm-kb0 = 85 adds, 263k ppm; the
                    # earlier 0-adds paralysis was keep_bonus=90090).
                    # Non-always keeps the production warm defaults.
                    if _solve_graph is not None:
                        _solve_graph.replay()
                        res_new = _solve_out
                    else:
                        res_new = _solve_eager()
                    _pt2 = time.perf_counter()
                    if always:
                        # always-migrate regime: the decision is foregone
                        # (trigger unconditional), so skip the 8-round
                        # cover decision entirely — it was ~half the K2
                        # solve+decision bracket. Placement identical;
                        # moves_add carries the -1 sentinel (real move
                        # counts come from lane.apply_moves). gain_ppm
                        # MUST be 0, not -1: apply_moves gates movement
                        # on gain_ppm >= threshold (0 here) — a -1
                        # sentinel silently killed ALL movement
                        # (caught 2026-08-25: ours_s2_moves==0).
                        verdict = {"trigger": 1, "gain_ppm": 0,
                                   "moves_add": -1, "moves_remove": -1}
                    else:
                        verdict = plfast.place_decision_fast(
                            tk_dev_solve, store.ion, res_new, L,
                            gain_threshold_ppm=max(
                                args.place_gain_threshold_ppm, 1),
                            mode="cover")
                    if args.s2_force_trigger and verdict["moves_add"] > 0:
                        verdict["trigger"] = 1
                    _pt3 = time.perf_counter()
                    # identity early-out (2026-08-25, remove no-op
                    # machinery): at the fixed point (stable traffic,
                    # always-solve) the warm solve returns the resident
                    # placement unchanged — adopt/finalize/plan-tensors/
                    # apply_moves would all be no-ops costing ~1.5-1.7 ms.
                    # primary+inst_nodes equality implies the whole slot
                    # table is identical (finalize is deterministic in
                    # them). One small host sync; stale/drift cells have
                    # nonzero diffs and take the full path.
                    # KNOB-GATED (2026-08-25, default OFF): with the
                    # early-out active, every stale-regime strict gate
                    # today failed (torn row / spin-hang at iter 3-4,
                    # 3/3 with vs 3/3 green without) — its mid-lane host
                    # sync shifts movement-issue timing into a latent
                    # race (suspect: shard/gateway mid-iteration
                    # ordering). Perf win (place 2.6-3.1 -> 1.6-2.0) is
                    # real and quiet-path-validated; re-enable via
                    # FLUX_OURS_PLACE_EARLYOUT=1 once the race is
                    # root-caused (next-session item).
                    _same = (bool(int(os.environ.get(
                                 "FLUX_OURS_PLACE_EARLYOUT", "0")))
                             and verdict["trigger"]
                             and torch.equal(res_new["primary"],
                                             store.primary)
                             and torch.equal(res_new["inst_nodes"],
                                             store.ion))
                    if _same:
                        lane.moves_this_iter = 0
                        lane.move_bytes_this_iter = 0
                        lane.trigger_fired = 0
                    if verdict["trigger"] and not _same:
                        hosts_new = store.adopt(res_new, finalize=True)
                        _pt4 = time.perf_counter()
                        p2l_n, l2p_n, lcnts_n = plan_tensors_from_hosts(
                            hosts_new, W, cfg.nlp)
                        _pt5 = time.perf_counter()
                        plan.p2l, plan.l2p, plan.lcnts = (
                            p2l_n, l2p_n, lcnts_n)
                        planner.refresh_placement()
                        lane.apply_moves(
                            p2l_n.long(), verdict["gain_ppm"],
                            wprobe_cb=(wprobe_cb if args.s2_wprobe
                                       else None))
                    move_stats.append((
                        int(verdict["trigger"]), lane.moves_this_iter,
                        lane.move_bytes_this_iter, verdict["gain_ppm"]))
                    if rank == 0 and args.check_iters:
                        # place-lane host-side split (gate mode only):
                        # drift | solve+decision | adopt | tensors | moves
                        _pt6 = time.perf_counter()
                        _sp = [_pt1 - _pt0, _pt3 - _pt1]
                        if verdict["trigger"] and not _same:
                            _sp += [_pt4 - _pt3, _pt5 - _pt4, _pt6 - _pt5]
                        print("[s2-split] iter %d: %s ms" %
                              (i, " ".join("%.2f" % (x * 1e3)
                                           for x in _sp)))
                        print(f"[s2] iter {i}: drift {drift} solve gain "
                              f"{verdict['gain_ppm']} adds "
                              f"{verdict['moves_add']} trigger "
                              f"{verdict['trigger']} moved "
                              f"{lane.moves_this_iter}")
                else:
                    move_stats.append((0, 0, 0, int(drift)))
                    if rank == 0 and args.check_iters:
                        print(f"[s2] iter {i}: drift {drift} < prefilter"
                              f" — quiet")
            place_end[i].record()
            _hb = (args.check_iters and lane is not None)

            def _hbp(tag):
                if _hb:
                    print(f"[hb-s2] r{rank} i{i} {tag}", flush=True)
            _hbp("plan")
            ip = planner.derive(d_gather_buf)
            runner.plan_meta(ip)
            plan_end[i].record()
            e2e_start[i].record()
            runner.issue_combine_meta(ip)
            gate_kw = None
            if lane is not None:
                if args.s2_join == "join":
                    lane.op_w1.join()
                else:
                    gate_kw = lane.gate_kwargs()
            _hbp("l0")
            l0_out = runner.l0_forward(inputs_shard, gate_kwargs=gate_kw)
            l0_end[i].record()
            if lane is not None:
                # FLUX_OURS_S2_W2_LATE: l1 weight pushes enqueue AFTER the
                # dispatch legs (no-op when the knob is off / no trigger)
                lane.issue_w2_late()
            intermediate = torch.nn.functional.gelu(l0_out)
            act_end[i].record()
            if lane is not None:
                lane.join_w2()
            _hbp("l1")
            out = runner.l1_forward(intermediate)
            e2e_end[i].record()
            if lane is not None:
                # end-of-iteration weight-signal drain (K2-4n stale hang
                # fix): after e2e_end (untimed gap) — see ours_s2.join_w1.
                lane.join_w1()
            _hbp("issued")
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
                                  "plan_comm_ms", "place_ms", "plan_ms",
                                  "total_ms")}
    for i in range(total_iters):
        e2e_end[i].synchronize()
        if i >= args.warmup_iters:
            iter_times["plan_comm_ms"].append(
                iter_start[i].elapsed_time(plan_comm_end[i]))
            iter_times["place_ms"].append(
                plan_comm_end[i].elapsed_time(place_end[i]))
            iter_times["plan_ms"].append(
                place_end[i].elapsed_time(plan_end[i]))
            iter_times["e2e_ms"].append(
                e2e_start[i].elapsed_time(e2e_end[i]))
            iter_times["l0_ms"].append(e2e_start[i].elapsed_time(l0_end[i]))
            iter_times["act_ms"].append(l0_end[i].elapsed_time(act_end[i]))
            iter_times["l1_ms"].append(act_end[i].elapsed_time(e2e_end[i]))
            iter_times["total_ms"].append(
                iter_start[i].elapsed_time(e2e_end[i]))
    if isolated:
        iter_times["iso_sync_ms"] = iso_sync_times[args.warmup_iters:]
    if lane is not None and move_stats:
        timed_ms = move_stats[args.warmup_iters:]
        RECORDER.emit_info(
            ours_s2_triggers=sum(m[0] for m in timed_ms),
            ours_s2_moves=sum(m[1] for m in timed_ms),
            ours_s2_move_bytes=sum(m[2] for m in timed_ms),
            ours_s2_last_gain_ppm=timed_ms[-1][3] if timed_ms else 0,
            ours_s2_join=args.s2_join,
            ours_s2_stale=args.s2_stale,
            ours_s2_wprobe=int(args.s2_wprobe),
            ours_s2_weight_shard=args.weight_shard,
            ours_s2_sched_moved_last=int(lane.sched_moved_last),
            ours_s2_w2_late=int(lane.w2_late),
            ours_s2_fcap_retries=int(planner.f_cap_retries_total),
            ours_s2_fcap_final=int(planner.f_cap_current),
        )

    def fmt(times):
        return ", ".join(f"{k[:-3]} {sum(v) / max(len(v), 1):.3f} ms"
                         for k, v in times.items())

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"ours #{rank}: {fmt(iter_times)}"))
    RECORDER.emit_iters("ours", iter_times)

    # ---- final deterministic iteration + correctness ----
    if lane is not None:
        # s2: the setup reference route was solved on the ORACLE placement;
        # re-solve deterministically on the CURRENT (adopted) tables so the
        # final iteration matches the slots' weights (untimed)
        phys_fin, _ = loccap_route_sl(
            topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
            cfg.nlp, L, args.eps, return_tables=True)
        plan.phys_override = phys_fin
    ip_ref = planner.derive_reference()
    runner.prep()
    runner.plan_meta(ip_ref)
    runner.issue_combine_meta(ip_ref)
    l0_out = runner.l0_forward(inputs_shard)
    intermediate = torch.nn.functional.gelu(l0_out)
    out = runner.l1_forward(intermediate)
    torch.cuda.synchronize()

    import hashlib
    sha = hashlib.sha256(
        out.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16]
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
