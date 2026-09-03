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
from flux.testing import ours_swap as oswap_rt
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
    p.add_argument("--routing_sched_files", type=str, default="",
                   help="ABLATION-ONLY topic-schedule harness (2026-09-02):"
                        " comma-separated extra routing files (same T/W/"
                        "topk as --routing_file; entry 0 MUST equal"
                        " --routing_file). Iteration i runs topic"
                        " ((i - last) // dwell) mod N so the LAST timed"
                        " iteration is topic 0 (correctness reference);"
                        " placement/tables carry over between topics (no"
                        " reset). Sizing envelopes are maxed over topics.")
    p.add_argument("--routing_dwell", type=int, default=1,
                   help="topic-schedule dwell: consecutive iterations per"
                        " topic (1 = a new topic every iteration).")
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
    p.add_argument("--route_global", type=int, default=0,
                   help="1: route-global restructure (handoff 26 §4) — ONE "
                        "topk+probs allgather replaces the d-allgather + "
                        "routed-decisions allgather; every rank recomputes "
                        "all ranks' assignment via the deterministic quota "
                        "route (route_global_quota; the relaxed kernel's "
                        "ticket pairing cannot be replicated cross-rank). "
                        "s1 only; f_cap runs uncapped (deterministic "
                        "realized == setup sizing).")
    p.add_argument("--pll_f_cap", type=int, default=-1,
                   help="-1 = auto from the reference tables")
    p.add_argument("--sizing", choices=["demand", "capacity"],
                   default="demand",
                   help="demand: realized reference + drift cushions "
                        "(default); capacity: provable caps everywhere")
    p.add_argument("--plan_overlap", type=int, default=0,
                   help="0 (default): combine meta inside the plan bracket; "
                        "1: pre-l0 side stream (8/25 structure — measured "
                        "to relabel plan_ms into l0_ms); 2: LATE issue "
                        "after the l0 enqueue (host meta work runs under "
                        "the executing GEMM, kernels on sm_margin SMs).")
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
    p.add_argument("--place_solver", choices=["pll", "pv2"], default="pll",
                   help="pll: PLACE-lambda FAST (warm-solve dynamic lane); "
                        "pv2: stateless node-aware greedy placement "
                        "(placement_v2 — marginals-only host solve, no "
                        "warm state/CUDA graph; runtime adoption == the "
                        "setup batch solve by purity, restoring the "
                        "handoff-22 sizing premise). Branch pv2 A/B.")
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
    p.add_argument("--swap_xport", choices=("nccl", "p2p"), default="nccl",
                   help="swap exchange transport: nccl = torch P2P ops in"
                        " the node subgroup (2026-08-27 baseline); p2p ="
                        " symmetric-heap staging + peer views, cudaMemcpy"
                        " over NVLink + zero-SM landed-signal wait")
    p.add_argument("--swap_issue", choices=("early", "late", "split"),
                   default="early",
                   help="where the exchange is enqueued: early = in the"
                        " place bracket right after the decision; late ="
                        " after the fused l0 forward is enqueued (moved"
                        " slot's tiles spin until landing); split = w1"
                        " early, w2 late")
    p.add_argument("--swap_overlap", type=int, default=1,
                   help="ABLATION-ONLY knob (2026-09-01). 1 (default) ="
                        " canon: the exchange rides the movement stream"
                        " and dispatch launches immediately, gated per"
                        " slot (overlapped expert dispatch). 0 = the"
                        " current stream WAITS for the exchange to land"
                        " before anything downstream is enqueued — swap"
                        " completes first, THEN dispatch (sequential;"
                        " requires --swap_issue early). The exposed"
                        " exchange lands in the place bracket: total_ms"
                        " sees it, e2e does not.")
    p.add_argument("--swap_reset", choices=("off", "every", "postwarmup"),
                   default="off",
                   help="ABLATION-ONLY: restore the ORACLE-BASIS placement"
                        " (tables + physical slot weights) in the untimed"
                        " gap — 'every' before each timed iteration (every"
                        " timed iter measures the full drift-event"
                        " response), 'postwarmup' once at the warmup ->"
                        " timed transition, 'off' (default) never.")
    p.add_argument("--swap_reset_period", type=int, default=1,
                   help="ABLATION-ONLY (2026-09-02 cycling ablation): with"
                        " --swap_reset every, reset only before timed"
                        " iterations whose index (from the first timed"
                        " one) is a multiple of this period — the dwell-N"
                        " topic-schedule proxy (1 event + N-1 quiet iters).")
    p.add_argument("--swap_rounds", choices=("1", "all"), default="1",
                   help="ABLATION-ONLY: '1' (default) = canon single"
                        " greedy round per iteration (<=1 exchanged slot"
                        " per rank); 'all' = capped tau=1 orbit to"
                        " fixpoint at decision time, executed as the"
                        " COMPOSED multi-slot exchange in one phase"
                        " (p2p + issue early only).")
    p.add_argument("--swap_pair_moves", type=int, default=1,
                   help="ABLATION (2026-09-02): exchanges taken per hot/cold"
                        " pair per orbit pass (1 = one per re-paired round,"
                        " the 8/28 greedy); >1 collapses the numpy rounds.")
    p.add_argument("--swap_max_moves", type=int, default=8,
                   help="staging cap for --swap_rounds all: max exchanged"
                        " slots per rank per iteration (round-granular"
                        " orbit truncation; also sizes the symmetric"
                        " staging = cap*(w1+w2) per rank).")
    p.add_argument("--swap_tables", choices=("upload", "device"),
                   default="device",
                   help="how a fired swap reaches the planner's device"
                        " tables: upload = refresh_placement (3 blocking"
                        " H2D, 8.27); device = SwapTableSync (pinned"
                        " mirrors + 2 non-blocking copies, zero syncs)")
    p.add_argument("--s2_swap", type=int, default=0,
                   help="1: intra-node expert SWAP lane (EPIC §4.3 analog,"
                        " ours_swap.py) — per-iteration greedy pair+swap"
                        " inside each node, exchanged over NVLink on the"
                        " movement stream, NO cross-node migration (the"
                        " pv2 adoption lane is disabled). Decision is"
                        " timed in the place bracket (total_ms).")
    p.add_argument("--swap_tau_rows", type=int, default=512,
                   help="EPIC tau in rows: accept a swap only if it drops"
                        " the pair max load by at least this many rows"
                        " (movement-cost threshold). A huge value gives"
                        " the decide-but-never-swap comparator arm.")
    p.add_argument("--wire", choices=["fused", "direct", "dov"],
                   default="fused",
                   help="l0/l1 transport: fused = the canonical Slipstream"
                   " ops; direct = the eplb_l01 staged All2AllSingle wire"
                   " (transport ablation, s1 only — same plan lane);"
                   " dov = direct-overlap: the same direct one-hop wire"
                   " but l0 = flat-mode GemmGroupedV2AGScatterOp (ring-"
                   "order puts + tile-spin GEMM overlap), combine as"
                   " direct (s1 only)")
    p.add_argument("--dwire_pair_sizing", choices=["capacity", "demand"],
                   default="capacity",
                   help="direct-wire All2AllSingle staging width: capacity ="
                   " max(realized pair ref, provable SL pair cap) + cushion"
                   " (the 8/30 default — blows the 16G symmetric heap at 16n"
                   " b32/b64 in the ctor, both models); demand = realized"
                   " pair ref + cushion only (the eplb arm's own sizing"
                   " philosophy; the per-iteration pair_max assert keeps"
                   " overflow loud). Alloc-only knob: max_split is not read"
                   " by any timed path.")

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
    # ---- topic-schedule harness (ablation-only; empty list = legacy) ----
    sched_topk_all = []
    if args.routing_sched_files:
        assert not args.route_global, "schedule harness: route_global off"
        for _sf in args.routing_sched_files.split(","):
            _ce = load_routing_file(_sf, args.G, args.topk)
            assert _ce.shape == choosed_experts.shape, (
                f"schedule file {_sf}: shape {tuple(_ce.shape)} !="
                f" {tuple(choosed_experts.shape)}")
            sched_topk_all.append(_ce.reshape(W, S, args.topk).cpu().int())
        assert torch.equal(sched_topk_all[0], topk_all), (
            "schedule entry 0 must be the --routing_file topic")
        # capacity envelope: per-source-rank per-expert loads, elementwise
        # max over the topic set (plan sizing input)
        tpe = torch.stack([loads_from_topk(cfg, t) for t in sched_topk_all]
                          ).max(0).values
        if rank == 0:
            print(f"[sched] {len(sched_topk_all)} topics, dwell"
                  f" {args.routing_dwell}, tpe envelope max"
                  f" {int(tpe.sum(1).max())} rows/rank", flush=True)

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
    use_pv2 = args.place_solver == "pv2"
    pv2mod = None
    if use_pv2:
        from flux.testing import placement_v2 as pv2mod
    torch.cuda.synchronize()
    t_ps = time.perf_counter()
    if use_pv2:
        pf_solve = None
        pv2_res = pv2mod.pv2_solve(
            plfast.demand_hist(tk_solve, L, args.G).cpu(), L, cfg.nlp)
        hosts_pll = pv2mod.hosts_lists(pv2_res, args.G)
        planner_stats = dict(pv2_res["stats"])
    else:
        pf_solve = plfast.build_placement_fast(tk_solve, L, cfg.nlp, args.G,
                                              **pf_cfg)
        hosts_pll = plfast.finalize_hosts(pf_solve, W, L, cfg.nlp,
                                          method="snake")
        planner_stats = plfast.stats_host(pf_solve)
    torch.cuda.synchronize()
    place_solver_ms = (time.perf_counter() - t_ps) * 1e3
    _st("placement solved")
    pblob = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
             "hosts": hosts_pll, "planner": planner_stats}
    plan = build_nodeaware_plan(cfg, tpe, pblob)
    _drift = plfast.drift_ppm(
        plfast.demand_hist(tk_dev, L, args.G),
        plfast.demand_hist(tk_solve, L, args.G))

    # ---- scenario 2 sizing set (handoff 22 §4, extended 2026-08-27 on
    #      branch pv2): the caps must cover EVERY placement the runtime
    #      can route on — the resident (oracle), the cold batch solve,
    #      the stale-rot resident (routed on only if a trigger ever
    #      forgoes adoption — cheap insurance), and, for the STATEFUL
    #      pll solver, the runtime warm-solve orbit from the resident
    #      seed (the 8/26 gate failure: warm adoptions left the
    #      {resident, cold-batch} envelope). pv2 is a pure function of
    #      the demand histogram, so its runtime adoption IS the batch
    #      solve — no orbit exists. ----
    from flux.testing.loccap_semantics import plan_tensors_from_hosts
    s2_size_plans = []              # [(tag, p2l, l2p, lcnts)]
    hosts_rot = None
    rot_tensors = None
    if args.scenario == "s2":
        if use_pv2:
            pv2_res_b = pv2mod.pv2_solve(
                plfast.demand_hist(tk_dev, L, args.G).cpu(), L, cfg.nlp)
            hosts_b = pv2mod.hosts_lists(pv2_res_b, args.G)
        else:
            pf_solve_b = plfast.build_placement_fast(tk_dev, L, cfg.nlp,
                                                     args.G, **pf_cfg)
            hosts_b = plfast.finalize_hosts(pf_solve_b, W, L, cfg.nlp,
                                            method="snake")
        s2_size_plans.append(
            ("batch",) + plan_tensors_from_hosts(hosts_b, W, cfg.nlp))
        if args.s2_stale == "rot":
            hosts_rot = [sorted((r + 1) % W for r in hs)
                         for hs in hosts_pll]
            rot_tensors = plan_tensors_from_hosts(hosts_rot, W, cfg.nlp)
            s2_size_plans.append(("rot",) + rot_tensors)
        if args.s2_swap:
            # intra-node swap orbit (ours_swap.py): the runtime swap
            # sequence on the fixed per-cell demand is setup-computable
            # (pure function of (d, p2l)) — fold every reachable swapped
            # placement into the sizing envelope, pv2-purity style.
            assert use_pv2, "--s2_swap is defined on the pv2 arm"
            assert args.s2_stale == "0", "--s2_swap excludes stale probes"
            assert args.swap_overlap or args.swap_issue == "early", (
                "--swap_overlap 0 (sequential ablation) requires"
                " --swap_issue early")
            assert args.swap_rounds == "1" or (
                args.swap_xport == "p2p"
                and args.swap_issue == "early"), (
                "--swap_rounds all needs --swap_xport p2p and"
                " --swap_issue early")
            from flux.testing import ours_swap as oswap
            _load_g = torch.bincount(tk_dev.reshape(-1),
                                     minlength=args.G).cpu().long()
            _orbit = oswap.swap_orbit(
                _load_g, plan.p2l, plan.l2p, plan.lcnts, L, cfg.nlp,
                args.swap_tau_rows)
            _fold = _orbit
            if args.swap_tau_rows < 0:
                # FORCE pre-converge (2026-08-29): the timed iterations only
                # ever alternate inside each node's cycle; the placements
                # before every node has entered its cycle are a warmup
                # transient that exists only because the probe starts from
                # the batch solve. Adopt the first all-nodes-in-cycle
                # placement as the initial tables (slots are filled from
                # plan.p2l below) and fold only the placements the run will
                # visit after it (one global period, capped by the run
                # length). 16n b64 K2: the 8-deep global fold (f_cap 824 /
                # recv 83k vs batch 554 / 66k) overflowed the 16G heap.
                _orb, _T, _per = oswap.swap_orbit_nodes(
                    _load_g, plan.p2l, plan.l2p, plan.lcnts, L, cfg.nlp,
                    args.swap_tau_rows)
                if _T is not None:
                    _nvisit = min(_per, args.warmup_iters + args.iters + 1)
                    if _T >= 0:
                        plan.p2l = _orb[_T][0].clone()
                        plan.l2p = _orb[_T][1].clone()
                    _fold = _orb[_T + 1:_T + 1 + _nvisit]
                    if rank == 0:
                        print(f"[swap-orbit] force pre-converge: entry {_T}"
                              f" (of {len(_orb)}), period {_per}, folding"
                              f" {len(_fold)} placements", flush=True)
                elif rank == 0:
                    print(f"[swap-orbit] force: no per-node cycle within"
                          f" {len(_orb)} rounds — full fold", flush=True)
            for _oi, (_p2l_o, _l2p_o) in enumerate(_fold):
                s2_size_plans.append(
                    (f"swap{_oi}", _p2l_o, _l2p_o, plan.lcnts))
            # topic-schedule harness: fold each other topic's orbit from
            # the resident placement too (the run visits chained
            # placements; the per-topic orbits from the basis bound them)
            for _ti, _tk in enumerate(sched_topk_all[1:], start=1):
                _lg_t = torch.bincount(_tk.reshape(-1).long(),
                                       minlength=args.G).cpu().long()
                _orb_t = oswap.swap_orbit(
                    _lg_t, plan.p2l, plan.l2p, plan.lcnts, L, cfg.nlp,
                    max(args.swap_tau_rows, 1))
                for _oi, (_p2l_o, _l2p_o) in enumerate(_orb_t):
                    s2_size_plans.append(
                        (f"t{_ti}swap{_oi}", _p2l_o, _l2p_o, plan.lcnts))
        if not use_pv2:
            # warm-solve orbit, mirroring the runtime _solve_kw exactly
            # (seed = the resident the loop starts from / resets to;
            # keep_bonus follows the always/threshold regime). Under a
            # stale probe the resident resets every iteration, so only
            # depth 1 is reachable; quiet runs iterate to a fixed point
            # (bounded depth — in practice depth 1).
            _always0 = args.place_gain_threshold_ppm == 0
            _wk = dict(passes_a=2, passes_b=1, repair_passes=1,
                       seed="warm",
                       keep_bonus=(0 if _always0 else 90090))
            if args.s2_stale == "rot":
                _ion_seed = plfast.hosts_to_ion(hosts_rot, W, L).cuda()
                _pri_seed = _ion_seed.long().argmax(dim=1)
            else:
                _pri_seed = pf_solve["primary"]
                _ion_seed = pf_solve["inst_nodes"]
            _seen = {tuple(map(tuple, hosts_pll)),
                     tuple(map(tuple, hosts_b))}
            for _depth in range(3):
                _r_o = plfast.build_placement_fast(
                    tk_dev, L, cfg.nlp, args.G,
                    seed_primary=_pri_seed, seed_inst_nodes=_ion_seed,
                    **_wk)
                _hosts_o = plfast.finalize_hosts(_r_o, W, L, cfg.nlp,
                                                 method="snake")
                _key = tuple(map(tuple, _hosts_o))
                if _key in _seen:
                    break           # fixed point / already covered
                _seen.add(_key)
                s2_size_plans.append(
                    (f"orbit{_depth}",)
                    + plan_tensors_from_hosts(_hosts_o, W, cfg.nlp))
                if args.s2_stale != "0":
                    break           # resident resets: depth 1 reachable
                _pri_seed = _r_o["primary"]
                _ion_seed = _r_o["inst_nodes"]

    # ---- setup reference route (deterministic torch; sizes buffers,
    #      binds the final correctness iteration) ----
    phys_ref, pll_aux = loccap_route_sl(
        topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
        cfg.nlp, L, args.eps, return_tables=True)
    pll_bounds = loccap_sl_bounds(pll_aux, W, args.pll_f_cap)
    args.pll_f_cap = pll_bounds["f_cap"]
    for _ti, _tk in enumerate(sched_topk_all[1:], start=1):
        # topic-schedule harness: the resident placement's reference
        # route on every other topic widens the caps (max)
        _, _aux_t = loccap_route_sl(
            _tk.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
            cfg.nlp, L, args.eps, return_tables=True)
        _b_t = loccap_sl_bounds(_aux_t, W, -1)
        args.pll_f_cap = max(args.pll_f_cap, _b_t["f_cap"])
        for k in ("recv_cap", "pair_cap"):
            pll_bounds[k] = max(pll_bounds[k], _b_t[k])
        if rank == 0:
            print(f"[sched-sizing] topic {_ti}: f_cap {_b_t['f_cap']} recv"
                  f" {_b_t['recv_cap']} pair {_b_t['pair_cap']}", flush=True)
    if args.route_global:
        # route-global: the deterministic quota route IS both the sizing
        # reference AND the runtime routing (bitwise — same pure function
        # of the same fixed per-cell topk), so realized == reference
        # exactly and f_cap runs uncapped (kstats[2] identically 0). The
        # SL bounds above are kept only for their aux diagnostics.
        assert args.scenario == "s1", "route_global v1 is s1-only"
        from flux.testing.placelambda_gpu import (
            instance_tables_gpu, route_global_quota)
        _rg_tab_cpu = instance_tables_gpu(
            plan.l2p, plan.lcnts, cfg.nlp, L, R=W)
        phys_ref = route_global_quota(
            topk_all.long().cpu(), _rg_tab_cpu, cfg.nlp, L, args.eps,
            f_cap=0)
        args.pll_f_cap = 0
    plan.phys_override = phys_ref
    _st("reference route + bounds ok")
    # r2 fix (2026-08-26) generalized (2026-08-27, branch pv2): every
    # placement in the s2 sizing set gets its own reference route; the
    # f_cap / recv / pair caps are maxed over the whole set (kstats[2]
    # ticket overflow at i0 was the runtime-adopted placement leaving the
    # setup envelope; exact no-op whenever the resident caps dominate).
    s2_refs = []
    for _tag, _p2l_x, _l2p_x, _lc_x in s2_size_plans:
        phys_x, aux_x = loccap_route_sl(
            topk_all.long().cpu(), _p2l_x, _l2p_x, _lc_x,
            cfg.nlp, L, args.eps, return_tables=True)
        bounds_x = loccap_sl_bounds(aux_x, W, -1)
        args.pll_f_cap = max(args.pll_f_cap, bounds_x["f_cap"])
        for k in ("recv_cap", "pair_cap"):
            pll_bounds[k] = max(pll_bounds[k], bounds_x[k])
        s2_refs.append((phys_x, aux_x))
        for _tk in sched_topk_all[1:]:
            # topic-schedule harness: every sizing placement x every topic
            _, _aux_t = loccap_route_sl(
                _tk.long().cpu(), _p2l_x, _l2p_x, _lc_x,
                cfg.nlp, L, args.eps, return_tables=True)
            _b_t = loccap_sl_bounds(_aux_t, W, -1)
            args.pll_f_cap = max(args.pll_f_cap, _b_t["f_cap"])
            for k in ("recv_cap", "pair_cap"):
                pll_bounds[k] = max(pll_bounds[k], _b_t[k])
        if rank == 0:
            print(f"[s2-sizing] {_tag}: f_cap {bounds_x['f_cap']} recv "
                  f"{bounds_x['recv_cap']} pair {bounds_x['pair_cap']}",
                  flush=True)
    if s2_size_plans and not args.s2_swap:
        # (swap lane excluded 2026-08-29: its runtime placements are exactly
        # the orbit fold above — a pure function of the fixed batch demand —
        # so the placement-independent bound is unnecessary, and at 16n b64
        # K2 it is 4.9x the fold's recv_cap (343805 vs 70156 rows), which
        # alone overflowed the 16G symmetric heap.)
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
    # s2: every sizing-set placement's twin meta joins the maxes below
    sps_all, uc_all = [sps_ref], [uc_ref]
    for phys_x, _aux_x in s2_refs:
        _prb = phys_x.view(ntokens, args.topk).long()
        vce_ref_b = (_prb // cfg.nlp) * gpe + 1 + _prb % cfg.nlp
        _, _, sps_b, uc_b = python_meta_from_vce(
            vce_ref_b.cuda(), W, S, gpe, nn, L)
        sps_all.append(sps_b)
        uc_all.append(uc_b)

    # kernel-drift cushion (handoff 17 / llc demand-sizing lineage)
    fp_slack = int(pll_aux["forced_pair"].sum(0).max())
    for _phys_x, aux_x in s2_refs:
        fp_slack = max(fp_slack, int(aux_x["forced_pair"].sum(0).max()))
    cushion = fp_slack + 8 * W

    # recv rows this rank computes (dispatch recv == combine send)
    recv_real = max(int(s[:, rank * gpe:(rank + 1) * gpe].sum())
                    for s in sps_all)
    if args.route_global:
        # deterministic route: runtime == reference bitwise, realized is
        # exact — the SL provable bound does not bound THIS algorithm
        recv_cap = recv_real + cushion
    elif args.sizing == "demand" and args.scenario == "s1" and sched_topk_all:
        # topic-schedule harness: recv_real is topic 0's realized demand;
        # size at the provable cap maxed over the scheduled topics
        recv_cap = pll_bounds["recv_cap"]
    elif args.sizing == "demand" and args.scenario == "s1":
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

    if args.wire in ("direct", "dov"):
        # transport ablation (2026-08-30): no fused combine-meta derive —
        # the plan-overlap machinery has nothing to overlap
        assert args.scenario == "s1", f"--wire {args.wire} is s1-only"
        args.plan_overlap = 0

    # late plan-overlap byte gate (canon-regen 8/29 bisect: mode 2 is
    # SAFE and winning at <= b16-class budgets but SIGABRTs/stalls b32+
    # cells at 4n AND 8n — an unexplained interaction with heavy wire
    # load; graphs and combine-idx kernel exonerated by the bisect; RCA
    # OPEN, handoff 26. Same engage-below-threshold shape as wave-adapt.
    if args.plan_overlap == 2:
        _ov2_max = int(os.environ.get("FLUX_OURS_OV2_MAX_MIB", "16"))
        _budget_mib = (S * args.chunk_bytes) / (1 << 20)
        if _budget_mib > _ov2_max:
            if rank == 0:
                print(f"[ours] plan_overlap 2 -> 0 (budget "
                      f"{_budget_mib:.0f} MiB/rank > OV2_MAX {_ov2_max})",
                      flush=True)
            args.plan_overlap = 0

    # ---- runner + planner (+ s2 movement lane) ----
    if args.wire in ("direct", "dov"):
        from flux.testing.ours_direct import (OursDirectRunner,
                                              OursDirectOverlapRunner)
        # All2AllSingle max_split: reference realized per-pair max vs the
        # provable SL pair cap, + the shared drift cushion (the runner
        # asserts the realized pair rows against it every iteration)
        _dest_ref = phys_ref.view(-1).long() // cfg.nlp
        _src_ref = torch.arange(W, dtype=torch.int64).repeat_interleave(
            S * args.topk)
        _pair_ref = int(torch.bincount(
            _src_ref * W + _dest_ref, minlength=W * W).max())
        if args.dwire_pair_sizing == "demand":
            # 2026-08-30 16n b32/b64 fix: the provable pair_cap floor alone
            # needs >16G of All2AllSingle staging (ctor NVSHMEM_MALLOC death,
            # handoff 30 SS5/SS8); realized-ref sizing mirrors
            # llc_sizing=demand (recv side, 8/25) and eplb's own max_split
            # convention. The cushion must be PAIR-LEVEL: the shared
            # `cushion` is the recv-side per-destination SUM (fp_slack + 8W
            # ~ 9k rows at K2 16n) and inflates the staging ~9x over the
            # realized pair. Loud contract preserved: the runner asserts
            # realized pair rows against max_split every iteration.
            # (Applies to dov's combine wire identically — same staging.)
            _fp_pair = int(pll_aux["forced_pair"].max())
            dwire_max_split = _pair_ref + _fp_pair + 64
            if rank == 0:
                print(f"dwire_pair_sizing=demand: max_split "
                      f"{dwire_max_split} (pair_ref {_pair_ref}, fp_pair "
                      f"{_fp_pair}) vs capacity "
                      f"{max(_pair_ref, int(pll_bounds['pair_cap'])) + cushion}",
                      flush=True)
        else:
            dwire_max_split = (max(_pair_ref, int(pll_bounds["pair_cap"]))
                               + cushion)
        if args.wire == "dov":
            # overlapped edition: the fused op's recv capacity is the
            # already-all-reduced l0_recv_rows (FLUX_A2AV_MAX_RECV_NTOKENS
            # exported above); the combine wire keeps the dwire staging
            # width. The runner forces the flat-mode ctor env (LB_UNION=0,
            # FLAT_FENCED_SIG=1, EARLY_LAUNCH=1) on every rank.
            runner = OursDirectOverlapRunner(
                TP_GROUP, EP_GROUP, DIST_ENV.NNODES, rank, L, cfg,
                args.ffn_hidden_size, input_dtype,
                l0_recv_rows, dwire_max_split, sm_margin=args.sm_margin)
            RECORDER.emit_info(
                ours_wire="dov",
                dwire_max_split=int(dwire_max_split),
                dwire_pair_ref=_pair_ref,
                dwire_pair_sizing=args.dwire_pair_sizing,
                dov_flat_fenced_sig=int(os.environ.get(
                    "FLUX_A2AV_FLAT_FENCED_SIG", "0")),
                dov_early_launch=int(os.environ.get(
                    "FLUX_A2AV_EARLY_LAUNCH", "0")))
            _st("runner (direct-overlap wire) constructed")
        else:
            runner = OursDirectRunner(
                TP_GROUP, rank, L, cfg, args.ffn_hidden_size, input_dtype,
                recv_cap, dwire_max_split,
                probs_all_setup[rank * S:(rank + 1) * S])
            RECORDER.emit_info(ours_wire="direct",
                               dwire_max_split=int(dwire_max_split),
                               dwire_pair_ref=_pair_ref,
                               dwire_pair_sizing=args.dwire_pair_sizing)
            _st("runner (direct wire) constructed")
    else:
        runner = OursRunner(
            TP_GROUP, EP_GROUP, DIST_ENV.NNODES, L, cfg,
            args.ffn_hidden_size, input_dtype, l0_recv_rows,
            sm_margin=args.sm_margin, plan_overlap=int(args.plan_overlap))
        _st("runner (fused ops) constructed")
    lane = None
    store = None
    swap_lane = None
    if args.scenario == "s1":
        w1v, w2v = build_slot_weights(plan.p2l, rank, cfg.nlp, gpe,
                                     args.ffn_hidden_size, args.H,
                                     input_dtype)
        runner.set_weights(w1v, w2v)
    else:
        from flux.testing.ours_s2 import OursMovementLane
        if not use_pv2:
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
        swap_lane = None
        if args.s2_swap:
            from flux.testing.ours_swap import OursSwapLane
            # per-node NCCL subgroups (collective creation on all ranks);
            # intra-node P2P rides NVLink on its own communicator
            _my_ng = None
            for _u in range(nn):
                _g = torch.distributed.new_group(
                    list(range(_u * L, (_u + 1) * L)))
                if _u == rank // L:
                    _my_ng = _g
            # EAGER communicator init (2026-08-27 qwen gate wedge RCA):
            # NCCL comm creation is COLLECTIVE over the subgroup, but the
            # first swap only makes P2P calls from the PAIR ranks — the
            # bystanders are already parked in the world allgather ->
            # circular wait (i0 deadlock, 925 ms partial-init then wedge).
            # One tiny all_reduce per node group at setup initializes the
            # communicator with all members present; issue() is then
            # enqueue-only.
            torch.distributed.all_reduce(
                torch.zeros(1, device="cuda"), group=_my_ng)
            torch.cuda.synchronize()
            if args.swap_rounds == "all":
                # ABLATION-ONLY: composed multi-slot exchange lane
                from flux.testing.ours_swap import OursSwapAllLane
                swap_lane = OursSwapAllLane(lane, rank, L, cfg.nlp,
                                            args.ffn_hidden_size, args.H,
                                            input_dtype,
                                            args.swap_max_moves,
                                            TP_GROUP)
            else:
                swap_lane = OursSwapLane(lane, _my_ng, rank, L, cfg.nlp,
                                         args.ffn_hidden_size, args.H,
                                         input_dtype,
                                         xport=args.swap_xport,
                                         issue=args.swap_issue,
                                         pg=TP_GROUP)
            swap_sync = None
            if args.swap_tables == "device":
                from flux.testing.ours_swap import SwapTableSync
                swap_sync = SwapTableSync(plan, torch.device("cuda"))
            swap_reset_snap = None
            if args.swap_reset != "off":
                # ABLATION-ONLY: pristine copies for the placement reset —
                # tables AND the physical slot weights (the exchange moves
                # real weights; a table-only reset would desynchronize).
                _s1 = lane.op_w1.prefetch_slots()
                _s2 = lane.op_w2.prefetch_slots()
                swap_reset_snap = {
                    "p2l": plan.p2l.clone(),
                    "l2p": plan.l2p.clone(),
                    "lcnts": plan.lcnts.clone(),
                    "w1": torch.stack([_s1[1 + j].clone()
                                       for j in range(cfg.nlp)]),
                    "w2": torch.stack([_s2[1 + j].clone()
                                       for j in range(cfg.nlp)]),
                }
    planner = OursIterPlanner(plan, rank, torch.device("cuda"), topk_all,
                              probs_all_setup, L, args.eps, args.pll_f_cap,
                              TP_GROUP, route_global=bool(args.route_global))
    # r2/s2 f_cap contract fix (handoff 22 §4): runtime-adopted placements
    # can exceed any setup-derived forced budget — enable the planner's
    # local escalate-and-reroute (kstats[2] breach -> 4x then uncapped).
    planner.f_cap_retry = (args.scenario == "s2")


    # s2 machinery: oracle snapshots for the stale probe + the weight probe
    pv2_state = None
    pv2_snap = None
    if args.scenario == "s2":
        if use_pv2:
            # pv2 resident state (host): the histogram the resident
            # placement was solved on (drift reference), its node map,
            # and the slot table identity
            pv2_state = {
                "hist": plfast.demand_hist(tk_solve, L, args.G)
                        .cpu().long(),
                "ion": pv2_res["ion"].clone(),
                "p2l": plan.p2l.long().clone(),
            }
        if args.s2_stale == "rot":
            # rank-rolled hosts (precomputed in the sizing set): every
            # expert's instances shift one rank — crosses node
            # boundaries, structurally suboptimal, so the runtime solve
            # always finds adds (movement fires per iteration)
            p2l_r, l2p_r, lcnts_r = rot_tensors
            ion_r_h = plfast.hosts_to_ion(hosts_rot, W, L)
            oracle_snap = {
                "p2l": p2l_r, "l2p": l2p_r, "lcnts": lcnts_r,
            }
            if store is not None:
                ion_r = ion_r_h.to(store.ion.device)
                primary_r = ion_r.long().argmax(dim=1)
                oracle_snap.update(
                    primary=primary_r.to(store.primary.dtype),
                    ion=ion_r,
                    hist=store.hist.clone(),
                    load_e=store.load_e.clone(),
                )
            else:
                pv2_snap = {
                    "hist": pv2_state["hist"].clone(),
                    "ion": ion_r_h.clone(),
                    "p2l": p2l_r.long().clone(),
                }
        else:
            oracle_snap = {
                "p2l": plan.p2l.clone(), "l2p": plan.l2p.clone(),
                "lcnts": plan.lcnts.clone(),
            }
            if store is not None:
                oracle_snap.update(
                    primary=store.primary.clone(),
                    ion=store.ion.clone(),
                    hist=store.hist.clone(),
                    load_e=store.load_e.clone(),
                )
            else:
                pv2_snap = {k: v.clone() for k, v in pv2_state.items()}
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
    if args.wire == "direct":
        # direct-wire edition: the adapter's splits/segments against the
        # same python recipe (sps_ref [W, E_virt] rows per (src, vslot))
        _sps = sps_ref.cpu().long().view(W, W, gpe)
        assert torch.equal(runner._in_splits.long().cpu(),
                           _sps[rank].sum(1)), (
            "direct in_splits != python_meta_from_vce")
        assert torch.equal(runner._out_splits.long().cpu(),
                           _sps[:, rank].sum(1)), (
            "direct out_splits != python_meta_from_vce")
        assert runner.n_recv == int(_sps[:, rank].sum()), (
            "direct n_recv != python_meta_from_vce")
        _seg_ref = _sps[:, rank].sum(0)  # per-vslot recv (pad-first)
        assert int(_seg_ref[0]) == 0, "rows in the pad slot"
        for _p, _s, _e in runner._segments:
            assert _e - _s == int(_seg_ref[1 + _p]), (
                f"direct segment rows drift at local slot {_p}")
        _m_ref = runner.n_recv
    elif args.wire == "dov":
        # direct-overlap edition: the op derive AND the combine metadata
        # against the python recipe — including the stable scatter index
        # (the OUT layout the combine permutation assumes) bitwise
        assert torch.equal(runner._sd.long().cpu(), sp_ref.long().cpu()), (
            "dov derive splits != python_meta_from_vce")
        assert torch.equal(runner._scd.cpu(), sc_ref.cpu().int()), (
            "dov derive scatter_index != python argsort(stable).argsort()")
        assert torch.equal(runner._sps.cpu(), sps_ref.cpu().int()), (
            "dov derive sps != python_meta_from_vce")
        _sps = sps_ref.cpu().long().view(W, W, gpe)
        assert torch.equal(runner._in_splits.long().cpu(),
                           _sps[rank].sum(1)), (
            "dov in_splits != python_meta_from_vce")
        assert torch.equal(runner._out_splits.long().cpu(),
                           _sps[:, rank].sum(1)), (
            "dov out_splits != python_meta_from_vce")
        assert runner.n_recv == int(_sps[:, rank].sum()), (
            "dov n_recv != python_meta_from_vce")
        _seg_ref = _sps[:, rank].sum(0)  # per-vslot recv (pad-first)
        assert int(_seg_ref[0]) == 0, "rows in the pad slot"
        for _p, _s, _e in runner._segments:
            assert _e - _s == int(_seg_ref[_p]), (
                f"dov segment rows drift at vslot {_p}")
        # the arrival->out permutation must be a permutation of [0, n)
        _ps = runner._place_slots.cpu()
        assert _ps.numel() == runner.n_recv
        assert torch.equal(torch.sort(_ps).values,
                           torch.arange(runner.n_recv)), (
            "dov place_slots is not a permutation")
        _m_ref = runner.n_recv
    else:
        assert torch.equal(runner._sd.long().cpu(), sp_ref.long().cpu()), (
            "derive splits != python_meta_from_vce")
        assert torch.equal(runner._sps.cpu(), sps_ref.cpu().int()), (
            "derive sps != python_meta_from_vce")
        assert torch.equal(runner._uc.cpu(), uc_ref.cpu().int()), (
            "derive uc != python_meta_from_vce")
        _m_ref = runner._m_this
    if rank == 0:
        print(f"setup audit OK: E_virt {E_virt} gpe {gpe} m_ref "
              f"{_m_ref}; placement basis {oracle_basis} "
              f"drift {_drift} ppm; solver {place_solver_ms:.1f} ms")
    _st("setup audit ok")

    RECORDER.emit_info(
        ours_fusion=("direct_a2av" if args.wire == "direct"
                     else "direct_overlap_a2av" if args.wire == "dov"
                     else "slipstream_v2"),
        ours_plan_overlap=int(bool(args.plan_overlap)),
        # plan-lane cost knobs (ours.py module header; all default OFF)
        ours_plan_xchg_narrow=planner.xchg_narrow,
        ours_plan_prealloc=int(planner.plan_prealloc),
        ours_plan_graph=int(planner.plan_graph),
        ours_plan_scale_graph=int(getattr(runner, "scale_graph", 0)),
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
        ours_place_solver=args.place_solver,
        ours_s2_sizing_plans=",".join(t[0] for t in s2_size_plans),
        **({f"pv2_{k}": v for k, v in pv2_res["stats"].items()
            if k != "mode"} if use_pv2 else {}),
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
    if lane is not None and not use_pv2:
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

    # pv2 host lane: pin to one intra-op thread for the timed loop —
    # tiny host tensor ops jitter badly under a contended OMP pool
    # (login-A100 bench: 6 -> 56 ms outliers at 128 threads); integer
    # results are thread-count-independent. Restored after the loop.
    _nt_prev = None
    if use_pv2 and lane is not None:
        _nt_prev = torch.get_num_threads()
        torch.set_num_threads(1)

    # ---- timed loop ----
    total_iters = args.warmup_iters + args.iters
    ev = lambda: [torch.cuda.Event(enable_timing=True)
                  for _ in range(total_iters)]
    iter_start, plan_comm_end, place_end, plan_end = ev(), ev(), ev(), ev()
    e2e_start, l0_end, act_end, e2e_end = ev(), ev(), ev(), ev()
    # graph-capture priming (canon-regen 8/29): capture the plan/scale
    # graphs HERE, all ranks quiesced between barriers — lazy first-use
    # capture inside an iteration SIGABRTs b32/b64 cells (torch's
    # pre-capture sync deadlocks against peers' in-flight collectives)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    planner.prime_graphs()
    runner.prime_scale_graph(planner)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    move_stats = []   # (trigger, moves, bytes, movement_ms, gain_ppm)
    out = None
    _sched_dev = [t.long().cuda() for t in sched_topk_all]
    _sched_cur = 0
    for i in range(total_iters):
        runner.prep()
        if _sched_dev:
            # topic-schedule harness: pick this iteration's topic OUTSIDE
            # the window; the last iteration lands on topic 0 (reference)
            _k = ((i - (total_iters - 1)) // args.routing_dwell) % len(_sched_dev)
            if _k != _sched_cur:
                planner.set_own_topk(_sched_dev[_k])
                _sched_cur = _k
                torch.cuda.synchronize()
        probe.step(i)
        if lane is not None and args.s2_stale != "0":
            # PROBE: reset the resident placement/tables to the oracle
            # solve OUTSIDE the window — trigger+movement re-fire every
            # timed iteration (worst-case movement overlap)
            plan.p2l = oracle_snap["p2l"].clone()
            plan.l2p = oracle_snap["l2p"].clone()
            plan.lcnts = oracle_snap["lcnts"].clone()
            if store is not None:
                store.primary.copy_(oracle_snap["primary"])
                store.ion.copy_(oracle_snap["ion"])
                store.hist.copy_(oracle_snap["hist"])
                store.load_e.copy_(oracle_snap["load_e"])
            else:
                pv2_state = {k: v.clone() for k, v in pv2_snap.items()}
            lane.resident_p2l = oracle_snap["p2l"].long().clone()
            planner.refresh_placement()
            torch.cuda.synchronize()
        if (lane is not None and args.s2_swap
                and args.swap_reset != "off"
                and ((i >= args.warmup_iters
                      and (i - args.warmup_iters) % args.swap_reset_period == 0)
                     if args.swap_reset == "every"
                     else i == args.warmup_iters)):
            # ABLATION-ONLY placement reset (untimed gap — enqueued
            # before iter_start.record, completed by the isolated sync):
            # restore the oracle-basis tables AND the physical slot
            # weights, so the timed iteration re-executes the full
            # drift-event swap + movement instead of the warmup-converged
            # fixpoint.
            plan.p2l = swap_reset_snap["p2l"].clone()
            plan.l2p = swap_reset_snap["l2p"].clone()
            plan.lcnts = swap_reset_snap["lcnts"].clone()
            _s1 = lane.op_w1.prefetch_slots()
            _s2 = lane.op_w2.prefetch_slots()
            for _j in range(cfg.nlp):
                _s1[1 + _j].copy_(swap_reset_snap["w1"][_j])
                _s2[1 + _j].copy_(swap_reset_snap["w2"][_j])
            if swap_sync is not None:
                swap_sync.apply(planner, plan)
            else:
                planner.refresh_placement()
            lane.resident_p2l = swap_reset_snap["p2l"].long().clone()
            torch.cuda.synchronize()
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < args.warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            iter_start[i].record()
            if args.route_global:
                # ONE collective per iteration: raw topk + probs (the
                # d-allgather is gone — d is a local pure function of the
                # gathered topk inside the route)
                planner.rg_exchange()
            else:
                torch.distributed.all_gather_into_tensor(
                    d_gather_buf, planner.local_loads(), group=TP_GROUP)
            plan_comm_end[i].record()
            if lane is not None and args.s2_swap:
                # SWAP place lane (timed, counts toward total_ms): D2H of
                # d -> greedy intra-node pair+swap decision (sub-ms host
                # integer, EPIC §4.3 analog) -> table transposition ->
                # NVLink exchange issued on the movement stream. No
                # cross-node migration; the router (next bracket) runs on
                # the swapped tables — same-iteration benefit.
                _pt0 = time.perf_counter()
                d_host = d_gather_buf.cpu()   # blocks on the in-flight
                _ptd = time.perf_counter()    # allgather: d2h span below
                load_g = d_host.long().sum(0)
                if args.swap_rounds == "all":
                    # ABLATION-ONLY: capped tau=1 orbit to fixpoint at
                    # decision time; the COMPOSED net intra-node
                    # permutation executes as one multi-slot phase.
                    _p2l_f, _l2p_f, _nr = oswap_rt.swap_orbit_capped(
                        load_g, plan.p2l, plan.l2p, plan.lcnts, L,
                        cfg.nlp, args.swap_max_moves,
                        per_pair=args.swap_pair_moves)
                    all_moves = oswap_rt.net_moves(plan.p2l, _p2l_f, L,
                                                   cfg.nlp)
                    swaps = [mv for lst in all_moves for mv in lst]
                    _pt1 = time.perf_counter()
                    if swaps:
                        plan.p2l, plan.l2p = _p2l_f, _l2p_f
                        if swap_sync is not None:
                            swap_sync.apply(planner, plan)
                        else:
                            planner.refresh_placement()
                    swap_lane.prepare(all_moves)
                else:
                    swaps, _Lr = oswap_rt.swap_plan(
                        load_g, plan.p2l, plan.lcnts, L, cfg.nlp,
                        args.swap_tau_rows)
                    _pt1 = time.perf_counter()
                    if swaps:
                        plan.p2l, plan.l2p = oswap_rt.apply_swaps(
                            plan.p2l, plan.l2p, swaps)
                        if swap_sync is not None:
                            swap_sync.apply(planner, plan)
                        else:
                            planner.refresh_placement()
                    swap_lane.prepare(swaps)
                swap_lane.issue_early()
                if not args.swap_overlap and swap_lane._issued:
                    # ABLATION-ONLY sequential mode: the exchange must
                    # LAND before anything downstream is enqueued — swap
                    # first, then dispatch (un-overlapped expert
                    # dispatch). Timed: the wait sits in the place
                    # bracket, so total_ms carries the exposed exchange.
                    torch.cuda.current_stream().wait_event(
                        swap_lane.ev_done)
                move_stats.append((int(bool(swaps)), len(swaps),
                                   swap_lane.move_bytes_this_iter, 0))
                if rank == 0 and args.check_iters:
                    _pt2 = time.perf_counter()
                    print(f"[swap/{args.swap_xport}/{args.swap_issue}]"
                          f" iter {i}: d2h "
                          f"{(_ptd - _pt0) * 1e3:.2f} plan "
                          f"{(_pt1 - _ptd) * 1e3:.2f} apply+issue "
                          f"{(_pt2 - _pt1) * 1e3:.2f} ms swaps "
                          f"{len(swaps)}: {swaps}", flush=True)
            elif lane is not None and use_pv2:
                # PV2 PLACE lane (timed): one D2H of d -> host drift ->
                # stateless marginals solve (placement_v2) -> gain /
                # trigger -> plan tensors + adoption + movement issue.
                # No warm state, no CUDA graph, no batch-size term; the
                # adopted placement equals the setup batch solve by
                # purity (sizing envelope exact).
                _pt0 = time.perf_counter()
                hist_now = (d_gather_buf.cpu().long()
                            .view(nn, L, args.G).sum(1))
                drift = plfast.drift_ppm(hist_now, pv2_state["hist"])
                _pt1 = time.perf_counter()
                always = args.place_gain_threshold_ppm == 0
                if always or drift >= args.place_drift_prefilter_ppm:
                    res_new = pv2mod.pv2_solve(hist_now, L, cfg.nlp)
                    p2l_new = res_new["p2l"].long()
                    changed_n = int((p2l_new != pv2_state["p2l"]).sum())
                    _pt2 = time.perf_counter()
                    if always:
                        # trigger foregone; gain MUST be 0 (apply_moves
                        # gates on gain >= threshold — pll-lane lesson)
                        verdict = {"trigger": 1, "gain_ppm": 0}
                    else:
                        rr_cur = pv2mod.pv2_remote_rows(
                            hist_now, pv2_state["ion"])
                        rr_new = pv2mod.pv2_remote_rows(
                            hist_now, res_new["ion"])
                        gain = (0 if rr_cur == 0 else
                                (rr_cur - rr_new) * 1_000_000 // rr_cur)
                        verdict = {
                            "trigger": int(gain >= max(
                                args.place_gain_threshold_ppm, 1)),
                            "gain_ppm": int(gain),
                        }
                    if args.s2_force_trigger and changed_n > 0:
                        verdict["trigger"] = 1
                    _pt3 = time.perf_counter()
                    if verdict["trigger"] and changed_n > 0:
                        plan.p2l, plan.l2p, plan.lcnts = (
                            res_new["p2l"], res_new["l2p"],
                            res_new["lcnts"])
                        planner.refresh_placement()
                        lane.apply_moves(
                            p2l_new, verdict["gain_ppm"],
                            wprobe_cb=(wprobe_cb if args.s2_wprobe
                                       else None))
                        pv2_state = {"hist": hist_now,
                                     "ion": res_new["ion"],
                                     "p2l": p2l_new}
                    else:
                        lane.moves_this_iter = 0
                        lane.move_bytes_this_iter = 0
                        lane.trigger_fired = 0
                        lane.last_gain_ppm = verdict["gain_ppm"]
                    move_stats.append((
                        int(verdict["trigger"]), lane.moves_this_iter,
                        lane.move_bytes_this_iter, verdict["gain_ppm"]))
                    if rank == 0 and args.check_iters:
                        _pt4 = time.perf_counter()
                        print("[pv2-split] iter %d: drift %.2f solve "
                              "%.2f decide %.2f adopt+moves %.2f ms" % (
                                  i, (_pt1 - _pt0) * 1e3,
                                  (_pt2 - _pt1) * 1e3,
                                  (_pt3 - _pt2) * 1e3,
                                  (_pt4 - _pt3) * 1e3))
                        print(f"[s2] iter {i}: drift {drift} gain "
                              f"{verdict['gain_ppm']} adds {changed_n} "
                              f"trigger {verdict['trigger']} moved "
                              f"{lane.moves_this_iter}")
                else:
                    move_stats.append((0, 0, 0, int(drift)))
                    if rank == 0 and args.check_iters:
                        print(f"[s2] iter {i}: drift {drift} < "
                              f"prefilter — quiet")
            elif lane is not None:
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
            if swap_lane is not None:
                gate_kw = swap_lane.gate_kwargs()
            elif lane is not None:
                if args.s2_join == "join":
                    lane.op_w1.join()
                else:
                    gate_kw = lane.gate_kwargs()
            _hbp("l0")
            l0_out = runner.l0_forward(inputs_shard, gate_kwargs=gate_kw)
            # late plan-overlap (mode 2): the combine-meta host work runs
            # HERE, while the GPU executes the just-enqueued l0 — host stays
            # ahead (l0 GPU span >> meta host span at every budget), so no
            # timed bracket inflates; the meta kernels ride the side stream
            # on the sm_margin headroom concurrently with the GEMM.
            runner.issue_combine_meta(ip, late=True)
            if swap_lane is not None:
                # late/split issue: the exchange rides under the enqueued
                # l0 (movement stream depends only on the pre-l0 event)
                swap_lane.issue_late()
            l0_end[i].record()
            if lane is not None:
                # FLUX_OURS_S2_W2_LATE: l1 weight pushes enqueue AFTER the
                # dispatch legs (no-op when the knob is off / no trigger)
                lane.issue_w2_late()
            intermediate = torch.nn.functional.gelu(l0_out)
            act_end[i].record()
            if swap_lane is not None:
                swap_lane.l1_wait()
            elif lane is not None:
                lane.join_w2()
            _hbp("l1")
            out = runner.l1_forward(intermediate)
            e2e_end[i].record()
            if lane is not None and swap_lane is None:
                # end-of-iteration weight-signal drain (K2-4n stale hang
                # fix): after e2e_end (untimed gap) — see ours_s2.join_w1.
                # Swap mode: signals are LOCAL writes — nothing crosses
                # the boundary; no drain needed.
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

    if _nt_prev is not None:
        torch.set_num_threads(_nt_prev)
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

    if args.wire in ("direct", "dov"):
        # diagnostic sub-phase metrics (must run BEFORE the final
        # deterministic iteration appends events past the timed loop)
        iter_times.update(runner.sub_times(args.warmup_iters))
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
    runner.issue_combine_meta(ip_ref, late=True)  # no-op unless mode 2
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
