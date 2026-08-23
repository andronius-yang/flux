################################################################################
#
# EPIC baseline layer0 arm for the sweep harness.
# Structural clone of test_moe_eplb_traffic.py; the placement algorithm, PEO
# group pipeline, and migration are python/flux/testing/epic_semantics.py
# (invariant-tested by test_epic_planner.py). Paper: flux/EPIC.pdf
# (SIGCOMM'26, Alibaba). Faithful baseline — authenticity over performance;
# NO flux GEMM-overlap machinery anywhere in this path.
#
################################################################################
"""COMET-style layer0 benchmark of EPIC: PEO per-expert-group overlap (§5.2)
+ EPIC-EPLB placement with replication (§4.2) + dynamic intra-host expert
migration (§4.3).

Pipeline per iteration (m = --groups; two CUDA streams):

  main:  plan_comm (all_gather [W, G] loads) -> [migration] -> pack
  comm:  wait(pack) -> D_0 -> D_1 -> ... -> D_{m-1}       (dispatch staging)
  main:  for g: wait(D_g) -> scatter_g -> GEMM_g          (compute staging)

Dispatch staging and compute staging (EPIC §5.2's two serialization rules)
fall out of stream in-order semantics; GEMM_g is gated on D_g by one event.
GEMMs are un-overlapped flux.GemmGroupedV2 launches, one per group, over the
contiguous slot-weight slice — launch-granularity faithful. m=1 is the
no-overlap arm and the validation anchor.

Transport (--transport, default hier_compress = EPIC's own Mode 2, §5.1
Figure 8(d): PXN relay + de-redundancy — the paper's distinctive transport):
hier_compress = fused-op dispatch_only over the virtual slot space (l01
combine via per-group TopkReduceScatterOp); nvshmem = Mode-1/DeepEP-default
analog, the staged per-entry wire (NO dedup) over flux's one-sided NVSHMEM
All2AllSingle; nccl = debug/parity-only NCCL alltoallv fallback (never a
faithful EPIC arm — pin it explicitly if you really want it).

Timing accounting (SCHEMA protocol rule 5, 2026-08-20): on the m=1 / D6
arms every batch-derived quantity — D6 quotas, the reroute expansion, wire
splits, scatter/segment/combine indices, the hc virtual-space metadata and
combine index sets — is recomputed PER ITERATION on device inside the
`plan` event bracket (flux.testing.epic_semantics.EpicIterPlanner;
timing_accounting=per_iter_gpu). m>1 and the loccap router keep the legacy
setup-time planning and are marked legacy_untimed_plan. Setup-time builds
survive only as buffer sizing, correctness references, and the loud
bitwise drift guard (planner.check_against at setup).

Migration (--migration on|inkernel) runs between plan and pack, per
Figure 5: replicated decision (per-node heaviest/lightest pairing,
tau-gated swaps) — on the rule-5 path a VECTORIZED GPU decision consuming
THIS iteration's derived instance loads (plan_migration_swaps_gpu; the
pre-2026-08-20 claim that the decision read the plan_comm gather was
wrong — the buffer was never consumed; it now is) — then intra-node weight
exchange (batched P2P, or the fused in-kernel phase 0), plan mutation,
layout rebuild + planner re-derive. Under this harness's static per-cell
routing the swaps converge during warmup; timed iterations measure the
honest steady-state per-step cost (decision + zero swaps), and the warmup
swap costs are book-kept (epic_migration_* facts).

Phase metrics (long-format, impl=epic): plan_comm_ms, plan_ms (the rule-5
per-iteration planner; ~0 on legacy arms), migration_ms, pack_ms,
disp{g}_ms (comm stream), stall{g}_ms (exposed wait before group g's
compute), scat{g}_ms, gemm{g}_ms, comm_ms (whole dispatch window), e2e_ms
(mig-end -> last GEMM: the comm-start->gemm-finish number, FAST
convention; planning is inside total_ms, not e2e_ms), total_ms
(start -> last GEMM).
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
from flux.testing.epic_semantics import (
    EpicIterPlanner,
    EpicLayer0Runner,
    build_epic_plan,
    build_fixed_plan,
    build_nodeaware_plan,
    plan_migration_swaps,
    plan_migration_swaps_gpu,
)
from flux.testing.placelambda_gpu import (
    build_placement_gpu,
    loccap_route_gpu,
    loccap_route_sl,
    loccap_sl_bounds,
    place_decision,
    placement_hash,
)
from flux.testing import placelambda_fast as plfast
from flux.testing.loccap_semantics import (
    d6_route,
    evensplit_route,
    incidence_stats,
    loccap_route,
    route_hash,
)
from flux.testing.ultraep_semantics import (
    UltraEPConfig,
    loads_from_topk,
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
    import socket

    host = socket.gethostname()
    hosts = [None] * DIST_ENV.WORLD_SIZE
    torch.distributed.all_gather_object(hosts, host, group=TP_GROUP)
    lw = DIST_ENV.LOCAL_WORLD_SIZE
    for node in range(DIST_ENV.WORLD_SIZE // lw):
        block = hosts[node * lw:(node + 1) * lw]
        assert len(set(block)) == 1, f"ranks not node-major: {hosts}"


@torch.no_grad()
def perf_epic(
    runner: EpicLayer0Runner,
    iter_planner,  # EpicIterPlanner (rule-5 path) or None (legacy m>1/loccap)
    ctx: MoeMlp1Ctx,
    probs_shard: torch.Tensor,
    loads_shard: torch.Tensor,
    loads_gather_buf: torch.Tensor,
    comm_stream,
    warmup_iters: int,
    iters: int,
    sm_margin: int,
    migration: str,  # "off" | "on" (host NCCL) | "inkernel" (fused phase 0)
    tau_tokens: float,
    single_stream: bool,
    place_fn=None,  # dynamic PLACE-lambda decision: called per iteration
    #                 INSIDE the timed bracket (place_ms); None = static
    #                 ablation (zero-width place event)
):
    m = runner.m
    l01 = runner.layers == "l01"
    disp_fn = (runner.dispatch_group_hc if runner.hc_enabled
               else runner.dispatch_group)
    total_iters = warmup_iters + iters
    g_names = (
        [f"disp{g}" for g in range(m)]
        + [f"gate{g}" for g in range(m)]
        + [f"scat{g}" for g in range(m)]
        + [f"gemm{g}" for g in range(m)]
    )
    if l01:
        g_names += (
            [f"act{g}" for g in range(m)]
            + [f"gemm1_{g}" for g in range(m)]
            + [f"cpack{g}" for g in range(m)]
            + [f"comb{g}" for g in range(m)]
            + [f"acc{g}" for g in range(m)]
            + ["sum_end"]
        )
    names = ["start", "plan_comm", "place", "plan", "mig", "pack"] + g_names
    ev = {
        name: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for name in names
    }
    torch.distributed.barrier()
    torch.cuda.synchronize()

    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    _hb = (print if int(os.getenv("FLUX_PLL_DEBUG", "0"))
           else (lambda *a, **k: None))
    _rand_payload = bool(int(os.getenv("FLUX_PLL_RANDOM_PAYLOAD", "0")))
    _rand_gen = (torch.Generator(device=ctx.inputs_shard.device)
                 .manual_seed(4242 + 7919 * runner.rank)
                 if _rand_payload else None)
    if _rand_payload:
        # provenance ledger for the correctness probe: payload 0 = the
        # pre-loop (setup) shard, payload k = the one dispatched in loop
        # iteration k-1
        ctx._pll_payloads = [ctx.inputs_shard.clone()]
    iso_sync_times = []
    mig_host_times = []
    mig_swaps_per_iter = []
    mig_mine_per_iter = []  # inkernel: 1 iff THIS rank launched a swap
    relayout_ms_total = 0.0
    for i in range(total_iters):
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        _hb(f"[pll-hb] r{runner.rank} iter {i} enter", flush=True)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            ev["start"][i].record()
            # Recurring counts derivation + exchange (rule 5: the histogram
            # is batch-derived, so the rule-5 path recomputes it in-bracket;
            # it is also the migration decision's load carrier).
            torch.distributed.all_gather_into_tensor(
                loads_gather_buf,
                (iter_planner.local_loads() if iter_planner is not None
                 else loads_shard),
                group=TP_GROUP,
            )
            ev["plan_comm"][i].record()
            # Dynamic PLACE-lambda lane (2026-08-21 amendment): when the
            # placement ablation is ON, the per-iteration solve + move
            # diff + trigger threshold run here, TIMED (place_ms). Static
            # arms record a zero-width event.
            if place_fn is not None:
                place_fn(i)
            ev["place"][i].record()
            # Per-iteration on-device planning (SCHEMA rule 5): NOTHING
            # derived from routing is carried across iterations on this
            # path. Legacy (m>1 / loccap) keeps setup-time planning and
            # emits timing_accounting=legacy_untimed_plan.
            ip = None
            if iter_planner is not None:
                ip = iter_planner.derive(loads_gather_buf)
                runner.bind_iter_plan(ip)
            ev["plan"][i].record()
            if migration != "off":
                t_mig = time.perf_counter()
                if iter_planner is not None:
                    # rule-5 fidelity fix: the decision consumes THIS
                    # iteration's derived loads (GPU decision, one small
                    # D2H), not the static plan tensors.
                    swaps = plan_migration_swaps_gpu(
                        iter_planner.p2l, ip.slot_loads, tau_tokens,
                        runner.ranks_per_node, runner.cfg.nlp,
                        runner.cfg.G)
                else:
                    swaps = plan_migration_swaps(
                        runner.plan, tau_tokens, runner.ranks_per_node)
                if swaps and migration == "inkernel":
                    # Paper-faithful §4.3: host decision + index rebuild
                    # only; the weight exchange runs fused as phase 0 of
                    # this iteration's group-0 hc dispatch (no NCCL, no
                    # sync — the prior epoch-close barrier already fences
                    # in-flight iterations from the scratch).
                    recv_b, relayout_ms = runner.apply_migration_inkernel(
                        swaps)
                    relayout_ms_total += relayout_ms
                    mig_mine_per_iter.append(1 if recv_b > 0 else 0)
                elif swaps:
                    # A swap stalls the pipeline (launch-granularity port of
                    # EPIC's in-kernel swap phase). Sync so the exchange
                    # cannot race in-flight prior iterations in e2e mode.
                    torch.cuda.synchronize()
                    _, relayout_ms = runner.apply_migration(swaps)
                    relayout_ms_total += relayout_ms
                else:
                    if migration == "inkernel":
                        mig_mine_per_iter.append(0)
                if swaps and iter_planner is not None:
                    # the swap mutated plan.p2l/l2p (host apply +
                    # rebuild): refresh the planner's placement and
                    # re-derive so the bound plan matches the new layout
                    # (swap iterations converge in warmup under static
                    # per-cell routing — timed iterations pay the
                    # decision only).
                    iter_planner.refresh_placement()
                    ip = iter_planner.derive(loads_gather_buf)
                    runner.bind_iter_plan(ip)
                mig_swaps_per_iter.append(len(swaps))
                mig_host_times.append((time.perf_counter() - t_mig) * 1e3)
            ev["mig"][i].record()
            _hb(f"[pll-hb] r{runner.rank} iter {i} planned", flush=True)
            if _rand_payload:
                # regression guard (2026-08-22): per-iteration payload
                # change makes stale-delivery visible even under identical
                # routing metadata (static payloads masked the relay-pull
                # race); check_correctness allgathers the FINAL shard, so
                # the deterministic final iteration stays consistent
                ctx.inputs_shard.copy_(
                    (torch.rand(ctx.inputs_shard.shape,
                                device=ctx.inputs_shard.device,
                                generator=_rand_gen) * 0.01)
                    .to(ctx.inputs_shard.dtype))
                ctx._pll_payloads.append(ctx.inputs_shard.clone())
            runner.pack(ctx.inputs_shard, probs_shard)
            ev["pack"][i].record()
            if single_stream:
                for g in range(m):
                    disp_fn(g)
                    ev[f"disp{g}"][i].record()
                    ev[f"gate{g}"][i].record()
                    runner.scatter_group(g)
                    ev[f"scat{g}"][i].record()
                    runner.gemm_group(g, sm_margin=sm_margin)
                    ev[f"gemm{g}"][i].record()
                    if l01:
                        runner.act_group(g)
                        ev[f"act{g}"][i].record()
                        runner.gemm1_group(g, sm_margin=sm_margin)
                        ev[f"gemm1_{g}"][i].record()
                        runner.combine_pack_group(g)
                        ev[f"cpack{g}"][i].record()
                        runner.combine_group(g)
                        ev[f"comb{g}"][i].record()
                        runner.accumulate_group(g)
                        ev[f"acc{g}"][i].record()
                if l01:
                    runner.finalize_sum()
                    ev["sum_end"][i].record()
            else:
                comm_stream.wait_event(ev["pack"][i])
                with torch.cuda.stream(comm_stream):
                    for g in range(m):
                        disp_fn(g)
                        ev[f"disp{g}"][i].record()
                for g in range(m):
                    torch.cuda.current_stream().wait_event(ev[f"disp{g}"][i])
                    ev[f"gate{g}"][i].record()
                    runner.scatter_group(g)
                    ev[f"scat{g}"][i].record()
                    runner.gemm_group(g, sm_margin=sm_margin)
                    ev[f"gemm{g}"][i].record()
                    if l01:
                        runner.act_group(g)
                        ev[f"act{g}"][i].record()
                        runner.gemm1_group(g, sm_margin=sm_margin)
                        ev[f"gemm1_{g}"][i].record()
                        # combine for group g rides the comm stream, gated
                        # on this group's GEMM1 (compute staging kept by
                        # comm-stream in-order execution).
                        comm_stream.wait_event(ev[f"gemm1_{g}"][i])
                        with torch.cuda.stream(comm_stream):
                            runner.combine_pack_group(g)
                            ev[f"cpack{g}"][i].record()
                            runner.combine_group(g)
                            ev[f"comb{g}"][i].record()
                            runner.accumulate_group(g)
                            ev[f"acc{g}"][i].record()
                if l01:
                    torch.cuda.current_stream().wait_event(
                        ev[f"acc{m - 1}"][i])
                    runner.finalize_sum()
                    ev["sum_end"][i].record()

    keys = (
        ["plan_comm_ms", "place_ms", "plan_ms", "migration_ms", "pack_ms",
         "comm_ms", "e2e_ms", "total_ms"]
        + [f"disp{g}_ms" for g in range(m)]
        + [f"stall{g}_ms" for g in range(m)]
        + [f"scat{g}_ms" for g in range(m)]
        + [f"gemm{g}_ms" for g in range(m)]
    )
    if l01:
        # combined-driver-compatible names (check_l01_identity.py reads
        # e2e_ms + act_ms; l1_ms = e2e - l0 - act by construction).
        keys += (
            ["l0_ms", "act_ms", "l1_ms", "sum_ms"]
            + [f"act{g}_ms" for g in range(m)]
            + [f"gemm1_{g}_ms" for g in range(m)]
            + [f"cpack{g}_ms" for g in range(m)]
            + [f"comb{g}_ms" for g in range(m)]
            + [f"acc{g}_ms" for g in range(m)]
        )
    times = {k: [] for k in keys}
    last = "sum_end" if l01 else f"gemm{m - 1}"
    for i in range(total_iters):
        ev[last][i].synchronize()
        if i < warmup_iters:
            continue
        times["plan_comm_ms"].append(
            ev["start"][i].elapsed_time(ev["plan_comm"][i]))
        times["place_ms"].append(
            ev["plan_comm"][i].elapsed_time(ev["place"][i]))
        times["plan_ms"].append(
            ev["place"][i].elapsed_time(ev["plan"][i]))
        times["migration_ms"].append(
            ev["plan"][i].elapsed_time(ev["mig"][i]))
        times["pack_ms"].append(ev["mig"][i].elapsed_time(ev["pack"][i]))
        times["comm_ms"].append(
            ev["pack"][i].elapsed_time(ev[f"disp{m - 1}"][i]))
        times["e2e_ms"].append(ev["mig"][i].elapsed_time(ev[last][i]))
        times["total_ms"].append(ev["start"][i].elapsed_time(ev[last][i]))
        prev_disp = ev["pack"][i]
        for g in range(m):
            times[f"disp{g}_ms"].append(
                prev_disp.elapsed_time(ev[f"disp{g}"][i]))
            prev_disp = ev[f"disp{g}"][i]
            prev_compute = ev["pack"][i] if g == 0 else ev[f"gemm{g - 1}"][i]
            times[f"stall{g}_ms"].append(
                prev_compute.elapsed_time(ev[f"gate{g}"][i]))
            times[f"scat{g}_ms"].append(
                ev[f"gate{g}"][i].elapsed_time(ev[f"scat{g}"][i]))
            times[f"gemm{g}_ms"].append(
                ev[f"scat{g}"][i].elapsed_time(ev[f"gemm{g}"][i]))
        if l01:
            act_total = 0.0
            for g in range(m):
                span = ev[f"gemm{g}"][i].elapsed_time(ev[f"act{g}"][i])
                times[f"act{g}_ms"].append(span)
                act_total += span
                times[f"gemm1_{g}_ms"].append(
                    ev[f"act{g}"][i].elapsed_time(ev[f"gemm1_{g}"][i]))
                # anchored at the gating event: queueing delay on the comm
                # stream + the pack work itself
                times[f"cpack{g}_ms"].append(
                    ev[f"gemm1_{g}"][i].elapsed_time(ev[f"cpack{g}"][i]))
                times[f"comb{g}_ms"].append(
                    ev[f"cpack{g}"][i].elapsed_time(ev[f"comb{g}"][i]))
                times[f"acc{g}_ms"].append(
                    ev[f"comb{g}"][i].elapsed_time(ev[f"acc{g}"][i]))
            l0_span = ev["mig"][i].elapsed_time(ev[f"gemm{m - 1}"][i])
            e2e_span = times["e2e_ms"][-1]
            times["l0_ms"].append(l0_span)
            times["act_ms"].append(act_total)
            times["l1_ms"].append(e2e_span - l0_span - act_total)
            times["sum_ms"].append(
                ev[f"acc{m - 1}"][i].elapsed_time(ev["sum_end"][i]))
    if isolated:
        times["iso_sync_ms"] = iso_sync_times[warmup_iters:]
    if migration != "off":
        times["migration_host_ms"] = mig_host_times[warmup_iters:]
    if migration == "inkernel":
        # Always-on swap-phase timing (cudaEvent pair around each fused
        # launch; same-stream elapsed = kernel residency = snapshot +
        # peer-wait + pull — under the sequential-phases design that IS the
        # exposed cost). Values arrive per-launch in launch order; expand to
        # per-iteration (0.0 when this rank launched nothing).
        launch_ms = runner._hc_ops[0].collect_swap_times()
        assert len(launch_ms) == sum(mig_mine_per_iter), (
            len(launch_ms), sum(mig_mine_per_iter))
        it_ms = iter(launch_ms)
        per_iter = [next(it_ms) if mine else 0.0
                    for mine in mig_mine_per_iter]
        times["swap_fused_ms"] = per_iter[warmup_iters:]
    mig_facts = {
        "swaps_total": sum(mig_swaps_per_iter),
        "swaps_timed": sum(mig_swaps_per_iter[warmup_iters:]),
        "rounds_to_converge": next(
            (n for n, c in enumerate(mig_swaps_per_iter) if c == 0),
            len(mig_swaps_per_iter),
        ) if migration != "off" else 0,
        "relayout_ms_total": relayout_ms_total,
    }
    return times, mig_facts


def check_correctness(runner, ctx, plan, topk_all, w_all, atol, rtol):
    """Bitwise dispatch/scatter + wire-quota audit + placed/migrated weight
    bitwise check + per-group GEMM allclose vs torch.matmul (collective)."""
    from flux.testing.ultraep_semantics import reroute_expand

    cfg = runner.cfg
    rank, R, S = runner.rank, cfg.R, cfg.S
    ok_bitwise = True
    if int(os.environ.get("FLUX_PLL_DEBUG", "0")):
        print(f"[pll-hb] r{rank} check_correctness enter", flush=True)

    # materialize the replicated inputs ONLY here (ctx opts out of the
    # resident [ntokens, h] copy — 7 GB @ K3 b56 32n); freed on return
    inputs_full = torch.empty(
        ctx.ntokens, ctx.inputs_shard.shape[1],
        dtype=ctx.inputs_shard.dtype, device=ctx.inputs_shard.device)
    torch.distributed.all_gather_into_tensor(
        inputs_full, ctx.inputs_shard, group=TP_GROUP
    )

    expected_hidden = torch.zeros_like(runner.hidden_buf)
    expected_probs = torch.zeros_like(runner.weights_buf)
    seg_fill = list(runner.elay.seg_start)
    n_rows_check = 0
    per_instance_rows = [0] * cfg.nlp
    _dbg = int(os.environ.get("FLUX_PLL_DEBUG", "0"))
    # vectorized expectation build (2026-08-21): the per-row python loop
    # was O(n_recv) GPU row-copies — minutes at K3 shape; this is seconds.
    dev_e = expected_hidden.device
    p2l_dev = plan.p2l.long().to(dev_e)
    for src in range(R):
        if _dbg:
            print(f"[pll-hb] r{rank} check src {src}", flush=True)
        tok, phys = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(phys * (S + 1) + tok, stable=True)
        tok, phys = tok[order], phys[order]
        msk = (phys // cfg.nlp) == rank
        tok, phys = tok[msk], phys[msk]
        if tok.numel() == 0:
            continue
        pl = (phys - rank * cfg.nlp).to(dev_e)
        tok_d = tok.to(dev_e)
        # rows arrive sorted by phys -> contiguous per-p_local runs;
        # slot = current seg_fill[p_local] + ordinal within the run
        first = torch.ones_like(pl, dtype=torch.bool)
        first[1:] = pl[1:] != pl[:-1]
        idx = torch.arange(pl.numel(), device=dev_e)
        starts = torch.where(first, idx, torch.zeros_like(idx))
        ordinal = idx - torch.cummax(starts, 0).values
        base = torch.tensor(seg_fill, dtype=torch.int64, device=dev_e)
        slot = base[pl] + ordinal
        expected_hidden[slot] = inputs_full[src * S + tok_d]
        logical = p2l_dev[phys.to(dev_e)]
        expected_probs[slot] = (
            w_all[src].to(dev_e)[tok_d, logical].to(expected_probs.dtype))
        cnt_pl = torch.bincount(pl, minlength=cfg.nlp).cpu()
        for b in range(cfg.nlp):
            c = int(cnt_pl[b])
            seg_fill[b] += c
            per_instance_rows[b] += c
        n_rows_check += int(tok.numel())

    assert n_rows_check == runner.n_recv, (
        f"rank {rank}: recomputed {n_rows_check} rows != runner {runner.n_recv}"
    )

    # Wire-quota audit. Under --router d6 the allocation IS the D6 table;
    # under a per-token router (plan.phys_override) the table is not the
    # allocation — audit the expansion against the override's own per-slot
    # counts instead (a cross-check of _expand_from_phys vs the raw
    # routing, not a tautology: the expansion path re-derives entries).
    ov = getattr(plan, "phys_override", None)
    if ov is not None:
        counts = torch.bincount(ov.reshape(-1).long(), minlength=cfg.P)
        for b in range(cfg.nlp):
            p = rank * cfg.nlp + b
            if int(plan.p2l[p]) < 0:
                continue
            expect = int(counts[p])
            got = per_instance_rows[b]
            if got != expect:
                ok_bitwise = False
                print(f"❌ rank {rank}: slot {b} rows {got} != router "
                      f"override count {expect}")
    else:
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
                        print(f"❌ rank {rank}: instance ({l},{j}) rows "
                              f"{got} != D6 rank-quota total {expect}")

    if not torch.equal(runner.hidden_buf[:runner.n_recv],
                       expected_hidden[:runner.n_recv]):
        ok_bitwise = False
        print(f"❌ rank {rank}: dispatched rows differ from plan prediction")
        if os.environ.get("FLUX_EPIC_CHECK_PROBE", "0") == "1":
            got = runner.hidden_buf[:runner.n_recv]
            exp = expected_hidden[:runner.n_recv]
            bad = (got != exp).any(dim=1)
            n_bad = int(bad.sum())
            zero_bad = int((got[bad] == 0).all(dim=1).sum())
            # multiset check: is the buffer a permutation of the expectation?
            gs = got.float().sum(dim=1).sort().values
            es = exp.float().sum(dim=1).sort().values
            perm = bool(torch.allclose(gs, es))
            # provenance of expected content for the bad rows: which source
            # NODE was supposed to fill them
            src_node = torch.full((expected_hidden.shape[0],), -1,
                                  dtype=torch.int64)
            fill_p = list(runner.elay.seg_start)
            for src in range(cfg.R):
                tok_p, phys_p = reroute_expand(cfg, plan, src,
                                               topk_all[src])
                order_p = torch.argsort(phys_p * (S + 1) + tok_p,
                                        stable=True)
                phys_p = phys_p[order_p]
                msk_p = (phys_p // cfg.nlp) == rank
                for p in phys_p[msk_p].tolist():
                    pl = p - rank * cfg.nlp
                    src_node[fill_p[pl]] = src // runner.ranks_per_node
                    fill_p[pl] += 1
            from collections import Counter
            bad_idx = bad.nonzero(as_tuple=True)[0].cpu()
            cnt = Counter(src_node[bad_idx].tolist())
            print(f"  probe rank {rank}: bad {n_bad}/{runner.n_recv} rows, "
                  f"{zero_bad} all-zero, permutation={perm}, "
                  f"bad-by-src-node {dict(sorted(cnt.items()))}")
            # provenance vs the payload ledger (FLUX_PLL_RANDOM_PAYLOAD):
            # which earlier payload do the bad rows carry? Also counts the
            # T2 poison sentinel (0xA5A5 in every element).
            ledger = getattr(ctx, "_pll_payloads", None)
            if ledger is not None and n_bad > 0:
                # per-row expected source index (src*S + tok), rebuilt from
                # the same expansion the expectation used
                src_row = torch.full((expected_hidden.shape[0],), -1,
                                     dtype=torch.int64)
                fill_q = list(runner.elay.seg_start)
                for src in range(cfg.R):
                    tok_q, phys_q = reroute_expand(cfg, plan, src,
                                                   topk_all[src])
                    order_q = torch.argsort(phys_q * (S + 1) + tok_q,
                                            stable=True)
                    tok_q, phys_q = tok_q[order_q], phys_q[order_q]
                    msk_q = (phys_q // cfg.nlp) == rank
                    for t, p in zip(tok_q[msk_q].tolist(),
                                    phys_q[msk_q].tolist()):
                        pl = p - rank * cfg.nlp
                        src_row[fill_q[pl]] = src * S + t
                        fill_q[pl] += 1
                bad_src = src_row[bad_idx].to(got.device)
                got_bad = got[bad_idx.to(got.device)]
                prov = {}
                for k, pay in enumerate(ledger):
                    full_k = torch.empty(
                        ctx.ntokens, pay.shape[1], dtype=pay.dtype,
                        device=pay.device)
                    torch.distributed.all_gather_into_tensor(
                        full_k, pay, group=TP_GROUP)
                    prov[k] = int((got_bad == full_k[bad_src]).all(
                        dim=1).sum())
                    del full_k
                sent = int((got_bad.view(torch.int16) == -23131).all(
                    dim=1).sum())
                print(f"  probe rank {rank}: bad-by-payload {prov} "
                      f"(ledger {len(ledger)}, last = expected), "
                      f"poison-sentinel {sent}", flush=True)
    if not torch.equal(runner.weights_buf[:runner.n_recv],
                       expected_probs[:runner.n_recv]):
        ok_bitwise = False
        print(f"❌ rank {rank}: route probs differ from plan prediction")

    # Weights: every local slot holds the canonical weights of its CURRENT
    # logical expert (post-placement, post-migration), bitwise.
    p2l = plan.p2l.long()
    for b in range(cfg.nlp):
        l = int(p2l[rank * cfg.nlp + b])
        if l < 0:
            continue
        exp1 = runner.make_canonical_fc1(l).to(runner.device)
        if not torch.equal(runner.slot_fc1[b], exp1):
            ok_bitwise = False
            print(f"❌ rank {rank}: fc1 slot {b} (expert {l}) mismatch")
        if runner.place_fc2:
            exp2 = runner.make_canonical_fc2(l).to(runner.device)
            if not torch.equal(runner.slot_fc2[b], exp2):
                ok_bitwise = False
                print(f"❌ rank {rank}: fc2 slot {b} (expert {l}) mismatch")

    # Per-group GEMM outputs vs torch.matmul per segment.
    ok_allclose = True
    seg_start = runner.elay.seg_start
    for grp in runner.elay.groups:
        out_g = runner.group_outputs[grp.g]
        if out_g is None:
            assert sum(grp.seg_rows) == 0
            continue
        base = seg_start[grp.slot_lo]
        for p, start, end, logical in runner.elay.gemm_segments:
            if not (grp.slot_lo <= p < grp.slot_hi):
                continue
            ref = torch.matmul(
                runner.hidden_buf[start:end].float(),
                runner.slot_fc1[p].float().t(),
            ).to(out_g.dtype)
            try:
                flux.torch_allclose(out_g[start - base:end - base], ref,
                                    atol=atol, rtol=rtol)
            except Exception:
                ok_allclose = False
                print(f"❌ rank {rank}: gemm group {grp.g} slot={p} "
                      f"expert={logical} mismatch")

    # Full-journey check (--layers l01): final [S, H] vs a torch chain from
    # the canonical generators, mimicking the pipeline's dtype casts stage
    # by stage (GEMM0 out bf16 -> gelu bf16 -> GEMM1 out bf16 -> fp32 scale
    # -> bf16 wire) so tolerances stay the house bf16 thresholds. Home-side
    # accumulation in fp32. Staging coverage is structurally guaranteed by
    # the comb_dst_slot permutation assert in the layout builder.
    if runner.layers == "l01":
        topk = topk_all[rank].long()

        def _l01_ref(x_shard):
            x = x_shard.float()
            ref = torch.zeros(cfg.S, cfg.H, dtype=torch.float32,
                              device=runner.device)
            for l in range(cfg.G):
                toks = (topk == l).any(dim=1).nonzero(as_tuple=True)[0]
                if toks.numel() == 0:
                    continue
                toks_dev = toks.to(runner.device)
                fc1 = runner.make_canonical_fc1(l).to(runner.device).float()
                fc2 = runner.make_canonical_fc2(l).to(runner.device).float()
                h0 = (x[toks_dev] @ fc1.t()).to(runner.dtype)
                a = torch.nn.functional.gelu(h0)
                y1 = (a.float() @ fc2.t()).to(runner.dtype)
                wgt = w_all[rank][toks, l].to(runner.device).unsqueeze(1)
                contrib = (y1.float() * wgt).to(runner.dtype).float()
                ref.index_add_(0, toks_dev, contrib)
            return ref

        ref = _l01_ref(ctx.inputs_shard)
        try:
            flux.torch_allclose(runner.final_out.float(), ref,
                                atol=atol, rtol=rtol)
        except Exception:
            ok_allclose = False
            out = runner.final_out.float()
            diff = (out - ref).abs()
            viol = diff > (atol + rtol * ref.abs())
            bad = viol.any(dim=1)
            msg = (f"❌ rank {rank}: l01 final output mismatch "
                   f"(max abs {float(diff.max()):.4f}; bad token rows "
                   f"{int(bad.sum())}/{cfg.S}, first {bad.nonzero(as_tuple=True)[0][:6].tolist()}, "
                   f"|ref| scale {float(ref.abs().mean()):.4g})")
            ledger = getattr(ctx, "_pll_payloads", None)
            if ledger is not None and len(ledger) >= 2 and int(bad.sum()) > 0:
                # provenance: do the bad rows equal the l01 chain on the
                # PREVIOUS payload (stale-by-one) ?
                ref_prev = _l01_ref(ledger[-2])
                close_prev = ((out[bad] - ref_prev[bad]).abs()
                              <= (atol + rtol * ref_prev[bad].abs())).all(dim=1)
                # or a MIX: the final payload's own contribution missing/duplicated?
                msg += (f"; of bad rows {int(close_prev.sum())} match the PREVIOUS "
                        f"payload's chain (stale-by-one), {int((~close_prev).sum())} neither")
            print(msg, flush=True)

    status = "✅" if (ok_bitwise and ok_allclose) else "❌"
    what = ("full journey" if runner.layers == "l01" else "gemm")
    print(f"{status} rank {rank}: epic dispatch content "
          f"{'bitwise-exact' if ok_bitwise else 'MISMATCH'}, "
          f"{what} {'allclose' if ok_allclose else 'MISMATCH'}")
    RECORDER.emit_correctness(bitwise=ok_bitwise, allclose=ok_allclose)
    # COLLECTIVE verdict (2026-08-21): a partial per-rank assert leaves the
    # surviving ranks wedged in the next barrier and the srun step alive —
    # reduce the flag so every rank raises (or passes) together and
    # torchrun tears down cleanly.
    flag = torch.tensor([int(ok_bitwise and ok_allclose)], device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN,
                                 group=TP_GROUP)
    assert int(flag) == 1, "correctness failed on at least one rank"
    return ok_bitwise and ok_allclose


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)  # exact bytes; numpy has no bf16
    return t.numpy().tobytes()


def run_one_forward(runner, ctx, probs_shard, sm_margin):
    """One untimed single-stream forward (the loccap_sl arm's FINAL
    DETERMINISTIC iteration: the runner is bound to the setup-reference
    routing before this call, so the output validates against
    plan.phys_override)."""
    disp_fn = (runner.dispatch_group_hc if runner.hc_enabled
               else runner.dispatch_group)
    runner.pack(ctx.inputs_shard, probs_shard)
    for g in range(runner.m):
        disp_fn(g)
        runner.scatter_group(g)
        runner.gemm_group(g, sm_margin=sm_margin)
        if runner.layers == "l01":
            runner.act_group(g)
            runner.gemm1_group(g, sm_margin=sm_margin)
            runner.combine_pack_group(g)
            runner.combine_group(g)
            runner.accumulate_group(g)
    if runner.layers == "l01":
        runner.finalize_sum()
    torch.cuda.synchronize()


def output_sha(runner) -> str:
    """Digest of the received rows + all group GEMM outputs (cross-m and
    single-stream-vs-two-stream identity lever). Under l01 the digest is of
    the FINAL [S, H] output (bitwise m-invariant by the staging design) —
    emitted under a DIFFERENT info key (epic_out_sha_l01) so cross-layer
    sha comparisons are impossible by construction."""
    h = hashlib.sha256()
    torch.cuda.synchronize()
    if runner.layers == "l01":
        h.update(_tensor_bytes(runner.final_out))
        return h.hexdigest()[:16]
    h.update(_tensor_bytes(runner.hidden_buf[:runner.n_recv]))
    for out in runner.group_outputs:
        if out is not None:
            h.update(_tensor_bytes(out))
    return h.hexdigest()[:16]


def probe_swap_ms(runner, n_probe: int = 3) -> float:
    """Setup-time probe: wall-clock one representative intra-node slot
    exchange between local ranks 0 and 1 (all ranks participate in the
    barrier so the number is comparable). Returns ms (0.0 if no local
    peer)."""
    dist = torch.distributed
    lw = DIST_ENV.LOCAL_WORLD_SIZE
    if lw < 2:
        return 0.0
    node = runner.rank // lw
    r0, r1 = node * lw, node * lw + 1
    buf = torch.empty_like(runner.slot_fc1[0])
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_probe):
        if runner.rank == r0:
            ops = [dist.P2POp(dist.isend, runner.slot_fc1[0].contiguous(),
                              peer=r1),
                   dist.P2POp(dist.irecv, buf, peer=r1)]
        elif runner.rank == r1:
            ops = [dist.P2POp(dist.isend, runner.slot_fc1[0].contiguous(),
                              peer=r0),
                   dist.P2POp(dist.irecv, buf, peer=r0)]
        else:
            ops = []
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / n_probe
    dist.barrier()
    if runner.place_fc2:
        ms *= 2.0  # a real swap moves fc1 + fc2
    return ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic_matrix", type=str, required=True)
    parser.add_argument("--routing_file", type=str, default=None)
    parser.add_argument("--oracle_routing_file", type=str, default=None,
                        help="scenario-1 placement oracle: solve the "
                        "static placement on THIS routing (the sampled "
                        "previous batch) instead of the evaluated batch "
                        "(placement placelambda_fast only; the dynamic "
                        "lane still observes the evaluated batch — its "
                        "trigger then measures oracle->realized drift)")
    parser.add_argument("--chunk_bytes", type=int, default=8192)
    parser.add_argument("--H", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--G", type=int, default=32)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup_iters", default=10, type=int)
    parser.add_argument("--sm_margin", default=0, type=int,
                        help="forwarded to GemmGroupedV2.forward")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--profile", default=False, action="store_true")
    # --- EPIC knobs ---
    parser.add_argument("--groups", type=int, default=2, choices=[1, 2, 4],
                        help="m: PEO expert groups (paper's values; m=1 = "
                        "no-overlap anchor)")
    parser.add_argument("--layers", default="l0", choices=["l0", "l01"],
                        help="l0 = dispatch + GEMM0 (v1); l01 = full journey"
                        " dispatch -> GEMM0 -> GELU -> GEMM1 -> per-group"
                        " combine -> terminal Sum (EPIC Fig 10(b))")
    parser.add_argument("--placement", default="epic",
                        choices=["none", "epic", "nodeaware",
                                 "placelambda_gpu", "placelambda_fast"],
                        help="epic = §4.2 (redundancy + GPU greedy + NIC "
                        "stage) from the pool load; none = fixed contiguous "
                        "homing, empty redundant slots; nodeaware = "
                        "PLACE-lambda co-occurrence partition + per-node-"
                        "first coverage replication from --placement_file")
    parser.add_argument("--placement_file", type=str, default=None,
                        help="<mid>.placement.json sidecar "
                        "(sweeps/predict_placement.py); required for "
                        "--placement nodeaware")
    parser.add_argument("--hc_meta", default="inwindow",
                        choices=["inwindow", "python"],
                        help="hc metadata derivation (campaign-2 v2b)."
                        " inwindow = CANONICAL: the op derives splits/"
                        "stable-scatter/splits_per_source/unique_counts"
                        " on device inside the dispatch bracket"
                        " (dispatch path calls derive_routed_meta; the"
                        " combine op rebuilds its index sets internally);"
                        " python = the campaign-1 torch-op derive, kept"
                        " for debug/A-B (requires the same binary)")
    parser.add_argument("--replica_select", default="local_static",
                        choices=["local_static", "local_spread"],
                        help="replica rule: local_static = the paper's"
                        " own D6 src-mod-C (DEFAULT — authentic EPIC);"
                        " local_spread = per-source equal split (SGLang"
                        " dynamic analog, ablation). Both sender-local.")
    parser.add_argument("--router", default="d6",
                        choices=["d6", "loccap", "evensplit", "loccap_gpu",
                                 "loccap_sl"],
                        help="replica selection: d6 = src mod lcnts (the "
                        "EPIC baseline rule; re-derived per iteration on "
                        "device under SCHEMA rule 5); loccap = per-token "
                        "tiered locality under compute caps (1+eps)*S*K, "
                        "the token-node-incidence minimizer — still a "
                        "once-per-cell python port (legacy_untimed_plan "
                        "accounting; not quotable in new-accounting "
                        "capsules until its GPU port lands); loccap_gpu = "
                        "the PLACE-lambda bounded-round device port "
                        "(flux.testing.placelambda_gpu) — rule-5 path, "
                        "re-derived per iteration in-window (plan_ms); "
                        "loccap_sl = the RELAXED sender-local FUSED KERNEL "
                        "(flux.placelambda_route_sl) — per-iteration "
                        "kernel + phys-row allgather in plan_ms, routing "
                        "varies legitimately, verified by invariants + "
                        "provable table bounds + a final deterministic "
                        "correctness iteration")
    parser.add_argument("--eps", type=float, default=0.25,
                        help="LocCap balance slack; 'inf' = pure locality")
    parser.add_argument("--l1_n_split", type=int, default=4,
                        help="l01 combine split-N pipeline depth (the sweep "
                        "passes the shape preset's n_split_l1: K3 -> 7, "
                        "n_per=512; H//n_split must be 8-aligned)")
    parser.add_argument("--pll_f_cap", type=int, default=0,
                        help="loccap_sl per-(src,dst) forced-admission "
                        "budget — the ONE sizing clamp that makes the "
                        "kernel arm's pair bounds provable; the kernel's "
                        "overflow counter must stay 0 (asserted). 0 = "
                        "AUTO from the reference forced-flow table "
                        "(2x max per-pair forced + 8; real-K3-measured)")
    parser.add_argument("--place_dynamic", default="static",
                        choices=["static", "dynamic"],
                        help="PLACE-lambda ablation toggle (placement "
                        "placelambda_gpu only). static = ideal stale "
                        "placement: one untimed setup solve "
                        "(place_solver_ms fact). dynamic = the placement "
                        "decision is part of the optimization under test: "
                        "per-iteration in-window solve + move diff + "
                        "trigger threshold, timed as place_ms")
    parser.add_argument("--place_gain_threshold_ppm", type=int,
                        default=50_000,
                        help="dynamic-placement trigger: fire when the "
                        "fresh solve's incidence-bound gain over the "
                        "resident placement clears this (ppm)")
    parser.add_argument("--migration", default="off",
                        choices=["off", "on", "inkernel"],
                        help="§4.3 per-step intra-host expert migration. "
                        "on = host NCCL weight exchange (launch-granularity "
                        "port; exchange cost lands in migration_ms). "
                        "inkernel = paper-faithful fused swap: the exchange "
                        "kernel runs as phase 0 of the group-0 hc dispatch "
                        "launch (weights complete, then the token wire; no "
                        "host sync) — hc transport only; the swap cost "
                        "lands INSIDE e2e_ms/disp0 and is also reported "
                        "as swap_fused_ms (compare arms on total_ms)")
    parser.add_argument("--tau_tokens", type=float, default=None,
                        help="migration gain threshold in tokens (overrides "
                        "--t_swap_ms/--t_token_us)")
    parser.add_argument("--t_swap_ms", type=float, default=None,
                        help="swap cost for tau = t_swap/t_token (absent = "
                        "the setup probe's measurement)")
    parser.add_argument("--t_token_us", type=float, default=None,
                        help="per-token expert compute time for tau")
    parser.add_argument("--epic_load_file", type=str, default=None,
                        help="predicted per-expert load JSON (the "
                        "<mid>.eplb_load.json sidecar; same convention as "
                        "the eplb arm). Absent = batch self-oracle (smoke)")
    parser.add_argument("--redundant_per_rank", type=int, default=2,
                        help="slot headroom (D2: togglable for MoonEP-"
                        "comparable configurations)")
    parser.add_argument("--weight_place", default="fc1fc2",
                        choices=["fc1fc2", "fc1"])
    parser.add_argument("--transport", default="hier_compress",
                        choices=["nccl", "nvshmem", "hier_compress"],
                        help="hier_compress (DEFAULT — EPIC's own Mode 2, "
                        "PXN relay + de-redundancy) = fused-op "
                        "dispatch_only over the virtual slot space (probs "
                        "ride the nvshmem side-wire; needs a post-S2 "
                        "binary); nvshmem = direct per-entry wire (Mode-1/"
                        "DeepEP-default analog); nccl = debug/parity "
                        "fallback only, never a faithful EPIC arm")
    parser.add_argument("--hc_relay", default="identity",
                        choices=["identity", "balanced"],
                        help="hier_compress relay shape: identity = "
                        "same-index-GPU PXN (faithful Mode 2), balanced = "
                        "our chunked-relay ablation")
    parser.add_argument("--hc_wire", default="relay_identity",
                        choices=["relay_identity", "lb_union"],
                        help="hier_compress dispatch wire: relay_identity = "
                        "faithful Mode-2 PXN gateway; lb_union = the Tier-B "
                        "fused lb_union wire (balanced chunked inter-node + "
                        "union-broadcast gateway) over the same virtual "
                        "slot space — the replicas x lb_union integration "
                        "arm")
    parser.add_argument("--hc_headroom", type=float, default=1.5,
                        help="capacity headroom for the per-group fused-op "
                        "instances (migration-proofing)")
    parser.add_argument("--num_comm_sm", type=int, default=8)
    parser.add_argument("--a2a_split_headroom", type=float, default=2.0,
                        help="All2AllSingle max_split = headroom * initial "
                        "max per-(group, pair) rows (migration can reshape "
                        "the wire; the runner asserts on overflow)")
    parser.add_argument("--gemm_backend", default="grouped",
                        choices=["grouped", "gemmonly"],
                        help="grouped = one GemmGroupedV2 per group "
                        "(faithful); gemmonly = per-segment loop (debug / "
                        "eplb-anchor parity)")
    parser.add_argument("--single_stream", default=False, action="store_true",
                        help="debug: serialize dispatch+compute on one "
                        "stream (output must be identical)")
    parser.add_argument("--no_interleave", default=False, action="store_true")
    parser.add_argument("--skip_correctness", default=False,
                        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_ep_group(DIST_ENV.WORLD_SIZE)
    if args.transport == "hier_compress" and not hasattr(
            flux.GemmGroupedV2AGScatterOp, "dispatch_only"):
        print("SKIP: this libflux binary lacks "
              "GemmGroupedV2AGScatterOp.dispatch_only (pre-S2 build); "
              "hier_compress arms need a rebuilt binary")
        sys.exit(0)
    if args.transport in ("nvshmem", "hier_compress"):
        assert DTYPE_MAP[args.dtype] != torch.float16, (
            "All2AllSingle instantiates BF16/FP32 only"
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
        choosed_experts = load_routing_file(args.routing_file, args.G,
                                            args.topk)
        assert choosed_experts.shape[0] % W == 0
        got = choosed_experts_to_matrix_chunks(choosed_experts, W,
                                               args.G // W)
        assert torch.equal(got * args.chunk_bytes, matrix)
        choosed_experts = choosed_experts.cuda()
        if TP_GROUP.rank() == 0:
            print(f"routing: REAL trace file {args.routing_file}")
    else:
        choosed_experts = traffic_matrix_to_choosed_experts(
            matrix, args.G, args.topk, args.chunk_bytes
        )
        if TP_GROUP.rank() == 0:
            print("routing: synthetic dealer assignment (max-dedup"
                  " construction; prefer trace-family cells)")
    ntokens = choosed_experts.shape[0]
    S = ntokens // W
    rank = TP_GROUP.rank()

    cfg = UltraEPConfig(
        S=S, K=args.topk, G=args.G, R=W, H=args.H,
        D=DIST_ENV.LOCAL_WORLD_SIZE,
        R_red=args.redundant_per_rank,
        locality_aware=False,
        interleave=not args.no_interleave,
    )

    topk_all = choosed_experts.reshape(W, S, args.topk).cpu().int()
    tpe = loads_from_topk(cfg, topk_all)

    if args.epic_load_file:
        with open(args.epic_load_file, "rb") as f:
            raw = f.read()
        blob = json.loads(raw)
        assert blob.get("version") == 1
        assert int(blob["G"]) == args.G
        pool_load = blob["load"]
        assert len(pool_load) == args.G
        load_source = "pool"
        load_sha = hashlib.sha256(raw).hexdigest()[:16]
    else:
        pool_load = tpe.long().sum(0).tolist()
        load_source = "batch"
        load_sha = ""
        if rank == 0 and args.placement == "epic":
            print("epic: NO --epic_load_file; placement from the batch's own"
                  " load (self-oracle — fine for smoke, not a headline cell)")

    placement_sha = ""
    t0 = time.perf_counter()
    if args.placement == "epic":
        plan = build_epic_plan(cfg, tpe, pool_load, num_nodes,
                               replica_select=args.replica_select)
    elif args.placement == "nodeaware":
        assert args.placement_file, "--placement nodeaware needs --placement_file"
        assert args.routing_file, (
            "--placement nodeaware is defined for trace cells only "
            "(--routing_file; the sidecar was simulated on that routing)")
        with open(args.placement_file, "rb") as f:
            raw_p = f.read()
        pblob = json.loads(raw_p)
        placement_sha = hashlib.sha256(raw_p).hexdigest()[:16]
        plan = build_nodeaware_plan(cfg, tpe, pblob)
    elif args.placement == "placelambda_gpu":
        # PLACE-lambda batch-observed solve ON DEVICE (our arm; the epic
        # driver is only the harness). static ablation: this one untimed
        # setup solve is the resident placement (ideal-stale semantics),
        # measured as place_solver_ms. dynamic ablation re-solves TIMED
        # per iteration in perf_epic (place_ms) without moving weights.
        assert args.routing_file, (
            "--placement placelambda_gpu is defined for trace cells only")
        tk_dev = topk_all.long().cuda()
        torch.cuda.synchronize()
        t_ps = time.perf_counter()
        pl_solve = build_placement_gpu(
            tk_dev, DIST_ENV.LOCAL_WORLD_SIZE, cfg.nlp, args.G)
        torch.cuda.synchronize()
        place_solver_ms = (time.perf_counter() - t_ps) * 1e3
        hosts_pll = pl_solve["hosts"]
        pblob = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
                 "hosts": hosts_pll, "planner": pl_solve["stats"]}
        if args.placement_file:
            # pre-registration gate: the offline CPU solve (sidecar) must
            # equal the on-device solve bit-for-bit (cross-device oracle)
            with open(args.placement_file, "rb") as f:
                raw_p = f.read()
            sblob = json.loads(raw_p)
            placement_sha = hashlib.sha256(raw_p).hexdigest()[:16]
            assert sblob["hosts"] == hosts_pll, (
                "on-device PLACE-lambda solve != the sidecar's offline "
                "solve — cross-device determinism bug, never noise")
            pblob["predicted"] = sblob.get("predicted", [])
        plan = build_nodeaware_plan(cfg, tpe, pblob)
    elif args.placement == "placelambda_fast":
        # PLACE-lambda FAST (session 8.22.placefast): batched bounded-pass
        # zero-D2H solver (flux.testing.placelambda_fast). Cold solve at
        # setup = the resident placement; rank-level finalize (Stage C)
        # runs here, OFF the per-iteration path. New arm — never compare
        # its placements bitwise against placelambda_gpu cells.
        assert args.routing_file, (
            "--placement placelambda_fast is defined for trace cells only")
        tk_dev = topk_all.long().cuda()
        tk_solve = tk_dev
        if args.oracle_routing_file:
            # scenario-1 (canonical): the placement observes the SAMPLED
            # PREVIOUS batch (window-w earlier decode slots, same layer),
            # never the evaluated batch — no self-oracle.
            _oc = load_routing_file(args.oracle_routing_file, args.G,
                                    args.topk)
            assert _oc.shape[0] % W == 0, "oracle rows not divisible by W"
            tk_solve = _oc.view(W, -1, args.topk).long().cuda()
        pf_cfg = dict(
            passes_a=int(os.environ.get("FLUX_PLACE_FAST_PA", "4")),
            passes_b=int(os.environ.get("FLUX_PLACE_FAST_PB", "3")),
            repair_passes=int(os.environ.get("FLUX_PLACE_FAST_REPAIR",
                                             "2")),
            seed=os.environ.get("FLUX_PLACE_FAST_SEED", "affinity"),
        )
        torch.cuda.synchronize()
        t_ps = time.perf_counter()
        pf_solve = plfast.build_placement_fast(
            tk_solve, DIST_ENV.LOCAL_WORLD_SIZE, cfg.nlp, args.G, **pf_cfg)
        hosts_pll = plfast.finalize_hosts(
            pf_solve, W, DIST_ENV.LOCAL_WORLD_SIZE, cfg.nlp,
            method=os.environ.get("FLUX_PLACE_FAST_FINALIZE", "snake"))
        torch.cuda.synchronize()
        place_solver_ms = (time.perf_counter() - t_ps) * 1e3
        pblob = {"version": 2, "G": args.G, "W": W, "nlp": cfg.nlp,
                 "hosts": hosts_pll,
                 "planner": plfast.stats_host(pf_solve)}
        plan = build_nodeaware_plan(cfg, tpe, pblob)
        _h_eval = plfast.demand_hist(tk_dev, DIST_ENV.LOCAL_WORLD_SIZE,
                                     args.G)
        _h_sol = plfast.demand_hist(tk_solve, DIST_ENV.LOCAL_WORLD_SIZE,
                                    args.G)
        _drift = plfast.drift_ppm(_h_eval, _h_sol)
        RECORDER.emit_info(
            epic_pll_oracle_file=args.oracle_routing_file or "",
            epic_pll_oracle_basis=("prev_batch"
                                   if args.oracle_routing_file
                                   else "self"),
            epic_pll_oracle_drift_ppm=_drift,
        )
        if rank == 0:
            print(f"placement basis: "
                  f"{'ORACLE ' + args.oracle_routing_file if args.oracle_routing_file else 'self (batch)'}"
                  f"; oracle->batch drift {_drift} ppm", flush=True)
    else:
        plan = build_fixed_plan(cfg, tpe)
    plan_host_ms = (time.perf_counter() - t0) * 1e3
    if args.placement not in ("placelambda_gpu", "placelambda_fast"):
        place_solver_ms = 0.0
        hosts_pll = None
    if args.placement != "placelambda_fast":
        pf_solve = None

    # Replica selection. D6 (the rule-5 arms) is re-derived per iteration
    # on device by EpicIterPlanner; this setup pass builds the CPU
    # reference for sizing + the drift guard. The loccap/evensplit python
    # routers still run once per cell (legacy_untimed_plan accounting —
    # SCHEMA rule 5 makes that amortization illegal for quotable arms, so
    # loccap arms are not quotable until the router's GPU port lands; it
    # is being ported in a parallel campaign).
    assert args.router == "d6" or args.migration == "off", (
        "--router loccap is incompatible with --migration: K_g and the RS "
        "capacity caps are frozen from the loccap layout at ctor")
    pll_bounds = None
    t_r = time.perf_counter()
    if args.router == "loccap" and rank == 0:
        # heartbeat: the python router port can be silent for minutes at
        # b64 tight-eps (repair phase) — keep the runner's idle watchdog
        # from conflating "computing" with "hung"
        print(f"loccap: routing {W}x{S}x{args.topk} at eps {args.eps:g} "
              "(host port, legacy_untimed_plan accounting) ...", flush=True)
    if args.router == "loccap":
        phys_all_route = loccap_route(
            topk_all.long(), plan.p2l, plan.l2p, plan.lcnts, cfg.nlp,
            DIST_ENV.LOCAL_WORLD_SIZE, args.eps).cpu()
        plan.phys_override = phys_all_route
    elif args.router == "loccap_gpu":
        # setup CPU reference = one side of the cross-device oracle; the
        # rule-5 planner re-derives on GPU per iteration and the setup
        # check_against asserts bitwise equality against this routing
        phys_all_route, _pll_rstats = loccap_route_gpu(
            topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
            cfg.nlp, DIST_ENV.LOCAL_WORLD_SIZE, args.eps,
            remote_cap_only=bool(int(os.environ.get(
                "FLUX_LOCCAP_REMOTE_CAP_ONLY", "0"))))
        plan.phys_override = phys_all_route
    elif args.router == "loccap_sl":
        # KERNEL arm (relaxed): the setup reference is the deterministic
        # torch sender-local route — it sizes every frozen buffer and is
        # the final-iteration correctness routing; its tables give the
        # PROVABLE per-pair/recv bounds every relaxed kernel iteration
        # obeys by construction (f_cap = the one admission clamp).
        phys_all_route, pll_aux = loccap_route_sl(
            topk_all.long().cpu(), plan.p2l, plan.l2p, plan.lcnts,
            cfg.nlp, DIST_ENV.LOCAL_WORLD_SIZE, args.eps,
            return_tables=True,
            remote_cap_only=bool(int(os.environ.get(
                "FLUX_LOCCAP_REMOTE_CAP_ONLY", "0"))))
        plan.phys_override = phys_all_route
        pll_bounds = loccap_sl_bounds(pll_aux, W, args.pll_f_cap)
        args.pll_f_cap = pll_bounds["f_cap"]  # resolve auto for planner/facts
    elif args.router == "evensplit":
        phys_all_route = evensplit_route(topk_all.long(), plan.l2p,
                                         plan.lcnts).cpu()
        plan.phys_override = phys_all_route
    else:
        phys_all_route = d6_route(topk_all.long(), plan.l2p, plan.lcnts)
    loccap_plan_host_ms = (time.perf_counter() - t_r) * 1e3
    route_stats = incidence_stats(phys_all_route, cfg.nlp,
                                  DIST_ENV.LOCAL_WORLD_SIZE)
    r_hash = route_hash(phys_all_route)
    if (args.placement in ("nodeaware", "placelambda_gpu")
            and args.router != "d6"):
        want = ("evensplit" if args.router == "evensplit"
                else f"loccap_gpu_eps{args.eps:g}"
                if args.router == "loccap_gpu"
                else f"loccap_sl_eps{args.eps:g}"
                if args.router == "loccap_sl"
                else f"loccap_eps{args.eps:g}")
        pred = [p for p in pblob.get("predicted", [])
                if p.get("router") == want]
        if pred:
            # the pre-registration gate: the sidecar's offline simulation
            # must equal the realized routing bit-for-bit (same code, same
            # inputs) — a mismatch is a determinism bug, never noise
            assert pred[0]["route_hash"] == r_hash, (
                "realized loccap routing != the sidecar's pre-registered "
                "simulation (route_hash mismatch)")
            assert pred[0]["incidence_remote"] == \
                route_stats["incidence_remote"]
        elif rank == 0:
            print(f"loccap: eps {args.eps:g} not in the sidecar's predicted "
                  "ladder — no pre-registration cross-check for this cell")

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
        alloc_input_full=False,
        gating_args=gating_args, skip_reference=True,
    )

    # Dedicated communicator for the comm-stream NCCL a2av calls
    # (moonep_overlap precedent: never interleave one communicator across
    # two streams).
    comm_group = DIST_ENV.new_group(list(range(W)))

    if args.layers == "l01":
        assert args.weight_place == "fc1fc2", "--layers l01 needs fc1fc2"
    runner = EpicLayer0Runner(
        plan, rank, TP_GROUP, torch.cuda.current_device(), topk_all,
        m=args.groups, dtype=input_dtype,
        ffn_size_shard=moe_ctx.ffn_size_shard,
        place_fc2=(args.weight_place == "fc1fc2"),
        ranks_per_node=DIST_ENV.LOCAL_WORLD_SIZE,
        comm_group=comm_group,
        layers=args.layers,
    )
    # loccap_sl capacity mode: recv-side buffers + wire panels are sized to
    # the PROVABLE table bounds (never to the reference's realized rows),
    # so every relaxed kernel iteration fits by construction.
    if args.router == "loccap_sl":
        runner.reserve_recv_capacity(pll_bounds["recv_cap"])
    if args.transport in ("nvshmem", "hier_compress"):
        runner.enable_nvshmem(DIST_ENV.LOCAL_WORLD_SIZE, args.num_comm_sm,
                              split_headroom=args.a2a_split_headroom,
                              max_split_floor=(pll_bounds["pair_cap"]
                                               if pll_bounds else 0))
    if args.migration == "inkernel":
        assert args.transport == "hier_compress", (
            "--migration inkernel is the fused hc-dispatch phase; use the "
            "hier_compress transport")
    if args.transport == "hier_compress":
        tp_env = flux.DistEnvTPWithEP(
            tp_group=TP_GROUP, nnodes=num_nodes, ep_group=EP_GROUP)
        assert args.hc_wire == "relay_identity" or \
            args.migration != "inkernel", (
                "--hc_wire lb_union x --migration inkernel is untested")
        hc_kwargs = {}
        if args.router == "loccap_sl":
            L_ = DIST_ENV.LOCAL_WORLD_SIZE
            NN_ = W // L_
            pu = pll_bounds["pair_ub"]
            # node-aggregated pair bounds: unique (dedup) counts never
            # exceed raw pair rows, so cross-node sums of pair_ub dominate
            # the stage/relay demands (mirrors required_a2av_knobs shapes)
            node_pair = pu.view(NN_, L_, NN_, L_).sum(3).sum(1)  # [NN, NN]
            offdiag = node_pair - torch.diag(torch.diag(node_pair))
            stage_ub = int(offdiag.sum(0).max()) if NN_ > 1 else 0
            hc_kwargs = dict(
                cap_floors={
                    "FLUX_A2AV_MAX_RECV_NTOKENS": pll_bounds["recv_cap"],
                    "FLUX_A2AV_MAX_STAGE_NTOKENS": stage_ub,
                    "FLUX_A2AV_MAX_RELAY_NTOKENS": stage_ub,
                },
                # K_g == K analytically at m=1 (every router emits exactly
                # K entries per token); the pin makes it explicit
                fixed_kg=[args.topk] * args.groups,
            )
        runner.enable_hier_compress(
            tp_env, DIST_ENV.LOCAL_WORLD_SIZE,
            headroom=args.hc_headroom, relay=args.hc_relay,
            inkernel_swap=(args.migration == "inkernel"),
            wire=args.hc_wire, **hc_kwargs)
        if args.layers == "l01":
            # Mode-2 combine: per-group TopkReduceScatterOp (S3)
            runner.enable_hc_combine(n_split=args.l1_n_split)

    place_bytes, place_ms = runner.place_weights(TP_GROUP)
    # AFTER place_weights: the grouped backend snapshots compacted weight
    # copies for groups containing zero-row slots (see enable_grouped_gemm).
    runner.enable_grouped_gemm(args.gemm_backend)

    # tau: explicit tokens, else t_swap/t_token with probed t_swap fallback.
    swap_probe_ms = (
        probe_swap_ms(runner) if args.migration in ("on", "inkernel") else 0.0)
    if args.tau_tokens is not None:
        tau_tokens = args.tau_tokens
    elif args.t_token_us is not None:
        t_swap_ms = (args.t_swap_ms if args.t_swap_ms is not None
                     else swap_probe_ms)
        tau_tokens = (t_swap_ms * 1e3) / args.t_token_us
    else:
        tau_tokens = 0.0

    gen = torch.Generator().manual_seed(777)
    w_all = torch.rand(W, S, args.G, dtype=torch.float32, generator=gen)
    probs_shard = w_all[rank].cuda()

    loads_shard = tpe[rank].cuda().contiguous()
    loads_gather_buf = torch.zeros(W * args.G, dtype=torch.int32,
                                   device="cuda")

    # Per-iteration timed GPU planning (SCHEMA rule 5), scoped to the m=1 /
    # D6 arms — the sweep's quotable configurations. m>1 and the loccap
    # router stay on the legacy setup-time path and are marked
    # legacy_untimed_plan in cells.csv (visible, never silently mixed).
    per_iter = (args.groups == 1
                and args.router in ("d6", "loccap_gpu", "loccap_sl"))
    iter_planner = None
    if per_iter:
        iter_planner = EpicIterPlanner(
            plan, rank, torch.device(torch.cuda.current_device()), topk_all,
            DIST_ENV.LOCAL_WORLD_SIZE,
            l01=(args.layers == "l01"),
            hc=runner.hc_enabled,
            hcc=getattr(runner, "hcc_enabled", False),
            kg_frozen=(runner._hc_kg[0] if runner.hc_enabled else None),
            replica_select=args.replica_select,
            inwindow_meta=(args.hc_meta == "inwindow"
                           and runner.hc_enabled),
            router=args.router,
            eps=(args.eps if args.router in ("loccap_gpu", "loccap_sl")
                 else None),
            route_group=TP_GROUP,
            f_cap=args.pll_f_cap,
        )
        # Setup-time drift guard (untimed): one derive from the known
        # loads must reproduce the CPU reference state bitwise.
        assert torch.equal(iter_planner.local_loads().cpu(), tpe[rank])
        loads_gather_buf.copy_(tpe.reshape(-1).to(loads_gather_buf.device))
        if args.router == "loccap_sl":
            # relaxed contract: bitwise guard runs on the DETERMINISTIC
            # reference ip; the real kernel derive is audited by
            # invariants + provable bounds + incidence band instead.
            ip_ref = iter_planner.derive_reference()
            iter_planner.check_against(ip_ref, runner)
            iter_planner.relaxed_bounds = pll_bounds
            iter_planner.ref_incidence = route_stats["incidence_remote"]
            iter_planner._check_iters = bool(int(os.environ.get(
                "FLUX_PLL_CHECK_ITERS", "0")))
            ip0 = iter_planner.derive(loads_gather_buf)
            facts0 = iter_planner.check_relaxed(
                ip0, pll_bounds,
                ref_incidence=iter_planner.ref_incidence)
            if rank == 0:
                print(f"loccap_sl setup audit: {facts0} "
                      f"(bounds recv_cap {pll_bounds['recv_cap']} "
                      f"pair_cap {pll_bounds['pair_cap']})", flush=True)
        else:
            ip0 = iter_planner.derive(loads_gather_buf)
            iter_planner.check_against(ip0, runner)
        if iter_planner.inwindow_meta:
            # v2b guard: the op's in-window derivation must be bitwise-
            # equal to the python reference bundle (stable scatter index
            # determinism is the load-bearing property). For loccap_sl the
            # guard consumes the REFERENCE vce (the setup bundle was built
            # from the reference routing).
            vce_chk = (ip_ref.hc_vce if args.router == "loccap_sl"
                       else ip0.hc_vce)
            b0 = runner._hc_bundles[0]
            sd, scd, sps_c, uc_c = runner._hc_ops[0].derive_routed_meta(
                vce_chk)
            if (args.router in ("loccap_sl", "loccap_gpu")
                    and int(os.environ.get("FLUX_PLL_FAST_TAIL", "1"))):
                # fast tail: vce is topk-column-ordered, so the setup
                # bundle's canonical-order meta is not bitwise-comparable.
                # The guard's meaning is op == python ON THE SAME vce —
                # recompute the python reference from vce_chk directly.
                from flux.testing.epic_semantics import python_meta_from_vce
                r_sd, r_scd, r_sps, r_uc = python_meta_from_vce(
                    vce_chk, W, S, iter_planner.gpe, iter_planner.nn,
                    DIST_ENV.LOCAL_WORLD_SIZE)
                assert torch.equal(sd.long().cpu(), r_sd.long().cpu()), (
                    "iw splits drift (fast tail)")
                assert torch.equal(scd.long().cpu(), r_scd.long().cpu()), (
                    "in-window stable scatter index != python-on-same-vce")
                assert torch.equal(sps_c.long().cpu(), r_sps.long().cpu())
                assert torch.equal(uc_c.long().cpu(), r_uc.long().cpu())
            else:
                assert torch.equal(sd.cpu(), b0.meta.splits), (
                    "iw splits drift")
                assert torch.equal(scd.cpu(), b0.meta.scatter_index), (
                    "in-window stable scatter index != python reference")
                assert torch.equal(sps_c, b0.meta.splits_per_source)
                assert torch.equal(uc_c, b0.meta.a2av_unique_counts)
        runner.bind_iter_plan(ip0)
    elif rank == 0:
        print("timing_accounting=legacy_untimed_plan (m>1 or a python "
              "router: the per-iteration GPU planner is scoped to the "
              "m=1 d6/loccap_gpu arms)")
    timing_accounting = "per_iter_gpu" if per_iter else "legacy_untimed_plan"

    # Dynamic PLACE-lambda ablation: per-iteration TIMED solve + move diff
    # + trigger decision (place_ms in perf_epic). The resident placement is
    # NOT mutated — weight dispatch is the queued fusion-pass mechanism;
    # this arm times the full decision apparatus.
    place_fn = None
    place_dec_log = []
    pf_dec_ring = None
    if args.place_dynamic == "dynamic" and pf_solve is not None:
        # FAST dynamic lane (session 8.22.placefast): warm-seeded
        # bounded-pass solve + tensorized decision, ZERO D2H in-loop —
        # verdicts land in a device ring buffer read once at teardown.
        # FLUX_PLACE_FAST_GRAPH=1 (default) CUDA-graph-captures the whole
        # solve+decision at setup and replays it per iteration.
        assert per_iter, (
            "dynamic placement is a rule-5 arm (m=1, router d6/loccap_gpu)")
        _tk_place = topk_all.long().cuda()
        if int(os.environ.get("FLUX_PLACE_FAST_STALE_RESIDENT", "0")):
            # trigger probe: resident = the CONTIG (fixed-style) layout
            # instead of the fresh cold solve — the decision must fire
            # loudly here and stay ~0 on the normal arm (resident==fresh)
            _res_primary = plfast.seed_contig(
                args.G, W, DIST_ENV.LOCAL_WORLD_SIZE,
                pf_solve["primary"].device)
            _res_ion = torch.zeros_like(pf_solve["inst_nodes"])
            _res_ion[torch.arange(args.G, device=_res_ion.device),
                     _res_primary] = True
        else:
            _res_primary = pf_solve["primary"].clone()
            _res_ion = pf_solve["inst_nodes"].clone()
        warm_cfg = dict(
            seed="warm", seed_primary=_res_primary,
            seed_inst_nodes=_res_ion,
            keep_bonus=int(os.environ.get("FLUX_PLACE_FAST_KEEP_BONUS",
                                          "90090")),  # LCM16//8
            move_margin=int(os.environ.get("FLUX_PLACE_FAST_MOVE_MARGIN",
                                           "0")),
            passes_a=int(os.environ.get("FLUX_PLACE_FAST_WARM_PA", "2")),
            passes_b=int(os.environ.get("FLUX_PLACE_FAST_WARM_PB", "1")),
            repair_passes=int(os.environ.get(
                "FLUX_PLACE_FAST_WARM_REPAIR", "1")))
        _n_iters_tot = args.warmup_iters + args.iters
        pf_dec_ring = torch.zeros(_n_iters_tot, 5, dtype=torch.int64,
                                  device="cuda")

        _pf_trigger = os.environ.get("FLUX_PLACE_FAST_TRIGGER", "cover")

        def _pf_hot():
            fresh = plfast.build_placement_fast(
                _tk_place, DIST_ENV.LOCAL_WORLD_SIZE, cfg.nlp, args.G,
                **warm_cfg)
            return plfast.place_decision_fast(
                _tk_place, _res_ion, fresh, DIST_ENV.LOCAL_WORLD_SIZE,
                to_host=False, mode=_pf_trigger,
                primary_cur=_res_primary)

        _pf_graph = None
        if bool(int(os.environ.get("FLUX_PLACE_FAST_GRAPH", "1"))):
            try:
                for _ in range(2):
                    _pf_hot()
                torch.cuda.synchronize()
                _pf_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(_pf_graph):
                    _pf_packed = _pf_hot()
                _pf_graph.replay()
                torch.cuda.synchronize()
            except Exception as _e:  # noqa: BLE001 — fall back eager
                if rank == 0:
                    print(f"place-fast graph capture failed "
                          f"({type(_e).__name__}: {_e}); running eager",
                          flush=True)
                _pf_graph = None
        if _pf_graph is not None:
            def place_fn(_i):
                _pf_graph.replay()
                pf_dec_ring[_i].copy_(_pf_packed)   # D2D, async
        else:
            def place_fn(_i):
                pf_dec_ring[_i].copy_(_pf_hot())    # D2D, async
        if rank == 0:
            print(f"place-fast dynamic lane: warm_cfg="
                  f"{ {k: v for k, v in warm_cfg.items() if not torch.is_tensor(v)} } "
                  f"graph={'on' if _pf_graph is not None else 'off'}",
                  flush=True)
    elif args.place_dynamic == "dynamic":
        assert per_iter, (
            "dynamic placement is a rule-5 arm (m=1, router d6/loccap_gpu)")
        _tk_place = topk_all.long().cuda()
        # resident placement = whatever the arm actually runs on: the
        # stale/oracle layout under none/epic/nodeaware (the meaningful
        # trigger ablation) or the batch-solved layout itself under
        # placelambda_gpu (the zero-gain apparatus-timing arm)
        _resident_hosts = (
            hosts_pll if hosts_pll is not None else
            [sorted((plan.l2p[g, :int(plan.lcnts[g])].long()
                     // cfg.nlp).tolist())
             for g in range(args.G)])

        def place_fn(_i):
            fresh = build_placement_gpu(
                _tk_place, DIST_ENV.LOCAL_WORLD_SIZE, cfg.nlp, args.G)
            place_dec_log.append(place_decision(
                _tk_place, _resident_hosts, fresh["hosts"],
                DIST_ENV.LOCAL_WORLD_SIZE,
                gain_threshold_ppm=args.place_gain_threshold_ppm))

    comm_stream = torch.cuda.Stream(priority=-1)

    if rank == 0:
        loads_g = tpe.long().sum(0)
        before = loads_g.reshape(W, cfg.epn).sum(1)
        mean = float(before.double().mean())
        imb_before = float(before.max()) / mean if mean else 1.0
        gemm_rows = plan.physical_rows_per_rank()
        imb_after = max(gemm_rows) / mean if mean else 1.0
        rep = plan.replica_summary()
        dup = runner.dup_stats()
        inter_send, inter_recv = runner.internode_rows()
        print(f"ntokens: {ntokens} ({S} per rank), topk: {args.topk}, "
              f"G: {args.G}, m: {args.groups}, placement: {args.placement}, "
              f"migration: {args.migration} (tau={tau_tokens:.1f} tok), "
              f"redundant/rank: {args.redundant_per_rank}, "
              f"transport: {args.transport}, gemm: {args.gemm_backend}, "
              f"load source: {load_source}")
        print(f"epic gemm rows per rank: {gemm_rows}")
        print(f"imbalance max/mean: before {imb_before:.3f} -> after "
              f"{imb_after:.3f}")
        print(f"router: {args.router}"
              + (f" (eps {args.eps:g})"
                 if args.router in ("loccap", "loccap_gpu", "loccap_sl") else "")
              + f"; incidence_remote {route_stats['incidence_remote']} "
              f"(mean nodes/token {route_stats['mean_nodes_per_token']:.3f});"
              f" loccap_plan_host_ms {loccap_plan_host_ms:.1f} "
              "(setup reference build; rule-5 arms re-derive per "
              "iteration as plan_ms)")
        print(f"replicas: {rep}; dup counterfactual: {dup}")
        print(f"one-time weight placement: {place_ms:.1f} ms "
              f"(recv {place_bytes} B on rank 0); plan_host_ms: "
              f"{plan_host_ms:.1f}")
        RECORDER.emit_info(
            timing_accounting=timing_accounting,
            planner_impl=("fused_dispatch"
                          if (per_iter and args.hc_meta == "inwindow"
                              and args.transport == "hier_compress")
                          else "torch_gpu" if per_iter else "legacy"),
            replica_select=args.replica_select,
            ntokens=ntokens,
            tokens_per_rank=S,
            gemm_rows_per_rank=gemm_rows,
            epic_groups=args.groups,
            epic_group_bounds=[(g.slot_lo, g.slot_hi)
                               for g in runner.elay.groups],
            epic_placement=args.placement,
            epic_migration=args.migration,
            epic_tau_tokens=tau_tokens,
            epic_t_swap_ms_probe=swap_probe_ms,
            epic_redundant_per_rank=args.redundant_per_rank,
            epic_imbalance_before=imb_before,
            epic_imbalance_after=imb_after,
            epic_replicas_total=rep["total_replicas"],
            epic_replicas_max_per_expert=rep["max_replicas_per_expert"],
            epic_plan_host_ms=plan_host_ms,
            epic_plan_comm_bytes=W * args.G * 4,
            epic_load_source=load_source,
            epic_load_file=args.epic_load_file or "",
            epic_load_sha=load_sha,
            epic_transport=args.transport,
            epic_gemm_backend=args.gemm_backend,
            epic_router=args.router,
            epic_loccap_eps=(f"{args.eps:g}"
                             if args.router in ("loccap", "loccap_gpu",
                                                "loccap_sl")
                             else ""),
            epic_place_dynamic=args.place_dynamic,
            epic_place_solver_ms=place_solver_ms,
            epic_placement_hash=(str(placement_hash(hosts_pll))
                                 if hosts_pll is not None else ""),
            epic_loccap_plan_host_ms=loccap_plan_host_ms,
            epic_incidence_remote=route_stats["incidence_remote"],
            epic_mean_nodes_per_token=route_stats["mean_nodes_per_token"],
            epic_route_imbalance=route_stats["imbalance_max_over_mean"],
            epic_route_hash=str(r_hash),
            epic_placement_file=args.placement_file or "",
            epic_placement_sha=placement_sha,
            epic_weight_place_ms_oneshot=place_ms,
            epic_single_stream=bool(args.single_stream),
            epic_migration_collective="subsumed_by_plan_comm",
            epic_nic_estimate_ignores_dedup=1,
        )
    # Per-rank facts (every rank).
    inter_send, inter_recv = runner.internode_rows()
    est_send = plan.epic_est_internode_send[rank]
    est_recv = plan.epic_est_internode_recv[rank]
    RECORDER.emit_info(
        epic_weight_place_bytes=place_bytes,
        epic_dup_stats=runner.dup_stats(),
        epic_internode_send_rows=inter_send,
        epic_internode_recv_rows=inter_recv,
        epic_est_internode_send_pool=est_send,
        epic_est_internode_recv_pool=est_recv,
    )
    if args.transport == "hier_compress":
        RECORDER.emit_info(
            epic_hc_relay=args.hc_relay,
            epic_hc_wire=args.hc_wire,
            epic_hc_kg=[b.K_g for b in runner._hc_bundles],
            epic_hc_pad_rows_total=[
                int(b.pad_rows_per_rank.sum()) for b in runner._hc_bundles],
            epic_hc_m_per_rank=[
                b.meta.m_per_rank.tolist() for b in runner._hc_bundles],
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    with flux.group_profile(
        name="moe_epic_traffic_" + os.environ["TORCHELASTIC_RUN_ID"],
        do_prof=args.profile,
        group=TP_GROUP,
    ):
        iter_times, mig_facts = perf_epic(
            runner, iter_planner, moe_ctx, probs_shard, loads_shard,
            loads_gather_buf, comm_stream, args.warmup_iters, args.iters,
            args.sm_margin, args.migration, tau_tokens, args.single_stream,
            place_fn=place_fn,
        )

    if pf_dec_ring is not None:
        # ONE teardown sync converts the device ring into the reference
        # dict form (zero-D2H contract held during the loop)
        ring_h = pf_dec_ring.cpu().tolist()
        for row in ring_h:
            lb_c, lb_n, gp, n_add, n_rem = (int(x) for x in row)
            place_dec_log.append({
                "lb_cur": lb_c, "lb_new": lb_n, "gain_ppm": gp,
                "moves_add": n_add, "moves_remove": n_rem,
                "trigger": int(gp >= args.place_gain_threshold_ppm),
            })
    if place_dec_log:
        # static per-cell routing => every iteration's decision is
        # identical by construction; record the constant + a sanity check
        d0 = place_dec_log[0]
        assert all(d == d0 for d in place_dec_log), (
            "dynamic place decision drifted across iterations on static "
            "routing — determinism bug")
        RECORDER.emit_info(
            epic_place_lb_cur=d0["lb_cur"],
            epic_place_lb_new=d0["lb_new"],
            epic_place_gain_ppm=d0["gain_ppm"],
            epic_place_moves_add=d0["moves_add"],
            epic_place_moves_remove=d0["moves_remove"],
            epic_place_trigger=d0["trigger"],
        )
    if args.migration != "off":
        RECORDER.emit_info(
            epic_migration_swaps_total=mig_facts["swaps_total"],
            epic_migration_swaps_timed=mig_facts["swaps_timed"],
            epic_migration_rounds_to_converge=mig_facts["rounds_to_converge"],
            epic_migration_swap_bytes=runner.migration_swap_bytes,
            epic_relayout_ms_total=mig_facts["relayout_ms_total"],
            # inkernel = fused §4.3 phase 0 (host-baked post-swap indices
            # stand in for the paper's device-resident placement map);
            # nccl_host = the launch-granularity port
            epic_swap_fused_path=(
                "inkernel" if args.migration == "inkernel" else "nccl_host"),
        )

    def fmt(times):
        keys = ["plan_comm_ms", "place_ms", "plan_ms", "migration_ms",
                "pack_ms", "comm_ms", "e2e_ms", "total_ms"]
        if args.layers == "l01":
            keys = ["plan_comm_ms", "place_ms", "plan_ms", "migration_ms",
                    "l0_ms", "act_ms", "l1_ms", "e2e_ms", "total_ms"]
        return ", ".join(
            f"{k[:-3]} {sum(times[k]) / max(len(times[k]), 1):.3f} ms"
            for k in keys
        )

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"epic #{rank}: {fmt(iter_times)}")
    )
    RECORDER.emit_iters("epic", iter_times)

    if args.router == "loccap_sl" and iter_planner is not None:
        # FINAL DETERMINISTIC ITERATION (untimed; user decision
        # 2026-08-21): bind the setup-reference routing and run one
        # forward so output_sha and check_correctness validate the data
        # plane against plan.phys_override, while the timed iterations
        # above used the relaxed kernel routing.
        _dbg = int(os.environ.get("FLUX_PLL_DEBUG", "0"))
        if _dbg:
            print(f"[pll-hb] r{rank} final: derive_reference", flush=True)
        ip_ref = iter_planner.derive_reference()
        runner.bind_iter_plan(ip_ref)
        if _dbg:
            print(f"[pll-hb] r{rank} final: forward", flush=True)
        run_one_forward(runner, moe_ctx, probs_shard, args.sm_margin)
        if _dbg:
            print(f"[pll-hb] r{rank} final: forward done", flush=True)
        RECORDER.emit_info(
            epic_route_relaxed=1,
            epic_pll_f_cap=args.pll_f_cap,
            epic_pll_recv_cap=pll_bounds["recv_cap"],
            epic_pll_pair_cap=pll_bounds["pair_cap"],
            epic_pll_kernel_stats=(iter_planner.last_kernel_stats or []),
        )

    sha = output_sha(runner)
    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"epic #{rank}: out_sha {sha}")
    )
    if args.layers == "l01":
        RECORDER.emit_info(epic_out_sha_l01=sha, epic_layers="l01")
    else:
        RECORDER.emit_info(epic_out_sha=sha, epic_layers="l0")

    if input_dtype == torch.float16:
        atol, rtol = 1e-2, 1e-3
    else:
        atol, rtol = 1e-2, 1.5e-2

    if not args.skip_correctness:
        check_correctness(
            runner, moe_ctx, plan, topk_all, w_all, atol, rtol,
        )

    TP_GROUP.barrier()
    torch.cuda.synchronize()
    RECORDER.flush()
