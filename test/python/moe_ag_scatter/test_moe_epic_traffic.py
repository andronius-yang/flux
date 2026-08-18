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

Migration (--migration on) runs between plan_comm and pack, per Figure 5:
replicated host decision (per-node heaviest/lightest pairing, tau-gated
swaps), intra-node weight exchange over batched P2P, plan mutation, layout
rebuild — then dispatch uses the updated mapping. The decision inputs are
derived from the SAME counts exchange the arm already pays every iteration
(plan_comm); adding a second collective would double-count it
(epic_migration_collective=subsumed_by_plan_comm). Under this harness's
static per-cell routing the swaps converge during warmup; timed iterations
measure the honest steady-state per-step cost (decision + zero swaps), and
the warmup swap costs are book-kept (epic_migration_* facts).

Phase metrics (long-format, impl=epic): plan_comm_ms, migration_ms, pack_ms,
disp{g}_ms (comm stream), stall{g}_ms (exposed wait before group g's
compute), scat{g}_ms, gemm{g}_ms, comm_ms (whole dispatch window), e2e_ms
(pack-start -> last GEMM: the comm-start->gemm-finish number, FAST
convention), total_ms (start -> last GEMM).
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
    EpicLayer0Runner,
    build_epic_plan,
    build_fixed_plan,
    plan_migration_swaps,
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
    ctx: MoeMlp1Ctx,
    probs_shard: torch.Tensor,
    loads_shard: torch.Tensor,
    loads_gather_buf: torch.Tensor,
    comm_stream,
    warmup_iters: int,
    iters: int,
    sm_margin: int,
    migration: bool,
    tau_tokens: float,
    single_stream: bool,
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
    names = ["start", "plan_comm", "mig", "pack"] + g_names
    ev = {
        name: [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for name in names
    }
    torch.distributed.barrier()
    torch.cuda.synchronize()

    isolated = bool(int(os.getenv("FLUX_SWEEP_ISOLATED_ITERS", "0")))
    iso_sync_times = []
    mig_host_times = []
    mig_swaps_per_iter = []
    relayout_ms_total = 0.0
    for i in range(total_iters):
        if isolated:
            t_iso = time.perf_counter()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            iso_sync_times.append((time.perf_counter() - t_iso) * 1e3)
        nvtx_tag = f"iter{i}_warmup" if i < warmup_iters else f"iter{i}"
        with torch.cuda.nvtx.range(nvtx_tag):
            ev["start"][i].record()
            # Recurring counts exchange (recv splits derive from this; also
            # the migration decision's load carrier).
            torch.distributed.all_gather_into_tensor(
                loads_gather_buf, loads_shard, group=TP_GROUP
            )
            ev["plan_comm"][i].record()
            if migration:
                t_mig = time.perf_counter()
                swaps = plan_migration_swaps(
                    runner.plan, tau_tokens, runner.ranks_per_node)
                if swaps:
                    # A swap stalls the pipeline (faithful: EPIC's in-kernel
                    # swap phase precedes dispatch). Sync so the exchange
                    # cannot race in-flight prior iterations in e2e mode.
                    torch.cuda.synchronize()
                    _, relayout_ms = runner.apply_migration(swaps)
                    relayout_ms_total += relayout_ms
                mig_swaps_per_iter.append(len(swaps))
                mig_host_times.append((time.perf_counter() - t_mig) * 1e3)
            ev["mig"][i].record()
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
        ["plan_comm_ms", "migration_ms", "pack_ms", "comm_ms", "e2e_ms",
         "total_ms"]
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
        times["migration_ms"].append(
            ev["plan_comm"][i].elapsed_time(ev["mig"][i]))
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
    if migration:
        times["migration_host_ms"] = mig_host_times[warmup_iters:]
    mig_facts = {
        "swaps_total": sum(mig_swaps_per_iter),
        "swaps_timed": sum(mig_swaps_per_iter[warmup_iters:]),
        "rounds_to_converge": next(
            (n for n, c in enumerate(mig_swaps_per_iter) if c == 0),
            len(mig_swaps_per_iter),
        ) if migration else 0,
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

    torch.distributed.all_gather_into_tensor(
        ctx.inputs, ctx.inputs_shard, group=TP_GROUP
    )

    expected_hidden = torch.zeros_like(runner.hidden_buf)
    expected_probs = torch.zeros_like(runner.weights_buf)
    seg_fill = list(runner.elay.seg_start)
    n_rows_check = 0
    per_instance_rows = [0] * cfg.nlp
    for src in range(R):
        tok, phys = reroute_expand(cfg, plan, src, topk_all[src])
        order = torch.argsort(phys * (S + 1) + tok, stable=True)
        tok, phys = tok[order], phys[order]
        msk = (phys // cfg.nlp) == rank
        tok, phys = tok[msk], phys[msk]
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

    # Wire-quota audit vs the D6 allocation.
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
                          f"D6 rank-quota total {expect}")

    if not torch.equal(runner.hidden_buf[:runner.n_recv],
                       expected_hidden[:runner.n_recv]):
        ok_bitwise = False
        print(f"❌ rank {rank}: dispatched rows differ from plan prediction")
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
        x = ctx.inputs_shard.float()
        topk = topk_all[rank].long()
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
        try:
            flux.torch_allclose(runner.final_out.float(), ref,
                                atol=atol, rtol=rtol)
        except Exception:
            ok_allclose = False
            diff = (runner.final_out.float() - ref).abs()
            print(f"❌ rank {rank}: l01 final output mismatch "
                  f"(max abs {float(diff.max()):.4f})")

    status = "✅" if (ok_bitwise and ok_allclose) else "❌"
    what = ("full journey" if runner.layers == "l01" else "gemm")
    print(f"{status} rank {rank}: epic dispatch content "
          f"{'bitwise-exact' if ok_bitwise else 'MISMATCH'}, "
          f"{what} {'allclose' if ok_allclose else 'MISMATCH'}")
    RECORDER.emit_correctness(bitwise=ok_bitwise, allclose=ok_allclose)
    assert ok_bitwise and ok_allclose
    return ok_bitwise and ok_allclose


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)  # exact bytes; numpy has no bf16
    return t.numpy().tobytes()


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
                        choices=["none", "epic"],
                        help="epic = §4.2 (redundancy + GPU greedy + NIC "
                        "stage) from the pool load; none = fixed contiguous "
                        "homing, empty redundant slots")
    parser.add_argument("--migration", default="off", choices=["off", "on"],
                        help="§4.3 per-step intra-host expert migration")
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

    t0 = time.perf_counter()
    if args.placement == "epic":
        plan = build_epic_plan(cfg, tpe, pool_load, num_nodes)
    else:
        plan = build_fixed_plan(cfg, tpe)
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
    if args.transport in ("nvshmem", "hier_compress"):
        runner.enable_nvshmem(DIST_ENV.LOCAL_WORLD_SIZE, args.num_comm_sm,
                              split_headroom=args.a2a_split_headroom)
    if args.transport == "hier_compress":
        tp_env = flux.DistEnvTPWithEP(
            tp_group=TP_GROUP, nnodes=num_nodes, ep_group=EP_GROUP)
        runner.enable_hier_compress(
            tp_env, DIST_ENV.LOCAL_WORLD_SIZE,
            headroom=args.hc_headroom, relay=args.hc_relay)
        if args.layers == "l01":
            # Mode-2 combine: per-group TopkReduceScatterOp (S3)
            runner.enable_hc_combine()

    place_bytes, place_ms = runner.place_weights(TP_GROUP)
    # AFTER place_weights: the grouped backend snapshots compacted weight
    # copies for groups containing zero-row slots (see enable_grouped_gemm).
    runner.enable_grouped_gemm(args.gemm_backend)

    # tau: explicit tokens, else t_swap/t_token with probed t_swap fallback.
    swap_probe_ms = probe_swap_ms(runner) if args.migration == "on" else 0.0
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
        print(f"replicas: {rep}; dup counterfactual: {dup}")
        print(f"one-time weight placement: {place_ms:.1f} ms "
              f"(recv {place_bytes} B on rank 0); plan_host_ms: "
              f"{plan_host_ms:.1f}")
        RECORDER.emit_info(
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
            runner, moe_ctx, probs_shard, loads_shard, loads_gather_buf,
            comm_stream, args.warmup_iters, args.iters, args.sm_margin,
            args.migration == "on", tau_tokens, args.single_stream,
        )

    if args.migration == "on":
        RECORDER.emit_info(
            epic_migration_swaps_total=mig_facts["swaps_total"],
            epic_migration_swaps_timed=mig_facts["swaps_timed"],
            epic_migration_rounds_to_converge=mig_facts["rounds_to_converge"],
            epic_migration_swap_bytes=runner.migration_swap_bytes,
            epic_relayout_ms_total=mig_facts["relayout_ms_total"],
        )

    def fmt(times):
        keys = ["plan_comm_ms", "migration_ms", "pack_ms", "comm_ms",
                "e2e_ms", "total_ms"]
        if args.layers == "l01":
            keys = ["plan_comm_ms", "migration_ms", "l0_ms", "act_ms",
                    "l1_ms", "e2e_ms", "total_ms"]
        return ", ".join(
            f"{k[:-3]} {sum(times[k]) / max(len(times[k]), 1):.3f} ms"
            for k in keys
        )

    flux.exec_in_rank_order(
        TP_GROUP, lambda: print(f"epic #{rank}: {fmt(iter_times)}")
    )
    RECORDER.emit_iters("epic", iter_times)

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
