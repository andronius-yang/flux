#!/usr/bin/env python3
"""Offline starvation predictor for the layer0 a2av lb_union (Tier B) dispatch.

Given a traffic matrix (a gen_matrix.py family or an on-disk matrix file), this
predicts, per destination rank, when each delivering window's signal fires vs
when the static tile-schedule wavefront demands it — i.e. where and how long
the grouped-GEMM starves — with ZERO GPU time. It exists to (a) select matrix
parameters before burning an allocation, (b) attribute windows at NN>2 by
their per-window row counts (matched against the tile-trace sidecar rows[]),
and (c) overlay predicted stalls on megadiagrams.

Everything geometric is an exact replica of the runtime, cited to
src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc @ ebb9d5c:
  - token->expert dealer: python/flux/testing/traffic_matrix.py
    traffic_matrix_to_choosed_experts (sorted column-major deal), replicated
    in pure stdlib here; --selftest-dealer proves bit-equality when torch is
    available.
  - canonical stream + balanced window cut: canon_start/chunk_bound (:1036-52)
  - recv layout + Tier B lane ends (gating geometry): region_rows/recv_off_u
    (:1203-1220), lane_end (:1257-1266); per-(expert,window) delivered rows
    reproduce the searchsorted gating cumsum (:1268-1274)
  - static schedule: shift_rank_to_order (sort_util.h:218-226), stage-major
    tile order with boundary tiles charged to the max stage
    (workspace_util.cu:36-103)
  - arrival model: relay wire puts round-serialized per NIC, targets
    tn=(my_node-dn)%NN descending (:2453-2455); gateway rounds ascending with
    head-of-line t_wait (:2519-2521); L serialized blocking NVLink puts per
    round, destination order dlg=(my_lr+1+dn+dl)%L, self at L-1-dn (:2526-39)

The timing constants (--bw-wire-gbps, --bw-nvlink-gbps, --gemm-rows-per-us,
--t0-wire-us, --t-local-us) are model parameters, NOT measurements; defaults
are rough A100/Slingshot figures. Calibrate them against an existing capture
(plot_a2av_trace.py --table / --gw-marks) before trusting absolute times; the
ORDER of events and the row/tile accounting are exact regardless.

Stdlib only. torch is needed only for --selftest-dealer.
"""

import argparse
import bisect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402


# ---------------------------------------------------------------------------
# dealer replica (traffic_matrix.py traffic_matrix_to_choosed_experts)
# ---------------------------------------------------------------------------
def deal_experts(chunks, tokens_per_rank, topk, G):
    """Return (cnt, tok_experts): cnt[s][e] = copies from source s to expert e;
    tok_experts[s][t] = sorted tuple of the topk experts of token t of rank s.

    Exact replica of the sorted column-major deal: per source, counts are
    spread round-robin over each destination rank's experts_per_rank experts
    (base + first-rem extra), the expert list is laid out ascending, and token
    t takes positions {j*T + t : j in [0, topk)}.
    """
    W = len(chunks)
    T = tokens_per_rank
    epr = G // W
    assert G % W == 0
    cnt = []
    tok_experts = []
    for s in range(W):
        counts = [0] * G
        for d in range(W):
            c = chunks[s][d]
            base, rem = divmod(c, epr)
            for j in range(epr):
                counts[d * epr + j] = base + (1 if j < rem else 0)
        assert sum(counts) == T * topk
        assert max(counts) <= T, "infeasible: an expert needs >T copies from one source"
        # cumsum over the sorted expert layout; expert_at(pos) via bisect
        cum = [0] * (G + 1)
        for e in range(G):
            cum[e + 1] = cum[e] + counts[e]
        experts_t = []
        for t in range(T):
            es = tuple(bisect.bisect_right(cum, j * T + t) - 1 for j in range(topk))
            assert len(set(es)) == topk, "dealer produced duplicate experts for a token"
            experts_t.append(es)
        cnt.append(counts)
        tok_experts.append(experts_t)
    return cnt, tok_experts


def selftest_dealer(chunks, tokens_per_rank, topk, G, chunk_bytes):
    """Bit-compare deal_experts against the real dealer (needs torch)."""
    import importlib.util

    import torch

    here = os.path.dirname(os.path.abspath(__file__))
    tm_path = os.path.join(here, "..", "python", "flux", "testing", "traffic_matrix.py")
    spec = importlib.util.spec_from_file_location("flux_traffic_matrix", tm_path)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    W = len(chunks)
    matrix = torch.tensor(chunks, dtype=torch.int64) * chunk_bytes
    ce = tm.traffic_matrix_to_choosed_experts(matrix, G, topk, chunk_bytes)
    _, mine = deal_experts(chunks, tokens_per_rank, topk, G)
    ref = ce.view(W, tokens_per_rank, topk).cpu()
    for s in range(W):
        for t in range(tokens_per_rank):
            got = tuple(sorted(ref[s][t].tolist()))
            want = tuple(sorted(mine[s][t]))
            assert got == want, f"dealer mismatch at (s={s}, t={t}): {got} != {want}"
    return True


# ---------------------------------------------------------------------------
# per-destination geometry: recv layout, windows, gating, tiles
# ---------------------------------------------------------------------------
class DestGeometry:
    """Exact recv/window/gating/tile geometry for one destination rank."""

    def __init__(self, chunks, cnt, tok_experts, W, L, T, topk, G, rank, tile_m):
        self.W, self.L, self.T, self.rank, self.tile_m = W, L, T, rank, tile_m
        NN = W // L
        self.NN = NN
        m, k = rank // L, rank % L
        self.m, self.k = m, k
        epr = G // W
        node_of_e = lambda e: (e // epr) // L  # noqa: E731
        rank_of_e = lambda e: e // epr  # noqa: E731

        # token -> destination structures
        # union token lists per (source rank s, dest node n), sorted ascending
        self.union_toks = [[None] * NN for _ in range(W)]
        # tokens needed per (source rank s, THIS dest rank)
        self.u_col = [0] * W
        for s in range(W):
            per_node = [set() for _ in range(NN)]
            u_here = 0
            for t, es in enumerate(tok_experts[s]):
                nodes = set()
                here = False
                for e in es:
                    nodes.add(node_of_e(e))
                    if rank_of_e(e) == rank:
                        here = True
                for n in nodes:
                    per_node[n].add(t)
                if here:
                    u_here += 1
            for n in range(NN):
                self.union_toks[s][n] = sorted(per_node[n])
            self.u_col[s] = u_here

        # canonical stream + balanced cut (canon_start/chunk_bound, .cc:1036-52)
        def canon_start(n, sl, mm):
            return sum(len(self.union_toks[n * L + s2][mm]) for s2 in range(sl))

        def chunk_bound(n, mm, kk):
            total = canon_start(n, L, mm)
            return (total // L) * kk + min(kk, total % L)

        self.canon_start = canon_start
        self.chunk_bound = chunk_bound

        # recv regions (region_rows, .cc:1203-1206) + slot row counts
        # slot s: local source -> u_col[s]; remote (ns,gl) -> window gl of ns
        self.slot_rows = [0] * W  # rows[] sidecar prediction (remote slots)
        for s in range(W):
            ns, gl = s // L, s % L
            if ns != m:
                self.slot_rows[s] = chunk_bound(ns, m, gl + 1) - chunk_bound(ns, m, gl)

        # per-expert rows keyed by (A-order position, delivering slot)
        # A-order within an expert = (source global rank, token) ascending;
        # the delivering slot is non-decreasing along it (windows cut the
        # canonical (sl, token)-ordered stream), so per-expert per-slot counts
        # fully determine tile spans.
        self.experts = list(range(rank * epr, (rank + 1) * epr))
        self.newly = {e: [0] * W for e in self.experts}  # rows per (expert, slot)
        for e in self.experts:
            for s in range(W):
                if cnt[s][e] == 0:
                    continue
                ns, sl = s // L, s % L
                if ns == m:
                    # local lane: whole region is slot s
                    self.newly[e][s] += cnt[s][e]
                else:
                    uni = self.union_toks[s][m]
                    base = canon_start(ns, sl, m)
                    bounds = [chunk_bound(ns, m, kk) for kk in range(L + 1)]
                    for t, es in enumerate(tok_experts[s]):
                        if e in es:
                            pos = base + bisect.bisect_left(uni, t)
                            gl = bisect.bisect_right(bounds, pos) - 1
                            self.newly[e][s - sl + gl] += 1

        # static schedule: stage of a slot (shift_rank_to_order)
        self.stage_of_slot = [
            ((s // L - m) % NN) * L + ((s % L - k) % L) for s in range(W)
        ]
        self.slot_of_stage = [None] * W
        for s in range(W):
            self.slot_of_stage[self.stage_of_slot[s]] = s

        # tiles: per expert, walk rows in slot-major A-order; a tile spanning
        # several slots is charged to the max stage (workspace_util.cu:46-68)
        self.tiles_by_stage = [0] * W
        self.rows_by_stage = [0] * W
        for e in self.experts:
            rows = self.newly[e]
            total = sum(rows)
            if total == 0:
                continue
            cum = [0] * (W + 1)
            for s in range(W):
                cum[s + 1] = cum[s] + rows[s]
                self.rows_by_stage[self.stage_of_slot[s]] += rows[s]
            for tm_i in range(math.ceil(total / tile_m)):
                r0, r1 = tm_i * tile_m, min((tm_i + 1) * tile_m, total) - 1
                stage_max = 0
                for s in range(W):
                    if cum[s + 1] > cum[s] and cum[s] <= r1 and cum[s + 1] > r0:
                        stage_max = max(stage_max, self.stage_of_slot[s])
                self.tiles_by_stage[stage_max] += 1


# ---------------------------------------------------------------------------
# arrival + wavefront model
# ---------------------------------------------------------------------------
def arrival_and_wavefront(geo, args):
    """Model signal times per slot and walk the static wavefront.

    Returns dict with per-stage table and summary. Times in us from GEMM t0.
    """
    W, L, NN, m, k = geo.W, geo.L, geo.NN, geo.m, geo.k
    row_bytes = args.chunk_bytes
    us_per_row_wire = row_bytes / (args.bw_wire_gbps * 1e3)  # GB/s -> B/us
    us_per_row_nvl = row_bytes / (args.bw_nvlink_gbps * 1e3)

    # wire arrival of window (ns, gl) at gateway (m, gl): the relay's rounds
    # descend the ring (tn = ns - dn), serialized on its NIC (.cc:2453-2455)
    arrive = {}
    for dn in range(1, NN):
        ns = (m + dn) % NN
        dnp = (ns - m) % NN  # this window is the relay's dnp-th put
        for gl in range(L):
            t = args.t0_wire_us
            for j in range(1, dnp + 1):
                tn = (ns - j) % NN
                rows_j = geo.chunk_bound(ns, tn, gl + 1) - geo.chunk_bound(ns, tn, gl)
                t += rows_j * us_per_row_wire
            arrive[(ns, gl)] = t

    # gateway (m, gl): rounds ascending, head-of-line wait, L serialized
    # blocking puts, dest order dlg=(gl+1+dn+dl)%L (.cc:2519-2539)
    signal = [0.0] * W  # per slot, us
    for s in range(W):
        if s // L == m:
            signal[s] = args.t_local_us
    for gl in range(L):
        t_free = 0.0
        for dn in range(1, NN):
            ns = (m + dn) % NN
            win_rows = geo.chunk_bound(ns, m, gl + 1) - geo.chunk_bound(ns, m, gl)
            put = win_rows * us_per_row_nvl
            start = max(t_free, arrive[(ns, gl)])
            pos = (k - gl - 1 - dn) % L  # my position in this gateway's rotation
            signal[ns * L + gl] = start + (pos + 1) * put if win_rows > 0 else start
            t_free = start + L * put

    # wavefront: stages in order; ready tiles are consumed at the aggregate
    # GEMM rate; a stage whose signal is late stalls the whole fleet (dense
    # static schedule, no skip-ahead)
    rate = args.gemm_rows_per_us
    t = 0.0
    stages = []
    total_stall = 0.0
    for st in range(W):
        s = geo.slot_of_stage[st]
        rows = geo.rows_by_stage[st]
        tiles = geo.tiles_by_stage[st]
        if rows == 0 and tiles == 0:
            continue
        ready = signal[s]
        stall = max(0.0, ready - t)
        t = max(t, ready) + rows / rate
        total_stall += stall
        stages.append(
            {
                "stage": st,
                "slot": s,
                "kind": "local" if s // L == m else f"win(n{s // L},l{s % L})",
                "rows": rows,
                "tiles": tiles,
                "signal_us": round(ready, 1),
                "wire_arrive_us": round(arrive.get((s // L, s % L), 0.0), 1)
                if s // L != m
                else None,
                "stall_us": round(stall, 1),
            }
        )
    span = t
    ideal = sum(geo.rows_by_stage) / rate
    return {
        "rank": geo.rank,
        "stages": stages,
        "span_us": round(span, 1),
        "ideal_us": round(ideal, 1),
        "stall_us": round(total_stall, 1),
        "deficit_frac_pred": round(total_stall / span, 3) if span else 0.0,
        "slot_rows": geo.slot_rows,
        "col_rows": sum(geo.rows_by_stage),
    }


# ---------------------------------------------------------------------------
# sidecar check
# ---------------------------------------------------------------------------
def check_sidecar(geo, path):
    """Compare predicted remote-slot rows[] against a tile-trace sidecar."""
    import plot_a2av_trace as pat

    iters = pat.read_sidecar(path)
    it = iters[-1]
    ok = True
    for s in range(geo.W):
        if s // geo.L == geo.m:
            continue
        got = it.rows[s] if it.rows else None
        want = geo.slot_rows[s]
        mark = "ok" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"  slot {s:2d} win(n{s // geo.L},l{s % geo.L}): sidecar {got} predicted {want} {mark}")
    return ok


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", help="gen_matrix family (alternative to --matrix)")
    ap.add_argument("--param", action="append", help="family param, e.g. nodefracs=0.9,0.1,0.9,0.1")
    ap.add_argument("--W", type=int)
    ap.add_argument("--ranks-per-node", type=int)
    ap.add_argument("--budget-mib", type=int)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--chunk-bytes", type=int, default=8192)
    ap.add_argument("--id", default="001")
    ap.add_argument("--matrix", help="path to <matrix_id>.txt (needs .meta.json for T)")
    ap.add_argument("--G", type=int, default=128)
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--ranks", default="all", help="'all' or comma list of dest ranks")
    # model constants — calibrate before trusting absolute numbers
    ap.add_argument("--bw-wire-gbps", type=float, default=22.0, help="per-relay-NIC wire GB/s")
    ap.add_argument("--bw-nvlink-gbps", type=float, default=80.0, help="gateway CE put GB/s")
    ap.add_argument("--gemm-rows-per-us", type=float, default=4.5)
    ap.add_argument("--t0-wire-us", type=float, default=500.0, help="pack+pull+launch offset")
    ap.add_argument("--t-local-us", type=float, default=300.0, help="local-lane signal time")
    ap.add_argument("--json", help="write full per-rank JSON here")
    ap.add_argument("--check-sidecar", help="tile-trace .bin to validate rows[] against")
    ap.add_argument("--selftest-dealer", action="store_true", help="bit-compare dealer (torch)")
    args = ap.parse_args()

    if args.matrix:
        with open(args.matrix) as f:
            lines = f.read().split()
        W = int(lines[0])
        vals = [int(x) for x in lines[1:]]
        chunks = [
            [vals[s * W + d] // args.chunk_bytes for d in range(W)] for s in range(W)
        ]
        meta_path = args.matrix.replace(".txt", ".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        T = meta["tokens_per_rank"]
        L = meta["ranks_per_node"]
        assert meta["topk"] == args.topk, "matrix topk != --topk"
    else:
        assert args.family and args.W and args.ranks_per_node and args.budget_mib
        params = dict(
            gen_matrix.FAMILY_DEFAULT_PARAMS[args.family],
            **gen_matrix.parse_params(args.param),
        )
        _, chunks, T = gen_matrix.generate(
            args.family, params, args.W, args.ranks_per_node,
            args.budget_mib, args.topk, args.chunk_bytes, args.id,
        )
        W, L = args.W, args.ranks_per_node

    if args.selftest_dealer:
        selftest_dealer(chunks, T, args.topk, args.G, args.chunk_bytes)
        print("dealer selftest: exact match vs traffic_matrix.py")

    cnt, tok_experts = deal_experts(chunks, T, args.topk, args.G)

    # closed-form cross-check of the dealt U against gen_matrix's law
    st = gen_matrix.dedup_round_stats(chunks, L, T)
    NN = W // L

    ranks = range(W) if args.ranks == "all" else [int(x) for x in args.ranks.split(",")]
    results = []
    for rank in ranks:
        geo = DestGeometry(chunks, cnt, tok_experts, W, L, T, args.topk, args.G, rank, args.tile_m)
        for n in range(NN):
            dealt = sum(len(geo.union_toks[n * L + sl][rank // L]) for sl in range(L))
            if n != rank // L:
                assert dealt == st["pair"][(n, rank // L)][1], "dealt U != closed form"
        res = arrival_and_wavefront(geo, args)
        results.append(res)
        print(
            f"rank {rank:2d} (node {rank // L}): span {res['span_us']:8.1f} us"
            f"  ideal {res['ideal_us']:8.1f}  stall {res['stall_us']:8.1f}"
            f"  deficit_pred {res['deficit_frac_pred']:.3f}  gemm_rows {res['col_rows']}"
        )
        for stg in res["stages"]:
            if stg["stall_us"] > 0 or stg["kind"] != "local":
                print(
                    f"    stage {stg['stage']:2d} {stg['kind']:12s} rows {stg['rows']:7d}"
                    f" tiles {stg['tiles']:5d} signal {stg['signal_us']:8.1f}"
                    f" stall {stg['stall_us']:8.1f}"
                )
        if args.check_sidecar:
            print(f"  sidecar rows[] check vs {args.check_sidecar}:")
            check_sidecar(geo, args.check_sidecar)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "model": {
                        k: getattr(args, k)
                        for k in (
                            "bw_wire_gbps", "bw_nvlink_gbps", "gemm_rows_per_us",
                            "t0_wire_us", "t_local_us", "tile_m", "G", "topk",
                        )
                    },
                    "headroom_closed_form": st["headroom"],
                    "ranks": results,
                },
                f, indent=2,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
