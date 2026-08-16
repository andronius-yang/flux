#!/usr/bin/env python3
"""Deterministic traffic-matrix generator for the sweep system (sweeps/SCHEMA.md).

Budget semantics: --budget-mib is STRICTLY the pre-topk send budget per source
rank — the bytes of unique tokens homed on the rank. The generated matrix's
row sums are budget_mib * 2^20 * topk (post-fanout wire rows), and
tokens_per_rank = budget_mib * 2^20 / chunk_bytes (independent of topk).

Identity: matrix_id is a pure function of (family, params, W, L, budget, topk,
chunk_bytes, id). The RNG seed is FNV-1a of the same canonical string, so the
same arguments always regenerate a byte-identical file on any platform. A
sidecar <matrix_id>.meta.json records everything including the .txt sha256.

Stdlib only — runs on login/head nodes without torch.

Families (all: zero diagonal, entries multiples of chunk_bytes, exact equal
row sums, per-(row,dst) routing feasibility for distinct-topk experts):
  uniform    — budget split equally over the W-1 off-diagonal destinations
  hotcol     — fraction `frac` (default 0.5) of every row to one seeded hot
               destination column, remainder uniform over the rest
  nodeskew   — intra-node share proportional to peer count; of the remote
               share, fraction `frac` (default 0.75) to one seeded hot remote
               node per source node, remainder uniform over other remote ranks
  remotefrac — per-rank inter-node skew: each rank sends fraction p of its
               budget uniformly to all remote ranks and 1-p uniformly to its
               L-1 intra-node peers, where p comes from a seeded per-node
               permutation of `fracs` (default: the measured set of the
               original 2n_16r_skew matrices)
  fanoutskew — per-NODE exporter skew: every rank of node i sends fraction
               nodefracs[i] uniformly to all remote ranks and 1-nodefracs[i]
               uniformly to its L-1 intra-node peers. No per-rank shuffle and
               no seeded state (--id still enters the identity/seed, but this
               family draws nothing from the RNG). len(nodefracs) must equal
               the node count; needs >= 3 nodes (below 3 the per-node wire
               stagger this family exists for cannot be expressed — see the
               dealer saturation law U = T*min(1, f*topk)).
  trace      — batches sampled from REAL MoE routing traces (Patterns behind
               Chaos dataset); implemented in gen_trace_routing.py. The one
               family with a NONZERO diagonal (real self-routed tokens) and a
               second artifact, <matrix_id>.routing.txt, holding the per-token
               expert ids — the bench must consume it via --routing_file or
               the dealer's max-dedup assignment misrepresents the trace.
"""

import argparse
import hashlib
import json
import math
import os
import time

GENERATOR_VERSION = 1

# measured per-rank remote fractions of the original (lost-generator)
# 2n_16r_skew dist_001 matrices — pinned as the family default
REMOTEFRAC_DEFAULT = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90)

FAMILY_DEFAULT_PARAMS = {
    "uniform": {},
    "hotcol": {"frac": 0.5},
    "nodeskew": {"frac": 0.75},
    "remotefrac": {"fracs": REMOTEFRAC_DEFAULT},
    # two hot exporter nodes, two thin — the NN=4 starvation-campaign arm 1
    "fanoutskew": {"nodefracs": (0.9, 0.1, 0.9, 0.1)},
    # real-trace routing (gen_trace_routing.py); `pools` and `layer` are
    # required, `poolsha` is injected from the trace content fingerprint
    "trace": {"model": "Qwen3-235B", "pool": "decode", "sem": "homog"},
}


def fnv1a(s: str) -> int:
    h = 2166136261
    for c in s.encode():
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def canonical_string(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance):
    param_str = ",".join(
        f"{k}={','.join(str(x) for x in v) if isinstance(v, (tuple, list)) else v}"
        for k, v in sorted(params.items())
    )
    return (
        f"{family}|{param_str}|w{W}x{L}|b{budget_mib}|k{topk}|c{chunk_bytes}"
        f"|id{matrix_instance}"
    )


def matrix_id_of(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance):
    defaults = FAMILY_DEFAULT_PARAMS.get(family, {})
    if params == defaults:
        param_slug = ""
    else:
        canon = canonical_string(family, params, W, L, budget_mib, topk, chunk_bytes, "")
        param_slug = f"-{fnv1a(canon) & 0xFFFFFF:06x}"
    return f"w{W}x{L}_{family}{param_slug}_b{budget_mib}_k{topk}_id{matrix_instance}"


def _apportion(weights, total):
    """Largest-remainder apportionment: integer counts >= 0 summing to total,
    proportional to weights, deterministic (ties by index)."""
    s = sum(weights)
    assert s > 0, "no positive destination weights"
    quotas = [w * total / s for w in weights]
    base = [math.floor(q) for q in quotas]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return base


def _row_weights(family, params, s, W, L, rng_derived):
    """Per-destination weights for source rank s (self weight must be 0)."""
    node = s // L
    nn = W // L
    w = [0.0] * W
    if family == "uniform":
        for d in range(W):
            if d != s:
                w[d] = 1.0
    elif family == "hotcol":
        frac = float(params["frac"])
        hot = rng_derived["hot_col"]
        if hot == s:  # the hot column's own row: uniform over the others
            for d in range(W):
                if d != s:
                    w[d] = 1.0
        else:
            others = W - 2  # excluding self and the hot column
            for d in range(W):
                if d == s:
                    continue
                w[d] = frac if d == hot else (1.0 - frac) / others
    elif family == "nodeskew":
        frac = float(params["frac"])
        intra_share = (L - 1) / (W - 1)
        remote_share = 1.0 - intra_share
        hot_node = rng_derived["hot_node"][node]
        other_remote = W - 2 * L  # remote ranks outside the hot node
        if other_remote == 0:
            frac = 1.0  # 2 nodes: the hot node is the only remote node
        for d in range(W):
            if d == s:
                continue
            dn = d // L
            if dn == node:
                w[d] = intra_share / (L - 1)
            elif dn == hot_node:
                w[d] = remote_share * frac / L
            else:
                w[d] = remote_share * (1.0 - frac) / other_remote
    elif family == "remotefrac":
        p = rng_derived["remote_frac"][s]
        for d in range(W):
            if d == s:
                continue
            if d // L == node:
                w[d] = (1.0 - p) / (L - 1)
            else:
                w[d] = p / (W - L)
    elif family == "fanoutskew":
        p = float(params["nodefracs"][node])
        for d in range(W):
            if d == s:
                continue
            if d // L == node:
                w[d] = (1.0 - p) / (L - 1)
            else:
                w[d] = p / (W - L)
    else:
        raise ValueError(f"unknown family: {family}")
    return w


def generate(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance):
    """Return (matrix_id, chunks) where chunks is a [W][W] list of chunk counts."""
    import random

    if family == "trace":
        raise ValueError("the trace family is generated via gen_trace_routing.ensure_trace_matrix")
    assert W % L == 0, f"W ({W}) must be a multiple of ranks_per_node ({L})"
    nn = W // L
    if family in ("nodeskew", "remotefrac"):
        assert nn >= 2, f"family {family} needs >= 2 nodes (W={W}, L={L})"
        # note: nodeskew with exactly 2 nodes degrades to frac=1.0 (the hot
        # node is the only remote node) — see _row_weights
    if family == "fanoutskew":
        assert nn >= 3, f"fanoutskew needs >= 3 nodes (W={W}, L={L})"
        assert len(params["nodefracs"]) == nn, (
            f"fanoutskew needs exactly one nodefrac per node:"
            f" got {len(params['nodefracs'])} for {nn} nodes"
        )

    budget_bytes = budget_mib * (1 << 20)
    assert (
        budget_bytes % chunk_bytes == 0
    ), f"budget ({budget_bytes} B) must be a multiple of chunk_bytes ({chunk_bytes})"
    tokens_per_rank = budget_bytes // chunk_bytes  # pre-topk tokens
    row_chunks = tokens_per_rank * topk  # post-fanout wire rows per source

    canon = canonical_string(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance)
    rng = random.Random(fnv1a(canon))
    rng_derived = {}
    if family == "hotcol":
        rng_derived["hot_col"] = rng.randrange(W)
    elif family == "nodeskew":
        # each source node picks a distinct-from-self hot remote node
        rng_derived["hot_node"] = [rng.choice([m for m in range(nn) if m != n]) for n in range(nn)]
    elif family == "remotefrac":
        fracs = [float(x) for x in params["fracs"]]
        assert (
            len(fracs) >= L
        ), f"remotefrac needs >= ranks_per_node ({L}) fractions, got {len(fracs)}"
        remote_frac = []
        for n in range(nn):
            perm = fracs[:L][:]
            rng.shuffle(perm)
            remote_frac.extend(perm)
        rng_derived["remote_frac"] = remote_frac

    chunks = []
    experts_hint = None
    for s in range(W):
        weights = _row_weights(family, params, s, W, L, rng_derived)
        row = _apportion(weights, row_chunks)
        assert row[s] == 0, f"internal: nonzero diagonal at row {s}"
        assert sum(row) == row_chunks
        chunks.append(row)

    mid = matrix_id_of(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance)
    return mid, chunks, tokens_per_rank


def dedup_round_stats(chunks, L, tokens_per_rank):
    """Closed-form dedup/wire statistics under the sorted column-major dealer
    (traffic_matrix_to_choosed_experts): a contiguous run of C copies to one
    destination node covers exactly min(C, T) distinct tokens, so
    U[s][n] = min(sum of chunks[s][d] over d in node n, tokens_per_rank).

    Returns a dict with:
      U            — [W][nn] unique-token counts (the runtime's U_mat)
      pair         — {(sn, dn): (copies, unique, dedup_ratio)} over ordered
                     node pairs, sn != dn
      round_profile— {dest_node: [(dn, src_node, U_rows), ...]} for dn
                     ascending (the gateway's round processing order)
      col_rows     — [W] per-destination-rank GEMM rows (column sums)
      headroom     — closed-form relay_balanced_bytes / relay_ident_bytes
                     (the test harness's metric, test_moe_ag_traffic.py)
    """
    W = len(chunks)
    nn = W // L
    T = tokens_per_rank
    U = [[min(sum(chunks[s][m * L + j] for j in range(L)), T) for m in range(nn)] for s in range(W)]
    return dedup_stats_from_U(chunks, U, L, tokens_per_rank)


def dedup_stats_from_U(chunks, U, L, tokens_per_rank):
    """Round/pair/headroom statistics from an explicit U ([W][nn] unique-token
    counts). dedup_round_stats feeds it the dealer closed form; the trace
    family (gen_trace_routing.py) feeds it the U measured from real routing —
    the closed form does NOT hold there."""
    W = len(chunks)
    nn = W // L
    pair = {}
    for sn in range(nn):
        for dn_ in range(nn):
            if dn_ == sn:
                continue
            copies = sum(chunks[sn * L + sl][dn_ * L + j] for sl in range(L) for j in range(L))
            uniq = sum(U[sn * L + sl][dn_] for sl in range(L))
            pair[(sn, dn_)] = (copies, uniq, (copies / uniq) if uniq else float("nan"))
    round_profile = {}
    for m in range(nn):
        round_profile[m] = [
            (dn, (m + dn) % nn, pair[((m + dn) % nn, m)][1]) for dn in range(1, nn)
        ]
    col_rows = [sum(chunks[s][d] for s in range(W)) for d in range(W)]
    ident = balanced = 0
    for n in range(nn):
        for dn in range(1, nn):
            tn = (n - dn + nn) % nn
            seg = [U[n * L + sl][tn] for sl in range(L)]
            ident += max(seg)
            balanced += (sum(seg) + L - 1) // L
    headroom = (balanced / ident) if ident else float("nan")
    return {
        "U": U,
        "pair": pair,
        "round_profile": round_profile,
        "col_rows": col_rows,
        "headroom": headroom,
    }


def dealer_dedup_u(chunks, tokens_per_rank):
    """Per-RANK dedup counts under the sorted column-major dealer: the same
    contiguous-run argument as dedup_round_stats's U — a run of C copies to
    one destination rank covers exactly min(C, T) distinct tokens (a rank's
    experts are contiguous in the dealt stream). Does NOT hold for real
    routing; use gen_trace_routing.real_dedup_stats there."""
    T = tokens_per_rank
    return [[min(c, T) for c in row] for row in chunks]


def a2av_knob_demands(chunks, u, U, L):
    """Exact a2av capacity demands in ROWS, replicating every runtime
    capacity FLUX_CHECK of gemm_grouped_v2_ag_scatter.cc for the flux-driver
    arms: recv copies-column max (the recv gate on non-compress paths),
    union-bcast dedup recv max (compress lb_union), identity/balanced-relay
    staging and non-compress hier staging. Mirrors the torch implementation
    in python/flux/testing/moonep_fused_map.py:required_a2av_knobs (parity
    unit test in sweeps/test_knob_demands.py); torch-free so the login-node
    runner can call it. chunks/u are [W][W] rows, U is [W][nn] rows."""
    W = len(chunks)
    nn = W // L

    recv_copies = max(sum(chunks[s][d] for s in range(W)) for d in range(W))

    def region_rows(s, d):
        if nn > 1 and s // L != d // L:
            return U[s][d // L]
        return u[s][d]

    recv_union = max(sum(region_rows(s, d) for s in range(W)) for d in range(W))

    def gr(l, n):  # global rank of local rank l on node n
        return n * L + l

    def node_chunk(s, n):
        return sum(chunks[s][n * L + j] for j in range(L))

    stage_hier = stage_ident = stage_lb = relay_lb = 0
    if nn > 1:
        for gn in range(nn):
            for gl in range(L):
                stage_hier = max(
                    stage_hier,
                    sum(node_chunk(gr(gl, ns), gn) for ns in range(nn) if ns != gn),
                )
                stage_ident = max(
                    stage_ident,
                    sum(U[gr(gl, ns)][gn] for ns in range(nn) if ns != gn),
                )

        def chunk_bound(n, m, k):
            # canonical stream of source node n's L union segments toward node
            # m, cut into L near-equal chunks; relay rank k carries chunk k
            total = sum(U[gr(j, n)][m] for j in range(L))
            return (total // L) * k + min(k, total % L)

        for n in range(nn):
            for k in range(L):
                stage_lb = max(
                    stage_lb,
                    sum(
                        chunk_bound(ns, n, k + 1) - chunk_bound(ns, n, k)
                        for ns in range(nn)
                        if ns != n
                    ),
                )
                relay_lb = max(
                    relay_lb,
                    sum(
                        chunk_bound(n, (n - dn + nn) % nn, k + 1)
                        - chunk_bound(n, (n - dn + nn) % nn, k)
                        for dn in range(1, nn)
                    ),
                )

    return {
        "recv_copies": recv_copies,
        "recv_union": recv_union,
        "stage_hier": stage_hier,
        "stage_ident": stage_ident,
        "stage_lb": stage_lb,
        "relay_lb": relay_lb,
    }


def a2av_rs_knob_demands(chunks, U, L):
    """Exact LAYER1 (gather-rs combine) capacity demands in ROWS, replicating
    the runtime FLUX_CHECKs of gemm_grouped_v2_gather_rs.cc (all collective:
    identical expressions on every rank, so an undersized knob aborts cleanly
    everywhere, never hangs — unlike the layer0 per-rank recv gate).

    Inputs stay in DISPATCH orientation (`chunks[h][o]` = rows homed at rank h
    whose expert copy is owned by rank o; `U[h][m]` = distinct tokens of home
    rank h with >= 1 copy on owner node m — the same [W][nn] dedup matrix the
    layer0 sizing uses). The layer1 wire runs owner->home, i.e. the transpose:
    the C++ `chunk_at(s, d)` equals `chunks[d][s]` here (its row-sum check
    :546-550 pins that orientation to the gemm rows).

    Demands (cc line anchors, 2026-08-16 tree):
      rs_send  (:562-571)  max over owner o of sum_h chunks[h][o] — the max
                           dispatch COLUMN sum; numerically identical to the
                           layer0 recv_copies bound.
      rs_stage (:600-615)  non-compress gateway staging at (gnode gn, lane gl):
                           rows homed on node gn owned by the lane-gl rank of
                           every other node.
      rs_conv  (:687-708)  compress convergence at (owner node n2, dest lane
                           dl): rows owned anywhere on n2, homed at lane-dl
                           ranks of remote nodes.
      rs_wire  (:687-710)  compress wire: one pre-reduced partial per distinct
                           (token, owner node), U-summed over remote home
                           ranks of lane dl.
    The recv panel is knob-free (max_m / W exact, :233). No legacy floor is
    applied here — the layer1 axis is new, there are no historical capsules
    whose env_json must stay byte-identical (contrast a2av_knob_demands).
    Torch-free so the login-node runner can call it; parity unit test in
    sweeps/test_knob_demands.py."""
    W = len(chunks)
    nn = W // L

    rs_send = max(sum(chunks[h][o] for h in range(W)) for o in range(W))

    def gr(l, n):  # global rank of local rank l on node n
        return n * L + l

    rs_stage = rs_conv = rs_wire = 0
    if nn > 1:
        for gn in range(nn):
            for gl in range(L):
                rs_stage = max(
                    rs_stage,
                    sum(
                        chunks[h][gr(gl, ns)]
                        for ns in range(nn)
                        if ns != gn
                        for h in range(gn * L, (gn + 1) * L)
                    ),
                )
        for n2 in range(nn):
            for dl in range(L):
                rs_conv = max(
                    rs_conv,
                    sum(
                        chunks[gr(dl, tn)][n2 * L + ls]
                        for tn in range(nn)
                        if tn != n2
                        for ls in range(L)
                    ),
                )
                rs_wire = max(
                    rs_wire,
                    sum(U[gr(dl, tn)][n2] for tn in range(nn) if tn != n2),
                )

    return {
        "rs_send": rs_send,
        "rs_stage": rs_stage,
        "rs_conv": rs_conv,
        "rs_wire": rs_wire,
    }


def check_feasible(chunks, W, topk, tokens_per_rank, nexperts=None):
    """Mirror of traffic_matrix_to_choosed_experts's routing constraint: with
    G experts (G % W == 0), each (source, expert) pair may receive at most
    tokens_per_rank copies. Returns the minimum feasible experts-per-rank."""
    max_entry = max(max(row) for row in chunks)
    # need ceil(max_entry / experts_per_rank) <= tokens_per_rank
    min_epr = math.ceil(max_entry / tokens_per_rank)
    if nexperts is not None:
        epr = nexperts // W
        if math.ceil(max_entry / epr) > tokens_per_rank:
            raise ValueError(
                f"infeasible: hottest (src,dst) entry has {max_entry} chunks; with"
                f" G={nexperts} ({epr}/rank) an expert would need"
                f" {math.ceil(max_entry / epr)} > tokens_per_rank ({tokens_per_rank})"
                f" copies from one source. Need G >= {min_epr * W} or a lower topk/frac."
            )
    return min_epr * W


def write_matrix(out_root, matrix_id, chunks, chunk_bytes, meta):
    os.makedirs(out_root, exist_ok=True)
    W = len(chunks)
    lines = [str(W)]
    for row in chunks:
        lines.append(" ".join(str(c * chunk_bytes) for c in row))
    body = "\n".join(lines) + "\n"
    path = os.path.join(out_root, f"{matrix_id}.txt")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
    os.rename(tmp, path)
    sha = hashlib.sha256(body.encode()).hexdigest()
    meta = dict(meta, sha256=sha, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta_path = os.path.join(out_root, f"{matrix_id}.meta.json")
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    os.rename(meta_path + ".tmp", meta_path)
    return path, sha


def ensure_matrix(
    family,
    params,
    W,
    L,
    budget_mib,
    topk,
    chunk_bytes,
    matrix_instance,
    out_root,
    nexperts=None,
    traces_root=None,
):
    """Generate the matrix if missing; verify sha if present. Returns
    (matrix_id, path, sha256)."""
    params = dict(FAMILY_DEFAULT_PARAMS[family], **(params or {}))
    if family == "trace":
        import gen_trace_routing

        return gen_trace_routing.ensure_trace_matrix(
            params,
            W,
            L,
            budget_mib,
            topk,
            chunk_bytes,
            matrix_instance,
            out_root,
            traces_root=traces_root,
            nexperts=nexperts,
        )[:3]
    mid = matrix_id_of(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance)
    path = os.path.join(out_root, f"{mid}.txt")
    meta_path = os.path.join(out_root, f"{mid}.meta.json")
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if sha != meta["sha256"]:
            raise RuntimeError(
                f"matrix {path} does not match its sidecar sha256 — refusing to"
                " overwrite; delete both files to regenerate"
            )
        return mid, path, sha
    mid2, chunks, tokens_per_rank = generate(
        family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance
    )
    assert mid2 == mid
    check_feasible(chunks, W, topk, tokens_per_rank, nexperts=nexperts)
    meta = {
        "matrix_id": mid,
        "family": family,
        "params": {k: list(v) if isinstance(v, (tuple, list)) else v for k, v in params.items()},
        "W": W,
        "ranks_per_node": L,
        "budget_mib": budget_mib,
        "budget_semantics": "pre-topk send budget; row_sum_bytes = budget_mib*2^20*topk",
        "topk": topk,
        "chunk_bytes": chunk_bytes,
        "tokens_per_rank": tokens_per_rank,
        "row_sum_bytes": budget_mib * (1 << 20) * topk,
        "seed": fnv1a(
            canonical_string(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance)
        ),
        "matrix_instance": matrix_instance,
        "generator_version": GENERATOR_VERSION,
    }
    path, sha = write_matrix(out_root, mid, chunks, chunk_bytes, meta)
    return mid, path, sha


def parse_params(kvs):
    params = {}
    for kv in kvs or []:
        k, _, v = kv.partition("=")
        if k in ("fracs", "nodefracs"):
            params[k] = tuple(float(x) for x in v.split(","))
        elif k == "layer":
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v  # string param (trace family: pools/sem/pool/model)
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=sorted(FAMILY_DEFAULT_PARAMS))
    ap.add_argument("--W", type=int, required=True, help="world size (total ranks)")
    ap.add_argument("--ranks-per-node", type=int, required=True)
    ap.add_argument("--budget-mib", type=int, required=True, help="PRE-TOPK send budget per rank")
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--chunk-bytes", type=int, default=8192)
    ap.add_argument("--id", default="001", help="instance id mixed into the seed")
    ap.add_argument("--G", type=int, default=None, help="check feasibility for this expert count")
    ap.add_argument("--out-root", default=".", help="directory for <matrix_id>.txt + meta")
    ap.add_argument("--param", action="append", help="family param override, e.g. frac=0.5")
    ap.add_argument("--print-only", action="store_true", help="print stats, write nothing")
    ap.add_argument(
        "--traces-root",
        default=os.path.expandvars("${PSCRATCH}/workspace/andrewy/moe_traces"),
        help="root of fetched trace pools (trace family only)",
    )
    args = ap.parse_args()

    params = dict(FAMILY_DEFAULT_PARAMS[args.family], **parse_params(args.param))
    if args.print_only and args.family == "trace":
        import gen_trace_routing

        mid, params, specs, pools_rows, routing, chunks, tokens_per_rank = (
            gen_trace_routing.generate_trace(
                params,
                args.W,
                args.ranks_per_node,
                args.budget_mib,
                args.topk,
                args.chunk_bytes,
                args.id,
                args.traces_root,
                args.G,
            )
        )
        print(f"matrix_id: {mid}")
        print(
            f"tokens_per_rank (pre-topk): {tokens_per_rank},"
            f" row_chunks: {tokens_per_rank * args.topk}"
        )
        print(f"pools: {['/'.join(s) for s in specs]} rows {[len(p) for p in pools_rows]}")
        gen_trace_routing.print_stats(
            routing, chunks, args.W, args.ranks_per_node, tokens_per_rank, args.G, args.topk
        )
        return
    if args.print_only:
        mid, chunks, tokens_per_rank = generate(
            args.family,
            params,
            args.W,
            args.ranks_per_node,
            args.budget_mib,
            args.topk,
            args.chunk_bytes,
            args.id,
        )
        min_g = check_feasible(chunks, args.W, args.topk, tokens_per_rank, nexperts=args.G)
        print(f"matrix_id: {mid}")
        print(
            f"tokens_per_rank (pre-topk): {tokens_per_rank}, row_chunks: {tokens_per_rank * args.topk}"
        )
        print(f"min feasible G: {min_g}")
        for s, row in enumerate(chunks):
            remote = sum(
                c for d, c in enumerate(row) if d // args.ranks_per_node != s // args.ranks_per_node
            )
            print(f"row {s:3d}: max {max(row):8d} remote_frac {remote / sum(row):.3f}")
        nn = args.W // args.ranks_per_node
        if nn >= 2:
            st = dedup_round_stats(chunks, args.ranks_per_node, tokens_per_rank)
            print("dedup per ordered node pair (closed form, current dealer):")
            for (sn, dn_), (copies, uniq, ratio) in sorted(st["pair"].items()):
                print(f"  n{sn}->n{dn_}: copies {copies:8d}  unique {uniq:8d}  dedup {ratio:.2f}")
            print("per-dest-node round profile (dn ascending = gateway order), U rows:")
            for m in range(nn):
                prof = "  ".join(f"dn{dn}(src n{sn}) {u:7d}" for dn, sn, u in st["round_profile"][m])
                print(f"  dest n{m}: {prof}")
            L = args.ranks_per_node
            per_node = [st["col_rows"][n * L : (n + 1) * L] for n in range(nn)]
            print("per-dest-rank GEMM rows (column sums), by node:")
            for n, cols in enumerate(per_node):
                print(f"  node {n}: {' '.join(f'{c:7d}' for c in cols)}")
            print(f"predicted headroom (relay_balanced/relay_ident): {st['headroom']:.3f}")
        return
    mid, path, sha = ensure_matrix(
        args.family,
        params,
        args.W,
        args.ranks_per_node,
        args.budget_mib,
        args.topk,
        args.chunk_bytes,
        args.id,
        args.out_root,
        nexperts=args.G,
        traces_root=args.traces_root,
    )
    print(f"{mid}\n{path}\nsha256 {sha}")


if __name__ == "__main__":
    main()
