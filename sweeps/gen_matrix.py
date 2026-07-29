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
    else:
        raise ValueError(f"unknown family: {family}")
    return w


def generate(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance):
    """Return (matrix_id, chunks) where chunks is a [W][W] list of chunk counts."""
    import random

    assert W % L == 0, f"W ({W}) must be a multiple of ranks_per_node ({L})"
    nn = W // L
    if family in ("nodeskew", "remotefrac"):
        assert nn >= 2, f"family {family} needs >= 2 nodes (W={W}, L={L})"
    if family == "nodeskew":
        assert nn >= 3 or W - 2 * L > 0 or float(params["frac"]) == 1.0, (
            f"nodeskew with {nn} nodes leaves no non-hot remote ranks; use frac=1.0"
            " or remotefrac instead"
        )

    budget_bytes = budget_mib * (1 << 20)
    assert budget_bytes % chunk_bytes == 0, (
        f"budget ({budget_bytes} B) must be a multiple of chunk_bytes ({chunk_bytes})"
    )
    tokens_per_rank = budget_bytes // chunk_bytes  # pre-topk tokens
    row_chunks = tokens_per_rank * topk  # post-fanout wire rows per source

    canon = canonical_string(family, params, W, L, budget_mib, topk, chunk_bytes, matrix_instance)
    rng = random.Random(fnv1a(canon))
    rng_derived = {}
    if family == "hotcol":
        rng_derived["hot_col"] = rng.randrange(W)
    elif family == "nodeskew":
        # each source node picks a distinct-from-self hot remote node
        rng_derived["hot_node"] = [
            rng.choice([m for m in range(nn) if m != n]) for n in range(nn)
        ]
    elif family == "remotefrac":
        fracs = [float(x) for x in params["fracs"]]
        assert len(fracs) >= L, (
            f"remotefrac needs >= ranks_per_node ({L}) fractions, got {len(fracs)}"
        )
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
):
    """Generate the matrix if missing; verify sha if present. Returns
    (matrix_id, path, sha256)."""
    params = dict(FAMILY_DEFAULT_PARAMS[family], **(params or {}))
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
        if k == "fracs":
            params[k] = tuple(float(x) for x in v.split(","))
        else:
            params[k] = float(v)
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
    args = ap.parse_args()

    params = dict(FAMILY_DEFAULT_PARAMS[args.family], **parse_params(args.param))
    if args.print_only:
        mid, chunks, tokens_per_rank = generate(
            args.family, params, args.W, args.ranks_per_node, args.budget_mib, args.topk,
            args.chunk_bytes, args.id,
        )
        min_g = check_feasible(chunks, args.W, args.topk, tokens_per_rank, nexperts=args.G)
        print(f"matrix_id: {mid}")
        print(f"tokens_per_rank (pre-topk): {tokens_per_rank}, row_chunks: {tokens_per_rank * args.topk}")
        print(f"min feasible G: {min_g}")
        for s, row in enumerate(chunks):
            remote = sum(
                c for d, c in enumerate(row) if d // args.ranks_per_node != s // args.ranks_per_node
            )
            print(f"row {s:3d}: max {max(row):8d} remote_frac {remote / sum(row):.3f}")
        return
    mid, path, sha = ensure_matrix(
        args.family, params, args.W, args.ranks_per_node, args.budget_mib, args.topk,
        args.chunk_bytes, args.id, args.out_root, nexperts=args.G,
    )
    print(f"{mid}\n{path}\nsha256 {sha}")


if __name__ == "__main__":
    main()
