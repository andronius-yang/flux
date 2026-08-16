#!/usr/bin/env python3
"""True cross-node dedup factor per (nodes, budget, family) — offline, no GPU.

docs/formalization.md §5.6: `wire_ratio` in cells.csv is `headroom`
(relay_balanced_bytes / relay_ident_bytes, a relay LOAD-BALANCE metric), not the
dedup factor. The dedup factor is the `pair` output of
gen_matrix.dedup_stats_from_U: for each ordered node pair, `copies` is how many
token-copies cross that boundary without dedup and `unique` is how many distinct
tokens actually need to, so `1 - unique/copies` is the byte saving dedup-merge
can possibly deliver.

Separating that ceiling from the measured latency gain is what isolates the
§5.4 coupling term: where the ceiling is large but the measured gain is zero,
the saving is real and being cancelled by something else.

Matrices are regenerated deterministically from the same (family, params, W, L,
budget, topk, chunk_bytes, instance) the sweep runner uses, so these are the
exact matrices the capsules ran on -- nothing is read back from PSCRATCH.

Usage:
  python3 sweeps/dedup_factor.py --topk 8 --budgets 1,2,8,16,32 --nodes 4,8,16
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402


def cross_node_dedup(chunks, L, tokens_per_rank):
    """-> (copies, unique, saving_fraction) summed over ordered node pairs."""
    st = gen_matrix.dedup_round_stats(chunks, L, tokens_per_rank)
    copies = unique = 0
    for (sn, dn), (c, u, _r) in st["pair"].items():
        if sn == dn:
            continue
        copies += c
        unique += u
    saving = (1.0 - unique / copies) if copies else float("nan")
    return copies, unique, saving, st["headroom"]


def analytic_saving(nn, topk):
    """iid-routing reference: expected distinct remote nodes hit vs remote
    crossings. Assumes each of topk picks lands uniformly on one of nn nodes."""
    if nn < 2:
        return float("nan")
    distinct = (nn - 1) * (1.0 - (1.0 - 1.0 / nn) ** topk)
    crossings = topk * (nn - 1) / nn
    return 1.0 - distinct / crossings


def routing_dedup(rows, nn, L, T, G, seed=1234):
    """Cross-node dedup ceiling for REAL routing re-mapped onto nn nodes.

    `rows` is the real per-token expert list from a <mid>.routing.txt. The
    routing (which experts a token wants, hence the real co-activation
    structure) is held fixed; only the expert->rank->node map changes, since
    epr = G // W depends on the world size.

    Rows are resampled with replacement to W*T tokens so that tokens-per-rank
    stays constant as nn grows -- otherwise node count would confound with T,
    and the dedup ceiling depends on T directly (U = min(copies, T)).

    NOTE this is a re-mapping of 4-node traces, not a measured 8/16-node trace
    run; it answers "does this routing DISTRIBUTION still collide at nn nodes",
    not "what did an 8-node trace capsule measure".
    """
    import random

    rng = random.Random(seed)
    W = nn * L
    epr = G // W
    if epr < 1:
        return float("nan")
    copies = unique = 0
    for s in range(W):
        src_node = s // L
        per_node_copies = [0] * nn
        per_node_unique = [0] * nn
        for _ in range(T):
            row = rows[rng.randrange(len(rows))]
            hit = set()
            for e in row:
                n = (e // epr) // L
                per_node_copies[n] += 1
                hit.add(n)
            for n in hit:
                per_node_unique[n] += 1
        for n in range(nn):
            if n == src_node:
                continue
            copies += per_node_copies[n]
            unique += per_node_unique[n]
    return (1.0 - unique / copies) if copies else float("nan")


def read_routing(path):
    """-> (rows, topk, G) from a <matrix_id>.routing.txt."""
    with open(path) as f:
        ntok, topk, G = (int(x) for x in f.readline().split())
        rows = [tuple(int(x) for x in line.split()) for line in f if line.strip()]
    assert len(rows) == ntok, "%s: header says %d tokens, found %d" % (path, ntok, len(rows))
    return rows, topk, G


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--routing", help="a <matrix_id>.routing.txt to re-map across --nodes")
    ap.add_argument("--tokens-per-rank", type=int, default=128, help="held constant across nn")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--budgets", default="1,2,8,16,32")
    ap.add_argument("--nodes", default="4,8,16")
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--chunk-bytes", type=int, default=8192)
    ap.add_argument("--instance", default="001")
    args = ap.parse_args()

    buds = [int(b) for b in args.budgets.split(",")]
    nodes = [int(n) for n in args.nodes.split(",")]
    L = args.ranks_per_node

    if args.routing:
        rows, topk, G = read_routing(args.routing)
        print("REAL routing re-mapped across node counts")
        print("  file: %s" % os.path.basename(args.routing))
        print("  %d real tokens, topk=%d, G=%d, T=%d held constant, L=%d\n"
              % (len(rows), topk, G, args.tokens_per_rank, L))
        print("%8s %8s %8s %14s %14s" % ("nodes", "W", "epr", "real ceiling", "dealer ceiling"))
        for nn in nodes:
            W = nn * L
            real = routing_dedup(rows, nn, L, args.tokens_per_rank, G)
            dealer = max(0.0, 1.0 - float(nn) / topk)
            print("%8d %8d %8d %13s %14s"
                  % (nn, W, G // W if G >= W else 0,
                     "n/a" if real != real else "%.1f%%" % (100 * real),
                     "%.1f%%" % (100 * dealer)))
        return

    # families as this campaign ran them: alternating skew, and the uniform null
    def fams(nn):
        out = [("uniform", {})]
        if nn >= 3:
            out.append(("fanoutskew", {"nodefracs": tuple([0.9, 0.1] * (nn // 2))}))
        return out

    print("cross-node dedup ceiling (1 - unique/copies), topk=%d, L=%d" % (args.topk, L))
    print("'analytic' = iid-routing reference (nn-1)(1-(1-1/nn)^k) / (k(nn-1)/nn)\n")
    for nn in nodes:
        W = nn * L
        print("=== %d nodes (W=%d) — analytic ceiling %.1f%% ===" % (nn, W, 100 * analytic_saving(nn, args.topk)))
        print("%-14s" % "family" + "".join("%13s" % ("b=%d" % b) for b in buds))
        for fam, params in fams(nn):
            row = "%-14s" % fam
            for b in buds:
                try:
                    _mid, chunks, T = gen_matrix.generate(
                        fam, dict(params), W, L, b, args.topk, args.chunk_bytes, args.instance
                    )
                    _c, _u, saving, _hr = cross_node_dedup(chunks, L, T)
                    row += "%13s" % ("%.1f%%" % (100 * saving))
                except Exception as e:  # infeasible config for this (W, budget)
                    row += "%13s" % ("n/a")
                    del e
            print(row)
        print()


if __name__ == "__main__":
    main()
