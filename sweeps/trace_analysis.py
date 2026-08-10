#!/usr/bin/env python3
"""Offline skew/dedup analysis of fetched trace pools — run BEFORE GPU time.

Two subcommands:

  scan   — per-layer expert-skew metrics for each pool across all (or chosen)
           MoE layers, plus cross-pool divergence per layer. Picks the layers
           worth sweeping: high per-rank load skew (receiver-side pressure)
           and high cross-pool divergence (what sem=pernode turns into
           exporter-side asymmetry).
  probe  — the go/no-go gate for a concrete trace-family arm: samples exactly
           what the generator will emit (same code path) and prints REAL
           dedup/headroom/per-round-U stats next to the dealer closed form,
           plus a fanoutskew reference. sem=pernode should show real headroom
           measurably < 1 and asymmetric per-round U segments; if it does not,
           re-arm (different pools/layer/pool=prefill) before burning GPU.

Stdlib only. Example:
    python sweeps/trace_analysis.py scan \
        --pool mmlu/college_mathematics --pool mmlu_ZH_CN/college_mathematics
    python sweeps/trace_analysis.py probe --sem pernode --layer 61 \
        --pool mmlu/college_mathematics --pool mmlu/high_school_world_history \
        --pool mmlu_ZH_CN/college_mathematics --pool mmlu_ZH_CN/high_school_world_history
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_matrix  # noqa: E402
import gen_trace_routing as gtr  # noqa: E402
from fetch_traces import MODEL_PREFIXES  # noqa: E402

DEFAULT_TRACES_ROOT = os.path.expandvars("${PSCRATCH}/workspace/andrewy/moe_traces")


def subject_dir(traces_root, model, pool_spec):
    bench, _, subject = pool_spec.partition("/")
    return os.path.join(traces_root, MODEL_PREFIXES[model], bench, subject)


def rank_loads(rows, G, W):
    """Normalized per-destination-rank GEMM-row shares of a pool under
    contiguous ownership."""
    epr = G // W
    loads = [0] * W
    for r in rows:
        for e in r:
            loads[e // epr] += 1
    total = sum(loads)
    return [x / total for x in loads]


def layer_metrics(rows, G, W, L):
    freq = [0] * G
    for r in rows:
        for e in r:
            freq[e] += 1
    total = sum(freq)
    mean = total / G
    rload = rank_loads(rows, G, W)
    nn = W // L
    nload = [sum(rload[n * L : (n + 1) * L]) for n in range(nn)]
    top = sorted(freq, reverse=True)
    k = len(rows[0])
    return {
        "max_expert_x": max(freq) / mean,  # hottest expert vs uniform
        "topk_share": sum(top[:k]) / total,  # share captured by the k hottest
        "max_rank_x": max(rload) * W,  # hottest dest rank vs uniform
        "max_node_x": max(nload) * nn,
        "rank_loads": rload,
    }


def pairwise_l1(loads_by_pool):
    """Mean pairwise L1 distance between per-rank load distributions (0 =
    identical preferences, 2 = disjoint). This is what sem=pernode converts
    into exporter-side asymmetry."""
    pools = list(loads_by_pool)
    dists = []
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            a, b = loads_by_pool[pools[i]], loads_by_pool[pools[j]]
            dists.append(sum(abs(x - y) for x, y in zip(a, b)))
    return sum(dists) / len(dists) if dists else 0.0


def cmd_scan(args):
    sdirs = {p: subject_dir(args.traces_root, args.model, p) for p in args.pool}
    for p, sdir in sdirs.items():
        gtr.build_layer_caches(sdir, None if args.layers is None else args.layers, args.token_pool)

    # discover layer set from the first pool's cache dir if not given
    if args.layers is None:
        first = next(iter(sdirs.values()))
        cache_dir = os.path.join(first, "pool_cache")
        layers = sorted(
            int(n.split("_")[0][len("layer") :])
            for n in os.listdir(cache_dir)
            if n.endswith(f"_{args.token_pool}.txt")
        )
    else:
        layers = args.layers

    report = {}
    rows_fmt = "{:>5} " + "{:>10} {:>9} {:>9} {:>9}" + "  {:>9}"
    print(rows_fmt.format("layer", "maxexp_x", "topk_sh", "maxrank_x", "maxnode_x", "xpool_L1"))
    for ly in layers:
        per_pool, loads = {}, {}
        for p, sdir in sdirs.items():
            rows = gtr.load_layer_pool(sdir, ly, args.token_pool)
            m = layer_metrics(rows, args.G, args.W, args.ranks_per_node)
            per_pool[p] = m
            loads[p] = m["rank_loads"]
        agg = {
            "max_expert_x": max(m["max_expert_x"] for m in per_pool.values()),
            "topk_share": max(m["topk_share"] for m in per_pool.values()),
            "max_rank_x": max(m["max_rank_x"] for m in per_pool.values()),
            "max_node_x": max(m["max_node_x"] for m in per_pool.values()),
            "xpool_l1": pairwise_l1(loads),
            "per_pool": {p: {k: v for k, v in m.items() if k != "rank_loads"}
                         for p, m in per_pool.items()},
        }
        report[ly] = agg
        print(
            rows_fmt.format(
                ly,
                f"{agg['max_expert_x']:.2f}",
                f"{agg['topk_share']:.3f}",
                f"{agg['max_rank_x']:.2f}",
                f"{agg['max_node_x']:.2f}",
                f"{agg['xpool_l1']:.3f}",
            )
        )

    by_div = sorted(report, key=lambda ly: -report[ly]["xpool_l1"])
    by_rank = sorted(report, key=lambda ly: -report[ly]["max_rank_x"])
    print(f"\ntop cross-pool divergence layers: {by_div[:8]}")
    print(f"top per-rank load skew layers:    {by_rank[:8]}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "pools": args.pool,
                    "token_pool": args.token_pool,
                    "G": args.G,
                    "W": args.W,
                    "ranks_per_node": args.ranks_per_node,
                    "layers": report,
                    "top_by_xpool_l1": by_div[:8],
                    "top_by_max_rank_x": by_rank[:8],
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"report: {args.report}")


def cmd_probe(args):
    params = {
        "model": args.model,
        "pool": args.token_pool,
        "sem": args.sem,
        "pools": "+".join(args.pool),
        "layer": args.layer,
    }
    mid, params, specs, pools_rows, routing, chunks, T = gtr.generate_trace(
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
    print(f"matrix_id (would be): {mid}")
    print(f"tokens_per_rank: {T}, pools: {['/'.join(s) for s in specs]}"
          f" rows {[len(p) for p in pools_rows]}")
    gtr.print_stats(routing, chunks, args.W, args.ranks_per_node, T, args.G, args.topk)

    # fanoutskew reference: same W/L/budget/topk, campaign-default nodefracs
    L, nn = args.ranks_per_node, args.W // args.ranks_per_node
    if nn >= 3:
        _, fchunks, _ = gen_matrix.generate(
            "fanoutskew",
            dict(gen_matrix.FAMILY_DEFAULT_PARAMS["fanoutskew"]),
            args.W,
            L,
            args.budget_mib,
            args.topk,
            args.chunk_bytes,
            args.id,
        )
        fst = gen_matrix.dedup_round_stats(fchunks, L, T)
        print(f"\nfanoutskew{gen_matrix.FAMILY_DEFAULT_PARAMS['fanoutskew']['nodefracs']}"
              f" reference headroom: {fst['headroom']:.3f}")

    # go/no-go verdict for lb-vs-union differentiation
    _, U = gtr.real_dedup_stats(routing, args.W, L, T, args.G)
    real = gen_matrix.dedup_stats_from_U(chunks, U, L, T)
    seg_spread = []
    for n in range(nn):
        for dn in range(1, nn):
            tn = (n - dn + nn) % nn
            seg = [U[n * L + sl][tn] for sl in range(L)]
            seg_spread.append(max(seg) / max(1, min(seg)))
    print(
        f"\nGO/NO-GO: real headroom {real['headroom']:.3f}"
        f" (lb_union reward exists iff < 1); per-(srcnode,dstnode) U segment"
        f" max/min spread: mean {sum(seg_spread)/len(seg_spread):.3f},"
        f" max {max(seg_spread):.3f} (asymmetry >~ 1.1 differentiates lb from union)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scan", cmd_scan), ("probe", cmd_probe)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--traces-root", default=DEFAULT_TRACES_ROOT)
        p.add_argument("--model", default="Qwen3-235B", choices=sorted(MODEL_PREFIXES))
        p.add_argument("--pool", action="append", required=True, help="bench/subject (repeat)")
        p.add_argument("--token-pool", default="decode", choices=gtr.POOLS,
                       help="which trace tokens form the pool (decode|prefill|all)")
        p.add_argument("--G", type=int, default=128)
        p.add_argument("--W", type=int, default=16)
        p.add_argument("--ranks-per-node", type=int, default=4)
    sp = sub.choices["scan"]
    sp.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=None)
    sp.add_argument("--report", default=None, help="write full JSON report here")
    pp = sub.choices["probe"]
    pp.add_argument("--layer", type=int, required=True)
    pp.add_argument("--sem", required=True, choices=gtr.SEMS)
    pp.add_argument("--budget-mib", type=int, default=32)
    pp.add_argument("--topk", type=int, default=8)
    pp.add_argument("--chunk-bytes", type=int, default=8192)
    pp.add_argument("--id", default="001")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
