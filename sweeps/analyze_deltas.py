#!/usr/bin/env python3
"""Within-capsule arm-vs-baseline deltas from sweep capsules.

Every delta is computed INSIDE one capsule, between cells that share
(nnodes, ranks_per_node, budget_mib, topk, mode, family, matrix_id) and differ
only in `variant`. That is SCHEMA.md protocol rule 4: `git_sha` is not a build
identity, so arms are only comparable when they came from the same binary, and
the only guarantee of that is same-capsule. Deltas from different capsules are
pooled for reporting but never computed across capsules.

Latency statistic (SCHEMA.md / the `/sweep` skill): for `isolated` cells quote
the **mean over iterations of the per-iteration max-across-ranks** of `e2e_ms`.
A layer is not done until its slowest rank is done, so the max-across-ranks is
the per-iteration latency; the mean over iterations is the point estimate.
`--stat median-all` reproduces the cruder median-over-all-(rank,iter) rows if
you need to compare against an older analysis that used it.

Usage:
  python3 sweeps/analyze_deltas.py --baseline hier \
      --arms hier_compress_lb_union,hier_compress_lb_union_eager \
      --runs 20260815-044144_perlmutter_22befac3 \
      --rows nnodes,budget_mib --cols family

  # all capsules, default axes
  python3 sweeps/analyze_deltas.py --baseline hier --arms hier_compress_union
"""
import argparse
import collections
import csv
import glob
import os
import statistics

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs")

# cells sharing these fields differ only in `variant` and are directly comparable
MATCH_KEYS = ("nnodes", "ranks_per_node", "budget_mib", "topk", "mode", "family", "matrix_id")


def cell_latency(rows, stat):
    """rows: list of (rank, iter, value_ms) for one cell. -> point estimate in ms."""
    if not rows:
        return None
    if stat == "median-all":
        return statistics.median([v for _, _, v in rows])
    per_iter = collections.defaultdict(list)
    for _, it, v in rows:
        per_iter[it].append(v)
    # per-iteration max across ranks, then mean over iterations
    return statistics.mean([max(vs) for vs in per_iter.values()])


def load_capsule(run_id, stat):
    """-> (cells_by_id, latency_by_cell_id). Only status=ok cells are returned."""
    cpath = os.path.join(RUNS_DIR, run_id, "cells.csv")
    mpath = os.path.join(RUNS_DIR, run_id, "metrics.csv")
    if not os.path.exists(cpath):
        return {}, {}
    cells = {r["cell_id"]: r for r in csv.DictReader(open(cpath)) if r.get("status") == "ok"}
    raw = collections.defaultdict(list)
    if os.path.exists(mpath):
        for m in csv.DictReader(open(mpath)):
            if m["metric"] != "e2e_ms" or m["cell_id"] not in cells:
                continue
            try:
                raw[m["cell_id"]].append((m["rank"], m["iter"], float(m["value_ms"])))
            except ValueError:
                pass
    lat = {cid: cell_latency(rows, stat) for cid, rows in raw.items()}
    return cells, {k: v for k, v in lat.items() if v is not None}


def build_sha(run_id):
    """The real build identity (SCHEMA rule 4). Capsules with different .so
    hashes must not have their numbers compared, even arm-to-arm."""
    import json

    p = os.path.join(RUNS_DIR, run_id, "manifest.json")
    if not os.path.exists(p):
        return None
    libs = json.load(open(p)).get("flux_libs") or []
    for l in libs:
        if l.get("path", "").endswith("libflux_cuda.so"):
            return (l.get("sha256") or "")[:12]
    return (libs[0].get("sha256") or "")[:12] if libs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True, help="variant every arm is measured against")
    ap.add_argument("--arms", required=True, help="comma-separated variants to compare")
    ap.add_argument("--runs", help="comma-separated run_ids (default: every capsule)")
    ap.add_argument("--rows", default="nnodes", help="cells.csv fields for table rows")
    ap.add_argument("--cols", default="budget_mib", help="cells.csv field for table columns")
    ap.add_argument("--topk", type=int, help="restrict to one topk")
    ap.add_argument("--mode", default="isolated", help="restrict to one mode (default isolated)")
    ap.add_argument("--stat", default="schema", choices=["schema", "median-all"])
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rowks = [k.strip() for k in args.rows.split(",") if k.strip()]
    colk = args.cols.strip()
    run_ids = (
        [r.strip() for r in args.runs.split(",")]
        if args.runs
        else sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(RUNS_DIR + "/*/cells.csv"))
    )

    # deltas[arm][rowkey][colkey] = list of percent deltas vs baseline
    deltas = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    shas, used_runs = set(), set()

    for rid in run_ids:
        cells, lat = load_capsule(rid, args.stat)
        groups = collections.defaultdict(list)
        for cid, c in cells.items():
            if cid not in lat:
                continue
            if args.mode and c["mode"] != args.mode:
                continue
            if args.topk and int(c["topk"]) != args.topk:
                continue
            groups[tuple(c[k] for k in MATCH_KEYS)].append(c)
        for _, members in groups.items():
            base = [c for c in members if c["variant"] == args.baseline]
            if not base:
                continue
            # several baseline cells in one group => same config, take the median
            b = statistics.median([lat[c["cell_id"]] for c in base])
            if b <= 0:
                continue
            for c in members:
                if c["variant"] not in arms:
                    continue
                rk = tuple(c[k] for k in rowks)
                deltas[c["variant"]][rk][c[colk]].append((lat[c["cell_id"]] - b) / b * 100.0)
                used_runs.add(rid)
                shas.add(build_sha(rid))

    if not deltas:
        print("no comparable cells found (need baseline + arm in the same capsule)")
        return

    print("baseline: %s | stat: %s | mode: %s" % (args.baseline, args.stat, args.mode or "any"))
    print("capsules: %d | distinct build sha256: %s" % (len(used_runs), ", ".join(sorted(s or "?" for s in shas))))
    if len(shas) > 1:
        print("  NOTE: multiple builds across these capsules. Deltas are still")
        print("  within-capsule and valid, but absolute ms are not comparable.")

    colvals = sorted(
        {c for a in deltas for r in deltas[a] for c in deltas[a][r]},
        key=lambda x: (float(x) if str(x).replace(".", "", 1).isdigit() else 1e9, str(x)),
    )
    for arm in arms:
        if arm not in deltas:
            continue
        print("\n=== %s  vs  %s   (negative = arm faster, %% ) ===" % (arm, args.baseline))
        print("%-22s" % "/".join(rowks) + "".join("%14s" % ("%s=%s" % (colk, c)) for c in colvals))
        for rk in sorted(deltas[arm], key=lambda t: tuple(_num(x) for x in t)):
            line = "%-22s" % "/".join(str(x) for x in rk)
            for c in colvals:
                d = deltas[arm][rk].get(c)
                line += "%14s" % ("%+.1f(%d)" % (statistics.median(d), len(d)) if d else ".")
            print(line)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("inf")


if __name__ == "__main__":
    main()
