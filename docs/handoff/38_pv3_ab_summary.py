"""Summarize a pv3-vs-LocCap A/B capsule (branch pv3, 2026-09-14): per
cell, the datapoint-campaign statistic (per-iteration MAX across ranks,
MEDIAN over timed iterations) of total / plan / plan_comm / l0 / l1 ms,
paired LocCap arm vs its _pv3 twin at the same budget.

Usage: python docs/handoff/38_pv3_ab_summary.py <run_id> [<run_id> ...]
(capsules under sweeps/results/runs/).
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
METRICS = ["total_ms", "plan_comm_ms", "plan_ms", "place_ms", "l0_ms",
           "l1_ms", "e2e_ms"]


def iter_max_median(rows):
    """rows: list of (rank, iter, value) -> median over iters of max over
    ranks."""
    per_iter = defaultdict(list)
    for r, i, v in rows:
        per_iter[i].append(v)
    return statistics.median(max(v) for v in per_iter.values())


def summarize(run_id):
    d = os.path.join(ROOT, "sweeps", "results", "runs", run_id)
    cells = {r["cell_id"]: r for r in csv.DictReader(
        open(os.path.join(d, "cells.csv")))}
    raw = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
        if r["metric"] in METRICS:
            raw[(r["cell_id"], r["metric"])].append(
                (int(r["rank"]), int(r["iter"]), float(r["value_ms"])))
    stat = {k: iter_max_median(v) for k, v in raw.items()}
    out = []
    for cid, c in cells.items():
        row = dict(run_id=run_id[:15], cell_id=cid, variant=c["variant"],
                   budget=int(c["budget_mib"]), status=c["status"],
                   deterministic=c["deterministic"])
        for m in METRICS:
            row[m] = stat.get((cid, m))
        out.append(row)
    return out


def main():
    rows = []
    for rid in sys.argv[1:]:
        rows += summarize(rid)
    # pair LocCap arm with its _pv3 twin
    by = {(r["variant"], r["budget"]): r for r in rows}
    print(f"{'variant':<58} {'b':>3} {'st':<3} {'total':>8} {'plan_c':>7} "
          f"{'plan':>7} {'place':>7} {'l0':>8} {'l1':>8}  dTotal%")
    for (v, b), r in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = v[:-4] if v.endswith("_pv3") else None
        d = ""
        if base and (base, b) in by and by[(base, b)]["total_ms"] and \
                r["total_ms"]:
            d = f"{100 * (r['total_ms'] / by[(base, b)]['total_ms'] - 1):+.1f}"
        f = lambda x: f"{x:8.3f}" if x is not None else f"{'-':>8}"
        print(f"{v:<58} {b:>3} {r['status'][:3]:<3} {f(r['total_ms'])} "
              f"{f(r['plan_comm_ms'])[1:]} {f(r['plan_ms'])[1:]} "
              f"{f(r['place_ms'])[1:]} {f(r['l0_ms'])} {f(r['l1_ms'])}  {d}")
    tag = "_".join(a[:15] for a in sys.argv[1:])
    outp = os.path.join(HERE, f"38_pv3_ab_{tag}.csv")
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", outp)


if __name__ == "__main__":
    main()
