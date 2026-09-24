"""Handoff 42 capsule summary: campaign statistic (per-iteration MAX over ranks,
MEDIAN over timed iterations) of total / l0 / l1 / place / plan per cell, with
deltas vs the first-listed arm of each (family, budget) group.

Usage: python docs/handoff/42_recon_summary.py <run_id> [<run_id> ...]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
METRICS = ("total_ms", "l0_ms", "l1_ms", "place_ms", "plan_ms")


def load(run_id):
    d = os.path.join(ROOT, "sweeps", "results", "runs", run_id)
    cells = {r["cell_id"]: r for r in csv.DictReader(open(os.path.join(d, "cells.csv")))}
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # cell->metric->iter->max
    for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
        if r["metric"] not in METRICS:
            continue
        k = (r["cell_id"], r["metric"])
        it = int(r["iter"])
        v = float(r["value_ms"])
        if v > per[k[0]][k[1]].get(it, -1):
            per[k[0]][k[1]][it] = v
    rows = []
    for cid, c in cells.items():
        wi = 0  # metrics.csv `iter` indexes TIMED iterations only (warmups are not recorded)
        stat = {}
        for m in METRICS:
            vals = [v for it, v in per[cid][m].items() if it >= wi]
            stat[m] = st.median(vals) if vals else float("nan")
        import re as _re
        _m = _re.search(r"pools[\"=: ]+([a-z_/]+)", c.get("family_params", "") or "")
        rows.append((c["family"] + "/" + (_m.group(1) if _m else "?"),
                     int(float(c["budget_mib"])), c["variant"], c["status"],
                     c.get("correct_bitwise", ""), c.get("correct_allclose", ""), stat))
    return rows


def main():
    rows = []
    for rid in sys.argv[1:]:
        rows += load(rid)
    groups = defaultdict(list)
    for fam, b, var, status, cb, ca, stat in rows:
        groups[(fam, b)].append((var, status, cb, ca, stat))
    for (fam, b), lst in sorted(groups.items()):
        print(f"\n== {fam} b{b}")
        ref = lst[0][4]
        print(f"{'variant':78s} {'st':>4s} {'total':>8s} {'d%':>6s} {'l0':>7s} {'l1':>7s} {'place':>7s} {'plan':>7s} chk")
        for var, status, cb, ca, stat in lst:
            dt = 100 * (stat["total_ms"] - ref["total_ms"]) / ref["total_ms"] if ref["total_ms"] else float("nan")
            print(f"{var:78s} {status:>4s} {stat['total_ms']:8.2f} {dt:+6.1f} {stat['l0_ms']:7.2f}"
                  f" {stat['l1_ms']:7.2f} {stat['place_ms']:7.2f} {stat['plan_ms']:7.2f} {cb}/{ca}")


if __name__ == "__main__":
    main()
