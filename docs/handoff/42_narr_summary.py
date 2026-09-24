"""Handoff 42 §6 narrative-check summary: per cell the rank-max per-iteration
total_ms (median over timed iterations, it0 = first timed iteration, rest_mean =
mean of the remaining timed iterations — the figs/ablation statistic) plus the
swap lane's recorded slot moves per timed iteration (records rank_000
`ours_s2_moves`; one move = one expert slot pair = w1 + w2 ≈ 2 x 29.4 MB at K2).

Usage: python docs/handoff/42_narr_summary.py <run_id> [...]
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(os.environ.get("PSCRATCH", ""), "workspace", "andrewy", "sweep_data")


def moves_of(run_id, cell_id):
    for f in glob.glob(os.path.join(DATA, run_id, "cells", cell_id, "records", "rank_000.jsonl")):
        for line in open(f):
            if "ours_s2_moves" in line:
                try:
                    d = json.loads(line)
                    while isinstance(d, dict) and "ours_s2_moves" not in d and len(d) == 1:
                        d = next(iter(d.values()))
                    for k in ("ours_s2_moves",):
                        if isinstance(d, dict) and k in d:
                            return int(d[k])
                except Exception:
                    m = re.search(r'"ours_s2_moves": (\d+)', line)
                    if m:
                        return int(m.group(1))
    return None


def main():
    rows = []
    for rid in sys.argv[1:]:
        d = os.path.join(ROOT, "sweeps", "results", "runs", rid)
        cells = {r["cell_id"]: r for r in csv.DictReader(open(os.path.join(d, "cells.csv")))}
        per = defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
            if r["metric"] != "total_ms":
                continue
            it = int(r["iter"]); v = float(r["value_ms"])
            per[r["cell_id"]][it] = max(per[r["cell_id"]].get(it, 0), v)
        for cid, c in cells.items():
            wi = 0  # metrics.csv `iter` indexes TIMED iterations only (warmups are not recorded)
            timed = [per[cid][i] for i in sorted(per[cid]) if i >= wi]
            if not timed:
                continue
            m = re.search(r"pools[\"=: ]+([a-z_/]+)", c.get("family_params", "") or "")
            sched = "sched" in (c.get("family_params", "") or "")
            mv = moves_of(rid, cid)
            rows.append((("sched:" if sched else "") + (m.group(1) if m else "?"), c["variant"], rid[:15],
                         st.median(timed), timed[0], st.mean(timed[1:]) if len(timed) > 1 else float("nan"),
                         (mv / len(timed)) if mv is not None else float("nan")))
    rows.sort()
    print(f"{'family':32s} {'arm':70s} {'run':15s} {'median':>7s} {'it0':>7s} {'rest':>7s} {'moves/it':>8s}")
    for fam, var, rid, med, it0, rest, mv in rows:
        print(f"{fam:32s} {var:70s} {rid:15s} {med:7.2f} {it0:7.2f} {rest:7.2f} {mv:8.1f}")


if __name__ == "__main__":
    main()
