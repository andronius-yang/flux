#!/usr/bin/env python3
"""v2chunk readout: per (arm, budget) mean-of-per-iter-max-across-ranks for
each metric, from capsule metrics.csv (isolated mode). Usage: <run_dir>..."""
import csv, os, sys
from collections import defaultdict

METRICS = ('total_ms', 'e2e_ms', 'l1_ms', 'l0_ms', 'plan_ms', 'place_ms')

def load(run_dir):
    cells = {c['cell_id']: c for c in csv.DictReader(open(os.path.join(run_dir, 'cells.csv')))}
    acc = defaultdict(lambda: defaultdict(dict))  # (cell, metric) -> iter -> rank -> v
    for r in csv.DictReader(open(os.path.join(run_dir, 'metrics.csv'))):
        acc[(r['cell_id'], r['metric'])].setdefault(int(r['iter']), {})[int(r['rank'])] = float(r['value_ms'])
    rows = {}
    for (cell, metric), its in acc.items():
        c = cells.get(cell)
        if c is None or metric not in METRICS:
            continue
        key = cell
        row = rows.setdefault(key, {
            'variant': c['variant'], 'budget': int(c['budget_mib']),
            'status': c['status'], 'run': run_dir.rstrip('/').rsplit('/', 1)[1][:15]})
        pim = [max(rk.values()) for it, rk in sorted(its.items())]
        row[metric] = sum(pim) / len(pim)
    return list(rows.values())

def main():
    rows = []
    for d in sys.argv[1:]:
        rows += load(d)
    rows.sort(key=lambda r: (r['budget'], r['variant']))
    print(f"{'variant':40s} {'b':>3s} {'stat':6s}" + ''.join(f"{m[:-3]:>9s}" for m in METRICS))
    last_b = None
    for r in rows:
        if last_b is not None and r['budget'] != last_b:
            print()
        last_b = r['budget']
        vals = ''.join(f"{r.get(m, float('nan')):9.2f}" for m in METRICS)
        print(f"{r['variant']:40s} {r['budget']:3d} {r['status'][:6]:6s}{vals}")

if __name__ == '__main__':
    main()
