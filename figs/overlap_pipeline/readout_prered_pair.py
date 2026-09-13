#!/usr/bin/env python
"""Readout for the streaming pre-reduce pair (capsule under a flux tree's sweeps/results/runs).

Usage (from the tree that ran the capsule, conda python):
  python figs/overlap_pipeline/readout_prered_pair.py <capsule_id> [--timeline <json>]

Isolated cells: per arm x family, median over timed iterations of the rank-max of
e2e_ms / l0_ms / l1_ms / place_ms (the ladder's statistic, SCHEMA rule 3: isolated = latency).
nsys cells (needs the extract_timeline.py JSON): per arm x family, 16-rank medians of the
combine landmarks relative to the l1 GEMM start (conv-0, first put, last put end, final
fold end, iteration end) and the l1 GEMM duration, on the case-study iterations
(iter33 = skewed proLaw block, iter4 = efficient plain lcb; falls back to the last timed
iteration present).
"""
import argparse, csv, json, os, statistics as st, collections, sys


def med(x):
    return st.median(x) if x else float('nan')


def isolated(cap_dir):
    rows = [r for r in csv.DictReader(open(os.path.join(cap_dir, 'metrics.csv'))) if r['mode'] == 'isolated']
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        by[r['cell_id']][r['metric']].setdefault(r['iter'], []).append(float(r['value_ms']))
    print('== isolated (median over iterations of the rank-max), ms')
    print(f"{'arm':58s} {'family':8s} {'total':>7s} {'l0':>6s} {'l1':>6s} {'place':>6s} n")
    out = {}
    for cell, mets in sorted(by.items()):
        arm = cell.split('_trace-')[0]
        fam = 'skewed' if 'b549f7' in cell else 'efficient'
        vals = {}
        for m in ('e2e_ms', 'l0_ms', 'l1_ms', 'place_ms'):
            if m in mets:
                vals[m] = med([max(v) for v in mets[m].values()])
        n = len(mets.get('e2e_ms', {}))
        out[(arm, fam)] = vals
        print(f"{arm:58s} {fam:8s} {vals.get('e2e_ms', float('nan')):7.2f} {vals.get('l0_ms', float('nan')):6.2f} "
              f"{vals.get('l1_ms', float('nan')):6.2f} {vals.get('place_ms', float('nan')):6.2f} {n}")
    return out


def timeline(path):
    j = json.load(open(path))
    print('== nsys combine landmarks (16-rank medians, ms rel. to l1 GEMM start unless noted)')
    print(f"{'arm':58s} {'fam':8s} {'iter':6s} {'l1GEMM':>6s} {'Σstart':>6s} {'Σspan':>6s} {'conv0':>6s} {'put0':>6s} {'putEnd':>7s} {'foldEnd':>7s} {'end':>6s}")
    for key, c in sorted(j['cells'].items()):
        arm = c['variant']
        fam = 'skewed' if 'b549f7' in key else 'efficient'
        want = 'iter33' if fam == 'skewed' else 'iter4'
        iters0 = c['ranks']['0']['iters']
        it = want if want in iters0 else [k for k in iters0 if not k.endswith('_warmup')][-1]
        rows = []
        for rk, rd in c['ranks'].items():
            r = rd['iters'].get(it)
            if r is None:
                continue
            ev = sorted(r['events'], key=lambda e: e['t0'])
            g1 = [e for e in ev if e['task'] == 'gemm.l1']
            g0 = [e for e in ev if e['task'] == 'gemm.l0']
            if not g1 or not g0:
                continue
            g1s, g1e, g0e = g1[0]['t0'], g1[-1]['t1'], g0[-1]['t1']
            puts = [e for e in ev if e['lane'] == 'nic' and e['t0'] >= g0e]
            nv = [e for e in ev if e['task'] == 'nvlink.token' and e['t0'] >= g1s]
            red = [e for e in ev if e['task'] == 'combine.reduce']
            pr = [e for e in ev if e['task'] == 'combine.prereduce']
            sw = [e for e in ev if e['task'] == 'nvlink.swap']
            w1 = [e for e in sw if e['t0'] < g0e]          # dispatch-side block (under l0)
            w2 = [e for e in sw if e['t0'] >= g0e]         # combine-side block (under l1)
            def ov(a, bs):                                  # busy time of a inside union of bs
                tot = 0.0
                for x in a:
                    for y in bs:
                        lo, hi = max(x['t0'], y['t0']), min(x['t1'], y['t1'])
                        tot += max(0.0, hi - lo)
                return tot
            rows.append(dict(
                w2s=(min(e['t0'] for e in w2) - g1s) if w2 else float('nan'),
                w2e=(max(e['t1'] for e in w2) - g1s) if w2 else float('nan'),
                w2busy=sum(e['t1'] - e['t0'] for e in w2),
                w2_under_prered=ov(w2, pr),
                w1s=(min(e['t0'] for e in w1) - g0[0]['t0']) if w1 else float('nan'),
                w1e=(max(e['t1'] for e in w1) - g0[0]['t0']) if w1 else float('nan'),
                g1d=g1e - g1s,
                conv0=(min(e['t0'] for e in nv) - g1s) if nv else float('nan'),
                put0=(min(p['t0'] for p in puts) - g1s) if puts else float('nan'),
                putend=(max(p['t1'] for p in puts) - g1s) if puts else float('nan'),
                foldend=(max(e['t1'] for e in red) - g1s) if red else float('nan'),
                end=r['dev_end_ms'] - g1s,
                prer=(max(e['t1'] for e in pr) - min(e['t0'] for e in pr)) if pr else float('nan'),
                prs=(min(e['t0'] for e in pr) - g1s) if pr else float('nan')))
        if not rows:
            continue
        m = {k: med([x[k] for x in rows]) for k in rows[0]}
        print(f"{arm:58s} {fam:8s} {it:6s} {m['g1d']:6.2f} {m['prs']:6.2f} {m['prer']:6.2f} {m['conv0']:6.2f} {m['put0']:6.2f} {m['putend']:7.2f} {m['foldend']:7.2f} {m['end']:6.2f}")
        if m['w2busy'] > 0 or m['w1e'] == m['w1e']:
            print(f"{'':58s} {'':8s} {'swap':6s} w1 {m['w1s']:.2f}-{m['w1e']:.2f} (rel l0 GEMM)  w2 {m['w2s']:.2f}-{m['w2e']:.2f} busy {m['w2busy']:.2f} under-Σ {m['w2_under_prered']:.2f} (rel l1 GEMM)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('capsule')
    ap.add_argument('--timeline')
    a = ap.parse_args()
    cap_dir = os.path.join('sweeps/results/runs', a.capsule)
    isolated(cap_dir)
    if a.timeline:
        timeline(a.timeline)
