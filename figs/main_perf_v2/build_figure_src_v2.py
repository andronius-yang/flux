#!/usr/bin/env python
"""Build figs/main_perf_v2/figure_src_v2.csv from v2 capsules (ADDITION to figs/main_perf,
never a replacement — see README.md).

Schema mirrors figs/main_perf/figure_src.csv: nodes, model, row_id, row_label, budget_mib,
total_ms (median over timed iterations of the rank-max e2e_ms — the ledger's
`iter_max_median`), stat, arm_variant, impl, capsule, cell_id, plus v2 columns:
l0_ms, l1_ms, sigma_kernel (stream|shipped), ledger_total_ms (the main_perf value for
the same nodes/model/row/budget, for the side-by-side), delta_pct.

Usage (from the tree that holds the capsules, e.g. flux-prered):
  python <repo>/figs/main_perf_v2/build_figure_src_v2.py --out <repo>/figs/main_perf_v2/figure_src_v2.csv \
      --ledger <repo>/figs/main_perf/figure_src.csv <capsule_id> [<capsule_id> ...]
"""
import argparse, csv, collections, os, statistics as st

ROW = {  # arm_variant -> (row_id, row_label, impl)
    'ours_l01_s1_pv2_r2_prs': ('ours12', '1+2', 'ours'),
    'ours_l01_s1_pv2_r2': ('ours12', '1+2', 'ours'),
    'ablation_l01_s2_swapall_nr_3d_dual3_str4_prs_p2p_r2': ('ours12_dispatch', '1+2 + expert dispatch (dual3, 3D-scheduled)', 'ours'),
}
MODEL = {'Kimi-K2': 'K2', 'Qwen3-235B': 'Qwen'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capsules', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', required=True)
    ap.add_argument('--runs', default='sweeps/results/runs')
    a = ap.parse_args()
    ledger = {}
    for r in csv.DictReader(open(a.ledger)):
        ledger[(r['nodes'], r['model'], r['row_id'], int(r['budget_mib']))] = float(r['total_ms'])
    out = []
    for cap in a.capsules:
        d = os.path.join(a.runs, cap)
        cells = {r['cell_id']: r for r in csv.DictReader(open(os.path.join(d, 'cells.csv')))}
        by = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in csv.DictReader(open(os.path.join(d, 'metrics.csv'))):
            if r['mode'] != 'isolated':
                continue
            by[r['cell_id']][r['metric']].setdefault(r['iter'], []).append(float(r['value_ms']))
        for cid, mets in by.items():
            c = cells[cid]
            if c['status'] != 'ok' or c['variant'] not in ROW:
                continue
            row_id, label, impl = ROW[c['variant']]
            try:
                import json
                model = MODEL.get(json.loads(c['family_params']).get('model', '?'), '?')
            except Exception:
                model = MODEL.get(c['family'].split('model=')[-1].split(';')[0], '?')
            med = lambda m: st.median([max(v) for v in mets[m].values()]) if m in mets else float('nan')
            tot = med('e2e_ms')
            key = (c['nnodes'], model, row_id, int(c['budget_mib']))
            led = ledger.get(key)
            out.append(dict(nodes=c['nnodes'], model=model, row_id=row_id, row_label=label,
                            budget_mib=c['budget_mib'], total_ms=f'{tot:.3f}', stat='iter_max_median',
                            arm_variant=c['variant'], impl=impl, capsule=cap, cell_id=cid,
                            l0_ms=f"{med('l0_ms'):.3f}", l1_ms=f"{med('l1_ms'):.3f}",
                            sigma_kernel='stream' if c['variant'].endswith('_prs') or '_prs_' in c['variant'] else 'shipped',
                            ledger_total_ms='' if led is None else f'{led:.3f}',
                            delta_pct='' if led is None else f'{(tot / led - 1) * 100:+.1f}'))
    out.sort(key=lambda r: (int(r['nodes']), r['model'], r['row_id'], r['sigma_kernel'], int(r['budget_mib'])))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print('wrote', a.out, len(out), 'rows')
    for r in out:
        print(f"  {r['nodes']}n {r['model']:4s} {r['row_id']:16s} {r['sigma_kernel']:7s} b{r['budget_mib']:<3s} {r['total_ms']:>8s}  ledger {r['ledger_total_ms']:>8s}  {r['delta_pct']:>6s}%")


if __name__ == '__main__':
    main()
