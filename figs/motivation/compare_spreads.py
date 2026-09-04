#!/usr/bin/env python3
"""Per-arm, all-16-rank spreads of the motivation figure's three imbalance
dimensions, for one or more phase JSONs (extract_phases.py output), side by
side. Dimensions (layer-0 window, middle timed iteration):

  ring  : NIC occupancy (inter-node puts) + wait-before-GEMM, expert GEMM
  COMET : inter-node fetch, expert GEMM
  EPLB  : placement span (one-shot), NIC occupancy of the exposed dispatch
          wire, expert GEMM (should stay flat)

  python figs/motivation/compare_spreads.py LABEL=phases_a.json[,phases_b.json] LABEL2=...
"""
import json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1"))
import build_v1_lanes as V  # noqa: E402


def sp(v):
    v = list(v); return f"{min(v):6.2f}–{max(v):6.2f}  {max(v) / min(v):4.2f}x  ({max(v) / st.mean(v):4.2f}x max/mean)"


def main():
    sets = []
    for arg in sys.argv[1:]:
        label, _, files = arg.partition("=")
        d = {"cells": {}}
        for f in files.split(","): d["cells"].update(json.load(open(f))["cells"])
        sets.append((label, d))
    for budget in (16, 32):
        print(f"\n===== b{budget} =====")
        for arm, title in V.ARMS:
            print(f"-- {title}")
            for label, d in sets:
                cells = {c["variant"]: c for c in d["cells"].values() if c["budget_mib"] == budget and c["status"] == "ok"}
                if arm not in cells: print(f"   {label:<14} (not captured)"); continue
                c = cells[arm]; it = sorted(c["ranks"]["0"]["iters"])[1]
                M, P = V.rank_data(c, arm, it)
                rows = c["info"]["gemm_rows_per_rank"]["0"] if arm.startswith("eplb") else c["rows_per_rank"]
                print(f"   {label:<14} GEMM ms  {sp(m['gemm'] for m in M.values())}   rows {min(rows)}–{max(rows)} ({max(rows) / min(rows):.2f}x)")
                print(f"   {'':<14} NIC ms   {sp(m['inter'] for m in M.values())}   wait-before-GEMM {sp(m['wait'] for m in M.values())}")
                if P:
                    print(f"   {'':<14} place ms {sp(p['span'] for p in P.values())}   puts {min(p['n_puts'] for p in P.values())}–{max(p['n_puts'] for p in P.values())}  inter GB {min(p['inter_bytes'] for p in P.values()) / 2**30:.2f}–{max(p['inter_bytes'] for p in P.values()) / 2**30:.2f}")
                    wb = c["info"]["eplb_wire_bytes"]["0"]; W, rpn = c["W"], c["rpn"]
                    recv = [sum(wb[s][dd] for s in range(W) if dd // rpn != s // rpn) / 2**20 for dd in range(W)]
                    print(f"   {'':<14} dispatch inter-node recv MB {sp(recv)}")
                else:
                    print(f"   {'':<14} dispatch inter-node recv MB {sp(b / 2**20 for b in c['recv_inter_bytes'])}")


if __name__ == "__main__":
    main()
