#!/usr/bin/env python3
"""Tidy aggregation for the 2026-08-24 authentic l01 campaign.
Conventions = handoff 15: e2e/plan/plan_comm/total = per-iter MAX across
ranks, MEDIAN over iters; l0/act/l1 = the CRITICAL rank's spans (rank with
max e2e that iter) so l0+act+l1 == e2e. MoonEP derives:
e2e = total - plan_comm - plan (per rank); l0 = pack+comm+scatter+prefetch+gemm;
l1 = gemm2+cpack+comb+acc; act from act_ms."""
import csv, os, sys, statistics as st
from collections import defaultdict

RUNS = "sweeps/results/runs"
BASELINE = {"l01_torch":"Torch+GEMM","l01_allgather_dense":"COMET",
    "l01_slipstream":"Slipstream","eplb_l01":"EPLB","epic_l01_hc_m1":"EPIC",
    "llc_l01_s1":"PLL","moonep_l01_nvshmem_getmem":"MoonEP"}
# cells.csv comm_pattern label -> variant key is messy; use cell_id prefix
def variant_of(cell_id):
    for k in sorted(BASELINE, key=len, reverse=True):
        if cell_id.startswith(k+"_"): return k
    return None
def budget_of(cell_id):
    for tok in cell_id.split("_"):
        if tok.startswith("b") and tok[1:].isdigit(): return int(tok[1:])
    return None
MOONEP_L0 = ["pack_ms","comm_ms","scatter_ms","prefetch_ms","gemm_ms"]
MOONEP_L1 = ["gemm2_ms","cpack_ms","comb_ms","acc_ms"]

rows_out = []
caps = sorted(sys.argv[1:])
for cap in caps:
    d = os.path.join(RUNS, cap)
    spec = open(os.path.join(d,"spec.yaml")).read()
    nodes = int([l for l in spec.splitlines() if l.startswith("nodes:")][0].split(":")[1])
    model = "K2" if "Kimi-K2" in spec else ("Qwen" if "Qwen" in spec else "?")
    # statuses from cells.csv (last col = status? use DictReader)
    status = {}
    with open(os.path.join(d,"cells.csv")) as f:
        for r in csv.DictReader(f):
            status[r["cell_id"]] = r.get("status","?")
    # metrics
    data = defaultdict(lambda: defaultdict(dict))  # cell -> (rank,iter) -> metric -> val
    with open(os.path.join(d,"metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["mode"] != "isolated": continue
            data[r["cell_id"]][(int(r["rank"]),int(r["iter"]))][r["metric"]] = float(r["value_ms"])
    for cell, per in sorted(data.items()):
        v = variant_of(cell); b = budget_of(cell)
        iters = sorted(set(i for (_,i) in per))
        ranks = sorted(set(r for (r,_) in per))
        agg = defaultdict(list)
        for it in iters:
            recs = {r: per[(r,it)] for r in ranks if (r,it) in per}
            if not recs: continue
            def val(rec):
                if "e2e_ms" in rec: return rec["e2e_ms"]
                return rec.get("total_ms",0)-rec.get("plan_comm_ms",0)-rec.get("plan_ms",0)
            crit = max(recs, key=lambda r: val(recs[r]))
            cr = recs[crit]
            agg["e2e"].append(val(cr))
            if "l0_ms" in cr:
                agg["l0"].append(cr["l0_ms"]); agg["l1"].append(cr["l1_ms"])
                agg["act"].append(cr.get("act_ms",0.0))
            else:
                agg["l0"].append(sum(cr.get(m,0) for m in MOONEP_L0))
                agg["l1"].append(sum(cr.get(m,0) for m in MOONEP_L1))
                agg["act"].append(cr.get("act_ms",0.0))
            for m,k in (("plan_ms","plan"),("plan_comm_ms","plan_comm"),("total_ms","total")):
                agg[k].append(max(rec.get(m,0) for rec in recs.values()))
        med = lambda k: round(st.median(agg[k]),3) if agg[k] else ""
        rows_out.append(dict(baseline=BASELINE.get(v,v), variant=v, model=model,
            nodes=nodes, budget_mib=b, status=status.get(cell,"?"), capsule=cap,
            cell_id=cell, e2e_ms=med("e2e"), l0_ms=med("l0"), act_ms=med("act"),
            l1_ms=med("l1"), plan_ms=med("plan"), plan_comm_ms=med("plan_comm"),
            total_ms=med("total"),
            e2e_min=round(min(agg["e2e"]),3) if agg["e2e"] else "",
            e2e_max=round(max(agg["e2e"]),3) if agg["e2e"] else "", fail_reason=""))
    # failed/skipped cells without metrics
    for cell, stt in status.items():
        if stt != "ok" and not any(r["cell_id"]==cell and r["capsule"]==cap for r in rows_out):
            rows_out.append(dict(baseline=BASELINE.get(variant_of(cell),"?"),
                variant=variant_of(cell), model=model, nodes=nodes,
                budget_mib=budget_of(cell), status=stt, capsule=cap, cell_id=cell,
                e2e_ms="",l0_ms="",act_ms="",l1_ms="",plan_ms="",plan_comm_ms="",
                total_ms="",e2e_min="",e2e_max="",fail_reason=stt))

cols = list(rows_out[0].keys())
w = csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()
for r in sorted(rows_out, key=lambda r:(r["nodes"],r["model"],r["baseline"],r["budget_mib"] or 0)):
    w.writerow(r)
