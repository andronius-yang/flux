"""Offline audit (2026-09-14, branch pv3): score the CURRENT main-perf
routing (LocCap sender-local reference, eps 1/16, pv2 placement, r2) and
the PV3 rotation water-fill against the paper's routing constraints on
the plotted main-perf cells' REAL inputs (same recipe as
figs/methodology/placement_routing/tier_fill_offline.py: pv2 solved from
the cell's oracle window, routed on the cell's batch routing file).

Per cell and router:
  c2_over/under_rows   rows outside [floor((1-C)q), ceil((1+C)q)] per replica
  c2_replicas_viol     replicas outside their band
  replica_ratio        min/max realized load / q over replicas with demand
  gpu_ratio            min/max realized GPU load / Q_j (Q_j = sum_e q_j^e)
  c3_real/rounded      GPUs violating |load - Q| <= C Q (real / +n_j rows)
  uniform_cap_over     rows above ceil((1+eps) S K) (LocCap's own cap)
  remote_rows / incid  inter-node rows and token-node incidence
CPU only, torch-only imports. Usage:
  python docs/handoff/38_pv3_audit_constraints.py [4] [8] [16]
"""
import csv
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TESTING = os.path.join(ROOT, "python", "flux", "testing")
METH = os.path.join(ROOT, "figs", "methodology", "placement_routing")
STAGING = "/pscratch/sd/y/yufeid/workspace/andrewy/sweep_data"
EPS = 0.0625
C_NUM, C_DEN = 1, 16
R_RED = 2
L = 4
SHAPE = {"K2": (384, 8), "Qwen": (128, 8)}


def load_by_path(name, path):
    mod = type(sys)(name)
    mod.__file__ = path
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    return mod


pv2 = load_by_path("pv2_off", os.path.join(TESTING, "placement_v2.py"))
plg = load_by_path("plg_off", os.path.join(TESTING, "placelambda_gpu.py"))
pv3 = load_by_path("pv3_off", os.path.join(TESTING, "pv3_route.py"))
lcs = load_by_path("lcs_off", os.path.join(TESTING, "loccap_semantics.py"))


def load_routing(path):
    with open(path) as f:
        n, k, g = (int(x) for x in f.readline().split())
        vals = torch.tensor([int(x) for x in f.read().split()],
                            dtype=torch.int64)
    return vals.view(n, k), g, k


def cell_inputs(capsule, cell_id):
    rec = os.path.join(STAGING, capsule, "cells", cell_id, "records",
                       "rank_000.jsonl")
    info = {}
    for line in open(rec):
        d = json.loads(line)
        if d.get("type") == "cell_info":
            info.update(d)
    ofile = info["epic_pll_oracle_file"]
    return ofile, ofile.replace(".oracle_routing.txt", ".routing.txt"), info


def score(tag, phys, tk, ipr, nlp, S, K):
    st = pv3.pv3_check(phys.long(), tk, ipr, nlp, L, C_NUM, C_DEN)
    inc, rem = pv3.incidence_remote(phys.long(), nlp, L)
    cap = int(math.ceil((1.0 + EPS) * S * K))
    rows = torch.bincount(phys.long().reshape(-1) // nlp,
                          minlength=ipr.shape[1])
    st["uniform_cap_over"] = int((rows - cap).clamp(min=0).sum())
    st["remote_rows"] = rem
    st["incidence"] = inc
    st["router"] = tag
    return st


def analyze(nodes, model, budget, capsule, cell_id):
    G, K = SHAPE[model]
    W = nodes * L
    nlp = G // W + R_RED
    ofile, rfile, info = cell_inputs(capsule, cell_id)
    orc, g1, k1 = load_routing(ofile)
    bat, g2, k2 = load_routing(rfile)
    assert g1 == g2 == G and k1 == k2 == K
    S = bat.shape[0] // W
    tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)),
                    torch.ones(orc.numel(), dtype=torch.int64),
                    accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    ipr = pv3.instance_phys_of_rank(res["l2p"], res["lcnts"], nlp, W)
    out = []
    base = dict(nodes=nodes, model=model, budget_mib=budget, S=S, K=K,
                nlp=nlp, cell_id=cell_id, rows_total=W * S * K,
                copies=int(res["lcnts"].sum()),
                drift_ppm=info.get("epic_pll_oracle_drift_ppm"))
    # --- current router: LocCap sender-local reference ---
    t0 = time.perf_counter()
    phys_lc, st_lc = plg.loccap_route_sl(tk, res["p2l"], res["l2p"],
                                         res["lcnts"], nlp, L, EPS)
    t_lc = time.perf_counter() - t0
    s = score("loccap_sl", phys_lc, tk, ipr, nlp, S, K)
    s.update(base, ref_ms=1e3 * t_lc, forced=st_lc["forced_overflow"],
             over_cap_rows=st_lc["over_cap_rows"])
    out.append(s)
    # --- pv3 ---
    t0 = time.perf_counter()
    phys_p3, st_p3 = pv3.pv3_route(tk, res["p2l"], res["l2p"], res["lcnts"],
                                   nlp, L, C_NUM, C_DEN)
    t_p3 = time.perf_counter() - t0
    s = score("pv3", phys_p3, tk, ipr, nlp, S, K)
    s.update(base, ref_ms=1e3 * t_p3, forced=0, over_cap_rows=0)
    out.append(s)
    # --- equal split (feasibility witness, locality-oblivious) ---
    phys_es = lcs.evensplit_route(tk, res["l2p"], res["lcnts"])
    s = score("evensplit", phys_es, tk, ipr, nlp, S, K)
    s.update(base, ref_ms=float("nan"), forced=0, over_cap_rows=0)
    out.append(s)
    return out


def main():
    want_nodes = {int(a) for a in sys.argv[1:]} or {4, 8, 16}
    win = [r for r in csv.DictReader(
        open(os.path.join(METH, "main_perf_winners.csv")))
        if r["plotted"] == "1" and "pv2" in r["winner_arm"]
        and int(r["nodes"]) in want_nodes]
    src = list(csv.DictReader(open(os.path.join(
        ROOT, "figs", "main_perf", "figure_src.csv"))))
    rows = []
    for w in win:
        s = [r for r in src if (r["nodes"], r["model"], r["budget_mib"]) ==
             (w["nodes"], w["model"], w["budget_mib"])
             and r["row_id"] == w["winner_row_id"]][0]
        for st in analyze(int(w["nodes"]), w["model"], int(w["budget_mib"]),
                          s["capsule"], s["cell_id"]):
            rows.append(st)
            print({k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in st.items()}, flush=True)
    keys = ["nodes", "model", "budget_mib", "router", "S", "K", "nlp",
            "rows_total", "copies", "drift_ppm", "ref_ms", "forced",
            "over_cap_rows", "uniform_cap_over", "c2_over_rows",
            "c2_under_rows", "c2_replicas_violating", "nonhost_rows",
            "replica_ratio_min", "replica_ratio_max", "gpu_ratio_min",
            "gpu_ratio_max", "c3_real_violations", "c3_rounded_violations",
            "rows_max", "Q_max", "Q_over_SK_max", "imbalance_max_over_mean",
            "remote_rows", "incidence", "cell_id"]
    tag = "_".join(str(n) for n in sorted(want_nodes))
    outp = os.path.join(HERE, f"38_pv3_audit_constraints_{tag}n.csv")
    with open(outp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", outp)


if __name__ == "__main__":
    main()
