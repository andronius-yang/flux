"""Offline pv3c (pv3 + per-replica-ticket cover/vacate pass) vs pv3 vs
LocCap on the plotted main-perf cells: constraint check + incidence.
Usage: python docs/handoff/38_pv3c_offline.py [4] [8] [16]"""
import csv, os, runpy, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, plg, pv3, L, EPS, R_RED, SHAPE, METH, ROOT = (A[k] for k in
    ("pv2", "plg", "pv3", "L", "EPS", "R_RED", "SHAPE", "METH", "ROOT"))
want_nodes = {int(a) for a in sys.argv[1:]} or {4}
want = [r for r in csv.DictReader(open(os.path.join(METH, "main_perf_winners.csv")))
        if r["plotted"] == "1" and "pv2" in r["winner_arm"] and int(r["nodes"]) in want_nodes]
src = list(csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", "figure_src.csv"))))
rows = []
print("cell            loccap_inc   pv3_inc  pv3c_inc  pv3c_vs_loccap  pv3_vs_loccap  c2viol  gpu_ratio_max  ref_s")
for w in want:
    s = [r for r in src if (r["nodes"], r["model"], r["budget_mib"]) == (w["nodes"], w["model"], w["budget_mib"]) and r["row_id"] == w["winner_row_id"]][0]
    nodes = int(w["nodes"]); G, K = SHAPE[w["model"]]; W = nodes * L; nlp = G // W + R_RED
    ofile, rfile, info = A["cell_inputs"](s["capsule"], s["cell_id"])
    orc, _, _ = A["load_routing"](ofile); bat, _, _ = A["load_routing"](rfile)
    S = bat.shape[0] // W; tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    phys_lc, _ = plg.loccap_route_sl(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, EPS)
    phys_p3, _ = pv3.pv3_route(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, 1, 16)
    t0 = time.perf_counter()
    phys_c, st = pv3.pv3c_route(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, 1, 16)
    tc = time.perf_counter() - t0
    i_lc, _ = pv3.incidence_remote(phys_lc.long(), nlp, L)
    i_p3, _ = pv3.incidence_remote(phys_p3.long(), nlp, L)
    i_c, _ = pv3.incidence_remote(phys_c.long(), nlp, L)
    row = dict(nodes=nodes, model=w["model"], budget=int(w["budget_mib"]), loccap=i_lc, pv3=i_p3, pv3c=i_c,
               c2viol=st["c2_replicas_violating"], gpu_max=st["gpu_ratio_max"], c3_round=st["c3_rounded_violations"])
    rows.append(row)
    print(f"{nodes:>2}n {w['model']:>4} b{w['budget_mib']:>2}   {i_lc:>9} {i_p3:>9} {i_c:>9}   {100*(i_c/i_lc-1):+6.1f}%        {100*(i_p3/i_lc-1):+6.1f}%       {st['c2_replicas_violating']}     {st['gpu_ratio_max']:.3f}     {tc:.0f}", flush=True)
tag = "_".join(str(n) for n in sorted(want_nodes))
with open(os.path.join(HERE, f"38_pv3c_offline_{tag}n.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
