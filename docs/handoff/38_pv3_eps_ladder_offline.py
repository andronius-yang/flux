"""Offline C (slack) ladder for pv3 / pv3c on the plotted cells vs LocCap
(eps 1/16): remote rows, token-node incidence, max GPU load (rows) and
max GPU load / Q_j. Usage: python docs/handoff/38_pv3_eps_ladder_offline.py [4] [8] [16]"""
import csv, os, runpy, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, plg, pv3, L, EPS, R_RED, SHAPE, METH, ROOT = (A[k] for k in
    ("pv2", "plg", "pv3", "L", "EPS", "R_RED", "SHAPE", "METH", "ROOT"))
LADDER = [(1, 16), (1, 8), (1, 4), (1, 2), (1, 1)]
want_nodes = {int(a) for a in sys.argv[1:]} or {4}
want = [r for r in csv.DictReader(open(os.path.join(METH, "main_perf_winners.csv")))
        if r["plotted"] == "1" and "pv2" in r["winner_arm"] and int(r["nodes"]) in want_nodes]
src = list(csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", "figure_src.csv"))))
rows = []
print("cell           router   C      remote   d_rem%  incid   d_inc%  rows_max  d_rows%  gpu/Q_max")
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
    ipr = pv3.instance_phys_of_rank(res["l2p"], res["lcnts"], nlp, W)
    phys_lc, _ = plg.loccap_route_sl(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, EPS)
    i_lc, r_lc = pv3.incidence_remote(phys_lc.long(), nlp, L)
    rows_lc = int(torch.bincount(phys_lc.long().reshape(-1) // nlp, minlength=W).max())
    st_lc = pv3.pv3_check(phys_lc.long(), tk, ipr, nlp, L, 1, 16)
    tag = f"{nodes}n {w['model']:>4} b{w['budget_mib']:>2}"
    print(f"{tag:<14} loccap   1/16  {r_lc:>8}   0.0%  {i_lc:>7}   0.0%  {rows_lc:>8}    0.0%   {st_lc['gpu_ratio_max']:.3f}")
    rows.append(dict(nodes=nodes, model=w["model"], budget=int(w["budget_mib"]), router="loccap", C="1/16", remote=r_lc, incid=i_lc, rows_max=rows_lc, gpu_ratio_max=st_lc["gpu_ratio_max"]))
    for cn, cd in LADDER:
        for name, fn in (("pv3", pv3.pv3_route), ("pv3c", pv3.pv3c_route)):
            ph, st = fn(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, cn, cd)
            assert st["c2_replicas_violating"] == 0, (tag, name, cn, cd, st)
            inc, rem = pv3.incidence_remote(ph.long(), nlp, L)
            rmax = int(torch.bincount(ph.long().reshape(-1) // nlp, minlength=W).max())
            print(f"{tag:<14} {name:<7} {cn}/{cd:<3} {rem:>8} {100*(rem/r_lc-1):+6.1f}%  {inc:>7} {100*(inc/i_lc-1):+6.1f}%  {rmax:>8}  {100*(rmax/rows_lc-1):+6.1f}%   {st['gpu_ratio_max']:.3f}", flush=True)
            rows.append(dict(nodes=nodes, model=w["model"], budget=int(w["budget_mib"]), router=name, C=f"{cn}/{cd}", remote=rem, incid=inc, rows_max=rmax, gpu_ratio_max=st["gpu_ratio_max"]))
tagn = "_".join(str(n) for n in sorted(want_nodes))
with open(os.path.join(HERE, f"38_pv3_eps_ladder_{tagn}n.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
