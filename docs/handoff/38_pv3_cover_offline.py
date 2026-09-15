"""Offline incidence check of the OPTIONAL cover-aware pv3 materialization
(pv3-cover) vs plain pv3 vs LocCap on the plotted 4n cells. Counts (and so
the constraints) are pv3's in both pv3 columns. CPU only.
Usage: python docs/handoff/38_pv3_cover_offline.py"""
import csv, os, runpy, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, plg, pv3, L, EPS, R_RED, SHAPE, METH, ROOT = (A[k] for k in
    ("pv2", "plg", "pv3", "L", "EPS", "R_RED", "SHAPE", "METH", "ROOT"))
want = [r for r in csv.DictReader(open(os.path.join(METH, "main_perf_winners.csv")))
        if r["plotted"] == "1" and "pv2" in r["winner_arm"] and r["nodes"] == "4"]
src = list(csv.DictReader(open(os.path.join(ROOT, "figs", "main_perf", "figure_src.csv"))))
print("cell          loccap_inc  pv3_inc  cover_inc  cover_vs_loccap  pv3_vs_loccap  c2viol")
for w in want:
    s = [r for r in src if (r["nodes"], r["model"], r["budget_mib"]) == (w["nodes"], w["model"], w["budget_mib"]) and r["row_id"] == w["winner_row_id"]][0]
    G, K = SHAPE[w["model"]]; W = 16; nlp = G // W + R_RED
    ofile, rfile, info = A["cell_inputs"](s["capsule"], s["cell_id"])
    orc, _, _ = A["load_routing"](ofile); bat, _, _ = A["load_routing"](rfile)
    S = bat.shape[0] // W; tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(4, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    ipr = pv3.instance_phys_of_rank(res["l2p"], res["lcnts"], nlp, W)
    phys_lc, _ = plg.loccap_route_sl(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, EPS)
    phys_p3, st = pv3.pv3_route(tk, res["p2l"], res["l2p"], res["lcnts"], nlp, L, 1, 16, return_tables=True)
    t0 = time.perf_counter()
    phys_cv = pv3.pv3_materialize_cover(tk, st["tables"], st["tables"]["rep"], ipr, nlp, L)
    tc = time.perf_counter() - t0
    chk = pv3.pv3_check(phys_cv, tk, ipr, nlp, L, 1, 16)
    assert bool(res["p2l"].long()[phys_cv].eq(tk).all())
    i_lc, _ = pv3.incidence_remote(phys_lc.long(), nlp, L)
    i_p3, _ = pv3.incidence_remote(phys_p3.long(), nlp, L)
    i_cv, _ = pv3.incidence_remote(phys_cv, nlp, L)
    print(f"4n {w['model']:>4} b{w['budget_mib']:>2}   {i_lc:>9} {i_p3:>8} {i_cv:>10}   {100*(i_cv/i_lc-1):+6.1f}%        {100*(i_p3/i_lc-1):+6.1f}%       {chk['c2_replicas_violating']}  ({tc:.1f}s ref)", flush=True)
