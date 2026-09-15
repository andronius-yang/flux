"""One-GPU pv3c route timing on the REAL 32n/16n K2 b1 inputs (matrix +
routing + pv2 placement of the campaign cells) — eager vs graph vs LocCap,
plus a per-kernel breakdown via CUDA events around each launch."""
import os, sys, time, statistics, runpy, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, plg, pv3, L, EPS, R_RED, SHAPE = (A[k] for k in ("pv2", "plg", "pv3", "L", "EPS", "R_RED", "SHAPE"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from flux.testing.pv3_ext import load_ext
import flux
ext = load_ext()
cells = [("32n K2 b1", 32, "20260915-131546_perlmutter_0a42f19c", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
         ("16n K2 b1", 16, "20260915-092116_perlmutter_8a8230ae", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
         ("32n K2 b64", 32, "20260915-131929_perlmutter_54c9a682", "ours_l01_s1_pv2_r2_trace-610042_b64_k8_isolated")]
print(f"{'cell':<11} {'router':<11} {'host us':>8} {'gpu us':>8} | breakdown (gpu us)")
for tag, nodes, cap, cell in cells:
    ofile, rfile, info = A["cell_inputs"](cap, cell)
    orc, _, _ = A["load_routing"](ofile); bat, _, _ = A["load_routing"](rfile)
    G, K = SHAPE["K2"]; W = nodes * L; nlp = G // W + R_RED
    S = bat.shape[0] // W; tk = bat.view(W, S, K)
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    d = torch.zeros(W, G, dtype=torch.int32, device="cuda")
    for r in range(W):
        d[r] = torch.bincount(tk[r].reshape(-1), minlength=G).int()
    l2p_d, lc_d = res["l2p"].int().cuda().contiguous(), res["lcnts"].int().cuda().contiguous()
    tk0 = tk[0].int().cuda().contiguous()
    wsc = torch.empty(ext.workspace_ints_c(G, W), dtype=torch.int32, device="cuda")
    print(f"{tag}: S={S} W={W} nlp={nlp} Cmax={l2p_d.shape[1]} max replicas={int(lc_d.max())} mean={float(lc_d.float().mean()):.2f}")
    def eager(cn=1, cd=2):
        return ext.route_pv3c(tk0, d, l2p_d, lc_d, 0, nlp, L, cn, cd, wsc)
    def loccap():
        return flux.placelambda_route_sl(tk0, d, l2p_d, lc_d, 0, nlp, L, 0.0625, 0)
    for _ in range(3): eager(); loccap()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        keep = eager()
    g.replay(); torch.cuda.synchronize()
    def timeit(fn, iters=40):
        for _ in range(5): fn()
        hs, gs = [], []
        for _ in range(iters):
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); e0.record(); fn(); t1 = time.perf_counter(); e1.record(); torch.cuda.synchronize()
            hs.append((t1 - t0) * 1e6); gs.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(hs), statistics.median(gs)
    for name, fn in (("loccap", loccap), ("pv3c 1/2", lambda: eager(1, 2)), ("pv3c 1/4", lambda: eager(1, 4)), ("pv3c graph", g.replay)):
        h, gp = timeit(fn)
        print(f"{tag:<11} {name:<11} {h:>8.0f} {gp:>8.0f}", flush=True)
    # per-kernel breakdown with nsys-free event timing: run the pieces via a profiler-less trick —
    # time route_pv3 (tables + route only) vs route_pv3c (adds budget + vacate)
    ws3 = torch.empty(ext.workspace_ints(G, W), dtype=torch.int32, device="cuda")
    h, gp = timeit(lambda: ext.route_pv3(tk0, d, l2p_d, lc_d, 0, nlp, L, 1, 2, ws3))
    print(f"{tag:<11} {'pv3 (tab+rt)':<11} {h:>8.0f} {gp:>8.0f}   -> budget+vacate = pv3c - pv3", flush=True)
