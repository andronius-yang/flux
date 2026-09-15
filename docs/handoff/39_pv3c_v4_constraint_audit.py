"""Constraint audit of the pv3c KERNEL v4 on the real campaign inputs: for
each plotted cell, run route_pv3c for EVERY rank (the production call, own
rank at a time), assemble the routing and score it with pv3_check — the
paper's constraints in integer form — plus conservation and non-host rows.
C = 1/4 and 1/2. Also the pv3 (no vacate) kernel product, and the reference
pv3c incidence for the small cells. Usage (1 GPU):
  python3 docs/handoff/39_pv3c_v4_constraint_audit.py > docs/handoff/39_pv3c_v4_constraint_audit.txt"""
import os, sys, csv, time, runpy, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, plg, pv3, L, EPS, R_RED, SHAPE = (A[k] for k in ("pv2", "plg", "pv3", "L", "EPS", "R_RED", "SHAPE"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
from flux.testing.pv3_ext import load_ext
ext = load_ext()
CELLS = [
    ("K2", 4, "20260915-055811_perlmutter_ef1c6bd9", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
    ("K2", 4, "20260915-053358_perlmutter_6ac9e4c1", "ours_l01_s1_pv2_r2_trace-610042_b16_k8_isolated"),
    ("Qwen", 4, "20260915-060043_perlmutter_0e2f3e28", "ours_l01_s1_pv2_r2_trace-a2e8ab_b1_k8_isolated"),
    ("Qwen", 4, "20260915-054236_perlmutter_7fbd9cf5", "ours_l01_s1_pv2_r2_trace-a2e8ab_b16_k8_isolated"),
    ("K2", 8, "20260915-084021_perlmutter_af0e914c", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
    ("K2", 8, "20260915-084021_perlmutter_af0e914c", "ours_l01_s1_pv2_r2_trace-610042_b16_k8_isolated"),
    ("Qwen", 8, "20260915-090203_perlmutter_294ed6a4", "ours_l01_s1_pv2_r2_trace-a2e8ab_b1_k8_isolated"),
    ("Qwen", 8, "20260915-090203_perlmutter_294ed6a4", "ours_l01_s1_pv2_r2_trace-a2e8ab_b16_k8_isolated"),
    ("K2", 16, "20260915-092116_perlmutter_8a8230ae", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
    ("K2", 16, "20260915-092116_perlmutter_8a8230ae", "ours_l01_s1_pv2_r2_trace-610042_b16_k8_isolated"),
    ("Qwen", 16, "20260915-094347_perlmutter_28d64661", "ours_l01_s1_pv2_r2_trace-a2e8ab_b1_k8_isolated"),
    ("Qwen", 16, "20260915-094347_perlmutter_28d64661", "ours_l01_s1_pv2_r2_trace-a2e8ab_b16_k8_isolated"),
    ("K2", 32, "20260915-131546_perlmutter_0a42f19c", "ours_l01_s1_pv2_r2_trace-610042_b1_k8_isolated"),
    ("K2", 32, "20260915-131929_perlmutter_54c9a682", "ours_l01_s1_pv2_r2_trace-610042_b64_k8_isolated"),
]
def find_capsule(prefix):
    import glob
    runs = os.path.join(HERE, "..", "..", "sweeps", "results", "runs")
    m = glob.glob(os.path.join(runs, prefix[:15] + "*"))
    return os.path.basename(m[0]) if m else prefix
print("model nodes b   C    router  rows     c2_over c2_under nonhost c3real c3round conserve  replica_ratio      gpu_ratio        remote  incidence  (ref pv3c inc)")
rows_out = []
for model, nodes, cap, cell in CELLS:
    cap = find_capsule(cap)
    try:
        ofile, rfile, info = A["cell_inputs"](cap, cell)
    except Exception as e:  # noqa: BLE001
        print(f"{model} {nodes}n {cell}: inputs missing ({e})"); continue
    orc, _, _ = A["load_routing"](ofile); bat, _, _ = A["load_routing"](rfile)
    G, K = SHAPE[model]; W = nodes * L; nlp = G // W + R_RED
    S = bat.shape[0] // W; tk = bat.view(W, S, K)
    b = cell.split("_b")[1].split("_")[0]
    node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
    hist = torch.zeros(nodes, G, dtype=torch.int64)
    hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
    res = pv2.pv2_solve(hist, L, nlp)
    p2l, l2p, lcnts = res["p2l"], res["l2p"], res["lcnts"]
    ipr = pv3.instance_phys_of_rank(l2p, lcnts, nlp, W)
    d = torch.zeros(W, G, dtype=torch.int32, device="cuda")
    for r in range(W):
        d[r] = torch.bincount(tk[r].reshape(-1), minlength=G).int()
    l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
    tk_d = tk.int().cuda()
    wsc = torch.empty(ext.workspace_ints_c(G, W), dtype=torch.int32, device="cuda")
    ws3 = torch.empty(ext.workspace_ints(G, W), dtype=torch.int32, device="cuda")
    for cn, cd in ((1, 4), (1, 2)):
        for router, fn, ws in (("pv3", ext.route_pv3, ws3), ("pv3c", ext.route_pv3c, wsc)):
            phys = torch.empty(W, S, K, dtype=torch.int32, device="cuda")
            bad = 0
            for r in range(W):
                ph, st = fn(tk_d[r].contiguous(), d, l2p_d, lc_d, r, nlp, L, cn, cd, ws)
                bad += int(st[2]) + int(st[0])
                phys[r] = ph
            torch.cuda.synchronize()
            phys_c = phys.cpu().long()
            conserve = bool(p2l.long()[phys_c].eq(tk.long()).all())
            st = pv3.pv3_check(phys_c, tk.long(), ipr, nlp, L, cn, cd)
            inc, rem = pv3.incidence_remote(phys_c, nlp, L)
            ref = ""
            if router == "pv3c" and b == "1" and nodes <= 16:
                phr, _ = pv3.pv3c_route(tk.long(), p2l, l2p, lcnts, nlp, L, cn, cd)
                ir, _ = pv3.incidence_remote(phr.long(), nlp, L)
                ref = f"{ir} ({100*(inc/max(ir,1)-1):+.1f}%)"
            line = (f"{model:<5} {nodes:>3}n b{b:<3} 1/{cd:<2} {router:<6} {S*K*W:>8} {st['c2_over_rows']:>7} {st['c2_under_rows']:>8} "
                    f"{st['nonhost_rows']:>7} {st['c3_real_violations']:>6} {st['c3_rounded_violations']:>7} {str(conserve):<9} "
                    f"[{st['replica_ratio_min']:.3f},{st['replica_ratio_max']:.3f}] [{st['gpu_ratio_min']:.3f},{st['gpu_ratio_max']:.3f}] "
                    f"{rem:>8} {inc:>9}  {ref}  kstats_bad={bad}")
            print(line, flush=True)
            rows_out.append(dict(model=model, nodes=nodes, budget=b, C=f"1/{cd}", router=router, rows=S*K*W,
                                 c2_over=st["c2_over_rows"], c2_under=st["c2_under_rows"], nonhost=st["nonhost_rows"],
                                 c3_real=st["c3_real_violations"], c3_rounded=st["c3_rounded_violations"], conserve=conserve,
                                 replica_ratio_min=round(st["replica_ratio_min"], 4), replica_ratio_max=round(st["replica_ratio_max"], 4),
                                 gpu_ratio_min=round(st["gpu_ratio_min"], 4), gpu_ratio_max=round(st["gpu_ratio_max"], 4),
                                 remote_rows=rem, incidence=inc, ref_incidence=ref, kstats_bad=bad, capsule=cap, cell=cell))
with open(os.path.join(HERE, "39_pv3c_v4_constraint_audit.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
print("wrote 39_pv3c_v4_constraint_audit.csv", len(rows_out), "rows")
