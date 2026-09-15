"""Reproduce the llc Qwen b16 pair-bound violation on one GPU: run the pv3c
kernel for every rank on the cell's exact inputs (pv2 placement from the
oracle window, batch routing file), compare (a) the pv3 stage counts vs
the reference tables M and (b) the assembled pv3c pairs vs pair_ub.
Usage: python docs/handoff/38_pv3c_pairbound_repro.py <routing.txt> <oracle_routing.txt> [C_num C_den]
"""
import os, runpy, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv2, pv3, L, R_RED = A["pv2"], A["pv3"], A["L"], A["R_RED"]
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "flux", "testing"))
import importlib.util
spec = importlib.util.spec_from_file_location("pv3_ext", os.path.join(HERE, "..", "..", "python", "flux", "testing", "pv3_ext.py"))
EXT = importlib.util.module_from_spec(spec); spec.loader.exec_module(EXT)
ext = EXT.load_ext()
rfile, ofile = sys.argv[1], sys.argv[2]
cn, cd = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, 4)
bat, G, K = A["load_routing"](rfile); orc, _, _ = A["load_routing"](ofile)
W = 16; nlp = G // W + R_RED
S = bat.shape[0] // W; tk = bat.view(W, S, K)
node_of_tok = (torch.arange(W).repeat_interleave(orc.shape[0] // W)) // L
hist = torch.zeros(W // L, G, dtype=torch.int64)
hist.index_put_((node_of_tok.repeat_interleave(K), orc.reshape(-1)), torch.ones(orc.numel(), dtype=torch.int64), accumulate=True)
res = pv2.pv2_solve(hist, L, nlp)
p2l, l2p, lcnts = res["p2l"], res["l2p"], res["lcnts"]
ipr = pv3.instance_phys_of_rank(l2p, lcnts, nlp, W)
# reference
phys_ref, st = pv3.pv3c_route(tk, p2l, l2p, lcnts, nlp, L, cn, cd, return_tables=True)
tab, rel_ref, ext_ref = st["tables"], st["release"], st["extra"]
M, rep = tab["M"], tab["rep"]
hosted = rep >= 0
# kernel, all ranks
d = torch.zeros(W, G, dtype=torch.int32, device="cuda")
for r in range(W): d[r] = torch.bincount(tk[r].reshape(-1), minlength=G).int()
l2p_d, lc_d = l2p.int().cuda().contiguous(), lcnts.int().cuda().contiguous()
ws3 = torch.empty(ext.workspace_ints(G, W), dtype=torch.int32, device="cuda")
wsc = torch.empty(ext.workspace_ints_c(G, W), dtype=torch.int32, device="cuda")
phys3 = torch.empty(W, S, K, dtype=torch.int32); physc = torch.empty(W, S, K, dtype=torch.int32)
kmax = 64
for r in range(W):
    tk_r = tk[r].int().cuda().contiguous()
    phys3[r], _ = ext.route_pv3(tk_r, d, l2p_d, lc_d, r, nlp, L, cn, cd, ws3)
    physc[r], stc = ext.route_pv3c(tk_r, d, l2p_d, lc_d, r, nlp, L, cn, cd, wsc)
    if r == 0:
        # dump the kernel's budgets for rank 0 from the workspace layout
        off = G * W + 2 * G * kmax + 2 * G + G * kmax * W + G * kmax
        rel_k = wsc[off: off + G * kmax].view(G, kmax).cpu()
        ext_k = wsc[off + G * kmax: off + 2 * G * kmax].view(G, kmax).cpu()
        # NOTE: after the vacate pass these hold the post-move budgets; compare totals instead
        print("rank0 kernel post-vacate rel sum", int(rel_k[:, :rep.shape[1]].sum()), "ext sum", int(ext_k[:, :rep.shape[1]].sum()),
              "| ref rel sum", int(rel_ref[:, 0].sum()), "ext sum", int(ext_ref[:, 0].sum()), "moved", int(stc[1]))
def counts(phys):
    src = torch.arange(W).view(W, 1, 1).expand_as(phys).reshape(-1)
    g = tk.reshape(-1); dst = phys.long().reshape(-1) // nlp
    return torch.bincount(src * G * W + g * W + dst, minlength=W * G * W).view(W, G, W)
c3 = counts(phys3)
# reference M as [W(src), G, W(dst)]
Mref = torch.zeros(W, G, W, dtype=torch.int64)
for jj in range(rep.shape[1]):
    v = hosted[:, jj]
    Mref[:, v, :] += 0
    for g in torch.nonzero(v).flatten().tolist():
        Mref[:, g, rep[g, jj]] += M[g, :, jj]
diff = (c3 != Mref)
print("pv3 stage: kernel counts != reference M entries:", int(diff.sum()), "| max abs diff", int((c3 - Mref).abs().max()))
if int(diff.sum()):
    idx = torch.nonzero(diff)[:8].tolist()
    for i, g, j in idx: print("  src", i, "g", g, "dst", j, "kernel", int(c3[i, g, j]), "ref", int(Mref[i, g, j]))
# pv3c pairs vs pair_ub
cc = counts(physc)
pair_k = cc.sum(1)
pair_ub = torch.zeros(W, W, dtype=torch.int64)
for jj in range(rep.shape[1]):
    for g in torch.nonzero(hosted[:, jj]).flatten().tolist():
        pair_ub[:, rep[g, jj]] += M[g, :, jj] + ext_ref[g, :, jj]
over = (pair_k - pair_ub).clamp(min=0)
print("pv3c pair rows over pair_ub:", int(over.sum()), "max", int(over.max()))
for i, j in torch.nonzero(over)[:6].tolist():
    gs = torch.nonzero((cc[i, :, j] - (Mref[i, :, j] + torch.stack([ext_ref[g, i, :][rep[g] == j].sum() if (rep[g] == j).any() else torch.tensor(0) for g in range(G)]))) > 0).flatten().tolist()[:6]
    print(f"  pair ({i}->{j}) kernel {int(pair_k[i,j])} ub {int(pair_ub[i,j])}; over-experts {gs}")
    for g in gs[:3]:
        jj = int(torch.nonzero(rep[g] == j)[0])
        print(f"     g{g}: kernel rows {int(cc[i,g,j])} M {int(M[g,i,jj])} extra {int(ext_ref[g,i,jj])} release {int(rel_ref[g,i,jj])} U {int(tab['U'][g])} fill {int(tab['fill'][g,jj])} Lb {int(tab['Lb'][g])} c {int(tab['c'][g])}")
stc_all = pv3.pv3_check(physc.long(), tk, ipr, nlp, L, cn, cd)
print("pv3c kernel constraint check:", {k: stc_all[k] for k in ("c2_over_rows", "c2_under_rows", "nonhost_rows")})
