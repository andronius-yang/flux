"""Analyze a check_relaxed pair-bound dump (FLUX_PLL_PAIR_DUMP): recompute
the pv3c reference tables from the dumped topk/l2p/lcnts, compare the
driver's pair_ub with the recomputed one, and break the offending pairs
down per expert (kernel rows vs M + extra).
Usage: python docs/handoff/38_pv3c_pairdump_analyze.py <dump_dir> [rank]"""
import os, runpy, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
A = runpy.run_path(os.path.join(HERE, "38_pv3_audit_constraints.py"), run_name="lib")
pv3 = A["pv3"]
d = sys.argv[1]; rk = int(sys.argv[2]) if len(sys.argv) > 2 else None
files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
print("dumps:", files)
f = files[0] if rk is None else f"rank{rk:03d}.pt"
D = torch.load(os.path.join(d, f))
phys, pair, pu, topk, l2p, lcnts, nlp, L = (D[k] for k in ("phys", "pair", "pair_ub", "topk", "l2p", "lcnts", "nlp", "L"))
R, S, K = topk.shape
G = int(lcnts.numel())
print(f"router {D['router']} eps {D['eps']} rank {D['rank']} R {R} S {S} K {K} G {G} nlp {nlp}")
import importlib.util
spec = importlib.util.spec_from_file_location("pv3_ext", os.path.join(HERE, "..", "..", "python", "flux", "testing", "pv3_ext.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
cn, cd = E.c_rational(float(D["eps"]))
p2l = D["p2l"]
phys_ref, st = pv3.pv3c_route(topk, p2l, l2p, lcnts, nlp, L, cn, cd, return_tables=True)
tab, ext = st["tables"], st["extra"]
M, rep = tab["M"], tab["rep"]; hosted = rep >= 0
pair_ub = torch.zeros(R, R, dtype=torch.int64)
for jj in range(rep.shape[1]):
    for g in torch.nonzero(hosted[:, jj]).flatten().tolist():
        pair_ub[:, rep[g, jj]] += M[g, :, jj] + ext[g, :, jj]
print("driver pair_ub == recomputed:", torch.equal(pu.long(), pair_ub), "| max abs diff", int((pu.long() - pair_ub).abs().max()))
over = (pair.long() - pu.long()).clamp(min=0)
print("pairs over:", torch.nonzero(over).tolist()[:8], "rows over", int(over.sum()))
src = torch.arange(R).view(R, 1).expand(R, S * K).reshape(-1)
g_all = topk.reshape(-1).long(); dst = phys.long().reshape(-1) // nlp
cnt = torch.bincount(src * G * R + g_all * R + dst, minlength=R * G * R).view(R, G, R)
for i, j in torch.nonzero(over).tolist()[:4]:
    print(f"pair ({i}->{j}): rows {int(pair[i,j])} ub {int(pu[i,j])}")
    for g in range(G):
        if not (rep[g] == j).any(): 
            if cnt[i, g, j] > 0: print(f"   g{g}: {int(cnt[i,g,j])} rows at a NON-HOSTING rank!")
            continue
        jj = int(torch.nonzero(rep[g] == j)[0])
        bound = int(M[g, i, jj] + ext[g, i, jj])
        if int(cnt[i, g, j]) > bound:
            print(f"   g{g}: kernel {int(cnt[i,g,j])} > M {int(M[g,i,jj])} + extra {int(ext[g,i,jj])}  (fill {int(tab['fill'][g,jj])} U {int(tab['U'][g])} Lb {int(tab['Lb'][g])} c {int(tab['c'][g])} d_i {int(st['d'][i,g])})")
# also: per-replica totals vs U on the assembled routing
load_gr = torch.bincount(g_all * R + dst, minlength=G * R).view(G, R)
for g in range(G):
    for jj in range(rep.shape[1]):
        j = int(rep[g, jj])
        if j >= 0 and int(load_gr[g, j]) > int(tab["U"][g]):
            print(f"REPLICA CAP VIOLATED g{g} rank{j}: {int(load_gr[g,j])} > U {int(tab['U'][g])}")
print("replica cap scan done")
