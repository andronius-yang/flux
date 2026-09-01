"""K2 minority-weight dial sweep: for eval topic professional_law, sweep the
oracle weight w of the eval topic (rest equal) and compute rmax under
contiguous (COMET GEMM proxy), pv2 static, and pv2+swap tau=1 — find the
dial where pv2-static <= contiguous (premise A) AND swap gain large
(premise B)."""
import sys, os, random, importlib.util
import numpy as np, torch
torch.set_num_threads(4)
WT = "/pscratch/sd/y/yufeid/workspace/andrewy/flux-het-oracle"
sys.path.insert(0, os.path.join(WT, "sweeps"))
import gen_matrix, gen_trace_routing as gtr, predict_placement as PP
spec = importlib.util.spec_from_file_location("pv2", os.path.join(WT, "python/flux/testing/placement_v2.py"))
PV2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(PV2)
spec2 = importlib.util.spec_from_file_location("osw", os.path.join(WT, "python/flux/testing/ours_swap.py"))
OSW = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(OSW)
W, L, NN, EPS = 16, 4, 4, 0.0625
POOLS = ["livecodebench/execution", "mmlu/clinical_knowledge",
         "mmlu/college_mathematics", "mmlu/electrical_engineering",
         "mmlu/high_school_psychology", "mmlu/high_school_world_history",
         "mmlu/philosophy", "mmlu/professional_law"]
EVAL = "mmlu/professional_law"
G, topk, nlp = 384, 8, 384 // W + 2
T, _ = gen_matrix.budget_tokens(4, 14336, topk)
troot = PP.load_platform("perlmutter")["traces_root"]
op = {s: [list(r) for r in gtr.resolve_pools(troot, "Kimi-K2", s, 5, "decode", slots=(32, 64))[1][0]] for s in POOLS}
ep = [list(r) for r in gtr.resolve_pools(troot, "Kimi-K2", EVAL, 5, "decode", slots=(64, 96))[1][0]]
rng = random.Random(11)
tk = torch.tensor([ep[rng.randrange(len(ep))] for _ in range(W * T)], dtype=torch.int64).reshape(W, T, topk)
load_g = torch.bincount(tk.reshape(-1), minlength=G)
# contiguous / COMET proxy: expert g on rank g//epr, rank load = sum expert loads
epr = G // W
contig = load_g.reshape(W, epr).sum(1)
print(f"COMET-contig rmax {int(contig.max())} imb {float(contig.max())/float(contig.float().mean()):.3f}")
def hosts_from_p2l(p2l):
    hosts = [[] for _ in range(G)]
    for i, e in enumerate(p2l.tolist()):
        if e >= 0: hosts[e].append(i // nlp)
    return [sorted(h) for h in hosts]
for name, w in [("equal(1/8)", 1.0), ("w=1/2-of-equal", 0.5), ("w=1/4-of-equal", 0.25),
                ("w=1/8-of-equal", 0.125), ("LOO(w=0)", 0.0)]:
    weights = [w if s == EVAL else 1.0 for s in POOLS]
    oh = torch.zeros(NN, G, dtype=torch.int64)
    cum = np.cumsum(weights) / sum(weights)
    for u in range(NN):
        for _ in range(T * L):
            p = op[POOLS[int(np.searchsorted(cum, rng.random()))]]
            for e in p[rng.randrange(len(p))]: oh[u, e] += 1
    sol = PV2.pv2_solve(oh, L, nlp)
    r_static = PP.simulate_arm(tk, PV2.hosts_lists(sol, G), nlp, L, "loccap", EPS)
    orbit = OSW.swap_orbit(load_g, sol["p2l"], sol["l2p"], sol["lcnts"], L, nlp, 1, max_rounds=16)
    r_swap = PP.simulate_arm(tk, hosts_from_p2l(orbit[-1][0]) if orbit else PV2.hosts_lists(sol, G), nlp, L, "loccap", EPS)
    st, sw = r_static["rows_per_rank_max"], r_swap["rows_per_rank_max"]
    print(f"{name:16s} static rmax {st} (imb {r_static['imbalance']:.3f}, vs contig {(st/float(contig.max())-1)*100:+.1f}%)"
          f"  swap {sw} ({(sw/st-1)*100:+.1f}%)  inter {r_static['internode_rows_dedup']}")
