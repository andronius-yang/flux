"""Postdoc scenario check: oracle = all-inclusive equal blend (anchor);
eval = PER-RANK single-topic batches (rank r <- pool r % P), plus the
per-NODE variant (node u <- pool u % P). Same-code chain as handoff 33.
Does this leave swap-capturable imbalance?"""
import sys, os, random, importlib.util
import numpy as np, torch
torch.set_num_threads(4)
WT = "/pscratch/sd/y/yufeid/workspace/andrewy/flux-het-oracle"
sys.path.insert(0, os.path.join(WT, "sweeps"))
import gen_matrix, gen_trace_routing as gtr, predict_placement as PP
def imp(n, rel):
    s = importlib.util.spec_from_file_location(n, os.path.join(WT, rel))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PV2 = imp("pv2", "python/flux/testing/placement_v2.py")
OSW = imp("osw", "python/flux/testing/ours_swap.py")
W, L, NN, EPS = 16, 4, 4, 0.0625
MODELS = {
    "Kimi-K2": (384, 8, 14336, ["livecodebench/execution","mmlu/clinical_knowledge",
        "mmlu/college_mathematics","mmlu/electrical_engineering","mmlu/high_school_psychology",
        "mmlu/high_school_world_history","mmlu/philosophy","mmlu/professional_law"]),
    "Qwen3-235B": (128, 8, 8192, ["livecodebench/execution","mmlu/college_mathematics",
        "mmlu/high_school_world_history","mmlu/philosophy",
        "mmlu_ZH_CN/college_mathematics","mmlu_ZH_CN/high_school_world_history"]),
}
troot = PP.load_platform("perlmutter")["traces_root"]
def hosts_from_p2l(p2l, G, nlp):
    hosts = [[] for _ in range(G)]
    for i, e in enumerate(p2l.tolist()):
        if e >= 0: hosts[e].append(i // nlp)
    return [sorted(h) for h in hosts]
for model, (G, topk, chunk, POOLS) in MODELS.items():
    nlp = G // W + 2
    T, _ = gen_matrix.budget_tokens(4, chunk, topk)
    op = {s: [list(r) for r in gtr.resolve_pools(troot, model, s, 5, "decode", slots=(32,64))[1][0]] for s in POOLS}
    ep = {s: [list(r) for r in gtr.resolve_pools(troot, model, s, 5, "decode", slots=(64,96))[1][0]] for s in POOLS}
    P = len(POOLS)
    for seed in (0, 1):
        rng = random.Random(100 + seed)
        # anchor blend oracle hist
        oh = torch.zeros(NN, G, dtype=torch.int64)
        for u in range(NN):
            for _ in range(T * L):
                p = op[POOLS[rng.randrange(P)]]
                for e in p[rng.randrange(len(p))]: oh[u, e] += 1
        sol = PV2.pv2_solve(oh, L, nlp)
        for variant, topic_of in (("per-rank", lambda r: POOLS[r % P]),
                                  ("per-node", lambda r: POOLS[(r // L) % P])):
            rows = []
            for r in range(W):
                pool = ep[topic_of(r)]
                rows += [pool[rng.randrange(len(pool))] for _ in range(T)]
            tk = torch.tensor(rows, dtype=torch.int64).reshape(W, T, topk)
            load_g = torch.bincount(tk.reshape(-1), minlength=G)
            st = PP.simulate_arm(tk, PV2.hosts_lists(sol, G), nlp, L, "loccap", EPS)
            orbit = OSW.swap_orbit(load_g, sol["p2l"], sol["l2p"], sol["lcnts"], L, nlp, 1, max_rounds=16)
            hf = hosts_from_p2l(orbit[-1][0], G, nlp) if orbit else PV2.hosts_lists(sol, G)
            sw = PP.simulate_arm(tk, hf, nlp, L, "loccap", EPS)
            bh = torch.stack([torch.bincount(tk[u*L:(u+1)*L].reshape(-1), minlength=G) for u in range(NN)])
            rs = PP.simulate_arm(tk, PV2.hosts_lists(PV2.pv2_solve(bh, L, nlp), G), nlp, L, "loccap", EPS)
            print(f"[{model} s{seed}] {variant:9s} static imb {st['imbalance']:.3f} rmax {st['rows_per_rank_max']}"
                  f" -> swap {sw['rows_per_rank_max']} ({(sw['rows_per_rank_max']/st['rows_per_rank_max']-1)*100:+.1f}%)"
                  f"  resolve {(rs['rows_per_rank_max']/st['rows_per_rank_max']-1)*100:+.1f}%"
                  f"  inter {st['internode_rows_dedup']}", flush=True)
