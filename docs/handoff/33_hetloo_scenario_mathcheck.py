#!/usr/bin/env python3
"""Scenario math check: which hetero-oracle scenario delivers the most
intra-node-swap-recoverable imbalance at 4n? Same-code chain as handoff 32
S2: pv2_solve + ours_swap.swap_orbit + LocCap simulate_arm (eps 0.0625, r2).
CPU only (predict_placement.py login-node precedent).

Scenarios (oracle basis -> placement; eval = per-topic homog [64,96)):
  anchor    equal mix of ALL pools' [32,64) (handoff 32 reproduction)
  mix2tv    equal mix of the 2 most TV-distant pools; eval on those 2
  advperm   pool A + hot<->cold expert-permuted copy of A; eval both halves
  loo       leave-one-out: mix of all pools EXCEPT the eval topic
  minority  eval topic at 1/16 weight, rest equal
  burst     anchor blend oracle, eval batch BLOCK-sampled (32 contiguous
            pool rows ~ one request) instead of iid; b4 and b1
Arms per cell: matched / static blend / +swap tau=1 fixpoint (load = eval
batch, tk_dev semantics) / re-solve on batch (full-s2 bound).
"""
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

torch.set_num_threads(4)
WT = "/pscratch/sd/y/yufeid/workspace/andrewy/flux-het-oracle"
sys.path.insert(0, os.path.join(WT, "sweeps"))
import gen_matrix  # noqa: E402
import gen_trace_routing as gtr  # noqa: E402
import predict_placement as PP  # noqa: E402  (brings LC, simulate_arm)


def _imp(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(WT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PV2 = _imp("placement_v2", "python/flux/testing/placement_v2.py")
OSW = _imp("ours_swap", "python/flux/testing/ours_swap.py")

W, L = 16, 4
NN = W // L
EPS = 0.0625
SEEDS = (0, 1)
ORACLE_WIN = (32, 64)
EVAL_WIN = (64, 96)

MODELS = {
    "Kimi-K2": dict(
        G=384, topk=8, chunk=14336, layer=5,
        pools=["livecodebench/execution", "mmlu/clinical_knowledge",
               "mmlu/college_mathematics", "mmlu/electrical_engineering",
               "mmlu/high_school_psychology",
               "mmlu/high_school_world_history", "mmlu/philosophy",
               "mmlu/professional_law"]),
    "Qwen3-235B": dict(
        G=128, topk=8, chunk=8192, layer=5,
        pools=["livecodebench/execution", "mmlu/college_mathematics",
               "mmlu/high_school_world_history", "mmlu/philosophy",
               "mmlu_ZH_CN/college_mathematics",
               "mmlu_ZH_CN/high_school_world_history"]),
}

plat = PP.load_platform("perlmutter")
TROOT = plat["traces_root"]


def load_pool(model, spec, layer, win):
    _, rows, _ = gtr.resolve_pools(TROOT, model, spec, layer, "decode",
                                   slots=win)
    return [list(r) for r in rows[0]]


def marginal(rows, G):
    h = np.zeros(G)
    for r in rows:
        for e in r:
            h[e] += 1
    return h / h.sum()


def sample_rows(pools, weights, n, rng):
    """n rows drawn iid from the weighted pool mixture."""
    out = []
    cum = np.cumsum(weights) / sum(weights)
    for _ in range(n):
        u = rng.random()
        p = pools[int(np.searchsorted(cum, u))]
        out.append(p[rng.randrange(len(p))])
    return out


def sample_block_rows(pool, n, rng, block):
    """n rows in aligned blocks of `block` contiguous pool rows (~request)."""
    out = []
    nblk = len(pool) // block
    while len(out) < n:
        b = rng.randrange(nblk)
        out.extend(pool[b * block:(b + 1) * block])
    return out[:n]


def hist_from(rows_per_node, G):
    h = torch.zeros(NN, G, dtype=torch.int64)
    for u in range(NN):
        for r in rows_per_node[u]:
            for e in r:
                h[u, e] += 1
    return h


def oracle_hist(pools, weights, T, rng, G):
    return hist_from([sample_rows(pools, weights, T * L, rng)
                      for _ in range(NN)], G)


def hosts_from_p2l(p2l, G, nlp):
    hosts = [[] for _ in range(G)]
    for i, e in enumerate(p2l.tolist()):
        if e >= 0:
            hosts[e].append(i // nlp)
    return [sorted(h) for h in hosts]


def eval_batch(pool, T, rng, G, topk, block=1):
    rows = []
    for _ in range(W):
        if block > 1:
            rows.extend(sample_block_rows(pool, T, rng, block))
        else:
            rows.extend(sample_rows([pool], [1.0], T, rng))
    return torch.tensor(rows, dtype=torch.int64).reshape(W, T, topk)


def arms_for_cell(model, mc, oracle_pools, oracle_w, ev_pool, T, seed,
                  block=1):
    G, topk, nlp = mc["G"], mc["topk"], mc["G"] // W + 2
    rng = random.Random(hash((model, seed, "hetmath")) & 0xFFFFFFFF)
    tk = eval_batch(ev_pool, T, rng, G, topk, block=block)
    load_g = torch.bincount(tk.reshape(-1), minlength=G)
    batch_hist = torch.stack([
        torch.bincount(tk[u * L:(u + 1) * L].reshape(-1), minlength=G)
        for u in range(NN)])
    res = {}

    def run(tag, hosts):
        res[tag] = PP.simulate_arm(tk, hosts, nlp, L, "loccap", EPS)

    # matched: solve on the eval topic's own oracle window
    mh = oracle_hist([ev_pool], [1.0], T, rng, G) if block > 1 else None
    # matched uses the topic's ORACLE window rows, not the eval pool; caller
    # passes matched hist via oracle_pools when needed. Simplest: caller
    # supplies matched separately; here compute blend + swap + resolve.
    bh = oracle_hist(oracle_pools, oracle_w, T, rng, G)
    sol = PV2.pv2_solve(bh, L, nlp)
    run("static", PP  # noqa: E501
        .fixed_hosts(G, W) if False else PV2.hosts_lists(sol, G))
    # swap tau=1 fixpoint on the eval batch load (tk_dev semantics)
    orbit = OSW.swap_orbit(load_g, sol["p2l"], sol["l2p"], sol["lcnts"],
                           L, nlp, 1, max_rounds=16)
    if orbit:
        p2l_f, _ = orbit[-1]
        run("swap", hosts_from_p2l(p2l_f, G, nlp))
        res["swap_rounds"] = len(orbit)
    else:
        res["swap"] = res["static"]
        res["swap_rounds"] = 0
    # full re-solve on the batch (s2-full bound; movement, off the table,
    # but reported for the wire context)
    sol_b = PV2.pv2_solve(batch_hist, L, nlp)
    run("resolve", PV2.hosts_lists(sol_b, G))
    return res


def matched_arm(model, mc, topic_oracle_pool, ev_pool, T, seed, block=1):
    G, topk, nlp = mc["G"], mc["topk"], mc["G"] // W + 2
    rng = random.Random(hash((model, seed, "hetmath")) & 0xFFFFFFFF)
    tk = eval_batch(ev_pool, T, rng, G, topk, block=block)
    oh = oracle_hist([topic_oracle_pool], [1.0], T, rng, G)
    sol = PV2.pv2_solve(oh, L, nlp)
    return PP.simulate_arm(tk, PV2.hosts_lists(sol, G), nlp, L, "loccap",
                           EPS)


def agg(cells, key):
    return float(np.mean([c[key] for c in cells]))


def main():
    out = {}
    for model, mc in MODELS.items():
        G, topk, chunk, layer = mc["G"], mc["topk"], mc["chunk"], mc["layer"]
        T4, _ = gen_matrix.budget_tokens(4, chunk, topk)
        T1, _ = gen_matrix.budget_tokens(1, chunk, topk)
        opools = {s: load_pool(model, s, layer, ORACLE_WIN)
                  for s in mc["pools"]}
        epools = {s: load_pool(model, s, layer, EVAL_WIN)
                  for s in mc["pools"]}
        margs = {s: marginal(opools[s], G) for s in mc["pools"]}
        # pairwise TV on oracle-window marginals
        specs = mc["pools"]
        tv = {}
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                tv[(specs[i], specs[j])] = 0.5 * float(
                    np.abs(margs[specs[i]] - margs[specs[j]]).sum())
        pair = max(tv, key=tv.get)
        print(f"[{model}] T(b4)={T4} T(b1)={T1}  max-TV pair = {pair} "
              f"(TV={tv[pair]:.3f}); TV range "
              f"{min(tv.values()):.3f}-{max(tv.values()):.3f}", flush=True)

        scen = {}

        def cell_set(tag, topics, oracle_of, weights_of, block=1, T=None):
            Tt = T or T4
            rows = []
            for t in topics:
                for seed in SEEDS:
                    opl, wts = oracle_of(t), weights_of(t)
                    r = arms_for_cell(model, mc, opl, wts, epools[t], Tt,
                                      seed, block=block)
                    r["matched"] = matched_arm(model, mc, opools[t],
                                               epools[t], Tt, seed,
                                               block=block)
                    rows.append(r)
            def m(arm, key):
                return float(np.mean([c[arm][key] for c in rows]))
            s = {}
            for arm in ("matched", "static", "swap", "resolve"):
                s[arm] = dict(
                    imb=round(m(arm, "imbalance"), 3),
                    rmax=round(m(arm, "rows_per_rank_max"), 1),
                    inter=round(m(arm, "internode_rows_dedup"), 1))
            s["swap_rounds"] = float(np.mean([c["swap_rounds"] for c in rows]))
            scen[tag] = s
            st, sw, mt, rs = s["static"], s["swap"], s["matched"], s["resolve"]
            print(f"  {tag:14s} imb {st['imb']:.3f}->{sw['imb']:.3f} "
                  f"(matched {mt['imb']:.3f})  rmax {st['rmax']:.0f}"
                  f"->{sw['rmax']:.0f} ({(sw['rmax']/st['rmax']-1)*100:+.1f}%"
                  f" swap, {(rs['rmax']/st['rmax']-1)*100:+.1f}% resolve, "
                  f"matched {(mt['rmax']/st['rmax']-1)*100:+.1f}%)  "
                  f"inter {st['inter']:.0f}->{rs['inter']:.0f}(resolve)",
                  flush=True)

        allp = specs
        eq = lambda t: [opools[s] for s in allp]  # noqa: E731
        eqw = lambda t: [1.0] * len(allp)  # noqa: E731
        cell_set("anchor", allp, eq, eqw)
        cell_set("mix2tv", list(pair),
                 lambda t: [opools[pair[0]], opools[pair[1]]],
                 lambda t: [1.0, 1.0])
        # advperm: steepest-skew pool A + hot<->cold permuted copy
        skew = {s: float((margs[s] ** 2).sum()) for s in specs}
        A = max(skew, key=skew.get)
        order = np.argsort(-margs[A])          # hot..cold expert ids
        sigma = np.empty(G, dtype=np.int64)
        sigma[order] = order[::-1]             # hot <-> cold
        operm = [[int(sigma[e]) for e in r] for r in opools[A]]
        eperm = [[int(sigma[e]) for e in r] for r in epools[A]]
        opools["_permA"], epools["_permA"] = operm, eperm
        cell_set("advperm", [A, "_permA"],
                 lambda t: [opools[A], operm], lambda t: [1.0, 1.0])
        cell_set("loo", allp,
                 lambda t: [opools[s] for s in allp if s != t],
                 lambda t: [1.0] * (len(allp) - 1))
        cell_set("minority", allp, eq,
                 lambda t: [(1.0 if s == t else 15.0 / (len(allp) - 1))
                            for s in allp])
        cell_set("burst_b4", allp, eq, eqw, block=32)
        cell_set("burst_b1", allp, eq, eqw, block=32, T=T1)
        cell_set("iid_b1", allp, eq, eqw, T=T1)
        out[model] = scen
    path = os.path.join(os.environ.get("CLAUDE_JOB_DIR",
                                       os.path.dirname(__file__) or "."),
                        "tmp", "het_scenarios.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
