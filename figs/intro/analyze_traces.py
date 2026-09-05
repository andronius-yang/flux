#!/usr/bin/env python3
"""Intro figure data stage: expert-selection skew and its NIC-traffic image.

Reads the real routing pools through sweeps/gen_trace_routing.load_layer_pool
(same code path as the sweeps; decode-window caches). Stdlib only.

Outputs (JSON to --out): per (model, topic, layer, window):
  - expert share vector (sorted + unsorted), max/uniform, topk share, gini
  - per-rank / per-node GEMM-row loads under contiguous ownership at W=16,L=4
  - sampled [16][16] chunk matrix (homog, T tokens/rank, fixed seed) + NIC
    egress/ingress per rank (off-node only)
  - cross-topic: pairwise L1 on expert marginals, top-N hot-set Jaccard,
    per-topic rank-load rows, and a pernode (4 topics x 4 nodes) matrix
  - cross-window: [32,64) vs [64,96) same topic (the oracle-basis drift)
"""
import argparse, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sweeps"))
import gen_trace_routing as gtr  # noqa: E402
from fetch_traces import MODEL_PREFIXES  # noqa: E402

TRACES_ROOT = os.path.expandvars("${PSCRATCH}/workspace/andrewy/moe_traces")
# derived data stays on PSCRATCH (user rule: no caches/large files under home)
DATA_ROOT = os.path.expandvars("${PSCRATCH}/workspace/andrewy/figs_data/intro")
MODELS = {"Kimi-K2": 384, "Qwen3-235B": 128}
W, L = 16, 4

def sdir(model, pool):
    b, _, s = pool.partition("/")
    return os.path.join(TRACES_ROOT, MODEL_PREFIXES[model], b, s)

def marginal(rows, G):
    f = [0] * G
    for r in rows:
        for e in r:
            f[e] += 1
    t = sum(f)
    return [x / t for x in f]

def gini(v):
    s = sorted(v); n = len(s); c = 0.0
    for i, x in enumerate(s, 1):
        c += i * x
    return 2 * c / (n * sum(s)) - (n + 1) / n

def rank_loads(share, G):
    epr = G // W
    rl = [sum(share[r * epr:(r + 1) * epr]) for r in range(W)]
    nl = [sum(rl[n * L:(n + 1) * L]) for n in range(W // L)]
    return rl, nl

def summarize(rows, G, topk):
    sh = marginal(rows, G)
    srt = sorted(sh, reverse=True)
    rl, nl = rank_loads(sh, G)
    return {
        "nrows": len(rows), "topk": topk,
        "share": sh, "share_sorted": srt,
        "max_expert_x": srt[0] * G, "topk_share": sum(srt[:topk]),
        "top10pct_share": sum(srt[:G // 10]), "gini": gini(sh),
        "n_experts_never": sum(1 for x in sh if x == 0),
        "n_experts_below_half": sum(1 for x in sh if x < 0.5 / G),
        "rank_load_x": [x * W for x in rl], "node_load_x": [x * (W // L) for x in nl],
        "max_rank_x": max(rl) * W, "min_rank_x": min(rl) * W,
        "max_node_x": max(nl) * (W // L),
    }

def chunk_matrix(routing, T, G):
    epr = G // W
    m = [[0] * W for _ in range(W)]
    for i, r in enumerate(routing):
        s = i // T
        for e in r:
            m[s][e // epr] += 1
    return m

def nic_stats(m):
    W = len(m)
    eg = [sum(m[s][d] for d in range(W) if d // L != s // L) for s in range(W)]
    ing = [sum(m[s][d] for s in range(W) if d // L != s // L) for d in range(W)]
    def x(v):
        mu = sum(v) / len(v); return [a / mu for a in v]
    return {"egress": eg, "ingress": ing, "egress_x": x(eg), "ingress_x": x(ing),
            "egress_max_x": max(x(eg)), "ingress_max_x": max(x(ing))}

def l1(a, b): return sum(abs(x - y) for x, y in zip(a, b))
def hotset(share, n): return set(sorted(range(len(share)), key=lambda i: -share[i])[:n])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--T", type=int, default=2048, help="tokens per source rank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "trace_stats.json"))
    a = ap.parse_args()
    topics = {
        "Kimi-K2": ["livecodebench/execution", "mmlu/professional_law", "mmlu/college_mathematics",
                    "mmlu/high_school_psychology", "mmlu/philosophy", "mmlu/clinical_knowledge",
                    "mmlu/electrical_engineering", "mmlu/high_school_world_history"],
        "Qwen3-235B": ["livecodebench/execution", "mmlu/college_mathematics",
                       "mmlu/high_school_world_history", "mmlu/philosophy"],
    }
    out = {"layer": a.layer, "W": W, "L": L, "T": a.T, "models": {}}
    for model, G in MODELS.items():
        mo = {"G": G, "topics": {}, "cross_topic": {}, "cross_window": {}}
        pools = {}
        for tp in topics[model]:
            for win in ((64, 96), (32, 64)):
                try:
                    rows = gtr.load_layer_pool(sdir(model, tp), a.layer, "decode", slots=win)
                except SystemExit as e:
                    print(f"skip {model} {tp} {win}: {e}", file=sys.stderr); continue
                pools[(tp, win)] = rows
                s = summarize(rows, G, len(rows[0]))
                if win == (64, 96):
                    rng = random.Random(a.seed)
                    routing = gtr.sample_routing([rows], "homog", W, L, a.T, rng)
                    m = chunk_matrix(routing, a.T, G)
                    s["matrix"] = m; s["nic"] = nic_stats(m)
                    # row (sender) vs column (receiver) spread
                    rs = [sum(r) for r in m]; cs = [sum(m[i][j] for i in range(W)) for j in range(W)]
                    s["row_sum_spread"] = max(rs) / min(rs); s["col_sum_spread"] = max(cs) / min(cs)
                mo["topics"].setdefault(tp, {})[f"{win[0]}_{win[1]}"] = s
                print(f"{model:10s} {tp:32s} L{a.layer} w{win}: rows={len(rows):6d} "
                      f"maxexp={s['max_expert_x']:.1f}x topk_share={s['topk_share']:.2f} "
                      f"gini={s['gini']:.2f} rank[{s['min_rank_x']:.2f},{s['max_rank_x']:.2f}]x "
                      f"node_max={s['max_node_x']:.2f}x never={s['n_experts_never']}"
                      + (f" NICin_max={s['nic']['ingress_max_x']:.2f}x NICout_max={s['nic']['egress_max_x']:.2f}x"
                         f" rowspread={s['row_sum_spread']:.2f} colspread={s['col_sum_spread']:.2f}" if "nic" in s else ""))
        # cross-topic on the evaluated window
        tps = [tp for tp in topics[model] if (tp, (64, 96)) in pools]
        shares = {tp: marginal(pools[(tp, (64, 96))], G) for tp in tps}
        ct = {"topics": tps, "l1": {}, "jaccard_hot": {}, "rank_load_x": {}}
        nhot = G // W  # one rank's worth of experts
        for i, t1 in enumerate(tps):
            ct["rank_load_x"][t1] = [x * W for x in rank_loads(shares[t1], G)[0]]
            for t2 in tps[i + 1:]:
                ct["l1"][f"{t1}|{t2}"] = l1(shares[t1], shares[t2])
                h1, h2 = hotset(shares[t1], nhot), hotset(shares[t2], nhot)
                ct["jaccard_hot"][f"{t1}|{t2}"] = len(h1 & h2) / len(h1 | h2)
                print(f"  {model} {t1} vs {t2}: L1={ct['l1'][f'{t1}|{t2}']:.3f} "
                      f"hot{nhot}-jaccard={ct['jaccard_hot'][f'{t1}|{t2}']:.2f}")
        # pernode matrix: node i draws topic tps[i] (first 4)
        if len(tps) >= 4:
            rng = random.Random(a.seed)
            routing = gtr.sample_routing([pools[(t, (64, 96))] for t in tps[:4]], "pernode", W, L, a.T, rng)
            m = chunk_matrix(routing, a.T, G)
            ct["pernode_topics"] = tps[:4]; ct["pernode_matrix"] = m; ct["pernode_nic"] = nic_stats(m)
            rs = [sum(r) for r in m]; cs = [sum(m[i][j] for i in range(W)) for j in range(W)]
            print(f"  {model} pernode {tps[:4]}: NICin_max={ct['pernode_nic']['ingress_max_x']:.2f}x "
                  f"NICout_max={ct['pernode_nic']['egress_max_x']:.2f}x rowspread={max(rs)/min(rs):.2f} colspread={max(cs)/min(cs):.2f}")
        mo["cross_topic"] = ct
        # cross-window drift within topic
        for tp in tps:
            if (tp, (32, 64)) in pools:
                s1, s2 = marginal(pools[(tp, (32, 64))], G), shares[tp]
                h1, h2 = hotset(s1, nhot), hotset(s2, nhot)
                r1, r2 = rank_loads(s1, G)[0], rank_loads(s2, G)[0]
                mo["cross_window"][tp] = {"l1": l1(s1, s2), "jaccard_hot": len(h1 & h2) / len(h1 | h2),
                                          "rank_l1": l1(r1, r2)}
                print(f"  {model} {tp} window[32,64) vs [64,96): L1={l1(s1,s2):.3f} hot-jaccard={mo['cross_window'][tp]['jaccard_hot']:.2f} rankL1={l1(r1,r2):.3f}")
        out["models"][model] = mo
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
