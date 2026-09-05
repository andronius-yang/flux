#!/usr/bin/env python3
"""Per-layer skew scan on the full decode pool caches (no window)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sweeps"))
import gen_trace_routing as gtr
from fetch_traces import MODEL_PREFIXES
from analyze_traces import sdir, summarize, W, DATA_ROOT
model, pool, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
G = {"Kimi-K2": 384, "Qwen3-235B": 128}[model]
out = {}
for ly in range(lo, hi + 1):
    cache = os.path.join(sdir(model, pool), "pool_cache", f"layer{ly}_decode.txt")
    if not os.path.exists(cache):
        continue
    rows = gtr.load_layer_pool(sdir(model, pool), ly, "decode")
    s = summarize(rows, G, len(rows[0]))
    out[ly] = {k: s[k] for k in ("max_expert_x", "topk_share", "gini", "max_rank_x", "min_rank_x", "max_node_x", "n_experts_never", "nrows")}
    print(f"{model} {pool} L{ly:2d}: maxexp={s['max_expert_x']:5.1f}x gini={s['gini']:.2f} rank[{s['min_rank_x']:.2f},{s['max_rank_x']:.2f}]x node={s['max_node_x']:.2f}x")
os.makedirs(DATA_ROOT, exist_ok=True)
json.dump(out, open(os.path.join(DATA_ROOT, f"layer_scan_{model}_{pool.replace('/','_')}.json"), "w"))
