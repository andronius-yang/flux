#!/usr/bin/env python3
"""Data stage for the intro figure: writes figure_src.csv (tidy, small) from
the real routing pools through the sweeps' own loader (stdlib only).

Rows:
  kind=expert  topic  expert_id  norm_count      per-expert routed-token count / uniform
  kind=cell    topic  src        dst   norm_chunks  sampled [16][16] chunk matrix / uniform cell
Provenance header lines (#) record model, layer, window, T, seed, nrows.
"""
import argparse, csv, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sweeps"))
import gen_trace_routing as gtr  # noqa: E402
from analyze_traces import sdir, marginal, chunk_matrix, W, L  # noqa: E402

MODELS = {"Kimi-K2": 384, "Qwen3-235B": 128}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Kimi-K2")
    ap.add_argument("--topics", default="livecodebench/execution,mmlu/professional_law")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--window", default="64:96", help="decode-slot window start:end (the evaluated window)")
    ap.add_argument("--T", type=int, default=2048, help="tokens per source rank in the sampled batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_src.csv"))
    a = ap.parse_args()
    G = MODELS[a.model]
    win = tuple(int(x) for x in a.window.split(":"))
    rows_out, prov = [], []
    for tp in a.topics.split(","):
        rows = gtr.load_layer_pool(sdir(a.model, tp), a.layer, "decode", slots=win)
        sh = marginal(rows, G)
        for e in range(G):
            rows_out.append(("expert", tp, e, "", f"{sh[e] * G:.5f}"))
        rng = random.Random(a.seed)
        routing = gtr.sample_routing([rows], "homog", W, L, a.T, rng)
        m = chunk_matrix(routing, a.T, G)
        mu = sum(map(sum, m)) / (W * W)
        for s in range(W):
            for d in range(W):
                rows_out.append(("cell", tp, s, d, f"{m[s][d] / mu:.5f}"))
        prov.append(f"# topic={tp} nrows={len(rows)} topk={len(rows[0])} hottest_expert_x={max(sh) * G:.2f}")
    with open(a.out, "w", newline="") as f:
        f.write(f"# model={a.model} G={G} layer={a.layer} pool=decode window={win[0]}:{win[1]} "
                f"W={W} L={L} T={a.T} seed={a.seed} sem=homog ownership=contiguous\n")
        for p in prov:
            f.write(p + "\n")
        w = csv.writer(f)
        w.writerow(["kind", "topic", "i", "j", "value"])
        w.writerows(rows_out)
    print("wrote", a.out, len(rows_out), "rows")

if __name__ == "__main__":
    main()
