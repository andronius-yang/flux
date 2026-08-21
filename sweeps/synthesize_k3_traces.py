#!/usr/bin/env python3
"""Synthesize Kimi-K3-shape trace pools from real Kimi-K2 routing marginals.

Method ("empirical-marginal hot-split v2", campaign 8.20.traffic-arch):
per (topic, layer), fit K2's 384-expert hotness marginal from decode rows;
upscale 384->896 by EXACT slot splitting — each K2 expert's cross-topic
share vector becomes 2 or 3 equal child slots (hottest 128 by mean -> 3),
preserving each topic's marginal mass-exactly AND cross-topic divergences
exactly, without inventing token-level expert groupings (the parent-split
lift was rejected for forcing sibling pairs per token; children here exist
only at the marginal level and tokens sample i.i.d.). A fixed seeded permutation then scatters the
hotness-ordered slots onto the 896 expert ids (real model ids are arbitrary
w.r.t. hotness; without this, contiguous rank ownership would hand rank 0
every hot expert). Tokens sample 16 DISTINCT experts via seeded
Gumbel-top-k, with an iterative weight calibration so the realized
without-replacement marginal matches the target curve.

Output: pools in the EXACT trace JSON format under
<traces_root>/Kimi-K3-synthesized/<bench>/<subject>/<n>.json — element 0 is
a prefill block [n_prefill][16], elements 1+ are decode steps [1][16], layer
keys are K3-relabeled ({"15":23,"30":46,"60":92} recorded in the manifest) —
plus a pool.manifest.json (fetch_traces discipline + provenance extensions).
Both sem=pernode and sem=homog consume these with zero downstream changes
(spec: family model=Kimi-K3-synth).

Self-test gates (hard-fail): realized-vs-target marginals (through the real
gen_trace_routing parser), cross-topic divergence matrix vs real K2, and
same-seed determinism. Numpy + stdlib only; CPU; minutes.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_traces  # noqa: E402
import gen_trace_routing as gtr  # noqa: E402

METHOD = "empirical-marginal hot-split v2"
SRC_MODEL = "Kimi-K2"
OUT_MODEL = "Kimi-K3-synth"
G_SRC, G_OUT, TOPK_OUT = 384, 896, 16
# K2 fit layer -> K3 label (93 layers, layer 0 dense; depths 0.25/0.5/late)
LAYER_MAP = {15: 23, 30: 46, 60: 92}
DEFAULT_POOLS = [
    "mmlu/college_mathematics",
    "mmlu/high_school_world_history",
    "mmlu/philosophy",
    "livecodebench/execution",
    "mmlu/professional_law",
    "mmlu/high_school_psychology",
    "mmlu/clinical_knowledge",
    "mmlu/electrical_engineering",
]
PCTS = (0.01, 0.05, 0.10, 0.25, 0.50)


def subject_dir(traces_root, model, pool_spec):
    bench, _, subject = pool_spec.partition("/")
    return os.path.join(traces_root, fetch_traces.MODEL_PREFIXES[model], bench, subject)


def marginal_from_rows(rows, g):
    freq = np.zeros(g, dtype=np.int64)
    for r in rows:
        freq[list(r)] += 1
    total = freq.sum()
    if total == 0:
        raise SystemExit("empty pool")
    return freq / float(total)


def curve_stats(shares, g):
    s = np.sort(shares)[::-1]
    cum = np.cumsum(s)
    out = {f"top{int(p * 100)}": float(cum[max(1, round(p * g)) - 1]) for p in PCTS}
    out["max_x"] = float(s[0] * g)
    idx = np.arange(1, g + 1)
    out["gini"] = float(np.sum((2 * idx - g - 1) * s[::-1]) / g)
    return out


def hot_split_upscale(m_src):
    """[G_SRC, P] share columns -> [G_OUT, P] by EXACT slot splitting.

    Each source expert's whole share VECTOR is split into k equal child
    slots (hottest 128 by mean share -> 3, remaining 256 -> 2; 128*3 +
    256*2 = 896). Every output slot derives from exactly ONE source row,
    so cross-topic L1 divergences are preserved EXACTLY (sum_k |A-B|/k =
    |A-B|) and marginals are mass-exact. v1 (piecewise-linear quantile
    interpolation along the mean ordering) averaged mean-adjacent rows —
    topic-opposites can be mean-neighbors — and compressed cross-topic
    divergence ~15% (caught by gate b). NOTE this creates marginal-level
    "children" only: tokens still sample i.i.d. from the marginal, no
    within-token sibling structure exists (the rejected parent-split lift
    forced sibling PAIRS per token — that is still not done)."""
    order = np.argsort(-m_src.mean(axis=1), kind="stable")
    k = np.full(G_SRC, 2, dtype=np.int64)
    k[order[:128]] = 3
    assert int(k.sum()) == G_OUT
    m_out = np.repeat(m_src / k[:, None], k, axis=0)
    m_out /= m_out.sum(axis=0, keepdims=True)
    return m_out


def gumbel_topk_rows(weights, n_rows, rng, chunk=20000):
    """[n_rows, TOPK_OUT] int32 rows of DISTINCT ids ~ Plackett-Luce(weights)."""
    logw = np.full(G_OUT, -np.inf, dtype=np.float32)
    nz = weights > 0
    logw[nz] = np.log(weights[nz]).astype(np.float32)
    out = np.empty((n_rows, TOPK_OUT), dtype=np.int32)
    done = 0
    while done < n_rows:
        b = min(chunk, n_rows - done)
        gum = rng.gumbel(size=(b, G_OUT)).astype(np.float32)
        score = logw[None, :] + gum
        idx = np.argpartition(-score, TOPK_OUT, axis=1)[:, :TOPK_OUT]
        srt = np.take_along_axis(score, idx, axis=1).argsort(axis=1)[:, ::-1]
        out[done : done + b] = np.take_along_axis(idx, srt, axis=1)
        done += b
    return out


def calibrate_weights(target, rng, n_probe=200000, iters=8, tol=0.02):
    # tol sits just above the L1 sampling-noise floor at n_probe tokens
    # (~896 * sqrt(1/896 / (n_probe*16)) ~= 0.013); tighter is unreachable
    """Adjust sampling weights so the realized without-replacement marginal
    matches the target (Gumbel-top-k flattens the head). Deterministic given
    the rng. Returns (weights, history)."""
    w = target.copy()
    hist = []
    for it in range(iters):
        rows = gumbel_topk_rows(w, n_probe, rng)
        realized = np.bincount(rows.ravel(), minlength=G_OUT) / float(rows.size)
        l1 = float(np.abs(realized - target).sum())
        hist.append(l1)
        if l1 < tol:
            break
        nz = target > 0
        adj = np.ones(G_OUT)
        adj[nz] = target[nz] / np.maximum(realized[nz], 1e-9)
        w = w * np.clip(adj, 0.5, 2.0)
        w[~nz] = 0.0
        w /= w.sum()
    return w, hist


def write_subject(sdir, per_layer_rows, n_files, prefill_rows, decode_per_file):
    os.makedirs(sdir, exist_ok=True)
    layers = sorted(per_layer_rows)  # K3 labels
    cursors = {ly: 0 for ly in layers}

    def take(ly, n):
        c = cursors[ly]
        cursors[ly] = c + n
        return per_layer_rows[ly][c : c + n]

    for fi in range(n_files):
        toks = []
        pre = {str(ly): [list(map(int, r)) for r in take(ly, prefill_rows)] for ly in layers}
        toks.append(pre)
        dec = {ly: take(ly, decode_per_file) for ly in layers}
        for ti in range(decode_per_file):
            toks.append({str(ly): [list(map(int, dec[ly][ti]))] for ly in layers})
        path = os.path.join(sdir, f"{fi + 1}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(toks, f, separators=(",", ":"))
        os.rename(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces-root",
                    default=os.path.expandvars(fetch_traces.DEFAULT_TRACES_ROOT))
    ap.add_argument("--pool", action="append", default=None,
                    help=f"bench/subject (repeat); default = the 8 EN pools: {DEFAULT_POOLS}")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--files-per-subject", type=int, default=75)
    ap.add_argument("--decode-per-file", type=int, default=400)
    ap.add_argument("--prefill-rows", type=int, default=100)
    ap.add_argument("--marginal-tol", type=float, default=0.015,
                    help="max abs cum-share drift, realized (via the real "
                    "parser) vs target, at each headline percentile")
    ap.add_argument("--divergence-tol", type=float, default=0.05,
                    help="max abs drift of any cross-topic L1 divergence "
                    "entry, synth vs real K2")
    args = ap.parse_args()
    pools = args.pool or DEFAULT_POOLS
    root = args.traces_root
    if "$" in root:
        raise SystemExit(f"unresolved traces root: {root}")
    P = len(pools)
    k2_layers = sorted(LAYER_MAP)

    # ---- fit: per (topic, K2 layer) marginals + K2 provenance ------------
    print(f"[fit] {P} pools x layers {k2_layers} from {SRC_MODEL}", flush=True)
    src_sha = {}
    m_src = {ly: np.zeros((G_SRC, P)) for ly in k2_layers}
    for pi, pool in enumerate(pools):
        sdir = subject_dir(root, SRC_MODEL, pool)
        man = gtr.load_manifest(sdir)
        src_sha[pool] = man["pool_sha"]
        gtr.build_layer_caches(sdir, k2_layers, pool="decode")
        for ly in k2_layers:
            rows = gtr.load_layer_pool(sdir, ly, pool="decode")
            m_src[ly][:, pi] = marginal_from_rows(rows, G_SRC)

    # ---- upscale + identity permutation ---------------------------------
    perm_rng = np.random.default_rng([args.seed, 0xD5])
    perm = perm_rng.permutation(G_OUT)  # hotness-rank -> expert id, shared
    m_out = {}  # per K3 layer: [G_OUT, P] target shares in EXPERT-ID space
    for ly in k2_layers:
        up = hot_split_upscale(m_src[ly])
        tgt = np.zeros_like(up)
        tgt[perm] = up
        m_out[LAYER_MAP[ly]] = tgt

    # ---- calibrate + sample + write -------------------------------------
    n_decode = args.files_per_subject * args.decode_per_file
    n_total = args.files_per_subject * (args.decode_per_file + args.prefill_rows)
    calib_log = {}
    for pi, pool in enumerate(pools):
        bench, _, subject = pool.partition("/")
        sdir = subject_dir(root, OUT_MODEL, pool)
        per_layer_rows = {}
        for ly3 in sorted(m_out):
            target = m_out[ly3][:, pi]
            crng = np.random.default_rng([args.seed, 0xCA, pi, ly3])
            w, hist = calibrate_weights(target, crng)
            calib_log[f"{pool}|{ly3}"] = [round(x, 4) for x in hist]
            srng = np.random.default_rng([args.seed, 0x5A, pi, ly3])
            per_layer_rows[ly3] = gumbel_topk_rows(w, n_total, srng)
        write_subject(sdir, per_layer_rows, args.files_per_subject,
                      args.prefill_rows, args.decode_per_file)
        man = fetch_traces.write_pool_manifest(
            sdir, OUT_MODEL, bench, subject, revision="synthesized")
        man.update(
            method=METHOD,
            source_model=SRC_MODEL,
            source_pool_shas=src_sha,
            source_pools=pools,
            layer_map={str(k): v for k, v in LAYER_MAP.items()},
            seed=args.seed,
            permutation_seed_key=[args.seed, 0xD5],
            topk=TOPK_OUT,
            nexperts=G_OUT,
            calibration_l1={k.split("|")[1]: v for k, v in calib_log.items()
                            if k.startswith(pool + "|")},
            note="rows across the three layers of one token are independent "
                 "draws (no cross-layer correlation is modeled; capsules "
                 "replay one layer at a time)",
        )
        mpath = os.path.join(sdir, "pool.manifest.json")
        with open(mpath + ".tmp", "w") as f:
            json.dump(man, f, indent=2, sort_keys=True)
            f.write("\n")
        os.rename(mpath + ".tmp", mpath)
        print(f"[write] {pool}: {args.files_per_subject} files, "
              f"{n_decode} decode rows/layer, pool_sha {man['pool_sha'][:12]}",
              flush=True)

    # ---- gate (a): realized marginals THROUGH THE REAL PARSER ------------
    print("[gate a] realized-vs-target marginals (via gen_trace_routing)",
          flush=True)
    realized = {ly: np.zeros((G_OUT, P)) for ly in m_out}
    worst = 0.0
    for pi, pool in enumerate(pools):
        sdir = subject_dir(root, OUT_MODEL, pool)
        for ly3 in sorted(m_out):
            rows = gtr.load_layer_pool(sdir, ly3, pool="decode")
            assert len(rows) == n_decode and len(rows[0]) == TOPK_OUT
            r = marginal_from_rows(rows, G_OUT)
            realized[ly3][:, pi] = r
            t, g = curve_stats(m_out[ly3][:, pi], G_OUT), curve_stats(r, G_OUT)
            for p in PCTS:
                d = abs(t[f"top{int(p * 100)}"] - g[f"top{int(p * 100)}"])
                worst = max(worst, d)
                if d > args.marginal_tol:
                    raise SystemExit(
                        f"GATE FAIL marginal {pool} layer {ly3} top"
                        f"{int(p * 100)}%: target {t}, realized {g}")
    print(f"  ok (worst cum-share drift {worst:.4f} <= {args.marginal_tol})")

    # ---- gate (b): cross-topic divergence matrix vs real K2 --------------
    print("[gate b] cross-topic L1 divergence, synth vs real", flush=True)
    worst = 0.0
    for ly in k2_layers:
        ly3 = LAYER_MAP[ly]
        for i in range(P):
            for j in range(i + 1, P):
                d_real = float(np.abs(m_src[ly][:, i] - m_src[ly][:, j]).sum())
                d_syn = float(
                    np.abs(realized[ly3][:, i] - realized[ly3][:, j]).sum())
                worst = max(worst, abs(d_real - d_syn))
                if abs(d_real - d_syn) > args.divergence_tol:
                    raise SystemExit(
                        f"GATE FAIL divergence {pools[i]} vs {pools[j]} layer "
                        f"{ly}->{ly3}: real {d_real:.4f} synth {d_syn:.4f}")
    print(f"  ok (worst divergence drift {worst:.4f} <= {args.divergence_tol})")

    # ---- gate (c): same-seed determinism (regenerate subject 0 rows) -----
    print("[gate c] determinism", flush=True)
    ly3 = sorted(m_out)[0]
    target = m_out[ly3][:, 0]
    crng = np.random.default_rng([args.seed, 0xCA, 0, ly3])
    w, _ = calibrate_weights(target, crng)
    srng = np.random.default_rng([args.seed, 0x5A, 0, ly3])
    again = gumbel_topk_rows(w, n_total, srng)
    sdir = subject_dir(root, OUT_MODEL, pools[0])
    rows = gtr.load_layer_pool(sdir, ly3, pool="decode")
    ref = np.array(rows, dtype=np.int32)
    # decode rows of file f occupy [f*(D+Pr)+Pr, (f+1)*(D+Pr)) of the
    # per-layer sample stream (write_subject consumes prefill first)
    d, pr = args.decode_per_file, args.prefill_rows
    idx = np.concatenate([np.arange(f * (d + pr) + pr, (f + 1) * (d + pr))
                          for f in range(args.files_per_subject)])
    if not np.array_equal(again[idx], ref):
        raise SystemExit("GATE FAIL determinism: regenerated rows differ")
    print("  ok")
    print(f"DONE: {P} synthesized pools under "
          f"{os.path.join(root, fetch_traces.MODEL_PREFIXES[OUT_MODEL])}")


if __name__ == "__main__":
    main()
