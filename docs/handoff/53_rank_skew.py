"""53_rank_skew.py -- handoff 53: rank skew of the plan-9 layer graph in serving, from one Nsight report per node
(logs/sglang/jobN15.sh: every node's server under nsys, the same capture window).

Per scheduler process (one rank) and per `lopep.step` range: the end of each all-gather, of GEMM 1 / GEMM 2, of the
combine (last bucket-reduce kernel), and the barrier's start / end (the activities launched by the range's API calls,
graph nodes included). Every rank leaves the end-of-layer barrier within microseconds of the others, so the barrier
ends are the common clock: within a node they share the clock; across nodes the instance shift and clock offset are
fitted on the barrier-end sequences. Per aligned instance: wait_r = barrier end - barrier start of rank r; the
slowest rank (smallest wait) gates the layer. Reports: the distribution of the slowest rank / node, the median wait
per rank, and for each segment (all-gather 2 end -> GEMM 1 end -> GEMM 2 end -> combine done -> barrier start) its
duration on the slowest rank vs the median rank.

usage: python 53_rank_skew.py n_..._n0.sqlite n_..._n1.sqlite ... [--skip 4]
"""
import argparse
import bisect
import collections
import sqlite3
import statistics as st


def load(path):
    """Per process (rank): its kernels split into layer-steps at the end-of-layer barrier kernel (one per layer-step
    on every rank; the lopep NVTX ranges exist on rank 0 only)."""
    con = sqlite3.connect(path)
    S = dict(con.execute("select id, value from StringIds"))
    K = collections.defaultdict(list)
    for s, e, n, gp in con.execute("select start, end, coalesce(shortName, demangledName), globalPid "
                                   "from CUPTI_ACTIVITY_KIND_KERNEL"):
        K[gp >> 24].append((s, e, S.get(n, "")))
    out = {}
    for pid, ks in K.items():
        ks.sort()
        bars = [i for i, x in enumerate(ks) if "barrier_on_stream" in x[2]]
        inst = []
        for b0, b1 in zip(bars, bars[1:]):
            seg = ks[b0 + 1:b1 + 1]
            gm = [x for x in seg if x[2] == "Kernel"]
            if len(gm) < 2:
                continue
            ag = [x for x in seg if "AllGather" in x[2] and x[0] < gm[0][0]]
            br = [x for x in seg if "bucket_reduce" in x[2]]
            if len(ag) < 2 or not br:
                continue
            bar = ks[b1]
            inst.append(dict(ag1=ag[-2][1], ag2=ag[-1][1], g1=gm[0][1], g2=gm[1][1], comb=max(x[1] for x in br),
                             bs=bar[0], be=bar[1]))
        if len(inst) >= 50:
            out[(path, pid)] = inst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--skip", type=int, default=4)
    a = ap.parse_args()
    ranks = {}
    for p in a.sqlite:
        ranks.update(load(p))
    keys = sorted(ranks)
    print(f"{len(keys)} ranks: " + ", ".join(f"{k[0].split('_n')[-1].split('.')[0]}:{k[1] % 100000}({len(ranks[k])})"
                                           for k in keys))
    # reference: the first rank; align every other rank by instance shift + clock offset on the barrier ends
    ref = ranks[keys[0]]
    rbe = [x["be"] for x in ref]
    aligned = {keys[0]: (0, 0)}
    for k in keys[1:]:
        be = [x["be"] for x in ranks[k]]
        best = None
        for shift in range(-30, 31):
            d = []
            for i in range(len(rbe)):
                j = i + shift
                if 0 <= j < len(be):
                    d.append(be[j] - rbe[i])
            if len(d) < 20:
                continue
            med = st.median(d)
            spread = st.median([abs(x - med) for x in d])
            if best is None or spread < best[0]:
                best = (spread, shift, med)
        aligned[k] = (best[1], best[2])
        print(f"  rank {k[1] % 100000} node {k[0].split('_n')[-1].split('.')[0]}: shift {best[1]}, offset "
              f"{best[2] / 1e3:.1f} us, residual {best[0] / 1e3:.1f} us")
    slowest = collections.Counter()
    waits = collections.defaultdict(list)
    seg_slow = collections.defaultdict(list)
    seg_med = collections.defaultdict(list)
    n = 0
    lastwait, spread = [], []
    for i in range(a.skip, len(ref)):
        rows = {}
        for k in keys:
            sh, off = aligned[k]
            j = i + sh
            if not (0 <= j < len(ranks[k])):
                break
            x = ranks[k][j]
            rows[k] = {f: (x[f] - off) for f in x}
        if len(rows) != len(keys):
            continue
        n += 1
        w = {k: (r["be"] - r["bs"]) / 1e3 for k, r in rows.items()}
        for k, v in w.items():
            waits[k].append(v)
        ks = min(w, key=w.get)
        slowest[ks] += 1
        lastwait.append(w[ks])
        spread.append(max(w.values()) - w[ks])
        segs = lambda r: {"ag2->g1": (r["g1"] - r["ag2"]) / 1e3, "g1->g2": (r["g2"] - r["g1"]) / 1e3,
                          "g2->comb": (r["comb"] - r["g2"]) / 1e3, "comb->bar": (r["bs"] - r["comb"]) / 1e3,
                          "ag1->ag2": (r["ag2"] - r["ag1"]) / 1e3}
        for s, v in segs(rows[ks]).items():
            seg_slow[s].append(v)
        allseg = [segs(r) for r in rows.values()]
        for s in allseg[0]:
            seg_med[s].append(st.median([x[s] for x in allseg]))
    print(f"aligned instances: {n}")
    print("slowest rank counts: " + ", ".join(f"{k[1] % 100000}@n{k[0].split('_n')[-1].split('.')[0]}:{c}"
                                             for k, c in slowest.most_common(8)))
    print("median barrier wait per rank (us): " + ", ".join(
        f"{k[1] % 100000}@n{k[0].split('_n')[-1].split('.')[0]}:{st.median(v):.0f}" for k, v in sorted(waits.items())))
    q = lambda v, f: sorted(v)[int(len(v) * f)]
    print(f"barrier time of the LAST arriving rank (us): median {st.median(lastwait):.0f} p10 {q(lastwait, .1):.0f} "
          f"p90 {q(lastwait, .9):.0f}; arrival spread (first -> last rank) median {st.median(spread):.0f} "
          f"p90 {q(spread, .9):.0f}")
    print("segments (median us): slowest rank vs median rank")
    for s in seg_slow:
        print(f"  {s:10s} slowest {st.median(seg_slow[s]):7.1f}   median rank {st.median(seg_med[s]):7.1f}")


if __name__ == "__main__":
    main()
