"""54_tile_trace.py -- handoff 54 (plan 10, S0 P1 / P2): GEMM 1 per-tile and bucket-reduce per-lane rings.

  python 54_tile_trace.py tile   <prefix>   # GEMM 1 tail after the last arrival, per receiving rank (median over runs)
  python 54_tile_trace.py reduce <prefix>   # combine receivers: per chain position wait / fold time, tail after the
                                            # last lane signal
<prefix>_r<rank>.npz are written by examples/serving_check.py --trace-out <prefix> (LOPEP_TILE_TRACE=1,
LOPEP_REDUCE_TRACE=1). POS=1: rings of a LOPEP_ROW_ORDER=1 run (segments are delivery positions). Ring layouts: lopep src/dispatch/ths_op/dispatch_gemm.cc (tile: 12 int64 per record) and
src/combine/ths_op/gemm_combine.cc (reduce: 8 int64 per record); row 0 is the header.

Delivery position of a source lane sid at receiving rank r (L ranks per node, NN nodes): node offset
dn = (node(sid) - node(r)) mod NN (dn 0 = own node, pushed over NVLink; dn >= 1 = wire round dn), then the local
offset (lr(sid) - lr(r)) mod L: lopep sort_util.h shift_rank_to_order.
"""
import glob
import os
import statistics as st
import sys

import numpy as np


def load(prefix):
    out = {}
    for f in sorted(glob.glob(prefix + "_r*.npz")):
        d = np.load(f)
        out[int(d["rank"])] = d
    assert out, f"no {prefix}_r*.npz"
    return out


POS = bool(int(os.environ.get("POS", "0")))   # LOPEP_ROW_ORDER=1 rings: the recorded segments are delivery positions


def order_of(sid, rank, L, NN):
    if POS:
        return sid, sid // L
    dn = (sid // L - rank // L) % NN
    return dn * L + (sid % L - rank % L) % L, dn


def tile(prefix):
    rings = load(prefix)
    rows = []
    for rank, d in sorted(rings.items()):
        if "tile" not in d:
            continue
        t = d["tile"]
        W, L = int(d["W"]), int(d["L"])
        NN = W // L
        recs = t[1:]
        recs = recs[recs[:, 0] > 0]
        for run in sorted(set(recs[:, 0].tolist())):
            r = recs[recs[:, 0] == run]
            last_dn = []
            for s0, s1 in zip(r[:, 5], r[:, 6]):
                dns = [order_of(int(s), rank, L, NN)[1] for s in range(int(s0), int(s1) + 1)] if s0 >= 0 else [0]
                last_dn.append(max(dns))
            last_dn = np.array(last_dn)
            te, tf, td = r[:, 9].astype(np.float64), r[:, 10].astype(np.float64), r[:, 11].astype(np.float64)
            t0 = te.min()
            last_fire = tf.max()
            tail = td.max() - last_fire
            late = last_dn == NN - 1
            cta = r[:, 7] >> 16
            late_per_cta = np.bincount(cta[late].astype(np.int64)).max() if late.any() else 0
            rows.append(dict(rank=rank, run=run, layer=int(r[0, 1]), tiles=len(r), gemm_us=(td.max() - t0) / 1e3,
                             fire_last_us=(last_fire - t0) / 1e3, tail_us=tail / 1e3,
                             late_tiles=int(late.sum()), late_frac=float(late.mean()),
                             late_per_cta=int(late_per_cta),
                             fire_after_us=float(np.median((td[late] - tf[late]) / 1e3)) if late.any() else 0.0,
                             tile_us=float(np.median((td - tf) / 1e3))))
    if not rows:
        print("no tile records")
        return
    print(f"{len(rows)} (rank, run) samples; per rank median over runs (us from the first tile entry):")
    print("rank  runs tiles  gemm  last_fire  tail  late_tiles late_frac late/cta  late_tile_us  tile_us")
    for rank in sorted(set(x["rank"] for x in rows)):
        R = [x for x in rows if x["rank"] == rank]
        m = lambda k: st.median(x[k] for x in R)
        print(f"{rank:4d} {len(R):5d} {m('tiles'):5.0f} {m('gemm_us'):6.1f} {m('fire_last_us'):9.1f} {m('tail_us'):6.1f} "
              f"{m('late_tiles'):10.0f} {m('late_frac'):9.2f} {m('late_per_cta'):8.0f} {m('fire_after_us'):12.1f} "
              f"{m('tile_us'):8.1f}")
    allm = lambda k: st.median(x[k] for x in rows)
    print(f"all  tail {allm('tail_us'):.1f} us, late-round tiles {allm('late_frac'):.2f} of the tiles, "
          f"max late tiles per CTA {allm('late_per_cta'):.0f}, gemm {allm('gemm_us'):.1f} us")


def reduce(prefix):
    rings = load(prefix)
    rows = []
    for rank, d in sorted(rings.items()):
        if "reduce" not in d:
            continue
        t = d["reduce"][1:]
        t = t[t[:, 0] > 0]
        for run in sorted(set(t[:, 0].tolist())):
            r = t[t[:, 0] == run]
            t0 = r[:, 5].min()
            per = {}
            for k in sorted(set(r[:, 1].tolist())):
                q = r[r[:, 1] == k]
                per[int(k)] = dict(lane=int(q[0, 2]), blocks=len(q), tokens=int(q[0, 4]),
                                   waited=(q[:, 6].max() - t0) / 1e3, exit=(q[:, 7].max() - t0) / 1e3,
                                   fold=float(np.median((q[:, 7] - q[:, 6]) / 1e3)))
            last_sig = max(v["waited"] for v in per.values())
            end = max(v["exit"] for v in per.values())
            rows.append(dict(rank=rank, run=run, per=per, tail=end - last_sig))
    if not rows:
        print("no reduce records")
        return
    print(f"{len(rows)} (rank, run) samples; tail after the last lane signal, median per rank (us):")
    for rank in sorted(set(x["rank"] for x in rows)):
        R = [x for x in rows if x["rank"] == rank]
        ks = sorted(set(k for x in R for k in x["per"]))
        parts = []
        for k in ks:
            v = [x["per"][k] for x in R if k in x["per"]]
            parts.append(f"c{k}(l{v[0]['lane']}) tok {st.median(z['tokens'] for z in v):.0f} "
                         f"sig {st.median(z['waited'] for z in v):.0f} fold {st.median(z['fold'] for z in v):.1f}")
        print(f"rank {rank:2d} tail {st.median(x['tail'] for x in R):6.1f} | " + " | ".join(parts))


if __name__ == "__main__":
    {"tile": tile, "reduce": reduce}[sys.argv[1]](sys.argv[2])
