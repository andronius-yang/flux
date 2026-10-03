"""Attention part of a decode layer (rope kernel of layer i back to the end of the layer's MoE gate topk / the first
MoE kernel): per-kernel median duration by name, and the span from the layer's input RMSNorm to the last attention
kernel. Layers 1..47 of each forward (layer 0 follows the step boundary)."""
import sqlite3, sys, statistics as st, collections
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
rope = [i for i, k in enumerate(K) if k[2].startswith("BatchQKApplyRotary")]
fw = [rope[i:i + 48] for i in range(0, len(rope) - 47, 48)]
dur = collections.defaultdict(list); span = []; seq = collections.Counter()
for f in fw:
    for j in range(1, 47):
        r = f[j]
        # attention part: from the FusedAddRMSNorm before the rope (input norm of this layer) to the topkGatingSoftmax /
        # first MoE kernel after it: walk back to the previous layer's last MoE kernel is fragile; use a fixed window:
        # 4 kernels before the rope (norm, qkv gemm, q/k norm x2) to the attention output GEMM + post-attention norm
        lo = r - 6; hi = r
        while hi < f[j + 1] and not (K[hi][2].startswith("topkGating") or K[hi][2].startswith("graph_stage") or K[hi][2].startswith("index_")):
            hi += 1
        win = K[lo:hi]
        seq[tuple(k[2][:18] for k in win)] += 1
        for k in win: dur[k[2][:50]].append(k[1] - k[0])
        span.append(K[hi - 1][1] - K[lo][0])
print(db.split('/')[-1], f"layers {len(span)}; attention window span median {st.median(span)/1e3:.1f} us; busy sum by kernel (median us):")
tot = 0
for n, v in sorted(dur.items(), key=lambda kv: -sum(kv[1])):
    m = st.median(v) / 1e3 * len(v) / len(span); tot += m
    print(f"  {m:7.2f}  ({len(v)/len(span):.1f} per layer, median {st.median(v)/1e3:6.2f})  {n}")
print(f"  busy {tot:.1f} us per layer; modal window {seq.most_common(1)[0][0]}")
