"""Attention part by position: the 10 kernels from the layer's input norm (rope index - 6) to the o_proj splitK
(rope + 3), median duration and idle-before per position, layers 1..46 of every forward."""
import sqlite3, sys, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
rope = [i for i, k in enumerate(K) if k[2].startswith("BatchQKApplyRotary")]
fw = [rope[i:i + 48] for i in range(0, len(rope) - 47, 48)]
rows = [[] for _ in range(10)]; span = []
for f in fw:
    for j in range(1, 47):
        r = f[j]; w = K[r - 6:r + 4]
        if not w[0][2].startswith("FusedAddRMSNorm"): continue
        for p, k in enumerate(w):
            rows[p].append((k[1] - k[0], k[0] - w[p - 1][1] if p else 0, k[2][:34]))
        span.append(w[-1][1] - w[0][0])
print(db.split('/')[-1], f"{len(span)} layers; span input-norm start -> o_proj end median {st.median(span)/1e3:.1f} us")
for p in range(10):
    print(f"  {rows[p][0][2]:34s} dur {st.median(x[0] for x in rows[p])/1e3:6.2f}  idle-before {st.median(x[1] for x in rows[p])/1e3:6.2f}")
