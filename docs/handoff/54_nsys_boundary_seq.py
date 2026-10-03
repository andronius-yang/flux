"""Kernel sequence of one boundary (lm head GEMM -> the next forward's layer-0 rope), median offsets over windows."""
import sqlite3, sys, collections, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
K = [(s, e, names[n]) for s, e, n in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
rope = [i for i, k in enumerate(K) if k[2].startswith("BatchQKApplyRotary")]
fw = [rope[i:i + 48] for i in range(0, len(rope) - 47, 48)]
seqs = []
for a, b in zip(fw, fw[1:]):
    i1 = b[0]
    lm = max(i for i in range(a[-1], i1) if K[i][2].startswith("ampere_bf16_s16816gemm_bf16_128x256"))
    if K[i1][0] - K[lm][0] > 20e6: continue
    t0 = K[lm][0]; seq = []; end = K[lm][0]
    for k in range(lm, i1 + 1):
        s, e, n = K[k]
        seq.append((n[:44], (s - t0) / 1e3, (e - s) / 1e3, max(0, s - end) / 1e3)); end = max(end, e)
    seqs.append(seq)
L = collections.Counter(len(s) for s in seqs).most_common(1)[0][0]
use = [s for s in seqs if len(s) == L]
print(f"{db.split('/')[-1]}: {len(seqs)} windows, {len(use)} with the modal {L} kernels; median us: start, dur, idle-before")
for j in range(L):
    print(f"  {use[0][j][0]:44s} {st.median(s[j][1] for s in use):8.1f} {st.median(s[j][2] for s in use):7.1f} {st.median(s[j][3] for s in use):7.1f}")
