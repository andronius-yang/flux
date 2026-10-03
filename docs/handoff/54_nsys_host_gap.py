"""Host API calls on the launching thread between two kernels' launches (by correlation id), for the boundary:
(a) the AllReduce u64 / reduce_kernel launches, (b) vectorized_gather -> layer-0 RMSNorm launches. Median over windows."""
import sqlite3, sys, collections, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
A, B = sys.argv[3], sys.argv[4]
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
K = [(s, e, names[n], cid) for s, e, n, cid in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
pid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? limit 1", (dev,)).fetchone()[0]
R = c.execute("select r.start, r.end, s.value, r.globalTid, r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.globalTid/0x1000000=? order by r.start", (pid // 0x1000000,)).fetchall()
bycid = {r[4]: r for r in R}
rope = [i for i, k in enumerate(K) if k[2].startswith("BatchQKApplyRotary")]
fw = [rope[i:i + 48] for i in range(0, len(rope) - 47, 48)]
agg = collections.defaultdict(list); spans = []
for a, b in zip(fw, fw[1:]):
    lo, hi = a[-1], b[0]
    ia = [i for i in range(lo, hi) if K[i][2].startswith(A)]
    ib = [i for i in range(lo, hi + 1) if K[i][2].startswith(B)]
    if not ia or not ib: continue
    ra, rb = bycid.get(K[ia[-1]][3]), bycid.get(K[ib[0]][3])
    if not ra or not rb or rb[0] < ra[0]: continue
    tid = ra[3]; spans.append(rb[0] - ra[1])
    cnt = collections.Counter()
    for r in R:
        if r[3] == tid and r[0] >= ra[1] and r[1] <= rb[0]: cnt[r[2].split('_v')[0]] += r[1] - r[0]
    for k, v in cnt.items(): agg[k].append(v)
print(f"{db.split('/')[-1]}: launch of last {A} -> launch of first {B} on the same thread: median {st.median(spans)/1e3:.1f} us over {len(spans)} windows")
for k, v in sorted(agg.items(), key=lambda kv: -st.median(kv[1]))[:12]:
    print(f"  {st.median(v)/1e3:8.1f} x{len(v):3d} {k}")
