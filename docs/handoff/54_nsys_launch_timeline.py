"""Per-layer timeline around a layer graph's launch (median over decode layers, times relative to the previous graph's
last kernel end): the eager kernels between the two graphs, the host's event-wait return and launch call, the graph's
first kernel."""
import sqlite3, sys, bisect, statistics as st, collections
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
pid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? limit 1", (dev,)).fetchone()[0]
K = [(s, e, cid, names[n]) for s, e, cid, n in c.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,))]
R = c.execute("select r.start, r.end, s.value, r.correlationId, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.globalTid/0x1000000=? order by r.start", (pid // 0x1000000,)).fetchall()
gl = {r[3]: r for r in R if r[2].startswith("cudaGraphLaunch")}
syn = [r for r in R if r[2].startswith("cudaEventSynchronize")]; ss = [r[1] for r in syn]
launch_of = {r[3]: r for r in R}
gidx = [i for i, k in enumerate(K) if k[2] in gl]
first_i = {}
for i in gidx:
    first_i.setdefault(K[i][2], i)
order = sorted(first_i.items(), key=lambda kv: kv[1])
rows = collections.defaultdict(list); seqs = collections.Counter()
for (cp, ip), (cn, inn) in zip(order, order[1:]):
    last = max(j for j in range(ip, inn) if K[j][2] == cp)
    t0 = max(K[j][1] for j in range(ip, last + 1))
    between = [j for j in range(last + 1, inn) if K[j][2] != cp]
    if not between or K[inn][0] - t0 > 3e6: continue
    L = gl[cn]
    j = bisect.bisect_right(ss, L[0]) - 1
    rows["eager first start"].append(K[between[0]][0] - t0)
    rows["eager last end"].append(max(K[b][1] for b in between) - t0)
    rows["eager kernels launched (last launch call end)"].append(max(launch_of[K[b][2]][1] for b in between if K[b][2] in launch_of) - t0)
    if j >= 0: rows["host wait return"].append(syn[j][1] - t0)
    rows["launch call start"].append(L[0] - t0); rows["launch call end"].append(L[1] - t0)
    rows["graph first kernel start"].append(K[inn][0] - t0)
    seqs[len(between)] += 1
print(db.split('/')[-1], f"{len(rows['launch call start'])} layer transitions; eager kernels between graphs (count: n) {dict(seqs.most_common(3))}")
for k, v in rows.items():
    print(f"  {k:48s} median {st.median(v)/1e3:8.1f} us  p10 {sorted(v)[len(v)//10]/1e3:8.1f}  p90 {sorted(v)[9*len(v)//10]/1e3:8.1f}")
