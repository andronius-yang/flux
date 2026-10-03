"""Host per-layer timeline on the launching thread, relative to the END of graph i-1's cudaGraphLaunch call: first and
last eager launch of layer i, the event wait (start / return), graph i's launch call start. Median over decode layers."""
import sqlite3, sys, bisect, statistics as st, collections
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
pid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? limit 1", (dev,)).fetchone()[0]
R = c.execute("select r.start, r.end, s.value, r.correlationId, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.globalTid/0x1000000=? order by r.start", (pid // 0x1000000,)).fetchall()
kc = set(cid for (cid,) in c.execute("select correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?", (dev,)))
gl = [r for r in R if r[2].startswith("cudaGraphLaunch")]
tid = collections.Counter(r[4] for r in gl).most_common(1)[0][0]
T = [r for r in R if r[4] == tid]
st_ = [r[0] for r in T]
rows = collections.defaultdict(list); cat = collections.defaultdict(list)
for a, b in zip(gl, gl[1:]):
    if b[0] - a[1] > 3e6: continue
    t0 = a[1]
    seg = T[bisect.bisect_right(st_, a[1]):bisect.bisect_left(st_, b[0])]
    launches = [r for r in seg if r[3] in kc]
    waits = [r for r in seg if r[2].startswith("cudaEventSynchronize")]
    if launches:
        rows["first eager launch start"].append(launches[0][0] - t0)
        rows["last eager launch end"].append(launches[-1][1] - t0)
    if waits:
        rows["last event wait start"].append(waits[-1][0] - t0); rows["last event wait return"].append(waits[-1][1] - t0)
    rows["next graph launch call start"].append(b[0] - t0)
    api = sum(r[1] - r[0] for r in seg)
    rows["CUDA API time in the gap (all calls)"].append(api)
    w = sum(r[1] - r[0] for r in waits)
    rows["  of which event waits"].append(w)
    rows["non-API host time in the gap (python etc.)"].append((b[0] - t0) - api)
print(db.split('/')[-1], f"{len(rows['next graph launch call start'])} gaps between consecutive graph launch calls (host clock)")
for k, v in rows.items():
    print(f"  {k:46s} median {st.median(v)/1e3:8.1f} us  p10 {sorted(v)[len(v)//10]/1e3:8.1f}  p90 {sorted(v)[9*len(v)//10]/1e3:8.1f}")
