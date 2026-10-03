"""Exposed layer-graph launch: per cudaGraphLaunch of the device's process, the GPU idle between the last kernel
before the graph's first kernel and that first kernel; plus the host timeline: wait return -> launch call start,
launch call duration. Median / p10 / p90 over all launches (decode)."""
import sqlite3, sys, bisect, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = sqlite3.connect(db)
pid = c.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? limit 1", (dev,)).fetchone()[0]
K = c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,)).fetchall()
first = {}
for i, (s, e, cid) in enumerate(K):
    if cid not in first: first[cid] = i
R = c.execute("select r.start, r.end, s.value, r.correlationId, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.globalTid/0x1000000=? order by r.start", (pid // 0x1000000,)).fetchall()
gl = [r for r in R if r[2].startswith("cudaGraphLaunch")]
syn = [r for r in R if r[2].startswith("cudaEventSynchronize")]
ss = [r[1] for r in syn]
idle, wait_to_call, call, call_end_to_first = [], [], [], []
for s, e, n, cid, tid in gl:
    i = first.get(cid)
    if i is None or i == 0: continue
    prev_end = max(k[1] for k in K[max(0, i - 8):i])
    idle.append(max(0, K[i][0] - prev_end))
    call.append(e - s)
    call_end_to_first.append(K[i][0] - e)
    j = bisect.bisect_right(ss, s) - 1
    if j >= 0 and syn[j][4] == tid and s - syn[j][1] < 5e6: wait_to_call.append(s - syn[j][1])
q = lambda x, p: sorted(x)[int(p * (len(x) - 1))] / 1e3
f = lambda x: f"median {st.median(x)/1e3:7.1f}  p10 {q(x,.1):7.1f}  p90 {q(x,.9):7.1f} us (n={len(x)})"
print(db.split('/')[-1])
print("  GPU idle before each graph's first kernel :", f(idle))
print("  cudaGraphLaunch call duration            :", f(call))
print("  launch call end -> graph's first kernel   :", f(call_end_to_first))
if wait_to_call: print("  event wait return -> launch call start   :", f(wait_to_call))
print(f"  exposed idle summed per forward (48 layers, median x 48): {st.median(idle)*48/1e6:.2f} ms")
