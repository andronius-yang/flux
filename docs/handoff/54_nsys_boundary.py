"""Step-boundary attribution on one device of a serving nsys capture (decode, 48 layers).
Layer marker = the attention rope kernel (one per layer). Boundary window = layer 47's rope -> next forward's layer 0
rope; a normal window = layer i's rope -> layer i+1's. Prints: spans, GPU idle gaps in the boundary window keyed by
(previous kernel -> next kernel), busy time by kernel name in the boundary window minus a normal window, and host
runtime API time on the launching threads inside the window."""
import sqlite3, sys, collections, statistics as st
db, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
NL = 48
c = sqlite3.connect(db)
names = dict(c.execute("select id, value from StringIds"))
K = c.execute("select start, end, shortName, correlationId, globalPid from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start", (dev,)).fetchall()
pid = K[0][4]
K = [(s, e, names[n], cid) for s, e, n, cid, _ in K]
rope = [i for i, k in enumerate(K) if k[2].startswith("BatchQKApplyRotary")]
# decode forwards: runs of NL ropes whose spacing is layer-like; take ropes in groups of NL from the first
fw = [rope[i:i + NL] for i in range(0, len(rope) - NL + 1, NL)]
def busy(i0, i1):
    """union of kernel intervals in [K[i0].start, K[i1].start) (multiple streams overlap)"""
    t0, t1 = K[i0][0], K[i1][0]
    iv = sorted((max(s, t0), min(e, t1)) for s, e, _, _ in K[i0:i1] if s < t1)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s > ce: tot += ce - cs; cs, ce = s, e
        else: ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot
norm_span, norm_busy = [], []
for f in fw:
    for j in range(NL - 1):
        norm_span.append(K[f[j + 1]][0] - K[f[j]][0]); norm_busy.append(busy(f[j], f[j + 1]))
bspan, bbusy, gaps, names_t = [], [], collections.defaultdict(list), collections.defaultdict(list)
win = []
for a, b in zip(fw, fw[1:]):
    i0, i1 = a[-1], b[0]
    if K[i1][0] - K[i0][0] > 20e6:      # an idle period between waves, not a step boundary
        continue
    win.append((i0, i1))
    bspan.append(K[i1][0] - K[i0][0]); bbusy.append(busy(i0, i1))
    end = K[i0][1]
    g = collections.Counter()
    for k in range(i0 + 1, i1 + 1):
        s, e, n, _ = K[k]
        if s - end > 15e3:
            g[(K[k - 1][2][:40], n[:40])] += s - end
        end = max(end, e)
    for key, v in g.items(): gaps[key].append(v)
    t = collections.Counter()
    for k in range(i0, i1): t[K[k][2][:60]] += K[k][1] - K[k][0]
    for key, v in t.items(): names_t[key].append(v)
m = lambda x: st.median(x) / 1e3 if x else 0
print(f"{db.split('/')[-1]} dev {dev}: {len(fw)} forwards, {len(win)} boundary windows")
print(f"normal layer window: span {m(norm_span):.1f} us busy {m(norm_busy):.1f} us | boundary window: span {m(bspan):.1f} "
      f"busy {m(bbusy):.1f} idle {m(bspan) - m(bbusy):.1f} | excess span {m(bspan) - m(norm_span):.1f} us")
print("idle gaps > 15 us in the boundary window (median us over windows where present, presence count):")
for key, v in sorted(gaps.items(), key=lambda kv: -st.median(kv[1]) * len(kv[1]))[:14]:
    print(f"  {st.median(v)/1e3:8.1f} x{len(v):3d}  {key[0]} -> {key[1]}")
print("kernel time by name in the boundary window (median us, top 22):")
for key, v in sorted(names_t.items(), key=lambda kv: -st.median(kv[1]))[:22]:
    print(f"  {st.median(v)/1e3:8.1f} x{len(v):3d}  {key}")
# host API on this process inside the windows
R = c.execute("select r.start, r.end, s.value, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.globalTid/0x1000000=? order by r.start", (pid // 0x1000000,)).fetchall()
api = collections.defaultdict(list)
import bisect
starts = [r[0] for r in R]
for i0, i1 in win:
    t0, t1 = K[i0][0], K[i1][0]
    a = collections.Counter()
    for r in R[bisect.bisect_left(starts, t0 - 5e6):bisect.bisect_left(starts, t1)]:
        s, e = max(r[0], t0), min(r[1], t1)
        if e > s: a[r[2].split('_v')[0]] += e - s
    for key, v in a.items(): api[key].append(v)
print("host runtime API time inside the boundary window (median us, all threads of the process):")
for key, v in sorted(api.items(), key=lambda kv: -st.median(kv[1]))[:12]:
    print(f"  {st.median(v)/1e3:8.1f} x{len(v):3d}  {key}")
