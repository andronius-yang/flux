import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
iters = cur.execute("SELECT globalTid, text, start, end FROM NVTX_EVENTS WHERE text LIKE 'iter%' ORDER BY start").fetchall()
timed = [(t>>24, txt, s, e) for t,txt,s,e in iters if not txt.endswith("warmup")]
pids = sorted({p for p,_,_,_ in timed})
KQ = ("SELECT s.value, k.start, k.end, k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId AND r.globalTid>>24=k.globalPid>>24 "
      "JOIN StringIds s ON s.id=k.shortName WHERE k.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY k.start")
MQ = ("SELECT m.copyKind, m.bytes, m.start, m.end, m.streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId AND r.globalTid>>24=m.globalPid>>24 "
      "WHERE m.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY m.start")
want = sys.argv[2] if len(sys.argv)>2 else None
for pid in pids:
    its = [x for x in timed if x[0]==pid]
    _,txt,s,e = its[-1] if want is None else [x for x in its if x[1]==want][0]
    ks = cur.execute(KQ,(pid,s,e)).fetchall(); mc = cur.execute(MQ,(pid,s,e)).fetchall()
    t0 = s
    dev_end = max([k[2] for k in ks]+[m[3] for m in mc])
    print(f"\n== pid {pid} {txt}: host-enqueue {(e-s)/1e6:.2f} ms, device tail {(dev_end-t0)/1e6:.2f} ms, {len(ks)} kernels, {len(mc)} memcpys ==")
    h = collections.defaultdict(lambda: [0,0.0,1e30,0])
    for n,a,b,st in ks:
        r=h[n]; r[0]+=1; r[1]+=(b-a)/1e6; r[2]=min(r[2],a); r[3]=max(r[3],b)
    for n,(c,ms,a,b) in sorted(h.items(), key=lambda x:x[1][2]):
        print(f"  {c:4d}x {ms:8.3f} ms  [{(a-t0)/1e6:7.2f} .. {(b-t0)/1e6:7.2f}]  {n[:80]}")
    hm = collections.defaultdict(lambda: [0,0,0.0,1e30,0])
    for ck,by,a,b,st in mc:
        r=hm[ck]; r[0]+=1; r[1]+=by; r[2]+=(b-a)/1e6; r[3]=min(r[3],a); r[4]=max(r[4],b)
    for ck,(c,by,ms,a,b) in sorted(hm.items()):
        print(f"  memcpy kind {ck}: {c:4d}x {by>>20:6d} MB {ms:8.3f} ms  [{(a-t0)/1e6:7.2f} .. {(b-t0)/1e6:7.2f}]")
    if pid==pids[0]:
        print("  -- ordered inter-node-ish ops (first 12 nvshmem/barrier/p2p):")
        evs = [(a,b,f"K {n[:50]} s{st}") for n,a,b,st in ks if ('nvshmem' in n.lower() or 'barrier' in n.lower() or 'proxy' in n.lower() or 'get' in n.lower())]
        evs += [(a,b,f"M kind{ck} {by>>20}MB s{st}") for ck,by,a,b,st in mc if ck not in (1,2)]
        for a,b,l in sorted(evs)[:14]: print(f"     [{(a-t0)/1e6:7.2f} .. {(b-t0)/1e6:7.2f}] {l}")
