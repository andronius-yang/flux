"""53_serving_segments.py -- handoff 53: per-layer segments and the combine tail of a serving Nsight capture (rank 0 process).

  python 53_serving_segments.py seg  <capture.sqlite>   # period, pre-MoE, MoE start -> GEMM 1, GEMMs, combine tail, end sync
  python 53_serving_segments.py tail <capture.sqlite>   # every activity from GEMM 2 to the end sync, offsets from GEMM 2 end
Layer periods are delimited by the end-of-layer barrier / sync kernels; decode-like periods = < 1.4x the median, or
BAND=lo_us,hi_us (e.g. a capture that also holds prefill steps).
"""
import sys

def seg():
    import sqlite3, sys, collections, statistics as st
    con=sqlite3.connect(sys.argv[1]); S=dict(con.execute("select id,value from StringIds"))
    pids=collections.Counter(gp>>24 for (gp,) in con.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL"))
    pid=sorted(pids)[0]
    K=sorted((s,e,S.get(n,''),gx) for s,e,n,gx,gp in con.execute("select start,end,coalesce(shortName,demangledName),gridX,globalPid from CUPTI_ACTIVITY_KIND_KERNEL") if gp>>24==pid)
    bars=[i for i,x in enumerate(K) if ('barrier_on_stream' in x[2] or 'sync_on_stream' in x[2])]
    rows=[]
    for b0,b1 in zip(bars,bars[1:]):
        seg=K[b0+1:b1+1]; t0=K[b0][1]
        st0=next((s for s,e,n,g in seg if n.startswith('graph_stage_in') or n.startswith('swap_decide') or 'AllGather' in n or n.startswith('step_head')),None)
        gm=[(s,e) for s,e,n,g in seg if n=='Kernel' and g>=64]
        if st0 is None or len(gm)<2: continue
        bs,be=K[b1][0],K[b1][1]
        rows.append(dict(period=(be-t0)/1e3, pre=(st0-t0)/1e3, to_g1=(gm[0][0]-st0)/1e3, g1=(gm[0][1]-gm[0][0])/1e3,
                         g1g2=(gm[1][0]-gm[0][1])/1e3, g2=(gm[1][1]-gm[1][0])/1e3, tail=(bs-gm[1][1])/1e3, bar=(be-bs)/1e3,
                         moe=(be-st0)/1e3))
    med=st.median(r['period'] for r in rows)
    import os
    band = os.environ.get("BAND")
    lo, hi = (float(v) for v in band.split(",")) if band else (0.0, 1.4 * med)
    R=[r for r in rows if lo <= r['period'] < hi]
    print("%s: %d layer periods (%d decode-like)"%(sys.argv[1].split('/')[-1],len(rows),len(R)))
    for k in ['period','pre','moe','to_g1','g1','g1g2','g2','tail','bar']:
        v=sorted(r[k] for r in R); print("  %-6s median %7.1f  p10 %7.1f  p90 %7.1f us"%(k,st.median(v),v[len(v)//10],v[9*len(v)//10]))

def tail():
    import sqlite3, sys, collections, statistics as st
    con=sqlite3.connect(sys.argv[1]); S=dict(con.execute("select id,value from StringIds"))
    pids=collections.Counter(gp>>24 for (gp,) in con.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL"))
    pid=sorted(pids)[0]
    K=[(s,e,S.get(n,''),gx) for s,e,n,gx,gp in con.execute("select start,end,coalesce(shortName,demangledName),gridX,globalPid from CUPTI_ACTIVITY_KIND_KERNEL") if gp>>24==pid]
    K+=[(s,e,'memcpy k%d'%k,b) for s,e,k,b,gp in con.execute("select start,end,copyKind,bytes,globalPid from CUPTI_ACTIVITY_KIND_MEMCPY") if gp>>24==pid]
    K.sort()
    bars=[i for i,x in enumerate(K) if ('barrier_on_stream' in x[2] or 'sync_on_stream' in x[2])]
    agg=collections.defaultdict(lambda: [[],[],[],0,[]]); n=0; tails=[]
    per=[K[b1][1]-K[b0][1] for b0,b1 in zip(bars,bars[1:])]; med=st.median(per)
    import os
    band = os.environ.get("BAND")
    lo, hi = (float(v) * 1e3 for v in band.split(",")) if band else (0.0, 1.4 * med)
    for (b0,b1),p in zip(zip(bars,bars[1:]),per):
        if not (lo <= p < hi): continue
        seg=K[b0+1:b1+1]
        gm=[(s,e) for s,e,nm,g in seg if nm=='Kernel' and g>=64]
        if len(gm)<2: continue
        g2s,g2e=gm[1]; n+=1; tails.append((K[b1][0]-g2e)/1e3)
        seen=collections.Counter()
        for s,e,nm,g in seg:
            if e<g2s: continue
            key=nm[:44]
            a=agg[key]; a[0].append((s-g2e)/1e3); a[1].append((e-g2e)/1e3); seen[key]+=1
            a[4].append(g if nm.startswith('memcpy') else 0)
        for k,c in seen.items(): agg[k][3]+=c
    print("%s: %d periods, tail (GEMM2 end -> barrier start) median %.0f us"%(sys.argv[1].split('/')[-1],n,st.median(tails)))
    rows=sorted(agg.items(), key=lambda kv: st.median(kv[1][0]))
    for k,a in rows:
        extra=" bytes %d"%st.median(a[4]) if k.startswith('memcpy') else ""
        print("  %-44s x%5.2f  start %8.1f  end %8.1f%s"%(k,a[3]/n,st.median(a[0]),st.median(a[1]),extra))

if __name__ == "__main__":
    mode = sys.argv.pop(1)
    seg() if mode == "seg" else tail()
