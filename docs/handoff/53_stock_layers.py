# 53_stock_layers.py -- handoff 53: per-layer breakdown of a STOCK SGLang serving capture (python 53_stock_layers.py <sqlite>)
# Stock serving capture (rank 0 process): per decoder layer, delimited by the DP all-gather kernel (one per layer):
# period, the MoE window (all-gather start -> the reduce-scatter's end, or the next kernel class after the experts),
# kernel classes inside the period (busy us), idle; medians over decode-like periods.
import sqlite3, sys, collections, statistics as st
con=sqlite3.connect(sys.argv[1]); S=dict(con.execute("select id,value from StringIds"))
pids=collections.Counter(gp>>24 for (gp,) in con.execute("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL"))
pid=sorted(pids)[0]
K=sorted((s,e,S.get(n,'')) for s,e,n,gp in con.execute("select start,end,coalesce(shortName,demangledName),globalPid from CUPTI_ACTIVITY_KIND_KERNEL") if gp>>24==pid)
K+=[(s,e,'memcpy k%d'%k) for s,e,k,gp in con.execute("select start,end,copyKind,globalPid from CUPTI_ACTIVITY_KIND_MEMCPY") if gp>>24==pid]
K.sort()
names=collections.Counter(n[:40] for s,e,n in K)
print("top kernels:", [(k,v) for k,v in names.most_common(25)])
ag=[i for i,x in enumerate(K) if 'AllGather' in x[2]]
per=[(K[b][0]-K[a][0],a,b) for a,b in zip(ag,ag[1:])]
med=st.median(p for p,a,b in per)
N=[(p,a,b) for p,a,b in per if p<1.4*med]
print("%d all-gathers, median period %.0f us, %d decode-like periods"%(len(ag),med/1e3,len(N)))
cls=collections.defaultdict(list); idle=[]; rs=[]
for p,a,b in N:
    seg=K[a:b]; cur=seg[0][0]; idl=0; c=collections.Counter()
    for s,e,n in seg:
        if s>cur: idl+=s-cur
        c[n[:40]]+=max(0,e-max(s,cur)); cur=max(cur,e)
    idle.append(idl/1e3)
    for k,v in c.items(): cls[k].append(v/1e3)
    r=[x for x in seg if 'ReduceScatter' in x[2] or 'reduce_scatter' in x[2].lower()]
    if r: rs.append((r[-1][1]-seg[0][0])/1e3)
print("idle per period median %.0f us; all-gather start -> last reduce-scatter end median %s us"%(st.median(idle), "%.0f"%st.median(rs) if rs else "n/a"))
for k,v in sorted(cls.items(), key=lambda kv:-sum(kv[1]))[:22]:
    print("   %-40s %7.1f us/period (in %d%% of periods)"%(k,sum(v)/len(N),100*len(v)//len(N)))
