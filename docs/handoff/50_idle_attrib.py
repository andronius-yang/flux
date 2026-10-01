"""50_idle_attrib.py <report.sqlite> [--range lopep.step] [--pid P] [--top N]: where does a layer-step's time go
when the GPU has nothing running? (handoff 50, plan 7 round 8)

For one process (default: the one with the most `lopep.step` NVTX ranges; --pid to pick another) and its device:
  * GPU busy = union of its kernels, memcpys and memsets; a layer-step's GPU window runs from the first GPU activity
    launched inside the step's host range to the last one; the idle gaps inside that window are attributed to what
    the process's main thread (the range's thread) was doing at that time:
      sync:<call>     the host sits in a synchronizing call (event / stream / device sync, blocking copy, query):
                      the GPU is idle because its streams wait on something not shown as an activity (a stream wait on
                      a peer's signal, or a host-issued op still to come) -> remote / ordering stall
      api:<call>      inside another CUDA runtime / driver call (launch / copy issue overhead)
      host:<phase>    between runtime calls: host code (Python / C++), attributed to the lopep ledger phase (NVTX mark
                      interval) and, when Python sampling is present, to the most frequent innermost lopep / sglang frame
  * GPU busy time is split by kernel class (gemm incl. arrival spin, nccl, nvshmem barrier, nvshmem proxy puts, a2av
    planning / pack / reduce kernels, memcpy, other).
Medians over layer-step instances are printed per category; --csv writes per-instance rows.
"""
import argparse
import bisect
import csv
import sqlite3
import statistics
from collections import Counter, defaultdict

SYNC = ("Synchronize", "cudaMemcpy_v", "cudaMemcpy2D_v", "cuMemcpyDtoH", "cuMemcpyHtoD", "Query", "cudaMemcpyFromSymbol")
CLASSES = (
    ("gemm", ("cutlass", "Gemm", "gemm", "GemmGrouped")),
    ("nccl", ("ncclDevKernel", "ncclKernel")),
    ("nvshmem_barrier", ("barrier_on_stream", "barrier_all", "sync_all")),
    ("nvshmem_proxy_put", ("nvshmemi_proxy", "proxy_rma", "signal_entrypoint")),
    ("a2av_plan", ("a2av_meta", "stable_scatter", "a2av_demands", "route_", "swap_decide", "lane_", "pad_rebuild",
                   "a2av_stage1", "consumer_build", "a2av_pack_scan", "gating_cumsum", "compress_plan", "combine_plan",
                   "combine_tables", "a2av_meta_arena", "prepare_workspace", "bucket_map")),
    ("a2av_move", ("prereduce", "combine_pack", "bucket_reduce", "a2av_combine_reduce", "index_select",
                   "indexSelect", "gather")),
    ("memcpy", ("memcpy", "Memcpy")),
    ("memset", ("memset", "Memset")),
)


def classify(name):
    for cls, pats in CLASSES:
        if any(p in name for p in pats):
            return cls
    return "other"


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def med(v):
    return statistics.median(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="lopep.step")
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--csv", default="")
    ap.add_argument("--skip", type=int, default=2, help="drop the first N instances (capture start)")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strs = dict(db.execute("select id, value from StringIds")) if "StringIds" in tabs else {}
    txt = lambda t, i: t if t is not None else strs.get(i)
    nv = [(s, e, txt(t, i), g) for (s, e, t, i, g) in db.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
    ranges = [r for r in nv if r[1] is not None and r[1] > r[0] and r[2] and r[2].startswith(a.range)]
    by_tid = defaultdict(list)
    for r in ranges:
        by_tid[r[3]].append(r)
    if not by_tid:
        print("no", a.range, "ranges")
        return
    cands = sorted(by_tid, key=lambda t: -len(by_tid[t]))
    tid = next((t for t in cands if a.pid == 0 or ((t >> 24) & 0xFFFFFF) == a.pid), cands[0])
    pid = (tid >> 24) & 0xFFFFFF
    inst = sorted(by_tid[tid])[a.skip:]
    marks = sorted((s, n) for (s, e, n, g) in nv if g == tid and n and (e is None or e == s) and not n.startswith("class "))
    mk_s = [m[0] for m in marks]
    # main-thread runtime calls
    rt = sorted((s, e, strs.get(n, str(n)), c) for (s, e, c, n) in db.execute(
        "select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?", (tid,)))
    rt_s = [r[0] for r in rt]
    # device activities of this process
    acts, by_corr = [], {}
    for tab, ncol in (("CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"), ("CUPTI_ACTIVITY_KIND_MEMCPY", None),
                      ("CUPTI_ACTIVITY_KIND_MEMSET", None)):
        if tab not in tabs:
            continue
        cols = "start, end, correlationId" + (f", {ncol}" if ncol else "")
        for row in db.execute(f"select {cols} from {tab} where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
            name = strs.get(row[3], str(row[3])) if ncol else tab.split("_")[-1].lower()
            acts.append((row[0], row[1], name, row[2]))
            by_corr.setdefault(row[2], []).append((row[0], row[1]))
    acts.sort()
    act_s = [x[0] for x in acts]
    busy_all = merge([(s, e) for (s, e, _, _) in acts])
    busy_s = [b[0] for b in busy_all]
    # python samples (if present): innermost lopep / sglang frame per sample time on this thread
    py = []
    for t in ("PYTHON_SAMPLES", "PYTHON_CALLCHAINS"):
        pass
    if "SAMPLING_CALLCHAINS" in tabs and "COMPOSITE_EVENTS" in tabs:
        try:
            q = ("select ce.start, sc.symbol, sc.stackDepth from COMPOSITE_EVENTS ce join SAMPLING_CALLCHAINS sc "
                 "on ce.id = sc.id where ce.globalTid = ? order by ce.start, sc.stackDepth")
            cur_t, best = None, None
            for st, sym, depth in db.execute(q, (tid,)):
                name = strs.get(sym, str(sym))
                if st != cur_t:
                    if cur_t is not None and best:
                        py.append((cur_t, best))
                    cur_t, best = st, None
                if best is None and any(k in name for k in ("lopep", "sglang", "serving.py", "overlap.py", "planner.py",
                                                            "layer.py", "dispatch_gemm", "gemm_combine", "a2av_")):
                    best = name
            if cur_t is not None and best:
                py.append((cur_t, best))
        except sqlite3.Error as ex:
            print("python/cpu sampling join failed:", ex)
    py_s = [p[0] for p in py]
    print(f"pid {pid} tid {tid}: {len(inst)} '{a.range}' instances, {len(rt)} main-thread runtime calls, {len(acts)} GPU "
          f"activities, {len(marks)} marks, {len(py)} attributed CPU samples")

    rows = []
    cat_us = defaultdict(list)
    cls_us = defaultdict(list)
    py_top = Counter()
    for k, (hs, he, _n, _) in enumerate(inst):
        # GPU window: activities launched by runtime calls inside the host range
        i0, i1 = bisect.bisect_left(rt_s, hs), bisect.bisect_right(rt_s, he)
        launched = [iv for r in rt[i0:i1] for iv in by_corr.get(r[3], [])]
        if not launched:
            continue
        gs, ge = min(x[0] for x in launched), max(x[1] for x in launched)
        # busy pieces inside [gs, ge] and idle gaps
        j0 = max(bisect.bisect_right(busy_s, gs) - 1, 0)
        busy = []
        for s, e in busy_all[j0:]:
            if s >= ge:
                break
            if e > gs:
                busy.append((max(s, gs), min(e, ge)))
        gaps, cur = [], gs
        for s, e in busy:
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < ge:
            gaps.append((cur, ge))
        cat = defaultdict(float)
        for g0, g1 in gaps:
            # walk the gap: inside runtime calls -> sync/api, else host code by phase
            t = g0
            r = max(bisect.bisect_right(rt_s, t) - 1, 0)
            while t < g1:
                while r < len(rt) and rt[r][1] <= t:
                    r += 1
                if r < len(rt) and rt[r][0] <= t < rt[r][1]:
                    end = min(rt[r][1], g1)
                    nm = rt[r][2].split("_v")[0]
                    key = ("sync:" if any(x in nm for x in SYNC) else "api:") + nm
                    cat[key] += end - t
                    t = end
                else:
                    nxt = rt[r][0] if r < len(rt) else g1
                    end = min(max(nxt, t + 1), g1)
                    mi = bisect.bisect_right(mk_s, t) - 1
                    phase = marks[mi][1] if mi >= 0 and marks[mi][0] >= hs else "pre_step"
                    cat["host:after_" + phase] += end - t
                    if py:
                        pi = bisect.bisect_left(py_s, t)
                        while pi < len(py) and py[pi][0] < end:
                            py_top[(phase, py[pi][1])] += 1
                            pi += 1
                    t = end
        cls = defaultdict(float)
        for s, e, name, _c in acts[bisect.bisect_left(act_s, gs):bisect.bisect_left(act_s, ge)]:
            cls[classify(name)] += min(e, ge) - s
        win = ge - gs
        idle = sum(g1 - g0 for g0, g1 in gaps)
        rows.append(dict(inst=k, host_us=(he - hs) / 1e3, gpu_window_us=win / 1e3, gpu_idle_us=idle / 1e3,
                         **{f"idle_{c}": v / 1e3 for c, v in cat.items()}, **{f"busy_{c}": v / 1e3 for c, v in cls.items()}))
        for c, v in cat.items():
            cat_us[c].append(v / 1e3)
        for c, v in cls.items():
            cls_us[c].append(v / 1e3)
    n = len(rows)
    print(f"\n{n} layer-steps: host range median {med([r['host_us'] for r in rows]):.0f} us, GPU window "
          f"{med([r['gpu_window_us'] for r in rows]):.0f} us, GPU idle inside it {med([r['gpu_idle_us'] for r in rows]):.0f} us")
    print("\nGPU idle by what the main thread was doing (mean over all steps = sum/n; median of steps where present):")
    for c, v in sorted(cat_us.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {c:48s} mean {sum(v) / n:8.1f} us   median-when-present {med(v):8.1f}   in {len(v)}/{n} steps")
    print("\nGPU busy by kernel class (summed activity time, may exceed wall time where streams overlap):")
    for c, v in sorted(cls_us.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {c:20s} mean {sum(v) / n:8.1f} us")
    if py_top:
        print("\nCPU samples during GPU-idle host code (phase, innermost lopep/sglang frame):")
        for (ph, fr), c in py_top.most_common(a.top):
            print(f"  {c:6d}  after_{ph:14s} {fr[:110]}")
    if a.csv:
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
