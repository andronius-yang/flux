"""gap_report2.py <report.sqlite> [--range NAME] [--csv out.csv] [--top N]

Per-phase attribution of an nsys trace (handoff 47). For every instance of an NVTX push/pop range
whose text starts with NAME (default: the most frequent of "zepp.step", "lopep.step", "moe_layer", "iter", "fwd"),
the phases are the intervals between successive NVTX marks emitted on the same thread inside the range
(a phase is named by the mark that closes it; the first phase runs from the range start to the first
mark). Every CUDA runtime call issued on that thread inside a phase interval is joined by
correlationId to its GPU activities (kernels, memcpys, memsets), so a phase owns the GPU work it
launched even when that work ran later. Reported per phase: host span, GPU busy (union), launches,
host-side sync time (event/stream/device synchronize + blocking memcpy), and the GPU time by kernel
class. Per range instance: span, GPU busy union, gap = span - busy, tail = last GPU end - range end.
Medians over instances. The GEMM class matches the CUTLASS kernel's demangled name, not "gemm"."""
import argparse
import csv
import sqlite3
import statistics
from collections import defaultdict

SYNC_CALLS = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cuStreamSynchronize",
              "cuCtxSynchronize", "cudaMemcpy_v", "cudaMemcpy2D_v", "cuMemcpyDtoH_v", "cuMemcpyHtoD_v", "cudaMemcpyToSymbol",
              "cudaMemcpyFromSymbol", "cudaEventQuery", "cuEventQuery")
CLASSES = (
    ("cutlass_gemm", ("cutlass", "Gemm", "gemm", "GemmGrouped")),
    ("triton_moe", ("fused_moe_kernel", "moe_align", "moe_sum_reduce", "silu_and_mul", "gelu_and_mul")),
    ("nccl", ("ncclDevKernel", "ncclKernel")),
    ("nvshmem_barrier", ("barrier_on_stream", "barrier_all", "sync_all")),
    ("nvshmem_proxy", ("nvshmemi_proxy", "proxy_rma", "signal_entrypoint")),
    ("a2av", ("a2av_", "route_", "stable_scatter", "prereduce", "combine_pack", "bucket_reduce")),
    ("memcpy", ("memcpy",)),
    ("memset", ("memset",)),
)


def classify(name):
    for cls, pats in CLASSES:
        if any(p in name for p in pats):
            return cls
    return "other"


def union(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="", help="range text prefix (default: auto)")
    ap.add_argument("--csv", default="", help="per-instance per-phase rows")
    ap.add_argument("--top", type=int, default=8, help="top kernel names per phase")
    ap.add_argument("--by-label", type=int, default=1, help="also summarize per range label (NVTX 'class ...' marks)")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    strs = dict(db.execute("select id, value from StringIds")) if "StringIds" in tabs else {}

    def txt(text, tid):
        return text if text is not None else strs.get(tid)

    nv = [(s, e, txt(t, tid), g) for (s, e, t, tid, g) in
          db.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
    ranges = [(s, e, n, g) for (s, e, n, g) in nv if e is not None and e > s and n]
    marks = [(s, n, g) for (s, e, n, g) in nv if (e is None or e == s) and n]
    prefixes = ["zepp.step", "lopep.step", "moe_layer", "iter", "fwd"]
    if a.range:
        pref = a.range
    else:
        counts = {p: sum(1 for r in ranges if r[2].startswith(p)) for p in prefixes}
        pref = max(counts, key=counts.get)
        if counts[pref] == 0:
            print("no known range found; range texts:", sorted({r[2] for r in ranges})[:20])
            return
    inst = [r for r in ranges if r[2].startswith(pref)]
    by_tid = defaultdict(list)
    for r in inst:
        by_tid[r[3]].append(r)
    tid = max(by_tid, key=lambda t: len(by_tid[t]))
    inst = sorted(by_tid[tid])
    pid = (tid >> 24) & 0xFFFFFF
    # runtime calls on this thread
    rt = [(s, e, c, strs.get(n, str(n))) for (s, e, c, n) in
          db.execute("select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?", (tid,))]
    rt.sort()
    acts = {}
    for tab, namecol in (("CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"), ("CUPTI_ACTIVITY_KIND_MEMCPY", None),
                         ("CUPTI_ACTIVITY_KIND_MEMSET", None)):
        if tab not in tabs:
            continue
        cols = "start, end, streamId, correlationId" + (f", {namecol}" if namecol else "")
        for row in db.execute(f"select {cols} from {tab} where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
            s, e, st, c = row[:4]
            name = strs.get(row[4], str(row[4])) if namecol else tab.split("_")[-1].lower()
            acts.setdefault(c, []).append((s, e, st, name))
    tmarks = sorted((s, n) for (s, n, g) in marks if g == tid and not n.startswith("class "))
    labels = sorted((s, n[len("class "):]) for (s, n, g) in marks if g == tid and n.startswith("class "))
    print(f"process {pid} thread {tid}: {len(inst)} '{pref}' ranges, {len(rt)} runtime calls, "
          f"{sum(len(v) for v in acts.values())} GPU activities, {len(tmarks)} marks")

    per_inst = []      # dict per instance
    phase_rows = []    # (inst idx, phase, host_us, busy_us, launches, sync_us, {cls: us}, {name: us})
    import bisect
    rt_starts = [r[0] for r in rt]
    mk_starts = [m[0] for m in tmarks]
    for k, (rs, re, name, _) in enumerate(inst):
        i0, i1 = bisect.bisect_left(mk_starts, rs), bisect.bisect_right(mk_starts, re)
        bounds = [(rs, "range_start")] + [tmarks[i] for i in range(i0, i1)]
        phases = []
        for j in range(len(bounds)):
            ps = bounds[j][0]
            pe = bounds[j + 1][0] if j + 1 < len(bounds) else re
            pname = bounds[j + 1][1] if j + 1 < len(bounds) else "tail"
            if pe > ps:
                phases.append((pname, ps, pe))
        all_iv, all_n, last_end = [], 0, re
        for (pname, ps, pe) in phases:
            j0, j1 = bisect.bisect_left(rt_starts, ps), bisect.bisect_left(rt_starts, pe)
            calls = rt[j0:j1]
            iv, by_cls, by_name, n_launch, sync_ns = [], defaultdict(int), defaultdict(int), 0, 0
            for (cs, ce, corr, cname) in calls:
                if any(cname.startswith(p) for p in SYNC_CALLS):
                    sync_ns += ce - cs
                for (s, e, st, an) in acts.get(corr, ()):
                    iv.append((s, e))
                    n_launch += 1
                    by_cls[classify(an)] += e - s
                    by_name[an] += e - s
                    last_end = max(last_end, e)
            busy = union(iv)
            all_iv += iv
            all_n += n_launch
            phase_rows.append((k, pname, (pe - ps) / 1e3, busy / 1e3, n_launch, sync_ns / 1e3,
                               {c: v / 1e3 for c, v in by_cls.items()}, {c: v / 1e3 for c, v in by_name.items()}))
        busy_all = union(all_iv)
        lab = [n for (s, n) in labels if rs <= s <= re]
        per_inst.append(dict(span=(re - rs) / 1e3, busy=busy_all / 1e3, gap=(re - rs - busy_all) / 1e3,
                             tail=(last_end - re) / 1e3, launches=all_n, name=name, label=lab[-1] if lab else ""))
    print(f"per instance (median of {len(per_inst)}): span {med([d['span'] for d in per_inst]):.1f} us, "
          f"GPU busy {med([d['busy'] for d in per_inst]):.1f}, gap {med([d['gap'] for d in per_inst]):.1f}, "
          f"GPU tail after range end {med([d['tail'] for d in per_inst]):.1f}, launches {med([d['launches'] for d in per_inst]):.0f}")
    groups = sorted({d["label"] for d in per_inst})
    if a.by_label and len(groups) > 1:
        for g in groups:
            idx = {i for i, d in enumerate(per_inst) if d["label"] == g}
            sub = [d for i, d in enumerate(per_inst) if i in idx]
            print(f"\n[label '{g or '-'}'] {len(sub)} instances: span {med([d['span'] for d in sub]):.1f} us, busy "
                  f"{med([d['busy'] for d in sub]):.1f}, gap {med([d['gap'] for d in sub]):.1f}, launches "
                  f"{med([d['launches'] for d in sub]):.0f}")
            names = []
            for r in phase_rows:
                if r[0] in idx and r[1] not in names:
                    names.append(r[1])
            for pn in names:
                rows = [r for r in phase_rows if r[0] in idx and r[1] == pn]
                clsm = defaultdict(list)
                for r in rows:
                    for c, v in r[6].items():
                        clsm[c].append(v)
                cls_txt = " ".join(f"{c}={med(v):.0f}" for c, v in sorted(clsm.items(), key=lambda kv: -med(kv[1])) if med(v) >= 1)
                print(f"  {pn:<14}{med([r[2] for r in rows]):>10.1f}{med([r[3] for r in rows]):>10.1f}"
                      f"{med([r[4] for r in rows]):>8.0f}{med([r[5] for r in rows]):>9.1f}  {cls_txt}")
    pnames = []
    for r in phase_rows:
        if r[1] not in pnames:
            pnames.append(r[1])
    print(f"{'phase':<14}{'host us':>10}{'gpu busy':>10}{'launch':>8}{'sync us':>9}  classes (median us)")
    for pn in pnames:
        rows = [r for r in phase_rows if r[1] == pn]
        clsm = defaultdict(list)
        for r in rows:
            for c, v in r[6].items():
                clsm[c].append(v)
        cls_txt = " ".join(f"{c}={med(v):.0f}" for c, v in sorted(clsm.items(), key=lambda kv: -med(kv[1])) if med(v) >= 1)
        print(f"{pn:<14}{med([r[2] for r in rows]):>10.1f}{med([r[3] for r in rows]):>10.1f}{med([r[4] for r in rows]):>8.0f}"
              f"{med([r[5] for r in rows]):>9.1f}  {cls_txt}")
    if a.top:
        tot = defaultdict(float)
        for r in phase_rows:
            for n, v in r[7].items():
                tot[n] += v
        print("top kernels by total time (us over all instances):")
        for n, v in sorted(tot.items(), key=lambda kv: -kv[1])[:a.top]:
            print(f"  {v:10.1f}  {n[:110]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["range", "instance", "phase", "host_us", "gpu_busy_us", "launches", "sync_us", "classes"])
            for r in phase_rows:
                w.writerow([per_inst[r[0]]["name"], r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", r[4], f"{r[5]:.1f}",
                            ";".join(f"{c}:{v:.1f}" for c, v in sorted(r[6].items()))])


if __name__ == "__main__":
    main()
