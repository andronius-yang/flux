"""49_hostcalls.py <report.sqlite> [--range lopep.step] [--phase NAME ...]: host-call deep dive per phase (handoff 49).

For every instance of an NVTX range (default lopep.step) on its busiest thread, the phases are the intervals between
successive NVTX marks (as in 47_gap_report2.py). Per phase, medians over instances of:
  host us            phase span on the host
  in-API us          host time inside CUDA runtime / driver calls (launch overhead, copies, syncs)
  host-code us       host us - in-API us (Python / C++ work between CUDA calls: metadata, tables, parsing)
  sync us            time inside blocking calls (event / stream / device synchronize, blocking memcpy)
  API calls by name  count per phase
  copies             by direction (H2D / D2H / D2D / P2P) and host memory kind (pageable / pinned), count and bytes
  kernels            name, count per phase, median GPU us
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict

COPY_KIND = {1: "H2D", 2: "D2H", 3: "H2A", 4: "A2H", 5: "A2A", 6: "A2D", 7: "D2A", 8: "D2D", 9: "H2H", 10: "P2P"}
MEM_KIND = {0: "pageable", 1: "pinned", 2: "device", 3: "array", 4: "managed", 5: "device_static", 6: "managed_static",
            7: "unknown"}   # nsys ENUM_CUDA_MEM_KIND
SYNC = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize", "cuStreamSynchronize",
        "cuCtxSynchronize", "cuEventSynchronize")


def med(v):
    return statistics.median(v) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="lopep.step")
    ap.add_argument("--phase", nargs="*", default=[], help="phases to detail (default: all)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    strs = dict(db.execute("select id, value from StringIds"))
    nv = [(s, e, t if t is not None else strs.get(tid), g) for (s, e, t, tid, g) in
          db.execute("select start, end, text, textId, globalTid from NVTX_EVENTS")]
    ranges = [(s, e, n, g) for (s, e, n, g) in nv if e is not None and e > s and n and n.startswith(a.range)]
    by_tid = defaultdict(list)
    for r in ranges:
        by_tid[r[3]].append(r)
    tid = max(by_tid, key=lambda t: len(by_tid[t]))
    inst = sorted(by_tid[tid])
    pid = (tid >> 24) & 0xFFFFFF
    marks = sorted((s, n) for (s, e, n, g) in nv if g == tid and (e is None or e == s) and n and not n.startswith("class "))
    rt = sorted((s, e, c, strs.get(n, str(n))) for (s, e, c, n) in db.execute(
        "select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?", (tid,)))
    kern = defaultdict(list)
    for (s, e, c, n) in db.execute("select start, end, correlationId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
                                   "where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
        kern[c].append((e - s, strs.get(n, str(n))))
    cps = defaultdict(list)
    for (s, e, c, k, sk, dk, b) in db.execute("select start, end, correlationId, copyKind, srcKind, dstKind, bytes from "
                                               "CUPTI_ACTIVITY_KIND_MEMCPY where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
        host_kind = MEM_KIND.get(sk if k in (1, 3) else dk if k in (2, 4) else 2, "?")
        cps[c].append((e - s, COPY_KIND.get(k, str(k)), host_kind, b))
    import bisect
    rs_ = [r[0] for r in rt]
    ms_ = [m[0] for m in marks]
    per = defaultdict(lambda: defaultdict(list))   # phase -> metric -> [per instance]
    order = []
    for (rs, re, _n, _g) in inst:
        i0, i1 = bisect.bisect_left(ms_, rs), bisect.bisect_right(ms_, re)
        bounds = [(rs, "range_start")] + marks[i0:i1]
        for j in range(len(bounds)):
            ps = bounds[j][0]
            pe = bounds[j + 1][0] if j + 1 < len(bounds) else re
            pn = bounds[j + 1][1] if j + 1 < len(bounds) else "tail"
            if pe <= ps:
                continue
            if pn not in order:
                order.append(pn)
            calls = rt[bisect.bisect_left(rs_, ps):bisect.bisect_left(rs_, pe)]
            api_ns = sum(min(ce, pe) - cs for (cs, ce, _c, _n) in calls)
            sync_ns = sum(ce - cs for (cs, ce, _c, n) in calls if n.startswith(SYNC) or
                          (n.startswith("cudaMemcpy") and "Async" not in n))
            d = per[pn]
            d["host"].append((pe - ps) / 1e3)
            d["api"].append(api_ns / 1e3)
            d["code"].append((pe - ps - api_ns) / 1e3)
            d["sync"].append(sync_ns / 1e3)
            cnt = defaultdict(int)
            kc = defaultdict(list)
            cc = defaultdict(lambda: [0, 0])
            for (_cs, _ce, c, n) in calls:
                cnt[n.split("_v")[0]] += 1
                for (dur, kn) in kern.get(c, ()):
                    kc[kn].append(dur / 1e3)
                for (dur, kind, hk, b) in cps.get(c, ()):
                    key = f"{kind}{'/' + hk if kind in ('H2D', 'D2H') else ''}"
                    cc[key][0] += 1
                    cc[key][1] += b
            for n, v in cnt.items():
                d["call:" + n].append(v)
            for n, v in kc.items():
                d["kern:" + n].append((len(v), sum(v)))
            for k, (n, b) in cc.items():
                d["copy:" + k].append((n, b))
    ninst = len(inst)
    print(f"{a.range}: {ninst} instances on thread {tid} (pid {pid})")
    print(f"{'phase':<13}{'host us':>9}{'in-API':>8}{'code':>8}{'sync':>8}")
    for pn in order:
        d = per[pn]
        print(f"{pn:<13}{med(d['host']):>9.1f}{med(d['api']):>8.1f}{med(d['code']):>8.1f}{med(d['sync']):>8.1f}")
    for pn in order:
        if a.phase and pn not in a.phase:
            continue
        d = per[pn]
        n_obs = len(d["host"])
        calls = sorted(((k[5:], sum(v) / n_obs) for k, v in d.items() if k.startswith("call:")), key=lambda x: -x[1])
        kerns = sorted(((k[5:], sum(c for c, _ in v) / n_obs, sum(t for _, t in v) / n_obs) for k, v in d.items()
                        if k.startswith("kern:")), key=lambda x: -x[2])
        copies = sorted(((k[5:], sum(c for c, _ in v) / n_obs, sum(b for _, b in v) / n_obs) for k, v in d.items()
                         if k.startswith("copy:")), key=lambda x: -x[1])
        print(f"\n== {pn}: host {med(d['host']):.1f} us (in API {med(d['api']):.1f}, host code {med(d['code']):.1f}, "
              f"sync {med(d['sync']):.1f}); per instance on average:")
        print("   API calls: " + ", ".join(f"{n} x{c:.1f}" for n, c in calls[:a.top] if c >= 0.05))
        if copies:
            print("   copies:    " + ", ".join(f"{k} x{c:.1f} ({b / c if c else 0:.0f} B each)" for k, c, b in copies))
        for (n, c, t) in kerns[:a.top]:
            if c >= 0.05:
                print(f"   kernel x{c:5.1f} {t:8.1f} us  {n[:100]}")


if __name__ == "__main__":
    main()
