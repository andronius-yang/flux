"""53_harness_kernels.py -- handoff 53: per-layer-step GPU breakdown of a serving_check --nsys capture (rank 0).

serving_check --nsys 1 synchronizes before and after every layer-step from step 20 on and wraps it in an NVTX range
"iter", so every GPU activity inside the range's time span belongs to that layer-step. Per layer-step: GPU window
(first activity start -> last activity end), busy (union of activity intervals), idle = window - busy, host range;
per kernel name: count, total and median duration and the median start / end offset from the window start (a
milestone timeline). Several captures side by side: pass several sqlite files.

usage: python 53_harness_kernels.py h_c4b.sqlite h_bare.sqlite [--top 30]
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict


def tables(con):
    return {r[0] for r in con.execute("select name from sqlite_master where type='table'")}


def load(path):
    con = sqlite3.connect(path)
    have = tables(con)
    strings = dict(con.execute("select id, value from StringIds"))
    ranges = []
    for start, end, text, tid, gtid in con.execute("select start, end, text, textId, globalTid from NVTX_EVENTS "
                                                   "where end is not null"):
        name = text if text is not None else strings.get(tid, "")
        if name == "iter":
            ranges.append((start, end, gtid >> 24))  # the process (a node-0 process-tree capture holds every local rank)
    ranges.sort()
    acts = []  # (start, end, name, kind, stream, process)
    for s, e, nid, sid, gp in con.execute("select start, end, coalesce(shortName, demangledName), streamId, globalPid "
                                          "from CUPTI_ACTIVITY_KIND_KERNEL"):
        acts.append((s, e, strings.get(nid, str(nid)), "K", sid, gp >> 24))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in have:
        for s, e, kind, nbytes, sid, gp in con.execute("select start, end, copyKind, bytes, streamId, globalPid "
                                                       "from CUPTI_ACTIVITY_KIND_MEMCPY"):
            acts.append((s, e, f"memcpy kind{kind}", "C", sid, gp >> 24))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in have:
        for s, e, sid, gp in con.execute("select start, end, streamId, globalPid from CUPTI_ACTIVITY_KIND_MEMSET"):
            acts.append((s, e, "memset", "S", sid, gp >> 24))
    acts.sort()
    return ranges, acts


def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyze(path, top, by_start=False):
    ranges, acts = load(path)
    import bisect
    starts = [a[0] for a in acts]
    per = []
    for rs, re_, pid in ranges:
        i = bisect.bisect_left(starts, rs)
        sel = []
        while i < len(acts) and acts[i][0] <= re_:
            if acts[i][5] == pid:
                sel.append(acts[i])
            i += 1
        if not sel:
            continue
        w0 = min(a[0] for a in sel)
        w1 = max(a[1] for a in sel)
        busy = union([(a[0], a[1]) for a in sel])
        per.append((rs, re_, w0, w1, busy, sel))
    if not per:
        print(f"{path}: no iter ranges with GPU work")
        return
    med = lambda v: statistics.median(v) if v else float("nan")
    win = [(p[3] - p[2]) / 1e3 for p in per]
    busy = [p[4] / 1e3 for p in per]
    host = [(p[1] - p[0]) / 1e3 for p in per]
    streams = [len({a[4] for a in p[5]}) for p in per]
    print(f"== {path}: {len(per)} layer-steps; median us: host range {med(host):.0f}, GPU window {med(win):.0f}, "
          f"busy (union) {med(busy):.0f}, idle {med([w - b for w, b in zip(win, busy)]):.0f}, "
          f"activities {med([len(p[5]) for p in per]):.0f}, streams {med(streams):.0f}")
    by = defaultdict(lambda: {"n": [], "dur": [], "s": [], "e": []})
    for p in per:
        w0 = p[2]
        cnt = defaultdict(int)
        dur = defaultdict(int)
        first = {}
        last = {}
        for s, e, name, kind, sid, _ in p[5]:
            cnt[name] += 1
            dur[name] += e - s
            first.setdefault(name, s - w0)
            last[name] = e - w0
        for name in cnt:
            by[name]["n"].append(cnt[name])
            by[name]["dur"].append(dur[name] / 1e3)
            by[name]["s"].append(first[name] / 1e3)
            by[name]["e"].append(last[name] / 1e3)
    rows = sorted(by.items(), key=lambda kv: -med(kv[1]["dur"]) * len(kv[1]["dur"]) / len(per))
    if by_start:
        rows = sorted(rows[:top], key=lambda kv: med(kv[1]["s"]))
    print(f"   {'kernel / activity':58s} {'n':>4s} {'sum_us':>8s} {'first_start':>11s} {'last_end':>9s}  (medians)")
    for name, d in rows[:top]:
        print(f"   {name[:58]:58s} {med(d['n']):4.0f} {med(d['dur']):8.1f} {med(d['s']):11.1f} {med(d['e']):9.1f}"
              f"{'' if len(d['n']) == len(per) else f'  (in {len(d[chr(110)])} of {len(per)})'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", nargs="+")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--by-start", action="store_true", help="the top rows ordered by their median start (timeline)")
    a = ap.parse_args()
    for p in a.sqlite:
        analyze(p, a.top, a.by_start)


if __name__ == "__main__":
    main()
