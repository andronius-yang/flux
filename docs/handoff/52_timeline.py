"""52_timeline.py <report.sqlite> [--pid P | --all-pids] [--skip N] [--csv PATH]: per-layer-step timeline metrics
T1-T8 of the lopep MoE layer in an SGLang serving capture (handoff 52, plan 8).

Input: an Nsight Systems SQLite export. A layer-step instance is an NVTX push/pop range `lopep.step` on the
scheduler's main thread (the thread holding the ranges); the ledger's NVTX marks inside it name the host phases.
Per process (default: the one with the most `lopep.step` ranges) the first --skip instances are dropped and every
metric is reported as median / p10 / p90 over the remaining instances (p = linear interpolation between ranks).

Step activity set: the GPU activities (kernels, memcpys, memsets) launched by main-thread runtime / driver calls
inside the step's host range, plus those launched by any other thread of the process (a wire proxy) between the
step's range start and the next step's range start. GPU window = first start .. last end of that set.

T1  planning end -> first wire copy.
    planning end = end of the last plan-PRODUCER kernel of the step that starts before the dispatch GEMM
    (`cutlass::Kernel<...agscatter...>`; in a step without one, before the step's first put kernel).
    Producers = the kernels whose output the host round trip / the wire consume: a2av_meta* (incl. the device
    arena), a2av_dispatch_plan, a2av_combine_plan_block, a2av_msplit_tables, a2av_combine_tables, a2av_demands.
    first wire = earliest step activity starting at or after the planning end that is a memcpy of kind
    Peer-to-Peer or Device-to-Device (any size; incl. cudaMemcpyBatchAsync copies), an NVSHMEM put / signal kernel
    (name contains nvshmem / putmem / proxy / signal) or a lopep copy kernel (LOPEP_COPY).
    Variants: T1p ends at the first PAYLOAD wire (memcpy >= --payload-bytes, rma put kernel, lopep copy kernel);
    T1L uses the literal planning set (every a2av planning kernel that starts before the dispatch GEMM, PLAN_ALL).
T2  GPU idle inside the GPU window, attributed (as in 50_idle_attrib.py) to what the main thread was doing:
    sync:<call> / api:<call> / host:after_<ledger phase>.
T3  host time blocked per step on the main thread (synchronize calls, synchronous memcpy calls, async D2H copies
    into pageable memory), polls (event / stream query) separately; host lead = GPU start of an activity minus the
    start of the runtime call that launched it (for the step's first activity, the dispatch GEMM, the first wire
    copy, the combine GEMM); host range duration vs GPU window.
T4  per step, count and API time of runtime / driver calls by group, main thread vs other threads of the process.
T5  wire-proxy latency (only if a non-main thread emits `lopep.proxy*` NVTX events): planning end -> the proxy's
    first API call of the step, -> the first wire copy, -> the first proxy-issued wire copy.
T6  R1 tripwire: module / library load and function-lookup calls longer than --lookup-us after the end of the
    `lopep.warmup` range (or after the end of the first `lopep.step` if there is none); plus kernels whose FIRST
    launch comes after that point with a launch call longer than --lookup-us (a lazy load hides inside the launch).
T7  R5 report: every kernel that STARTS while a spinner kernel (the lopep GEMMs, a2av_combine_pack,
    a2av_combine_prereduce, wait_geq) is resident on the same device, grouped by (kernel, spinner) with grid, block,
    registers per thread, static + dynamic shared memory; flagged when block threads x registers > 32768 or shared
    memory > 100 KB per block.
T8  sequential phases of the GPU window: [start -> planning end] [planning end -> GEMM1 start] [GEMM1]
    [GEMM1 end -> GEMM2 start] [GEMM2] [GEMM2 end -> last activity]; the floor list ranks these (with
    [planning end -> GEMM1 start] split at the first wire copy) by median, with the GPU idle inside each segment,
    its attribution and its busiest kernel classes.

Plain Python 3.6+, sqlite3, statistics; imports helpers from 50_idle_attrib.py in the same directory.
"""
import argparse
import bisect
import csv
import heapq
import importlib.util
import math
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict, namedtuple

sys.dont_write_bytecode = True      # do not leave a __pycache__ next to the handoff files
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("idle_attrib50", os.path.join(_HERE, "50_idle_attrib.py"))
ia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ia)

# ---------------------------------------------------------------------------------------------------- name sets
PRODUCER = ("a2av_meta", "a2av_dispatch_plan", "a2av_combine_plan_block", "a2av_msplit_tables", "a2av_combine_tables",
            "a2av_demands")
PLAN_ALL = PRODUCER + ("a2av_stage1", "consumer_build", "gating_cumsum", "swap_decide", "lane_", "prepare_workspace",
                       "compress_plan_scan")
PUT = ("nvshmem", "putmem", "proxy", "signal")
PAYLOAD_PUT = ("putmem", "proxy_rma", "rma_")
LOPEP_COPY = ("copy_continous_aligned", "a2av_wire_copy", "a2av_relay_copy", "a2av_desc_copy")
SPINNERS = (("dispatch GEMM", lambda n: "cutlass::Kernel<" in n and "agscatter" in n),
            ("combine GEMM", lambda n: "cutlass::Kernel<" in n and "gatherrs" in n),
            ("flux GEMM", lambda n: "cutlass::Kernel<flux" in n),
            ("combine_pack", lambda n: "a2av_combine_pack_kernel" in n),
            ("combine_prereduce", lambda n: "a2av_combine_prereduce_kernel" in n),
            ("wait_geq", lambda n: "wait_geq_kernel" in n))
LOOKUP = ("ModuleLoad", "LibraryLoad", "LibraryGetKernel", "LibraryGetModule", "GetFuncBySymbol", "ModuleGetFunction",
          "KernelGetFunction", "FuncGetAttributes", "LazyLoad")
REG_LIMIT, SMEM_LIMIT = 32768, 100 * 1024
MEMCPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
WIRE_KINDS = (8, 10)
SEGS = ("start->plan_end", "plan_end->first_wire (T1)", "first_wire->GEMM1", "GEMM1 (dispatch)", "GEMM1->GEMM2",
        "GEMM2 (combine)", "GEMM2->last")
T8_PHASES = (("start->plan_end", "P1_start_to_plan_end", (0,)), ("plan_end->GEMM1 start", "P2_plan_end_to_gemm1", (1, 2)),
             ("GEMM1 (dispatch)", "P3_gemm1", (3,)), ("GEMM1 end->GEMM2 start", "P4_gemm1_to_gemm2", (4,)),
             ("GEMM2 (combine)", "P5_gemm2", (5,)), ("GEMM2 end->last activity", "P6_gemm2_to_last", (6,)))
T4_GROUPS = ("launch", "event record", "stream wait event", "wait/write value", "memcpy sync", "memcpy async",
             "memcpy batch", "memset", "synchronize", "query", "other")

Call = namedtuple("Call", "start end corr name tid drv")
Act = namedtuple("Act", "start end kind name corr stream dev bytes ckind skind dkind regs grid block ssm dsm")


def is_gemm1(n):
    return "cutlass::Kernel<" in n and "agscatter" in n


def is_gemm2(n):
    return "cutlass::Kernel<" in n and "gatherrs" in n


def has(n, pats):
    return any(p in n for p in pats)


def act_class(a):
    """Kernel class for busy-time tables: 50_idle_attrib's classes, with the two lopep GEMMs, the planning kernels
    and the wire / host copies told apart (50's 'gemm' pattern also matches the workspace kernels)."""
    if a.kind == "memcpy":
        return "memcpy_wire" if a.ckind in WIRE_KINDS else "memcpy_host"
    if a.kind == "memset":
        return "memset"
    if is_gemm1(a.name):
        return "gemm1"
    if is_gemm2(a.name):
        return "gemm2"
    if has(a.name, PLAN_ALL) or "make_workspace" in a.name:
        return "a2av_plan"
    return ia.classify(a.name)


def t4_group(name):
    if "LaunchKernel" in name or "LaunchCooperative" in name or "GraphLaunch" in name or name.startswith("cuLaunch"):
        return "launch"
    if "EventRecord" in name:
        return "event record"
    if "StreamWaitEvent" in name:
        return "stream wait event"
    if "WaitValue" in name or "WriteValue" in name or "BatchMemOp" in name:
        return "wait/write value"
    if "MemcpyBatch" in name:
        return "memcpy batch"
    if "Memcpy" in name:
        return "memcpy async" if "Async" in name else "memcpy sync"
    if "Memset" in name:
        return "memset"
    if "Synchronize" in name:
        return "synchronize"
    if "Query" in name:
        return "query"
    return "other"


def sync_kind(name):
    """T3: 'sync' for synchronize calls, 'memcpy' for synchronous copy calls, 'query' for polls, else None."""
    if "Synchronize" in name:
        return "sync"
    if "Query" in name:
        return "query"
    base = name.split("_v")[0]
    if "Memcpy" in base and "Async" not in base and "Batch" not in base:
        return "memcpy"
    return None


def short(n, width=90):
    """Demangled kernel name without the leading 'void ' and the argument list (template-depth aware)."""
    s = n[5:] if n.startswith("void ") else n
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0 and i > 0:
            s = s[:i]
            break
    return s if len(s) <= width else s[:width - 3] + "..."


def pct(v, q):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    x = q * (len(v) - 1)
    i = int(x)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (x - i)


def st3(v):
    v = [x for x in v if x is not None and x == x]
    if not v:
        return "        n/a"
    return "%8.1f  [p10 %8.1f  p90 %8.1f]" % (pct(v, 0.5), pct(v, 0.1), pct(v, 0.9))


def st3c(v):
    """Compact 'median (p10-p90)' for table cells."""
    v = [x for x in v if x is not None and x == x]
    if not v:
        return "n/a"
    return "%.1f (%.1f-%.1f)" % (pct(v, 0.5), pct(v, 0.1), pct(v, 0.9))


def mean(v):
    return sum(v) / len(v) if v else 0.0


# ---------------------------------------------------------------------------------------------------- loading
class Capture:
    """Process-independent tables: strings, NVTX, thread names, processes."""

    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        self.tabs = {r[0] for r in self.db.execute("select name from sqlite_master where type='table'")}
        self.strs = dict(self.db.execute("select id, value from StringIds")) if "StringIds" in self.tabs else {}
        cols = {r[1] for r in self.db.execute("pragma table_info(NVTX_EVENTS)")}
        et = "eventType" if "eventType" in cols else "0"
        self.ranges, self.marks = defaultdict(list), defaultdict(list)   # globalTid -> [(start, end, text)]
        for s, e, t, i, g, typ in self.db.execute(
                "select start, end, text, textId, globalTid, %s from NVTX_EVENTS" % et):
            n = t if t is not None else self.strs.get(i)
            if not n or g is None:
                continue
            if e is not None and e > s and typ in (0, 59, 60):
                self.ranges[g].append((s, e, n))
            elif (e is None or e == s) and typ in (0, 34):
                self.marks[g].append((s, n))
        self.tname = {}
        if "ThreadNames" in self.tabs:
            for nid, _p, g in self.db.execute("select nameId, priority, globalTid from ThreadNames"):
                self.tname[g] = self.strs.get(nid, str(nid))
        self.pname = {}
        if "PROCESSES" in self.tabs:
            for pid, name in self.db.execute("select pid, name from PROCESSES"):
                self.pname[pid] = name

    def step_threads(self, rng):
        """{pid: (main tid, sorted step ranges)} for every process holding `rng` ranges (thread with the most)."""
        out = {}
        for g, rr in self.ranges.items():
            k = [r for r in rr if r[2].startswith(rng)]
            if not k:
                continue
            pid = (g >> 24) & 0xFFFFFF
            if pid not in out or len(k) > len(out[pid][1]):
                out[pid] = (g, sorted(k))
        return out

    def gpu_pids(self):
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in self.tabs:
            return []
        return sorted(r[0] for r in self.db.execute(
            "select distinct (globalPid >> 24) & 0xFFFFFF from CUPTI_ACTIVITY_KIND_KERNEL"))

    def load_process(self, pid):
        """API calls of every thread of the process (runtime + driver) and its GPU activities."""
        S = self.strs
        calls = []
        for tab in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if tab not in self.tabs:
                continue
            cols = {r[1] for r in self.db.execute("pragma table_info(%s)" % tab)}
            ec = "eventClass" if "eventClass" in cols else ("1" if tab.endswith("DRIVER") else "0")
            for s, e, c, n, g, cl in self.db.execute(
                    "select start, end, correlationId, nameId, globalTid, %s from %s "
                    "where (globalTid >> 24) & 0xFFFFFF = ?" % (ec, tab), (pid,)):
                calls.append(Call(s, e, c, S.get(n, str(n)), g, cl == 1 or tab.endswith("DRIVER")))
        calls.sort()
        acts = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tabs:
            for (s, e, n, c, st, dev, reg, gx, gy, gz, bx, by, bz, ssm, dsm) in self.db.execute(
                    "select start, end, demangledName, correlationId, streamId, deviceId, registersPerThread, gridX, "
                    "gridY, gridZ, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory "
                    "from CUPTI_ACTIVITY_KIND_KERNEL where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
                acts.append(Act(s, e, "kernel", S.get(n, str(n)), c, st, dev, 0, 0, 0, 0, reg or 0, gx * gy * gz,
                                bx * by * bz, ssm or 0, dsm or 0))
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in self.tabs:
            for s, e, b, ck, c, st, sk, dk, dev in self.db.execute(
                    "select start, end, bytes, copyKind, correlationId, streamId, srcKind, dstKind, deviceId "
                    "from CUPTI_ACTIVITY_KIND_MEMCPY where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
                acts.append(Act(s, e, "memcpy", "memcpy " + MEMCPY_KIND.get(ck, str(ck)), c, st, dev, b, ck, sk, dk,
                                0, 0, 0, 0, 0))
        if "CUPTI_ACTIVITY_KIND_MEMSET" in self.tabs:
            for s, e, b, c, st, dev in self.db.execute(
                    "select start, end, bytes, correlationId, streamId, deviceId "
                    "from CUPTI_ACTIVITY_KIND_MEMSET where (globalPid >> 24) & 0xFFFFFF = ?", (pid,)):
                acts.append(Act(s, e, "memset", "memset", c, st, dev, b, 0, 0, 0, 0, 0, 0, 0, 0))
        acts.sort()
        return calls, acts


# ---------------------------------------------------------------------------------------------------- per step
def attribute_gap(g0, g1, mcalls, mcall_s, marks, mk_s, hs):
    """Split an idle gap [g0, g1) into pieces labelled by what the main thread was doing (50_idle_attrib logic):
    inside a call -> sync:<call> (synchronize / blocking copy / query) or api:<call>; between calls -> host code,
    named by the last ledger mark of this step at or before the piece start."""
    pieces = []
    t = g0
    r = max(bisect.bisect_right(mcall_s, t) - 1, 0)
    while t < g1:
        while r < len(mcalls) and mcalls[r].end <= t:
            r += 1
        if r < len(mcalls) and mcalls[r].start <= t < mcalls[r].end:
            end = min(mcalls[r].end, g1)
            full = mcalls[r].name
            nm = full.split("_v")[0]
            pieces.append((t, end, ("sync:" if any(x in full for x in ia.SYNC) else "api:") + nm))
            t = end
        else:
            nxt = mcalls[r].start if r < len(mcalls) else g1
            end = min(max(nxt, t + 1), g1)
            mi = bisect.bisect_right(mk_s, t) - 1
            phase = marks[mi][1] if mi >= 0 and marks[mi][0] >= hs else "pre_step"
            pieces.append((t, end, "host:after_" + phase))
            t = end
    return pieces


def first_wire(L, t0, payload_bytes, proxy_tids=None, call_of=None):
    """Earliest step activity starting at or after t0 that is a wire copy (see module doc). payload_bytes > 0 keeps
    only payload copies; proxy_tids keeps only activities launched from those threads."""
    best = None
    for a in L:
        if a.start < t0:
            continue
        if best is not None and a.start >= best.start:
            break
        if proxy_tids is not None:
            c = call_of.get(a.corr)
            if c is None or c.tid not in proxy_tids:
                continue
        if a.kind == "memcpy":
            ok = a.ckind in WIRE_KINDS and (payload_bytes <= 0 or a.bytes >= payload_bytes)
        elif a.kind == "kernel":
            ok = has(a.name, LOPEP_COPY) or (has(a.name, PAYLOAD_PUT) if payload_bytes > 0 else has(a.name, PUT))
        else:
            ok = False
        if ok:
            best = a
    return best


def wire_desc(a, call_of):
    if a is None:
        return "none"
    c = call_of.get(a.corr)
    api = c.name.split("_v")[0] if c else "?"
    if a.kind == "memcpy":
        size = "%d B" % a.bytes if a.bytes < 4096 else ">= 4 KB"
        return "%s %s (%s)" % (a.name, size, api)
    return "kernel %s" % short(a.name, 60)


def analyze(cap, pid, tid, steps_all, calls, acts, a):
    """Per-instance rows plus aggregated tables for one process."""
    call_of = {c.corr: c for c in calls}
    mcalls = [c for c in calls if c.tid == tid]
    mcall_s = [c.start for c in mcalls]
    ocalls = [c for c in calls if c.tid != tid]
    ocall_s = [c.start for c in ocalls]
    by_corr = defaultdict(list)
    for x in acts:
        by_corr[x.corr].append(x)
    act_s = [x.start for x in acts]
    busy_all = ia.merge([(x.start, x.end) for x in acts])
    busy_s = [b[0] for b in busy_all]
    marks_all = sorted(cap.marks.get(tid, []))
    marks = [m for m in marks_all if not m[1].startswith("class ")]
    mk_s = [m[0] for m in marks]
    labels = [m for m in marks_all if m[1].startswith("class ")]
    lab_s = [m[0] for m in labels]
    # wire proxy: non-main threads of this process emitting lopep.proxy NVTX events
    proxy_tids = set()
    for g in set(cap.ranges) | set(cap.marks):
        if g != tid and ((g >> 24) & 0xFFFFFF) == pid:
            if any(r[2].startswith("lopep.proxy") for r in cap.ranges.get(g, ())) or \
                    any(m[1].startswith("lopep.proxy") for m in cap.marks.get(g, ())):
                proxy_tids.add(g)
    proxy_calls = sorted(c for c in ocalls if c.tid in proxy_tids)
    proxy_s = [c.start for c in proxy_calls]
    proxy_rng = sorted((s, e) for g in proxy_tids for (s, e, n) in cap.ranges.get(g, ()) if n.startswith("lopep.proxy"))

    rows, cat_us, cls_us = [], defaultdict(list), defaultdict(list)
    seg_idle_cat = [defaultdict(float) for _ in SEGS]
    seg_cls = [defaultdict(float) for _ in SEGS]
    seg_n = 0
    t4_main = {g: [[], []] for g in T4_GROUPS}         # group -> [counts per step, us per step]
    t4_oth = {g: [[], []] for g in T4_GROUPS}
    t4_main_drv = Counter()
    t4_oth_thr = Counter()
    t4_other_names = Counter()
    plan_end_names, wire_names, wirep_names = Counter(), Counter(), Counter()
    items = defaultdict(float)
    n_noplan = n_nogemm = 0
    starts_all = [r[0] for r in steps_all]
    for k, (hs, he, _n) in enumerate(steps_all):
        if k < a.skip:
            continue
        nxt = starts_all[k + 1] if k + 1 < len(starts_all) else he
        i0, i1 = bisect.bisect_left(mcall_s, hs), bisect.bisect_right(mcall_s, he)
        own_calls = mcalls[i0:i1]
        j0, j1 = bisect.bisect_left(ocall_s, hs), bisect.bisect_left(ocall_s, max(nxt, he))
        oth_calls = ocalls[j0:j1]
        L = [x for c in own_calls for x in by_corr.get(c.corr, ())] + \
            [x for c in oth_calls for x in by_corr.get(c.corr, ())]
        if not L:
            continue
        L.sort()
        gs, ge = L[0].start, max(x.end for x in L)
        li = bisect.bisect_right(lab_s, he) - 1
        label = labels[li][1][len("class "):] if li >= 0 and labels[li][0] >= hs else ""
        if a.label and not label.startswith(a.label):
            continue
        row = dict(pid=pid, inst=k, host_start_ns=hs, label=label, host_us=(he - hs) / 1e3,
                   gpu_window_us=(ge - gs) / 1e3, gpu_start_minus_host_start_us=(gs - hs) / 1e3,
                   gpu_end_minus_host_end_us=(ge - he) / 1e3)
        # -- landmarks
        g1 = [x for x in L if x.kind == "kernel" and is_gemm1(x.name)]
        g2 = [x for x in L if x.kind == "kernel" and is_gemm2(x.name)]
        g1s = g1[0].start if g1 else None
        g1e = max(x.end for x in g1) if g1 else None
        g2s = g2[0].start if g2 else None
        g2e = max(x.end for x in g2) if g2 else None
        if g1s is not None:
            bound = g1s
        else:
            puts = [x.start for x in L if x.kind == "kernel" and has(x.name, PUT)]
            bound = min(puts) if puts else ge
        prod = [x for x in L if x.kind == "kernel" and x.start < bound and has(x.name, PRODUCER)]
        plan_end = max(x.end for x in prod) if prod else None
        litp = [x for x in L if x.kind == "kernel" and x.start < bound and has(x.name, PLAN_ALL)]
        plan_end_lit = max(x.end for x in litp) if litp else None
        fw = first_wire(L, plan_end, 0) if plan_end is not None else None
        fwp = first_wire(L, plan_end, a.payload_bytes) if plan_end is not None else None
        fwl = first_wire(L, plan_end_lit, 0) if plan_end_lit is not None else None
        if plan_end is None:
            n_noplan += 1
        else:
            plan_end_names[short(max(prod, key=lambda x: x.end).name, 60)] += 1
            wire_names[wire_desc(fw, call_of)] += 1
            wirep_names[wire_desc(fwp, call_of)] += 1
        row["T1_us"] = (fw.start - plan_end) / 1e3 if fw is not None else None
        row["T1p_payload_us"] = (fwp.start - plan_end) / 1e3 if fwp is not None else None
        row["T1L_literal_us"] = (fwl.start - plan_end_lit) / 1e3 if fwl is not None else None
        row["plan_end_rel_us"] = (plan_end - gs) / 1e3 if plan_end is not None else None
        # -- T2: idle gaps in the window, attributed
        jb = max(bisect.bisect_right(busy_s, gs) - 1, 0)
        gaps, cur = [], gs
        for s, e in busy_all[jb:]:
            if s >= ge:
                break
            if e <= gs:
                continue
            s, e = max(s, gs), min(e, ge)
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < ge:
            gaps.append((cur, ge))
        pieces = []
        for g0, gg1 in gaps:
            pieces += attribute_gap(g0, gg1, mcalls, mcall_s, marks, mk_s, hs)
        cat = defaultdict(float)
        for p0, p1, c in pieces:
            cat[c] += p1 - p0
        idle = sum(g[1] - g[0] for g in gaps)
        row["gpu_idle_us"] = idle / 1e3
        for top in ("sync", "api", "host"):
            row["idle_%s_us" % top] = sum(v for c, v in cat.items() if c.startswith(top + ":")) / 1e3
        for c, v in cat.items():
            row["idle_" + c] = v / 1e3
            cat_us[c].append(v / 1e3)
        win_acts = acts[bisect.bisect_left(act_s, gs):bisect.bisect_left(act_s, ge)]
        cls = defaultdict(float)
        for x in win_acts:
            cls[act_class(x)] += min(x.end, ge) - x.start
        for c, v in cls.items():
            cls_us[c].append(v / 1e3)
        for x in L:
            if x.kind == "kernel":
                items[short(x.name, 70)] += (x.end - x.start) / 1e3
            elif x.kind == "memcpy" and x.ckind in WIRE_KINDS:
                items["memcpy %s (wire)" % MEMCPY_KIND[x.ckind]] += (x.end - x.start) / 1e3
        # -- T3: blocked host time, leads
        blk = Counter()
        for c in own_calls:
            sk = sync_kind(c.name)
            if sk:
                blk[sk] += c.end - c.start
                blk[sk + "_n"] += 1
            elif "Memcpy" in c.name and "Async" in c.name and any(
                    x.kind == "memcpy" and x.ckind == 2 and x.dkind == 0 for x in by_corr.get(c.corr, ())):
                blk["d2h_pageable"] += c.end - c.start
                blk["d2h_pageable_n"] += 1
        row["T3_blocked_us"] = (blk["sync"] + blk["memcpy"] + blk["d2h_pageable"]) / 1e3
        row["T3_sync_calls_us"] = blk["sync"] / 1e3
        row["T3_sync_memcpy_us"] = blk["memcpy"] / 1e3
        row["T3_d2h_pageable_us"] = blk["d2h_pageable"] / 1e3
        row["T3_query_us"] = blk["query"] / 1e3
        row["T3_query_n"] = blk["query_n"]
        row["T3_blocked_n"] = blk["sync_n"] + blk["memcpy_n"] + blk["d2h_pageable_n"]

        def lead(x):
            c = call_of.get(x.corr) if x is not None else None
            return (x.start - c.start) / 1e3 if c is not None else None
        row["T3_lead_first_us"] = lead(L[0])
        row["T3_lead_gemm1_us"] = lead(g1[0]) if g1 else None
        row["T3_lead_first_wire_us"] = lead(fw)
        row["T3_lead_gemm2_us"] = lead(g2[0]) if g2 else None
        # -- T4: API calls by group
        cm, co = Counter(), Counter()
        for c in own_calls:
            gname = t4_group(c.name)
            cm[gname] += 1
            cm[gname + "_ns"] += c.end - c.start
            if c.drv:
                t4_main_drv[gname] += 1
            if gname == "other":
                t4_other_names[c.name.split("_v")[0]] += 1
        for c in oth_calls:
            gname = t4_group(c.name)
            co[gname] += 1
            co[gname + "_ns"] += c.end - c.start
            t4_oth_thr[cap.tname.get(c.tid, str(c.tid & 0xFFFFFF))] += 1
        for gname in T4_GROUPS:
            t4_main[gname][0].append(cm[gname])
            t4_main[gname][1].append(cm[gname + "_ns"] / 1e3)
            t4_oth[gname][0].append(co[gname])
            t4_oth[gname][1].append(co[gname + "_ns"] / 1e3)
            key = gname.replace(" ", "_").replace("/", "_")
            row["T4_main_%s_n" % key] = cm[gname]
            row["T4_main_%s_us" % key] = cm[gname + "_ns"] / 1e3
            row["T4_other_%s_n" % key] = co[gname]
            row["T4_other_%s_us" % key] = co[gname + "_ns"] / 1e3
        # -- T5: proxy
        if proxy_tids and plan_end is not None:
            pi = bisect.bisect_left(proxy_s, plan_end)
            pc = proxy_calls[pi] if pi < len(proxy_calls) and proxy_calls[pi].start < max(nxt, he) else None
            row["T5_proxy_first_api_us"] = (pc.start - plan_end) / 1e3 if pc else None
            row["T5_first_wire_us"] = row["T1_us"]
            fpx = first_wire(L, plan_end, 0, proxy_tids, call_of)
            row["T5_first_proxy_wire_us"] = (fpx.start - plan_end) / 1e3 if fpx is not None else None
            ri = bisect.bisect_left(proxy_rng, (hs, 0))
            pr = [r for r in proxy_rng[ri:] if r[0] < max(nxt, he)]
            row["T5_proxy_ranges_n"] = len(pr)
            row["T5_proxy_ranges_us"] = sum(e - s for s, e in pr) / 1e3
        # -- T8: sequential segments
        if g1s is None or g2s is None:
            n_nogemm += 1
        elif plan_end is not None:
            b = [gs, plan_end, fw.start if fw is not None and fw.start < g1s else g1s, g1s, g1e, g2s, g2e, ge]
            for i in range(1, len(b)):
                b[i] = min(max(b[i], b[i - 1]), ge)
            seg_n += 1
            for i, nm in enumerate(SEGS):
                row["seg%d_us" % i] = (b[i + 1] - b[i]) / 1e3
            idle_seg = [0.0] * len(SEGS)
            for p0, p1, c in pieces:
                for i in range(len(SEGS)):
                    o = min(p1, b[i + 1]) - max(p0, b[i])
                    if o > 0:
                        idle_seg[i] += o
                        seg_idle_cat[i][c] += o / 1e3
            for i in range(len(SEGS)):
                row["seg%d_idle_us" % i] = idle_seg[i] / 1e3
            for x in win_acts:
                for i in range(len(SEGS)):
                    o = min(x.end, b[i + 1]) - max(x.start, b[i])
                    if o > 0:
                        seg_cls[i][act_class(x)] += o / 1e3
            for _nm, key, idx in T8_PHASES:
                row["T8_" + key + "_us"] = sum(row["seg%d_us" % i] for i in idx)
        rows.append(row)
    agg = dict(cat_us=cat_us, cls_us=cls_us, seg_idle_cat=seg_idle_cat, seg_cls=seg_cls, seg_n=seg_n,
               t4_main=t4_main, t4_oth=t4_oth, t4_main_drv=t4_main_drv, t4_oth_thr=t4_oth_thr, t4_other_names=t4_other_names,
               plan_end_names=plan_end_names, wire_names=wire_names, wirep_names=wirep_names, items=items,
               n_noplan=n_noplan, n_nogemm=n_nogemm, proxy_tids=proxy_tids, mcalls=len(mcalls))
    return rows, agg


# ---------------------------------------------------------------------------------------------------- T6 / T7
def where(t, steps_all, marks, mk_s):
    """'step k, after mark m' for a host time t (k = index in the full range list), or '' outside the steps."""
    if not steps_all:
        return ""
    k = bisect.bisect_right([r[0] for r in steps_all], t) - 1
    if k < 0 or t > steps_all[k][1]:
        return "between steps"
    if not marks:
        return "ref. process step %d" % k
    mi = bisect.bisect_right(mk_s, t) - 1
    m = marks[mi][1] if mi >= 0 and marks[mi][0] >= steps_all[k][0] else "step start"
    return "step %d, after mark '%s'" % (k, m)


def tripwire(cap, pid, steps_all, calls, acts, thr_us, marks=()):
    out = []
    marks = [m for m in marks if not m[1].startswith("class ")]
    mk_s = [m[0] for m in marks]
    warm = [r for g, rr in cap.ranges.items() if ((g >> 24) & 0xFFFFFF) == pid for r in rr
            if r[2].startswith("lopep.warmup")]
    if warm:
        cutoff, how = max(r[1] for r in warm), "end of the lopep.warmup range (%d ranges)" % len(warm)
    elif steps_all:
        cutoff, how = steps_all[0][1], "end of the first lopep.step range (no lopep.warmup range in this process)"
    else:
        cutoff, how = min((x.start for x in acts), default=0), "start of the process's GPU activity (no ranges)"
    t_ref = steps_all[0][0] if steps_all else cutoff
    out.append("  cutoff: %s" % how)
    hits, per = [], defaultdict(lambda: [0, 0, 0])
    for c in calls:
        if c.start < cutoff or not has(c.name, LOOKUP):
            continue
        d = c.end - c.start
        p = per[c.name.split("_v")[0]]
        p[0] += 1
        p[1] = max(p[1], d)
        if d > thr_us * 1e3:
            p[2] += 1
            hits.append(c)
    for nm, (n, mx, nb) in sorted(per.items()):
        out.append("  %-34s %7d calls after the cutoff, max %8.1f us, %d above %.0f us" % (nm, n, mx / 1e3, nb, thr_us))
    if not per:
        out.append("  no module / library load or function-lookup calls traced after the cutoff")
    out.append("  calls above %.0f us: %d%s" % (thr_us, len(hits), "  <-- R1 VIOLATION" if hits else ""))
    for c in hits[:40]:
        out.append("    t=%+10.3f ms (rel. first step)  %-22s %-36s %9.1f us  %s" % (
            (c.start - t_ref) / 1e6, cap.tname.get(c.tid, str(c.tid & 0xFFFFFF))[:22], c.name[:36],
            (c.end - c.start) / 1e3, where(c.start, steps_all, marks, mk_s)))
    # first launches after the cutoff
    call_of = {c.corr: c for c in calls}
    first = {}
    for x in acts:
        if x.kind == "kernel" and x.name not in first:
            first[x.name] = x
    late = []
    for nm, x in first.items():
        c = call_of.get(x.corr)
        t = c.start if c else x.start
        if t >= cutoff:
            late.append((t, nm, (c.end - c.start) / 1e3 if c else float("nan"), c))
    slow = [r for r in late if r[2] > thr_us]
    nslow = sum(1 for x in acts if x.kind == "kernel" and x.corr in call_of and call_of[x.corr].start >= cutoff
                and call_of[x.corr].end - call_of[x.corr].start > thr_us * 1e3)
    out.append("  kernel launch calls above %.0f us after the cutoff: %d (any launch; host jitter included)" % (
        thr_us, nslow))
    out.append("  kernels first launched after the cutoff: %d, with a launch call above %.0f us: %d%s" % (
        len(late), thr_us, len(slow), "  <-- first launch slow: lazy load suspect" if slow else ""))
    for t, nm, d, c in sorted(slow)[:40]:
        out.append("    t=%+10.3f ms  launch %8.1f us  %-14s %-38s %s" % (
            (t - t_ref) / 1e6, d, (cap.tname.get(c.tid, "?") if c else "?")[:14],
            where(t, steps_all, marks, mk_s)[:38], short(nm, 110)))
    return out


def r5_report(acts, top):
    out = []
    kern = [x for x in acts if x.kind == "kernel"]
    spin = []
    for x in kern:
        for lab, f in SPINNERS:
            if f(x.name):
                spin.append((x.start, x.end, lab, x))
                break
    spin.sort(key=lambda r: r[0])
    sres = defaultdict(Counter)
    for s, e, lab, x in spin:
        sres[lab][(x.grid, x.block, x.regs, x.ssm + x.dsm)] += 1
    out.append("  spinner blocks (grid, block threads, regs/thread, smem B): " + "; ".join(
        "%s %s x%d" % (lab, k, n) for lab, c in sres.items() for k, n in c.most_common(2)))
    agg = {}
    active = []          # heap of (end, idx)
    si = 0
    for x in kern:
        while si < len(spin) and spin[si][0] <= x.start:
            heapq.heappush(active, (spin[si][1], si))
            si += 1
        while active and active[0][0] <= x.start:
            heapq.heappop(active)
        labs = sorted({spin[i][2] for _e, i in active if spin[i][3] is not x and spin[i][3].dev == x.dev})
        if not labs:
            continue
        key = (short(x.name, 70), "+".join(labs))
        r = agg.get(key)
        if r is None:
            r = agg[key] = dict(n=0, gmin=x.grid, gmax=x.grid, block=set(), regs=0, ssm=0, dsm=0, breg=0)
        r["n"] += 1
        r["gmin"], r["gmax"] = min(r["gmin"], x.grid), max(r["gmax"], x.grid)
        r["block"].add(x.block)
        r["regs"] = max(r["regs"], x.regs)
        r["ssm"], r["dsm"] = max(r["ssm"], x.ssm), max(r["dsm"], x.dsm)
        r["breg"] = max(r["breg"], x.block * x.regs)
    flagged = []
    out.append("  %-70s %-50s %7s %11s %6s %4s %7s %13s  %s" % ("kernel starting during a spinner", "spinner(s) resident",
                                                               "count", "grid", "block", "reg", "blk*reg",
                                                               "smem st+dyn B", "flag"))
    for (nm, labs), r in sorted(agg.items(), key=lambda kv: (kv[0][1], -kv[1]["n"])):
        flag = []
        if r["breg"] > REG_LIMIT:
            flag.append("REG>%d" % REG_LIMIT)
        if r["ssm"] + r["dsm"] > SMEM_LIMIT:
            flag.append("SMEM>100KB")
        if flag:
            flagged.append((nm, labs, r, flag))
        elif r["breg"] == REG_LIMIT:
            flag.append("(at the register limit)")
        out.append("  %-70s %-50s %7d %11s %6s %4d %7d %6d+%-6d  %s" % (
            nm, labs[:50], r["n"], "%d-%d" % (r["gmin"], r["gmax"]) if r["gmin"] != r["gmax"] else str(r["gmin"]),
            "/".join(str(b) for b in sorted(r["block"])), r["regs"], r["breg"], r["ssm"], r["dsm"], " ".join(flag)))
    out.append("  flagged (block threads x regs > %d or smem > 100 KB): %d%s" % (
        REG_LIMIT, len(flagged), "" if not flagged else ": " + "; ".join(
            "%s during %s (%s)" % (nm, labs, " ".join(f)) for nm, labs, r, f in flagged)))
    return out


# ---------------------------------------------------------------------------------------------------- report
def report(cap, pid, tid, steps_all, rows, agg, calls, acts, a):
    P = print
    n = len(rows)
    pr = "  %-44s %s"

    def col(key):
        return [r.get(key) for r in rows]
    P("=" * 118)
    P("capture %s" % cap.path)
    P("pid %d (%s), main thread %d '%s': %d '%s' ranges, %d analysed (first %d dropped), %d main-thread API calls, "
      "%d GPU activities" % (pid, cap.pname.get(pid, "?"), tid & 0xFFFFFF, cap.tname.get(tid, "?"), len(steps_all),
                             a.range, n, a.skip, agg["mcalls"], len(acts)))
    labs = Counter(r["label"] for r in rows)
    P("step labels: " + ", ".join("%s x%d" % (k or "-", v) for k, v in labs.most_common()))
    P("values in us: median [p10 p90] over instances")

    P("\n== T1 planning end -> first wire copy")
    P("  planning end = end of the last producer kernel (%s) starting before the dispatch GEMM;" % "/".join(PRODUCER))
    P("  first wire = first PtoP / DtoD memcpy (any size), NVSHMEM put / signal kernel or lopep copy kernel after it")
    P(pr % ("T1   (headline)", st3(col("T1_us"))))
    P(pr % ("T1p  (first PAYLOAD wire, >= %d B or rma put)" % a.payload_bytes, st3(col("T1p_payload_us"))))
    P(pr % ("T1L  (literal: all planning kernels pre-GEMM1)", st3(col("T1L_literal_us"))))
    P(pr % ("planning end, rel. GPU window start", st3(col("plan_end_rel_us"))))
    P("  planning-end kernel: " + "; ".join("%s x%d" % kv for kv in agg["plan_end_names"].most_common(3)))
    P("  first wire (T1):     " + "; ".join("%s x%d" % kv for kv in agg["wire_names"].most_common(3)))
    P("  first wire (T1p):    " + "; ".join("%s x%d" % kv for kv in agg["wirep_names"].most_common(3)))
    if agg["n_noplan"]:
        P("  steps without a producer kernel before GEMM1: %d (T1 undefined there)" % agg["n_noplan"])
    if len(labs) > 1:
        for lab, _c in labs.most_common():
            sub = [r for r in rows if r["label"] == lab]
            P("  [%-14s n=%4d] T1 %s   T1p %s" % (lab or "-", len(sub), st3([r.get("T1_us") for r in sub]),
                                                st3([r.get("T1p_payload_us") for r in sub])))

    P("\n== T2 GPU idle inside the step's GPU window")
    P(pr % ("GPU window", st3(col("gpu_window_us"))))
    P(pr % ("GPU idle", st3(col("gpu_idle_us"))))
    for top in ("sync", "api", "host"):
        P(pr % ("  idle while main thread in %s" % top, st3(col("idle_%s_us" % top))))
    P("  by main-thread activity (mean per step = sum / n; median of steps where present):")
    for c, v in sorted(agg["cat_us"].items(), key=lambda kv: -sum(kv[1]))[:a.top * 2]:
        P("    %-46s mean %8.1f   median-when-present %8.1f   in %d/%d" % (c, sum(v) / n, pct(v, 0.5), len(v), n))
    P("  GPU busy by class (summed activity time in the window, mean per step; overlapping streams add up):")
    P("    " + "  ".join("%s %.1f" % (c, sum(v) / n) for c, v in sorted(agg["cls_us"].items(), key=lambda kv: -sum(kv[1]))))

    P("\n== T3 host blocking and host lead (main thread)")
    P(pr % ("blocked: synchronize + sync memcpy + D2H pageable", st3(col("T3_blocked_us"))))
    P(pr % ("  synchronize calls", st3(col("T3_sync_calls_us"))))
    P(pr % ("  synchronous memcpy calls", st3(col("T3_sync_memcpy_us"))))
    P(pr % ("  async D2H into pageable memory", st3(col("T3_d2h_pageable_us"))))
    P(pr % ("  blocking calls per step (count)", st3(col("T3_blocked_n"))))
    P(pr % ("polls (event / stream query), not counted", st3(col("T3_query_us"))))
    P(pr % ("lead: first activity (GPU start - call start)", st3(col("T3_lead_first_us"))))
    P(pr % ("lead: dispatch GEMM", st3(col("T3_lead_gemm1_us"))))
    P(pr % ("lead: first wire copy", st3(col("T3_lead_first_wire_us"))))
    P(pr % ("lead: combine GEMM", st3(col("T3_lead_gemm2_us"))))
    P(pr % ("host range duration", st3(col("host_us"))))
    P(pr % ("GPU window duration", st3(col("gpu_window_us"))))
    P(pr % ("GPU start - host range start", st3(col("gpu_start_minus_host_start_us"))))
    P(pr % ("GPU end - host range end", st3(col("gpu_end_minus_host_end_us"))))

    P("\n== T4 API calls per step (main thread: inside the host range; other threads: range start .. next step start)")
    P("  %-18s %22s %24s %9s   %12s %12s" % ("group", "main count", "main us", "main drv", "other count",
                                                "other us"))
    for g in T4_GROUPS:
        m, o = agg["t4_main"][g], agg["t4_oth"][g]
        P("  %-18s %22s %24s %9.1f   %12.1f %12.1f" % (g, st3c(m[0]), st3c(m[1]), agg["t4_main_drv"][g] / max(n, 1),
                                                     mean(o[0]), mean(o[1])))
    tot_m = [sum(agg["t4_main"][g][1][i] for g in T4_GROUPS) for i in range(n)]
    tot_mn = [sum(agg["t4_main"][g][0][i] for g in T4_GROUPS) for i in range(n)]
    P("  %-18s %22s %22s" % ("total main", st3c(tot_mn), st3c(tot_m)))
    P("  (main: median (p10-p90) per step; 'main drv' = mean driver-API calls per step in the group; other threads:"
      " mean per step)")
    P("  'other' on the main thread (mean per step): " + ", ".join(
        "%s %.1f" % (k, v / max(n, 1)) for k, v in agg["t4_other_names"].most_common(8)))
    P("  other-thread calls in step slots, by thread name (mean per step): " + (", ".join(
        "%s %.1f" % (k, v / max(n, 1)) for k, v in agg["t4_oth_thr"].most_common(6)) or "none"))

    P("\n== T5 wire-proxy latency")
    if not agg["proxy_tids"]:
        P("  n/a (proxy off: no thread other than the main thread emits lopep.proxy NVTX events)")
    else:
        P("  proxy threads: " + ", ".join("%d '%s'" % (g & 0xFFFFFF, cap.tname.get(g, "?")) for g in agg["proxy_tids"]))
        P(pr % ("planning end -> proxy first API call", st3(col("T5_proxy_first_api_us"))))
        P(pr % ("planning end -> first wire copy (=T1)", st3(col("T5_first_wire_us"))))
        P(pr % ("planning end -> first proxy-issued wire", st3(col("T5_first_proxy_wire_us"))))
        P(pr % ("lopep.proxy ranges per step (count)", st3(col("T5_proxy_ranges_n"))))
        P(pr % ("lopep.proxy range time per step", st3(col("T5_proxy_ranges_us"))))

    P("\n== T6 R1 tripwire (lookups / loads > %.0f us after warm-up)" % a.lookup_us)
    for line in tripwire(cap, pid, steps_all, calls, acts, a.lookup_us, cap.marks.get(tid, [])):
        P(line)

    P("\n== T7 R5 report (kernels starting while a spinner is resident, whole capture)")
    for line in r5_report(acts, a.top):
        P(line)

    P("\n== T8 sequential phases of the GPU window (%d steps with both GEMMs; %d without)" % (
        agg["seg_n"], agg["n_nogemm"]))
    for nm, _key, idx in T8_PHASES:
        v = [sum(r["seg%d_us" % i] for i in idx) for r in rows if "seg0_us" in r]
        vi = [sum(r["seg%d_idle_us" % i] for i in idx) for r in rows if "seg0_us" in r]
        P("  %-28s %s   idle %s" % (nm, st3(v), st3(vi)))
    P("\n  floor list (segments ranked by median; [plan_end -> GEMM1] split at the first wire copy):")
    segs = []
    for i, nm in enumerate(SEGS):
        v = [r["seg%d_us" % i] for r in rows if "seg0_us" in r]
        vi = [r["seg%d_idle_us" % i] for r in rows if "seg0_us" in r]
        segs.append((pct(v, 0.5), i, nm, v, vi))
    win = pct([r["gpu_window_us"] for r in rows if "seg0_us" in r], 0.5)
    sn = max(agg["seg_n"], 1)
    for rank, (m, i, nm, v, vi) in enumerate(sorted(segs, reverse=True), 1):
        P("  %d. %-26s %s  (%4.1f %% of window)  GPU idle %s" % (rank, nm, st3(v), 100 * m / win if win else 0, st3(vi)))
        ic = sorted(agg["seg_idle_cat"][i].items(), key=lambda kv: -kv[1])[:3]
        bc = sorted(agg["seg_cls"][i].items(), key=lambda kv: -kv[1])[:4]
        if ic:
            P("       idle while: " + ", ".join("%s %.0f" % (c, t / sn) for c, t in ic))
        if bc:
            P("       busy:       " + ", ".join("%s %.0f" % (c, t / sn) for c, t in bc))
    P("  (idle / busy contributors: mean us per step)")
    if len(labs) > 1:
        P("\n  by step label (medians, us):")
        P("    %-14s %5s %8s %8s %7s %7s %7s %7s %7s %7s %7s" % ("label", "n", "window", "idle", "T1", "P1", "P2",
                                                              "P3", "P4", "P5", "P6"))
        for lab, _c in labs.most_common():
            sub = [r for r in rows if r["label"] == lab]
            sg = [r for r in sub if "seg0_us" in r]
            ph = [pct([sum(r["seg%d_us" % i] for i in idx) for r in sg], 0.5) for _nm, _k, idx in T8_PHASES]
            P("    %-14s %5d %8.1f %8.1f %7.1f " % (lab or "-", len(sub), pct([r["gpu_window_us"] for r in sub], 0.5),
                                                 pct([r["gpu_idle_us"] for r in sub], 0.5),
                                                 pct([r.get("T1_us") for r in sub if r.get("T1_us") is not None], 0.5))
              + " ".join("%7.1f" % x for x in ph))
        P("    (P1..P6 = the T8 phases in order)")
    P("\n  largest GPU items per step (summed duration of the step's own activities, mean per step):")
    for nm, t in sorted(agg["items"].items(), key=lambda kv: -kv[1])[:a.top]:
        P("    %8.1f  %s" % (t / max(n, 1), nm))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite")
    ap.add_argument("--range", default="lopep.step")
    ap.add_argument("--pid", type=int, default=0, help="process to analyse (default: most lopep.step ranges)")
    ap.add_argument("--all-pids", action="store_true", help="report every GPU process (T6/T7 only without ranges)")
    ap.add_argument("--skip", type=int, default=2, help="drop the first N instances (capture start)")
    ap.add_argument("--csv", default="", help="per-instance rows with every scalar metric")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--payload-bytes", type=int, default=4096, help="T1p: smallest memcpy counted as payload")
    ap.add_argument("--lookup-us", type=float, default=100.0, help="T6 threshold")
    ap.add_argument("--label", default="", help="keep only steps whose ledger 'class' label starts with this")
    a = ap.parse_args()
    cap = Capture(a.sqlite)
    st = cap.step_threads(a.range)
    if a.all_pids:
        pids = sorted(set(cap.gpu_pids()) | set(st))
    elif a.pid:
        pids = [a.pid]
    elif st:
        pids = [max(st, key=lambda p: len(st[p][1]))]
    else:
        print("no '%s' ranges in %s" % (a.range, a.sqlite))
        return 1
    all_rows = []
    for pid in pids:
        calls, acts = cap.load_process(pid)
        if pid not in st:
            print("=" * 118)
            print("capture %s\npid %d (%s): no '%s' ranges -> T1-T5 / T8 need the ledger's step ranges; T6 / T7 only"
                  % (cap.path, pid, cap.pname.get(pid, "?"), a.range))
            print("\n== T6 R1 tripwire (lookups / loads > %.0f us after warm-up)" % a.lookup_us)
            ref = st[max(st, key=lambda p: len(st[p][1]))][1] if st else []
            for line in tripwire(cap, pid, ref, calls, acts, a.lookup_us):
                print(line)
            print("\n== T7 R5 report (kernels starting while a spinner is resident, whole capture)")
            for line in r5_report(acts, a.top):
                print(line)
            continue
        tid, steps_all = st[pid]
        rows, agg = analyze(cap, pid, tid, steps_all, calls, acts, a)
        if not rows:
            print("pid %d: no analysable instances" % pid)
            continue
        report(cap, pid, tid, steps_all, rows, agg, calls, acts, a)
        all_rows += rows
        sys.stdout.flush()
    if a.csv and all_rows:
        keys, seen = [], set()
        for r in all_rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: ("%.3f" % v if isinstance(v, float) and not math.isnan(v) else v)
                            for k, v in r.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
