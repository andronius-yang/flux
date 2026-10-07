#!/usr/bin/env python3
"""figs/case_study: timeline extraction for the SERVING-path case study (CS_v7, 2026-10-06).

Input: one serving_case_study.py capture directory (<out>/<family>_nsys/{nsys/*.nsys-rep, records/traces_r*.npz}).
Output: the JSON / CSV of extract_timeline.py (cells -> ranks -> iters -> events with lane / task), so
build_case_study.py draws it unchanged (cell ids carry the variant suffix `_serving`).

The serving path issues its communication from kernels, so nsys shows kernel spans, not copy-engine records:
  GPU lane      kernels by name (GEMMs, reduces, planning), as extract_timeline.py
  NIC lane      the device wire's %globaltimer stamps (lopep LOPEP_WIRE_TRACE, read back per iteration):
                dispatch round = [put issued (slot ready seen), put returned (the group's quiet + signals)],
                combine position = [pre-reduce flag seen, put returned]
  NVLink lane   pack_push kernel spans (own-node rows stored into the node peers), gateway forwards [remote window landed, forwarded] (block 0 of the forward kernel),
                the combine pack's per-block push windows (LOPEP_PACK_TRACE) and the swap lane's pushes
                (lane_push_kernel = Expert Swap); the relay kernel's pulls are NOT drawn (no timestamps, the
                kernel span is mostly waiting for announces and free slots)
  not drawn     kernels whose span is mostly a spin on a signal (wire, combine wire, relay, forward, lane wait,
                lane commit: its copy is local, staging -> slot, see the cs_v3 note)
Trace stamps are aligned per (rank, iteration): offset = nsys start of the stamped kernel - its entry stamp
(wire_kernel for the dispatch rows, combine_wire_kernel for the combine row, the earliest pack block for the
pack trace); the offsets of one GPU agree to a few microseconds (printed).

usage: python figs/case_study/extract_serving_timeline.py <capture_dir> --out <sqlite dir> --json <out.json> --csv <out.csv>
"""
import argparse, csv, glob, json, os, re, sqlite3, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import extract_timeline as ET  # noqa: E402

CELL = "ablation_l01_s2_swapall_rst_3d_dual3_str4_p2p_r2_serving_{tag}_b64_k8_nsys"
FAMILY = {"sc": ("trace-b549f7", "trace:model=Kimi-K2;pools=livecodebench/execution;layer=5;sem=homog;dslots=64:32;"
                 "sched=8 topics (S-C, dwell 4)"),
          "lcb": ("trace-610042", "trace:model=Kimi-K2;pools=livecodebench/execution;layer=5;sem=homog;dslots=64:32")}

# serving-path kernel classes (first match wins); lane None = not drawn (spin-dominated span)
SERVING_CLASSES = [
    (r"^Kernel$", ("gpu", "gemm")),
    (r"^(pack_push_kernel)", ("nvlink", "nvlink.token")),
    (r"^(relay_kernel|wire_kernel|forward_kernel|combine_wire_kernel|a2av_lane_wait_kernel|lane_commit_kernel|wait_geq_kernel)", (None, "spin")),
    # the device dual3 lane (python/flux/testing/serving_dual3.py): push_w1 = W1 phase under the dispatch GEMM,
    # pull_w2 = W2 phase under the combine GEMM; both spin on their GEMM's start mark before copying (trimmed below)
    (r"^(lane_push_kernel|pull_w2_kernel|push_w1_kernel)", ("nvlink", "nvlink.swap")),
    (r"^(wait_acks_kernel|wait_pushed_kernel)", (None, "spin")),
    (r"^a2av_combine_prereduce", ("gpu", "combine.prereduce")),
    (r"^a2av_combine_(pack|tail_push)", ("gpu", "combine.pack")),
    (r"^a2av_combine_bucket_reduce", ("gpu", "combine.reduce")),
    # the deferred verdict's all-reduce closes the forward and waits for every rank (an end-of-layer barrier)
    (r"^ncclDevKernel_AllReduce", ("wait", "barrier")),
    (r"^(barrier_on_stream|nvshmemi_signal_wait)", ("wait", "barrier")),
    (r"^nvshmemi_proxy_rma", ("nic", "nic.put")),     # host-issued blocking put (stream busy while the NIC moves data)
    (r"^ncclDevKernel_(AllGather|Broadcast)", ("gpu", "plan.comm")),
    (r"^(vectorized_|unrolled_)?elementwise_kernel$|^reduce_kernel$|^fill", ("gpu", "misc")),
    (r".*", ("gpu", "plan.compute")),          # planning chain, metadata, tables, step head, activation helpers
]


def classify(name):
    for pat, cls in SERVING_CLASSES:
        if re.match(pat, name):
            return cls
    return ("gpu", "plan.compute")


def kernel_events(cur, pid, s, e):
    ev = []
    for n, a, b, st, c, enq in cur.execute(ET.KQ, (pid, s, e)):
        lane, task = classify(n)
        ev.append(dict(kind="k", name=n, lane=lane, task=task, t0=a, t1=b, stream=st, cid=c, enq=enq))
    for ck, by, a0, b0, st, c, enq in cur.execute(ET.MQ, (pid, s, e)):
        kind = ET.MEMCPY_KIND.get(ck, f"copy{ck}")
        if kind == "p2p":
            lane, task = "nvlink", "nvlink.token"
        elif kind == "d2d":
            lane, task = "gpu", "copy.d2d"
        else:
            lane, task = "host", f"copy.{kind}"
        ev.append(dict(kind="m", name=f"memcpy_{kind}", lane=lane, task=task, bytes=by, t0=a0, t1=b0, stream=st, cid=c, enq=enq))
    ev.sort(key=lambda x: x["t0"])
    n = 0
    for x in ev:
        if x["task"] == "gemm":
            x["task"] = "gemm.l0" if n == 0 else ("gemm.l1" if n == 1 else f"gemm.{n}")
            n += 1
    # dual3 phase kernels are launched before their GEMM and wait for its start mark: the copy starts with the GEMM
    g = {x["task"]: x["t0"] for x in ev if x["task"] in ("gemm.l0", "gemm.l1")}
    for x in ev:
        if x["name"].startswith("push_w1_kernel") and "gemm.l0" in g:
            x["t0"] = min(max(x["t0"], g["gemm.l0"]), x["t1"]); x["phase"] = "late"
        elif x["name"].startswith("pull_w2_kernel") and "gemm.l1" in g:
            x["t0"] = min(max(x["t0"], g["gemm.l1"]), x["t1"]); x["phase"] = "l1"
    return ev


def first(ev, pat):
    m = [x for x in ev if re.match(pat, x["name"])]
    return m[0] if m else None


def trace_events(ev, wire, pack, s):
    """device-stamp events for one (rank, iteration); ev = that iteration's kernels (ns, absolute)"""
    out, notes = [], {}
    wk, ck, pk = first(ev, r"^wire_kernel"), first(ev, r"^combine_wire_kernel"), first(ev, r"^a2av_combine_(pack|tail_push)")
    off_d = wk["t0"] - int(wire[0][0]) if (wk is not None and wire is not None and wire[0][0]) else None
    off_c = ck["t0"] - int(wire[1][0]) if (ck is not None and wire is not None and wire[1][0]) else None
    pe = pack[:, 0][pack[:, 0] > 0] if pack is not None and pack.size else np.zeros(0)
    off_p = pk["t0"] - int(pe.min()) if (pk is not None and pe.size) else None
    notes.update(off_d=off_d, off_c=off_c, off_p=off_p)
    rel = lambda t, off: (int(t) + off - s) / 1e6
    if off_d is not None:
        ready, ret = {}, {}
        for dn in range(1, 21):
            a, b = int(wire[0][1 + 3 * (dn - 1)]), int(wire[0][2 + 3 * (dn - 1)])
            if a and b:
                ready[dn], ret[dn] = a, b
                out.append(dict(kind="t", name=f"dispatch put round {dn}", lane="nic", task="nic.put",
                                t0=rel(a, off_d), t1=rel(b, off_d), rows=int(wire[0][3 + 3 * (dn - 1)])))
        # the relay kernel's NVLink pulls are not drawn: it has no timestamps and its span is mostly waiting
        for dn in range(1, 21):   # gateway forward (block 0 of the forward kernel)
            a, b = int(wire[2][1 + 2 * (dn - 1)]), int(wire[2][2 + 2 * (dn - 1)])
            if a and b and b > a:
                out.append(dict(kind="t", name=f"gateway forward window {dn}", lane="nvlink", task="nvlink.token",
                                t0=rel(a, off_d), t1=rel(b, off_d)))
    if off_c is not None:
        for gi in range(20):
            a, b = int(wire[1][1 + 3 * gi]), int(wire[1][2 + 3 * gi])
            if a and b:
                out.append(dict(kind="t", name=f"combine put {gi}", lane="nic", task="nic.put",
                                t0=rel(a, off_c), t1=rel(b, off_c), rows=int(wire[1][3 + 3 * gi])))
    if off_p is not None:
        iv = []
        for blk in pack:
            if not blk[0]:
                continue
            for gi in range(20):
                a, b = int(blk[1 + 3 * gi]), int(blk[2 + 3 * gi])
                if a and b and b > a:
                    iv.append((a, b))
        iv.sort(); merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            out.append(dict(kind="t", name="combine NVLink pushes", lane="nvlink", task="nvlink.token",
                            t0=rel(a, off_p), t1=rel(b, off_p)))
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--csv", required=True); ap.add_argument("--ffn", type=int, default=2048); ap.add_argument("--H", type=int, default=7168)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    result = {"capsule": os.path.basename(os.path.normpath(a.capture)), "cells": {}}
    rows_out, offs = [], []
    for fam, (tag, fparams) in FAMILY.items():
        cdir = os.path.join(a.capture, f"{fam}_nsys")
        reps = sorted(glob.glob(os.path.join(cdir, "nsys", "*.nsys-rep")))
        if not reps:
            print("no reports:", cdir, file=sys.stderr); continue
        cid = CELL.format(tag=tag)
        cell = dict(variant=cid.rsplit("_" + tag, 1)[0], budget_mib=64, status="ok", W=16, nnodes=4, rpn=4,
                    family_params=fparams, H=a.H, ffn=a.ffn, slot_bytes=a.ffn * a.H * 2, ranks={}, recorded={}, info={})
        traces = {}
        for f in glob.glob(os.path.join(cdir, "records", "traces_r*.npz")):
            z = np.load(f)
            traces[int(z["rank"])] = (z["wire"], z["pack"], int(z["warmup"]))
        for rep in reps:
            nid = int(re.search(r"node(\d+)_", os.path.basename(rep)).group(1))
            db = sqlite3.connect(ET.export(rep, a.out, cid)); cur = db.cursor()
            pids = sorted({t >> 24 for (t,) in cur.execute("SELECT DISTINCT globalTid FROM NVTX_EVENTS")})
            for pid in pids:
                dev = cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid>>24=?", (pid,)).fetchall()
                if not dev: continue
                rank = nid * 4 + int(dev[0][0])
                nv = cur.execute(ET.NQ, (pid,)).fetchall()
                iters = [(t, s, e) for t, s, e, _ in nv if t and re.fullmatch(r"iter\d+(_warmup)?", t)]
                rk = dict(node=nid, device=int(dev[0][0]), iters={})
                wire_all, pack_all, _w = traces.get(rank, (None, None, 0))
                for txt, s, e in iters:
                    i = int(re.match(r"iter(\d+)", txt).group(1))
                    kev = kernel_events(cur, pid, s, e)
                    wire = wire_all[i] if (wire_all is not None and wire_all.ndim == 3 and i < len(wire_all)) else None
                    pack = pack_all[i] if (pack_all is not None and pack_all.ndim == 3 and i < len(pack_all)) else None
                    tev, notes = trace_events(kev, wire, pack, s)
                    if notes.get("off_d") is not None and notes.get("off_c") is not None:
                        offs.append(abs(notes["off_d"] - notes["off_c"]) / 1e3)
                    ev = []
                    for x in kev:
                        if x["lane"] is None: continue
                        # the staged lane launches its push on every rank every step; a rank with nothing to send
                        # returns at once (one slot's two matrices take ~0.3 ms): only real pushes are swaps
                        if x["task"] == "nvlink.swap" and x["t1"] - x["t0"] < 50_000: continue
                        x = dict(x); x["t0"] = (x["t0"] - s) / 1e6; x["t1"] = (x["t1"] - s) / 1e6
                        x["enq"] = (x["enq"] - s) / 1e6 if x.get("enq") is not None else None
                        ev.append(x)
                    ev += tev
                    ev.sort(key=lambda x: x["t0"])
                    dev_end = max([x["t1"] for x in ev], default=0.0)
                    it = dict(host_ms=(e - s) / 1e6, dev_end_ms=dev_end, events=ev, host_ranges=[], proxy_ranges=[], align=notes)
                    busy = {}
                    for x in ev:
                        busy.setdefault(x["task"], []).append((x["t0"], x["t1"]))
                    summ = {k: round(ET.union_busy(v), 3) for k, v in busy.items()}
                    for ln in ("gpu", "nic", "nvlink"):
                        summ[f"_lane_{ln}"] = round(ET.union_busy([(x["t0"], x["t1"]) for x in ev if x["lane"] == ln]), 3)
                    it["summary"] = summ
                    rk["iters"][txt] = it
                    rows_out.append(dict(cell_id=cid, rank=rank, iter=txt, host_ms=round(it["host_ms"], 3), dev_end_ms=round(dev_end, 3), **summ))
                cell["ranks"][str(rank)] = rk
            db.close()
        result["cells"][cid] = cell
        print(cid, "ranks:", len(cell["ranks"]), file=sys.stderr)
    if offs:
        print(f"wire/combine-wire alignment disagreement (us): median {np.median(offs):.2f} max {max(offs):.2f}", file=sys.stderr)
    json.dump(result, open(a.json, "w"))
    keys = sorted({k for r in rows_out for k in r}, key=lambda k: (k not in ("cell_id", "rank", "iter", "host_ms", "dev_end_ms"), k))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows_out: w.writerow(r)
    print("wrote", a.json, a.csv, file=sys.stderr)


if __name__ == "__main__":
    main()
