#!/usr/bin/env python3
"""figs/case_study: per-rank, per-iteration device/host timeline extraction
for the OURS case-study nsys capsules (CASE-STUDY ONLY, 2026-09-05).

Derived from figs/motivation/extract_phases.py (same nsys sqlite export +
correlationId attribution: a device event belongs to the iteration whose
host `iterN` NVTX range enqueued it). Adds:
  * event CLASSES (GPU / NIC / NVLink / host lanes, see CLASSIFY) so the
    timeline figure can colour by task,
  * host NVTX ranges (plan.* / swap.* under FLUX_OURS_NVTX=1) and the a2av
    proxy ranges (i<e>.src<s>.wait|pending|compute, inter/intra_epoch),
  * a per-(cell, rank, iter) summary CSV of busy ms per class.

nsys mode = timelines only, never latency (SCHEMA protocol rule 3).

usage: python figs/case_study/extract_timeline.py <capsule_run_id> \
          --out <sqlite cache dir> --json <out.json> --csv <summary.csv>
"""
import argparse, csv, glob, json, os, re, sqlite3, subprocess, sys

NSYS = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys"
KQ = ("SELECT s.value, k.start, k.end, k.streamId, k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId AND r.globalTid>>24=k.globalPid>>24 "
      "JOIN StringIds s ON s.id=k.shortName WHERE k.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY k.start")
MQ = ("SELECT m.copyKind, m.bytes, m.start, m.end, m.streamId, m.correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId AND r.globalTid>>24=m.globalPid>>24 "
      "WHERE m.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY m.start")
NQ = ("SELECT coalesce(s.value, n.text), n.start, n.end, n.globalTid FROM NVTX_EVENTS n "
      "LEFT JOIN StringIds s ON s.id=n.textId WHERE n.globalTid>>24=? AND n.end IS NOT NULL ORDER BY n.start")

# kernel-name -> (lane, task). Order matters: first match wins.
KERNEL_CLASSES = [
    (r"^Kernel$", ("gpu", "gemm")),                       # CUTLASS grouped GEMM (l0 then l1 by order)
    (r"^a2av_combine_prereduce", ("gpu", "combine.prereduce")),  # pre-topk reduce
    (r"^a2av_combine_pack", ("gpu", "combine.pack")),
    (r"^a2av_combine_bucket_reduce", ("gpu", "combine.reduce")),
    # dense (COMET) layer-1: fused gather + top-k reduce + reduce-scatter kernel and its inter-node reduce
    (r"^ep_topk_gather_rs", ("gpu", "combine.gather_rs")),
    (r"^internode_reduce", ("gpu", "combine.reduce")),
    (r"^nvshmemi_signal_wait", ("wait", "barrier")),
    (r"^nvshmemi_proxy_rma", ("nic", "nic.put")),          # blocking wire put (stream busy while NIC moves data)
    (r"^barrier_on_stream", ("wait", "barrier")),
    (r"^ncclDevKernel_(AllGather|AllReduce|Broadcast)", ("gpu", "plan.comm")),
    (r"^(pll_|compress_plan_|a2av_meta_counts|a2av_stable_scatter|a2av_consumer_build|a2av_stage1|prepare_workspace|DeviceRadixSort|searchsorted|indexSelect|indexFunc|kernelHistogram|CatArrayBatchedCopy|calc_gather_index|sort_scatter_index|device_kernel|make_workspace)", ("gpu", "plan.compute")),
    (r"^(vectorized_|unrolled_)?elementwise_kernel$|^reduce_kernel$|^fill", ("gpu", "misc")),
    (r".*", ("gpu", "other")),
]
MEMCPY_KIND = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p"}


def classify_kernel(name):
    for pat, cls in KERNEL_CLASSES:
        if re.match(pat, name):
            return cls
    return ("gpu", "other")


def export(rep, out, tag):
    db = os.path.join(out, tag + "__" + os.path.basename(rep).replace(".nsys-rep", ".sqlite"))
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type=sqlite", "--force-overwrite=true", "-o", db, rep],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return db


def events(cur, pid, s, e, slot_bytes):
    ev = []
    for n, a, b, st, c in cur.execute(KQ, (pid, s, e)):
        lane, task = classify_kernel(n)
        ev.append(dict(kind="k", name=n, lane=lane, task=task, t0=a, t1=b, stream=st, cid=c))
    for ck, by, a, b, st, c in cur.execute(MQ, (pid, s, e)):
        kind = MEMCPY_KIND.get(ck, f"copy{ck}")
        if kind == "p2p":
            # swap copies are EXACTLY one bf16 expert slot (verified 9/5: 29,360,128 B on the
            # movement stream; token chunks range 23-56 MB and never hit the exact size)
            lane, task = "nvlink", ("nvlink.swap" if by == slot_bytes else "nvlink.token")
        elif kind == "d2d":
            lane, task = "gpu", "copy.d2d"
        else:
            lane, task = "host", f"copy.{kind}"
        ev.append(dict(kind="m", name=f"memcpy_{kind}", lane=lane, task=task, bytes=by, t0=a, t1=b, stream=st, cid=c))
    ev.sort(key=lambda x: x["t0"])
    # l0 vs l1 GEMM: by order within the iteration
    n = 0
    for x in ev:
        if x["task"] == "gemm":
            x["task"] = "gemm.l0" if n == 0 else ("gemm.l1" if n == 1 else f"gemm.{n}")
            n += 1
    return ev


def read_matrix(path):
    ls = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    nums = [[int(float(x)) for x in l] for l in ls if all(x.replace(".", "").replace("-", "").isdigit() for x in l)]
    W = max(len(l) for l in nums)
    return [l for l in nums if len(l) == W][-W:]


def records(cell_dir):
    per, info = {}, {}
    for f in sorted(glob.glob(os.path.join(cell_dir, "records", "rank_*.jsonl"))):
        rk = None
        for line in open(f):
            d = json.loads(line)
            if d["type"] == "meta": rk = d["rank"]
            elif d["type"] == "iters": per.setdefault(d["metric"], {})[rk] = d["values_ms"]
            elif d["type"] == "cell_info":
                for k, v in d.items():
                    if k == "type": continue
                    info.setdefault(k, {})[rk] = v
    return per, info


def union_busy(intervals):
    """total ms covered by the union of [t0,t1) intervals (overlap-safe)."""
    tot, cur0, cur1 = 0.0, None, None
    for a, b in sorted(intervals):
        if cur1 is None or a > cur1:
            if cur1 is not None: tot += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None: tot += cur1 - cur0
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capsule"); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    result = {"capsule": a.capsule, "cells": {}}
    rows_out = []
    cap_dir = os.path.join("sweeps/results/runs", a.capsule)
    spec_txt = open(os.path.join(cap_dir, "spec.yaml")).read()
    ffn = int(re.search(r"^ffn_hidden:\s*(\d+)", spec_txt, re.M).group(1))
    for row in csv.DictReader(open(os.path.join(cap_dir, "cells.csv"))):
        if row["mode"] != "nsys" or not row["nsys_path"]: continue
        rpn = int(row["ranks_per_node"]); M = read_matrix(row["matrix_path"]); W = len(M)
        H = int(row["H"]); slot_bytes = ffn * H * 2          # one expert-slot weight (bf16) = the swap copy unit
        cell = dict(variant=row["variant"], budget_mib=int(row["budget_mib"]), status=row["status"], W=W,
                    nnodes=int(row["nnodes"]), rpn=rpn, family_params=row["family_params"], H=H, ffn=ffn,
                    slot_bytes=slot_bytes, ranks={})
        cell_dir = os.path.dirname(row["nsys_path"])
        per, info = records(cell_dir)
        cell["recorded"] = {m: {str(r): v for r, v in d.items()} for m, d in per.items()}
        cell["info"] = {k: {str(r): v for r, v in d.items()} for k, d in info.items()
                        if k.startswith(("ours_", "pv2_", "epic_pll"))}
        for rep in sorted(glob.glob(os.path.join(row["nsys_path"], "*.nsys-rep"))):
            nid = int(re.search(r"node(\d+)_", os.path.basename(rep)).group(1))
            db = sqlite3.connect(export(rep, a.out, row["cell_id"])); cur = db.cursor()
            pids = sorted({t >> 24 for (t,) in cur.execute("SELECT DISTINCT globalTid FROM NVTX_EVENTS")})
            for pid in pids:
                dev = cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid>>24=?", (pid,)).fetchall()
                if not dev: continue
                rank = nid * rpn + int(dev[0][0])
                nv = cur.execute(NQ, (pid,)).fetchall()
                iters = [(t, s, e) for t, s, e, _ in nv if t and re.fullmatch(r"iter\d+(_warmup)?", t)]
                rk = dict(node=nid, device=int(dev[0][0]), iters={})
                for txt, s, e in iters:
                    ev = events(cur, pid, s, e, slot_bytes)
                    for x in ev: x["t0"] = (x["t0"] - s) / 1e6; x["t1"] = (x["t1"] - s) / 1e6
                    dev_end = max([x["t1"] for x in ev], default=0.0)
                    # host ranges opened inside the iteration bracket (plan.*, swap.*), proxy ranges whose
                    # start falls before the device tail of this iteration
                    hr = [dict(name=t, t0=(s0 - s) / 1e6, t1=(e0 - s) / 1e6) for t, s0, e0, _ in nv
                          if t and s0 >= s and s0 <= e and (t.startswith("plan.") or t.startswith("swap."))]
                    pr = [dict(name=t, t0=(s0 - s) / 1e6, t1=(e0 - s) / 1e6) for t, s0, e0, _ in nv
                          if t and re.match(r"i\d+\.", t) and s0 >= s and (s0 - s) / 1e6 <= dev_end]
                    it = dict(host_ms=(e - s) / 1e6, dev_end_ms=dev_end, events=ev, host_ranges=hr, proxy_ranges=pr)
                    rk["iters"][txt] = it
                    busy = {}
                    for x in ev:
                        busy.setdefault(x["task"], []).append((x["t0"], x["t1"]))
                    summ = {k: round(union_busy(v), 3) for k, v in busy.items()}
                    summ["_lane_gpu"] = round(union_busy([(x["t0"], x["t1"]) for x in ev if x["lane"] == "gpu"]), 3)
                    summ["_lane_nic"] = round(union_busy([(x["t0"], x["t1"]) for x in ev if x["lane"] == "nic"]), 3)
                    summ["_lane_nvlink"] = round(union_busy([(x["t0"], x["t1"]) for x in ev if x["lane"] == "nvlink"]), 3)
                    summ["_bytes_p2p_swap"] = sum(x.get("bytes", 0) for x in ev if x["task"] == "nvlink.swap")
                    summ["_bytes_p2p_token"] = sum(x.get("bytes", 0) for x in ev if x["task"] == "nvlink.token")
                    summ["_host_swap_ms"] = round(union_busy([(h["t0"], h["t1"]) for h in hr if h["name"].startswith("swap.")]), 3)
                    summ["_host_plan_ms"] = round(union_busy([(h["t0"], h["t1"]) for h in hr if h["name"].startswith("plan.")]), 3)
                    it["summary"] = summ
                    rows_out.append(dict(cell_id=row["cell_id"], variant=row["variant"], family=row["family_params"],
                                         rank=rank, iter=txt, host_ms=round(it["host_ms"], 3), dev_end_ms=round(dev_end, 3), **summ))
                cell["ranks"][str(rank)] = rk
            db.close()
        result["cells"][row["cell_id"]] = cell
        print(row["cell_id"], row["status"], "ranks:", len(cell["ranks"]), file=sys.stderr)
    json.dump(result, open(a.json, "w"))
    keys = sorted({k for r in rows_out for k in r}, key=lambda k: (not k.startswith(("cell_id", "variant", "family", "rank", "iter", "host_ms", "dev_end")), k))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows_out: w.writerow(r)
    print("wrote", a.json, a.csv, file=sys.stderr)


if __name__ == "__main__":
    main()
