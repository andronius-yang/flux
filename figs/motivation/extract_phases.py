#!/usr/bin/env python3
"""Per-rank, per-iteration device event lists from a sweep capsule's nsys reps.

For every nsys cell of the capsule: export each node rep to sqlite (cached in
--out), map pid -> global rank (node index from the rep filename, device id
from CUPTI), attribute every kernel / memcpy to the timed `iterN` NVTX range
whose host bracket launched it (correlationId join — the host range closes at
enqueue, never filter by timestamp), and rebase to that iteration's host start.
Also extracts the EPLB one-time placement puts (NVTX `place_put ...` ranges)
and per-rank stats (matrix rows / inter-node bytes, recorder ledgers).
Writes one JSON per capsule. Stdlib only.
"""
import argparse, csv, glob, json, os, re, sqlite3, subprocess, sys, collections

NSYS = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys"
KQ = ("SELECT s.value, k.start, k.end, k.streamId, k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId AND r.globalTid>>24=k.globalPid>>24 "
      "JOIN StringIds s ON s.id=k.shortName WHERE k.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY k.start")
MQ = ("SELECT m.copyKind, m.bytes, m.start, m.end, m.streamId, m.correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
      "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId AND r.globalTid>>24=m.globalPid>>24 "
      "WHERE m.globalPid>>24=? AND r.start>=? AND r.start<=? ORDER BY m.start")

def export(rep, out, tag):
    # tag = cell id: rep basenames repeat across cells (node<N>_<job>), never cache by basename alone
    db = os.path.join(out, tag + "__" + os.path.basename(rep).replace(".nsys-rep", ".sqlite"))
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return db

def events(cur, pid, s, e):
    ev = [dict(kind="k", name=n, t0=a, t1=b, stream=st, cid=c) for n, a, b, st, c in cur.execute(KQ, (pid, s, e))]
    ev += [dict(kind="m", name=f"memcpy{ck}", bytes=by, t0=a, t1=b, stream=st, cid=c) for ck, by, a, b, st, c in cur.execute(MQ, (pid, s, e))]
    return sorted(ev, key=lambda x: x["t0"])

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capsule"); ap.add_argument("--out", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    result = {"capsule": a.capsule, "cells": {}}
    for row in csv.DictReader(open(os.path.join("sweeps/results/runs", a.capsule, "cells.csv"))):
        if row["mode"] != "nsys" or not row["nsys_path"]: continue
        rpn = int(row["ranks_per_node"]); M = read_matrix(row["matrix_path"]); W = len(M); cb = int(row["chunk_bytes"])
        node = lambda r: r // rpn
        cell = dict(variant=row["variant"], budget_mib=int(row["budget_mib"]), status=row["status"], W=W, nnodes=int(row["nnodes"]),
                    rpn=rpn, tokens_per_rank=int(row["tokens_per_rank"]), H=int(row["H"]), chunk_bytes=cb,
                    rows_per_rank=[sum(M[s][d] for s in range(W)) // cb for d in range(W)],
                    send_inter_bytes=[sum(M[s][d] for d in range(W) if node(d) != node(s)) for s in range(W)],
                    send_intra_bytes=[sum(M[s][d] for d in range(W) if node(d) == node(s) and d != s) for s in range(W)],
                    recv_inter_bytes=[sum(M[s][d] for s in range(W) if node(d) != node(s)) for d in range(W)],
                    ranks={})
        cell_dir = os.path.dirname(row["nsys_path"])
        per, info = records(cell_dir)
        cell["recorded"] = {m: {str(r): v for r, v in d.items()} for m, d in per.items()}
        cell["info"] = {k: {str(r): v for r, v in d.items()} for k, d in info.items()
                        if k in ("gemm_rows_per_rank", "eplb_weight_place_sends", "eplb_weight_place_bytes", "eplb_imbalance_before",
                                 "eplb_imbalance_after", "eplb_rehomed_slots", "eplb_remote_frac", "eplb_wire_bytes", "eplb_replicas_total")}
        for rep in sorted(glob.glob(os.path.join(row["nsys_path"], "*.nsys-rep"))):
            nid = int(re.search(r"node(\d+)_", os.path.basename(rep)).group(1))
            db = sqlite3.connect(export(rep, a.out, row["cell_id"])); cur = db.cursor()
            nv = cur.execute("SELECT globalTid, text, start, end FROM NVTX_EVENTS WHERE text IS NOT NULL ORDER BY start").fetchall()
            pids = sorted({t >> 24 for t, *_ in nv})
            for pid in pids:
                dev = cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid>>24=?", (pid,)).fetchall()
                if not dev: continue
                rank = nid * rpn + int(dev[0][0])
                rk = dict(node=nid, device=int(dev[0][0]), iters={}, place=None)
                for t, txt, s, e in nv:
                    if t >> 24 != pid: continue
                    if re.fullmatch(r"iter\d+", txt):
                        ev = events(cur, pid, s, e)
                        for x in ev: x["t0"] = (x["t0"] - s) / 1e6; x["t1"] = (x["t1"] - s) / 1e6
                        rk["iters"][txt] = dict(host_ms=(e - s) / 1e6, events=ev)
                    elif txt == "eplb_place_weights":
                        ev = events(cur, pid, s, e)
                        puts = [(pt, ps, pe) for pt2, pt, ps, pe in nv if pt2 >> 24 == pid and pt.startswith("place_put")]
                        # label each event by the put range whose host bracket launched it
                        cid2put = {}
                        for pt, ps, pe in puts:
                            for cid, in cur.execute("SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid>>24=? AND start>=? AND start<=?", (pid, ps, pe)):
                                cid2put[cid] = pt
                        for x in ev:
                            x["put"] = cid2put.get(x["cid"]); x["t0"] = (x["t0"] - s) / 1e6; x["t1"] = (x["t1"] - s) / 1e6
                        rk["place"] = dict(host_ms=(e - s) / 1e6, n_puts=len(puts), events=ev)
                cell["ranks"][str(rank)] = rk
            db.close()
        result["cells"][row["cell_id"]] = cell
        print(row["cell_id"], row["status"], "ranks:", len(cell["ranks"]), file=sys.stderr)
    json.dump(result, open(a.json, "w"))
    print("wrote", a.json, file=sys.stderr)

if __name__ == "__main__":
    main()
