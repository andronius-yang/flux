#!/usr/bin/env python3
"""H100/ALPS weak-scaling dataset builder (docs/handoff/35_h100_alps_weak_scaling.md).

Reads the lane's capsules under sweeps/results/runs/ and writes
figs/weak_scaling/h100/figure_src.csv with EXACTLY the column contract of
figs/weak_scaling/figure_src.csv (figure_src.md), so the unchanged generator
renders the H100 figure:

    python3 figs/weak_scaling/h100/build_figure_src.py            # alps capsules, 2..32n
    python3 figs/weak_scaling/make_figure.py --baseline nvshmem --stacked --src-dir figs/weak_scaling/h100
    python3 figs/weak_scaling/make_figure.py --baseline nvshmem --src-dir figs/weak_scaling/h100   # verA/verB

Statistic (verified 2026-09-06 against the Perlmutter figure rows to 3 decimals):
    total_ms = MEAN over all (rank, timed iteration) samples of the recorder's
               total_ms  (rule-5 plan-inclusive: plan_comm + plan + e2e), isolated mode.
    A max-across-ranks mean is written alongside as total_ms_maxrank for reference.
Row selection: per (nodes, system, budget) the NEWEST capsule whose cell is
status ok; a cell with any other status is written with an empty total_ms and
its status (the generator draws the fail note for missing reference cells).
"Ours" convention (main figure / ver4): the generator takes min(ours, dwire)
per node; the ours row's speedup columns are computed against that min here.
Build identity: the flux_libs sha of every capsule used is checked to be ONE
binary (SCHEMA rule 4); a mismatch is printed loudly and recorded in the note.

Self-test on Perlmutter capsules:
    python3 figs/weak_scaling/h100/build_figure_src.py --platform perlmutter \
        --capsules 20260904-004243_perlmutter_da6f9186 --nodes 2 --out /tmp/x.csv
    -> 2n nvshmem 5.003 / ours 4.240 (b1), 105.516 / 36.097 (b64) = figure_src.md values.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "sweeps", "results", "runs")
HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM_OF = {
    "ours_l01_s1_pv2_r2": "ours",
    "ours_l01_s1_pv2_r2_dwire": "dwire",
    "l01_nvshmem": "nvshmem",
    "l01_allgather_dense": "comet",   # not part of the H100 scope; picked up if ever run
}
COLUMNS = ["nodes", "ranks", "system", "budget_mib", "total_ms", "tokens_per_rank",
           "total_tokens", "throughput_mtok_s", "speedup_vs_comet", "speedup_vs_nvshmem",
           "status", "capsule", "cell_id", "source_dataset", "note", "total_ms_maxrank",
           "binary"]


def capsule_dirs(platform, explicit):
    if explicit:
        out = []
        for c in explicit:
            hits = sorted(glob.glob(os.path.join(RUNS, c + "*")))
            if not hits:
                raise SystemExit(f"capsule not found: {c}")
            out += hits
        return out
    return sorted(glob.glob(os.path.join(RUNS, f"*_{platform}_*")))


def load_capsule(d):
    cells = {r["cell_id"]: r for r in csv.DictReader(open(os.path.join(d, "cells.csv")))}
    per = defaultdict(lambda: defaultdict(dict))  # cell -> iter -> rank -> total_ms
    with open(os.path.join(d, "metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["metric"] == "total_ms" and r["mode"] == "isolated":
                per[r["cell_id"]][int(r["iter"])][int(r["rank"])] = float(r["value_ms"])
    m = json.load(open(os.path.join(d, "manifest.json")))
    libs = {os.path.basename(l["path"]): l["sha256"][:8] for l in m.get("flux_libs", [])}
    binary = f"{libs.get('libflux_cuda_ths_op.so', '?')}/{libs.get('libflux_cuda.so', '?')}"
    return cells, per, binary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="alps")
    ap.add_argument("--capsules", nargs="*", help="explicit run_id prefixes (default: every <platform> capsule)")
    ap.add_argument("--nodes", default="2,4,8,16,32")
    ap.add_argument("--budgets", default="1,64")
    ap.add_argument("--out", default=os.path.join(HERE, "figure_src.csv"))
    args = ap.parse_args()
    nodes = [int(x) for x in args.nodes.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]

    best = {}   # (nodes, system, budget) -> row-ish dict; newest ok capsule wins
    binaries = defaultdict(set)
    for d in capsule_dirs(args.platform, args.capsules):
        cells, per, binary = load_capsule(d)
        cap = os.path.basename(d)
        for cid, c in cells.items():
            sysname = SYSTEM_OF.get(c["variant"])
            if sysname is None or c["mode"] != "isolated":
                continue
            n, b = int(c["nnodes"]), int(c["budget_mib"])
            if n not in nodes or b not in budgets:
                continue
            key = (n, sysname, b)
            ok = c["status"] == "ok" and cid in per and per[cid]
            prev = best.get(key)
            if prev is not None and prev["ok"] and not ok:
                continue   # never let a failed re-run shadow an ok cell
            if prev is not None and prev["ok"] and ok and prev["capsule"] > cap:
                continue   # newest ok capsule wins
            row = dict(ok=ok, capsule=cap, cell_id=cid, status=c["status"], binary=binary,
                       ranks=int(c["world_size"]), tokens_per_rank=int(c["tokens_per_rank"] or 0),
                       deterministic=c.get("deterministic", ""))
            if ok:
                samples = [v for it in per[cid].values() for v in it.values()]
                row["total_ms"] = st.mean(samples)
                row["total_ms_maxrank"] = st.mean(max(it.values()) for it in per[cid].values())
                row["iters"] = len(per[cid])
                if str(c.get("deterministic", "0")) not in ("0", ""):
                    row["status"] = "INVALID_deterministic=" + str(c["deterministic"])
                    row["ok"] = False
            best[key] = row
            binaries[key].add(binary)

    used_binaries = {r["binary"] for r in best.values() if r["ok"]}
    binary_note = ""
    if len(used_binaries) > 1:
        binary_note = "BUILD MISMATCH across capsules: " + " | ".join(sorted(used_binaries))
        print("WARNING:", binary_note, "(SCHEMA rule 4 — the figure must come from ONE binary)")

    out_rows = []
    for b in budgets:
        for n in nodes:
            ref = {s: best.get((n, s, b)) for s in ("nvshmem", "comet", "ours", "dwire")}
            ours_cands = [ref[s] for s in ("ours", "dwire") if ref[s] and ref[s]["ok"]]
            ours_min = min((r["total_ms"] for r in ours_cands), default=None)
            for s in ("ours", "dwire", "nvshmem", "comet"):
                r = ref[s]
                if r is None:
                    if s in ("ours", "nvshmem"):
                        print(f"MISSING: {n}n {s} b{b} — no capsule has this cell")
                    continue
                tpr = r["tokens_per_rank"]
                total_tokens = tpr * r["ranks"]
                o = dict.fromkeys(COLUMNS, "")
                o.update(nodes=n, ranks=r["ranks"], system=s, budget_mib=b, status=r["status"],
                         capsule=r["capsule"], cell_id=r["cell_id"], tokens_per_rank=tpr,
                         total_tokens=total_tokens, binary=r["binary"],
                         source_dataset=os.path.relpath(os.path.join(RUNS, r["capsule"]), ROOT))
                notes = []
                if r["ok"]:
                    o["total_ms"] = f"{r['total_ms']:.3f}"
                    o["total_ms_maxrank"] = f"{r['total_ms_maxrank']:.3f}"
                    o["throughput_mtok_s"] = f"{total_tokens / r['total_ms'] / 1000.0:.4f}"
                    if r["iters"] != 10:
                        notes.append(f"{r['iters']} timed iters")
                    if s == "ours" and ours_min is not None:
                        for refname in ("comet", "nvshmem"):
                            rr = ref[refname]
                            if rr and rr["ok"]:
                                o[f"speedup_vs_{refname}"] = f"{rr['total_ms'] / ours_min:.3f}"
                        if ref["dwire"] and ref["dwire"]["ok"] and ref["dwire"]["total_ms"] < r["total_ms"]:
                            notes.append("Ours = dwire at this node count (min over arms); speedups vs that min")
                else:
                    notes.append(f"cell status {r['status']} — no latency")
                if binary_note:
                    notes.append(binary_note)
                o["note"] = "; ".join(notes)
                out_rows.append(o)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {args.out}: {len(out_rows)} rows from {len({r['capsule'] for r in best.values()})} capsules;"
          f" binaries used: {sorted(used_binaries) or 'none'}")
    for o in out_rows:
        print(f"  {o['nodes']:>2}n {o['system']:8s} b{o['budget_mib']:<3} total {o['total_ms'] or '-':>9}"
              f"  maxrank {o['total_ms_maxrank'] or '-':>9}  sp_vs_ring {o['speedup_vs_nvshmem'] or '-':>6}"
              f"  {o['status']}  {o['capsule']}")


if __name__ == "__main__":
    main()
