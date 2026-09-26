#!/usr/bin/env python3
"""Build figs/main_perf_v5/figure_src.csv (2026-09-26).

v5 = v4's baseline rows UNCHANGED + the Ours rows for the plotted budgets (b1 / b4 / b16) re-measured
from the open-source repository ($PSCRATCH/workspace/andrewy/moe_ep, final binary e252095: the
extracted system with the workspace-kernel race fixed) via `scripts/reproduce.sh`, i.e. the
same protocol (isolated mode, 5 warm-up + 10 timed iterations, per-iteration MAX across ranks,
MEDIAN over iterations, bit-identical trace batches). v4's Ours rows at the non-plotted budgets
(b2, b64) are carried over unchanged and flagged. When a cell has several same-binary
measurements (the grid value plus deliberate re-runs), the row is the MEDIAN of them and the
flag lists the count.

Ours row mapping (open-source knobs -> figure row_id):
  comm_strategy=overlap, swap=off -> ours12          ("1+2")
  comm_strategy=overlap, swap=on  -> ours12_dispatch ("1+2 + expert swap (3D-scheduled, band trigger)")
  comm_strategy=direct            -> ours2_direct    ("2Ours (expert balance + Routing + direct a2av)")
"""
import collections
import csv
import glob
import os
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
V4 = os.path.join(REPO, "figs", "main_perf_v4", "figure_src.csv")
OSS = os.path.join(os.environ["PSCRATCH"], "workspace", "andrewy", "moe_ep")
MEASURED = os.path.join(OSS, "results", "measured")
OSS_SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=OSS, capture_output=True, text=True).stdout.strip()
PLOTTED = (1, 4, 16)
MODEL = {"qwen3": "Qwen", "k2": "K2"}
ROW = {("overlap", False): ("ours12", "1+2"),
       ("overlap", True): ("ours12_dispatch", "1+2 + expert swap (3D-scheduled, band trigger)"),
       ("direct", False): ("ours2_direct", "2Ours (expert balance + Routing + direct a2av)")}


def cell_medians(path):
    """(nodes, model, budget, strategy, swap) -> [iter_max_median of each measurement in the file].
    A re-run appends another block of iterations to the same cell; blocks are separated by the
    (rank 0, iteration 0) row that opens each measurement."""
    per = collections.defaultdict(list)      # key -> list of {iter: max over ranks}
    for r in csv.DictReader(open(path)):
        if r["metric"] != "total_ms":
            continue
        key = (int(r["nodes"]), r["model"], int(float(r["budget_mib"])), r["comm_strategy"], r["swap"] in ("1", "True", "on"))
        it, rank = int(r["iter"]), int(r["rank"])
        if rank == 0 and it == 0 or not per[key]:
            per[key].append({})
        d = per[key][-1]
        d[it] = max(d.get(it, 0.0), float(r["value_ms"]))
    return {key: [st.median(d.values()) for d in blocks] for key, blocks in per.items()}


meas = collections.defaultdict(list)
files = sorted(glob.glob(os.path.join(MEASURED, "main_perf_*n.csv"))) + sorted(glob.glob(os.path.join(MEASURED, "rerun_*.csv")))
for p in files:
    for key, vals in cell_medians(p).items():
        meas[key].extend(vals)
assert meas, f"no measured CSVs under {MEASURED}"

with open(V4, newline="") as f:
    hdr = next(csv.reader(f))
out = []
with open(V4, newline="") as f:
    for r in csv.DictReader(f):
        if not r["row_id"].startswith("ours"):
            out.append(r)                                        # baselines verbatim
        elif int(r["budget_mib"]) not in PLOTTED:
            r = dict(r); r["flag"] = (r["flag"] + ";" if r["flag"] else "") + "carried_from_v4"
            out.append(r)                                        # b2 / b64 Ours rows: unchanged
n_new = 0
for (nodes, model, b, strat, swap), vals in sorted(meas.items()):
    if b not in PLOTTED:
        continue
    row_id, label = ROW[(strat, swap)]
    t = st.median(vals)
    flag = f"open_source_build_{OSS_SHA}"
    if len(vals) > 1:
        flag += f"_median_of_{len(vals)}"
    out.append({"nodes": str(nodes), "model": MODEL[model], "row_id": row_id, "row_label": label, "budget_mib": str(b),
                "total_ms": f"{t:.3f}", "recorded_ms": "", "stat": "iter_max_median",
                "arm_variant": f"{strat}{'+swap' if swap else ''}", "impl": "moe_ep",
                "capsule": f"results/measured (moe_ep {OSS_SHA})", "cell_id": f"{model}_b{b}_{strat}_swap{int(swap)}_{nodes}n",
                "branch": "moe_ep", "git_sha": OSS_SHA, "flag": flag})
    n_new += 1
out.sort(key=lambda r: (int(r["nodes"]), r["model"], r["row_id"], int(r["budget_mib"])))
with open(os.path.join(HERE, "figure_src.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out)
print(f"figure_src.csv: {len(out)} rows, {n_new} Ours rows from the open-source build {OSS_SHA}")
