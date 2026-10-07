#!/usr/bin/env python3
"""Build figs/main_perf_v6/figure_src.csv (2026-10-07).

v6 = v5's baseline rows UNCHANGED (incl. the COMET+EPLB rows `comet_eplb`, now plotted) + the Ours rows for the
plotted budgets (b1 / b4 / b16) re-measured with `gpu-plan` ON (the layer planned and issued on the GPU: device
router, swap decision, metadata, kernel-issued dispatch and combine wire, CUDA layer graphs, device dual3 swap
lane) from the Zepp release candidate (clone $PSCRATCH/workspace/andrewy/zepp_gpuplan, branch gpuplan d0ab91f,
libzepp_cuda.so sha256 beb332a88464f324...) via `scripts/reproduce.sh --gpu-plan 1`: the same protocol as v5
(isolated mode, 5 warm-up + 10 timed iterations, per-iteration MAX across ranks, MEDIAN over iterations,
bit-identical trace batches). The measurements are archived here (gpuplan_measured/on, gzipped long-format CSVs);
gpuplan_measured/off holds the same-day gpu-plan OFF grids (not plotted). v5's Ours rows at the non-plotted
budgets (b2, b64) are carried over and flagged.

Ours row mapping (release knobs -> figure row_id), as in v5:
  comm_strategy=overlap, swap=off -> ours12
  comm_strategy=overlap, swap=on  -> ours12_dispatch
  comm_strategy=direct            -> ours2_direct      (gpu-plan does not apply; host-planned)
"""
import collections
import csv
import glob
import gzip
import io
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
V5 = os.path.join(REPO, "figs", "main_perf_v5", "figure_src.csv")
MEASURED = os.path.join(HERE, "gpuplan_measured", "on")
BUILD = "zepp_gpuplan d0ab91f (libzepp_cuda.so beb332a88464f324)"
PLOTTED = (1, 4, 16)
MODEL = {"qwen3": "Qwen", "k2": "K2"}
ROW = {("overlap", False): ("ours12", "1+2"),
       ("overlap", True): ("ours12_dispatch", "1+2 + expert swap (3D-scheduled, band trigger)"),
       ("direct", False): ("ours2_direct", "2Ours (expert balance + Routing + direct a2av)")}


def cell_medians(path):
    """(nodes, model, budget, strategy, swap) -> median over iterations of the max over ranks of total_ms."""
    per = collections.defaultdict(dict)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            if r["metric"] != "total_ms":
                continue
            key = (int(r["nodes"]), r["model"], int(float(r["budget_mib"])), r["comm_strategy"],
                   r["swap"] in ("1", "True", "on"))
            it = int(r["iter"])
            per[key][it] = max(per[key].get(it, 0.0), float(r["value_ms"]))
    return {key: st.median(d.values()) for key, d in per.items()}


meas = {}
for p in sorted(glob.glob(os.path.join(MEASURED, "main_perf_*n.csv.gz"))):
    meas.update(cell_medians(p))
assert len(meas) == 52, f"expected 18+18+16 cells, got {len(meas)}"

with open(V5, newline="") as f:
    hdr = next(csv.reader(f))
out = []
with open(V5, newline="") as f:
    for r in csv.DictReader(f):
        if not r["row_id"].startswith("ours"):
            out.append(r)                                        # baselines verbatim (comet_eplb included)
        elif int(r["budget_mib"]) not in PLOTTED:
            r = dict(r)
            r["flag"] = (r["flag"] + ";" if r["flag"] else "") + "carried_from_v5"
            out.append(r)                                        # b2 / b64 Ours rows: unchanged
for (nodes, model, b, strat, swap), t in sorted(meas.items()):
    row_id, label = ROW[(strat, swap)]
    out.append({"nodes": str(nodes), "model": MODEL[model], "row_id": row_id, "row_label": label,
                "budget_mib": str(b), "total_ms": f"{t:.3f}", "recorded_ms": "", "stat": "iter_max_median",
                "arm_variant": f"{strat}{'+swap' if swap else ''}{'+gpu_plan' if strat == 'overlap' else ''}",
                "impl": "zepp", "capsule": f"gpuplan_measured/on/main_perf_{nodes}n.csv.gz",
                "cell_id": f"{model}_b{b}_{strat}_swap{int(swap)}_{nodes}n", "branch": "gpuplan",
                "git_sha": "d0ab91f", "flag": f"gpu_plan_build {BUILD}" if strat == "overlap" else f"build {BUILD}"})
out.sort(key=lambda r: (int(r["nodes"]), r["model"], r["row_id"], int(r["budget_mib"])))
with open(os.path.join(HERE, "figure_src.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    w.writerows(out)
print(f"figure_src.csv: {len(out)} rows, {len(meas)} Ours rows from {BUILD}")
