#!/usr/bin/env python3
"""Build figs/main_perf_v4/figure_src.csv (2026-09-25).

v4 = v3's baseline rows UNCHANGED + the Ours rows replaced by the FINAL candidate campaign on the
consolidated implementation (handoff 42 §12): pv3c router (C = 1/4 at <= 8n, 1/2 at 16n), minimal-move
relay partition, per-round staging + P2P pulls, band swap trigger, CUDA_DEVICE_MAX_CONNECTIONS=24
(direct wire keeps its canonical 8). The plotted Ours bar = min over the three candidate rows
(make_figure.py OURS_CANDIDATES). Statistic: per-iteration MAX across ranks, MEDIAN over the 10 timed
iterations (iter_max_median), recomputed from each capsule's metrics.csv.
"""
import csv, os, re, subprocess, statistics as st, collections
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
V3 = os.path.join(REPO, "figs", "main_perf_v3", "figure_src.csv")
SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
# candidate campaign capsules (final3_*): direct / fused / fused + 3D band swap, b1/b4/b16
FINAL3 = {4: ["20260925-200150_perlmutter_d35537f2", "20260925-200638_perlmutter_bab63c3c"],
          8: ["20260925-203721_perlmutter_e8536692", "20260925-204147_perlmutter_76f9e262"],
          16: ["20260925-204837_perlmutter_ac1689a0", "20260925-205308_perlmutter_09e338eb"]}
# b64 rows: the fused arm on the same implementation + pin, same day (handoff 42 §11 "fused c24")
B64 = {4: ["20260925-151206_perlmutter_" , ], }  # filled below from the run logs
ROW = {"dwire": ("ours2_direct", "2Ours (expert balance + Routing + direct a2av)"),
       "swapall": ("ours12_dispatch", "1+2 + expert swap (3D-scheduled, band trigger)"),
       "s1": ("ours12", "1+2")}
def row_of(cid):
    if "dwire" in cid: return ROW["dwire"]
    if "swapall" in cid: return ROW["swapall"]
    return ROW["s1"]
def cells(rid):
    p = os.path.join(REPO, "sweeps", "results", "runs", rid, "metrics.csv")
    per = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in csv.DictReader(open(p)):
        if r["metric"] == "total_ms": per[r["cell_id"]][int(r["iter"])] = max(per[r["cell_id"]][int(r["iter"])], float(r["value_ms"]))
    return {cid: st.median(d.values()) for cid, d in per.items()}
out = []
with open(V3, newline="") as f:
    hdr = next(csv.reader(f))
with open(V3, newline="") as f:
    for r in csv.DictReader(f):
        if not r["row_id"].startswith("ours"): out.append(r)          # baselines verbatim (incl. comet_eplb, comet_v3anchor)
def add(nodes, rid, flag, only=None):
    for cid, t in cells(rid).items():
        model = "K2" if "610042" in cid else "Qwen"; b = int(re.search(r"_b(\d+)_", cid).group(1))
        if only and b not in only: continue
        row_id, label = row_of(cid); arm = cid.split("_trace")[0]
        out.append({"nodes": str(nodes), "model": model, "row_id": row_id, "row_label": label, "budget_mib": str(b),
                    "total_ms": f"{t:.3f}", "recorded_ms": "", "stat": "iter_max_median", "arm_variant": arm, "impl": "ours",
                    "capsule": rid, "cell_id": cid, "branch": "pv3", "git_sha": SHA, "flag": flag})
for nodes, rids in FINAL3.items():
    for rid in rids: add(nodes, rid, "final_candidate_campaign_20260925")
# b64 fused rows (ours12 only): from the same-day conn=24 capsules listed in logs/pv3 (c24v / cmid / pin24)
import glob
L = os.environ.get("PSCRATCH", "") + "/workspace/andrewy/logs/pv3"
b64src = {4: glob.glob(f"{L}/run_c24v_c24v_4n_*_c24.log"), 8: glob.glob(f"{L}/run_cmid_cmid_8n_*_c24.log"), 16: glob.glob(f"{L}/run_pin24_pin24_16n_*.log")}
seen = set()
for nodes, logs in b64src.items():
    for lg in logs:
        for line in open(lg):
            m = re.search(r"run_id: (\S+)", line)
            if not m or m.group(1) in seen: continue
            seen.add(m.group(1))
            if os.path.exists(os.path.join(REPO, "sweeps", "results", "runs", m.group(1), "metrics.csv")):
                add(nodes, m.group(1), "b64_fused_conn24_20260925_not_plotted", only=(64,))
# de-duplicate b64 (several reps): keep the mean over reps per (nodes, model)
b64 = collections.defaultdict(list); rest = []
for r in out:
    if r["row_id"] == "ours12" and r["budget_mib"] == "64" and r["flag"].startswith("b64_fused"): b64[(r["nodes"], r["model"])].append(r)
    else: rest.append(r)
for k, rs in b64.items():
    r = dict(rs[0]); r["total_ms"] = f"{st.mean(float(x['total_ms']) for x in rs):.3f}"; r["capsule"] = ";".join(x["capsule"] for x in rs); r["cell_id"] = rs[0]["cell_id"]; r["flag"] += f"_mean_of_{len(rs)}_reps"; rest.append(r)
out = rest
out.sort(key=lambda r: (int(r["nodes"]), r["model"], r["row_id"], int(r["budget_mib"])))
with open(os.path.join(HERE, "figure_src.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(out)
n_ours = sum(1 for r in out if r["row_id"].startswith("ours")); print(f"figure_src.csv: {len(out)} rows, {n_ours} ours rows, sha {SHA}")
