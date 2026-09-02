#!/usr/bin/env python3
"""Cycling-ablation dataset builder (2026-09-02, K2 4n b64).

Reads the phase-1 (per-topic reset-proxy) and phase-2 (topic-schedule)
capsules listed in CAPSULES, recomputes per-cell statistics from the raw
metrics.csv (per-iteration MAX across ranks of total_ms), and writes:
  results_tidy.csv  — one row per (capsule, cell[, topic position])
  summary_*.md      — the tables quoted in docs/handoff/34_cycling_ablation.md
Run from the repo root:  python figs/ablation_cycling/build_dataset.py
"""
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "sweeps", "results", "runs")
OUT = os.path.dirname(os.path.abspath(__file__))

ARM = {
    "l01_allgather_dense": ("COMET", 0),
    "l01_slipstream": ("1 token-comm overlap", 1),
    "ablation_l01_pr0_pv2_r2": ("2 placement only (pr0)", 2),
    "ours_l01_s1_pv2_r2": ("1+2 static", 3),
    # phase-1 reset proxies
    "ablation_l01_s2_swapall_noov_p2p_r2": ("1+2 full-orbit swap SEQ d1", 4),
    "ablation_l01_s2_swapall_p2p_r2": ("1+2 full-orbit swap OVL d1", 5),
    "ablation_l01_s2_swapall_rp4_noov_p2p_r2": ("1+2 full-orbit swap SEQ d4", 6),
    "ablation_l01_s2_swapall_rp4_p2p_r2": ("1+2 full-orbit swap OVL d4", 7),
    "ablation_l01_s2_swap_t1_rst_noov_p2p_r2": ("1+2 one-round swap SEQ d1", 8),
    "ablation_l01_s2_swap_t1_rst_p2p_r2": ("1+2 one-round swap OVL d1", 9),
    "ablation_l01_s2_swap_t1_rp4_noov_p2p_r2": ("1+2 one-round swap SEQ d4", 10),
    "ablation_l01_s2_swap_t1_rp4_p2p_r2": ("1+2 one-round swap OVL d4", 11),
    # phase-2 schedule arms (dwell lives in the family)
    "ablation_l01_s2_swapall_nr_noov_p2p_r2": ("1+2 full-orbit swap SEQ", 4),
    "ablation_l01_s2_swapall_nr_p2p_r2": ("1+2 full-orbit swap OVL", 5),
    "ablation_l01_s2_swap_t1_noov_p2p_r2": ("1+2 one-round swap SEQ", 8),
    "ours_l01_s2_swap_p2p_t1_r2": ("1+2 one-round swap OVL", 9),
}
SHORT = {"livecodebench/execution": "lcb", "mmlu/clinical_knowledge": "clinical",
         "mmlu/college_mathematics": "colmath", "mmlu/electrical_engineering": "elecEng",
         "mmlu/high_school_psychology": "hsPsych", "mmlu/high_school_world_history": "hsWorldHist",
         "mmlu/philosophy": "philosophy", "mmlu/professional_law": "proLaw"}


def scenario_of(fp):
    op = fp.get("opool", "")
    n = len([p for p in op.split("+") if p])
    if n == 8:
        return "S-A seen-8"
    if n == 4:
        return "S-B unseen-4"
    if n == 7:
        return "S-C LOO-proLaw-8"
    return f"opool{n}"


def load(cap_dir):
    cells = {r["cell_id"]: r for r in csv.DictReader(open(os.path.join(cap_dir, "cells.csv")))}
    per = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(os.path.join(cap_dir, "metrics.csv"))):
        if r["metric"] == "total_ms":
            per[r["cell_id"]][int(r["iter"])][int(r["rank"])] = float(r["value_ms"])
    m = json.load(open(os.path.join(cap_dir, "manifest.json")))
    libs = {os.path.basename(l["path"]): l["sha256"][:8] for l in m.get("flux_libs", [])}
    binary = f"{libs.get('libflux_cuda_ths_op.so', '?')}/{libs.get('libflux_cuda.so', '?')}"
    rows = []
    for cid, its in per.items():
        c = cells[cid]
        if c["status"] != "ok" or c["variant"] not in ARM:
            continue
        fp = json.loads(c["family_params"])
        mx = [max(its[i].values()) for i in sorted(its)]
        T = len(mx)
        label, order = ARM[c["variant"]]
        base = dict(capsule=os.path.basename(cap_dir), cell_id=cid, variant=c["variant"], arm=label,
                    arm_order=order, scenario=scenario_of(fp), binary=binary, iters=T,
                    eval_pool=SHORT.get(fp.get("pools"), fp.get("pools")))
        if fp.get("sched"):
            sched = fp["sched"].split("+")
            dw = int(fp.get("dwell", 1))
            n = len(sched)
            base.update(protocol="schedule (carried-over placement)", dwell=dw)
            bt = defaultdict(list)
            for i, x in enumerate(mx):
                bt[((i - (T - 1)) // dw) % n].append(x)
            rows.append(dict(base, topic="ALL", mean_ms=st.mean(mx), median_ms=st.median(mx), sd_ms=st.pstdev(mx), n_iters=T))
            for k in sorted(bt):
                rows.append(dict(base, topic=SHORT.get(sched[k], sched[k]), mean_ms=st.mean(bt[k]),
                                 median_ms=st.median(bt[k]), sd_ms=st.pstdev(bt[k]), n_iters=len(bt[k])))
        else:
            dw = 4 if "rp4" in c["variant"] else (1 if ("rst" in c["variant"] or "swapall_p2p" in c["variant"] or "swapall_noov" in c["variant"]) else 0)
            base.update(protocol="reset-proxy (per-topic cell)", dwell=dw)
            rows.append(dict(base, topic=base["eval_pool"], mean_ms=st.mean(mx), median_ms=st.median(mx), sd_ms=st.pstdev(mx), n_iters=T))
    return rows


def main():
    caps = sorted(glob.glob(os.path.join(RUNS, "20260902-*")))
    rows = []
    for c in caps:
        note = open(os.path.join(c, "spec.yaml")).read()
        if "ablcycle" not in note and "ablsched" not in note:
            continue
        if "ablcycle gate" in note:
            continue   # correctness gate (random payload) — not a perf regime
        rows += load(c)
    # the flux-driver cells of the first two schedule capsules ran BEFORE the
    # runner passed the schedule flags to the l01 driver (topic-0 only);
    # their valid twins are the FLUX-ARMS RECOLLECT capsules
    PRE_FIX_FLUX = ("9d6bcf40", "57c8b387")
    rows = [r for r in rows if not (r["capsule"].endswith(PRE_FIX_FLUX) and r["variant"].startswith("l01_"))]
    keys = ["capsule", "scenario", "protocol", "dwell", "arm", "arm_order", "variant", "topic", "eval_pool",
            "mean_ms", "median_ms", "sd_ms", "n_iters", "iters", "binary", "cell_id"]
    with open(os.path.join(OUT, "results_tidy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["scenario"], r["protocol"], r["dwell"], r["arm_order"], r["topic"])):
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]) for k in keys})
    print("rows", len(rows), "->", os.path.join(OUT, "results_tidy.csv"))
    # summary tables: mean over topics (phase 1) / mean over reps of ALL (phase 2)
    lines = []
    for proto in ("reset-proxy (per-topic cell)", "schedule (carried-over placement)"):
        for scen in sorted({r["scenario"] for r in rows}):
            sel = [r for r in rows if r["protocol"] == proto and r["scenario"] == scen]
            if not sel:
                continue
            if proto.startswith("reset"):
                topics = sorted({r["topic"] for r in sel})
                by = defaultdict(lambda: defaultdict(list))
                for r in sel:
                    by[(r["arm_order"], r["arm"])][r["topic"]].append(r["mean_ms"])
                lines.append(f"\n### {scen} — {proto}, mean over timed iters, per topic (n reps in parens)\n")
                lines.append("| arm | " + " | ".join(topics) + " | MEAN |")
                lines.append("|---|" + "---|" * (len(topics) + 1))
                for (o, a) in sorted(by):
                    vals = [st.mean(by[(o, a)][t]) if t in by[(o, a)] else float("nan") for t in topics]
                    nrep = max(len(v) for v in by[(o, a)].values())
                    lines.append(f"| {a} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {st.mean([v for v in vals if v == v]):.2f} ({nrep}) |")
            else:
                for dw in sorted({r["dwell"] for r in sel}):
                    s2 = [r for r in sel if r["dwell"] == dw]
                    positions = [t for t in sorted({r["topic"] for r in s2}) if t != "ALL"]
                    by = defaultdict(lambda: defaultdict(list))
                    for r in s2:
                        by[(r["arm_order"], r["arm"])][r["topic"]].append(r["mean_ms"])
                    lines.append(f"\n### {scen} — {proto}, dwell {dw}: mean over the whole schedule (ALL) and per topic block\n")
                    bym = defaultdict(list)
                    for r in s2:
                        if r["topic"] == "ALL":
                            bym[(r["arm_order"], r["arm"])].append(r["median_ms"])
                    lines.append("| arm | ALL mean (reps) | ALL median | " + " | ".join(positions) + " |")
                    lines.append("|---|---|---|" + "---|" * len(positions))
                    for (o, a) in sorted(by):
                        allv = by[(o, a)]["ALL"]
                        allstr = f"{st.mean(allv):.2f} (n={len(allv)}" + (f", sd {st.pstdev(allv):.2f})" if len(allv) > 1 else ")")
                        medstr = f"{st.mean(bym[(o, a)]):.2f}"
                        lines.append(f"| {a} | {allstr} | {medstr} | " + " | ".join(f"{st.mean(by[(o, a)][t]):.2f}" if t in by[(o, a)] else "" for t in positions) + " |")
    with open(os.path.join(OUT, "summary_tables.md"), "w") as f:
        f.write("# Cycling-ablation summary tables (generated by build_dataset.py)\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
