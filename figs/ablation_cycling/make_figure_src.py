#!/usr/bin/env python3
"""Figure-source dataset for the two-panel ablation (decision 2026-09-02).

Reads results_tidy.csv (build_dataset.py) and the S-C dwell-4 capsules'
raw metrics, writes:
  figure_src.csv              — one row per (panel, arm, statistic): value, sd, n
  sc_d4_iteration_series.csv  — per-iteration max-rank total_ms of every arm on
                                the S-C dwell-4 schedule, mean over the 3 reps
                                (the professional-law block is iterations 27–30)
Panel A = S-A seen-8, per-topic reset-proxy cells, mean over the 8 topics.
Panel B = S-C LOO-proLaw, dwell-4 topic schedule with carried-over placement,
          whole-schedule mean/median and the professional-law block, 3 reps.
Reference = S-B unseen-4 dwell-4 schedule (rejected; kept for the record).
"""
import csv
import glob
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "sweeps", "results", "runs")

ARMS = [  # (arm label in results_tidy, figure label, rung)
    ("COMET", "COMET", "COMET"),
    ("1 token-comm overlap", "1: token-comm overlap", "1"),
    ("2 placement only (pr0)", "2: placement + routing", "2"),
    ("1+2 static", "1+2 (no expert movement)", "1+2"),
    ("1+2 full-orbit swap SEQ", "1+2 + expert swap, sequential", "1+2+swap seq"),
    ("1+2 full-orbit swap OVL", "1+2 + expert swap, overlapped", "1+2+swap ovl"),
    ("1+2 one-round swap SEQ", "1+2 + one-round swap, sequential", "1+2+swap1 seq"),
    ("1+2 one-round swap OVL", "1+2 + one-round swap, overlapped", "1+2+swap1 ovl"),
    ("1+2 full-orbit swap SEQ d4", "1+2 + expert swap, sequential (reset d4)", "1+2+swap seq"),
    ("1+2 full-orbit swap OVL d4", "1+2 + expert swap, overlapped (reset d4)", "1+2+swap ovl"),
    ("1+2 one-round swap SEQ d4", "1+2 + one-round swap, sequential (reset d4)", "1+2+swap1 seq"),
    ("1+2 one-round swap OVL d4", "1+2 + one-round swap, overlapped (reset d4)", "1+2+swap1 ovl"),
]
LABEL = {a: (l, r) for a, l, r in ARMS}


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results_tidy.csv"))))
    out = []

    def emit(panel, arm, stat, vals, caps, topic="", note=""):
        if not vals:
            return
        lab, rung = LABEL.get(arm, (arm, ""))
        out.append(dict(panel=panel, rung=rung, arm=lab, tidy_arm=arm, topic=topic, statistic=stat,
                        value_ms=f"{st.mean(vals):.3f}", sd_ms=f"{st.pstdev(vals):.3f}" if len(vals) > 1 else "",
                        n=len(vals), capsules="+".join(sorted({c[-8:] for c in caps})), note=note))

    # ---- Panel A: S-A per-topic reset-proxy cells ----
    sa = [r for r in rows if r["scenario"] == "S-A seen-8" and r["protocol"].startswith("reset")]
    topics = sorted({r["topic"] for r in sa})
    for arm in [a for a, _, _ in ARMS]:
        sel = [r for r in sa if r["arm"] == arm]
        if not sel:
            continue
        per_topic = {}
        for t in topics:
            v = [float(r["mean_ms"]) for r in sel if r["topic"] == t]
            if v:
                per_topic[t] = st.mean(v)
                emit("A: seen basis (S-A per-topic)", arm, "mean over 16 timed iters", v,
                     [r["capsule"] for r in sel if r["topic"] == t], topic=t)
        emit("A: seen basis (S-A per-topic)", arm, "mean over the 8 topics", list(per_topic.values()),
             [r["capsule"] for r in sel], topic="ALL", note="value = mean of per-topic means; sd = spread over topics")

    # ---- Panel B: S-C dwell-4 schedule ----
    for scen, panel in (("S-C LOO-proLaw-8", "B: drift schedule (S-C dwell 4, placement carried over)"),
                        ("S-B unseen-4", "reference: S-B dwell 4 (rejected: placement arms sink to COMET)")):
        sc = [r for r in rows if r["scenario"] == scen and r["protocol"].startswith("schedule") and r["dwell"] == "4"]
        for arm in [a for a, _, _ in ARMS]:
            sel = [r for r in sc if r["arm"] == arm]
            if not sel:
                continue
            allr = [r for r in sel if r["topic"] == "ALL"]
            emit(panel, arm, "whole-schedule mean, mean over reps", [float(r["mean_ms"]) for r in allr],
                 [r["capsule"] for r in allr], topic="ALL")
            emit(panel, arm, "whole-schedule median, mean over reps", [float(r["median_ms"]) for r in allr],
                 [r["capsule"] for r in allr], topic="ALL")
            for t in sorted({r["topic"] for r in sel if r["topic"] != "ALL"}):
                bl = [r for r in sel if r["topic"] == t]
                emit(panel, arm, "topic-block mean, mean over reps", [float(r["mean_ms"]) for r in bl],
                     [r["capsule"] for r in bl], topic=t)

    keys = ["panel", "rung", "arm", "tidy_arm", "topic", "statistic", "value_ms", "sd_ms", "n", "capsules", "note"]
    with open(os.path.join(HERE, "figure_src.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print("figure_src.csv rows:", len(out))

    # ---- S-C d4 per-iteration series, mean over reps ----
    caps = sorted({r["capsule"] for r in rows if r["scenario"] == "S-C LOO-proLaw-8"
                   and r["protocol"].startswith("schedule") and r["dwell"] == "4"})
    series = defaultdict(lambda: defaultdict(list))   # arm -> iter -> [values over reps]
    for cap in caps:
        d = os.path.join(RUNS, cap)
        cells = {r["cell_id"]: r for r in csv.DictReader(open(os.path.join(d, "cells.csv")))}
        per = defaultdict(lambda: defaultdict(dict))
        for r in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
            if r["metric"] == "total_ms":
                per[r["cell_id"]][int(r["iter"])][int(r["rank"])] = float(r["value_ms"])
        for cid, its in per.items():
            v = cells[cid]["variant"]
            arm = {"l01_allgather_dense": "COMET", "l01_slipstream": "1 token-comm overlap",
                   "ablation_l01_pr0_pv2_r2": "2 placement only (pr0)", "ours_l01_s1_pv2_r2": "1+2 static",
                   "ablation_l01_s2_swapall_nr_noov_p2p_r2": "1+2 full-orbit swap SEQ",
                   "ablation_l01_s2_swapall_nr_p2p_r2": "1+2 full-orbit swap OVL",
                   "ablation_l01_s2_swap_t1_noov_p2p_r2": "1+2 one-round swap SEQ",
                   "ours_l01_s2_swap_p2p_t1_r2": "1+2 one-round swap OVL"}.get(v)
            if arm is None or cells[cid]["status"] != "ok":
                continue
            for i in sorted(its):
                series[arm][i].append(max(its[i].values()))
    sched = ["lcb", "clinical", "colmath", "elecEng", "hsPsych", "hsWorldHist", "philosophy", "proLaw"]
    with open(os.path.join(HERE, "sc_d4_iteration_series.csv"), "w", newline="") as f:
        w = csv.writer(f)
        arms = [a for a, _, _ in ARMS if a in series]
        w.writerow(["iter", "topic_block"] + [LABEL[a][0] for a in arms] + [f"sd:{LABEL[a][0]}" for a in arms])
        T = 32
        for i in range(T):
            k = ((i - (T - 1)) // 4) % 8
            w.writerow([i, sched[k]] + [f"{st.mean(series[a][i]):.2f}" for a in arms]
                       + [f"{st.pstdev(series[a][i]):.2f}" for a in arms])
    print("sc_d4_iteration_series.csv written from", [c[-8:] for c in caps])


if __name__ == "__main__":
    main()
