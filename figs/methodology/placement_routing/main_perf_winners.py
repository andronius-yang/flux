"""Which OURS arm the main-perf figure actually plots, per group.

Reads figs/main_perf/figure_src.csv (the committed data authority of the
main performance figure) and writes main_perf_winners.csv: for every
(nodes, model, budget) group the OURS candidate with the minimum
total_ms — i.e. the arm the figure's "Ours" bar shows (SPEC §2.2
best-of rule) — plus the runner-up and the margin. The methodology
figures must describe the winners' mechanisms (directive 2026-09-05).

Usage: python figs/methodology/placement_routing/main_perf_winners.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "main_perf", "figure_src.csv")
OUT = os.path.join(HERE, "main_perf_winners.csv")
PLOTTED = (1, 4, 16)  # main_perf SPEC §1: 1 / 4 / 16 MiB groups


def main():
    rows = list(csv.DictReader(open(SRC)))
    groups = {}
    for r in rows:
        if not r["row_id"].startswith("ours"):
            continue
        k = (int(r["nodes"]), r["model"], int(r["budget_mib"]))
        groups.setdefault(k, []).append(
            (float(r["total_ms"]), r["row_id"], r["arm_variant"], r["capsule"]))
    out = []
    for k in sorted(groups):
        v = sorted(groups[k])
        win, second = v[0], v[1]
        out.append(dict(
            nodes=k[0], model=k[1], budget_mib=k[2],
            plotted=int(k[2] in PLOTTED),
            winner_row_id=win[1], winner_arm=win[2],
            winner_total_ms=f"{win[0]:.3f}", winner_capsule=win[3],
            runner_up_row_id=second[1], runner_up_arm=second[2],
            runner_up_total_ms=f"{second[0]:.3f}",
            margin_pct=f"{100.0 * (second[0] - win[0]) / second[0]:.1f}",
            n_candidates=len(v)))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    tally = {}
    for o in out:
        if o["plotted"]:
            tally[o["winner_arm"]] = tally.get(o["winner_arm"], 0) + 1
    print(f"wrote {OUT} ({len(out)} groups)")
    print("plotted-group winners:", tally)


if __name__ == "__main__":
    main()
