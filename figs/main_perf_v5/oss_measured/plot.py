"""Bar chart of measured main-performance cells (bench/replay.py --out CSVs) next to the
published numbers in results/expected_main_perf.csv.

  python results/plot.py results/measured/main_perf_4n.csv [more csvs ...] [--out main_perf.png]

One panel per model; x = (nodes, budget); bars = the three settings (overlap, overlap + swap,
direct); a marker on each group = the published best of the three."""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [("overlap", False, "overlap"), ("overlap", True, "overlap + swap"), ("direct", False, "direct")]


def load_measured(paths):
    """Median over iterations of the max over ranks of total_ms, per cell."""
    per_iter = defaultdict(lambda: defaultdict(float))
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["metric"] != "total_ms":
                    continue
                key = (int(r["nodes"]), r["model"], float(r["budget_mib"]), r["comm_strategy"], r["swap"] in ("1", "True", "on"))
                it = int(r["iter"])
                per_iter[key][it] = max(per_iter[key][it], float(r["value_ms"]))
    out = {}
    for key, its in per_iter.items():
        v = sorted(its.values())
        out[key] = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    return out


def load_expected(path):
    best = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            best[(int(r["nodes"]), r["model"], float(r["budget_mib"]))] = float(r["plotted_min_ms"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--expected", default=os.path.join(HERE, "expected_main_perf.csv"))
    ap.add_argument("--out", default="main_perf.png")
    a = ap.parse_args()
    got = load_measured(a.csvs)
    exp = load_expected(a.expected)
    models = sorted({k[1] for k in got})
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 3.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        groups = sorted({(k[0], k[2]) for k in got if k[1] == model})
        width = 0.26
        for i, (strat, swap, label) in enumerate(SETTINGS):
            ys = [got.get((n, model, b, strat, swap), float("nan")) for n, b in groups]
            ax.bar([g + (i - 1) * width for g in range(len(groups))], ys, width, label=label)
        pub = [exp.get((n, model, b), float("nan")) for n, b in groups]
        ax.plot(range(len(groups)), pub, "k_", markersize=18, markeredgewidth=2, label="published best")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{n}n\n{b:g} MiB" for n, b in groups])
        ax.set_ylabel("ms per layer step")
        ax.set_title({"qwen3": "Qwen3-235B", "k2": "Kimi-K2"}.get(model, model))
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
