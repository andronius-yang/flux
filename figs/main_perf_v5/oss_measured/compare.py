"""Compare a benchmark CSV (bench/replay.py --out) against results/expected_main_perf.csv.

  python results/compare.py results/2026-09-26/main_perf.csv [--tolerance 0.05]

Prints one line per cell: measured, expected, delta, and whether the cell is within tolerance
of the expected value and below the reference ceiling (the best published alternative)."""
import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_expected(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["nodes"]), r["model"], float(r["budget_mib"]), r["comm_strategy"], r["swap"] == "on")
            out[key] = (float(r["total_ms"]), float(r["plotted_min_ms"]),
                        float(r["reference_ceiling_ms"]) if r["reference_ceiling_ms"] else None)
    return out


def load_measured(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["metric"] != "total_ms_iter_max_median":
                continue
            key = (int(r["nodes"]), r["model"], float(r["budget_mib"]), r["comm_strategy"], r["swap"] in ("1", "True", "on"))
            out[key] = float(r["value_ms"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measured")
    ap.add_argument("--expected", default=os.path.join(HERE, "expected_main_perf.csv"))
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args()
    exp = load_expected(a.expected)
    got = load_measured(a.measured)
    print(f"{'nodes':>5} {'model':>6} {'MiB':>5} {'strategy':>8} {'swap':>4} {'measured':>9} {'expected':>9} {'delta':>7}  verdict")
    n_ok = n_all = 0
    mins = defaultdict(lambda: float("inf"))
    for key in sorted(got):
        nodes, model, b, strat, swap = key
        m = got[key]
        mins[(nodes, model, b)] = min(mins[(nodes, model, b)], m)
        if key not in exp:
            print(f"{nodes:>5} {model:>6} {b:>5g} {strat:>8} {'on' if swap else 'off':>4} {m:>9.3f} {'-':>9} {'-':>7}  (no expected value)")
            continue
        e, _pmin, _ceil = exp[key]
        d = (m - e) / e
        ok = abs(d) <= a.tolerance
        n_all += 1
        n_ok += ok
        print(f"{nodes:>5} {model:>6} {b:>5g} {strat:>8} {'on' if swap else 'off':>4} {m:>9.3f} {e:>9.3f} {d:>+6.1%}  {'ok' if ok else 'OUT'}")
    print(f"\ncells within {a.tolerance:.0%}: {n_ok}/{n_all}")
    for (nodes, model, b), m in sorted(mins.items()):
        ceil = next((v[2] for k, v in exp.items() if k[:3] == (nodes, model, b) and v[2]), None)
        pmin = next((v[1] for k, v in exp.items() if k[:3] == (nodes, model, b)), None)
        if ceil is not None:
            print(f"{nodes:>5} {model:>6} {b:>5g}  best {m:.3f} ms  vs published best {pmin:.3f}  vs reference ceiling {ceil:.3f}  "
                  f"{'below' if m < ceil else 'NOT below'}")


if __name__ == "__main__":
    main()
