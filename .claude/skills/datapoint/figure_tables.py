#!/usr/bin/env python3
"""Generate the figure-shaped table CSV from a campaign tidy CSV.

Usage: python3 figure_tables.py <tidy.csv> <out.csv>

Layout (matches the intended 2x3 figure: rows = models, cols = topology,
each cell = grouped stacked bars over budgets):
- Block order topology-major: Qwen 4n, K2 4n, Qwen 8n, K2 8n, Qwen 16n, K2 16n.
- One row per figure arm in the fixed order below (arms mapped to tidy
  baselines; unmapped arms emit fully-empty rows — e.g. FAST and the
  ours-combined arms until a campaign measures them).
- Column groups left->right: TOTAL (plan-inclusive total_ms, the headline),
  PLAN (plan_ms+plan_comm_ms), LAYER0 (l0_ms+act_ms), LAYER1 (l1_ms), with a
  blank spacer column between groups. Segments stack to ~total (<1%).
- Empty non-total cells inside a mapped arm = pre-skipped OOM-class cells.
"""
import csv, sys

BUDS = [1, 2, 4, 8, 16, 32, 64]
ORDER = [("Torch + GEMM", "Torch+GEMM"),
         ("FAST + GEMM", None),               # broken/excluded as of 8/24
         ("Comet", "COMET"),
         ("MoonEP", "MoonEP"),
         ("EPLB", "EPLB"),
         ("EPIC", "EPIC"),
         ("1 Ours(Token Comm)", "Slipstream"),
         ("2 Ours(expert balance + Routing)", "PLL"),
         ("1+2", None),                       # combined arm: not yet measured
         ("1+2 + expert dispatch overlap", None)]
BLOCKS = [("Qwen", "Qwen 3", 4), ("K2", "Kimi K2", 4),
          ("Qwen", "Qwen 3", 8), ("K2", "Kimi K2", 8),
          ("Qwen", "Qwen 3", 16), ("K2", "Kimi K2", 16)]
METRICS = ["total", "plan", "layer0", "layer1"]

def main(tidy_path, out_path):
    rows = list(csv.DictReader(open(tidy_path)))
    def cell(model, n, base, b):
        for r in rows:
            if (r["model"] == model and r["nodes"] == str(n)
                    and r["baseline"] == base and r["budget_mib"] == str(b)):
                if r["status"] != "ok":
                    return None
                f = lambda k: float(r[k]) if r[k] else 0.0
                return dict(total=f("total_ms"),
                            plan=round(f("plan_ms") + f("plan_comm_ms"), 3),
                            layer0=round(f("l0_ms") + f("act_ms"), 3),
                            layer1=f("l1_ms"))
        return None
    out = open(out_path, "w")
    out.write("# Figure tables generated from %s\n" % tidy_path)
    out.write("# See figure_tables.py docstring for layout/conventions.\n")
    hdr = ["arm"]
    for m in METRICS:
        hdr += ["%s_b%d" % (m, b) for b in BUDS] + [""]
    hdr = hdr[:-1]
    for model, label, n in BLOCKS:
        out.write("\n# === %s | %dn ===\n" % (label, n))
        out.write(",".join(hdr) + "\n")
        for fig, base in ORDER:
            vals = [fig]
            for m in METRICS:
                for b in BUDS:
                    c = cell(model, n, base, b) if base else None
                    vals.append("" if c is None else "%g" % c[m])
                vals.append("")
            out.write(",".join(vals[:-1]) + "\n")
    out.close()
    print("wrote", out_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
