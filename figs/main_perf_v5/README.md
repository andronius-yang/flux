# main_perf_v5 — Ours rows re-measured from the open-source repository (2026-09-26)

**Status: ADDITION to v4 (never an overwrite).** v5 = v4 with ONE change: the Ours rows at the
plotted budgets (b1 / b4 / b16) are re-measured from the extracted open-source repository
(`$PSCRATCH/workspace/andrewy/moe_ep`, final binary = build of e252095: the same reconciled
mechanisms with the dispatch workspace-kernel race fixed; results committed there under
`results/measured/`). Baseline rows are v4's, unchanged; v4's Ours rows at the non-plotted budgets
(b2, b64) are carried over and flagged `carried_from_v4`. Same protocol as v4: `scripts/reproduce.sh`
(isolated mode, 5 warm-up + 10 timed iterations, per-iteration MAX across ranks, MEDIAN over
iterations), bit-identical trace batches. Where a cell has several same-binary measurements (the grid
value plus deliberate re-runs) the row is their MEDIAN (`_median_of_N` flag).

Ours row mapping (open-source knob -> row_id): overlap -> `ours12`; overlap + swap -> `ours12_dispatch`;
direct -> `ours2_direct`. The plotted Ours bar = min of the three (make_figure.py OURS_CANDIDATES).
16n direct b16 is absent as in v4 (symmetric-heap wall of the direct arm; skipped by the script).

Plotted Ours (min of three) v4 -> v5: within +-3 % in 15 of 18 cells; 8n Qwen b1 -9.6 % (the fused
arm now beats v4's swap-arm min), 16n K2 b4 +4.7 %, 8n K2 b4 +2.9 %. The swap row itself is 5-12 %
faster than v4 everywhere (the band-first decision costs ~0.2 ms instead of ~0.5-0.7 ms of host
time; movement and schedule unchanged, paper §4.3 semantics unchanged) but rarely is the min.

Files: `build_figure_src_v5.py` (from v4's figure_src.csv + the open-source CSVs), `figure_src.csv`,
`make_figure.py` (v4's, output names only), `main_perf_v5.{pdf,png}`, `_samebinary` variant as in v4.

## oss_measured/ (added 2026-09-28)

The open-source repository's `results/` directory as it stood at the rename to LoPEP (build e252095):
`expected_main_perf.csv` (published values + reference ceilings), `measured/main_perf_{4,8,16}n.csv`
(long-format per-rank per-iteration metrics), the compare printouts, the re-run note, `main_perf.png`,
and the `compare.py` / `plot.py` helpers. These files were REMOVED from the public repository on
2026-09-28 (rule: the public tree carries no latency data); this directory is their only copy, and
`build_figure_src_v5.py` reads them from here.
