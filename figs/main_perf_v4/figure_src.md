# main_perf_v4 `figure_src.csv` provenance

- Baseline rows (`fast_gemm`, `nvshmem_gemm`, `moonep`, `eplb`, `epic`, `comet`, `comet_eplb`,
  `comet_v3anchor`, all budgets 1/2/4/16/64): copied verbatim from `figs/main_perf_v3/figure_src.csv`
  (see that file's `figure_src.md` for their capsules and the 9/1–9/17 ledgers).
- Ours rows: rebuilt by `build_figure_src_v4.py` from the 2026-09-25 candidate campaign capsules
  (`flag = final_candidate_campaign_20260925`), `total_ms` = per-iteration MAX across ranks, MEDIAN
  over the 10 timed iterations, recomputed from each capsule's `metrics.csv`; `git_sha` = the tree at
  build time (pv3). b64 `ours12` rows: `flag = b64_fused_conn24_20260925_not_plotted_mean_of_N_reps`.
- Absent by ruling: 16n `ours2_direct` b16 (K2, Qwen; setup failure), all Ours b2 rows.
