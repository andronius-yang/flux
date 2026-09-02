# Weak-scaling figure — `figure_src.csv` notes

Source of truth for `make_figure.py` (verA latency / verB throughput).
Derived 2026-09-02 from `weak_scaling_speedup_table.csv` (the campaign's
assembled 2–32n dataset, see `README.md`), restricted to the **64 MiB**
per-rank budget, K2 shape (the only model in the weak-scaling campaign).

## Columns

- `nodes`, `ranks` (4 GPUs/node on Perlmutter), `system` (`ours` =
  `ours_l01_s1_pv2_r2`, the s1 stack; `comet` = `l01_allgather_dense`).
- `total_ms` — isolated-mode planning-inclusive total (SCHEMA rule 5), the
  campaign's per-iter-max / median-over-iters aggregate.
- `tokens_per_rank` = **4,680** pre-topk input tokens: the 64 MiB budget is
  realized as `effective_budget_bytes` 67,092,480 B at 14,336 B/token
  (H 7168 × bf16), exact in every b64 capsule (2n cell records
  ntokens 37,440 = 4,680 × 8). USER RULING 2026-09-02: tokens = pre-topk
  input tokens per rank (not the topk-replicated wire rows).
- `total_tokens` = tokens_per_rank × ranks (weak scaling: per-rank work
  fixed, total grows with nodes).
- `throughput_mtok_s` = total_tokens / total_ms, in million tokens per
  second — the synthesized MoE-layer throughput.
- `speedup_vs_comet` = comet_total_ms / ours_total_ms (identical to the
  throughput ratio; tokens cancel). Present only where COMET ran (2–16n).
- `status`, `capsule`, `cell_id`, `source_dataset` — provenance. 2n and
  32n rows come from this campaign's capsules; 4/8/16n from the
  handoff-30 same-binary dataset (USER RULING 2026-09-02: keep the
  weak-scaling table's cells even though the main figure quotes different
  capsules for the same nominal 4n/16n cells — ~5% cross-build drift,
  README-bounded ±3.6%).

## COMET at 32 nodes (`does_not_run`)

Root-caused 2026-09-02 from capsule `20260901-075429_perlmutter_04875008`
and source, no re-run: COMET's dense (all-gather) dispatch sorts each
expert's gathered rows by source rank in `AgScatterSortOpV2`
(`src/moe_ag_scatter/sort_util.cu`), whose shared-memory per-source-rank
counter is statically sized `MaxTpRanks = 64` and unguarded. At 128 ranks
source ranks 64–127 overflow into the adjacent expert prefix sums, the
gather indices are corrupted, and the grouped GEMM faults (illegal memory
access on 2/128 ranks in the first forward; the rest wedge on the dense
barrier). b1–b4 faulted, b8/b16 wedged to the watchdog, b64 was never
launched (allocation expired). The constant is inherited from upstream
ByteDance Flux (MoE release db4ffe0, still 64 on origin/main); 16n = 64
ranks is exactly the boundary. OURS is unaffected (a2av kernels, caps
raised to 128 in 4b7d266). The NVSHMEM ring baseline (32n data, no 2n) is
in the campaign table but not in this figure (USER RULING: bars vs COMET
only).
