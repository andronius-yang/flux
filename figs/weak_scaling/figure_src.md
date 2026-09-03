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

## COMET at 32 nodes, 64 MiB (`oom_40gb_a100`)

Two-stage story, both verified from capsules (no guesswork):

1. **Original failure (capsule `20260901-075429`, 2026-09-01):** COMET's dense
   dispatch sorted gathered rows through a statically 64-rank shared-memory
   table (`MaxTpRanks = 64`, `AgScatterSortOpV2`, `sort_util.cu`, unguarded,
   inherited from upstream Flux). At 128 ranks it overflowed, corrupted the
   gather indices, and the GEMM faulted (b1–b4) or wedged (b8/b16). Root-caused
   2026-09-02 from the rank logs + source.
2. **After raising the table to 128 (+ a loud guard) — session 78f1b4cd,
   2026-09-02/03:** COMET **runs at 32n for 1–16 MiB** (capsule
   `20260902-235755`: 43.3 / 48.6 / 60.0 / 86.4 / 144.2 ms; Ours speedup
   1.13–2.69×). The **64 MiB cell does not fit an A100-40GB**: the dense
   gathered input is ~8.6 GB/rank at W=128; NVSHMEM symmetric-heap allocation
   fails at 13/15/17 G, and at a 24 G heap torch itself OOMs (needs 16.1 GiB +
   ~6.1 GiB non-torch overhead) — heap >17 + 16.1 + 6.1 > 39.5 GB, so the cell
   is infeasible on this GPU (probes `20260903-014242`, `20260903-023656`;
   would fit an 80 GB part). Hence the figure's 32n COMET marker reads
   **OOM**, not "does not run".

The NVSHMEM ring baseline (32n data, no 2n) is in the campaign table but not
in this figure (USER RULING: bars vs COMET only).
