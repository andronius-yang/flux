# Weak-scaling dataset (2-32 nodes, Kimi-K2)

Speedup-vs-scale study of the OURS s1 stack (`ours_l01_s1_pv2_r2`: pv2
placement + LocCap routing + r2 + slipstream comm/comp overlap) against
COMET (`l01_allgather_dense`) and the barebones NVSHMEM blocking-put ring
(`l01_nvshmem`). Isolated mode, K2 shape on captured routing, budgets =
strict per-rank pre-topk send budget (weak scaling in node count).

- `weak32_speedup_figure.html` — the published figure (artifact
  e558bd31-76fe-4e6c-87a8-82bbdd91a43d): two speedup panels, failure band,
  data table, method notes.
- `weak_scaling_results_tidy.csv` — per-cell means (handoff-30 schema)
  for every 32n and 2n capsule of this campaign, including the three
  failed 32n bring-up smokes (one cap per round: l1 node cap, l0 owner
  bitmask, l0 stage-1 smem window; fixed in commit 4b7d266) and the COMET
  32n failure record (no budget completes at 128 ranks). `(gate)` rows
  are correctness-ON smoke cells — never compare their latencies against
  perf-lane rows.
- `weak_scaling_speedup_table.csv` — the assembled 2-32n figure dataset.
  2n/32n from this campaign's capsules (2n = one-capsule head-to-head
  20260901-101958); 4/8/16n OURS+COMET from the handoff-30 tidy dataset,
  4/8/16n ring from the 8/31 ladder capsules (pre-cap-fix binary;
  cross-build error bounded ±3.6% by the 4n re-run 20260901-021249).
  Ring has no 2n cells and no 4-16n b8 cells.

Headline: vs COMET at b64 the speedup runs 1.01x (2n, COMET wins below
b64) -> 1.28x -> 1.86x -> 2.59x -> COMET fails at 32n. FAST excluded
(16-server hard limit, handoff 24).
