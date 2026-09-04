# Weak-scaling dataset (2-32 nodes, Kimi-K2)

Speedup-vs-scale study of the OURS s1 stack (`ours_l01_s1_pv2_r2`: pv2
placement + LocCap routing + r2 + slipstream comm/comp overlap) against
COMET (`l01_allgather_dense`) and the barebones NVSHMEM blocking-put ring
(`l01_nvshmem`). Isolated mode, K2 shape on captured routing, budgets =
strict per-rank pre-topk send budget (weak scaling in node count).

- `weak32_speedup_figure.html` — v1 figure (artifact
  e558bd31-76fe-4e6c-87a8-82bbdd91a43d): ring panel 4-32n.
- `weak32_speedup_figure_v2.html` — v2 figure: BOTH panels 2-32n (ring 2n
  from capsule 20260904-004243, ring+s1 in one capsule).
- `weak_scaling_results_tidy.csv` — per-cell means (handoff-30 schema)
  for every 32n and 2n capsule of this campaign, including the three
  failed 32n bring-up smokes (one cap per round: l1 node cap, l0 owner
  bitmask, l0 stage-1 smem window; fixed in commit 4b7d266), the original
  COMET 32n failure record (upstream MaxTpRanks=64 dense-dispatch sort
  table, unguarded), and the post-fix COMET 32n capsule 20260902-235755
  (MaxTpRanks=128: b1-b16 run; b64 does not fit a 40 GB A100 — heap probes
  15G/17G exhaust NVSHMEM, 24G OOMs torch at 16.1 GiB; capsules
  20260903-014242 / 20260903-023656, spec weak32_comet_b64_k2.yaml). `(gate)` rows
  are correctness-ON smoke cells — never compare their latencies against
  perf-lane rows.
- `weak_scaling_speedup_table.csv` — the assembled 2-32n figure dataset.
  2n/32n from this campaign's capsules (2n = one-capsule head-to-head
  20260901-101958); 4/8/16n OURS+COMET from the handoff-30 tidy dataset,
  4/8/16n ring from the 8/31 ladder capsules (pre-cap-fix binary;
  cross-build error bounded ±3.6% by the 4n re-run 20260901-021249).
  2n ring speedups use the s1 column of the ring's own 2n capsule (agrees
  with the COMET-capsule s1 within ±1.5%). Ring has no 4-16n b8 cells.

Headline: vs COMET at b16 the speedup runs 0.92x (2n, COMET wins) ->
1.16x -> 1.56x -> 2.26x -> 2.69x (32n); at b64 1.01 -> 1.28 -> 1.86 ->
2.59 -> COMET does not fit at 32n (40 GB A100 memory wall, bracketed). FAST excluded
(16-server hard limit, handoff 24).

## ver4 (REV 3.0): stacked 1 MiB / 64 MiB figure, Ours = min over arms

`make_figure.py --baseline nvshmem --stacked` -> `weak_scaling_nvshmem_stacked.{pdf,png}`.
"Ours" follows the main figure's convention (fastest of s1 and the
direct-wire arm per node). New same-binary one-capsule sets at 1 MiB:
16n `20260904-120809` (dwire 7.20 / s1 12.77 / ring 24.77) and 32n
`20260904-124333` (dwire 9.25 / s1 39.19 / ring 81.77; the 32n dwire cell's
timed loop is complete but the process hung at teardown -> status timeout,
flagged in figure_src.csv). 1 MiB speedups vs ring: 1.18 / 1.33 / 1.62 /
3.44 / 8.84x.
