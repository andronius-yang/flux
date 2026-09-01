# Ablation figure source data (2026-09-01)

Source of record: `docs/handoff/33_hetloo_severity.md` §2l (tables) and
§2j/§2k/§2k-reps (protocol + statistics). K2 4n, b64, isolated mode,
per-iteration max-rank total_ms. Binary sha16 505e4bed (ths_op) for all
9/1 capsules.

Files:
- `ablation_iter_tidy.csv` — long format: study (loo|matched), arm, rep,
  capsule (short id), iter (0-9 timed; the postwarmup placement reset
  fires before iter 0, so iter 0 = the drift-event iteration for the
  swap arms), total_ms.
- `ablation_tables.csv` — the summary tables: metric it0 (event
  iteration) and rest_mean (mean of iters 1-9 per rep, then over reps),
  n_reps, mean, sd over reps.

Arms:
- moonep — always-balance ceiling (cross-node movement every iteration).
  SINGLE RUN both studies; the matched row comes from the 8/31 hetero
  campaign capsule a25828a5 (oracle-insensitive arm, ce939eb9 binary —
  annotate if cross-compared).
- comet — dense allgather baseline. slipstream_comm_only — comm/comp
  overlap only. placement_swap_seq — placement/routing + one up-front
  sequential swapall, LEGACY comm bracket (the ours-driver proxy for the
  llc isolate, NOT the epic transport). full_stack_seq / full_stack_ovl
  — full comm canon + swapall postwarmup, sequential vs overlapped
  expert movement.

Caveats for the figure session:
- Only within-study, within-protocol comparisons are quotable: ~1.4 ms
  cross-allocation ambient offsets were observed between same-config
  capsules.
- placement_swap_seq LOO it0 has sd 14.2 (one rep spiked) — show reps
  or a box, not a bare mean, if that cell is drawn.
- matched it0: full_stack_seq == placement_swap_seq statistically
  (per-rep delta bounces +-5, mean +0.10); do not draw an ordering
  between them there.
