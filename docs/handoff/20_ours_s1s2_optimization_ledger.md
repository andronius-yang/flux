# Handoff 20 — OURS s1/s2 optimization ledger (2026-08-25)

Written record of every optimization applied to the OURS arms today, with
validation status. Record capsules for the campaign ran from the frozen
run tree at a4d2e47; items marked HELD are committed on the dev branch
but deliberately NOT in the record binary/config (post-record canon bump,
gated at 4n first).

## In the record (validated, capsule-backed)

1. `CUDA_DEVICE_MAX_CONNECTIONS=32` family pin (SCHEMA rule 14). Fixes
   two channel-aliasing defects (s2 stale-b32 deadlock; K2 torn-row
   race); perf −4..−5% at 8n and 16n-b64 (16n-b32 mixed, +5% — flagged).
2. Warm solve with keep_bonus=0 in the always-migrate regime (the
   kb=90090 replica-pull caused 0-adds paralysis).
3. plan_tensors_from_hosts vectorized (bitwise-verified; 6.9 -> 0.7 ms
   at G=384).
4. Cover-decision skip under always (trigger foregone; sentinel MUST be
   gain=0 — the -1 sentinel silently killed all movement via the
   apply_moves gain gate; caught by move-liveness records, fixed).
5. CUDA-graph capture of the warm solve (handoff-12 containment
   recovered; ~200 launches -> 1 replay).
6. End-of-iteration join_w1 fabric drain: one-sided remote epoch-SETs
   are NOT bounded by the destination's sync+barrier; zero-row moved
   slots' late SETs regressed raised flags -> K2-4n stale deadlock.
   Destination wait-all inside the boundary closes it (untimed gap).
7. apply_moves vectorization + keep-mask/shard-plan hoist (computed once
   for w1+w2; parity 300/300).

Net place lane: 13.7/19.5 ms -> 3.1/3.4 ms (qwen/K2 4n quiet-always);
s2 quiet totals: 4n 22.1/33.7 -> 11.5/13.3, 8n 29.3/33.7 -> 16.6/18.5.

## FINAL CANON (2026-08-25 user ruling, post-validation)

- **Place identity early-out: CANON** (53e1baf). Validated: quiet place
  2.6-3.1 -> 1.6-2.0 ms at 4n, in-cell correctness green on all 8 grid
  cells, both models; final s2 gate (ov0 + early-out) green.
- **plan_overlap: REMOVED EVERYWHERE** ("we don't overlap the plan").
  The s1-side flip was perf-positive and gate-green, but under s2 the
  ov side-stream x movement-lane timing re-exposed a mid-iteration
  signal race (2/2 failures: one torn row @iter3, one spin-hang @iter4;
  ov0 3x green same day). Rather than a per-scenario split, the user
  ruled ov out entirely. The ~0.5-1 ms plan saving is forgone; the
  ov x movement race is a documented open item (suspect: shard/gateway
  chain mid-iteration ordering; WPM wire itself is blocking-default).

## Explicitly NOT applied

- Anything from the l1-combine ablation (user directive 2026-08-25);
  see the SCHEMA rule-13 amendment for the msplit attribution ruling.
- Counts-only exchange (plan-lane redesign, deferred session).
- prod-arm prefilter recalibration (user decision pending: 10k ppm
  never gates under ~180k ppm window-to-window trace drift).
