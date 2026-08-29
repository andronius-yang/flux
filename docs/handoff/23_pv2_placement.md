# Handoff 23 — PV2: stateless node-aware greedy placement (branch pv2, 2026-08-27)

User-directed: implement the postdoc's node-aware greedy expert-placement
heuristic as branch **pv2**, minimize the duration of EVERY
placement-related calculation (solve AND tails), and A/B it against
PLACE-lambda-FAST (pll) at 4n (interactive) and 8n (debug). Preceded by
the same-day offline assessment (memory
`greedy-place-heuristic-assessment`): affinity spread ties/beats pll at
16n offline, loses qwen 4n on incidence; blind spread is catastrophic.

## 1. What PV2 is

`python/flux/testing/placement_v2.py` — the whole placement is a **pure
function of the demand histogram** `hist[NN, G]` (from the d[R, G]
allgather the plan lane already pays):

- counts: EPLB-global greedy `argmax(load/c)`, hard cap `c <= NN`
  (integer heap, ties lower g);
- node assignment: exact sequential greedy **affinity spread** (share
  desc; each instance to the max-residual-demand free non-hosting node),
  pure-python integer inner loop (GIL-bound — immune to OMP-pool
  contention that made tiny tensor ops jitter 6→56 ms on a busy host);
  leftover slots backfilled to the highest-share non-hosting experts;
- ranks: within-node snake (finalize_hosts semantics), tensorized;
- plan tensors: the canonical slot recipe, produced directly from
  (g, rank) tensors — **no hosts-list detour** (bitwise equal to
  `plan_tensors_from_hosts`, unit-gated).

Total solve 0.9–1.9 ms host, **no batch-size term** (b1 == b64), no CUDA
graph, no warm state. Deterministic integers + stable sorts: every rank
computes the identical placement.

Driver: `--place_solver pv2` (`test_moe_ours_traffic.py`). The pv2 s2
lane = one D2H of d → host drift → solve → remote-rows gain proxy →
plan-tensor adoption + `apply_moves` (movement machinery SHARED with
pll — deliberately, so the A/B isolates the placement lane). Timed loop
pins `torch.set_num_threads(1)` (restored after).

Unit gates `test/python/moe_ag_scatter/test_placement_v2.py` (CPU, login
node): determinism, recipe equality, counts == brute-force greedy,
structure, runtime==setup placement identity, real-trace quality vs the
pll cold solve.

## 2. Sizing-envelope fix — the handoff-22 §4 blocker CLOSED

s2 caps are now maxed over the **full set of runtime-reachable
placements**: cold batch + stale-rot resident + (pll only) the runtime
warm-solve **orbit** from the resident seed, each reference-routed at
setup (`[s2-sizing]` prints). Measured at the K2 4n gate: resident/batch
f_cap ≈ 102, **orbit0 f_cap 398** — exactly the gap that made
`ours_l01_s2_gate_r2` assert `kstats[2] != 0` on 8/26.

PV2 needs no orbit: its runtime adoption IS the setup batch solve
(stateless purity — the sizing premise handoff 22 declared FALSE for the
warm solver is restored by construction).

Result: **`ours_l01_s2_gate_r2` (pll) gates GREEN at 4n K2 + Qwen** —
the s2×r2 lane is un-blocked. PV2's own gates (s1 + s2 stale-rot +
force-trigger + rule-6c weight probe): green K2 + Qwen, 8/8 per-iteration
output checks, correctness PASS all ranks.

Gate capsules: 20260827-085903 (K2 3/3), 20260827-090902 (Qwen 3/3).

## 3. A/B capsules (one binary — python-only changes on the 8/26 build)

Arms per capsule (r2 slack parity; never-mix vs R_red=0 and vs pll
out_sha): `ours_l01_{s1,s1_pv2,s2,s2_pv2,s2_stale,s2_stale_pv2}_r2`,
b1/b8/b64, isolated, 5+10.

| capsule | topo/model | status |
|---|---|---|
| 20260827-091901_perlmutter_1e3ad650 | 4n K2 | 18/18 ok (1 flaky-race retry, §5) |
| 20260827-094758_perlmutter_12fd2fb3 | 4n Qwen | 18/18 ok |
| 20260827-100404_perlmutter_a8696113 | 8n K2 | 18/18 ok |
| 20260827-101618_perlmutter_7b9c1ff6 | 8n Qwen | 18/18 ok (1 flaky-race retry, §5) |

## 4. Results (median of 10, place_ms / total_ms)

**Quiet always-solve s2 (the headline A/B):** pv2 place is FLAT
(~1.1–1.7 ms) at every budget; pll grows with batch (3.1 → 10.0 ms at
qwen b64):

| cell | pll place | pv2 place | pll total | pv2 total | Δtotal |
|---|---|---|---|---|---|
| K2 4n b1/b8/b64 | 3.06/3.39/4.74 | 1.50/1.50/1.51 | 9.47/13.25/53.07 | 7.44/11.49/47.28 | −21/−13/−11% |
| Qwen 4n b1/b8/b64 | 2.55/3.13/9.98 | 1.09/1.11/1.10 | 6.55/10.72/49.70 | 4.75/8.53/42.07 | −27/−20/−15% |
| K2 8n b1/b8/b64 | 3.09/4.10/6.68 | 1.68/1.69/1.70 | 11.58/17.10/70.48 | 9.81/14.06/63.14 | −15/−18/−10% |
| Qwen 8n b1/b8/b64 | 2.65/3.13/6.98 | 1.19/1.20/1.20 | 8.69/13.55/61.88 | 7.13/11.14/51.05 | −18/−18/−17% |

**Static s1 (placement quality only):** parity at K2 (4n: 5.83 vs 5.85 /
9.53 vs 9.59 / 48.40 vs 47.31; 8n: 7.65 vs 7.65 / 12.71 vs 12.12 /
65.60 vs 64.52) and a pv2 WIN at Qwen 4n b64 (43.28→39.97 = −7.6%) —
the offline qwen-4n incidence deficit (+15–17%) does not surface there
(better rank balance / lower forced overflow compensate). The one
quality loss on hardware: **Qwen 8n s1**, pv2 +3.6…+10.6% total
(5.84→6.46 / 10.62→11.00 / 57.58→61.11) — while the s2 quiet pair at the
same geometry favors pv2 (batch-basis solve vs the s1 oracle-window
basis; single capsule, e2e spread at 8n is a few percent — a repeat
capsule would firm this up). Net: quality parity ±10% everywhere
measured, tilting pv2 at large budgets and K2, tilting pll at Qwen 8n
static.

**Stale (migrate every iteration):** place_ms ≈ 85–136 ms BOTH arms —
the shared adoption/movement tail (apply_moves gateway/shard planning +
WPM issue at ~400 adds/iter under r2) dominates and is solver-agnostic,
exactly as the offline assessment predicted. Totals still favor pv2 by
0–15%. **Per-step greedy re-placement remains tail-bound: the next
blocker is vectorizing/overlapping the movement-issue chain, not the
solve.**

## 5. Incidents / caveats

- `ours_l01_s2_stale_pv2_r2` at **b1** went stuck on first attempt in
  BOTH topologies (K2 4n: rank 0 deadlocked mid-iteration 10/15; Qwen
  8n: same class) and passed on the runner's retry both times, and at
  b8/b64 first-try — a REPEATABLE-preference signature of the known-open
  WPM mid-iteration movement race (handoff 20; shard/gateway ordering,
  flux-ours session owns the root-cause). PV2's host-timing profile
  (short place lane, no cover decision) reliably widens the race window
  at fast small-budget iterations under migrate-every-iteration. Not a
  pv2 logic bug (gates 8/8 green incl. weight probe; pll had 3/3
  analogous failures on 8/25 in other configs) — but stale-b1 pv2 cells
  should be run with retries until the race is fixed.
- Login-node background runs of the sweep runner were reaped at ~6–8 min
  twice (no capsule written; no orphan steps). Workaround that held: run
  the runner inside a persistent Monitor task. Two lost partial runs cost
  ~15 min of the 4n window.
- The working tree carries the other session's uncommitted FAST-lane
  files; capsules record git_dirty accordingly. Binary identity is the
  8/26 rebuild (unchanged — all pv2 work is python-only).
- pv2 arms are a NEW family: never bitwise-compare against pll cells;
  r2 slack boundary applies as always.

## 6. Cost

4n interactive window ~1.0 h × 4 nodes ≈ 4.1 nh (gates + 2 A/B capsules
+ 2 reaped partials). 8n debug: 2 windows (~15 + ~17 min) × 8 nodes ≈ 4.3 nh; total ≈ 8.4 nh.
Every allocation scancelled at completion.

## 7. State / next

- Branch `pv2` (local, never pushed): code + specs + capsules committed.
- Next: 16n A/B (the offline assessment predicts the pv2 quality WIN
  grows there: −3% K2, −14…−20% Qwen wire visits); movement-tail
  vectorization for true per-step migration; root-cause the WPM race
  (flux-ours session); optional prod-regime (prefilter) A/B under real
  trace drift.
