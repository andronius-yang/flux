# Handoff 19 — Topic-drift ablation: the cost of adapting expert placement (PROPOSED)

Status: **proposed 2026-08-25** (user + session design discussion, OURS
record campaign). Not yet run. A later session will detail specs, cells,
and budgets; this doc fixes the rationale, the protocol skeleton, and the
machinery inventory so that session starts warm.

## Rationale

Every expert-placement baseline implicitly answers: *what happens when the
placement is wrong?* The existing dataset measures steady states (matched
placement, or per-iteration synthetic staleness) but never the **moment of
adaptation** — the iteration where traffic shifts topic and the resident
experts are stale. That moment is where the schools actually differ:

- Static placement (EPLB) pays the imbalance forever — and measured
  mismatch (8/14 A/B) is **worse than random** (imbalance 1.87–2.03,
  e2e +31..43%): a matched-then-drifted placement concentrated capacity
  on the wrong experts, below uniform-random's replica coverage.
- Always-migration (MoonEP faithful) never has a stale placement but pays
  the movement rent every iteration (loses b8; wins b32 vs EPLB — the
  rent-vs-imbalance crossover is budget-dependent).
- Conditional overlapped movement (OURS s2) idles when stable (quiet
  ≈ upkeep only) and pays a *partially hidden* move at the drift.
- No placement at all (Slipstream) is the imbalance-tolerant reference:
  flat, drift-insensitive, no apparatus.

The ablation prices arms 1–3 against reference 4 **through a drift
event**, not a steady state, at ≥2 budgets (the crossover demands it).

## Protocol: the 5-step timeline (one run = one curve per arm)

Per-iteration metrics are already recorded, so each cell yields a
drift-response curve (total_ms vs iteration):

1. **S1 warmup** (untimed): matched placement solved from pool-A oracle;
   pipeline warm.
2. **S2 matched steady state** (timed, ~3 iters): pool-A traffic on
   pool-A placement. Baseline level per arm.
3. **S3 DRIFT EVENT**: traffic switches to pool B mid-run (routing
   windows drawn from pool B from iteration k on). Placement is now
   stale. Nothing else changes.
4. **S4 adaptation transient** (timed, per-iteration — the money
   window): each arm reacts per its school — A1 absorbs imbalance, A2
   stops the world and moves, A3 moves under overlap, A4 doesn't notice.
5. **S5 post-drift steady state** (timed, ~3-4 iters): who returned to
   a matched-level floor, and how much of the S4 spike amortizes.

Drift severity is a ladder: adjacent-topic pool pair (mild), far-topic
pair (hard), `rot` reset (adversarial synthetic bound, existing probe).

## Arms

| arm | meaning | machinery |
|---|---|---|
| A1 stale-no-move | placement frozen at pool-A solve | s2 driver, trigger disabled (threshold=inf) — EXISTS |
| A2 staged move-first | timed bracket: solve -> move -> **block until landed** -> run | NEEDS driver bracket (~30 lines: apply_moves + join both layers pre-forward, separately timed) |
| A3 overlapped+sharded (OURS s2) | movement under dispatch/GEMM via weight-gated tiles + shard | EXISTS (canon: conn32, warm-kb0, drift prefilter) |
| A4 no-placement reference | local routing + comm-comp overlap, take the imbalance | l01_slipstream — EXISTS |

Cross-baseline points on the same protocol: EPLB matched/mismatch cells
(exist, 8/14 A/B), EPIC placement arm (exists in epic driver), MoonEP
faithful getmem (natively an always-move point; NIC-shard arms are OUR
optimization, ablation-only per variants.py note).

## Existing machinery (verified through the s2 campaign, 8/25)

- s2 driver: `--s2_stale {oracle,rot}`, `--s2_join`, `--s2_wprobe`,
  gain threshold + drift prefilter, warm-kb0 always-solve, place-lane
  sub-timers (gate mode), per-iteration move_stats (trigger/moves/bytes).
- Movement lane: WPM multicast + NIC-shard + weight-gated tiles / join;
  channel-aliasing fixed by family pin CUDA_DEVICE_MAX_CONNECTIONS=32.
- Mismatch instruments: EPLB pool A/B machinery; content-addressed
  per-pool routing sidecars (K2+Qwen, 8 EN pools each).
- Per-iteration recorder + tidy aggregation (handoff 18 tooling).

## Needed (small)

1. Driver: second routing file + `--drift_at_iter k` (switch windows).
2. Driver: A2 staged-migration bracket (timed move_block_ms).
3. Spec family for drift cells (pool-pair axis, budget axis b8/b32).
4. Decide S4 length (adaptation typically 1 trigger + landing; give it
   3-4 iters) and repeats (drift curves are per-iteration — n>=3 seeds).

Validate at 4n, record at 8n/16n. Wire-order rule 6 applies to every
movement leg (blocking or probe-proven); correctness cells randomize
payload per rule.
