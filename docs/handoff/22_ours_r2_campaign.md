# Handoff 22 — OURS r2 authentic-numbers campaign (2026-08-26)

User-directed campaign: authentic OURS numbers at **slack parity**
(`--redundant_per_rank 2`, matching every other expert-movement baseline —
the 2026-08-25 finding was that OURS alone ran R_red=0). Datapoint-skill
conventions (handoff 18 = parent protocol): isolated mode only, 5+10
iters, s1-canon inputs (SCHEMA rule 10), budgets b1–b64, K2 + Qwen,
4n/8n/16n. Headline metric = plan-inclusive `total_ms`.

**Scope outcome: s1-only.** `ours_l01_s1_r2` delivered in full (42/42
record cells ok, zero failures, zero wedges). `ours_l01_s2_r2` is
**BLOCKED** at its 4n gate — see §4.

Files: `22_ours_r2_results_tidy.csv` (tidy dataset),
`22_ours_r2_figure_tables.csv` (figure-table blocks; only the ours row is
populated — this was a single-arm campaign).

## 1. Binary + never-mix

One binary served every capsule: main @ merge `f96f5bc` (ours→main, full
`ours` branch line incl. r2 arms) + same-day incremental rebuild
(2026-08-26 12:27/12:28 PDT `.so` pair, lib64→lib synced), CUDA 12.4 pins.
Tag audit + `find src -newer` clean; dry-runs 6/6 clean. The new
`ours_l01_s2_r2`/`ours_l01_s2_gate_r2` variants and the driver f_cap fix
(§4) are python-only — binary identity holds across the campaign.

NEVER-MIX: r2 cells never compare to R_red=0 cells (slack boundary,
2026-08-25); today's capsules never compare against 8/25 capsules
(different builds, rule 4). The vs-8/25 deltas in §3 are indicative only
and marked as cross-build.

## 2. Record capsules (all `ours_l01_s1_r2`, 7/7 ok each)

| Topo | Model | run_id |
|---|---|---|
| 4n | K2 | 20260826-201042_perlmutter_42c8fd19 |
| 4n | Qwen | 20260826-201555_perlmutter_ae55d0d4 |
| 8n | K2 | 20260826-202657_perlmutter_dc75c754 |
| 8n | Qwen | 20260826-203200_perlmutter_e2e1b008 |
| 16n | K2 | 20260826-202541_perlmutter_6b4ce193 |
| 16n | Qwen | 20260826-203704_perlmutter_ed48058d |

Gates/canary: 20260826-193442_perlmutter_8895e6ad (K2 4n: s1 gate ok
447 s **+ s2 gate stuck** — the §4 evidence), 20260826-200723_perlmutter_fa149af4
(Qwen 4n s1 gate ok 49 s), 20260826-202149_perlmutter_b9e5e475 (K2 4n s2
re-gate post-fix, still stuck), 20260826-202219_perlmutter_7c86cb27 (16n
s1 b1 canary ok 156 s — r2-at-16n was previously unmeasured).

## 3. Headline numbers — `total_ms` (plan-inclusive), median of 10

| b (MiB) | K2 4n | Qwen 4n | K2 8n | Qwen 8n | K2 16n | Qwen 16n |
|---|---|---|---|---|---|---|
| 1 | 5.89 | 3.64 | 7.53 | 5.87 | 13.65 | 12.89 |
| 2 | 6.44 | 4.13 | 8.20 | 6.36 | 14.43 | 13.80 |
| 4 | 7.29 | 5.28 | 9.62 | 7.65 | 16.49 | 15.34 |
| 8 | 9.58 | 7.60 | 12.71 | 10.85 | 20.58 | 18.39 |
| 16 | 15.28 | 12.61 | 20.44 | 17.46 | 29.85 | 26.36 |
| 32 | 26.09 | 22.91 | 36.59 | 29.86 | 47.51 | 43.84 |
| 64 | 47.79 | 42.49 | 68.31 | 55.94 | 86.43 | 78.73 |

plan_ms stays 1.2–2.2 (4n) / 1.3–4.7 (8n) / 1.6–8.3 (16n) across the
ladder; e2e/l0/l1 splits in the tidy CSV. Ladders monotone, no outliers.

Cross-build indicative delta (NOT rule-4 comparable): vs the 8/25 r0
s1 16n grid (qwen 117.75 / K2 106.60 at b64), today's r2 16n b64 is
78.73 / 86.43 — the slack-parity prediction that the r2 win grows with
scale holds direction-wise at 16n, previously unmeasured.

## 4. s2_r2 blocker (for the flux-ours s2 session)

`ours_l01_s2_gate_r2` (K2 4n b8) fails deterministically at i0:
`plan_meta` assert **"loccap_sl forced-budget overflow (kstats[2] != 0)"**
— asserting ranks are the non-rank-0 node leads (r4/r8/r12); the other
ranks park in collectives → cell reported "stuck" at idle-timeout.

RCA chain, two layers:
1. **Driver sizing bug (fixed, fix KEPT, uncommitted):**
   `test_moe_ours_traffic.py` computed the s2 batch-placement bounds via
   `loccap_sl_bounds(pll_aux_b, W, args.pll_f_cap)` — i.e. under the
   RESIDENT-derived f_cap; the batch placement's own forced-pair peaks
   never raised `f_cap` (only recv/pair caps were maxed). Fix: auto-derive
   (`f_cap=-1`) and `args.pll_f_cap = max(resident, batch)`. Exact no-op
   at r0 and whenever the resident cap dominates; s1 path untouched.
2. **Open mechanism gap (why the re-gate STILL fails):** under the gate's
   stale-rot + force-trigger regime, movement fires at i0
   (`trigger 1 moved 400`) and routing then runs under the
   **runtime-adopted** placement, whose forced geometry exceeds even
   max(resident, cold-batch) f_cap. The driver's sizing premise — "the
   stale probe oscillates between exactly these two placements" — is
   FALSE at r2: warm-solve adoptions under replica headroom leave the
   setup-derived forced-budget envelope. Contract options: recompute/raise
   f_cap on adoption, a provable placement-independent forced bound, or a
   kernel-side graceful path for budget misses. This is s2-schedule
   mechanism work (related open item: mid-iteration movement race,
   handoff 20 / ours-record close-out), left to the flux-ours session.

Until re-gated green, s2×r2 numbers must not be quoted anywhere.

## 5. Cost + ops

≈ **14.7 node-hours** total: 4n gates/records 2×4n windows ≈ 4.1 nh
(includes ~1.5 nh of s2 gate diagnostics: two stuck cells + one killed
retry), 8n lane 1.33 nh (one debug window, ~10 min), 16n lane ≈ 9.3 nh
(one regular -t 45 window, ~35 min; grant landed during the gate grace
window — zero idle). No wedges, no lost windows, every allocation
scancelled at completion. Same-day context: campaign ran the evening
after the 08-26 site-wide Lustre outage (partial dead OSTs; see the
session memory) — build + launch waited for OST recovery.

## 6. Uncommitted state (a human commits)

- 10 capsules under `sweeps/results/runs/20260826-*` (6 records + 3 gate
  + 1 canary).
- `sweeps/variants.py`: `ours_l01_s2_r2` + `ours_l01_s2_gate_r2` arms.
- `test/python/moe_ag_scatter/test_moe_ours_traffic.py`: the §4 f_cap fix.
- `docs/handoff/22_*` (this file + 2 CSVs).
- The `ours`→`main` merge `f96f5bc` is already committed (user-directed
  wiring); nothing pushed anywhere.
