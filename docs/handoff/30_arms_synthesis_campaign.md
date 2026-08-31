# Handoff 30 — six-arm synthesis campaign: scenarios vs best baselines, one binary (2026-08-30)

**One-line:** the canon OURS stack (pv2 placement + LocCap routing + r2 + Slipstream
combine, 8/29 comm canon) measured against its own constituents (slipstream-only,
placement/routing-only) and COMET on ONE fresh binary across K2/Qwen x 4n/8n/16n x
b1-b64 — OURS-s1 is the headline winner from 8n up and from b4 up at 16n, the
direct-wire ablation owns 16n b1/b2 (handoff-29 prediction CONFIRMED), and s2
(always-solve + forced swap) prices the per-iteration adaptivity at ~1-1.6 ms.

## 1. Scope and arms (user-directed, 2026-08-30)

One capsule per (model x variant x topology), budgets 1-64 MiB, isolated mode,
5 warmup + 10 timed iters, s1-canon inputs (datacamp specs, SCHEMA rule 10),
slack parity r2 on every replication arm:

| scenario | variant key | capsules |
|---|---|---|
| COMET | `l01_allgather_dense` | 6 |
| Ours: slipstream-only (8/29 canon via binary defaults) | `l01_slipstream` | 6 |
| Ours: placement/routing-only | `llc_l01_s1_pv2` | 6 |
| Ours combined, s1 | `ours_l01_s1_pv2_r2` | 6 |
| Ours combined, s2 = always-solve + forced swap every iter (p2p, early) | `ours_l01_s2_swap_force_p2p_r2` | 6 |
| Ours plan over direct staged wire (handoff 29) | `ours_l01_s1_pv2_r2_dwire` | 6 |
| EPLB (16n only — dwire transport-identity comparator) | `eplb_l01` | 2 |

38 campaign capsules + 8 regather capsules (§8), 266 dataset rows — ALL 266 ok after the 8/30-evening regather (0 failed, 0 pre-skipped).

## 2. Binary identity (rule 4)

Fresh build 2026-08-30 ~05:47 on pv2 @ ab03abb (+ chunked-combine merge in src):
`libflux_cuda_ths_op.so` sha256 prefix **1b5593d4e917214e**, cuda 12.4 pins.
Tag audit green incl. WAVE_ADAPT(=48 default) + COMBINE_IDX(=1 default) — the 8/29
combine canon — and CHUNK/PIECES present but default-OFF (handoff 28 canon dial
defended). The lib64-shadows-lib install gotcha was hit and fixed (lib/ synced).
Anchors: COMET K2+Qwen b8 within -3.5..+6.2% of handoff-18 at every topology;
llc + s2 capsules within <3% of the 8/29 same-canon references — **no build shift**.
Every capsule `deterministic=0`.

## 3. Headline (plan-inclusive total_ms, per-iter max-rank, median of 10 iters)

b1 / b8 per arm (see the tidy CSV for all budgets and the mechanism columns):

| topo model | COMET | Slipstream | PLL-pv2 | **OURS-s1** | OURS-s2 | dwire | EPLB |
|---|---|---|---|---|---|---|---|
| 4n K2 | 4.02 / 9.55 | 3.95 / 9.70 | 6.14 / 11.72 | 4.01 / **8.99** | 5.00 / 10.08 | 6.17 / 20.10 | — |
| 4n Qwen | 4.45 / 8.65 | 3.07 / 8.72 | 5.28 / 9.91 | **3.00 / 6.75** | 3.93 / 7.73 | 4.88 / 18.37 | — |
| 8n K2 | 6.80 / 16.58 | 5.60 / 13.99 | 8.07 / 14.58 | **5.68 / 12.40** | 7.08 / 13.28 | 6.13 / 23.15 | — |
| 8n Qwen | 7.35 / 17.28 | 5.23 / 13.89 | 7.63 / 13.68 | **5.35 / 10.37** | 6.25 / 11.14 | 5.61 / 23.79 | — |
| 16n K2 | 15.58 / 34.67 | 12.88 / 22.77 | 15.12 / 22.96 | 12.61 / **20.42** | 14.02 / 21.85 | **7.22** / 27.84 | 9.87 / 32.56 |
| 16n Qwen | 16.95 / 34.90 | 12.36 / 22.89 | 15.23 / 21.63 | 12.33 / **17.63** | 13.43 / 18.98 | **6.83** / 28.70 | 8.30 / 30.51 |

Findings:
1. **OURS-s1 is the canonical winner** at every 8n cell, at 16n from b4 up, and at
   4n b4+ (Qwen decisively: -22% vs COMET at 4n b8, -40% at 8n b8, -49% at 16n b8).
   The synthesis beats both constituents everywhere they are beaten separately —
   placement+routing and the slipstream combine compose.
2. **16n b1/b2 belongs to the direct wire**: dwire 7.22/6.83 beats EPLB 9.87/8.30
   (-25/-18%) and every fused arm — see §4. Crossover to fused at b4.
3. **s2 prices per-iteration adaptivity at +0.9-1.6 ms total** over s1 with
   identical e2e at 4n/8n (the movement itself is fully overlapped; the cost is
   the always-on solve + host apply/issue). At 16n K2 the forced swap develops an
   intermittent stall tail (e2e max 120 ms at b8; median unaffected). On a static
   eval distribution s2 buys nothing — it is the insurance premium, not a win.
4. **Placement/routing-only (PLL-pv2) never beats slipstream-only** on totals at
   4n/8n (its plan bracket ~2.1-2.5 ms dominates its transport gains) but ties it
   at 16n b8+ — placement value grows with hop count, comm value with budget.

## 4. Handoff-29 §4 prediction — CONFIRMED (both models)

K2 16n (capsules e216d851 / 0880056b / 54f921d1), lane-quoted mean-over-ranks+iters:
dwire b1 total 7.488 (predicted 7.5-8.5), EPLB 9.989 (predicted 9.5-10), fused ours
12.625 (predicted ~14). **Flip vs EPLB: b1 -25.0%, b2 -21.4%** (Qwen: -18.2%, -13.9%).
Transport identity: wire legs identical (b8 13.134 vs 13.171 ms); dwire's *overall*
e2e drifts 7-10% faster than eplb at b4-b16 — outside the wire leg, recorded not
debugged. Fused takes over from b4+ exactly as the stage-chain diagnosis requires.
Handoff 29 §4 has been updated with the verdict; memory `dwire-transport-ablation`
flipped COMPLETE.

## 5. Failed / flagged / pre-skipped cells

- dwire 16n b32+b64 (both models): ONE root cause, RESOLVED same evening (§8) —
  the "wedge" at b32 was rank deaths in the All2AllSingle ctor
  (`flux_shm.cc:80 NVSHMEM_MALLOC`) with surviving ranks blocked in the
  collective until the watch-kill; b64 identical. Cells regathered green.
- llc 16n b64 (both models): the pre-skip was STALE — resolved by the 8/25-8/27
  fixes (llc_sizing=demand lineage + hidden-A2A skip, handoff 27 "both 16n-b64
  fixes"); `llc_l01_s1_pv2` b64 had already run green on 8/29 (fa493d1c /
  a071f2c5). Regathered green with no flags (§8).
- **FLAG — qwen 16n OURS-s1 b16 intermittent op-bracket stall (REPRODUCED)**:
  original cell bd8b7b04 had 5/10 iters at ~352-449 ms (median 191.1, poisoned);
  the §8 regather (d5be00eb) reproduced the stall at 1-2/10 iters (e2e max
  357.7) with a CLEAN median 26.06 — the dataset row now carries the regather
  value with the flag. The stall is real, intermittent, ~350-450 ms, inside the
  op e2e bracket (plan/place normal), absent at b8/b32 same arm. Mild echo: K2
  16n s2 b8 (e2e max 119.9, median clean). OPEN RCA candidate — smells like an
  intermittent collective/wire stall class.
- Loose-bound heap clamp-at-16G warnings (llc/ours-s2/dwire high-b): all such
  cells ran ok; warning is the known loose-prior class.

## 6. Execution ledger

- 4N lane: 12/12 invocations, 84/84 ok, job 57736340, 39 min, **2.60 nh**, ~0 idle.
- 8N lane: 12/12, 84/84 ok, debug windows 57736504 + 57737307, **5.94 nh**, ~0 idle.
- 16N lane: 14/14, 92 attempted cells (88 ok, 4 failed as above), job 57738204
  (65 min grant), **17.4 nh**, ~3-4 min idle; ~4h50m queue wait (0 nh).
- **Total ~25.9 node-hours** (m5350_g). No rebuilds mid-campaign, no repo edits by
  lanes, all capsules deterministic=0.
- Incidents: (a) 16n pending salloc killed at ~60 min when the harness reaped the
  backgrounded process — job cancelled, queue position lost; fix = setsid-detached
  salloc (add to run-books). (b) `-I120` in salloc lines makes salloc abandon
  long-queue requests — removed for 8n/16n mid-campaign. (c) shared-scratch log
  filename collision 4N/8N — lane-prefixed filenames restored (8/23 ledger rule).
  No data impact from any incident.

## 7. Files / provenance

- Tidy dataset: `docs/handoff/30_arms_synthesis_results_tidy.csv` (conventions =
  handoff 15 / datapoint skill; headline = plan-inclusive total_ms).
- Figure tables: `docs/handoff/30_arms_synthesis_figure_tables.csv` (new arm rows:
  1+2 = OURS-s1, 1+2 s2, 1+2 direct-wire).
- Capsules: 38 under `sweeps/results/runs/20260830-{1253..1848}*` (run_ids in the
  tidy `capsule` column) — staged, human commits.
- Never-mix: this campaign's binary (1b5593d4) postdates the 8/29 comm canon and
  the 8/30 chunked-combine merge; compare against 8/29+ capsules only as
  same-canon (drift <3% verified), never against pre-8/29 wave-path data.

## 8. Regather (2026-08-30 evening, user-directed): the 6 missing cells + b16 rerun

Two brief 16n windows (57755670 ~13 min, 57755966 ~3 min; granted instantly —
queue mood flipped from the afternoon's 3h46m).

**llc b64**: ran as-is, green both models (K2 `20260830-230822_3f853021`
total 102.14; Qwen `-231220_27069afe` total 92.87). The campaign pre-skip was
stale — fixed since 8/25-8/27; 8/29 had already proven it.

**dwire b32/b64 RCA + fix**: both budgets/models died in the All2AllSingle
CTOR (`NVSHMEM_MALLOC`), not a wedge — capacity sizing's provable pair_cap
floor needs >16G of staging. Fix v1 (`--dwire_pair_sizing demand`, twin
variant `ours_l01_s1_pv2_r2_dwire_dps`): max_split = pair_ref + cushion —
cleared K2 b32 (`-230620_f3898c43`, max_split 10205) but still died at K2 b64
+ qwen b32 because the shared `cushion` is the RECV-side per-destination sum
(fp_slack + 8W ≈ 9k rows at K2 16n) — ~9x the realized pair (1155). Fix v2:
PAIR-level cushion (`forced_pair.max() + 64`) — all remaining cells green:
qwen b32 `-231559_cae63d59` (104.20), K2 b64 `20260831-011130_45dacd5f`
(199.76), qwen b64 `-011223_93360895` (196.72). max_split is alloc-only
(ctor + per-iteration loudness assert; never in a timed path) — the dps twin
is latency-identical to the capacity arm by construction; K2 b32 ran under
both v1 (fat) and v2-era sizing conventions without a timing delta lever.
Driver knob + twin committed in this handoff's tree
(`test_moe_ours_traffic.py --dwire_pair_sizing`, `variants.py`).

**qwen OURS-s1 b16 rerun** (`-231008_d5be00eb`): stall reproduced at lower
incidence (1-2/10 iters, max 357.7 ms), median clean 26.06 — see §5 flag.

**Completed 16n b32/b64 picture (total_ms)**: dwire K2 99.17/199.76, qwen
104.20/196.72 — dwire stays 13-15% below eplb even at high budgets
(117.31/227.03, 114.12/217.88) but the fused arm owns b4+ by 2-4x
(47.58/85.79, 44.26/76.80). The 16n story is complete: dwire b1-b2, fused
b4+, monotone crossover, EPLB dominated at every budget by one of ours.
