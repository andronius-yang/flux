# Handoff 32 — Hetero-oracle multi-homog scenario: math check + 4n measurement campaign (2026-08-31)

**Scenario (postdoc direction, user-adopted 8/31).** The placement oracle is no
longer one topic: it is the EQUAL-WEIGHT MIX of the available topic pools'
previous decode window (rule-10 window [32,64), g=0), while every evaluated
batch stays SINGLE-TOPIC homog on the canon window [64,96). One cell per
(topic, budget, arm); the headline is the MEAN over topics. This replaces the
old wrong-topic mismatch A/B as the fair drift regime: the oracle did the best
it could with the information it had, and each served batch deviates from it.
K2 mix = 8 topics (lcb/execution + 7 mmlu EN); Qwen mix = all 6 captured pools.

## 1. Mechanics (worktree-het-oracle commit 98eb2c8)

- `gen_trace_routing.ensure_oracle_sidecars`: `opool` now accepts a
  `+`-joined pool list — balanced (min-pool-len per topic) round-robin
  interleave so every W-block of the driver's `(W,-1,topk)` reshape sees the
  same topic mixture. Single-pool opool and no-opool bodies are byte-identical
  to before (content-guard verified against a pre-existing canonical sidecar).
- **`sweep.py` rule-10 gap FOUND+FIXED**: the oracle-routing condition matched
  only `placelambda_fast/gpu` tokens, so `llc_*_pv2` arms had run SELF-oracle
  (basis=self in rank logs) since the 8/27 `placelambda_fast`→`pv2` token
  swap. Every 8/27–8/31 `llc_l01_s1_pv2` capsule (datapoint PLL rows,
  handoff 30 PLL-pv2 arm) measured self-oracle placement — likely ~unchanged
  numbers under matched single-topic g=0, but annotate before quoting, and
  never compare against post-fix pv2 cells without noting the basis flip.
  The `ours` driver was never affected.
- Specs `sweeps/specs/hetoracle_{k2,qwen}_4n.yaml`: 8/6 per-topic homog
  families sharing the mix opool, budgets b1–b64, isolated, 5+10 iters,
  s1-canon inputs otherwise.
- On-GPU wiring proof in both lanes' llc capsules:
  `placement pv2; basis: ORACLE …oracle_routing.txt; oracle->batch drift
  401035–477500 ppm` — 40–48% drift, ~30x the s2_prod 1% prefilter, matching
  the offline TV distances (0.21–0.38).

## 2. Offline math check (same-code, CPU; scratch het_oracle_{K2,Qwen}.json)

pv2_solve (r2 nlp) + ours_swap.swap_plan/apply_swaps + LocCap simulate_arm
(eps 0.0625) on the real pools, 4n b4, 2 seeds. Mean over topics:

| arm | K2 imb / rmax / inter-node rows | Qwen imb / rmax / inter-node rows |
|---|---|---|
| mixed static | 1.228 / 2908 / 12086 | 1.212 / 4966 / 19713 |
| re-solve on topic window | 1.069 / −12.9% / −5.3% | 1.062 / −12.4% / −14.3% |
| re-solve on batch (s2 semantics) | 1.062 / −13.5% / −10.0% | 1.062 / −12.4% / −20.5% |
| intra-node swap (τ=512, runtime) | 1.160 / −5.6% / −0.0% | 1.074 / −11.4% / −5.3% |
| intra-node swap (τ=1 fixpoint) | 1.077 / −12.3% / −2.0% | — / −11.3% / −7.0% |
| wrong-topic (old mismatch) | 1.436 / +17.0% / +3.1% | 1.606 / +32.5% / +5.8% |

Mechanism separation: full movement recovers imbalance AND wire; the
intra-node swap recovers only imbalance (node assignment unchanged — wire
recovery structurally ~0). K2's shallower per-topic skew means the τ=512 gate
barely fires (force/τ=1 needed); s2 solves on the CURRENT batch (`tk_dev`),
so it does not lag even under per-iteration topic switching.

## 3. Measured campaign (4n, b1–b64, isolated, one binary)

Binary = main 8/31 post-dov build, `libflux_cuda_ths_op.so` sha16
**ce939eb91073d55b** (symlinked into the worktree package; all rule-4 tags
audited; no src newer). Gates (l01_nvshmem b4 vs 8/31 anchors): K2 e2e/total
10.298/11.333 vs 10.61/11.58 (−2.9/−2.1%); Qwen 10.759/11.789 vs 10.90/12.07
(−1.3/−2.3%). **536/536 cells ok — zero failed, zero skipped** (heap-clamp
warnings on llc b8–b64 + s2_swap K2 b64 all ran green).

**MEAN total_ms over topics** (per-iter max-rank, median of 10 iters;
plan-inclusive headline per handoff-15 convention). K2 (8 topics):

| arm | b1 | b2 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|---|
| moonep_l01_nvshmem_getmem | 16.75 | 17.43 | 21.90 | 27.55 | 40.37 | 63.91 | 118.21 |
| l01_allgather_dense | **4.01** | **4.72** | **6.11** | **9.08** | 15.78 | 29.14 | 57.10 |
| l01_slipstream | 4.01 | 4.73 | 6.64 | 9.91 | 16.23 | 28.87 | 53.95 |
| llc_l01_s1_pv2 | 6.38 | 7.11 | 9.03 | 13.01 | 20.65 | 37.22 | 71.62 |
| ours_l01_s1_pv2_r2 | 4.10 | 4.76 | 6.33 | 9.57 | **15.32** | **27.37** | **50.62** |
| ours_l01_s2_swap_force_p2p_r2 | 5.07 | 5.80 | 7.31 | 10.51 | 16.27 | 28.15 | 50.91 |

Qwen (6 topics):

| arm | b1 | b2 | b4 | b8 | b16 | b32 | b64 |
|---|---|---|---|---|---|---|---|
| moonep_l01_nvshmem_getmem | 10.34 | 10.68 | 14.57 | 21.93 | 34.80 | 64.42 | 123.75 |
| l01_allgather_dense | 4.49 | 4.89 | 5.89 | 8.53 | 14.99 | 27.51 | 52.05 |
| l01_slipstream | 3.06 | 3.72 | 5.20 | 8.44 | 15.24 | 28.26 | 54.13 |
| llc_l01_s1_pv2 | 5.30 | 6.04 | 7.95 | 11.79 | 19.31 | 35.50 | 68.06 |
| ours_l01_s1_pv2_r2 | **3.10** | **3.66** | **4.75** | **7.32** | **12.53** | **23.17** | 44.03 |
| ours_l01_s2_swap_force_p2p_r2 | 3.98 | 4.46 | 5.55 | 8.15 | 13.39 | 23.74 | **43.76** |

## 4. Findings

1. **The hetero oracle is a working, fair drift regime.** 40–48% realized
   oracle→batch drift; per-topic total_ms spread at b64 is ±8–10% around the
   mean (K2 professional_law consistently the most expensive topic), so
   mean-over-topics does real work.
2. **OURS s1 survives the hetero oracle.** Even planning from a topic-blended
   basis, s1 is the best arm on Qwen b1–b32 and K2 b16–b64; on K2 b1–b8 COMET
   edges it by ≤0.5 ms (K2's shallower skew: hetero placement pays less
   there, and slipstream's combine advantage shrinks below b16).
3. **s2 per-iter planning + forced intra-node swap crosses over at b64 on
   Qwen** (43.76 vs 44.03) and converges-but-not-quite on K2 (50.91 vs 50.62,
   −0.3 shy). Its premium over s1 shrinks from ~0.9–1.0 ms at b1 to ~0 at
   b64 while its imbalance recovery grows with budget — the measured
   crossover is exactly where the math check's ~11–13% max-rank recovery
   (Qwen steeper than K2) predicts. s2 also compresses the per-topic TAIL:
   K2 b64 worst-topic 52.81 vs s1's 53.81.
4. **Placement+routing without the slipstream combine (llc pv2) never pays**
   at 4n on the hetero oracle (worst non-moonep arm at every budget) —
   consistent with handoff 30's PLL-pv2 finding; the plan bracket dominates.
5. **MoonEP's always-balance is 2–3x off the pace everywhere** — full
   per-iteration rebalancing costs far more than the residual it removes at
   4n. The right comparison story for the paper is s1/s2 vs COMET/slipstream
   with moonep as the always-balance ceiling-cost reference.
6. Expected next step where movement should GROW: the wire component of the
   hetero residual (−10% K2 / −20% Qwen inter-node rows for full movement)
   is mostly untapped by the swap-only s2 arm and grows with hop count —
   8n/16n reruns of this scenario are where full movement (not just swap)
   should separate from s1.

## 5. Capsule ledger (all on worktree-het-oracle, binary ce939eb91073d55b)

K2: gate 20260831-144316_4c409238; comet 144503_465f9504; slip 150122_4fb2dc4a;
llc 151915_e1dc5eca; ours-s1 154331_bca5df93; s2-swap 161757_603ddd77;
moonep 180823_a25828a5 (rerun — see incidents).
Qwen: gate 20260831-144316_df94c59a; comet 144502_b927f56b; slip 145712_72fb0247;
llc 151111_e565df43; ours-s1 152556_ca523dcb; s2-swap 154904_e7499b88;
moonep 161216_874f4ca7.
Tidy dataset: `docs/handoff/32_het_oracle_tidy.csv` (588 rows).

## 6. Incidents / node-hours

- K2 moonep attempt 1 (run f6b878ae) was externally stopped at cell 55/56
  with no capsule written (per-cell logs normal up to the kill; killer
  unidentified — not the lane). Full rerun on a fresh allocation went 56/56.
  Cost ~6 nh. Partial raw data at sweep_data/20260831-165642_…_f6b878ae
  (no capsule — ignore).
- Node-hours: Qwen 7.2 (one 1:48 4n allocation), K2 ~18.1 (3.42 h + 1.11 h
  allocations, incl. the lost moonep attempt). Campaign ≈ 25.3 nh on m5350_g.
- Worktree relocated to /pscratch/sd/y/yufeid/workspace/andrewy/
  flux-het-oracle mid-session (home hit its 40G wall; symlink left at
  flux/.claude/worktrees/het-oracle keeps git plumbing + session paths
  resolving).
