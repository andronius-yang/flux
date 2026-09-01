# Handoff 33 — Hetero-oracle severity: scenario math check + LOO GPU A/B (2026-08-31/09-01)

**Question (user, 8/31 eve):** handoff 32 showed s1 surviving the equal-blend
hetero oracle (swap premium only repaid at b64). Find an *extreme but
realistic* scenario — real traces only, no synthesized fanout skew — where
always-swap intra-node balancing (`ours_l01_s2_swap_force_p2p_r2`) pays off
severely in total_ms. Constraint: inter-node expert movement is OFF the
table (wire movement is the limiter; no overlap slack under the dispatch
wire).

## 1. Scenario math check (offline, same-code, CPU — 4n)

Driver: `33_hetloo_scenario_mathcheck.py` (this dir; results JSON alongside).
Same-code chain: `pv2_solve` -> `ours_swap.swap_orbit` (tau=1 fixpoint on the
eval batch = tk_dev semantics) -> LocCap `simulate_arm` (eps 0.0625, r2 slots
nlp=G/W+2). Eval = per-topic homog [64,96) batch, T from the b4 budget,
2 seeds, mean over topics. Arms: matched (own-topic [32,64) basis) / static
(scenario oracle basis) / +swap / full re-solve (bound; needs movement).

Mean static->swap imbalance (rmax delta % vs static; resolve % as bound):

| scenario | K2 static->swap (swap%, resolve%) | Qwen static->swap (swap%, resolve%) |
|---|---|---|
| anchor (equal blend, = handoff 32) | 1.183->1.086 (-8.1, -10.0) | 1.255->1.100 (-12.3, -15.3) |
| mix2tv (2 most-TV-distant pools) | 1.103->1.062 (-3.7, -2.8) | 1.201->1.082 (-9.9, -11.5) |
| advperm (hot<->cold permuted twin; synthetic ceiling probe) | 1.177->1.066 (-9.5, -8.0) | 1.319->1.098 (-16.7, -19.5) |
| **loo (oracle mix EXCLUDES eval topic)** | **1.414->1.138 (-19.5, -24.8)** | **1.435->1.138 (-20.7, -26.0)** |
| minority (eval topic 1/16 weight) | 1.279->1.089 (-14.9, -16.8) | 1.339->1.112 (-16.9, -20.6) |
| burst_b4 (block-32 request sampling) | 1.258->1.087 (-13.6) | 1.218->1.117 (-8.3) |
| burst_b1 / iid_b1 control | 1.224 vs 1.249 (~equal) | 1.250 vs 1.248 (~equal) |

Findings:
1. **Severity is monotone in how little the oracle saw of the eval topic**:
   equal blend 1.18-1.26 -> 1/16 share 1.28-1.34 -> never-seen (LOO)
   1.41-1.44. LOO is the realistic extreme; swap recovers 75-90% of the
   full-resolve rmax gap in every scenario, with ~zero wire recovery
   (structural — node assignment untouched).
2. **Pool-selection decorrelation is a dead end** (mix2tv < anchor on both
   models): a 2-pool blend is half-matched to each side.
3. **Intra-topic burstiness is a dead end**: block-by-request sampling ==
   iid at both b4 and b1 on both models (real per-request routing within a
   topic is close to the topic marginal).
4. Per-topic LOO picker (Qwen, b4): college_mathematics steepest (static
   imb 1.770, rmax 7248), philosophy 1.577, ZH college_math 1.725; TV range
   0.233-0.573 (max pair = EN world_history vs ZH college_math — language
   decorrelates more than subject).

## 2. GPU A/B (Qwen, LOO oracle excl. college_mathematics, homog college_mathematics eval)

Specs `sweeps/specs/hetloo_qwen_{4,8}n.yaml` (budgets 1,2,4,8,16,64 per user
ask — b32 deliberately absent). Arms in ONE capsule per topology:
`ours_l01_s1_pv2_r2` (static LOO-basis placement) vs
`ours_l01_s2_swap_force_p2p_r2` (per-iter solve + forced intra-node P2P
swap). Isolated mode, 10 iters, deterministic=0. Per-iter max-rank, median.

**4n** (gate a0c34746 GREEN: b4 nvshmem 10.496/11.746 vs anchor
10.904/12.072, -3.7/-2.7%; arms capsule **20260901-020046_8b32ee18**,
12/12 ok, alloc 57800982 ~0.52 nh):

| total_ms | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| s1 | 3.18 | 3.81 | 5.19 | 7.48 | 13.29 | 49.00 |
| s2 | 3.92 | 4.49 | 5.58 | 8.11 | **13.17** | **44.12** |
| s2 vs s1 | +23.3% | +17.8% | +7.5% | +8.4% | **-0.9%** | **-10.0%** |
| e2e delta | -1.7% | -1.4% | -1.3% | -2.6% | **-7.7%** | **-12.4%** |

**8n** (gate b24b7364 GREEN: 16.365/17.689 vs anchor 16.140/17.509,
+1.4/+1.0%; arms capsule **20260901-021323_59f6ad19**, 12/12 ok, alloc
57801006 7:54 = 1.05 nh):

| total_ms | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| s1 | 5.35 | 6.15 | 7.63 | 10.87 | 18.11 | 62.66 |
| s2 | 6.41 | 7.35 | 8.66 | 12.29 | 20.10 | 64.15 |
| s2 vs s1 | +19.8% | +19.5% | +13.5% | +13.1% | +11.0% | +2.4% |
| e2e delta | +3.4% | -3.1% | 0.0% | -0.3% | +2.5% | +1.1% |

## 2b. K2 4n GPU A/B (added 9/1, user-directed)

Spec `hetloo_k2_4n.yaml`: LOO oracle = 7 K2 pools excl. professional_law,
eval = homog professional_law (per-topic picker: static imb **1.970**, the
runaway steepest across both models; next K2 topic 1.469). Gate capsule
1ae41cf5: 12.944 total vs the 8/31 execution-family anchor 11.584 =
**+11.7% — a TOPIC-LOAD CONFOUND, not drift** (the gate family follows the
spec's eval topic; professional_law is K2's most expensive topic — handoff
32 §4.1; true build drift is bounded 1-4% by the same-binary Qwen gates).
Arms capsule **20260901-040925_1d60b9d8**, 12/12 ok, alloc 57804191
(~10 min, ~0.7 nh):

| total_ms | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| s1 | 4.35 | 5.09 | 7.98 | 11.25 | 18.88 | 72.20 |
| s2 | 5.22 | 6.16 | **7.78** | 11.54 | **17.38** | **55.19** |
| s2 vs s1 | +20.0% | +21.0% | **-2.5%** | +2.6% | **-7.9%** | **-23.6%** |
| e2e delta | -1.1% | -0.2% | -15.7% | -9.3% | -14.5% | **-25.7%** |

**The strongest severity result of the campaign**: -17.0 ms total at b64
(-23.6%), e2e -18.0 ms (-25.7%), total crossover at b4, e2e wins at every
budget. K2's larger H (7168, chunk 14336) makes each recovered row worth
more milliseconds, and professional_law's 1.97 LOO residual is the deepest
realistic imbalance found.

## 2b-full. K2 4n LOO 6-arm ladder (added 9/1, user-directed)

Full ablation roster in ONE capsule (**20260901-042235_f7603e26**, 33/36
ok, alloc 57804409 TIMEOUT 30:09 ~2 nh): moonep (always-balance incl.
cross-node) / COMET dense / slipstream (comm only) / llc-pv2 (placement+
routing only) / s1 (placement+routing+comm, static) / s2 (+ per-iter
intra-node force-swap). total_ms (per-iter max-rank, median):

| arm | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| moonep | 14.93 | 16.34 | 22.37 | 32.44 | 47.16 | 129.66 |
| comet | 4.05 | 4.89 | 6.51 | 9.74 | 17.27 | 63.51 |
| slipstream | 4.14 | 4.82 | 7.78 | 10.14 | 17.39 | 58.19 |
| llc-pv2 | 6.76 | 8.20 | 11.24 | 17.57 | 31.32 | stuck |
| s1 | 4.30 | 5.17 | 7.88 | 11.14 | 19.08 | 72.20* |
| s2 | 5.20 | 6.12 | 8.14 | 11.30 | **17.34** | **55.19*** |

\* s1/s2 b64 from the same-binary same-matrix 2-arm capsule 1d60b9d8: the
llc b64 cell went STUCK for 654 s (heap sizing clamped at the 16G platform
cap — the handoff-17 llc/PLL b64 class, first seen at 4n here) and burned
the allocation; the three trailing b64 cells failed at 0 s on the expired
allocation. Not a code failure — 1d60b9d8 ran both cells green.

**The ladder reframes the story**: under LOO drift the static placement is
FRAGILE — s1 loses to plain COMET from b4 up (b64: 72.20 vs 63.51, s1
+13.7% WORSE than no placement), because a placement solved on the wrong
basis concentrates load worse than the neutral contiguous layout. The
always-swap arm rescues exactly that fragility: s2 b64 beats slipstream
(-5.2%) and COMET (-13.1%), and at b16 matches the best baselines while
s1 trails ~10%. b64 ranking: **s2 > slip > comet >> s1 >> moonep**; llc
never pays anywhere (and moonep is 3-4x off at every budget). So the
honest ablation claim is not "swap improves our stack" but "**runtime
intra-node rebalancing is what makes placement SAFE under drift** — and
then best-in-field."

## 2b-qwen. Qwen 4n LOO 6-arm ladder (added 9/1) — BOTH PREMISES HOLD

Capsule **20260901-045411_3062cc8a**, 36/36 ok, alloc 57805035 (~1.3 nh).
total_ms:

| arm | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| moonep | 8.36 | 11.03 | 13.52 | 20.26 | 33.73 | 119.33 |
| comet | 4.24 | 4.72 | 5.58 | 8.15 | 14.09 | 49.23 |
| slipstream | **3.09** | 3.67 | 5.40 | 8.48 | 14.48 | 54.20 |
| llc-pv2 | 5.47 | 6.77 | 8.97 | 14.05 | 23.95 | 91.42 |
| s1 | **3.09** | **3.68** | **4.85** | **7.70** | **13.22** | 49.12 |
| s2 | 4.08 | 4.67 | 5.54 | 8.28 | 13.51 | **44.12** |

Unlike K2, **s1 beats COMET at EVERY budget** under LOO (b1 -27%, b4 -13%,
b16 -6%, b64 tie) and beats slipstream from b4 up — Qwen's deep structural
expert skew means even a wrong-basis placement+routing outperforms the
contiguous layout. And s2 still wins the money column: b64 -10.2% vs s1,
-10.4% vs COMET, best arm outright. Note the margin EROSION: s1's lead
over COMET shrinks to ~0 at b64 — drift burns the static margin exactly
where bytes are largest, and the swap restores a clear -10%. The
two-premise scenario (all optims >= COMET, s2 >> s1) therefore EXISTS
as-is on Qwen 4n LOO; K2 professional_law LOO is the over-rotated corner.

**K2 dial (offline, dial_sweep_k2.py, single seed):** COMET-contiguous
rmax for the professional_law eval batch is 5219 (imb 2.204) — WORSE than
even the LOO pv2 static (4789): the K2 GPU flip is NOT raw GEMM balance
but the wire/overlap side (COMET's dense allgather is drift-immune while
a wrong-basis sparse dispatch concentrates hot lanes). Minority-weight
dial: at w = 1/2-of-equal share (~6.7% of the blend) static rmax is
still -28% vs contiguous while the swap gain rises to -28.3% (vs -11.8%
at equal blend) — the candidate K2 two-premise cell if one is wanted on
K2 as well (one 6-arm capsule at that weight would confirm).

## 2c. CPU decomposition of the 8n collapse (Qwen, anchor+loo, 1 seed)

het_scenarios_8n.json (job scratch; NODES=8 W=32): anchor 1.340->1.169
(swap -12.8% rmax), **loo 1.938->1.346 (swap -30.6%, resolve -45.2%)** —
the LOO residual EXPLODES at 8n and the swap's offline recovery GROWS,
yet the GPU 8n A/B shows s2 >= s1 everywhere. Conclusion: the 8n collapse
is pure TRANSLATION failure — the wire share dominates the bracket at 8n
and the swap recovers zero wire (inter rows unchanged, ~62k) — not a
shortage of recoverable imbalance. (K2 8n anchor partial datum before the
login-node kill: 1.291 static, swap -14.4% — same direction.)

**16n (landed 9/1, 33_hetloo_scenario_mathcheck_16n_qwen.json)** completes
the scaling law: LOO static imb 1.435 -> 1.938 -> **2.105** at 4/8/16n
(anchor 1.255 -> 1.340 -> 1.444), while the swap's share of the resolve
gap collapses **80% -> 68% -> 25%** (swap -12.3% vs resolve -49.5% at
16n). With 2 experts/GPU and 8/node at Qwen 16n the residual is almost
entirely BETWEEN nodes — structurally unreachable by any intra-node
mechanism. This is the quantitative form of the verdict-3 handover:
drift-time imbalance grows with scale, but past the island size it can
only be addressed by node-level placement (the movement/wire regime). Login-node NOTE: three pure-background attempts of
this script were externally killed minutes in (same unidentified-killer
signature as handoff 32's moonep incident); the foreground-migrated
invocation survived.

## 2d. Postdoc scenario check (9/1): per-rank topic alternation — NO capturable imbalance

Proposal: keep the all-inclusive blend oracle, but assign EACH RANK its own
single in-pool topic (round-robin) instead of homog. Same-code verify
(33_hetloo_perrank_check.py, 2 seeds): per-rank static imb = **1.062 on
both models = the LocCap (1+eps) capacity floor**; the swap orbit finds
ZERO swaps and full re-solve recovers 0.0%. Per-node variant: K2
anchor-level only (1.11-1.19, swap captures all of it), Qwen at floor.
Mechanism: grouped-GEMM load is a function of the GLOBAL routed expert
histogram; summing one-topic batches across ranks reconstructs the blend
the oracle planned for — per-rank topic identity moves the structure to
the WIRE (per-source destination patterns), not to row balance. Balanced
per-rank alternation is therefore a wire scenario, not a balance one.
Salvageable variant if rank-level heterogeneity is wanted: SKEWED per-rank
assignment (majority of ranks on one topic) — shifts the global histogram
continuously toward the minority/LOO regime while keeping per-rank
single-topic realism.

## 2f. Swap-OVERLAP ablation (9/1, user-directed port) — expert-dispatch overlap isolated

New ABLATION-ONLY machinery (commit on this branch; python-only, no .so
rebuild): `--swap_overlap 0` on the ours driver makes the current stream
wait the swap lane's ev_done right after issue_early — the exchange LANDS
before anything downstream is enqueued (swap first, THEN dispatch;
requires issue=early; exposed exchange sits in the place bracket:
total_ms sees it, e2e does not). Three ablation_-prefixed arms in
variants.py (never headline): ablation_l01_s2_swap_noov_p2p_r2 (canon
comm, sequential swap), ablation_l01_pr_swap_p2p_r2 /
_noov (legacy comm bracket — wave-adapt/combine-idx/plan-graphs OFF,
plan_overlap 0 — with overlapped / sequential swap).

Capsules (4 arms each, one capsule, LOO cells as §2b): K2
**20260901-061332_c20b66a6** (24/24), Qwen **20260901-063110_df91c5c7**
(24/24). Serialization cost (noov - ovl, total_ms, s2 stack):

| model | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| K2 | +0.53 | +0.55 | +0.17 | +0.68 | +0.72 | +0.46 |
| Qwen | +0.18 | +0.09 | +0.15 | +0.02 | +0.08 | +0.19 |

Findings: (1) the overlap hides essentially the ENTIRE per-iteration
exchange — the overlapped arm shows no residual spin penalty; (2) the
exposed cost is BUDGET-FLAT and scales with EXPERT BYTES (K2 ~59 MB pair
-> ~0.5-0.7 ms; Qwen ~25 MB -> ~0.1-0.2 ms), i.e. large relative at small
budgets (K2 b1: ~10% of total) and negligible at b64 (<1%); (3) the pr
(legacy-comm) twins show the same magnitudes — expert-dispatch overlap is
INDEPENDENT of the slipstream token-side comm canon (separable
mechanisms). Design note: per iteration each rank exchanges at most ONE
expert pair (up to L/2 swaps/node); convergence accumulates across
iterations, so the hidden amount is bounded by per-iter movement — a
rounds-per-iteration extension would scale both the movement and the
ablation gap.

## 2g. Swapall ablation: reset-every + full-orbit composed exchange (9/1)

New ABLATION-ONLY machinery (committed this branch, python-only):
`--swap_reset {off,every,postwarmup}` restores the oracle-basis placement
— tables AND physical slot weights from pristine device copies — in the
untimed inter-iteration gap, so timed iterations re-execute the full
drift-event rebalance instead of the warmup-converged fixpoint.
`--swap_rounds all` + `--swap_max_moves 8`: capped tau=1 orbit at
decision time (round-granular truncation; K2 LOO: 8 rounds, 52
slot-moves, max 8/rank) executed by OursSwapAllLane as the COMPOSED
intra-node permutation in one push/signal/pull phase (pushes precede
pulls in stream order; deadlock-free; per-slot final-write gates).

Capsules (canon force ref [NO reset — continuity anchor, NOT
apples-to-apples vs the reset arms] + swapall ovl + swapall noov):
K2 **20260901-070601_bf186315**, Qwen **20260901-071849_403cd4bf**,
18/18 each. total_ms; the measurement = swapall noov - ovl:

| model | b1 | b2 | b4 | b8 | b16 | b64 |
|---|---|---|---|---|---|---|
| K2 overlap saving | +1.61 | +1.37 | +1.17 | +1.74 | +1.91 | +4.61 |
| Qwen overlap saving | +1.35 | +1.25 | +1.58 | +0.95 | +1.45 | +1.07 |

Findings: (1) with real per-iteration movement (~5-8 slots/rank, up to
~470 MB on the hottest K2 rank) the overlap recovers 1-2 ms/iter flat
and 4.6 ms at K2 b64 — the concrete overlapped-expert-dispatch number;
(2) at small budgets even the overlapped twin cannot fully hide 8
slots/rank (dispatch+GEMM span < exchange time), so the ovl-noov delta
UNDERSTATES total movement there; (3) the swapall arms carry a large
timed premium vs canon force (K2 ~+13 ms at b1): the per-iteration
capped-orbit host path costs ~5 ms (above the sub-ms target — numpy
single-conversion rewrite identified as the fix, ~0.5-1 ms reachable)
plus the un-hideable movement. Both swapall arms run the FULL slipstream
comm canon; only --swap_overlap differs. OPEN (user decision): reset-
every single-swap reference rung; legacy-comm swapall pair (2x2 cross);
numpy fast path.

## 2h. Drift-event payoff: postwarmup reset, iteration-resolved (9/1, K2 4n LOO b64)

Fastpath landed first: swap_orbit_capped rewritten single-conversion
numpy (~5 -> **2.0 ms** for the full 8-round decision; post-fixpoint
iterations pay only the ~0.3 ms empty-plan check — under postwarmup the
decision cost sits once, in the event iteration). True sub-ms for the
event itself needs numba (env change) or a compiled core (.so rebuild) —
parked pending user sign-off.

Capsule **20260901-074630_a6e16e81** (5/5): COMET / canon force (no
reset) / force+postwarmup / swapall+postwarmup ovl / noov. Per-iteration
max-rank total_ms, timed iters 0-9 (reset fires before iter 0):

| arm | it0 (event) | rest mean | rest median | full mean |
|---|---|---|---|---|
| comet | 63.28 | 63.84 | 63.76 | 63.78 |
| force canon (no reset) | 55.88 | 60.93 | 58.65 | 60.43 |
| force + pw reset | 71.46 | 60.88 | 56.59 | 61.93 |
| **swapall + pw, overlapped** | **63.33** | **57.01** | 56.99 | **57.64** |
| swapall + pw, sequential | 69.33 | 56.86 | 55.91 | 58.11 |

Findings:
1. **Pre-paying the whole rebalance in ONE overlapped iteration wins.**
   The drift-event iteration costs 63.33 — COMET PARITY — because the 53
   slot-moves hide under the b64 dispatch; every following iteration sits
   at the ~57.0 floor (-11% vs COMET). Window mean 57.64 = -9.6% vs COMET.
2. **The overlap is worth -6.0 ms in the event iteration** (sequential
   twin: 69.33) — the concrete "issue and hide the expert transfers
   first" number.
3. **Gradual one-swap-per-iteration does NOT converge cleanly under
   force**: force+pw spikes 70-72 every ~3 iterations (the fixpoint
   oscillation revisits bad placements) — rest mean 60.88 vs swapall's
   57.01, so the up-front rebalance beats gradual by ~3.9 ms per
   steady-state iteration AND kills the spikes. The canon force arm (no
   reset) shows the same periodic 69-70 spikes — it has been paying
   oscillation costs at b64 all along.
4. Design implication (CANDIDATE, user decision): the production s2
   should be tau=1-converge-then-hold (swapall semantics: solve to
   fixpoint, move once, then only the ~0.3 ms empty check per iteration)
   rather than force-oscillate — strictly better steady state in this
   trace.

## 2i. CLEAN ABLATION DATASET (9/1, user-directed): K2 4n LOO, b4/16/64

Capsule **20260901-091051_5be2f7eb** (18/18, one binary, one allocation).
Six arms forming the full ablation ladder; the three swapall arms use the
postwarmup reset (drift event fires in timed iter 0). Arm 4 note: the
"placement/routing+swap only" rung is the legacy-comm PROXY on the ours
driver (wave-adapt/combine-idx/plan-graphs OFF, plan_overlap 0), not the
epic-driver llc transport.

Medians (per-iter max-rank, 10 timed iters — steady-state dominated):

| arm | b4 | b16 | b64 |
|---|---|---|---|
| moonep | 22.29 | 46.60 | 130.58 |
| comet | 6.56 | 16.94 | 63.24 |
| slipstream (comm only) | 7.52 | 17.15 | 58.26 |
| pr + swapall, sequential | 9.10 | 18.08 | 56.85 |
| full + swapall, sequential | 7.26 | 16.99 | **55.43** |
| full + swapall, overlapped | 7.59 | 17.32 | 57.25 |

Drift-event iteration (it0) vs post-event mean (it1-9):

| arm | b4 ev/rest | b16 ev/rest | b64 ev/rest |
|---|---|---|---|
| pr + swap, seq | 22.70 / 9.73 | 25.36 / 18.64 | 68.46 / 57.96 |
| full + swap, seq | 21.07 / 7.95 | 25.14 / 17.82 | 68.78 / 56.52 |
| full + swap, ovl | 20.28 / 7.96 | **20.75** / 17.59 | **64.01** / 57.18 |

Readings:
1. **Expert-overlap prices the event**: ovl vs seq event iteration =
   -0.8 / **-4.4** / **-4.8** ms at b4/16/64; at b64 the overlapped
   event lands at COMET PARITY (64.01 vs 63.24), so even a
   one-iteration topic residency does no worse than COMET while every
   further iteration gains ~-6 ms.
2. **Token-comm contribution under identical placement+swap** (pr vs
   full, seq): steady-state -1.8 / -0.8 / -1.4 ms.
3. **Placement+swap contribution on top of comm-only** (slip vs
   full-ovl steady): +0.4 / +0.4 / -1..-3 ms — placement pays at b64,
   costs slightly at small budgets (plan bracket).
4. Steady-state ovl-vs-seq medians agree within run noise (+-2%) — the
   twins share the same converged placement; ALL the overlap value is in
   the event iteration, which is exactly the ablation claim.
5. Event iterations at b4/b16 carry a fixed ~8-12 ms cost beyond the
   NVLink bytes (orbit 2 ms + apply/refresh + gate stalls where
   dispatch+GEMM is too short to hide movement) — the known
   small-budget limit from §2f/§2h.

## 2j. THE four-tier ablation data point (9/1): K2 4n LOO b64, 5 repetitions

User ask: one data point showing COMET > {comm-only, placement-only} >
combined-unoverlapped > combined-overlapped. Structurally b64-only (at
b4/b16 COMET is FASTEST — dense allgather is cheap when bytes are small)
and window-mean-over-the-drift-event (postwarmup, 1 event + 9 steady):
the only cut where combined-seq amortizes below the singles while still
differing from combined-ovl. Single-run steady noise (~+-0.7) had flipped
the last rung; 5 same-allocation repetitions resolve it. Capsules
a03516f9 / 4695b36d / 788d611f / d19ddb24 / e3c5ff32 (25/25).

| tier | arm | window mean +- sd |
|---|---|---|
| 1 (worst) | COMET | 64.35 +- 0.44 |
| 2 | pr placement+swap, seq | 59.53 +- 1.59 |
| 2 | slipstream (comm only) | 58.66 +- 0.80 |
| 3 | full stack, swap UNoverlapped | 58.12 +- 0.52 |
| 4 (best) | full stack, swap OVERLAPPED | **57.13 +- 0.22** |

seq - ovl window delta: +0.99 +- 0.24 (sem), POSITIVE IN ALL 5 REPS
(+0.41..+1.53); event-iteration delta +7.55 mean (all reps +5.9..+9.2).
Softest boundary: slip (58.66) vs full-seq (58.12) — 0.54 margin, 3/5
reps cleanly ordered, correct in expectation. The ladder as a claim:
each mechanism contributes — comm overlap and placement+swap each beat
COMET alone, combining them beats either, and overlapping the expert
movement is a further, statistically solid -1.0 ms on the window (-7.5
ms at the drift event itself).

## 3. Verdict

1. **The severe case exists and is 4n LOO**: s2 always-swap crosses at b16
   and wins b64 by **-4.9 ms total (-10.0%) / -5.75 ms e2e (-12.4%)** —
   vs -0.27 ms (-0.6%) at b64 under the equal blend (handoff 32). In the
   e2e bracket s2 wins at EVERY budget; the small-b total_ms losses are the
   flat ~0.7-0.9 ms host plan/apply premium. The CPU-predicted chain (LOO
   1.77 static imb -> swap recovery -> rows-bound ms) closed quantitatively.
2. **8n is the honest boundary: intra-node swap leverage collapses.** s2
   pays at every budget (total +2.4% even at b64), e2e a wash. Consistent
   with structure: at 8n the LOO residual increasingly sits BETWEEN nodes
   (swap cannot touch node assignment; the 4n math check already showed
   ~zero wire recovery), Qwen r2 headroom becomes 50% (2 spare slots on 4
   experts/GPU) so replication pre-absorbs hot experts, and the wire share
   of total grows, diluting the rows-bound component. The 8n/16n CPU
   extension (running at session close) will decompose shrink-of-residual
   vs failure-to-translate.
3. Paper framing: the always-swap contribution = **latency under topic
   drift at moderate scale (<=4n island) and the tail** (worst-topic
   compression, handoff 32 §4.3); at >=8n the story hands over to
   node-level placement (which is the movement/wire regime, currently
   off the table).

## 4. Incidents

- **Binary drift (rule 4).** The 8/31 campaign binary ths_op
  `ce939eb91073d55b` no longer exists: another session rebuilt the main
  tree's `python/flux/lib` (worktree symlink target) 8/31 ~17:55 PDT.
  All four hetloo capsules ran ths_op `505e4bedb6d1502f` / libflux_cuda
  `3d251dba29378431`. Within-capsule A/Bs valid; tonight's 4n-vs-8n
  same-build; NEVER-MIX against the 8/31 hetero campaign capsules. Gates
  green within +-5% of the ce939eb9-era anchors bound the drift (~1-4%).
- Both lane subagents stalled waiting on watchers that die on agent stop;
  orchestrator drove the runs directly (dup gate run 021256_1be08650 killed
  before its srun; no capsule dir left). One lane/orchestrator srun overlap
  risk resolved by pid-kill (never pkill -f).
- Self-oracle scope re-verified in the 98eb2c8 diff: only the epic-driver
  (llc) token list was broken 8/27-8/31; the ours driver always received
  --oracle_routing_file. llc lost everywhere even WITH self-oracle basis —
  a fortiori.

## 5. Ledger

Capsules (branch worktree-het-oracle, commit c0810d1): 4n gate
20260901-015953_a0c34746, 4n arms 20260901-020046_8b32ee18, 8n gate
20260901-021258_b24b7364, 8n arms 20260901-021323_59f6ad19. Specs commit
98d3b31. Math check: 33_hetloo_scenario_mathcheck.{py,json} (4n; 8n/16n
extension pending at write time). GPU cost ~1.6 nh m5350_g total.
