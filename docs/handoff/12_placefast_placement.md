# 12 — PLACE-lambda FAST: per-iteration placement (session 8.22.placefast)

Status 2026-08-22: **landed on branch `place-fast`** (worktree
`../flux-place-fast`, base main 030bb97, python-only against main's
binary; commit 58896c4 + this doc). Report artifact:
https://claude.ai/code/artifact/6e5e1ab0-5017-4975-9e45-0c88383b501d
Memory: `placefast-campaign`. Eval battery + smoke logs lived in session
scratch (numbers reproduced in the report and below).

**USER DIRECTION (2026-08-22, end of session): near-term scope is the
STATIC flavor only — the goal is a same-capsule comparison of static
PLACE-lambda placement against EPIC / EPLB / MoonEP. The dynamic lane
below is validated and parked; weight-dispatch mechanisms are a later
date.**

## What this is

`python/flux/testing/placelambda_fast.py` — the per-iteration
expert-placement solver for the LLC arm (PLACE-lambda placement + LocCap
routing on the epic harness). Same integer objective as the exact
reference (`placelambda_gpu.build_placement_gpu` — kept in-tree), new
schedule: batched bounded passes, exact-integer GEMM engine (one-hot
X [T,G]; every count < 2^24 so fp32 accumulation is bit-exact and
order-free on any device), zero-D2H hot path (no item/nonzero/scalar
index-put; fixed pass counts; masked no-op passes), CUDA-graph-capturable
end to end. NEW ARM (`pllf_*`): never compare its placements bitwise
against `placelambda_gpu` cells (same never-mix rule as loccap_gpu vs
exact loccap).

Headline numbers (job 57423410, 4n, one binary; login-A100 benches):
solve+decision 1.03 s -> 4.8 ms (K3 4n graphed), 2.7 s -> 65 ms (K3 32n,
R=128, offline); dynamic place lane 92.7 -> 2.50 (cover) / 1.94 ms/iter
(proxy) at qwen3 b8; l01 totals fast-dyn 30.8 vs exact-dyn 123.9 (the
exact solver also inflates l0 exec 8.2 -> 5.4 ms via SM contention).
Quality through the real router: parity to better everywhere post-fix
(qwen3 8n −15.6% vs exact; K3 4n −4% confirmed e2e incidence 45,629 vs
47,506, e2e 8.10 vs 8.44 ms; worst observed +0.8% at 32n pA4, parity at
pA8). Unit gates `test/python/moe_ag_scatter/test_placelambda_fast.py`
ALL OK incl. **CPU==GPU bit-identity** (cross-device-oracle
precondition) and graph-replay stability.

## Solver structure (what a later session needs to know before touching)

- **Stage A** batched FM: one [G,NN] delta table per pass (single GEMM
  `Xᵀ[Z|W|z_hY|w_hY]`), every expert nominates its best target, admits
  granted in canonical (dst, delta, g) order via sorted segment
  prefix-sums against destination caps frozen at pass start
  (conservative — sources only shed). First `repair_passes` are BALANCE
  REPAIR: count-unblock swaps (lightest experts INTO the hottest
  over-load node to free count slots) then cheapest-delta evictions
  (zero-load filtered). Eviction-only repair provably deadlocks
  (measured at qwen3 b64; the swap fix flipped that case from +6.7% to
  −4..−7.5% vs exact).
- **Stage B** replication: one [G,NN,K+1] histogram per pass, per-node
  top-slots admit in the reference tie order (two stable sorts).
  passes_b=1 empirically spends all slots.
- **Stage C** (rank-level finalize; snake default, reference LPT option)
  is OFF the hot path — incidence depends only on node-level placement;
  finalize runs at setup / on adopt.
- **Seeds**: `affinity` (cold: per-node demand greedy under
  seed_cnt_cap = ceil(G/NN)+1 — seeding to the full cnt_cap deadlocks
  repair; measured FM-locally-optimal at qwen3 b8), `contig`
  (reference), `warm` (resident primary — the per-iteration path; must
  be device-resident, asserted).
- Knob defaults: cold pA=4 pB=3 repair=2; warm pA=2 pB=1 repair=1,
  keep_bonus=LCM16//8=90090, move_margin=0. Knee: pA=4; pA=8 matches
  exact at 32n. Damping / max_moves_per_pass measured neutral-to-harmful
  — leave off.

## Static vs dynamic — the design (user Q&A 2026-08-22)

Orthogonal axes: exact-vs-fast = which solver (both OURS); static-vs-
dynamic = WHEN it runs. Static = one untimed setup solve (ideal-stale;
scenario 1). Dynamic = warm solve + trigger decision inside the timed
window every iteration (scenario 2). Same planner, two cadences.

### Trigger rule

`gain_ppm = (lb_cur − lb_new)·1e6/lb_cur`, `trigger = gain_ppm >=
--place_gain_threshold_ppm` (default 50,000 = 5% predicted inter-node
wire-row reduction; inherited from the exact lane, not derived — but
validated: same-distribution noise ~0 ppm, mild b64 drift 66,000 ppm
(fires), structural drift 450-620k ppm). Live proof both directions
(job 57423410): resident==fresh arms read 0/0/trigger=0 every iteration
(driver asserts all-iterations-identical — doubles as a determinism
check); stale-resident probe (`FLUX_PLACE_FAST_STALE_RESIDENT=1`) read
lb 45,510→22,525, gain 505,053 ppm, trigger=1, 41 adds / 9 removes.

KNOWN GAP: the trigger is **gain-only** — it returns the movement size
(`moves_add` × bytes/instance) but does not charge for it. The
principled rule is amortization: trigger iff gain_ppm × comm_ms ×
expected-epochs-until-next-shift > adds × bytes_per_instance /
move_bandwidth. Deliberately deferred with the dispatch mechanism (the
cost side depends on HOW weights move). Pre-trigger movement bounds that
DO exist: keep_bonus (resident replicas preferred at equal quality),
move_margin (delta floor on Stage A moves).

### Placement storage (`PlacementStore`)

Device tensors, node-level first: `primary [G]`, `ion [G,NN]` (this pair
IS the placement identity), `hist [NN,G]` = the demand histogram the
placement was **solved on** (the drift reference), `load_e`, lazy
`hosts` (built by finalize only inside `adopt()`), `epoch`.
`moves_from(res)` = ion diff → adds/removes (the movement-bytes
counter), sync-free. CUDA-graph subtlety: resident tensors are captured
BY ADDRESS — adopting = `copy_` into the same buffers, no recapture.

### Comparison, three tiers by cost

1. `drift_ppm(hist_now, hist_ref)` — O(NN·G) L1, ~µs. Quiet-iteration
   prefilter. Wired in the module, NOT yet in the driver lane (smoke
   arms solve every iteration deliberately, to price the worst case).
2. Decision statistic: `cover` (fixed-round masked greedy node cover —
   the router's own rule, comparable and near-realized; min(K,NN) GEMM
   rounds/side; reuses the solver's X) or `proxy` (one-shot
   serving-node incidence, home-if-hosted-else-primary; O(E); upper
   bound, fair because identical on both sides; the large-NN choice —
   env `FLUX_PLACE_FAST_TRIGGER`).
3. Move diff — only consulted on trigger.

Verdict `[lb_cur, lb_new, gain_ppm, adds, removes]` lands in a device
ring buffer; one D2H at teardown; consumable one iteration late
(deferred-trigger — free, since weight movement is asynchronous anyway).

### Scenario-2 policy (from the drift battery)

Warm refinement from a good resident BEATS a cold re-solve under
same-distribution drift (fewer moves AND better incidence) — never
cold-reseed on a routine trigger. Structural drift: warm lands
1.2–1.5× off cold, trigger reports huge gain → escalate passes_a, and
cold-reseed only if gain stays large after escalation.

## Solver input semantics — the "oracle" question

The e2e static cells are **batch-observed**: `build_placement_fast`
consumes the cell's full allgathered routing `topk_all [R,S,K]` (user
design 2026-08-21: the in-pipeline solver observes ROUTING, not oracle
pools). The demand histogram `hist[NN,G]` is derived from it by one
index_add and drives only the SEED; Stages A/B consume token-level
co-occurrence (the X matrix) — a marginal histogram alone cannot supply
that, and the co-occurrence term is precisely what separates
PLACE-lambda from EPLB (which needs only loads). For a pool-oracle
static arm (scenario 1 proper), feed a pool-sampled pseudo-batch, not a
histogram; the offline planner (`predict_placement.build_placement`)
already implements the pool-statistics path for the exact solver.
FAIRNESS RULE for the coming baseline capsule: EPIC/EPLB/MoonEP and
pllf must share the information basis — either all self-oracle on the
batch (what the smoke ladders did: epic baseline ran `load source:
batch`) or all pool-oracle (EPLB gets eplb_load; pllf gets a
pool-sampled pseudo-batch). Never mix bases within a comparison.

## Attribution protocol vs EPIC/EPLB (user ruling 2026-08-22)

The harness (EpicIterPlanner derive tail: splits/scatter-index/vce/
binding) is SHARED — optimizing it speeds d6/evensplit baselines too,
which is correct and required (rule-4 fairness). Arm-private costs:
the routing decision itself (d6≈0; loccap_sl kernel 0.4 ms), the
phys-row allgather (only our router), the placement lane (place_ms;
baselines run 0). Every capsule must include `pll(f)_hc_d6` (d6 on OUR
placement) + the epic-placement d6 baseline, and report:
**plan premium** = plan_ms(ours) − plan_ms(d6-same-placement),
**place premium** = place_ms, vs **payoff** = comm/e2e delta. The arm
wins fairly iff premium < payoff. Same-capsule EPIC numbers already
measured (4n b8, one binary): EPIC placement is near incidence-neutral
(43,803 vs fixed 45,510) vs pllf 17,214; pllf comm 4.21 vs 5.12, l0 e2e
4.80 vs 5.71, l01 e2e 9.81 vs 11.07 — totals still favor EPIC only via
the 18.5 ms torch router (the loccap_sl kernel replaces exactly that).

## Open items / next session order

1. **Static-flavor baseline capsule** (near-term scope): same-capsule
   pllf-static vs EPIC vs EPLB-load vs (MoonEP — different driver;
   decide comparability) at 4n + 8n (`-q regular`; interactive caps at
   4 nodes). Needs the shared-information-basis rule above.
2. loccap_sl kernel composition (replace the 18.5 ms torch router in
   pllf arms) + the shared derive-tail zero-D2H/graph treatment
   (capacity mode already froze shapes); counts-only exchange replaces
   the phys allgather (arm-private win).
3. `predict_placement` sidecar mode `placelambda_fast` (offline CPU
   solve; CPU==GPU identity already unit-proven) → quotable capsules.
4. NVSHMEM heap: pllf K3 l01 needs `NVSHMEM_SYMMETRIC_SIZE=4G`, qwen3
   b64 needs 8G (fast placement's replica spread tips the default 1G;
   the exact twin barely fit). Fold into arm env or size from placement.
5. Movement-aware trigger (needs the dispatch cost model); drift-gated
   cadence in the driver; 16n quality (never evaluated); 32n e2e +
   proxy trigger at scale; chunked-X for b64-class T at 32n.
6. Open design questions posed to the user (unanswered): (a) online
   estimator for epochs-until-next-shift from the stored hist stream —
   drift_ppm's own history is the natural one; (b) proxy-trigger bias
   vs topk/NN structure — where consolidation onto shared remote nodes
   matters, cover is the safer statistic.
