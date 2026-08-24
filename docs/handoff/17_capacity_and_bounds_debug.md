# 17 — datacamp failure-class debug: llc recv-bound fix + capacity audit (2026-08-24)

Follow-up to handoff 15's failure-attribution ledger (user directive: root-cause
the PLL 16n qwen assertion before the next launch; audit the OOM classes).
All code changes UNCOMMITTED on main. Related: handoff 13 (LLC stack),
memory `llc-recv-bound-fix`.

## 1. llc 16n qwen b4–b32 "recv bound violated" — ROOT-CAUSED + FIXED

**Symptom** (datacamp, 5 cells): every rank asserts at the SETUP audit
(`check_relaxed`, epic_semantics.py:2241) — `recv bound violated: 114 rows
over` (b4) / `839 rows over` (b8); b1–b2 pass; every K2 16n cell passes.

**Root cause** (offline repro on the login A100 reproduced the campaign
numbers exactly: 111@b4 / 836@b8, single violating dst): a **pipeline-stage
mismatch between the two implementations of the same forced-fallback rule**.

- The CUDA kernel (`pll_shares_kernel`, src/cuda/moe_utils.cu:412) computes
  its per-expert forced-fallback table ONCE from the **post-tier-2** load
  (`key = load*R + rank`, load written by q1/t2clip kernels).
- The torch reference `loccap_route_sl` (placelambda_gpu.py:~650) keyed its
  forced rows on the **live tier-3-updated** load (`load += spent` inside the
  round loop) — same key formula, different load snapshot.
- `recv_ub[dst] = load_t2 + Σ_src myshare + 2·Σ_src fp_ref + 8R` budgets each
  destination's forced slack from the REFERENCE's realized forced-pair table
  `fp_ref`. When the two argmins diverge for experts carrying forced flow,
  the kernel concentrates forced ingress on a dst where `fp_ref ≈ 0` — dst 6
  (node 1) here: ref sends it 0 forced rows, kernel sends 686 (b4) / 1464
  (b8) — blowing the fixed 8R = 512 cushion. Everywhere else the two forced
  distributions matched within ~10 rows.

**Why qwen@16n specifically**: forced volume ∝ tier-3 residual. Qwen W=64 has
nlp = 128/64+2 = 4 (replication 2.0) and eps .0625 → ~7% of all entries go
forced (31k rows @b4, 62k @b8); the divergence scales with S → b1–b2 sit
inside the 512-row cushion, b4+ do not. K2 (nlp=8, G=384) keeps the two
distributions aligned and margins ≥ 656.

**Fix** (python-only, no rebuild; placelambda_gpu.py, `loccap_route_sl`):
snapshot `load_fb = load.clone()` right after the share tables (post-tier-2)
and key the forced fallback on `load_fb` — the reference now realizes forced
rows under the kernel's exact static rule, making `fp_ref` predictive again.

**Validation** (all on login A100, module env):
- Full llc 16n ladder, both models, b1–b64 routing (repro4): kernel recv over
  `recv_ub` ≤ 0 on **14/14 cells** — min margin = the full 8R cushion (512)
  at qwen, 656+ at K2; pair bounds clean everywhere; conservation holds.
- `test_placelambda_kernel.py` ALL OK (parity + latency 0.28–0.36 ms/rank),
  `test_loccap_router.py` ALL OK, `test_placelambda_gpu.py` ALL OK incl.
  **CPU/GPU bit-identical** (the cross-device determinism contract survives).

**NEVER-MIX note**: the deterministic loccap_sl reference routing changes
slightly (forced destinations move to the static-key argmin). Correctness
semantics are unchanged (any hosting rank is valid; conservation asserted),
but future llc capsules are routing-non-identical to pre-fix ones at
forced-heavy cells. Sizing tables (`recv_cap`/`pair_cap`) shift only via fp.
Repro scripts: session scratchpad `llcdbg/repro{,2,3,4}.py`.

## 2. PLL/llc 16n b64 heap deaths (K2 + qwen) — AUTHENTIC capacity class

Both b64 cells die in `enable_hier_compress` at the **dispatch op ctor**
(`flux.GemmGroupedV2AGScatterOp`, epic_semantics.py:2712):
`flux_shm.cc:80 NVSHMEM_MALLOC failed` — the heap is exhausted before the
combine panels even allocate. Static arithmetic at b64 (post-fix bounds):
capacity-mode combine (recv_cap≈154k/69k rows qwen/K2) + All2AllSingle
staging (max_split ≥ pair_cap 8850/3863 × W rows × 2 ops × in+out) ≈ 10/4.3
GiB **before** the dispatch panels (recv/stage/relay each ≈ 8× their b8
values ≈ 100k–770k rows × 8 KiB) — total well past the 16 GiB platform cap
(`sym_size_max_g`, platforms/*.yaml). Verdict: same class as the torch
gathered-bytes OOM — **pre-skip in specs** (the 40 GB card cannot host heap +
data plane at b64 regardless of sizer cleverness). The recv-bound fix does
not change this.

## 3. EPLB 16n qwen b64 CUDA OOM — reproducible, NOT authentic

Traceback: `reroute_expand_all_gpu_fast` (ep_gpu_plan.py:488),
`prm = rqp_all.long().reshape(R*G,-1).cummax(1).values[re]` → tried to
allocate **4.03 GiB** = E×Cmax×8 B with E = R·S·K = 64·8192·8 = 4,194,304 and
Cmax = max_replicas_dim = max(R, 1+R·R_red) = **129** (eplb_semantics.py:109)
— exact match. Ambient pressure: 16 G NVSHMEM reservation (row-sum sizer at
the cap) + ~8.6 G torch data plane → the 4 GiB **planner intermediate** tips
a 39.5 GiB card. Deterministic (all ranks OOM identically).

**Authenticity verdict**: a realistic EPLB deployment would NOT OOM here —
the [E, Cmax] int64 gather is a port convenience of our sync-free planner
twin (a production planner walks the quota prefix in O(E) memory), and the
16 G heap reservation is our loose row-sum prior, not EPLB's requirement.
Candidate fix (NOT implemented — `_derive_fast` is inside the timed plan
bracket, so any change is a rule-4/never-mix boundary and a user decision):
chunk the `[re]` gather (4×1 GiB peak) or carry prm in int32 (halves it).

## 4. Runner hardening + audit procedure (sweeps/sweep.py, uncommitted)

Two sizing regimes exist in `build_cell_env`:
- **Exact** (gather_rs / l01 flux arms): knob demands from the matrix replicate
  the ops' FLUX_CHECKs; `_A2AV_SYM_G_REQUIRED` is recorded and run_cell
  **pre-skips** (`skipped_capacity`) when demand > cap. Already safe.
- **Loose row-sum priors** (fast / moonep* / ultraep / eplb / epic): the
  helpers clamp to the cap SILENTLY — this is how the llc/eplb b64 cells ran
  into guaranteed deaths. Auto-skip on these bounds would be WRONG (they
  over-estimate ~4×: llc 16n b8 runs fine clamped), so the hardening is
  visibility: build_cell_env now sets `_A2AV_SYM_G_AT_CAP` whenever a
  loose-bound cell lands at the cap, and run_cell prints a WARNING naming the
  cell (also visible under `--dry-run`).

**Pre-launch audit procedure** (the answer to "make sure this class never
recurs"): run `python sweeps/sweep.py run <spec> --dry-run` for every planned
spec and grep the output for `skipped_capacity` (exact arms — these cells are
already safe) and the new at-cap WARNING (loose arms — decide pre-skip vs
accept per the known-class table below). Known classes to pre-skip at 16n:
torch b32+ (handoff 15), llc/PLL b64 (§2), eplb qwen b64 (§3, until the
planner intermediate is fixed).
