# 08 — Follow-up queue after the comm-only layer-axis campaign

> **STATUS 2026-08-17**: items 1–4 DONE (commits 58efc02 + docs follow-up;
> smoke capsule `20260817-025014_perlmutter_cc9e7e98`, uncommitted — human
> commits capsules). User-directed deviations from the queue as written:
> the FANOUT arm was NOT deleted and the F/E ablation arms were KEPT — all
> eager/FANOUT mechanisms retained but marked CLOSED-LOSER (decisive,
> case closed, ablation-only); explicit-off arms added as the post-flip
> ablation axis. Item 4 verdict: UNREACHABLE on deployed paths (one latent
> PCIe-only analog recorded in handoff 07 §6, not fixed). Remaining: items
> 5 (l01_fast bring-up) and 6 (16n W64 + first 16n layer1 closure).

Handoff for a fresh session starting the post-campaign work. Read
`docs/handoff/07_comm_only_layer_axis_campaign.md` FIRST — it is the campaign
authority (verdicts, capsule ledger, both bug narratives); `sweeps/SCHEMA.md`
governs all sweep interpretation; `docs/handoff/07_dashboard.html` is the
self-contained visual summary (open locally in a browser). The auto-memory
files (`comm-only-l0-l1-sweep-plan`, `l1-a2av-lazy-load-hang`,
`l1-cascade-empty-expert-hang`) carry the same facts in ledger form.

State of the tree: everything is merged to local `main` (NOT pushed anywhere
except the campaign branch `worktree-comm-sweep-layer-axis` on the fork).
The installed binary in `python/flux/lib/` is **binary C** (both hang fixes;
ths-op sha256 prefix `bd25c058…` was binary B — verify against the newest
capsule manifests, e.g. `20260816-397ac0fa`, before trusting). Worktrees
`comm-sweep-layer-axis`, `l1-hang-debug`, `l1-nn4-debug` are all merged —
purge when convenient (`git worktree remove`), remembering the
lib64-shadows-lib and stale-.so gotchas in the memory ledger.

Environment: `source ./module.sh`; account **m5350_g** (m4243_g exhausted);
salloc+srun only, never sbatch; scancel the moment jobs finish; take jobids
only from your own salloc output (two sessions collided on this once).

---

## 1. Canonicalization patches (highest value, smallest risk)

**What**: make the campaign's winner configs the defaults.
- Flip `FLUX_A2AV_FUSED_STAGE2` and `FLUX_A2AV_EARLY_LAUNCH` to default-ON
  when `FLUX_A2AV_LB_UNION=1` (env reads in
  `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`; EARLY_LAUNCH
  keeps its existing guards — forbidden with PACK_OVERLAP, needs conn>1 on
  compress paths, which the variant env already satisfies).
- Delete the `hier_compress_lb_union_eager` (FANOUT) A/B arm from
  `sweeps/variants.py` — its comment explicitly awaited this verdict —
  and collapse the factorial corner arms into base + explicit ablations.

**Why**: three-run sign-agreement verdicts (handoff 07 §2): F wins
−0.2…−0.7 ms, E wins −0.3…−1.8 ms, N (FANOUT) loses +0.05…+0.6 ms on real
4n trace routing.

**How/validate**: after the default flip, old capsules' env_json means
something different — never byte-compare env across the boundary (add the
same style of dated note the conn=8 pin has in variants.py). Rebuild, then
one 2n correctness-ON smoke of lb_union base (which now silently runs F+E)
+ `--skip_correctness` off; update SCHEMA/SKILL notes. C++ change ⇒ rebuild
required (login builds ≤8 jobs; tests OFF — the test binaries hit a GLIBCXX
link error vs libnvshmem, which is why build.sh must run with `--no_test`
and a fresh cmake cache).

## 2. Combined default pairing = compress

**What**: make `lb_union(F+E) + a2av_hier_compress` the reference combined
configuration: in `sweeps/variants.py`, promote the winning pairing (see
`l01_lbunion_compress` added by the 4n session; align naming), and document
in SCHEMA that combined cells inherit compress's CSRs (amortized semantics)
so compress's isolated-mode build penalty does not apply there.

**Why**: the W=16 best-pairing A/B (capsules `9378fed5`/`397ac0fa`, 14/14
×2) — compress pairing wins EVERY budget: 10.6 ms b8, −52% vs stock,
−45…−52% across b2–b64. Standalone-l1 verdicts differ (hier wins iso at
small budgets) — keep both stories straight per handoff 07 §3.1/§4.1.

## 3. Eager arrival-order reduce: disposition

**What**: decide the fate of `FLUX_A2AV_RS_EAGER`. Evidence: regression at
BOTH scales standalone (+15–35% W=8, +2–11% W=16) AND the entire combined
composition penalty (+18% identity violation at b8, ablation-attributed —
handoff 07 §4). Options: (a) keep as opt-in ablation with a verdict comment
(cheapest, preserves history), (b) delete kernel + knob. Recommend (a) now,
(b) only with user sign-off. Either way, annotate `l1_hier_eager` /
`l1_compress_eager` in variants.py with the verdict so nobody re-runs them
expecting a win.

## 4. gemm_rs `set_barrier_ptr` audit (correctness risk, unaudited)

**What**: the empty-expert split-cascade bug (fixed in moe_gather_rs by
`c9b82b6`) divides the full problem list by the non-empty problem count.
The dense-MLP sibling `src/gemm_rs/` has its own `set_barrier_ptr`
machinery that was NEVER audited for the same pattern.

**How**: read the gemm_rs split-cascade / barrier-pointer code for the same
full-list-index vs non-empty-count mismatch; the trigger is any zero-size
segment. If affected: minimal repro (a split with an empty segment), apply
the analogous one-liner, 2n validation. If clean: record the audit result
in handoff 07's open-items so it stops being listed.

## 5. `l01_fast` (fast+fast combined arm)

**What**: `test/python/moe_combined/test_moe_l0l1_traffic.py --impl fast`
is a deliberate `NotImplementedError` stub. Open question: one
`flash_comm_t` doing TWO alltoallv calls per timed window (dispatch matrix
M then combine Mᵀ) — do FAST's credits/signals need an in-window
`alltoallv_reset` between the calls? Fallback design: two comm objects.

**How**: resolve at a 2n bring-up (capacity formula `4·max(row,col sum)` is
transpose-symmetric, so sizing is fine either way); validate against the
torch two-layer reference; then add the `l01_fast` variant (e2e-only,
≥2 nodes, `requires_file` libflash) and a small capsule. Reference
implementations: `test/python/moe_gather_rs/test_moe_gather_rs_fast_baseline.py`
(the l1 direction, validated) and the layer0 fast bench.

## 6. 16n W64 closure debt (pre-campaign, still open)

**What**: the 2026-08-15 W64 fixes (predicate segment gate 55e0273, exact
knobs, retry-routing) were never re-validated at 16n: allgather
correctness-ON b1–b16, `hier_compress_lb_union` b32/b64, and (if EP arms
matter again) moonep_fused at W64. Now ALSO worth folding in: layer1 at
16n has never run (the empty-expert fix makes it possible for the first
time).

**How**: needs `-q regular` (interactive QOS caps at 4 nodes), G question
(G=192 was the historical 16n deviation — the trace generator will state
the minimum; document whichever is used), NVSHMEM_SYMMETRIC_SIZE handled by
the exact sizers. Budget several hours of queue+run; capsule protocol as
always (fwd+rev if any verdict is claimed).

## Worktree state & the base-ref trap

Debug worktrees `l1-hang-debug` / `l1-nn4-debug` are PURGED (2026-08-16;
branches kept, everything merged — a stray uncommitted copy of the
empty-expert fix in l1-hang-debug was verified byte-identical to main
before removal). The campaign worktree `comm-sweep-layer-axis` remains on
disk, clean and even with main — a new session may either enter it
(EnterWorktree with its path) or remove it and start fresh; there is
nothing unmerged in it either way, and fresh-per-task branches are the
convention.

**TRAP**: Claude Code's EnterWorktree with a NAME branches from
**origin/main = upstream bytedance/flux**, not local main — a fresh
worktree created that way is missing months of local work (this bit the
campaign on day one). Create worktrees manually instead:
`git worktree add -b <branch> .claude/worktrees/<name> main`
then EnterWorktree with the `path`. (Or immediately
`git reset --hard main` inside a name-created worktree.)

## 7. Small hygiene items

- The campaign's variants.py still carries the five factorial corner arms
  and the binary-A-era comments; prune once item 1 lands.
- `docs/handoff/00_START_HERE.md` should gain a one-line pointer to
  handoff 07 as the layer-axis campaign authority.
- The two debug fixes (`1550b67` lazy-load preload, `c9b82b6` empty-expert)
  are candidates for an upstream PR to bytedance/flux — user's call.
- Capsule count is growing (150+ under sweeps/results/runs) — no action
  needed, but analysis scripts should filter by the campaign notes strings.

---

Protocol reminders that bit us and will bite again: compare arms only
within one capsule/one BUILD (manifest `flux_libs` sha, not git_sha);
ordering-sensitive claims need fwd+rev sign agreement; trace cells with
`routing_mode=''` are dealer-poisoned; W=8 and W=16 never mix; when trace
cells hang, bincount the routing for empty experts before anything else.

## v3 fixes (2026-08-17, branch epic-v3-fixes)

- **FIXED**: the epic_l01_hc_m4 b2 deterministic hang — root cause was NOT a
  missing ladder signal (the combine always signals zero-row lanes) but an
  UNPRIMED NVSHMEM transport kernel: the bare inter-node signal_op, emitted
  iff a lane has zero rows (U[d][n]==0), first-launch module-loads behind
  the resident pack/pre-reduce spin kernels (the 1550b67 lazy-load class).
  Fix: two signal_op primes added to the TopkReduceScatterOp ctor priming
  block; the same block was ported to the dispatch op's ctor (which had
  NONE — latent for the fused flux arms under EARLY_LAUNCH). Hardening:
  FLUX_A2AV_RS_SPIN_LIMIT (0=off) traps the two combine spin loops instead
  of hanging; wire_total-vs-U consistency asserts in both compress-CSR
  builders.
- **FIXED**: GemmGroupedV2 zero-split skew (weights AND fp8 per-expert
  scales now stay associated with their own expert across zero splits);
  regression mode `--zero_splits` in test_grouped_gemm_v2_only.py.
- **DOCUMENTED, NOT FIXED** (out of sm80 scope, marked with code comments
  pointing at the V2 fix): identical weight-pointer skew in
  gemm_grouped_v3.cc (Hopper) and blockscale_gemm.cc::forward_grouped
  (fp8; weight AND weight-scale pointers).
- **FIXED**: All2AllSingle cross-rank contract — ctor now allgathers
  {max_split, n_dim, local_world_size, dtype} and FLUX_CHECKs equality
  (loud, hang-free abort); debug-build device asserts guard per-call
  splits <= max_split; probe_a2a_single.py --mismatch is the negative test.

## nodeaware/LocCap campaign follow-ups (2026-08-19, session 8.19.theory)

- **Terminology, recorded per user note: "D6" = the MODDED replica-selection
  rule** — source rank src sends ALL its tokens for expert l to instance
  `src mod lcnts[l]` (EPIC port design decision #6; the paper is silent on
  replica selection). Token-oblivious; the LocCap baseline anchor.
- **DEFERRED baseline arm (user-requested, run after the current program):
  `evensplit` router** — per expert l, order its tokens canonically (global
  token id ascending) and split CONTIGUOUSLY into lcnts[l] equal chunks,
  chunk j -> instance j ("first 1/2 to replica 0, next 1/2 to replica 1";
  brute-force even sharding, source-oblivious — distinct from BOTH d6's
  per-source modding AND UltraEP's per-(source,expert) largest-remainder
  quota interleave). Implement as a third --router option emitting
  plan.phys_override (~20 lines beside loccap_route; offline-simulable in
  predict_placement identically, so it gets pre-registered incidence
  numbers for free). Compare within one capsule against d6 + loccap.
- lc0625@b64 cell: blocked on the bit-identical router vectorization
  (repair-phase python loops; >25 min on login node — sidecar route_hash is
  the identity oracle). Then a b64-only patch twin pair.
- 8n stage: blocked on the epic dispatch_only W32 delivery race (NR-16
  amendment); needs its own 2n->4n->8n bring-up after the 4n program and
  any rebuild boundary.
- **LocCap routing inside the FULLY-FUSED op (user-requested todo,
  post-session)**: today's loccap arms run on the epic staged-compute class
  (fused dispatch_only wire, unfused per-group GEMMs). Capsule B's b8
  verdict shows tile overlap dominates there (fused lb_union 3.93 vs best
  staged 4.99), while at b64 locality routing wins outright (na_lc125
  35.58 beats fused 37.84, same capsule). The composition — plan.phys_
  override-style replica selection feeding the full fused forward
  (GemmGroupedV2AGScatterOp with GEMM inside, moonep_fused-style virtual
  space + weights) — is the v2 arm: locality's byte/balance wins on top of
  tile overlap. Needs: virtual_choosed emission from loccap (exists),
  weight manifestation for replicas on the fused path (place_weights
  equivalent), and the pad handling noted in the campaign plan (dummy
  pad-weight expert or padless m=1 space).

## PLACE-lambda GPU port (2026-08-21, session 8.21.place) — LANDED; fusion next

- **Landed** (`flux.testing.placelambda_gpu`, pll_* variants, SCHEMA rule-5
  placement amendment): `loccap_gpu` router = bounded-round vectorized
  LocCap (greedy cover, clipped proportional fills, bounded repair;
  integer-deterministic, **CPU==GPU bit-identical** — proven on the login
  A100), rule-5 TIMED per-iteration via EpicIterPlanner router branch
  (plan_ms). `build_placement_gpu` = batch-observed steepest-descent
  PLACE-lambda (Stage A/B vectorized, LCM-integer bundle credit);
  `place_dynamic` toggle: static = untimed setup solve (place_solver_ms),
  dynamic = per-iteration TIMED solve + move-diff + trigger (place_ms,
  epic_place_* facts). Sidecar mode `placement_placelambda_gpu` (offline
  CPU solve of the same file; driver hard-asserts on-device equality).
  eps default 0.0625 (working default, NOT canonical). loccap_gpu is a
  NEW arm — never compare against exact-loccap cells.
- **NOT yet done**: (a) multi-rank GPU bring-up of the pll arms
  (2n -> 4n smoke, then the placement-axis capsule); (b) **the FUSION
  pass — immediately after bring-up verification (user directive)**: mine
  place_ms/plan_ms for removable latency (CUDA-graph capture of the
  fixed-shape kernel sequence, fold the router into the
  FusedEpDispatch-style in-window derive, batch the small kernels);
  (c) actual expert-weight dispatch on trigger (the permanent-weights
  end design: oracle ground truth vs per-batch solve -> move decision;
  epic in-kernel swap machinery is the reuse candidate); (d) the v2
  fully-fused composition above (loccap_gpu phys feeding
  GemmGroupedV2AGScatterOp) — unchanged, still queued.
- **Measured starting point (login A100, real Qwen3 4n routing, eps
  0.0625)** — the fusion pass's mining targets: router 52 ms (b8) / 67 ms
  (b64); placement solve 201/156 ms; trigger decision 118 ms. Quality:
  incidence vs fixed/d6 −47.2% (b8) / −26.7% (b64); placement contributes
  most, router −8..−15% on top. The latency is NOT kernel-bound — it is
  ~200 small launches + dozens of host syncs (`bool()/int()` early-exits
  in the round loops) + per-call rebuild of placement-static tables.
  Fusion order: (1) hoist instance tables/grids to refresh_placement,
  (2) remove per-round early-exit syncs (fixed round count), (3) tensorize
  place_decision's host-side hosts conversion, (4) CUDA-graph the fixed-
  shape sequence. Sub-ms plausibly needs (4); (1)-(3) alone should give
  ~10x. Quality flag: real-routing skew leaves over_cap_rows 8241@b8 /
  47824@b64 (rows_max ~1.2x cap) — the bounded repair cannot fix
  single-instance hot experts (exact loccap's augmenting chains can);
  consider a replication-aware repair or extra rounds in the fusion pass.
- **2026-08-21 (later, same session): SENDER-LOCAL + FUSED KERNEL LANDED,
  SUB-MS ACHIEVED.** User rulings: bit-identity RELAXED (invariants +
  incidence band replace route_hash), kernel = native flux binary (paper
  artifact). Sender-local redesign (`loccap_route_sl` torch reference in
  placelambda_gpu.py): shared tables = order-independent functions of the
  d[R,G] allgather; per-row decisions rank-owned; NO donation across
  sources. Offline gate PASSED: sl never loses to global loccap_gpu on
  incidence (up to −13% better at b8 tight eps; identical at eps=inf).
  CUDA kernel `placelambda_route_sl` in src/cuda/moe_utils.cu (9 small
  launches, O(n^2) largest-remainder ranking, relaxed atomic tickets),
  pybind in src/pybind/ths_op.cc, flux.placelambda_route_sl, tag
  FLUX_PLACELAMBDA_ROUTE_SL_TAG (in libflux_cuda.so, probe-visible).
  Parity test test_placelambda_kernel.py: **0.40-0.45 ms/rank warm
  median** (vs 52-67 ms torch), incidence within +0.5% of reference,
  b8 AND b64 (latency launch-bound, not compute-bound).
- **OPEN for kernel-arm bring-up (the one real design item):** per-
  iteration routing now VARIES across iterations (relaxed atomics) while
  the hc runner's buffers/K_g/RS-caps are frozen at setup — the
  `kg_iter <= kg_frozen` assert and knob sizing need slack (headroom on
  the setup-routed K_g / rows), or first-iteration sizing + margin.
  Solve at 2n bring-up, not blind. Driver/planner wiring for router
  `loccap_sl` (kernel + phys-row allgather in derive, relaxed
  check_against) is NOT yet written — the torch `loccap_gpu` global
  router remains the integrated deterministic arm. Next fusion targets:
  persist workspace/memsets across iterations, fold into the dispatch
  launch (planner_impl=fused_dispatch), counts-only exchange instead of
  the phys-row allgather.
- **2026-08-22 — DISPATCH WIRE ORDERING BUG ROOT-CAUSED (the "op-level
  cross-iteration staleness" of 08-21 PART 5).** Toggle ladders on 4n K3 b7
  (job 57405130, ~2 nh): relay-pull fence (T1), relay-panel poison (T2),
  blocking pull (T3) — all REFUTED (C2 dead; poison sentinel never
  delivered); epoch quiets (T4) reduce 10x but leave node2←node3 (round-1)
  residue; `FLUX_PLL_RANDOM_PAYLOAD=1` (new driver guard: per-iteration
  payload randomization + payload-provenance probe) showed the bug is
  PRODUCTION-WIDE: d6, loccap_gpu, loccap_sl all fail 16/16 with ~35-45%
  inter-node rows carrying EXACTLY the previous epoch's payload (ledger
  probe), in e2e, isolated and single-stream modes; FORCE_REF's lockstep
  standalone forward was the only pass. **Mechanism:** the inter-node
  chunk `putmem_signal_nbi` lets the gateway's `node_sig` wait pass before
  the chunk bytes land → the lb_union forward ships the previous epoch's
  stage. **Verified fix: `FLUX_A2AV_BLOCKING_WIRE=1`** → 16/16 bitwise on
  every arm INCLUDING the kernel arm (G1 re-gate PASSED: loccap_sl plan
  4.5 ms, comm 5.4, total 13.3 ms) at comm +16-21% / e2e +10-14%.
  Candidate F2 `FLUX_A2AV_WIRE_SIGNAL_FENCE=1` (data nbi → one quiet →
  signals; tag kept) built and REFUTED on job 57405794 (16/16 stale on
  loccap_gpu, 4/16 on the kernel arm = T4-like residue) — a stream quiet
  before the signal is not sufficient. Same-build 10-iter A/B: BLOCKING
  comm 5.81 / e2e 8.58 ms vs nbi 5.07 / 7.85 (+15% / +9%). Stage 2b
  derive_combine_meta drift guard PASSED at setup on the epic l01 K3 cell
  (the cell then failed only on the wire staleness under F2). F3 candidate
  = GPUDirect-RDMA visibility: gateway node_sig wait is a raw
  CUStreamWaitValue64 GEQ poll without CU_STREAM_WAIT_VALUE_FLUSH → toggle
  `FLUX_A2AV_WAIT_FLUSH=1` (flush flag on all a2av front-end waits, tag
  FLUX_A2AV_WAIT_FLUSH_TAG) — DEAD here: Perlmutter A100 reports
  cudaDevAttrCanFlushRemoteWrites=0 (job 57406271, ctor guard fired on
  every F3 cell). F4 `FLUX_A2AV_NVSHMEM_WAIT=1` (tag
  FLUX_A2AV_NVSHMEM_WAIT_TAG): the union gateway node_sig wait and the
  gather-gateway t_wait use nvshmemx_signal_wait_until_on_stream (NVSHMEM
  proxy-enforced GDR consistency; gather_rs.cc:2039 already does this for
  its inter-node signals) — built and REFUTED (ladder10, job 57406405:
  loccap_gpu + d6 16/16 one-epoch-stale; BLOCKING 16/16 again, comm 5.74
  vs nbi 5.06 = +13%). ROOT MECHANISM (NVSHMEM libfabric transport
  source): put_signal = data `fi_write` (RDMA into GPU memory) + signal as
  a separate `fi_send` message applied by the TARGET host proxy, no
  FI_FENCE between them; `fi_info -p cxi` shows msg_order [] on the
  endpoint type in use → flag-before-data is legal; sender quiet
  (delivery-complete ≠ GPU-visible) and receiver NVSHMEM wait cannot
  order it; the blocking variant works because the data completes before
  the flag message is issued. **SHIPPING FIX = `FLUX_A2AV_BLOCKING_WIRE=1`**
  (4/4 cells incl. 10-iter + kernel arm). DECISIONS FOR THE USER: (1)
  canonicalize (flip the default; rule-4 boundary + SCHEMA note) or keep
  env-carried; (2) the expert-movement sweep (plan Stage 4) and any
  re-quoted comm-only numbers must run with it (prior hc capsules are
  never-mix). NEXT: audit gather_rs (8 nbi put_signal sites) and
  fused_ep_dispatch under FLUX_PLL_RANDOM_PAYLOAD; G2 capsule with the
  fix; Stages 2-4. SCHEMA protocol rule 6 records the never-mix
  boundary and the new correctness requirement. OPEN: audit layer1
  gather_rs (8 nbi put_signal sites) + fused_ep_dispatch (eplb/epic) under
  the same probe; decide the canonical wire fix (blocking vs F2) by a
  same-capsule A/B; Stage 2b derive_combine_meta compile+validate; then
  G2 and the plan's Stages 2-4.
- **2026-08-22 (night) — WIRE-ORDERING AUDIT, rounds 1-3 (payload probe = U[0,0.01)
  with alternating sign per iteration; blocking-default binary; 4n K3 b7, 10 iters).**
  PASS 16/16: pll loccap_gpu l0, pll d6 l0, eplb fused l0 + l01 (FusedEpDispatch
  already fences), moonep staged l0 (nvshmem All2AllSingle + getmem), flux
  allgather l0, stock COMET allgather_dense l01, lbunion_hier l01, lbunion_compress
  l01, gather_rs compress/hier/dense l1. FAIL/OPEN: (a) layer-1 nbi combine wire
  fails (epic l01 with FLUX_A2AV_RS_BLOCKING_WIRE=0: dispatch rows bitwise-exact,
  final output wrong 16/16) -> gather_rs blocking default justified; blocking must be
  INTER-NODE only (the on-stream blocking variant faults on intra-node host-staged
  sources: "unspecified launch failure" until fixed in 1197597). (b) epic l01 still
  mismatches with RS blocking (max 0.33) -> NOT the wire; bisect (probe off /
  FLUX_EPIC_HCC_DERIVE=0) in round 4 — Stage 2b derive_combine_meta suspect. (c)
  FUSED-forward-path residual: l0 hc_lb_union (comm-only) 4/16 ranks with exactly 1
  wrong row each (rank 7 row 9936, 1270/3072 elems, max 0.046 vs |y|~0.007),
  moonep_fused l0 3/16 ranks 1 row each on the SAME virtual experts every run —
  deterministic; dispatch_only arms clean; provenance (prev-payload torch reference)
  in round 4. (d) moonep l01 / moonep l1 standalone / moonep_fused l01 host-OOM at
  K3 under manual sizing — need the sweep's per-cell sizing, not a verdict. Runner
  lessons: `srun` inside `while read` must take `< /dev/null`; per-rank raises wedge
  the step -> collective verdicts everywhere; flag names differ per driver (l01/l1
  use -G). sweep.py now exports FLUX_RANDOM_PAYLOAD=1 for every correctness cell.
- **2026-08-22 (later) — audit rounds 4-5.** (1) epic l01 probe-only mismatch is NOT
  Stage 2b (FLUX_EPIC_HCC_DERIVE=0 gives the identical max-abs per rank) and NOT the
  RS wire: provenance shows ~45 % of token rows off by ~7 % (max 0.41 on |ref| 5.9),
  only a handful equal the previous payload's chain -> a MINORITY OF PER-EXPERT
  CONTRIBUTIONS is stale-by-one inside the epic runner's layer-1 path (dispatch rows
  bitwise-exact). Prime suspect: stream ordering between the runner's combine_pack
  (group1_outputs -> e["inbuf"] on comm_stream) and TopkReduceScatterOp.run's
  internal streams / host-staged intra-node nbi puts reading inbuf; static payloads
  hide it. epic l01 (and the future pll l01) numbers are correctness-suspect until
  fixed. (2) FUSED-forward-path residual (comm-only hc_lb_union l0, moonep_fused l0):
  one fixed row per failing rank (r7 row 9936, r10 row 18028, r13 row 7722), wrong-
  element count varies 88..2344/3072 per run, and when nearly whole it EQUALS the
  previous payload -> the fused GEMM reads that row while it is still being
  overwritten (element-granular old/new mix at a structural position = tile/window
  boundary of the Tier-B window gating). dispatch_only arms (pll/epic l0, eplb fused)
  are clean. Both are pre-existing bugs made visible by the payload probe.
- **2026-08-22 — bug (a) FIXED (plan eager-juggling-glacier, dissect-and-fix):** epic
  runner `combine_group` now brackets `TopkReduceScatterOp.run` with
  `_hcc_stream.wait_stream(current)` (IN) and `current.wait_stream(_hcc_stream)` (OUT,
  before the closing barrier / flag zero_ / accumulate), + `record_stream` on the
  per-iteration index tensors + inbuf rows assert. Probe: A1 16/16; A0a (unfenced,
  single-stream) and A0b (unfenced, isolated) controls still fail identically -> the
  cp_stream seam was the bug, not the two-stream loop. Side effect: epic l01 `comb_ms`
  was launch-latency before (SCHEMA rule 8, never-mix).
