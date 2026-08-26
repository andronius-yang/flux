# H4 / H1 scoping + implementation report (worktree flux-ours, branch ours, 2026-08-25)

All paths relative to `/global/u1/y/yufeid/workspace/changchen/andrewy/flux-ours`.
Line numbers are post-edit for gather_rs files, pre-edit (unchanged) for ag_scatter.

## H4 — combine receiver head-of-line: IMPLEMENTED (arrival-dynamic bucket receiver)

### Problem recap (verified in code)
`TopkReduceScatterOpImpl::run_a2av_hier`, bucket receiver
(`src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc:2310-2414` post-edit): S = L + NN - 1
sequential `CUStreamWaitValue64` waits on ONE reduce stream in EXPECTED arrival order
(own-node lanes first, then remote same-lr lanes in descending ring), each releasing an
`a2av_combine_bucket_reduce` fold of the tokens completing at that chain position. The
bucket-prefix guarantee REQUIRES the sequential chain (bucket k's tokens may need any lane
at positions <= k), so one inverted arrival head-of-line blocks every landed lane behind it.

### Why the task's design (a) (multi-stream subset chains) is structurally void here
Under `CUDA_DEVICE_MAX_CONNECTIONS=1` (launch.sh contract), all streams multiplex ONE
front-end channel; a pending `CUStreamWaitValue` on any stream blocks later-enqueued ops on
OTHER streams (stated as the enqueue-schedule rule at gemm_grouped_v2_gather_rs.cc:1982-1989
and the wire-ladder comment at :869-877 "at conn=1 the split ... buys nothing"). K_r
sub-chains would not give arrival independence: a stalled wait in one sub-chain still parks
the channel for the others. The only real fix is device-side consumption (no front-end waits),
i.e. design (b).

### What was implemented (design b, refined)
`FLUX_A2AV_RS_RECV_DYN=1` (default **0**; binary tag string `FLUX_A2AV_RS_RECV_DYN_TAG`,
registered exactly like `FLUX_A2AV_RS_BUCKET_TAG` — a `(void)get_int_from_env("...TAG", 0)`
probe in the knob helper, gemm_grouped_v2_gather_rs.cc:277-306): ONE persistent kernel on the
reserved reduce SMs replaces the whole wait chain.

Refinement over the task's letter-(b): instead of scatter-adding lanes into an fp32 scratch +
finalize (lane-chain's 4-5x byte amplification + a full ntok x n finalize pass through only
8 reduce CTAs as exposed tail), the kernel keeps the BUCKET fold's 1x-byte / fold-at-token-
completion semantics, made arrival-dynamic with per-token outstanding-contribution counters:

- Warp protocol (`a2av_combine_dyn_reduce_kernel`, `src/moe_gather_rs/a2av_combine.cu:560-676`):
  - leader polls the S per-lane epoch signals with `ld.global.acquire.sys.b64`
    (`load_acquire_sys_u64`, the eager kernel's proven pattern) against `run_id` (GEQ);
  - claims `chunk_rows` (knob `FLUX_A2AV_RS_RECV_DYN_CHUNK`, default 4) of any ARRIVED lane
    via per-lane `atomicAdd` cursors;
  - for each claimed row: `t = token_of[row]`; `atomicSub(remain[t], 1)`; the warp observing
    `old == 1` folds token t immediately: fp32 accumulate over `red_ptr[t]..red_ptr[t+1]`
    (the reduce CSR) **in CSR order**, one cast, one output write — the bucket fold's exact
    per-element arithmetic;
  - exits when every lane is arrived + fully claimed. Zero-row lanes are excluded host-side
    (they still signal; nothing can complete there — same rule as the bucket chain).
- Plan-side precursors, all enqueued on the reduce stream BEFORE any host wait reaches the
  conn=1 channel (the eager kernel's launch-order discipline, mandatory for any persistent
  kernel): cursor memset + `a2av_lane_token_map` extended with an optional `remain` output
  (`remain[t] = red_ptr[t+1] - red_ptr[t]`).
- Buffer reuse (no new allocations on the bucket path): `remain` = `a2av_bucket_comp_`
  ([ntok_max] i32), cursors = `a2av_bucket_meta_`; `a2av_token_of_row_` is now allocated for
  lane-chain OR recv_dyn (gemm_grouped_v2_gather_rs.cc:1050-1057).

### Files/lines touched (H4)
- `include/flux/args/moe_gather_rs.h`:
  - `A2AVLaneTokenMapArguments`: trailing `int32_t *remain = nullptr` (positional aggregate
    init at existing call sites unchanged).
  - new `struct A2AVDynReduceArguments` + `a2av_combine_dyn_reduce` decl (after the bucket
    section, ~:511-560).
- `src/moe_gather_rs/a2av_combine.cu`:
  - token-map kernel writes `remain` when non-null (:404-417);
  - new `a2av_combine_dyn_reduce_kernel<T>` (:560-676), launcher (:938-960), preload
    attribute query in `a2av_combine_preload` (:717-720).
- `src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc`:
  - knob helpers `get_a2av_rs_recv_dyn` / `get_a2av_rs_recv_dyn_chunk` + TAG (:277-306);
  - member `const bool a2av_recv_dyn_` (:938-941), ctor init (`a2av_bucket_ && knob`, :1293),
    loud contract check when env-explicit (:1305-1311);
  - `a2av_token_of_row_` allocation condition (:1041-1057);
  - dyn launch block in `run_a2av_hier` after the eager block (:1910-1980);
  - receiver-branch guards: bucket branch `&& !a2av_recv_dyn_` (:2310), legacy wait-all
    branch `&& !a2av_recv_dyn_` (:2417).

### Race analysis (why this is safe)
1. **Row claim exactly-once**: per-lane atomic cursor; each row processed by exactly one warp.
2. **Token fold exactly-once**: `remain[t]` atomic total order — exactly one decrementer
   observes `old == 1`. Every recv row belongs to exactly one token (`token_of` is total over
   the C' image: own-node lanes are copies of my tokens, remote lanes are per-token merged
   partials — the same coverage the lane-chain receiver already relies on).
3. **No concurrent same-row output writes**: each output token row written once, by its
   completing warp (the bucket property). No fp32 scratch, no atomicAdd on data.
4. **Signal trust / wire-ordering rule**: unchanged. A lane's rows are only decremented after
   that lane's signal acquire-loads >= run_id; payload precedes signal by the blocking-wire
   discipline (SCHEMA rule 6). Wait semantics were not weakened — only consumption order.
5. **Cross-lane visibility at fold time**: the folding warp reads rows of OTHER lanes than the
   one it just decremented. Chain: those rows' decrements happened-before (same-location
   atomic order on `remain[t]`) and were issued after their lane's acquire; the payload was
   system-visible before the signal (blocking wire); a `__threadfence()` sits between the
   old==1 observation and the fold loads; every panel line is read for the FIRST time in this
   kernel (L1 is invalidated at kernel launch; each row is folded once), so loads are served
   L2-fresh.
6. **Termination**: every lane signals every epoch (always-signal invariant, payload or not);
   cursors are monotone; warps exit when all lanes arrived + exhausted. `FLUX_A2AV_RS_SPIN_LIMIT`
   watchdog wired (trap + printf, same semantics as eager/prered).
7. **Launch-order deadlock class**: persistent kernel enqueued before any host wait (conn=1
   rule) and preloaded in the ctor (lazy-load deadlock class, 2026-08-16 root cause).
8. **SM budget**: `get_a2av_reduce_blocks()` CTAs (default 8) — the same reservation the
   bucket folds used; coexists with the prered spin kernel exactly as the eager receiver did.

### Numerics / gate expectations
- Knob OFF (default): bit-identical binary behavior (all new code is behind `a2av_recv_dyn_`;
  no schedule or arithmetic change).
- Knob ON: output is **BITWISE-identical** to the bucket receiver / wait-all reduce (per-token
  fp32 accumulation in fixed CSR order; only the fold *schedule* is arrival-dependent). The
  ours gate (`test/python/moe_ag_scatter/test_moe_ours_traffic.py:904`, `torch.allclose` +
  bad-row count, `--check_iters 1` with per-iteration random payload) must stay green with
  bit-exact expected under FLUX_TEST_DETERMINISTIC=1 as well; any allclose-only pass would
  itself be a red flag for a claim/fold bug.
- NN scaling: lane arrays sized `kA2AVMaxWorld` (64) and a 64-bit lane mask → S = 35 at 32n
  fits. (Pre-existing global cap: `kA2AVMaxNodes = 16` in moe_gather_rs.h bounds the whole
  combine at 16 nodes; bumping it is a separate, orthogonal change.)

### Residual risks (H4)
- Not compiled here (orchestrator builds): risk of minor type/signature slips; kept to the
  file's existing idioms (PackU/loadPack/storePack, tuple_return_if, designated init).
- Perf risk: 128 warps polling S acquire loads per sweep is more signal traffic than the
  eager kernel's per-element polls in aggregate? No — polls are per-warp per-sweep (S loads),
  orders of magnitude fewer than eager's per-element polling; but the fold-inline-in-warp
  shape serializes rows within a chunk (leader atomics). `FLUX_A2AV_RS_RECV_DYN_CHUNK` is the
  spread dial if fold-triggering warps become the bottleneck.
- The exposed tail is now (last lane's rows + its completed tokens) through 8 CTAs; if the
  last lane is the own-node GEMM-tail lane (canon own-last production), most tokens complete
  at it — `FLUX_A2AV_RS_OWN_FIRST=1` is the documented receiver-overlap precondition and
  should be part of the A/B matrix (dyn x own_first).
- `remain`/`token_of` add one small plan kernel pass (ntok threads) to the timed bracket —
  same cost class as the bucket map/scan/scatter it replaces.

## H1 — dispatch union-broadcast closure: DESIGN-ONLY (not implemented)

### Verified mechanism (anchors in `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`)
- Tier-B gateway loop :3150-3226: per round dn (source node ns), after `node_sig[ns]`
  arrives, gateway (my_node, my_lr) puts its WHOLE staged window [win_a, win_b) to ALL L
  local ranks at `recv_off_of_u(ns*L, d) + win_a`, fused signal on the window's gating slot
  `signal_base + ns*L + my_lr`. Recv volume = L x U per round → the measured 2.15x-needed at
  16n b64 (overlap ~1.86x/U), trending to Lx at 32n.
- Window partition: `chunk_bound` lambdas :1308-1329 — "SINGLE source of truth", derived
  purely from host U_mat (untimed uc metadata, :3987-3998: `a2av_unique_counts` [W, W+NN]).
- Recv layout: `region_rows` :1500-1503 — a remote-node source's region holds the whole
  U[s][node(d)] union, IDENTICAL on all L destinations; interiors ascending token index.
- Consumer identity (the load-bearing assumption): the dedup consumer reads recv row
  `C[t]` = exclusive cumsum of the per-token keep flag over global tokens
  (:2191-2231 ATen path; fused path :2039-2127 passes `c_excl` straight into
  `a2av_consumer_build_kernel` — `sort_util.cu:675`: `recv_row = args.c_excl[p / args.topk]`).
  Valid only because regions are source-contiguous, token-ascending, and identical across
  destinations.
- Tier-B tile gating: `gate_q`/`lane_end` :1541-1571 — per expert, per LANE (window) ends via
  ONE searchsorted over the (expert, recv-row) key; fused build re-keys A rows per
  (expert, lane) via `a2av_gating_cumsum_` (:2088-2116); the GEMM tile gate partitions an
  expert's A rows by lane and spins on the window slots. Lane keying REQUIRES slot regions to
  be disjoint, contiguous, ascending in the recv image.
- Sender parcel order: stage1 union keep-flags (`sort_util.cu:573,597`) + `a2av_pack_scan`
  (ascending-token exclusive scan per segment, `sort_util.cu:635`) + wave-pack per-segment
  `index_select`s :1773-1814.
- The "no SM gather in the fused pipeline" constraint: :4000-4010 — union_bcast is exempt
  from `sm_margin >= 1` precisely because its forwards are pure copy-engine puts; the exact-
  delivery alternative (relay gather gateway :3227-3311) pays `t_index_select` SM gathers +
  a host D2H sync (`fwd_cnt_event_` :3233), which is what Tier B was built to avoid.

### The design that closes recv amplification without SM gathers
Reorder each (ns -> tn) union stream, WITHIN each gateway window, into need-set groups, and
GLUE each window's exclusive group to its shared slice:

1. Sender-side stream order per (source s, target node tn):
   `[excl(dl0) | sh_0 | excl(dl1) | sh_1 | excl(dl2) | sh_2 | excl(dl3) | sh_3]`
   where excl(dlk) = union tokens needed by EXACTLY local rank k of tn, and the shared group
   (needed by >= 2 lanes) is cut into L balanced slices sh_k. Window k (gateway k's parcel)
   = `[excl(dlk) | sh_k]` — still ONE contiguous wire put per (round, gateway).
2. Gateway forward per round: to dest k: ONE put of the whole window (excl_k + sh_k are
   adjacent both in the stage and in k's recv image); to dest d != k: ONE put of the sh_k
   subrange only. Put count per (round, gateway) stays L; all puts remain contiguous
   copy-engine `putmem_signal` — the :4007 exemption survives.
3. Destination recv image per (source region, dest d):
   `[sh_0 | .. | sh_{d-1} | excl_d | sh_d | .. | sh_{L-1}]` (excl_d glued before sh_d) —
   slot k's delivered piece is contiguous (`sh_k`, or `[excl_d|sh_d]` for k == d), pieces are
   disjoint and ascending in slot order → the lane_end searchsorted keying and the tile-gate
   slot discipline survive VERBATIM in structure; only the boundary values change.
   Recv volume per round per dest = excl_d + shared (vs U today); total = Sum(excl) + L*shared.
4. Metadata: the group counts per (s, tn) — L+1 ints (or 2^L - 1 for the exact-delivery
   generalization below) — ride the EXTENDED uc contract (:3991 explicitly calls uc an
   extension of splits_per_source "NOT derivable from cnt"): every rank computes all
   boundaries host-side, no new D2H sync, no wire metadata. uc becomes [W, W + NN + NN*(L+1)]
   (or a second tensor).

### Why this is NOT the "contained" case → not implemented
The claimer survives, but the SCATTER/plan assumptions break at four coordinated sites, three
of them device kernels, across three distinct roles that must agree bit-exactly:
1. **Consumer gather identity** (`c_excl` one-cumsum, sort_util.cu:675 + ATen twin :2191-2231):
   recv content/order becomes per-destination and per-piece; `C[t]` must be replaced by a
   per-dest `row_of_token[t]` built from group classification + per-piece cumsums (the
   substitution POINT is clean — the kernel reads `c_excl[t]` as an opaque table — but the
   table build is a new multi-piece scan pass, and the keep-flag semantics split into
   "occupies a row at d" (excl_d OR shared) vs "needed by d").
2. **Sender pack order** (stage1 flags + `a2av_pack_scan` + wave-pack): needs group ids per
   (seg, token) and a two-level (window, group) scan instead of the single ascending-token
   scan — a kernel rewrite on the timed plan path.
3. **All host boundary formulas** keyed off `chunk_bound`/`region_rows`/`recv_off_u`/
   `lane_end`/`nvtx_window_rows_` (:1308-1329, :1494-1571) plus the wire-put/stage lambdas
   (:3091-3136) — the file itself warns "any divergent re-derivation silently corrupts wire
   offsets" (:1299-1306).
4. **Plan producer** (python `derive_routed_meta` / testing harness) must produce the extended
   uc columns, and the Tier-B fused-stage2 audit invariants (:2128-2188) must be rewritten.
Each is individually moderate; jointly they are a new wire layout = a rule-4 never-mix
boundary with silent-corruption failure modes, and no GPU validation was available in this
session. Per the task's criterion this is the DO-NOT-IMPLEMENT branch.

### Win estimate + a dial the implementer should know
Let o = Sum_d(need_d)/U (measured ~1.86 at 16n b64). The (L+1)-group design's recv =
excl + L*shared, where shared depends on the multiplicity distribution: if most shared tokens
have multiplicity 2, shared ≈ (oU - excl)/2 and the recv saving over L*U can be as small as
~10-15%; if multiplicity is high (union concentrated), the saving approaches the full
L*U -> oU (2.15x -> 1x needed). The exact-delivery generalization — order by full need-set
(2^L - 1 = 15 groups, canonical fixed order chosen to minimize per-dest run count; each dest
receives its groups as <= ~4-6 contiguous CE puts per window) — always achieves recv = oU at
put-count inflation ~4-6x per (round, gateway, dest). Both share the same 4-site surgery
above; group-count K ∈ {L+1, 2^L-1} is a spec knob, not a design fork. RECOMMENDATION: before
building any of it, measure the shared-multiplicity histogram from an existing captured trace
(pure python over routing) — if multiplicity-2 dominates, only the 15-group variant pays, and
the put-count trade needs its own microbench.

### Suggested knob names if/when implemented
`FLUX_A2AV_LB_UNION_XSPLIT` (0 = union broadcast, 1 = excl/shared glued windows,
2 = full need-set groups) + `FLUX_A2AV_LB_UNION_XSPLIT_TAG`; extended-uc presence check must
be a loud FLUX_CHECK (uc width), collective-evaluation style.

## OPEN DEFECT (2026-08-25 field report): dyn receiver hangs at 16n b32/b64

Symptom: every FLUX_A2AV_RS_RECV_DYN=1 arm hangs at 16n qwen b32/b64 (silent cell,
900s idle kill, retry-identical); b1/b8 gates + twins green; wirebal twin (dyn off)
green at b32. Static re-audit findings, ranked:

### #1 (fix first): garbage token_of on any lane row not covered by the reduce CSR
Kernel reads t = token_of[row] for EVERY row in [lane_row_lo, +lane_rows)
(a2av_combine.cu:617-621); lane extents come from uc via chunk_cp/recv_off_cp
(gemm_grouped_v2_gather_rs.cc:1946-1953); token_of is UNINITIALIZED at alloc
(empty_with_uninitialized_data, :1053) and the map kernel writes only CSR-covered
rows (a2av_combine.cu:404-417). One uncovered row => arbitrary int32 t =>
atomicSub(remain + t, 1) is the kernel's ONLY OOB-write vector (+-8 GiB around
a2av_bucket_comp_). Three faces, all matching the symptom exactly:
 (a) corrupt lane_cursor (adjacent a2av_bucket_meta_) to large negative =>
     claims return start < rows forever on garbage rows => perpetual progress =>
     livelock that never naps and is IMMUNE to the spin watchdog (spins reset on
     progress, a2av_combine.cu:672-674) => silent 900s cell;
 (b) garbage red_ptr[t] fold bounds => ~2^31-iteration fold loop => same;
 (c) unmapped-VA atomic => device fault => rank dies, 63 peers spin => same
     job-level silence.
Status of the trigger premise: I VERIFIED statically that U == CSR coverage in the
steady plan path — a2av_meta_counts_kernel (sort_util.cu:1275-1301, per-token
owner-rank mask -> node mask) and compress_plan_token_kernel red_flags
(a2av_combine.cu:852-861) apply the same predicate to the same vce tensor
(ours.py:436 -> 463-467, one derive_routed_meta feeds _sd/_scd/_sps/_uc). So if #1
is the root cause, the divergence lives in an edge not refutable from here:
rem_base/rem_pos position alignment vs recv_off_cp under demand-sized caps, or
LocCap forced-fallback interplay — regimes that bind exactly at b32+/16n.
KEY ASYMMETRY: bucket/wirebal walk CSR entries only and are structurally immune to
uncovered rows; dyn's lane row-range walk is the ONE new data dependency.

### #2: spin-limit expiry = loud rank death that presents as job silence (Q1)
The loop honors FLUX_A2AV_RS_SPIN_LIMIT (a2av_combine.cu:658-671): spins counts
consecutive NO-PROGRESS sweeps (~1-3us each: 19 acquire polls + 200ns nap); expiry
does NOT silently exit — leader printf "[a2av-combine] dyn reduce SPIN LIMIT: warp
%d done_mask ..." then whole-warp __trap() kills that rank's context; peers then
spin forever => silent cell. remain counters are irrelevant to kernel exit (exit is
cursor-based). ours_gate specs arm no SPIN_LIMIT; several 16n specs arm 2e7
(gen7gate_*_16n.yaml) ~= 30-60s of continuous no-progress — far above any legit b64
drain (~45ms), so a trap implies the kernel was ALREADY stuck (#1-class).
DISCRIMINATOR (zero rebuild): grep the b32/b64 rank logs for "dyn reduce SPIN
LIMIT" and for unspecified-launch-failure / Xid lines. The printed done_mask names
the stuck lane set.

### #3: Q2 sizes/overflow — AUDIT CLEAN
remain = a2av_bucket_comp_ sized [ntok_max = max_m/topk/W] (TOKENS, :1057-1061),
t in [0, ntok_local <= ntok_max] when covered; cursors: Sd <= 19 of bucket_meta's
3*64+2 int32. Cursor growth bounded (rows + n_warps*chunk ~ 65536 + 512 at qwen
b64); per-token counts <= topk+NN-1 = 23. Nothing crosses a capacity between b8
(8192 rows) and b32 (32768 rows).

### #4: Q3 double/missed decrement — CLEAN as designed
Claims exactly-once (atomic cursor chunks); decrement only after that warp's leader
acquire-observed the lane signal; arrival re-polled every sweep; exactly one
decrementer sees old==1. The only breaker is #1's OOB corruption — Q3's suspicion
is a downstream symptom of #1, not an independent cause.

### #5: Q4 grid sizing — CLEAN
No shared memory, no one-pass assumption (chunked); 8 CTAs is throughput-only; map
kernel's 32 blocks at b64 fit the free-SM budget (margin 25 - pack 10 - prered 6 =
9 SMs) as b8's 4 blocks do. Eager precedent (persistent 8-CTA device-polling
receiver ran 16n b64 in the fpwp arms) rules out SM-residency/poll-pressure.

### Minimal fix (next binary, single change): token_of sentinel hardening
1. Launch block (gemm_grouped_v2_gather_rs.cc:~1967): cudaMemsetAsync(token_of,
   0xFF, image_rows * 4, reduce_stream) BEFORE the map kernel (0xFF bytes = -1).
2. Kernel row loop (a2av_combine.cu:~618): add int64_t ntokens to
   A2AVDynReduceArguments; `if (t < 0 || t >= args.ntokens) continue;` — slack rows
   still claimed (lanes exhaust, kernel exits) but never decrement or fold.
3. Diagnostic rider: on spin-limit expiry, warp 0 leader prints the per-lane
   (signal value, cursor, rows) table before __trap.
Cost: one ~256KB memset/iter + one compare/row; covered-row folds stay
bitwise-identical. Removes the ENTIRE #1 class regardless of which upstream edge
produces uncovered rows.

### No-rebuild diagnostics for the coordinator
- grep rank logs: "dyn reduce SPIN LIMIT", "unspecified launch failure", Xid.
- One b32 dyn cell with FLUX_A2AV_RS_SPIN_LIMIT=200000 (~0.5s): stuck-wait traps
  fast and prints done_mask; a livelock (perpetual progress) will NOT trap —
  discriminating #1(a) from a genuinely missing signal.
- cuda-gdb attach to a hung cell; warp PCs distinguish poll loop vs fold loop vs
  claim loop instantly.

### FIX APPLIED (2026-08-25, user-directed; source-only, uncompiled): RECV_DYN v2
Field confirmation: no trap lines in hung rank logs => livelock face #1(a).
Applied exactly the minimal fix above, plus the diagnostic rider and a tag bump:
- include/flux/args/moe_gather_rs.h: A2AVDynReduceArguments gains int64_t
  ntokens_local (between remain and lane_cursor; designated-init order kept).
- src/moe_gather_rs/a2av_combine.cu: (a) row-loop slack guard
  `if (t < 0 || t >= args.ntokens_local) continue;` (claimed, never decremented);
  (b) spin-limit expiry now prints a per-lane (sig, want, cursor, rows) table
  from warp 0's leader before the per-warp done_mask printf + __trap;
  (c) launcher FLUX_CHECK_GT(ntokens_local, 0).
- src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc: (a) -1-fill
  (cudaMemsetAsync 0xFF) of the ENTIRE a2av_token_of_row_ buffer
  (a2av_recv_rows_ int32s = max_m/W >= cpr >= image rows >= every lane extent,
  guarded by the :1466 cpr check and :1650 image check) BEFORE the map kernel,
  on reduce_stream; (b) .ntokens_local = ntok_local in the launch init;
  (c) TAG BUMP: FLUX_A2AV_RS_RECV_DYN_TAG -> FLUX_A2AV_RS_RECV_DYN_V2_TAG
  (old literal removed from all compiled sources; v1 dyn binaries never-mix).
- sweeps/variants.py: _DYN_REQUIRES now requires FLUX_A2AV_RS_RECV_DYN_V2_TAG
  (a v1 binary can no longer satisfy dyn arms).
Knob-off default remains bit-identical (all changes inside the a2av_recv_dyn_
branch / dyn-only kernel). Residual risks: the underlying uc-vs-CSR slack (if
that is the divergence) is now TOLERATED, not root-caused — slack rows silently
skip; recommend one post-fix b32 cell with the diag armed (small SPIN_LIMIT)
plus a bincount audit of U vs CSR coverage to close the plan-side question.
