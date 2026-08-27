# Handoff 22 — s2 exposed-movement-latency: moved-last GEMM schedule + late-w2 flow (2026-08-26)

Goal (user-directed): minimize the exposed latency of s2 expert movement
with EAGER adoption kept (no next-iteration adoption), by (a) rescheduling
the fused l0 GEMM around the per-iteration moved set and (b) scheduling
the weight flow so token dispatch owns the wire head.

## Motivating derivation (existing data only)
- Supply (OURS s1 16n b64 nsys, capsule 20260825-183714): NIC idle
  40.7 ms of a 113 ms iteration (10.6 ms mid-window, 27.3 ms tail);
  dispatch wire busy 38.2 ms front-loaded under the l0 GEMM.
- Demand (stale K2 8n capsule 20260825-123035): 364 moves/iter global
  = ~0.67 GB/rank/iter — fits the ~1 GB idle capacity ONLY if scheduled.
- Contention (same capsule): stale vs quiet l0 5.9→46.8 ms, place
  4.1→48.7 — ≈ equal parts wire contention and movement-issue cost.

## Mechanisms (both knob-gated, default OFF; new binary = rule-4 boundary)
1. **Moved-last schedule** `FLUX_OURS_SCHED_MOVED_LAST=1`: the static
   schedule's bijective remap (`calc_sorted_problem_schedule_v2`) now
   takes a per-iteration `sched_expert_order` int32[gpe] (bit 30 =
   deferred class = THIS iteration's moved slots; low bits = rank) +
   `sched_n_front`; front class fills the schedule prefix stage-major.
   Release-time criterion: under s2 every tile of a moved slot is truly
   blocked until its push lands (weight gate is per-expert), so deferring
   exactly that class demotes no runnable tile — unlike NR-14's retracted
   `sched_prefetch_last` (its class held resident-weight tiles; ~29%
   regression; that knob stays OFF and is overridden by this one).
   Plumbing: args struct → forward/forward_impl (all 5 sites) → pybind
   (`sched_expert_order`, `sched_n_front`) → `ours_s2.build_sched_order`
   → `gate_kwargs()`. FLUX_CHECK'd to weight-gated static-schedule
   forwards only.
2. **Late-w2 issue** `FLUX_OURS_S2_W2_LATE=1`: `apply_moves` issues only
   w1 (+relay roles); w2 legs are stashed and issued by the driver via
   `lane.issue_w2_late()` immediately AFTER the fused l0 forward is
   enqueued — dispatch legs reach the proxy queue first, w2 rides the l0
   compute shadow (runway = l0 + gelu; `join_w2` unchanged, with a
   failsafe auto-issue). NO current-stream dependency is taken at late
   issue (a wait_stream would park w2 behind the gated GEMM = deadlock);
   w_stream FIFO orders w2 after w1.

## Validation state
- CPU logic test (scratchpad `test_moved_last_logic.py`): 500 trials —
  remap bijection, front-before-deferred consumption, within-class
  order, exact F-D equivalence when the moved set = {eid >= gate}. PASS.
- py_compile clean; C++ diff reviewed; NOT yet compiled (Lustre outage —
  torch headers unreachable). Build with the cudatoolkit 24.5/12.4 pin.
- Arms: `ours_l01_s2_stale_{ml,w2l,mlw2}`, quiet null `ours_l01_s2_mlw2`,
  gates `ours_l01_s2_gate_{ml,mlw2}`. Specs `mlgate_{k2,qwen}_4n.yaml`,
  `mlab_{k2,qwen}_{4,8,16}n.yaml`. Chain: scratchpad `mlrun.sh`
  (probe → build → gate → ab4 → ab8 → ab16), explicit --jobid.

## Predictions to test (from the theory thread)
- Stale mid-scale (8n, moderate moves): clearest ml win (weight landing
  slow, unmoved prefix thick). Quiet: strict null. 16n heavy-move: watch
  for NR-14-style pacing regression (should not appear — the deferred
  class is genuinely blocked and the unmoved prefix carries most rows).
- w2l alone should cut the l0-window wire contention ~half (w2 = half
  the movement bytes); ml+w2l compose.
- Failure modes to watch: gate torn-row family (rule-6 lineage — any
  reorder shifts movement-issue timing; 3/3-green rule before canon),
  join_w2 hang if a driver path skips issue_w2_late (failsafe covers).

## Addendum 2026-08-26 (post-merge): s2xr2 f_cap contract fix

Merged main @ c27b576 (r2 campaign; driver batch-placement f_cap fix
kept). Resolution of handoff 22 §4's open mechanism gap: the planner
(`OursIterPlanner.derive`) now does a rank-LOCAL escalate-and-reroute
when the route kernel reports forced-budget breach (kstats[2] != 0):
re-route with 4x f_cap, then uncapped (kernel forced_left tickets are
per-call workspace ints; f_cap<=0 = INT_MAX/2 already in-kernel).
Sound because the route is sender-local and the breach check happens
BEFORE the phys/probs allgather — no collective asymmetry; nothing
persistent is sized by f_cap; plan_meta's recv_cap assert remains the
loud sizing backstop. The raised cap sticks (adopted geometry is
persistent). s2-only (driver sets planner.f_cap_retry), s1 untouched;
cost = one early device sync per s2 iteration. Provenance:
ours_s2_fcap_retries / ours_s2_fcap_final in records. Re-gate specs:
mlgate_{k2,qwen}_4n_r2.yaml (base r2 first; ml/mlw2 x r2 gates after).

### Layer 2 (found by the first re-gate, 20260826-234231): recv envelope

With the route unblocked, K2 4n died one layer deeper at i0: l1
`a2av_hier send panel overflow` (max_send_rows 7086 > a2av_send_rows_
6212). The C++ check is the collective max over every rank's outbound
combine rows == that rank's dispatch recv — i.e. the adopted placement
concentrated 7086 recv rows on one rank vs the setup envelope 6212
(recv_cap; setup real 5234). Same premise failure as f_cap, one contract
out: reference-derived recv bounds cover only resident/batch placements.
FIX (56e0c92): s2 sizes recv_cap at a placement-INDEPENDENT provable
ceiling — sum of the nlp hottest experts' total batch demands (a rank
hosts <= nlp slots; no expert routes more entries to a rank than its
total demand). l0_recv and RS_MAX_SEND_ROWS inherit via their existing
max(..., recv_cap). Qwen's first re-gate was already green (596 moves,
0 f_cap retries — the merged batch-f_cap max sufficed there; cap 434).

### Layer 3 + RESOLUTION (20260826-2354/2357 gates: BOTH GREEN)

Second re-gate: K2 failed one panel further (combine conv 4641 > 4199).
FIX (27b826a): the whole RS panel family floored at placement-independent
per-dest-population ceilings — column sums are fixed (every owner
receives exactly cpr = S*K), so conv/stage <= (nn-1)*S*K and wire
(unique rows) <= (nn-1)*S; send inherits recv_cap. Third re-gate:
**K2 ok (1600 moves / 4 triggers) + Qwen ok (596 / 4)** — s2 x r2
UNBLOCKED. Note fcap_retries = 0 in the green gates: with sound
envelopes the batch-max f_cap sufficed; the layer-1 escalate-and-reroute
remains armed as the safety net (and the recv/panel FLUX_CHECKs remain
the loud contract). Capsules: 234231/234904 (failed, evidence),
234402/235025/235437/235716 (green).

### noshard crash RCA (capsule 5e606d1f) + issue-cost results

bigchunk (FLUX_OURS_S2_SHARD_CHUNK=8MB) = **-27% K2 4n stale**
(203->149 b8, 221->162 b32; place 121->33ms) — the host-enqueue volume
finding confirmed. noshard died with an ATen IndexKernel device assert:
ROOT CAUSE = my apply_moves hoist took the w_stream.wait_stream snapshot
BEFORE the host prep, so `keep`'s H2D (current stream) could race
w_stream's index_fill_; every arm carries the latent race, but only
noshard lacks the ~100ms plan_weight_shards host gap that hides it.
FIXED: wait_stream moved to just before the w_stream block. noshard
retest queued post-fix; round-2 chunk ladder (8 vs 16MB, ml/w2l stacked
on bigchunk) in flight.

### Issue-cost round 2 (e042d29f, 8/8 ok, K2 4n stale)

Chunk ladder: 16MB == 8MB (+0.03/+0.6ms) — 8MB saturates the lever.
ml on bigchunk: null (+0.1/+1.3) — with fast issue the K2-4n regime is
weight-WIRE-bound during l0 (66ms); every moved tile blocks on landing
regardless of order, so the schedule cannot create bandwidth. w2l on
bigchunk: place 33->10 but total +1.5/+4.0 (serialized wire relocates).
CANON CANDIDATE at K2-scale movement: bigchunk-8MB alone (-27%);
ml stays the Qwen/moderate-movement win. Remaining stale place = 33ms
(plan_weight_shards python + residual issue) — next lever if needed.

### The race family converges (2026-08-27 ~01:00)

K2 8n A/B (ec58b712): stale_ml b8 STUCK at window 12/15 — no asserts,
pure spin-wedge. Trigger inventory for the latent mid-iteration movement
race now spans FOUR independent timing perturbations of the movement
lane: (1) place early-out host sync (handoff 20, 3/3 fail), (2)
plan_overlap side stream (2/2 fail), (3) noshard = no plan_weight_shards
host gap (IndexKernel corruption, persists post wait_stream fix), (4)
moved-last at 8n (adds build_sched_order host work to the place lane;
4n was 10/10 green). Same code green when timing is undisturbed. The
noshard DEBUG_SYNC cell (FLUX_OURS_S2_DEBUG_SYNC) is in flight to
localize the faulting op; corruption-vs-wedge are likely two faces of
one ordering bug in the issue/role/epoch chain. 16n legs proceed
(idle_timeout bounds wedge cost; wedge locations are themselves data).

### Race RCA (debug-sync Heisenbug -> allocator lifetime)

The DEBUG_SYNC noshard cell ran GREEN (08dd0e3b) — per-step syncs remove
the fault => concurrency bug. RCA: the apply_moves hoist allocates
`keep` on the driver stream but consumes it on w_stream with NO
record_stream; on return the caching allocator can recycle the block
mid-index_fill_. Garbage indices: in-bounds -> mis-raised slot epochs ->
unmoved-slot gate never raised -> tile spin-wedge (the stale_ml 8n
face); out-of-bounds -> IndexKernel assert (the noshard face). Triggers
= anything that shrinks the host gap between issue and return. FIX:
keep.record_stream in _issue_op (covers late-w2 reuse too). Historical
8/25 triggers (early-out/ov) predate the hoist — possibly a distinct
sibling; revalidation tells. noshard free-running revalidation launched.
