# Handoff 36 — 3D scheduling: expert movement under token comm AND compute (2026-09-08/09)

Worktree `$PSCRATCH/workspace/andrewy/flux-3dsched`, branch `3dsched` (off main
cdc9fa7 + main's 9/5 uncommitted case-study code). Binary: ths_op sha
**a4f418de** (built 9/9 00:00 on nid001084, cuda 24.5/12.4 pins, script
`$PSCRATCH/workspace/andrewy/build_3dsched.sh`). Never mix with the 9/5
case-study capsules (121506 / 141104 = pre-3dsched binaries).

## 1. Ask (user, 9/8)

Paper section "three phase scheduling": the expert swap was overlapped only
inside the host planning gap (early issue lands in the ~2.4 ms plan window,
nsys shows NVLink idle under the GEMM/wire). Required: (a) total_ms equal or
lower, (b) the NVLink expert-movement block visible UNDER the inter-node
blocking puts and the GEMM, (c) BEST = the dispatch-side weights (w1) and the
combine-side weights (w2) as TWO SEPARATE blocks — one under layer 0, one
under layer 1 — not two contiguous copies. Lane = the composed 8-slot orbit
(`swapall`, cap 8 slots/rank) so the NVLink volume is visible. Case study
only (K2 4n b64, S-C dwell-4 schedule + plain lcb). Rebuild + recapture
approved.

## 2. Prior record (why the copy was never on the critical path)

Handoff 25 (8/28-29, one-slot force lane, 4n): late issue == early at b64
(K2 46.95 vs 47.41, Qwen 41.06 vs 41.39), +0.5-0.7 ms at b1 (moved-slot
spin); late nsys = copies 100 % under GEMM / 84 % under the dispatch wire.
9/5 case study (t1 lane, early issue): ovl 49.6 (S-C block) / 46.7 (plain)
vs sequential twin 51.6 / 46.3 -> the 0.66 ms copy is already hidden; only
the host chain (decide + apply + issue, ~0.8 ms) is on total_ms. 9/2 nsys
(8-slot orbit, early): 8 x 29 MB on one stream = 2.4 ms, lands 5 ms BEFORE
the l0 GEMM; 4 streams -> 0.9 ms; l0 moved-last HURT (+3 ms GEMM).

## 3. What was built (commit cc15f44 + 080bc36 + 6c94e24)

**Layer-1 combine-side weight gate (kernel + op, rebuild).**
`gather_rs_gemm_grouped_with_absmax.h`: new `prob_wgate_map` (per-problem
index into `weight_signal_ptr`, -1 = ungated) + `weight_signal_expected`;
a gated problem's tiles spin at tile start (thread 0, system-scope acquire)
until the slot's landed epoch >= expected — the layer-1 twin of the layer-0
per-slot gate. `GemmGroupedV2GatherRSOp.set_weight_gate(weight_signal,
epoch, gate_of_expert)` arms the NEXT forward (one-shot). Inside the msplit
forward the gated experts are ordered LAST inside every combine wave
(`idx_of` permutation; `prob_eid` + `prob_group_map` shipped so
make_workspace / the cascade flags follow the permuted list; the wave =
destination-node axis is untouched); the per-problem map is host-built on
the final order and staged through a pinned buffer with one async H2D.
Legacy (wave-adapt collapsed) path = gate only.

**Swap lane issue modes (`OursSwapAllLane(issue=...)`, python-only).**
`early` (9/1 ablation: both matrices in the place bracket) / `late` (both
after the l0 enqueue) / `split` (w1 early, w2 late) / **`dual`**: w1 phase
after the l0 enqueue (dispatch side; l0 per-slot gate, optional moved-last),
w2 phase enqueued right before l1 with the movement streams waiting `ev_l0`
(= l0 complete), so the block sits under the l1 GEMM + combine wire; the l1
per-problem gate (`l1_gate_kwargs`) replaces the stream-level w2 join;
`l1_join()` after the l1 enqueue keeps the next iteration's table/signal
writes ordered after the pulls (an EMPTY gated expert schedules no tile).
Each matrix is one push-all / join / pull-all phase over N movement streams
(`FLUX_OURS_SWAP_STREAMS`). Fix: moved-last no longer requires the exchange
to be issued before `gate_kwargs()` (late/split/dual issue after the l0
enqueue).

Driver: `--swap_issue dual` (swap_rounds all), `swap.issue_l1` NVTX range,
`OursRunner.l1_forward(gate_kwargs)`. Extractor (`figs/case_study/
extract_timeline.py`): swap copies tagged by issue phase (early/late/l1),
per-phase spans, swap-under-GEMM / swap-under-NIC overlap ms.

Arms: `ablation_l01_s2_swapall_{nr,rst}_3d_{late,dual}{,_ml,_str4,_ml_str4}_p2p_r2`
(+ `_gate`), `..._3d_{early,noov}_str4_...`. Specs: `abl3d_gate_k2_4n`,
`abl3d_ab_k2_4n`, `abl3d_ab2_k2_4n`, `casestudy3_3d_sc_d4_k2_4n_nsys`.

## 4. Gates (capsule 20260909-070206_perlmutter_e158d083, 4/4 ok)

proLaw severe cell, reset-every (every timed iteration = the full 8-slot
orbit event, 253 global swaps/iter), `--check_iters 1`: dual_ml_str4,
dual_ml (1 stream), dual_str4, late_ml_str4 — 176/176 per-iteration output
checks OK, 0 BAD on every cell. Host split under dual: d2h 0.2-0.7 / decide
2.1 / apply+issue 0.19 ms (the ~80 enqueues moved off the place bracket).
Gate-mode (perturbed) totals: dual_str4 58.1 (l0 23.0) vs dual_ml_str4 64.3
/ dual_ml 63.9 / late_ml_str4 63.8 (l0 28.9-29.6) -> **l0 moved-last costs
~6 ms of l0 on the 8-slot lane** (9/2 finding reproduced under late issue).

## 5. A/B-1 on the NO-RESET base (capsule 20260909-071236_perlmutter_8bf1c017, 16/16 ok)

`swapall_nr` (placement carried across topics, the cycling-ablation harness):
the composed orbit CONVERGES, so after a block entry nothing moves — place
bracket 0.5 ms in every arm, movement only at block entries (0.7-1.8 ms).
Rank-max medians, K2 4n b64 isolated (S-C dwell-4 schedule / plain lcb):

| arm (nr base, 4 streams unless noted) | S-C total | plain total |
|---|---|---|
| early (9/1 issue point) | 50.43 | 45.95 |
| noov (sequential) | 51.23 | 46.18 |
| early, 1 stream (9/2 arm) | 51.40 | 45.87 |
| late + l0 moved-last | 51.49 | 45.65 |
| dual | 51.80 | 45.92 |
| dual + l0 moved-last | 51.43 | 45.99 |
| dual + moved-last, 1 stream | 51.45 | 45.57 |
| COMET overlapped (no-gate c8) | 52.34 | 54.34 |

Verdict: within noise (spread ~1.4 ms S-C, ~0.6 plain) — the nr base cannot
discriminate issue modes and would show ~no NVLink in the figure. The
user's "8 swaps" requirement is met by the RESET-EVERY base (A/B-2, §6).

## 6. A/B-2 on the RESET-EVERY base (capsule 20260909-072958_perlmutter_7d007d60, 12/12 ok)

`swapall_rst`: the placement is restored to the oracle basis before EVERY
timed iteration, so every iteration carries the full composed orbit (the
9/2 nsys proxy; 253 global swaps/iter on the proLaw block). K2 4n b64,
isolated, rank-max per iteration; S-C schedule = median / mean over the 32
iterations (8 topic blocks of 4) and the proLaw block median; plain = median.

| arm (rst base) | S-C med | S-C mean | proLaw block | plain med | l0 / l1 (S-C med) |
|---|---|---|---|---|---|
| early, 4 streams (current issue point) | 52.57 | 54.40 | 61.4 | 48.02 | 20.65 / 27.29 |
| noov (sequential), 4 streams | 53.25 | 54.57 | 64.9 | 47.55 | 20.36 / 26.97 |
| **late**, 4 streams (both under l0) | **51.35** | **52.33** | 58.1 | 47.59 | 20.66 / 26.93 |
| **dual**, 4 streams (w1 under l0, w2 under l1) | 53.49 | 53.44 | **57.0** | **47.21** | 20.55 / 28.29 |
| dual, 1 stream | 52.54 | 53.00 | 59.3 | 47.28 | 20.97 / 27.43 |
| dual + l0 moved-last, 4 streams | 52.92 | 54.21 | 63.1 | 47.89 | 21.13 / 27.61 |

Reading:
- Moving the issue point out of the host gap is free on l0 at b64 (late
  l0 == early l0) and REMOVES ~0.9 ms of place bracket + ~1.1 ms of plan
  bracket (the ~80 enqueues + the copy/plan contention leave the timed
  host chain): late = -1.2 ms median / -2.1 ms mean vs early.
- dual keeps the l0 gain but its l1 is +1.0-1.4 ms on the mild blocks
  (median 53.49 vs late 51.35); on the severe proLaw block dual is the
  fastest arm (57.0 vs 58.1 late / 61.4 early / 64.9 sequential). vs the
  current early point: dual is better on the mean (-1.0..-1.4) and on plain
  (-0.8), at parity (1 stream) / +0.9 (4 streams) on the S-C median.
- l0 moved-last stays a loser on the 8-slot lane (+0.5-1.6 ms; the deferred
  class = ~30 % of the rank's experts makes a poorly filled tail) — the
  case-study arm runs WITHOUT it.
- 1 vs 4 streams: within noise on total (4 streams shorten the NVLink block
  0.9 vs 2.4 ms; the l1-side cost does not scale with the block length, so
  it is a landing-time / contention effect, not copy time — see the capture).

## 7. Case-study capture 3 (capsule 20260909-074415_perlmutter_05252de1, 20/20 ok)

Spec `casestudy3_3d_sc_d4_k2_4n_nsys` (nsys + isolated, one binary): COMET
overlapped (no-gate c8) / gated COMET / OURS 8-slot reset-every early /
sequential / **3D dual** (4 streams, no l0 moved-last). Isolated rank-max
per-iteration latency (S-C median / mean / proLaw block; plain median):

| arm | S-C med | S-C mean | proLaw | plain |
|---|---|---|---|---|
| COMET overlapped (no-gate c8) | 51.43 | 54.81 | 52.7 | 54.20 |
| COMET gated | 54.71 | 56.64 | 64.8 | 60.00 |
| OURS early (host-gap issue) | 53.33 | 53.98 | 59.2 | 48.02 |
| OURS sequential (noov) | 53.60 | 54.87 | 65.6 | 48.44 |
| **OURS 3D dual** | **51.98** | 53.97 | **58.7** | **47.09** |

Same-capsule verdict: dual <= early on both families (S-C -1.35 median,
mean equal; plain -0.9; proLaw block -0.5), l1 27.54 vs 27.73 -> the
+1.4 ms l1 seen in A/B-2 was run-to-run variation, not a systematic cost.
The user condition (total_ms equal or lower vs the host-gap overlap) holds
for dual in every same-capsule comparison of capture 3 and in 3 of 4 in
A/B-2. Supplement `casestudy3b_3d_late_k2_4n_nsys` = the late arm's
timeline rows (best total in A/B-2), same binary/session.

### 7.1 Timelines (extractor: figs/case_study/extract_timeline.py, JSON on
PSCRATCH figs_data/case_study/timeline_20260909-074415.json + -080251.json)

Medians over the 16 ranks of the device-time placement of the swap copies
(ms from the iteration start; "under" = swap busy time concurrent with a
GEMM kernel / a NIC put kernel):

| arm, case | l0 GEMM | l1 GEMM | NIC puts | swap block(s) | under GEMM / NIC |
|---|---|---|---|---|---|
| early, skewed iter33 | 11.7-29.3 | 32.8-42.9 | 11.7-51.9 | host gap 2.6-3.9 | 0 / 0 |
| sequential, skewed | 17.2-33.1 | 38.2-48.3 | 17.0-58.7 | host gap 2.7-4.1 (+5.5 ms wait) | 0 / 0 |
| late (3b), skewed | 7.8-23.9 | 31.1-41.2 | 7.6-51.4 | after l0 enqueue 5.7-7.5 | 0 / 0.3 |
| dual, skewed | 8.2-24.0 | 30.7-40.9 | 8.0-49.8 | w1 6.2-6.7, w2 29.1-29.6 | 0 / 0 |
| dual, efficient iter4 | 6.5-20.9 | 23.5-34.4 | 6.3-41.3 | w1 4.6-4.9, w2 22.2-22.6 | 0 / 0 |

**Finding (the reason for round 2):** enqueue order alone does NOT place
the copies under the GEMM. The host runs ~2 ms ahead of the GPU, so a
block enqueued right after the l0 forward (late, dual-w1) executes while
the GPU is still in the plan tail and finishes 1.5 ms before the GEMM
starts and before the first NIC put; dual-w2, gated on l0 COMPLETION,
lands in the 3-6 ms l0->l1 gap (gelu + combine meta) and finishes before
the l1 GEMM starts. Every arm therefore reads 0 ms under GEMM. What late /
dual DO buy is the host chain: the l0 GEMM starts 3.5 ms earlier than
early on the skewed block (8.2 vs 11.7) because the ~80 enqueues leave the
place bracket — that is the -1..-2 ms total_ms of §6/§7, not overlap.

Round 2 (`late2` / `dual2`, python-only, same binary): the phases wait on
DEVICE events — w1 on "the stream reaches the l0 op" (recorded right before
the l0 enqueue), w2 on "the stream reaches the l1 op" (recorded in
issue_l1). Both fused ops open with a node barrier, so the peers' pushes
start together; the l0 per-slot gate / l1 per-problem gate + in-wave
moved-last absorb the ~1 ms landing. Specs abl3d_gate2 -> abl3d_ab3 ->
casestudy3c (job 58110152).

## 7. Case-study capture 3 (capsule 20260909-074415_perlmutter_05252de1, 20/20 ok)

Spec `casestudy3_3d_sc_d4_k2_4n_nsys` (nsys + isolated, one binary): COMET
overlapped (no-gate c8) / gated COMET / OURS 8-slot reset-every early /
sequential / **3D dual** (4 streams, no l0 moved-last). Isolated rank-max
per-iteration latency (S-C median / mean / proLaw block; plain median):

| arm | S-C med | S-C mean | proLaw | plain |
|---|---|---|---|---|
| COMET overlapped (no-gate c8) | 51.43 | 54.81 | 52.7 | 54.20 |
| COMET gated | 54.71 | 56.64 | 64.8 | 60.00 |
| OURS early (host-gap issue) | 53.33 | 53.98 | 59.2 | 48.02 |
| OURS sequential (noov) | 53.60 | 54.87 | 65.6 | 48.44 |
| **OURS 3D dual** | **51.98** | 53.97 | **58.7** | **47.09** |

Same-capsule verdict: dual <= early on both families (S-C -1.35 median,
mean equal; plain -0.9; proLaw block -0.5), l1 27.54 vs 27.73 -> the
+1.4 ms l1 seen in A/B-2 was run-to-run variation, not a systematic cost.
The user condition (total_ms equal or lower vs the host-gap overlap) holds
for dual in every same-capsule comparison of capture 3 and in 3 of 4 in
A/B-2. Supplement `casestudy3b_3d_late_k2_4n_nsys` = the late arm's
timeline rows (best total in A/B-2), same binary/session.

### 7.1 Timelines (extractor: figs/case_study/extract_timeline.py, JSON on
PSCRATCH figs_data/case_study/timeline_20260909-074415.json + -080251.json)

Medians over the 16 ranks of the device-time placement of the swap copies
(ms from the iteration start; "under" = swap busy time concurrent with a
GEMM kernel / a NIC put kernel):

| arm, case | l0 GEMM | l1 GEMM | NIC puts | swap block(s) | under GEMM / NIC |
|---|---|---|---|---|---|
| early, skewed iter33 | 11.7-29.3 | 32.8-42.9 | 11.7-51.9 | host gap 2.6-3.9 | 0 / 0 |
| sequential, skewed | 17.2-33.1 | 38.2-48.3 | 17.0-58.7 | host gap 2.7-4.1 (+5.5 ms wait) | 0 / 0 |
| late (3b), skewed | 7.8-23.9 | 31.1-41.2 | 7.6-51.4 | after l0 enqueue 5.7-7.5 | 0 / 0.3 |
| dual, skewed | 8.2-24.0 | 30.7-40.9 | 8.0-49.8 | w1 6.2-6.7, w2 29.1-29.6 | 0 / 0 |
| dual, efficient iter4 | 6.5-20.9 | 23.5-34.4 | 6.3-41.3 | w1 4.6-4.9, w2 22.2-22.6 | 0 / 0 |

**Finding (the reason for round 2):** enqueue order alone does NOT place
the copies under the GEMM. The host runs ~2 ms ahead of the GPU, so a
block enqueued right after the l0 forward (late, dual-w1) executes while
the GPU is still in the plan tail and finishes 1.5 ms before the GEMM
starts and before the first NIC put; dual-w2, gated on l0 COMPLETION,
lands in the 3-6 ms l0->l1 gap (gelu + combine meta) and finishes before
the l1 GEMM starts. Every arm therefore reads 0 ms under GEMM. What late /
dual DO buy is the host chain: the l0 GEMM starts 3.5 ms earlier than
early on the skewed block (8.2 vs 11.7) because the ~80 enqueues leave the
place bracket — that is the -1..-2 ms total_ms of §6/§7, not overlap.

Round 2 (`late2` / `dual2`, python-only, same binary): the phases wait on
DEVICE events — w1 on "the stream reaches the l0 op" (recorded right before
the l0 enqueue), w2 on "the stream reaches the l1 op" (recorded in
issue_l1). Both fused ops open with a node barrier, so the peers' pushes
start together; the l0 per-slot gate / l1 per-problem gate + in-wave
moved-last absorb the ~1 ms landing. Specs abl3d_gate2 -> abl3d_ab3 ->
casestudy3c (job 58110152).

## 6. Case-study capture 3 — TBD

## 8. Round 2 — device-gated issue (gate-2 20260909-081315 3/3, A/B-3 20260909-082135 12/12)

Gate-2 (reset-every proLaw, check_iters=1): dual2 4-stream / dual2 1-stream
/ late2 4-stream = 176/176 OK, 0 BAD each (the l1 per-problem gate holds
with the copies landing under the l1 GEMM).

A/B-3, RESET-EVERY base, K2 4n b64 isolated, rank-max per iteration:

| arm | S-C med | S-C mean | proLaw | plain med | l0 / l1 / place (S-C med) |
|---|---|---|---|---|---|
| early, 4 streams (current issue point) | 52.66 | 53.65 | 60.4 | 47.84 | 20.52 / 26.77 / 2.23 |
| dual (capture-3 semantics), 4 streams | 52.33 | 52.79 | 59.2 | 47.68 | 20.55 / 27.49 / 1.40 |
| **late2**, 4 streams | **51.70** | **52.12** | 58.8 | **47.07** | 20.74 / 26.76 / 1.39 |
| **dual2**, 4 streams | 52.26 | 52.53 | 59.6 | 47.48 | 20.78 / 27.16 / 1.47 |
| **dual2**, 1 stream | 51.90 | 52.45 | **58.2** | 47.33 | 20.66 / 27.11 / 1.39 |
| dual2 + l0 moved-last, 4 streams | 52.49 | 54.26 | 64.5 | 47.79 | 21.11 / 27.02 / 1.40 |

Verdict: with the copies now genuinely under the GEMMs (§9), dual2 is at
parity or better than the host-gap issue point on every statistic
(S-C -0.4..-0.8 median / -1.1 mean; plain -0.4..-0.5; proLaw block -0.8..
-2.2). The landing under the GEMM costs l0 +0.2 / l1 +0.4 ms (gated tiles
+ NVLink contention) and buys -0.8 ms of place bracket + ~-1 ms of plan
bracket. late2 (both under l0) remains ~0.5 ms better than dual2 — the
price of the two-sided picture. l0 moved-last is a loser again (+0.2 med,
+1.7 mean, +4.9 on proLaw). 1 vs 4 streams: 1 stream is marginally better
on total (longer, thinner NVLink block: less contention with the token
forwards); either is fine for the figure.

## 9. Capture 3c timelines (device-gated late2/dual2; capsule 20260909-083619, 6/6)

| arm, case | l0 GEMM | l1 GEMM | NIC puts | swap block(s) | under GEMM / NIC |
|---|---|---|---|---|---|
| late2 4str, skewed iter33 | 8.4-24.3 | 32.2-42.3 | 8.3-50.6 | 6.3-8.1 | 0.2 / 0 |
| dual2 4str, skewed | 7.7-25.0 | 30.8-41.0 | 7.6-50.7 | w1 5.6-6.1, w2 29.5-29.9 | 0 / 0 |
| dual2 1str, skewed | 7.4-23.4 | 28.3-38.6 | 7.1-47.9 | w1 5.4-6.2, w2 27.0-27.7 | 0 / 0 |
| dual2 4str, efficient iter4 | 6.4-20.9 | 23.4-34.5 | 6.2-41.7 | w1 4.4-4.8, w2 22.5-22.9 | 0 / 0 |

**Finding (the reason for round 3):** "the stream reaches the op" is still
1-2 ms before the GEMM kernel: both fused ops run a metadata/staging
prologue (dispatch: meta + node-pack + relay staging; combine: workspace +
combine meta) before launching their GEMM, and the l0 NIC puts start only
~0.3 ms before the GEMM. A 0.5-0.7 ms block that starts with the op
therefore still ends before the GEMM. The only device point that is the
GEMM is the GEMM launch itself.

Round 3 (`late3` / `dual3`, C++ + python, rebuild): both fused ops get a
one-shot GEMM-START MARK — `set_gemm_start_mark(epoch)` arms the next
forward to write `epoch` into a device int64 on the forward stream
immediately before the GEMM kernel launch (l0: `Step 5: launch GEMM`; l1:
after the pre-GEMM node barrier); `gemm_start_mark()` returns the flag.
The lane's movement streams wait on it with the zero-SM
cuStreamWaitValue64 (GEQ), so the NVLink block starts WITH the GEMM (w1)
and WITH the l1 GEMM (w2); the l0 per-slot gate / l1 per-problem gate +
in-wave moved-last cover the ~1 ms landing. Specs abl3d_gate3 ->
abl3d_ab4 -> casestudy3d.

## 10. Round 3 status (binary ths_op d3bb40c7 = a4f418de + GEMM-start mark)

Gate-3 (capsule 20260909-102602, debug window 58113218): **late3** (both
matrices gated on the l0 GEMM-start mark) 176/176 OK, 0 BAD. **dual3**
(w1 on the l0 mark, w2 on the l1 mark) WEDGES in the first iteration's
layer 1 on all 16 ranks (4 and 1 streams alike; the exchange was issued,
every rank printed its "i0 l1" heartbeat and none returned). The l1
per-problem gate was never actually exercised under spin before (in dual2
the copies landed before the l1 GEMM started), so the wedge is specific to
copies landing while the l1 GEMM (+ the combine's persistent pack /
pre-reduce / reduce blocks, which fill the SM margin) runs. Hypotheses:
(a) SM starvation of something the swap needs an SM for while every SM is
held by the gated GEMM + persistent combine blocks; (b) a gate-map / index
mismatch that only shows when a gated tile really has to wait. Diagnostic
queued: `abl3d_gate3b` (dual3 with sm_margin 16, short idle timeout) in
window 2 (job 58113504) after `casestudy3e` (late3 nsys rows) and
`abl3d_ab5` (late3 vs early, isolated).

Interim recommendation: **late3** = NVLink block starting WITH the l0 GEMM
(under GEMM + dispatch puts), total_ms <= early (gate-mode 58.1 vs dual
58.05 earlier; isolated A/B-5 pending); the two-sided (w2 under l1)
variant needs the wedge resolved first.
