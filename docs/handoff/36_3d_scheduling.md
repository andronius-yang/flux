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

### 7.1 Timelines — TBD (extractor run)

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

### 7.1 Timelines — TBD (extractor run)

## 6. Case-study capture 3 — TBD
