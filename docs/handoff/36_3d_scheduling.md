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

## 6. A/B-2 on the RESET-EVERY base — TBD

## 7. Case-study capture 3 — TBD

## 6. Case-study capture 3 — TBD
