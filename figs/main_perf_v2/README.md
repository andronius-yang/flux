# main_perf_v2 — provisional re-measurement lane (opened 2026-09-10 eve)

## STANDING RULING (user, 2026-09-10) — read before touching figs/main_perf

**This lane is an ADDITION, not a replacement.** Nothing here overwrites,
edits, or supersedes `figs/main_perf/figure_src.csv`, its SPEC, or the
main-perf ledger. The re-measured numbers are stored HERE (`figure_src_v2.csv`
when it exists) until the speedups have been shown to be RECREATABLE
(independent re-runs agree within the run band); only then, and only by an
explicit user ruling, may v2 be considered for promotion to the official
figure. Until that ruling, every paper number still comes from
`figs/main_perf`. This survives context compaction: if in doubt, v2 is
provisional and main_perf is authoritative.

## Why a v2 exists

The combine-side Σ pre-reduce was rewritten as a residency-safe streaming
kernel (`FLUX_A2AV_RS_PRERED_STREAM`, `figs/overlap_pipeline/00_brief.md`
§8–§12): layer-1 −2..−4 ms at K2 4n b64 on both the early and the dual3
arms. It changes every arm whose layer-1 transport is `hier_compress`
(Ours s1/s2/dual3, slipstream, LLC/2Ours, EPIC) and no baseline (NVSHMEM,
FAST, Comet, EPLB, MoonEP, and the direct-a2av row, which does NOT run Σ —
see below). The user waived SCHEMA rule 4 (same-binary) for this lane.

## Scope (user rulings 2026-09-10)

1. **Main perf v2 arms** = Ours `ours_l01_s1_pv2_r2` (best-of rule stays;
   expected winner on every plotted group) + **dual3**
   (`ablation_l01_s2_swapall_rst_3d_dual3_str4_p2p_r2` lineage, replacing the
   swap-force row as the recorded "expert dispatch" row — it will NOT be
   drawn: on the oracle-basis pools it pays ~1.2 ms of place bracket and
   moves nothing) + the **direct-a2av row** for "placement + routing only"
   once its transport is tightened (below). Budgets 1/2/4/16/64 MiB,
   K2 + Qwen, 4n/8n/16n. Baselines are NOT re-run.
2. **Case study**: recapture (casestudy3d-style, dual3 rows + COMET
   overlapped / host-gap / sequential / late3) on the new binary.
3. **On hold**: weak scaling (2..32n) and the ablation-cycling figure. When
   the ablation is redone, bar 4 (expert-dispatch overlap) becomes dual3
   (movement genuinely under compute), not the host-gap swap.
4. **NOT canonicalized (user, 2026-09-10 eve).** The binary default stays
   OFF because we do not yet know whether the streaming kernel hangs at
   8n/16n. v2 cells run knob-ON arms (`ours_l01_s1_pv2_r2_prs`,
   `ablation_l01_s2_swapall_nr_3d_dual3_str4_prs_p2p_r2`, env
   `FLUX_A2AV_RS_PRERED_STREAM=1 / STAGES=2 / BLOCKS=6`) on the existing
   knob binary (flux-prered `libflux_cuda.so` a0c60c75, ths_op d3bb40c7);
   no rebuild unless a fix is needed. Gate-first at EVERY topology
   (`--check_iters 1`, random payload, b1 and b16) before any timed cell.
   Budgets **1/2/4/16 only** (no 64 MiB: too much changes, hang risk).
   Protocol = the 8/29 datapoint capsules' (iters 10, warmup 5, isolated,
   lcb homog dslots 64:32 oracle g=0, K2 shape k2 / Qwen shape qwen3).

## Direct-a2av row — what it is and what "fix" means

`ours_l01_s1_pv2_r2_dwire` is ALREADY direct and simple on BOTH layers
(`python/flux/testing/ours_direct.py`): dispatch = pack → All2AllSingle
NVSHMEM one-sided a2av (one `putmem_nbi_block` per destination rank, team
barriers, no signal gating) → place → per-segment un-overlapped GEMM;
combine = per-segment GEMM2 → combine-pack → the same All2AllSingle →
deterministic home accumulation (permutation + one K-axis sum). It runs NO
gateway, NO dedup, NO Σ. Its 16n b16+ collapse (52 ms vs Ours 28 at K2
b16; 200 vs 86 at b64) is the put-count wall: W−1 = 63 puts per rank per
layer through the NVSHMEM host proxy at 0.12–0.6 ms each (handoff 16 §1,
26 §3b), plus un-deduped bytes. "As simple as it gets" cannot remove that
wall on CXI (no IBGDA); DeepEP-LL's cost model assumes GPU-initiated RDMA.
What CAN be tightened (candidate rebuild, user to confirm): one grouped
un-overlapped GEMM per layer instead of per-segment GemmOnly launches;
fold the pack/place index passes into the GEMM epilogue / wire order; one
fused full-grid reduce kernel (CSR scatter-add, bandwidth-bound) instead of
permutation + sum. Expected effect: a few ms at 4n/8n, invisible at 16n
b16+ where the wire wall dominates.

## Files
- (to come) `figure_src_v2.csv`, `figure_src_v2.md`, capsule list.

## Incident log

- **2026-09-10 eve, dual3 (no-reset base) + streaming Σ at K2 4n b1: stuck at
  iteration 0, layer 0** (rank heartbeat parks at `i0 l0`; capsule
  20260911-054339 gate, 968 s idle timeout). Layer 1 — and therefore the
  streaming Σ kernel — was never reached, so this is the dual3 arm at a
  1 MiB budget (never gated below 64 MiB before; cf. the recorded first-try
  b1 hang class of the pv2 s2 arms, handoff 23 §5). Handling: dual3 is timed
  at b2/b4/b16 only, s1 at b1/2/4/16; a discriminator cell (canon dual3,
  shipped Σ, b1) runs last to confirm the hang exists without the new kernel.
  The s1 + streaming Σ b1 gate cell PASSED (106 s, per-iteration checks).
- **Same session, dual3 (no-reset) + streaming Σ at K2 4n b16: also stuck at
  `i0 l0` on all 16 ranks** (977 s). Both stuck cells show "placement basis
  prev_batch drift 16–46 %", i.e. the no-reset dual3 arm on the plain pools
  moves experts at iteration 0 and its first dispatch forward never returns.
  dual3 × no-reset base was never gated (handoff 36 gated dual3 on the
  reset-every base at b64 only; the no-reset A/B used the older `dual`
  issue mode). Ruling for v2: the dual3 recorded row is DEFERRED until the
  3D-scheduling lane root-causes the i0-l0 hang on the no-reset base; v2
  carries Ours s1 (+ the case-study dual3 rows on the reset-every base,
  which run clean with the streaming Σ). Not a Σ-kernel failure.

## Results so far (2026-09-11 00:10) — 4n, Ours s1, knob binary a0c60c75

`figure_src_v2.csv` (16 rows; builder `build_figure_src_v2.py`). Median over 10
timed iterations of the rank-max e2e (the ledger statistic), ms:

| model | budget | ledger (8/29) | v2 shipped Σ (control) | v2 streaming Σ | kernel effect (stream − shipped) |
|---|---|---|---|---|---|
| K2 | 1 | 4.00 | 3.15 | 3.19 | +0.04 (l1 +0.01) |
| K2 | 2 | 4.68 | 3.67 | 3.74 | +0.07 (l1 +0.05) |
| K2 | 4 | 6.05 | 4.92 | 5.10 | +0.18 (l1 +0.12) |
| K2 | 16 | 14.41 | 12.35 | 12.73 | +0.39 (l0 +0.45, l1 −0.03) |
| Qwen | 1 | 3.02 | 2.29 | 2.26 | −0.03 |
| Qwen | 2 | 3.79 | 2.74 | 2.68 | −0.06 |
| Qwen | 4 | 4.71 | 3.71 | 3.68 | −0.03 |
| Qwen | 16 | 11.72 | 9.95 | 10.23 | +0.27 (l0 +0.35, l1 −0.07) |

Capsules: 063901 (K2 stream), 064208 (Qwen stream), 070236 (K2 control),
070505 (Qwen control); gates 054339 / 060414 / 064122 (s1 stream b1 + b16 both
models, per-iteration checks green).

**Findings.**
1. **At 4n and 1–16 MiB the streaming Σ is a no-op on the s1 arm**: l1 moves
   by ≤0.12 ms in either direction; the only ≥0.3 ms totals differences are
   in l0, which the kernel never touches (run-to-run band). Its measured
   benefit (l1 −2..−4 ms) is at 64 MiB and on the swap arms (brief §10–§11).
   Consequence: the PLOTTED main-perf groups (1/4/16 MiB) will not change
   from the kernel; the kernel's story lives in the case study and at b64.
2. **The −12..−29 % versus the 8/29 ledger is drift, not the kernel**: the
   same-binary control reproduces it (−14..−28 %). The 8/29 ledger and this
   session differ in binary generation (505e4bed-era vs a0c60c75/d3bb40c7)
   and in session; this is precisely the "recreatable" question this lane
   exists for, and it says the ledger's absolute Ours numbers are NOT
   reproduced today (baselines were not re-run here, so their drift is
   unknown — rule 4 was waived by ruling, and this is what it costs).
3. dual3 (no-reset) at b1 AND b16 hangs at `i0 l0` with the shipped kernel
   too (discriminator capsule 065410, stuck 493 s): the arm, not the kernel.
4. Case-study v2 recapture (capsule 064254, 10/10, nsys): every OURS row
   now has Σ resident from t=0 and the first combine put at +4.6..5.5 ms
   (was +7.5..8.3); figure recut as `case_study_cs3_v2prs.{svg,png,drawio}`
   (builder `figs/case_study/build_case_study.py --rows cs3v2`). Iteration
   ends: efficient dual3 ≈ 43 ms (COMET overlapped ≈ 53), skewed dual3 ≈ 58.

**Open before 8n/16n:** given finding 1, the 8n/16n s1 cells at 1–16 MiB would
re-measure drift, not the kernel; the informative cells are b64 (excluded by
ruling for hang risk) and the swap/case-study arms. Decide whether 8n/16n
proceed as planned or move to a b64-gated pair per topology.
