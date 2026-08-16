# 07 — Comm-only layer-axis campaign (2026-08-16)

The first sweep campaign with a **layer dimension**: layer0 (dispatch), layer1
(combine), and the combined continuous pass (l01). Comm-only arms — no
MoonEP/DeepEP/UltraEP/EPLB. Real Qwen3-235B trace routing
(`pools=mmlu/high_school_world_history;layer=92;sem=homog`, matrix_instance
001), topk=8, G=128, H=ffn=4096, bf16, budgets 1–64 MiB, isolated mode,
Perlmutter 4n/W16 (m5350_g). Verdict protocol: forward + reversed-arm-order
capsule pairs; a knob effect is real only if the sign agrees in both.

**STATUS: layer0 COMPLETE (verdict-grade). Layer1/l01 in flight** on the
post-fix binaries (this doc updates as those capsules land).

## 1. Layer0 headline (final)

Isolated per-layer latency, mean-over-iters of per-iteration max-across-ranks
e2e_ms, forward capsules (7ff7098d blow / 01f7e410 bhigh; fast 03ddc517 /
4124b1a5); reverse twins 78f371c6 / c914e099 agree:

| budget | torch (unfused) | fast (BvN a2av) | allgather (stock flux) | lb_union base | **lb_union fused+early** |
|---|---|---|---|---|---|
| b1  | 1.923 | 4.644 | 1.661 | 1.581 | **1.238** |
| b2  | 3.342 | 5.085 | 2.296 | 2.002 | **1.673** |
| b4  | 6.425 | 7.275 | 3.534 | 2.845 | **2.588** |
| b8  | 12.784 | 11.624 | 6.114 | 4.721 | **4.469** |
| b16 | 25.484 | 20.731 | 11.773 | **9.093** | 9.154 |
| b32 | — (OOM) | 40.085 | 23.843 | 19.705 | **19.227** |
| b64 | — (OOM) | — (heap fail) | 52.201 | 45.697 | **44.447** |

- torch reference is infeasible ≥ b32 (materializes the full unsharded
  tensor) — a real feasibility boundary, not a harness gap.
- fast's flat in-window BvN schedule recompute floors it at small budgets
  (crosses torch at b8); b64 hits the known 16G symmetric-heap team-creation
  ceiling (re-confirmed; documented loss).
- Winner vs stock flux: **−26..−29% at b2–b32**, −16% at b64, 2.9× vs torch
  at b8.

## 2. The 2³ factorial (final; the campaign's core)

Knobs on the `hier_compress_lb_union` base: F=`FLUX_A2AV_FUSED_STAGE2`,
N=`FLUX_A2AV_FANOUT`, E=`FLUX_A2AV_EARLY_LAUNCH`; all 8 corners, fwd+rev.
Sign-stable verdicts (fwd/rev agreement, per budget):

- **F: WIN** (−0.24..−0.53 ms wherever both orderings agree; b16/b64
  inconclusive — the NR-14 noise zone). → canonicalization-grade evidence
  for flipping the FUSED_STAGE2 default.
- **E: WIN** (−0.04 at b1 → −1.78 ms at b64, growing with budget). →
  EARLY_LAUNCH graduates from "its own configuration" to a recommended
  default on lb_union (guards: conn>1 on compress paths; never with
  PACK_OVERLAP).
- **N: LOSS** (+0.02 at b1 → +1.2 ms at b64). The 2026-08-07 lb_union_eager
  A/B finally resolves: **fanout off / delete the temp arm**. (Note: the 2n
  synthetic smoke had suggested otherwise — real-trace 4n reversed it; the
  repeat-with-reversed-order protocol is what made this trustworthy.)
- **N×E interaction: beneficial and sign-stable at every budget** (fanout
  hurts much less under early-launch) — mechanistically interesting (both
  contend for issue order; early-launch drains the contention), practically
  moot since N's main effect loses.
- FN, FE: budget-dependent, no stable verdict.

Winner configuration: **lb_union + FUSED_STAGE2 + EARLY_LAUNCH** — the l01
pairing arm (`l01_lbunion_hier_eager`) carries it.

## 3. Two layer1 hang bugs found and fixed (bring-up ladder + debug track)

The campaign's bring-up ladder caught, and the parallel debug track fixed,
two distinct never-before-seen hangs (layer1 multi-node had only ever run
2n × synthetic before today):

1. **Lazy-load deadlock** (2n eager/compress): `CUDA_MODULE_LOADING=LAZY`
   defers first-launch kernel module loads; eager/compress launch persistent
   spin kernels first, and NVSHMEM 3.2.5 delivers every on-stream signal via
   a device kernel whose first-ever launch then queues behind the
   never-exiting spinner → circular wait. Fix: ctor-time kernel preload +
   NVSHMEM transport priming (branch worktree-l1-hang-debug, 1550b67,
   merged). Validated 2n: eager + compress green, 8/8 allclose (compress
   with CHECK_IDENTITY on its first correct GPU run ever).
2. **Empty-expert split-cascade mis-bucketing** (ALL l1 arms × trace
   matrices, any node count): `set_barrier_ptr` in
   gather_rs_gemm_grouped_with_absmax.h divided the full-list problem_idx by
   the NON-empty problem count; real-trace routing leaves ~15/128 experts
   with zero tokens, mis-bucketing completions so a per-split barrier never
   fires. This is why every l1 trace cell hung (capsule 20260816-134402,
   20/20 stuck) while synthetics were green — the "4n hang" was a
   matrix-family confound. Fix: one-liner (branch worktree-l1-nn4-debug,
   c9b82b6, merged). Validated 4n on the exact hang config: hier 16/16
   allclose 8.12 ms, dense 15.54 ms. Sibling `gemm_rs` set_barrier_ptr:
   unaudited, flagged.

Method note for the ledger: the two bugs masked each other across scales
(2n synthetic green + 4n trace stuck read as "node-count bug"), and the
family confound was only broken by a same-binary same-nodes A/B of
remotefrac-vs-trace. Lesson: bring-up ladders must include a REAL-routing
rung, not just synthetic.

## 4. Layer1 + combined (PENDING — capsules in flight)

- Layer1 five-arm capsule pairs (dense, hier, hier_eager, compress,
  compress_eager × tmiso/tmamo) re-run on the fixed binary; early 2n
  observations: hier tmiso−tmamo gap ≈ 0.33–0.43 ms (the schedule-
  inheritance value, as predicted); eager costs +80% over legacy at 2n b8
  and the gap persists at 4n (6.88 vs 3.78 ms remotefrac) — the eager-vs-
  legacy verdict on real traces is the open headline question.
- l1_fast baseline: port validated (bitwise NCCL cross-check + allclose,
  first GPU run green); b1–b4 capsule cells landed (e21d3f34), b8+ rerun
  pending.
- Combined l01: bench + identity checker landed
  (test/python/moe_combined/test_moe_l0l1_traffic.py,
  sweeps/check_l01_identity.py); first GPU bring-up + capsules pending.
  Acceptance gate: e2e(l01) ≈ e2e(l0 iso) + act + e2e(l1 tmamo) within 10%.

## 5. Capsule index (this campaign, committed)

P2 bring-up: 20260816-123413 (l1 identity rung), -125015 (clean l1 rung),
-131627 (factorial 8-corner smoke, 8/8 green).
P3 layer0: -132227 (blow fwd 45/45), -133612 (bhigh fwd 18/18), -161758
(blow REV 45/45), -163134 (bhigh REV 18/18); fast: -134133 (5/5), -134308
(1/2, b64 documented loss). Third factorial confirmation: 1451c017.
Layer1 negatives: -134402 (20/20 stuck, pre-fix trace hang, the bug-2
evidence), -170140 (binary-B 4n smoke negative — the discriminator).
Binary discipline: fwd/rev pairs share a binary; layer0 capsules = pre-fix
binary; layer1/l01 results quote post-fix binaries only. Never compare
across.

## 6. Open items

- Layer1/l01 capsules on the fixed binary (in flight) + this doc's §4.
- Canonicalization patches: FUSED_STAGE2 + EARLY_LAUNCH defaults on
  lb_union; delete FLUX_A2AV_FANOUT + the lb_union_eager arm (verdict §2).
- gemm_rs set_barrier_ptr audit (same pattern as bug 2's site).
- trace-matrix compress/eager smoke post-fix (debug-agent recommendation).
- 16n closure cells (separate campaign; W64 fixes still unvalidated at 16n).
