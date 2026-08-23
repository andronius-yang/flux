# 14 — Mission-4 dispatch audit: hier_compress lb_union under the comm-only runner

Session 2026-08-23 (continuation of the LLC campaign, worktree `flux-place-fast`,
branch `place-fast`). Mission: audit + optimize the isolated l01 latency of the
best comm+comp fusion arm (`l01_lbunion_compress`) against the comm-only
baselines (torch+gemm, fast+gemm, flux dense allgather), canon rule-10 data
(K2 + Qwen3, livecodebench/execution layer 5 homog), b1–b64, 4n/16r.
Binary: fresh worktree build (pre-fix sha `62383055…`, candidate binary after
commit `80e500c`). All capsules committed on `place-fast`; jobs 57460652 +
57466107 (~75 min of 4n total).

## A. Baseline table (iso totals, mean-of-per-iter-max, 10 iters; FAST = e2e-only lane)

| b | K2: compress / dense / torch | qwen: compress / dense / torch | qwen fast (e2e) |
|---|---|---|---|
| 1 | 5.94 / 6.80 / 9.85 | 4.21 / 5.54 / — | 12.75 |
| 2 | 6.52 / 9.68 / 10.41 | 4.72 / 6.23 / — | 12.46 |
| 4 | 7.66 / 10.69 / 12.12 | 6.07 / 7.15 / — | 15.59 |
| 8 | 10.38 / 13.22 / 15.68 | 8.95 / 9.93 / 13.08 | 39.10(b16) |
| 16 | 16.45 / 18.80 / 25.76 | 15.49 / 16.46 / — | — |
| 32 | 29.04 / 33.91 / 48.49 | 27.39 / 30.39 / — | 62.06 |
| 64 | 53.55 / 66.24 / 90.00 | 52.58 / 57.71 / — | 122.83 |

compress wins EVERY budget both models (K2 −12..−33% vs dense; qwen −6..−24%).
l0 ≈ l1 at high budgets (K2 b64: 25.0/26.5) — the combine side carries half.
compress plan lane grows with budget (0.70 → 1.55/2.02 ms at b64): the
in-window unique-counts derive + host collapse; a 16n+ optimization surface.
**K2 × FAST is broken**: the l01_fast b8 cell hangs (killed by the stuck
detector, 2 attempts; qwen fast green in 25 s) — first-light issue in the FAST
lane at K2 shape (G=384/H=7168), diagnosis deferred, K2 fast cells dropped.
K2 first-light fix that unblocked everything: `SHAPE_PRESETS["k2"]` now pins
`n_split_l1=7` (7168/7=1024; the driver asserts N/n_split % 512 == 0).

## B. The audit (schedule map + verdicts)

Full stream/signal map produced by subagent (5-section engineering map: l0
schedule, LB_UNION arithmetic, l1 combine cascade, starvation surface,
ingestion boundary) — see the session transcript; headline verdicts:

1. **The schedule is logical and the orders are right.** The descending-ring
   relay put order makes source (my+dn) send to me in ITS round dn, so the
   gateway's ascending-dn wait order matches expected arrival. The l1 pack
   kernel produces `group_flags` in exactly the rotation the conv/wire
   ladders consume (no head-of-line by construction — my C10-conv candidate
   dissolved on inspection). Tier B's window keying, the always-signal
   invariant, and the CHECK_COMPRESS lane-monotonicity are all sound.
2. **Two real serialization findings, one big one:**
   - the l1 wire ladder serialized `n_split·(NN−1)` independent blocking
     puts on ONE internode stream → **FLUX_A2AV_RS_WIRE_STREAMS=2**
     (parity-split; shipped as the winning knob, below);
   - the l0 relay enqueues ALL rounds' pulls before the first wire put
     (stream FIFO ⇒ round-1's put waits round-3's staging, zero data dep)
     → **FLUX_A2AV_RELAY_PULL_STREAM=1** (2-stream + per-round events)
     — implemented, measured ≈ 0: the pull mass on real traces is too
     small (NVLink ≈ 12× wire BW) and the extra stream costs at b64.
     CLOSED-NOISE, knob kept as ablation.
3. **Kernel starve found — not comm/comp, but comm/comm:** the l1 ladders'
   fixed CTA budgets (PACK/REDUCE/PRERED 6/6/4, canonicalized at K3 shape)
   are payload-proportional work behind fixed grids. At b64 they starve:
   **10/8/6 cuts l1 by 1.9–2.1 ms at b64 both models** (monotone 4→6→10
   dose-response; audit item 4e) and ties at b2/b8 (the 16-extra-CTA GEMM
   tax never bites — comm-bound where it matters).
4. **FANOUT re-test under blocking wire + real traces:** still mixed
   (b32 −0.5 / b64 +1.2 at K2; sign-unstable at qwen) — the NR-06/8.17
   closure STANDS.

## C. The sender-local partition question (user's exploratory ask) — ANSWERED

The lb partition (`chunk_bound`) is ALREADY sender-locally computable: every
rank evaluates it from replicated U at zero exchange cost. The union's real
price is (a) the union-build metadata and (b) the phase-1 intra-node pull.
A sender-local scheme that avoids the pull cannot balance: layer0 bytes
originate at each token's home rank, so per-rank wire bytes are fixed by
routing unless bytes move intra-node first. **Boundary-snapping is refuted by
arithmetic**: pull volume per round == the L1 distance between the segment
boundaries and the equal cut == exactly the imbalance the partition removes
from the wire; snapping a cut by δ saves δ/12 (NVLink) and costs up to δ
(wire) on the critical path — net-negative whenever the wire is critical.
The equal per-round cut is essentially optimal at BW_nvlink ≫ BW_wire; the
code already takes the free part (own_only / single-source fast paths).
Genuine generalization recorded for other fabrics: `chunk_bound` may be any
pure function of replicated U (e.g. NIC-capability-weighted cuts).

## D. Shipped result (canonicalization = user decision)

**rswire2 + CTA 10/8/6** (both pure env deltas on one binary, correctness
5/5+5/5 with per-iteration random payload, wire rule intact — every
inter-node put stays blocking putmem_signal):

| b | K2 best vs control | qwen best vs control |
|---|---|---|
| 2 | 6.26 vs 6.52–7.11 | 4.53 vs 4.72–5.00 |
| 8 | 9.95 vs 10.04–10.38 | 8.62 vs 8.89–8.95 |
| 32 | 27.82 vs 29.04–29.41 | 26.32 vs 27.02–27.05 |
| 64 | **50.73** vs 53.22–53.95 | **49.99** vs 51.49–52.04 |

(ranges = the 3 control repetitions across capsules.) −2..−6% total,
rswire2's l1 share sign-stable 16/16 (fwd+rev × models × budgets).
NEVER-MIX: candidate-binary capsules (post-80e500c) vs baseline capsules
(62383055) — compare arms only within one capsule, as always.

## E. Scalability notes (32n/128r, untestable now)

- Tier B window-count invariant ((NN−1)·L ≡ remote sources) holds to NN=8
  (64-bit ballot masks); kMaxBuckets=129 clears NN=8/L=4 windows.
- rswire2 generalizes: at NN=8 the wire ladder has 7·n_split independent
  cells — parity-split helps MORE; consider round-robin over >2 streams
  (guard: CUDA_DEVICE_MAX_CONNECTIONS ≥ streams+6).
- CTA heuristic: pack/prered work scales with rows·H; propose
  PACK ≈ clamp(6 + 2·(b/16), 6, 12) style budget-conditional sizing, or
  simply canonicalize 10/8/6 (validated tie-at-low-b here).
- The host metadata collapse (~50–150 µs at 4n) scales O(W·E + W²·NN):
  at 32n/128r it becomes ~0.5–1 ms — the C2 split (stage-1 needs only
  seg_off; two-slice pinned arena H2D) is mapped and becomes worth doing
  at 16n+. Same for plan_ms (2 ms at qwen b64 already).
- C7′ (wire sub-chunking, k_s=2–4: sub-signal per chunk slice, gateway
  forwards sub-windows, consumer lanes = sub-windows) is the designed
  structural attack on wire-serialization-into-tile-release; ~1 week,
  needs signal-space widening + gating W→W′ + CHECK extension. The only
  candidate that attacks the dominant term; do it before 32n bring-up.

## F. LocCap → dispatch boundary (for the future port)

The op's ingestion is already clean: per-call device tensors
(splits_gpu/scatter_index) + 2 CPU matrices (splits_per_source,
a2av_unique_counts) + in-op `derive_routed_meta` for the rule-5 path (one
pinned D2H + eventSync, the honest in-window sync). Nothing in this
session's changes adds host round-trips; the loccap tail's vce output maps
onto the same derive contract (`python_meta_from_vce` precedent). When the
port lands, keep: no bincount/nonzero on the hot path, the always-signal
invariant, blocking inter-node put_signal, and the never-cache rule.

## G. Closed / deferred ledger

CLOSED this session: pull2s (noise; ablation knob kept), fanout re-test
(mixed again), single-stream pull/put interleave (pessimization — blocking
puts would park pulls), conv-ladder reorder (no HOL exists), boundary-
snapped cuts (12× arithmetic), sub-windows without wire sub-signals (no
serialized work removed), C2-into-plan-bracket (accounting, not removal).
DEFERRED: C2 split (16n+), C7′ wire sub-chunking (designed), K2×FAST hang
(first-light), l0-lane dispatch_only first-barrier drop (l0-only lane, not
l01; needs W=32 probe), PACK_OVERLAP×LB_UNION unblock (e2e-mode-only win,
2–4 days), epoch-close barrier-chain lightening (3 global barriers per l01
iter; needs epoch-safety design — credit-based fence).
