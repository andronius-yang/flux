# Handoff 26 — l1 combine weight traffic: the low-budget K2 floor (2026-08-29)

Context: closing the paper's last experimentation question — why COMET
(`l01_allgather_dense`) beats OURS at 4n K2 low budgets while OURS wins
Qwen everywhere (user directive: fix it fairly, and do NOT adopt flux's
column split — n_split column pipelining stays a flux-unique mechanism).

## 1. Diagnosis (measured; step-0 capsules)

Step-0 nsys attribution (capsules `20260829-081712` K2 / `20260829-081959`
Qwen, 4n b1+b8, ours canon `ours_l01_s1_pv2_r2` + COMET, instrumented —
never latency): the K2 low-budget gap decomposes into exactly two items.

**(a) The l1 grouped GEMM re-reads expert weights once per msplit wave.**
The msplit destination-node row-split builds `n_waves x E` full-N
sub-problems (`make_workspace_kernel`); every wave's sub-problems stream
every expert's whole w2 panel from HBM again. At 4n that is 4 weight
passes vs COMET's 1 (its column split leaves weight traffic invariant —
each weight column block is read once regardless of n_split). The
arithmetic closes to ~5%:

| l1 GEMM span (nsys, 4n b1) | measured | model |
|---|---|---|
| ours K2 (4 passes x 764 MB w2/rank + flops) | 1.95 ms | ~2.0 |
| COMET K2 (1 pass + flops) | 0.69 ms | ~0.6 |
| ours Qwen (4 x 126 MB) | 0.43 ms | ~0.4 |
| ours K2 b8 (4 passes + 8x flops) | 2.84 ms | ~2.9 |

The model-dependence of the combine floor is exactly weights-per-rank
(K2 764 MB vs Qwen 126 MB), not G per se. The wire (3 blocking puts,
~0.15 ms each) and the bucket receiver (0.07 ms) hide under the GEMM at
b1 — slot-coalescing/receiver theories are DEAD.

**Hardware A/B (clean isolated, one binary, capsules `20260829-084113` K2
/ `20260829-084646` Qwen, 4n):** arms canon / `wn3` (WAVE_NODES=3, 2-3
passes) / `msp0` (msplit+fused-pack off = 1 pass) / `msp0_nb` (+bucket
off):

| total_ms | canon | wn3 | msp0 |
|---|---|---|---|
| K2 b1 | 5.92 | 5.46 | **4.74** (l1 2.84 -> 1.60) |
| K2 b8 | 9.76 | 9.68 | **9.57** |
| Qwen b1 | 3.76 | 3.71 | **3.66** |
| Qwen b8 | **7.48** | 7.90 | 8.00 |

Collapse wins everywhere measured EXCEPT wire-bound cells (Qwen b8) —
msplit's early wire release is worth keeping exactly where wire bytes are
large. Reread/wire byte ratios at the measured points: K2 b1 ~370, K2 b8
~62, Qwen b1 ~60 (all collapse-win/neutral), K2 b16 ~31 (unmeasured),
Qwen b8 ~7.6 (collapse loses) → threshold 48 separates them.

**(b) The plan lane is ~70% dispatch/combine metadata derivation, not
routing.** Host NVTX medians (b1, model-independent): derive_combine_meta
0.64 (a ~15-op torch dispatcher chain + two PAGEABLE H2Ds ≈ hidden syncs;
only ~0.03 ms GPU), derive_routed_meta 0.26, scale build 0.23, vce tail
0.21, route kernel 0.11, phys+probs allgather 0.11. COMET pays 0.26 total.

## 2. Implemented (this session, 8/29 rebuild; rule-4 boundary)

1. **Byte-adaptive wave collapse** (`FLUX_A2AV_RS_WAVE_ADAPT`, default 0 =
   bit-identical legacy; tag `FLUX_A2AV_RS_WAVE_ADAPT_TAG`). Per-iteration
   host rule in `forward_gather_rs`, from the same cnt table the wave
   build reads: collapse to the legacy single-gate problem structure (one
   weight pass, `set_msplit_waves(..., 0)` keeps pack/relay/receiver
   coherent, fused-pack off for the iteration) when
   `(n_waves-1) * E * N * K * elt > ratio * remote_wire_bytes`.
   Arm `ours_l01_s1_pv2_r2_wa` (ratio 48) + gate twin. This trades
   GEMM<->wire overlap away only when the overlap cannot pay for the
   re-reads.
2. **Kernel-side combine index build** (`FLUX_A2AV_RS_COMBINE_IDX_KERNEL`,
   default ON, tag `..._TAG`; torch chain kept as the
   FLUX_A2AV_RS_CHECK_IDENTITY reference): `a2av_combine_plan` — two
   direct-write kernels (upper-bound over host prefix tables) + pinned
   async H2D replace the dispatcher chain in
   `build_a2av_combine_indices`. Targets derive_combine_meta 0.64 →
   ~0.1-0.2 ms.
3. Plan-lane NVTX sub-ranges (`FLUX_OURS_NVTX`, ours.py, default off).

Diagnostic arms kept for reproduction: `_wn3`, `_msp0`, `_msp0_nb`
(never headline).

## 3. v2 DESIGN — overlap without re-reads (not yet implemented)

The collapse forfeits the early wire release. The structural fix keeps
ONE weight pass AND early release, staying row/wave-shaped (no column
split):

- **Problem order: expert-chunk-outer, wave-inner.** Order the msplit
  sub-problem list as chunks of experts sized to L2 (A100 40 MB; K2
  w2 panel 14.7 MB → 2 experts/chunk): for each chunk C, emit problems
  (C, wave 0..NN-1) adjacently. The 2nd..Nth wave of a chunk hits L2, so
  HBM weight traffic collapses to ~1 pass while the (wave, expert)
  sub-problem semantics — and the receiver contract — are unchanged.
- **Release granularity: per (chunk, wave), not per wave.** Today the
  pack/wire gate waits for a whole wave (all experts), which under any
  weight-efficient order completes only near GEMM end. Add cascade flags
  per (chunk, wave) — the epilogue already counts per-problem tiles; the
  flag index becomes problem-linear instead of wave-linear. The pack
  kernel flushes each dest node's rows of newly completed chunks
  (sub-panel ranges are contiguous if the send panel is ordered
  (dest, expert, token) — which it already is: panel order = A-order
  restricted per dest). Wire puts become per (dest, chunk-watermark,
  lane) with **signal-ADD row counts**; the receiver already supports
  additive multi-contributor signals (handoff 21; handoff 24 principle
  4), gating on plan-known totals. First flush then lands after ~1 chunk
  (~1/13 of the GEMM at K2) instead of after a full weight pass —
  overlap strictly better than today's wave release.
- **Put-count control:** flush at chunk watermarks only (13 chunks x 3
  dests at K2 4n vs today's 3) and floor the flush size (merge
  watermarks below ~1 MB) — same lane discipline, rail-aligned, blocking
  puts (wire rule 6a untouched).
- Expected: K2 b1 l1 GEMM ~1.95 → ~0.7 with wire still overlapped at
  b16-b64 (removes the ~1.5 ms constant from EVERY budget, not just
  small ones — at b64 that is the 4-pass tax msplit currently pays).
- **Put-fragmentation constant (user concern 8/29, quantified from
  existing data): ~120 us per blocking put_signal through the CXI proxy,
  serial per rank.** Two independent fits agree: (a) the handoff-16 8n K2
  n_split ladder at b1 — ns1 4.75 / ns2 5.65 / ns7 9.75 / ns14 15.46 e2e,
  i.e. +0.90/+4.10/+5.71 ms for +7/+35/+49 per-(split,dest) puts =
  129/117/116 us/put; (b) the step-0 4n timeline put spans: 131-161 us at
  2.1 MB and 430-490 us at 17 MB fit constant ~120 us + ~50 GB/s
  streaming. CONSEQUENCE: v2 must treat pieces-per-dest P as a dial with
  a byte floor, NEVER a per-chunk flush. P=1 is the small-b fixed point
  (v2 degenerates to the shipped collapse: same put count as today, one
  weight pass, no fragmentation); P=2..4 engages only where the wire is
  multi-ms (b16+), where (NN-1)x(P-1)x0.12 ms of extra serial proxy
  constants stays <5% of wire and the (NN-1)-pass weight saving (4n K2
  ~1.5 ms, 16n ~2.3 ms with E=8/rank) nets positive. At 16n P=2 costs
  15x0.12=1.8 ms of constants vs 2.3 saved — marginal; P must shrink (or
  collapse) as NN grows. Pre-implementation gate: a piece-ladder cell at
  fixed bytes (b16/b32, P=1..8) on the current binary to re-fit a+b*P at
  4n/8n before any kernel work.
- Other risks: L2 hit rate under concurrent pack/prereduce CTAs (measure
  with the step-0 recipe); tile-scheduler adjacency across SMs;
  flag-region sizing (13x4 flags fit the 128-padded barrier region at 4n
  but not 128 chunks x waves at 16n — cap chunk count or widen the
  region).

## 3b. USER RULING (2026-08-29): the wave dial

The combine GEMM/wire schedule is a three-position dial, and the
destination-major implementation is KEPT as a first-class alternative,
never removed:

| setting | behavior | when |
|---|---|---|
| `FLUX_A2AV_RS_WAVE_ADAPT=0` (binary default) | always destination-major waves (today's canon; bit-identical legacy) | wire-bound cells / pre-8/29 reproduction |
| `FLUX_A2AV_RS_WAVE_ADAPT=R` (arm `_wa`, R=48) | per-iteration byte rule: collapse when reread > R x wire | the record candidate |
| `FLUX_A2AV_RS_MSPLIT=0` | always collapsed (single-gate, one pass) | ablation / floor probes |

Accepted premise (measured, not assumed): **at b1-class budgets no
overlap is optimal** — the remote wire (~0.45 ms at 4n) is smaller than
msplit's weight-pass rent (~1.5 ms) AND smaller than the put-constant
cost any finer-grained overlap scheme would pay (~0.36 ms at P=2); the
serial minimum-put schedule is within ~0.1 ms of any scheme's bound.
v2 (expert-major + progressive flush) remains a b16+ experiment behind
the piece-ladder go/no-go in §3, with P=1 degenerating to the collapse.

## 4. Remaining plan-lane queue

- Fuse/de-serialize derive_routed_meta + combine meta into one derive
  with a single honest sync (0.26 + residual combine host cost).
- The validated-but-off planfast knobs (PLAN_GRAPH + SCALE_GRAPH,
  lossless; ~0.1-0.3 ms) — canonicalization = user decision.
- Counts-only exchange (handoff 20 deferred) stays the endgame for the
  route+exchange 0.5 ms.

## 5. Where this leaves the b1 fight (4n K2, vs COMET 4.04 total)

canon 5.92 → +collapse 4.74 (measured) → +combine-idx kernel ~4.3 (est)
→ +routed-meta fusion / planfast ~4.0-4.1 (est) → v2 keeps the b16+ wins
intact and removes the same tax at every budget. Qwen untouched at b8+
by construction (ratio gate).
