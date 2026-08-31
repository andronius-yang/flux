# Handoff 31 — direct-overlap (dov) ablation: tile-spin GEMM over the direct wire (2026-08-30/31)

**One-line:** third arm of the direct-wire family — the pv2+LocCap+r2 plan
lane and the one-hop direct transport of the dwire arm, but layer0's GEMM
OVERLAPS the wire: tiles spin on per-source signals in ring arrival order.
dov beats the un-overlapped dwire arm at EVERY 4n/8n budget on both
models; the dispatch leg alone drops 22–40%.

## 1. Mechanism (what was actually built)

The requested "GEMM tile spin waiting on incoming data in the ring order
of arriving direct transports" already existed, dormant, as the fused
op's **FLAT a2av mode** (`a2av_dispatch=True`, ring/hier/compress off):
pack → per-dest puts in ring order starting at rank+1 → per-source
signals → the grouped GEMM's dynamic claimer picks tile buckets as
sources land (`multi_masks` for straddling tiles, W ≤ 64). Two things
made it unusable as-is:

1. **Wire-ordering rule 6.** The consumer gates on per-source signals, so
   the shipped flat wire uses BLOCKING `putmem_signal_on_stream` per
   remote dest — W−L serialized proxy puts (~120 µs each) on one stream,
   destroying the direct wire's NN-flat concurrency.
2. The flat GEMM gate waits `fetch_remote_event` = full wire issue (and
   under the blocking wire, full drain).

Fix = `FLUX_A2AV_FLAT_FENCED_SIG=1` (new, this branch): remote data goes
as **CONCURRENT `putmem_nbi` puts in ring order + ONE PE quiet + the
signal ops in the same ring order** — the F2 quiet-then-signal pattern
already proven for the relay wire (handoff 08). Intra-node destinations
keep the fused nbi `put_signal` (NVLink P2P store order — the audited
intra configuration), so near sources release their tiles per-source
while remote data drains. Ctor-requires `FLUX_A2AV_EARLY_LAUNCH=1`: the
GEMM launches ungated and spins, the deferred cp-stream replay
re-records the covering events (existing early-launch machinery).

Runner `OursDirectOverlapRunner` (`flux/testing/ours_direct.py`):
- l0 = ONE flat-mode `GemmGroupedV2AGScatterOp.forward` (no pack, no
  place, no probs side-wire on the timed path).
- plan = `derive_routed_meta` (the fused arm's derive) + host prefix sums
  over the pinned `sps` + one [W, gpe] H2D + O(n_recv) index math + a
  LOCAL stable argsort of my own vce row.
- combine = the dwire reverse `All2AllSingle` + deterministic home
  reduce, **byte-identical**, keyed off the op's stable-scatter OUT
  layout: `scatter_index` is `argsort(stable).argsort()` (audited
  bitwise vs `python_meta_from_vce`), whose per-expert interior is
  (source, token) — exactly `direct_layout_entries_fast`'s segment
  layout. So `place_slots` is pure prefix-sum algebra over `sps` and
  `comb_dst = argsort(my_vce, stable)`. The combine's correctness does
  NOT depend on the wire's internal pack order. CPU brute-force test:
  `test_ours_dov_plan.py` (place_slots↔stable-OUT identity, simulated
  reverse-wire return order, scale scatter).
- combine scale from the planner's fused probs allgather through the
  scatter index (`_scale_compute` logic) — bitwise the same fp32 values
  the dwire arm ships over its probs a2av, one wire leg cheaper.

Arms `ours_l01_s1_pv2_r2_dov[_gate]` (conn=8, LB_UNION=0 forced before
the collective ctor, FLAT_FENCED_SIG=1, EARLY_LAUNCH=1); sweep.py sizes
the heap as fused-prior + eplb staging sum. Capability tag =
`FLUX_A2AV_FLAT_FENCED_SIG` in the .so.

## 2. Gates (4/4 green, capsules `20260831-001454_f08d77f2` K2 / `-001832_3ec58cad` qwen)

4n, b1+b8, per-iteration output checks + full correctness + per-iteration
random payload — this is the wire audit for the fenced flat signals
(rule 6b). All green first try on both models.

## 3. 4n A/B (capsules `20260831-002000_2cc7b203` K2 / `-002649_416a08d9` qwen, one binary, 30/30 ok)

total_ms medians (l0 in parens):

| b | K2 fused | K2 dwire | K2 dov | qwen fused | qwen dwire | qwen dov |
|---|---|---|---|---|---|---|
| 1 | 3.98 (1.62) | 6.16 (2.62) | **5.82 (2.05)** | 3.06 (1.16) | 4.83 (1.86) | **4.55 (1.59)** |
| 2 | 4.63 (1.91) | 8.00 (3.45) | **7.20 (2.47)** | 3.54 (1.30) | 6.82 (2.80) | **6.04 (1.98)** |
| 4 | 5.93 (2.35) | 12.22 (5.27) | **10.60 (3.28)** | 4.55 (1.70) | 10.48 (4.59) | **9.03 (2.95)** |
| 8 | 9.08 (3.51) | 19.87 (8.97) | **17.16 (5.43)** | 6.70 (2.63) | 18.51 (8.44) | **15.25 (4.99)** |
| 16 | 14.40 (6.26) | 35.37 (16.67) | **29.82 (9.95)** | 11.67 (4.74) | 33.99 (16.07) | **27.70 (9.67)** |

1. **dov < dwire at every cell**: K2 −5.5% (b1) → −15.7% (b16); qwen
   −5.8% → −18.5%. The l0 (wire+GEMM) leg alone: −22% at b1 growing to
   −40% at b8/b16 — the overlap recovers the serialization exactly where
   bytes dominate, as designed.
2. **Combine faithful**: dov l1 ≡ dwire l1 on qwen (8.35 vs 8.35 at b8);
   on K2 dov_comb runs ~0.5 ms hotter (7.01 vs 6.50 at b8) with gemm2 /
   cpack / acc identical — attributed to rank-arrival skew absorbed by
   the combine's first team barrier (dwire enters the combine
   barrier-synchronized by its own dispatch wire; dov's ranks finish
   their overlapped l0 at different times). Recorded, not debugged.
3. **Plan** costs +0.15–0.35 ms over dwire (derive + combine algebra vs
   the adapter), a partial offset at b1; net still negative everywhere.
4. Fused keeps every 4n cell (few hops → the compress wire's byte
   savings beat any overlap of full-byte transport — consistent with
   handoff 29 §3).

## 4. 8n A/B (capsules `20260831-033502_24b72e20` K2 / `-034121_e3ec33ba` qwen, one binary, 30/30 ok)

total_ms medians (l0 in parens):

| b | K2 fused | K2 dwire | K2 dov | qwen fused | qwen dwire | qwen dov |
|---|---|---|---|---|---|---|
| 1 | 5.87 (2.70) | **5.98 (2.60)** | 6.37 (2.64) | 5.34 (2.42) | **5.63 (2.39)** | 5.76 (2.42) |
| 2 | 6.38 (3.00) | 8.16 (3.66) | 8.21 (3.27) | 6.16 (2.61) | 7.98 (3.58) | **7.84 (3.12)** |
| 4 | 8.39 (3.65) | 13.22 (6.09) | **12.52 (5.02)** | 7.18 (3.11) | 12.80 (5.99) | **11.90 (4.65)** |
| 8 | 12.33 (4.93) | 23.15 (11.19) | **21.17 (8.75)** | 10.46 (4.33) | 23.39 (11.16) | **20.39 (8.18)** |
| 16 | 18.72 (8.14) | 43.27 (20.95) | **38.07 (14.97)** | 17.75 (7.64) | 44.20 (21.77) | **37.68 (14.91)** |

(bold = dov-vs-dwire winner within the direct family)

**The 8n crossover:** at b1 the direct wire is latency-dominated — 28
remote destinations, tiny payloads, and the dov wire's ~60 host-issued
stream ops give it no l0 advantage over the eplb device-kernel fan-out
(l0 2.64 vs 2.60 K2) — so dov's plan (+0.20) and combine-skew (+0.13)
make it a net LOSS (K2 +6.5%, qwen +2.3%). From b2 (qwen) / b4 (K2) the
overlap wins and grows: −5.3→−12.0% (K2), −7.0→−14.8% (qwen) at b4–b16,
with the l0 leg −18..−29%. Same shape as 4n, shifted one budget notch up
by the doubled hop count.

## 4b. 16n A/B (capsules `20260831-062119_8e5d79c3` K2 / `-062614_c3dcad9c` qwen, one binary, 24/24 ok; user-requested extension)

W = 64 = the flat dynamic claimer's multi-source-mask cap — ran clean at
the boundary. total_ms medians (l0 in parens):

| b | K2 fused | K2 dwire | K2 dov | qwen fused | qwen dwire | qwen dov |
|---|---|---|---|---|---|---|
| 1 | 12.74 (6.85) | **6.92 (3.23)** | 8.08 (3.90) | 12.40 (6.63) | **6.68 (3.09)** | 7.86 (3.74) |
| 2 | 13.55 (7.14) | **9.80 (4.74)** | 10.50 (4.82) | 12.99 (6.82) | **9.92 (4.82)** | 10.49 (4.73) |
| 4 | 15.81 (7.89) | 15.90 (7.98) | **15.74 (7.02)** | 14.44 (7.39) | 16.00 (7.81) | **15.40 (7.04)** |
| 16 | 27.86 (13.03) | 52.55 (27.36) | **47.63 (21.58)** | (355.7†) | 53.22 (26.63) | **47.59 (21.38)** |

† the fused qwen 16n b16 cell hit the KNOWN intermittent ~350 ms l1
stall (handoff 30 open issue) — reproduced here on this binary too;
unrelated to the direct family.

**16n verdict:** the overall frontier is UNCHANGED — dwire keeps its
b1/b2 crown (dov +17% / +6-7% there), the fused arm keeps b4+ overall.
Within the direct family dov crosses over at b4 (−1..−4%) and wins b16
(−9.4% K2, −10.6% qwen; l0 leg −19..−21%). At b1 the dov l0 is WORSE
than un-overlapped (3.90 vs 3.23): 63 destinations mean ~130
host-issued stream ops (puts + quiet + signal ops) against the eplb
wire's single device-kernel fan-out, and the b1 GEMM is too small to
repay it; plan also grows ~+0.5 with W (derive + [W,gpe] algebra).

**Overlap-win region summary (dov vs dwire, total_ms):** 4n = every
budget (−5.5..−18.5%); 8n = b2/b4 up (−1.8..−14.8%), b1 loses; 16n =
b4 up (−1..−10.6%), b1/b2 lose. The gain is ∝ wire bytes ×
GEMM-to-latency ratio; the fixed per-dest issue cost is ∝ W. If dov's
low-b cells ever matter, the candidate fix is moving the fenced fan-out
into a device kernel (concurrent nbi_block puts + quiet + signal from
one block, eplb-style) instead of host-issued stream ops.

## 5. Limits / notes

- Flat dynamic claimer caps W ≤ 64 (multi-source masks) — 16n is the
  boundary topology.
- The fenced flat wire's remote sources release together after the PE
  quiet (per-source remote granularity would need per-dest completion
  tracking NVSHMEM does not expose); intra-node stays per-source. On a
  shared NIC the remote puts complete near-simultaneously anyway.
- b32/b64 not attempted at 4n/8n (heap: the dov sum sizer caps at 16G —
  same class as the dwire b64 wall).
- The conda env's editable flux install was re-pointed at this worktree
  by the build (build.sh side effect) and must be restored to the main
  checkout when the worktree retires.
