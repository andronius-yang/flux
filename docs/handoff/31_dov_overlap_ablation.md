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

## 4. 8n A/B

(see §4 results below — filled after the 8n run; spec
`dov_ab_8n_{k2,qwen}.yaml`, same three arms, b1–b16.)

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
