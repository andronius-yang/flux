# v2 chunked combine — design + campaign notes (worktree v2-combine, 2026-08-29)

Goal (handoff 26 §3): l1 combine GEMM<->wire overlap WITHOUT the msplit
weight re-read tax. User pre-registration: at b1/b2, splitting the
inter-node puts nets nothing (accepted premise, handoff 26 §3b) — every
mechanism here must degenerate to the shipped collapse at small budgets.

## M1 — chunk-ordered problems (this commit)

Knob `FLUX_A2AV_RS_CHUNK_E` (default 0 = OFF, bit-identical legacy order;
-1 = auto floor(L2_eff / (N*K*elt)) clamp [1,E]; N = explicit width).
`FLUX_A2AV_RS_CHUNK_L2_MIB` (default 30) sets L2_eff for auto.

- Problem list order becomes chunk-outer, wave-inner:
  (c0: w0 e0..e1, w1 e0..e1, ..) (c1: ..). Same n_waves x E problems,
  same wave_M/wave_off math, same per-wave cascade flags/targets, same
  pack/conv/prereduce/wire/receiver. Only ORDER moves.
- Two device maps ride the existing pinned arena H2D:
  prob_eid[i] (workspace kernel: eid no longer i % E),
  prob_group_map[i] (cascade: group_idx no longer problem_idx / stride).
- Effect: chunk c's expert panels stay in L2 across its waves ->
  ~1 HBM weight pass. Wave flags fire near GEMM end (all waves ~
  simultaneously) => comm timing ~ collapse; weight cost ~ collapse;
  M1 alone is infra + the L2-hypothesis hardware probe, NOT a win arm.
- Expected measurables (K2 4n, vs 20260829 step-0 numbers):
  b16/b64 l1 GEMM span with waves armed ~ msp0 floor (reread gone);
  b1 partial only (padding tiles remain: 4x tiles, L2-fed).
- Risks: L2 hit under concurrent pack/prered/reduce CTAs (measure);
  in-flight tile window spanning >2 chunks at tiny problems (b1 only).

## piece-ladder pre-fit (before M2 kernel work; handoff 26 §3 gate)

On the CURRENT binary: msp0 (collapse) + n_split in {1,2,4,8} at K2
b16/b32, 4n + 8n, isolated. Wire puts = n_split*(NN-1) blocking
putmem_signal; fit e2e = a + b*puts to re-quote the ~120us/put CXI
constant at both scales. Note ns>1 disables bucket (ns1-only) — deltas
are within-ladder only, never vs canon.

## M2 — progressive piece release (after M1 gates + ladder fit)

Dedup-preserving: wire rows (one per (tn, dest token), compress C')
get a two-side-derivable READY-PIECE key = max over the row's conv
contributors of chunk_of(contributor expert on its owner rank); chunk
schedules are a pure function of splits (deterministic), so sender
wire_csr AND receiver red_row can both order rows (tn, ready_piece,
token) — replacing today's (tn, token) contract. No wire-byte
inflation (each row still sent once, at its last contributor's chunk).
- conv ladder: per (tn, dl) slices per piece — send-panel per-dest rows
  are (expert, token)-ordered, so piece = contiguous slice; conv
  signals become per (lane, piece).
- prereduce: consumes pieces in order per tn (spin per (lane,piece)),
  emits wire rows of ready_piece <= p, flips per (tn, piece) wire flags.
- wire ladder: <=P blocking puts per tn (piece watermarks merged under
  a byte floor; P=1 == today's single put == the collapse fixed point).
  Signals: per-put SIGNAL_ADD of row counts on a rolling uint64
  cumulative expectation (both sides track cum rows per lane; no
  resets, epoch-safe), receiver lane completes at cum target — bucket
  receiver unchanged except the wait value; dyn receiver can later
  consume pieces as sub-lanes (M3).
- Auto-engage rule mirroring wave-adapt: pieces only when
  (n_waves-1) weight-pass bytes > ratio * (NN-1)*(P-1) put-constant
  cost AND wire bytes > floor; else P=1. At 8n puts double (7 vs 3
  per piece step) — P must shrink with NN (user flag).

## Campaign (user ask 8/29): l1 + total_ms vs baselines,
b {1,2,4,16,64}, 4n + 8n, K2 + Qwen, one binary per capsule:
arms = canon ours_l01_s1_pv2_r2_wa_pf(ov2 where green) / _wa twin /
msp0 twin / chunk (M1) / chunk+pieces (M2, P ladder 2..4).

## 4n RESULTS (2026-08-29 night; capsules 20260830-013149 K2,
## -014319 Qwen, -015422 nsys; gates -0128xx green w/ reference checks)

l1_ms (isolated, mean of per-iter max-rank), K2 4n:
  b:      1     2     4     16    64
  canon   1.61  1.97  2.77  7.15  24.26
  wa0     2.84  3.14  3.53  7.00  24.66
  msp0    1.62  1.95  2.80  7.94  28.92
  chunk   3.01  3.43  4.19  8.94  29.56
Qwen 4n: chunk-msp0 = +0.21/+0.22/+0.52 at b1/16/64; wa0-msp0 =
-1.09/-4.23 at b16/64 (overlap value).

nsys l1 GEMM spans (K2 b16): msp0 2.32 / chunk 3.44 / wa0 3.87.
- L2 reuse WORKS: chunk beats wa0's GEMM by 0.43 (the true marginal
  HBM-reread cost — rereads partially hide under MMA, so the 1.4 ms
  bandwidth model overstates it).
- The SPLIT structure costs +1.12 over the collapse (padding ~0.5 +
  L2 re-traversal ~0.46) — bigger than the reread it was built to fix.
- Overlap value (wa0 vs msp0): 0.94 (b16) -> 4.26 (b64) K2; 4.2 Qwen
  b64. M1 (no release) loses to wa0 wherever waves win, as designed.
- b1-b4: collapse owns them; the b1 wave tax is mostly PADDING FLOPS
  (~104 padded tiles ~1.9 ms), refining handoff 26's pass model.

DESIGN CONSEQUENCE for M2: drop the (chunk, wave) sub-problem
structure; go NO-SPLIT per-tile-flag (one problem per expert,
dest-sorted rows — already the layout; epilogue per-tile counters
fire per (expert, dest-range); pack/conv/prereduce/wire consume
(dest, chunk-watermark) pieces as in §M2 above). Ceiling at K2 b16:
~6.1 vs canon 7.15; auto-collapse at b1-b4 (1 tile/expert). The
piece-ladder put-constant fit still gates the wire fragmentation.
