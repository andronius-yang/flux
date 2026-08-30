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

## M2 IMPLEMENTATION SPEC (settled 8/30 ~02:40, after the 4n verdict)

Structure: NO-SPLIT (one problem per expert; kills the +0.5-1.9ms padding)
+ per-chunk flags via M1's prob_group_map (group = expert-chunk; the
existing per-problem cascade IS the producer — zero new kernel-side GEMM
code) + piece-granular comm chain. Knob FLUX_A2AV_RS_PIECES (0=off,
N=max pieces/dest, -1=auto); requires msplit+fused_pack+bucket+ns1+NG1.

- M2a (host-only): waves=[one wave, all nodes] => E full-row problems;
  prob_group_map=chunk_of(e); gemm n_split arg = n_chunks;
  non_empty_per_group[c] = nonempty experts of chunk c; ws preset for
  empty chunks (new ws_args.n_flags decoupling problem_count from flag
  count). Comm: pack waits ALL chunk flags (collapse timing) — gate
  arm equals msp0 within noise; scaffolding only.
- Piece merge (host): per-chunk send bytes from cnt -> pieces (<=P,
  byte floor FLUX_A2AV_RS_PIECE_FLOOR_MIB), piece_of_chunk[].
- Pack relay per (node, piece): group_flags -> [NN x P]; wait the
  piece's chunk flags in order, flip per (tn, p).
- Conv ladder per (tn, dl, piece): send-panel per-dest rows are
  (expert, token) asc => piece slices are contiguous prefixes; offsets
  from per-(d, piece) cnt prefixes. conv_sig -> [L x NN x P], epoch
  SET per slot (intra-node NVLink, stream-ordered, wire rule safe).
- Prereduce per (tn, piece): wait L conv sigs of (tn, p); wire rows
  sorted (seg, ready_piece, token) => contiguous per-piece ranges via
  wire_piece_start[NN-1][P]; flip wire_flags[tn*P + p].
- Wire ladder per (tn, piece): blocking put of the piece range into
  the dest C' lane at the piece base; SIGNALS = per (lane, piece)
  slots [W x P], SET run_id (NO signal-ADD needed — distinct slots
  keep the epoch trust contract; zero-row pieces bare-signal as today).
- Receiver: bucket lane wait becomes P sequential CUStreamWaitValue64
  (piece asc). Piece-progressive folding (sub-lane buckets) DEFERRED.
- Plan (two-sided contract change): ready_piece(seg token) = max over
  the token's my-node contributors of chunk_of(e on its owner rank)
  (same Ec on all ranks — deterministic from splits; BOTH sides hold
  e_of_copy). Sender: compress_plan_token_kernel adds atomicMax
  wire_piece[(seg,t)]; phase A scan becomes per-(seg,piece) counting
  order. Dest: phase C rem_pos becomes per (node col, piece) bases +
  token-asc positions; red kernel emits accordingly. The .cc sort
  reference gets wkey = (seg, ready_piece, token) for CHECK_IDENTITY;
  python _dev twin + sim: TODO note (update before running the sim).
- Sizing: flag/sig regions ctor-sized at max(n_split, 8) pieces.
- Auto rule: engage pieces only when waves would run (reuse the
  wave-adapt byte rule) AND wire_bytes/(NN-1) >= floor; P shrinks
  with NN (put constants), P=1 == collapse fixed point.

## M2 4n K2 RESULTS (8/30, capsule 20260830-023028, gates green both
## models incl. per-iter reference checks — first-execution green)

l1_ms K2 4n:  b1: canon 1.61 / msp0 1.67 / P2 1.75 / P4 2.15 / wa0 2.82
             b16: wa0 7.06 / canon 7.19 / P2 7.35 / P4 7.44 / msp0 8.15
             b64: wa0 24.53 / canon 25.05 / P4 25.40 / P2 26.45 / msp0 29.87
- Pieces = best non-reread schedule: beats collapse by 0.8 (b16) /
  4.5 (b64) — 70-84% of the wave overlap recovered at ~1 weight pass,
  no padding. Piece-machinery rent at b1: +0.08 (P2) / +0.48 (P4).
- But waves still win b16/b64 by 0.3-0.9: the LAST-CONTRIBUTOR SKEW
  (P(row ready by chunk c) ~ (c/n_chunks)^k_node, k~4) leaves the wire
  tail exposed; that costs more than the reread+padding tax waves pay
  at 4n. Canon dial keeps the record at every budget.
- P direction flips with budget (P2 best at b16, P4 at b64) — put
  constants vs piece mass, as modeled.
- FOLLOW-UPS: skew-aware piece boundaries (cut chunk index at
  n_chunks*(p/P)^(1/4) to balance wire mass); 8n regime (tax/overlap
  ratio shifts); placement-aware chunking (cluster contributors).
