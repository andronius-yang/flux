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
