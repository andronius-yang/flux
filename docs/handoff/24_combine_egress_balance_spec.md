# Handoff 24 — Combine egress balancing: implementation spec (refines handoff-21 TODO 1)

Status: **SPEC ONLY, 2026-08-27 pm** (drafted during the Perlmutter outage;
user-directed — no code changes yet). Supersedes the one-paragraph TODO-1
sketch in handoff 21 with the design settled in the 8/27 debate. Authority
for the mechanism numbers: handoff 21 (nsys pair 20260825-183714) + the
8/27 NVLink-idle measurement (capsule 20260827-140515, s2_pv2 K2 4n b8:
CE busy 1.70 / 11.18 ms per iteration = **84.8% NVLink idle**, with the
two intra-node rounds visible as P2P bursts bracketing each wire put).

## 0. Problem (measured)

The NVSHMEM host proxy is a PER-RANK serial resource (~7.6 GB/s on CXI).
At 16n qwen b64 the combine's merged wire bytes (~462 MB/node — already
byte-minimal: one row per (token, source node), nothing to dedup) drain
through the node's four proxies as 272/121/53/15 MB — the node finishes
at the pace of ONE proxy (35.8 ms) while a balanced drain would take
~15 ms. Placement cannot fix it (wirebal proven empty; hot-expert duty
irreducible at whole-expert granularity). This is a TRANSPORT fix: same
bytes, balanced carriers.

## 1. Design principles (settled in the 8/27 debate)

1. **The wire stays rail-aligned** (same-local-rank pairing, one flow per
   (source node, dest rail), NN-1 blocking puts/rank at ns1). A
   direct-to-home variant from balanced egress proxies was considered and
   REJECTED: it multiplies dest-NIC incast from NN-1 to L*(NN-1)
   concurrent flows (15 -> 60 at 16n) and abandons the per-target lane
   discipline the l1 wire ladder is tuned around. Only per-rail BYTE
   COUNTS change; the flow topology is bit-identical to today.
2. **Two intra-node NVLink hops** carry the rebalancing, both sitting in
   the measured ~85% CE-idle windows:
   - source side: sub-panel slices moved to their assigned egress rail's
     staging (the balancing);
   - destination side: ingress rail forwards each home-rank range to its
     final buffer (the "one extra jump at the end").
3. **Sub-panel splitting at cumsum row boundaries is REQUIRED, not
   optional.** At ns1 the natural unit is one panel per dest node: NN-1
   panels over L rails cannot balance at small NN (3 items over 4 rails
   at 4n), and a single hot panel is itself the imbalance. Rows are
   unique and order-insensitive at the receiver, so any panel splits at
   arbitrary row offsets; the balancer becomes exact proportional
   assignment instead of bin packing.
4. **Count-based completion (signal-ADD) at every tier.** Receiver-side
   lanes ALREADY support additive signals with multiple contributors
   (handoff 21 notes this explicitly), so k split-pieces arriving from k
   rails need no new receiver contract: every put signal-ADDs its row
   count; consumers gate on `counter >= plan_total` (plan-known,
   replicated). Split-topology- and order-invariant; post-iteration
   `counter == total` is a free structural audit.
5. **Plan-side cost is sub-ms and the toggle is free.** Per-(source-node
   -> dest-node) panel byte counts are exact functions of tables the
   combine meta already builds (U / sps). Balancer: equalize per-rail
   egress totals SUMMED OVER DESTINATIONS (rail j serves every dest
   node), i.e. minimize max_j Sum_v piece_j(u->v), by exact proportional
   cumsum splits, keeping per-(rail, dest) pieces MB-class. Deterministic
   integer host math, same class as the pv2/swap decisions. Engage the
   whole path ONLY when the plan's per-rail spread exceeds a threshold —
   small-b cells keep today's schedule bit-for-bit.
6. **v1 has ZERO kernel changes** (the user's fan-in complexity concern):
   prereduce writes locally exactly as today; each contributing rank's
   STREAM then issues CE puts of its slice to the egress rail's staging
   followed by an on-stream signal-ADD of the moved row count; the
   egress wire put gates on the piece counter. "Writers signal" at
   stream granularity — per-piece pipelining preserved (egress puts
   piece j while j+1 stages), kernels untouched.
   **v2 (optional, only if v1's CE hop ever measures):** fold the
   redistribution into the prereduce/epilogue store path with a
   per-WAVE-uniform peer base pointer (all rows of one wave/piece target
   one peer -> no intra-CTA divergence; align split boundaries to CTA
   work boundaries). Bounded complexity; measure before building.

## 2. Ownership chain as pinned from the code (question (a), 2026-08-27)

Read from `src/moe_gather_rs/a2av_combine.cu` +
`ths_op/gemm_grouped_v2_gather_rs.cc`:

1. GEMM1 rows live on the rank hosting the serving expert instance;
   under fused pack (gen-8c) the epilogue scatters rows directly into
   the LOCAL dest-major send panel per (wave = dest node, expert)
   sub-problem, cascade-flagged per wave.
2. CONV ladder (intra-node, CE): each local peer (n, ls) puts its slice
   for dest rank (tn, lr) into the SAME-LR rank's conv panel and raises
   `conv_signals[ls][tn][sid]` (uint64 epoch, additive discipline).
3. PREREDUCE runs on the same-lr rank (`a2av_combine_prereduce_kernel`,
   a2av_combine.cu:258): thread 0 spins the L conv signals per (tn),
   blocks merge the CSR (`wire_ptr/wire_copy`) into the wire panel
   segment (token-ascending, `U[(tn, my_lr)][my_node]` rows), and a
   block-count handshake flips `wire_flags[tn][sid]`.
4. WIRE ladder: host-issued BLOCKING putmem_signal per (tn) — **"source
   rank (n, lr) owns all wire rows to rank (tn, lr) (same-lr
   end-to-end)"** (build_a2av_compress_indices doc, gather_rs.cc:~470).
   So wire duty is DEST-LR-KEYED, already rail-aligned.
5. Receiver: per-source-node lanes under the C' image; red CSR per local
   token; Slipstream v2 = arrival-order bucketed receiver (starts
   consuming early lanes — softens the forward hop's tail).

**Open attribution item (decisive, cheap, blocked on PSCRATCH):** the
272/121/53/15 spread must equal the per-rail quantity
`Sum_tn U[(tn, lr)][n] * row_bytes` if the staged spec governs the fused
path too. Recompute U from the 183714 cell's trace + placement and
compare per-rail sums to the measured proxy bytes. If EQUAL: the skew is
dest-home-lr demand structure (and handoff 21's "follows expert hosting"
phrasing is imprecise — hosting sets only the intra-node CONV traffic;
this also explains wirebal's emptiness at the mechanism level: hosting
permutations never touched wire duty at all). If NOT equal: the fused
wave path's wire enqueue deviates from the staged spec — trace the
ladder enqueue before implementing. THE DESIGN IS SKEW-SOURCE-AGNOSTIC
(it balances planned bytes whatever causes them); this check only
sharpens the narrative + the balancer's input table.

## 3. New machinery (v1), concretely

- Plan lane (+sub-ms, timed): per-rail egress totals; threshold check;
  piece table {(u-rail j) -> [(dest tn, row range, staging offset)]};
  per-piece expected counts; per-home-range forward table for the
  ingress side. All from existing replicated tables.
- Source: per-contributing-stream CE puts slice->egress staging +
  signal-ADD; egress wire put per (tn, piece) gates on piece counter
  (CUStreamWait/host ladder, same discipline as wire_flags today).
- Wire: unchanged blocking putmem_signal, rail-aligned, signal-ADDs row
  counts into the ingress lane counter (receiver lanes already additive).
- Destination: per-piece forward — CE puts of home-rank ranges into
  final recv regions + NVLink signal-ADD to each home counter; own-rail
  range can land pre-placed (its final home IS the ingress rank). Final
  reduce gates on per-source counters == plan totals.
- Buffers: egress staging ~ wire-payload-class per node on the symmetric
  heap — SIZING RISK at 16n b64 (llc cells already brush the 16G cap);
  size from the plan's piece maxima with the r2-envelope discipline.

## 4. Protocol / gates

- New wire schedule => new binary tag (FLUX_A2AV_RS_EGRESS_BAL_TAG) +
  never-mix boundary; knob default OFF until the A/B verdict.
- Rule-6: every wire put stays blocking; intra-node signal ordering is
  NVLink-domain but ALL gates run random payloads (rule 6b); test
  small budgets at scale FIRST (the 8/25 wedge lesson).
- Verdict cells (inherited from TODO 1): 16n qwen b64 combine wire span
  (nsys) < 20 ms AND l1 gap vs llc <= 0; plus b1-b8 parity (threshold
  toggle should make small-b bit-identical); plus the counter==total
  audits green.
- Composition: orthogonal to pv2 placement and the s2 swap lane
  (movement never touches the combine wire path); if the qwen-8n-pv2 l1
  regression is confirmed as wire concentration, this fix addresses it
  placement-independently.

## 5. Explicitly deferred

- v2 epilogue-remote-store (build only on v1 evidence).
- TODO 2 (CE-parcel union closure) — unchanged, separate ticket.
- Any change to receiver lane semantics beyond additive counts.
