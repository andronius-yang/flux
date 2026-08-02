# Layer0 dedup (a2av_hier_compress) — variants and machinery

Companion to `layer0_a2av_walkthrough.md` (raw a2av + a2av_hier). Started
2026-07-31. Constants as before: L local ranks, N (=NN) nodes, W = N*L
ranks, T tokens/shard, k = topk. Code references are
`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc` unless noted.

## 0. High-level map of the compress family

### The shared dedup core

Raw/hier ship COPIES (a token routed to 3 experts of node m crosses the
wire 3x). Compress changes the wire contract to UNIQUE TOKENS, tracked by
two locally-derived, globally-replicated matrices:

- `u[s][d]` — unique tokens of source s needed by destination rank d;
- `U[s][n]` — unique tokens of s needed by ANY rank of node n (union).

Wire: intra puts carry u[s][d] rows; inter-node, ONE union aggregate of
U[s][n] rows per remote node, staged at the gateway, re-fanned locally.
Recv side: each token lands ONCE per source region; the GEMM's multiple
logical A rows alias one physical recv row via `sorted_gather_index`
(one-cumsum identity C[t], build_stage2 compress branch) — dedup is free
at the GEMM because gather indices were always arbitrary. Producer side:
the send buffer becomes SEGMENTS of unique tokens (L per-rank segments
for my node + one union segment per remote node, nseg = L + NN - 1),
built by the flag/scan pack (`a2av_pack_flag_` -> `a2av_pack_scan` ->
`pack_gather`, fused with stage1).

### Variant map (sweeps/variants.py)

| variant | env | one-line |
|---|---|---|
| `hier_compress` (default) | — | dedup + **balanced inter-node relay** (load balancing) |
| `hier_compress_identity` | `FLUX_A2AV_RELAY_IDENTITY=1` | A/B control: fixed relay=self wire, byte-identical to pre-relay compress |
| `hier_compress_union` | `FLUX_A2AV_UNION_BCAST=1` | SM-free gateway: broadcast whole union, receivers alias subsets (implies identity wire) |
| `hier_compress_pack` | union + `FLUX_A2AV_PACK_OVERLAP=1` + `CUDA_DEVICE_MAX_CONNECTIONS=2` | iteration n+1's pack overlaps iteration n's GEMM (parity double-buffered send) |

### 0b. Q&A — is the gather-at-gateway design still resident?

Yes — it is the DEFAULT. Exact forwarding (union lands in gateway
staging; gateway index_selects each local rank's subset into scratch —
only then contiguous — and ships one put per local destination) is the
forwarding path for both `hier_compress` (balanced relay) and
`hier_compress_identity`; only the union variants bypass it (`!union_bcast_`
guard in the compress put block, ~1738+). Supporting machinery:

- forward indices (posU = token's rank within the staged union) built on
  the main stream WHILE the wire flies: batched over all NN-1 rounds in
  identity mode (1292+); windowed per-round in balanced-relay mode
  (1138+), where a staged segment's rows arrive split across L relays —
  hence the pinned cnt_before/cnt_in [2, NN-1, L, L] table + fwd_cnt_event_.
- the gather is a real kernel -> sm_margin >= 1 enforced (the one SM
  consumer in the family).
- the forward put is deliberately NON-nbi: local completion must be known
  before the next round's gather reuses the scratch.

Family tree: exact forwarding is the trunk; union_bcast is the branch
asking "what if the subset is never made contiguous at all?" — it then
exists only logically, as receiver consumer indices aliasing scattered
rows of the U region at GEMM-read time (no kernel/scratch/sm_margin/
non-nbi constraint; cost L*U NVLink bytes vs sum_d u). Empirically
(variants.py comment) union + pack overlap is the validated best config
so far — deleting the gather kernel has beaten shipping fewer NVLink
bytes on this hardware. Exact forwarding remains default: it is the
general design and the only one compatible with the balanced relay.

### 1. Default = balanced relay (members 363-395, partition 829-860)

Problem: with relay=self, rank l's inter-node bytes = U[(n,l)][m] —
hostage to its own routing; skew saturates one NIC while siblings idle;
rounds finish at the max, not the mean.

Fix: per (source node n -> target node m) round, the node's L union
segments concatenate into ONE canonical stream (ascending source local
rank, token-ascending interiors == pack order), cut into L near-equal
chunks; relay rank k stages chunk k (intra-node piece puts deliver the
parts owned by siblings) and wire-puts it to the same-local-rank gateway
on m. Balance is exact BY CONSTRUCTION and every boundary derives from
the replicated U matrix — sender/relay/gateway/destination agree with
zero extra metadata (the partition lambdas `canon_start`/`chunk_bound`
are the single source of truth; comment warns any re-derivation silently
corrupts wire offsets).

Price: intra piece-put stage; relay staging buffer (`a2av_relay_stage_`);
two signal layers (`a2av_relay_sig_` slot (dn-1)*L+src_lr,
`a2av_gw_round_sig_` slot (dn-1)*L+gw_lr); per-destination signal
AGGREGATION on dedicated `cp_stream_signal` (pure front-end
CUStreamWaitValue64/WriteValue64, zero SMs — must not ride cp_stream:
would couple gateway rounds across local ranks; nor cp_stream_inter_node:
would poison fetch_remote_event, which gates the GEMM launch — 243-247);
new GEMM-gate event `relay_send_event_`; relay windows cnt_in/cnt_before
(pinned [2, NN-1, L, L], `fwd_cnt_event_`).

### 2. union_bcast (members 300-307)

Baseline compress forwarding index_selects each local rank's exact subset
from the staged union — a kernel, hence sm_margin >= 1. Union-bcast
deletes it: gateway forwards the WHOLE staged union to every local rank
(contiguous nbi puts straight from symmetric staging — no gather, no
scratch, no SM reservation); each receiver aliases its subset out of the
U-sized region through consumer indices. RECV layout changes:
remote-source regions hold U[s][n] rows, not u[s][d]. Trade: NVLink
carries L*U instead of sum_d u — waste when tokens are needed by few
local ranks, free when needed by all L. Implies identity wire (ORs into
relay_identity_) — union and balanced relay are currently exclusive.

### 3. pack overlap (members 249-258)

Different axis: removes the producer pack from the critical path.
Iteration n+1's pack chain (meta H2D -> stage1 -> pack scan -> send
gather) runs on dedicated `pack_stream_`, overlapping iteration n's GEMM.
Needs parity double-buffering of send buffer + meta arena (run_id_ & 1;
the 2x symmetric allocation is collective — env must match on all ranks)
and CUDA_DEVICE_MAX_CONNECTIONS=2 (with 1 hw queue it is a wall-clock
no-op). Consequence: ready_event no longer implies the previous close
barrier -> explicit `barrier_done_event_` waits on the put streams.
Contract: caller must keep inputs_shard/scatter_index/splits alive and
unmutated until the next forward (pack_stream reads without allocator
bookkeeping).

**Caveat — cross-iteration overlap and what it measures (Q&A 2026-07-31):**
the overlap is literally iteration n+1's pack under iteration n's GEMM of
the SAME op in the harness loop. Within ONE forward pass there is no
slack: the pack's inputs (inputs_shard, scatter_index) are produced by
gating immediately before dispatch, so the pack is critical-path for a
one-shot call — pack_overlap buys NOTHING there (no previous GEMM to
hide under; the 2x send allocation is still paid). It is real, however,
wherever the same layer runs back-to-back on independent data: training
(grad accumulation, 1F1B pipeline microbatches — the dominant Comet
regime) and continuous-batching inference. It is fake for single-stream
decode (step n+1 depends on step n's full model output). Precise framing:
a THROUGHPUT optimization measured faithfully by the steady-state loop,
but a misleading ONE-SHOT-LATENCY number — quoting hier_compress_pack
against other variants embeds a deployment-cadence assumption the other
cells don't. IRL integration contract: forward() inputs device-ready and
unmutated until the next call (pack_stream_ reads outside allocator
bookkeeping).

Orientation: the DEFAULT is the hardest — the only variant adding new
distributed machinery (piece puts, two signal layers, aggregation, relay
windows). union_bcast and pack_overlap are REMOVALS (of the gateway
gather kernel / of the pack's critical-path slot) bought with extra
NVLink bytes and extra symmetric memory respectively.

## 1. hier_compress_identity — timeline and new machinery (2026-07-31)

The logical successor to a2av_hier: same stream/signal/round discipline,
but bytes cross each link once per unique token. Timeline (N=4, L=4):

```
host   [u/U from pinned tables][seg_off·recv_off_u·fwd_col_off][meta H2D][enqueue all]   «no wait C — compress requires host metadata (line 821)»
main   [memset flags]──[stage1: decode+flags]──[scan]──[pack gather]──R──[fwd-idx build: posU columns, N−1 rounds, one batched pass]──F──[stage2: dedup consumer C[t] one-cumsum]······································(wait H·I)┌►[ GEMM ▓ tiles spin: sig[s] ≥ run_id ▓ ]──(wait AG)──[█ close barrier █]
                                                                      │                                               │               │                                                                                          │                                          │
cp     ·······························································├─[self u·sig[me]][p+s→ℓ₁ u][p+s→ℓ₂ u][p+s→ℓ₃ u]H╌╌╌╌╌╌╌╌(wait F)─[w1][g1·SM][f1: p+s→ℓ₀…ℓ₃]─[w2][g2·SM][f2: p+s→ℓ₀…ℓ₃]─[w3][g3·SM][f3: p+s→ℓ₀…ℓ₃]─(wait I)┼─AG╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘
                                                                      │                                                                  │                          │                          │                                 │
wire   ·······························································│································································[ns1]······················[ns2]······················[ns3]·······························│
                                                                      │                                                                                                                                                          │
inter  ·······························································└─[agg p+s→G@(n−1,me) U·nodesig][agg p+s→G@(n−2,me) U·nodesig][agg p+s→G@(n−3,me) U·nodesig]·······························································I
```

(Both gate waits enqueue together at 2748-2764; H's vertical stops at the
connector because main is mid-stage2 at that wall-clock — release is at
the gate.)

New machinery forced by the unique guarantee (vs a2av_hier):

1. **C is gone** — u/U underivable from per-expert counts (expert overlap
   invisible), so compress REQUIRES host tables (splits_per_source +
   a2av_unique_counts [W, W+NN]; 821-822). Host computes seg_off_h /
   recv_off_u / fwd_col_off from pinned tables, one async H2D (960-1034).
2. **Pack = flag→scan→gather** — dedup physically happens in stage1's
   IDEMPOTENT flag write (seg-major [nseg,T]; duplicate (token,seg)
   copies write 1 again; sort_util.cu 577-589); per-segment exclusive
   scan ranks flagged tokens; one index_select gathers. Send buffer =
   nseg = L+N−1 segments of unique tokens (L per-rank u segments + one
   U union per remote node), not per-destination copy chunks (1177-1239).
3. **Intra puts shrink** to u(me,d) rows, same discipline (1815-1836).
4. **Union aggregate** of U(me,tn) rows per remote node — token-
   ascending, NOT destination-major → forces 5-6 (1848-1872).
5. **New event F (fwd_index_event_)** — forward indices (posU = each
   destination-row's rank within the union), built on main in one
   batched pass over all rounds, overlapping the wire; gates the
   gateway gathers (wait at 1917; build 1292-1341).
6. **Gateway rounds gain a kernel** — per round: doorbell wait →
   index_select ON cp_stream (SM kernels: birth of sm_margin >= 1) into
   scratch → NON-nbi forward puts (local completion before next round's
   scratch reuse); gateway's own subset gathered straight into recv +
   bare signal (1913-1990).
7. **Consumer aliasing** — dedup recv layout (recv_off_u); one-cumsum
   C[t] makes every copy of t read one recv row; GEMM logical semantics
   (chunks64, M_this_ep, group tables) deliberately untouched (955-958).
8. **Collective overflow checks** — staging capacity FLUX_CHECKed with
   the same U-derived expression on every rank → collective failure, not
   a one-rank hang (1772-1789).

One-sentence delta: hier's forwarding was free because the aggregate
arrived destination-major; compress buys "bytes once" by giving that up,
and items 5-6 (fwd indices, SM gather, non-nbi scratch discipline) are
exactly the bill.

### 1b. Q&A — the [meta H2D] box, and which path the earlier timelines drew

The meta upload is NOT compress-specific: it belongs to the use_meta path
(splits_per_source provided), which the traffic test enables BY DEFAULT
(derive mode is the opt-in flag, test line 391). So:

- raw a2av / hier, derive path = what the layer0_a2av_walkthrough
  timelines drew: [1KB D2H] -> C blocks the host; no meta H2D.
- raw a2av / hier, metadata path = what the sweeps actually measure:
  C vanishes; a [meta H2D] box appears on main (group tables cumA / offA /
  offR_of_A / expert_base / ssc in one pinned arena, single async H2D —
  334, 880).
- compress: metadata path mandatory; the upload additionally carries
  seg_off + fwd_col_off (958, 1088).
- baseline allgather: neither — sort kernels derive on device.

Edge-kind contrast (why it is a BOX, not a vertical): C is device->host
and BLOCKS the CPU (the one host-horizon stall). Meta H2D is host->device
and blocks nobody — async, ordered before its consumers by plain stream
order; its only sync is the pinned-arena reuse guard
cudaEventSynchronize(counts_event_[par]) (889), which doubles as the
H2D-completion event and returns immediately in steady state. Op inputs
(inputs_shard / scatter_index / splits_gpu) are uploaded by the CALLER
before forward() in every variant — the stand-in for the real metadata
all-gather, outside all timelines and outside sweep timing.

### 1c. Step-by-step: host = geometry, main = membership (identity)

The wire pattern is inherited from a2av_hier verbatim; all new
intelligence is computation. Division of labor: host computes everything
SMALL and shape-defining (O(W^2) offsets/capacities); main computes
everything O(tokens) (membership, ranks-within-sets, permutations).

HOST (never blocks): H1 pinned-arena guard (889, steady-state no-op).
H2 logical copy counts chunks64 + M_this_ep from cnt[s][e] (892-926) —
GEMM geometry deliberately unchanged by dedup. H3 stage2 group tables
cumA/offR_of_A/ssc/expert_base (930-955). H4 u_mat/U_mat from uc_host +
sanity checks u <= U <= sum u (960-1004). H5 recv_off_u, seg_off_h
(L per-rank u segments + N-1 union U segments), total_send_rows
(1016-1034). H6 fwd_col_off_h — exact-packed a2av_fwd_idx_ columns per
(round, local dest) (1036-1090). H7 pinned meta arena -> ONE async H2D
(1088-1092). H8 collective staging checks + put composition from the
layout lambdas (1750-1789).

MAIN: M1 memset flags (1179). M2 stage1: decode + IDEMPOTENT flag
writes — THE DEDUP — + pack keys (1185-1207). M3 per-segment exclusive
scan -> pack_gather (1220-1227). M4 one index_select -> send segments ->
R (1237). M5 fwd-idx build -> posU columns -> F (1292-1341). M6 stage2:
perm_a + mine_token -> one cumsum C[t] -> sorted_gather_index — the
k-copies-alias-1-row map (1466-1533).

**Where dedup happens (Q&A):** at M2's flag write, NOT the gather. Flag
slot = (seg, lp/topk) with seg derived from the OWNER RANK (owner = e/E),
not the expert: two experts on one rank -> same seg; two ranks on one
remote node -> that node's single union seg (node-level dedup, same
mechanism, coarser segment). Plain =1 suffices: set membership, not
multiplicity; identical-value concurrent stores are a benign race. The
scan counts the 1s; the gather materializes them.

**Why F exists (Q&A):** R certifies PAYLOAD (send buffer), F certifies
INDICES (posU). R is recorded before the fwd-idx kernels run, and
producer (main) / consumer (cp index_select) are different streams —
stream order never spans streams; without F there is NO event between
them and the gather can read a2av_fwd_idx_ mid-write.

**The invariant that makes three independent index structures agree with
zero negotiation: every ordering is token-ascending within its segment.**
Pack scan emits ascending-token interiors -> staged union is
ascending-token -> posU (computed from metadata alone) predicts physical
staging rows -> forwarded subsets arrive ascending-token per source ->
C[t] (a cumsum over ascending token) predicts physical recv rows.
Producer, transit, consumer never exchange an index; each rederives the
same canonical order from the same replicated metadata. Break the
convention at any one site and the other two silently read wrong rows —
hence the centralized layout lambdas and the "any divergent re-derivation
silently corrupts wire offsets" warning.

### 1d. Q&A — the token trace, and the breakeven resolved

Trace (token t of source s, needed by experts on (m,l1) x2 and (m,l2)):
ONE flag slot in node m's union segment (sort_util.cu 582-586: own node =
L per-rank segments, each remote node = one union segment; all copies of
t collapse to (seg_m, t)); one send row; one EFA row; TWO gateway-forward
rows (l1, l2); one recv row at l1; two GEMM A rows aliasing it.

General law: a token needed by r local ranks of node m crosses EFA once
but NVLink r times — "bytes once" is a PER-LINK guarantee, deduping the
expensive link and re-expanding on the cheap one.

Breakeven (resolved): exact forwarding moves sum_d u = r_bar*U NVLink
rows; union-bcast moves L*U. Since 1 <= r_bar <= L, union-bcast NEVER
wins on bytes (worse by L/r_bar). Its empirical win (validated best in
sweeps) is entirely kernel economics: deletes the index_select (SM
contention with the GEMM — the family's only regime-A residue), the
scratch + non-nbi round serialization, and launch count on the single hw
queue, replacing them with contiguous CE puts. Real tradeoff:
(L - r_bar)*U extra NVLink TIME vs gather time + SM interference. High
topk / broad routing (r_bar -> L, e.g. k=8 on L=8) makes union nearly
byte-free — exactly the sweep regime. EFA bytes identical either way (U),
so the wire lane never moves between the two variants.
