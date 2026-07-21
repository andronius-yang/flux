# COMET Layer0 a2av Dispatch: Design Walkthrough

This document explains the raw-alltoallv (a2av) dispatch mode added to the sm80
MoE layer0 op (`GemmGroupedV2AGScatterOp`, ctor flag `a2av_dispatch`), from the
communication-patterns perspective: what the dense baseline does, why the a2av
variant is shaped the way it is, and what each change in the code accomplishes.
For how to *run* it, see `comet_traffic_matrix_tests.md`.

## 1. The baseline: why layer0 is a dense allgather

Flux layer0 splits routing into two planes:

- **Metadata plane** (tiny, routing-dependent): `splits[e]` and
  `scatter_index[t][j]` describe the *global* routing. Every rank holds an
  identical copy before the op runs.
- **Payload plane** (large, routing-*independent*): every rank fetches every
  other rank's entire `tokens_per_rank x hidden` shard into a replicated buffer.
  Source, size, and destination offset of every transfer are pure functions of
  shapes and rank ids — a **static communication plan** that could be printed
  before the gate even runs.

The static plan is what enables COMET's overlap: mat A is pre-sorted by
`(source_rank, expert)` so every GEMM tile depends on exactly **one** source
shard, a per-source-rank flag flips as each shard lands (stream-serialized, so
flags flip in exactly the enqueue order), and the tile scheduler walks tiles in
ring-stage order, spinning on the one flag each tile needs. Communication stays
routing-oblivious; all routing-dependence is pushed into local *index arrays*
(`sorted_gather_index`, `sorted_scatter_index`) that address the replicated
buffer.

The cost: wire traffic is `(W-1)/W * ntokens * hidden` per rank **regardless of
routing**. Under the traffic-matrix harness (EP = world, each expert fully owned
by one rank), the true demand per rank is only ~`topk/W` of the tokens, so the
dense AG moves ~`W/topk` times more bytes than the matrix specifies — and the
matrix only shapes *logical* dispatch, not wire bytes. The a2av mode exists to
make the wire bytes equal the matrix, so the profiling harness measures the
communication pattern it prescribes.

## 2. The a2av pattern: what replaces what

Raw alltoallv dispatch: each `(token, topk-slot)` copy travels **once**,
directly producer rank -> expert-owner rank. No node-level dedup, no topk dedup
— deliberately, so wire bytes s->d equal exactly `M[s][d]` from the matrix
(this is the pattern the nccl-alltoallv baselines measure).

Replaying the three static-plan ingredients under a2av shows exactly what must
change:

| ingredient | dense AG | a2av |
|---|---|---|
| transfer sizes | shape-derived constants | `chunks[s][d]` — routing-dependent, per iteration |
| buffer offsets | `rank * tokens_per_rank` | prefix sums over the chunk matrix |
| arrival order | stream-serialized ring, guaranteed | whenever each source finishes — **not** ordered |

So the design has three corresponding pieces: (a) a producer dispatch that
computes offsets from the routing and pushes payload with signals, (b) a
consumer layout that keeps the GEMM's per-source decomposition intact, and (c)
a tile scheduler that no longer assumes arrival order.

One thing deliberately does **not** change: the metadata plane. In the harness
every rank already holds the global `scatter_index` (deterministically built
from the matrix on all ranks), so every offset on both sides is computed
locally with no exchange. A real system would allgather per-(source, expert)
counts first — a few KB, latency-bound — noted in the code where it would go.

## 3. Producer side: pack once, one put per destination

Implemented in `a2av_dispatch()`
(`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc`).

**Index math (ATen, on-device, ~10 ops).** From the global metadata each rank
derives, for every copy `(t, j)`: its expert `e` (searchsorted over
`cumsum(splits)`), source rank `s = t / tokens_per_rank`, owner
`d = e / ep_nexperts`; then the full `[W, W]` chunk matrix
(`bincount(s * W + d)`), whose row/column prefix sums are the send/recv
offsets. One small D2H copy of this matrix (the only host sync) feeds the put
loop. The layout contract was validated by a standalone simulation before any
GPU run (pack -> contiguous puts -> gather -> scatter reconstruction).

**Why pack at all.** The rows a producer sends to one destination are scattered
across its shard. Sending them row-by-row would mean thousands of ~8 KB
messages — latency-dominated. Instead a single `index_select` gathers all
outgoing copies into a symmetric send buffer ordered
**(destination, expert, dst_row)**. That order is chosen so the segment for
each destination is *exactly* the interior layout the consumer wants (see §4),
which makes each s->d transfer **one contiguous put** and eliminates any
consumer-side unpack.

**Why host-issued `putmem_signal` (not a device kernel).** At W=16 the loop is
at most 15 puts — negligible CPU cost — and host-issued stream-ordered puts are
the pattern this repo already validated on Slingshot (the multi-node layer1
ladder in `gemm_grouped_v2_gather_rs.cc`). Device-initiated puts would route
through the NVSHMEM proxy anyway and add device-link build complexity for no
win at this message count. `putmem_signal` fuses the readiness notification
with the payload: NVSHMEM guarantees the signal value lands *after* the payload
is fully visible at the destination — this is the whole correctness story for
the consumer's wait.

**Signal discipline.** One `uint64 signal[W]` symmetric array; slot `s` on a
consumer is owned by source `s`. The value written is a monotonically
increasing per-op epoch (`run_id_`), and the consumer waits for
`signal[s] >= run_id_`. Because the epoch only grows, the signal array is
**never reset** — no memset, no barrier to clear flags between iterations (the
idiom comes from `src/coll/all2all_op.cc` / gather_rs's `run_id_`). Two rules
keep this sound:

- *Every rank signals every rank every iteration*, including zero-payload pairs
  (plain `signal_op` when `chunks==0`). A GEMM tile's segment span can include
  a zero-count middle rank, and the claimer polls all sources — a missing
  signal would hang the epoch forever.
- The trailing `nvshmemx_barrier_all_on_stream` after the GEMM (same place the
  dense multi-node path has it) both quiets outstanding `nbi` puts and prevents
  iteration n+1's puts from racing iteration n's GEMM reads of the recv buffer.

Self-traffic skips the network: the self-destined segment of the send buffer is
one local `cudaMemcpyAsync` into the own recv region, then a local signal.
Remote puts go out in ring order starting at `rank+1` so all ranks don't hammer
the same destination simultaneously (incast).

## 4. Consumer side: keep the GEMM's world, change only its schedule

The guiding principle: the grouped GEMM already has exactly the right
decomposition — problems split by `(source_rank, expert)`, A rows read through
a gather index, per-tile waits on per-source flags. The a2av mode should *feed*
that machinery a different buffer and a different schedule, not rewrite it.

**Recv layout & index reuse.** The recv buffer is ordered
**(source, expert, dst_row)** — source-major regions so each incoming put is
contiguous; expert-major inside so the wire layout is a per-source slice of the
sorted mat A. The GEMM's A-iterator already reads rows through
`sorted_gather_index`, so the layout change is absorbed entirely by
*recomputing that index* to point into the recv buffer (a permutation,
`argsort().argsort()` over the canonical sort key) — zero changes to the
iterators. `sorted_scatter_index` (output row = `dst_row - P[e]`) and
`sorted_splits_cumsum` (the `[ep_nexperts][W]` per-expert cumsum the kernel's
segment ballot reads) keep their exact dense-path semantics, so the epilogue
and the per-tile segment computation are untouched. Both sides sort by the same
composite int64 key `((.) << 32) | dst_row`, giving a deterministic tie-break —
deliberately *not* reusing `AgScatterSortOpV2`, whose within-group order is
atomicAdd-nondeterministic and would break the producer/consumer contract.

**Per-tile wait, retyped.** The existing spin
(`ag_scatter_gemm_grouped_with_absmax.h`) computes which source segments a
tile's M-range touches and waits on those flags — it was already "wait only for
what this tile needs". The a2av branch changes only the flag type: a
system-scope acquire load on the `uint64` signal, compared `>= run_id`. System
scope because the writer is the NVSHMEM proxy/NIC, not a GPU thread; acquire so
the subsequent A reads are ordered after the payload the signal vouches for.

**Dynamic tile claiming — why ring order had to go.** In the dense path,
arrival order is *guaranteed* to match the ring, so enumerating tiles in
ring-stage order is optimal. Under a2av, arrival order is whatever the network
delivers; a CTA marching through a fixed order would block on a slow source
while other sources' tiles sit ready. Since there is no longer any order worth
encoding statically, the schedule becomes a claim structure
(`fill_problem_info_a2av` in `workspace_util.cu`, built on-GPU per iteration):

- tiles bucketed by the single source they depend on (the vast majority);
  multi-source boundary tiles go to bucket `W` with a source bitmask;
- per-bucket atomic cursors; a persistent CTA (thread 0, result broadcast
  through one smem int) scans sources — staggered by `blockIdx.x` to spread
  contention — relaxed-polls each signal as a *heuristic*, and `atomicAdd`-pops
  a tile from the first ready bucket with work remaining
  (`cutlass_impls/a2av_tile_claimer.hpp`);
- multi-source tiles are claimed when their whole mask has arrived, or when
  nothing else is left (at that point blocking in the per-tile spin is optimal
  anyway); `__nanosleep` backoff when nothing is claimable.

The division of labor is the elegant part: the claimer's signal reads need **no
memory ordering at all** — a stale read in either direction only affects *which
tile is picked next*, never correctness, because the authoritative per-tile
acquire spin still guards every tile's data. Claim-exactly-once falls out of
the atomic cursors (over-claims past the bucket size simply retry), and
termination is all-cursors-exhausted.

**Workspace plumbing without the mirroring hazard.** The dense workspace layout
is computed twice (host `to_gemm_args_impl` and the device prepare kernel) and
must agree byte-for-byte — a classic silent-corruption trap. The new a2av
regions (bucket tiles/offsets/cursors/masks, appended after `problem_info`)
sidestep it: pointers are computed **host-side only**
(`get_a2av_schedule_workspace` in `workspace_util.h`) and passed into the
prepare kernel by value, so there is no device-side offset mirror to keep in
sync.

## 5. Packaging decisions

- **Ctor flag on the existing op**, not a sibling class: the a2av mode shares
  op selection, tuning configs, workspace management, GEMM launch, and the
  output path; the dense path is bit-identical to before (all new kernel
  behavior is gated on `signal_ptr != nullptr`). The same compiled GEMM ops
  serve both modes — no OpRegistry/generator changes.
- **v1 scope** (FLUX_CHECK-guarded): `ep_size == world_size`, single weight
  group, bf16/fp16, no drop-token, no `allgather_output` (there is no dense
  gathered buffer to return), no triton path.
- **Recv capacity knob**: symmetric allocations must be sized up front and
  uniformly across PEs, but per-rank receive volume is routing-dependent.
  Default is 2x the balanced load (`ntokens * topk / W * 2`), overridable via
  `FLUX_A2AV_MAX_RECV_NTOKENS`; overflow hits a loud FLUX_CHECK. Note the
  failure mode in practice: the *hot* rank raises, the other ranks wait in
  collectives — it looks like a hang. The real 4n_16r matrices have ~3x hot
  columns; 4x average is a safe setting.

## 6. What the first measurements say

Correctness: allclose vs the torch reference on every rank in every tested
configuration (1 node x 4r uniform/skewed/first-epoch; 4 nodes x 16r,
16mib and 64mib `dist_001`). Timing (10 iters): 16mib — dense 2.76 ms, a2av
2.99 ms; 64mib — dense 8.58 ms, a2av 9.61 ms, despite a2av moving ~15x fewer
bytes. The gap is the pre-comm critical path: the per-iteration ATen argsorts
and the D2H sync of the chunk matrix run *before* any byte moves, serialized
with everything else. In the harness the routing is identical every iteration,
so caching the index tensors and offsets across forwards (keyed on
`scatter_index` identity) is the obvious next optimization and should recover
the wire-byte advantage; a second lever is overlapping the index math of
iteration n+1 with the GEMM of iteration n.

## 7. File map

| file | role |
|---|---|
| `include/flux/args/moe_ag_scatter.h` | `signal_ptr` / `signal_expected` on the op arguments (null = dense mode) |
| `src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h` | dual-mode `operator()` (claimer loop vs ProblemVisitor), `process_tile` refactor, uint64 signal wait |
| `src/moe_ag_scatter/cutlass_impls/a2av_tile_claimer.hpp` | dynamic per-source-bucket tile claiming |
| `src/moe_ag_scatter/workspace_util.{h,cu}` | `A2AVScheduleWorkspace` (host-computed region pointers), `fill_problem_info_a2av` bucket builder |
| `src/moe_ag_scatter/gemm_grouped_v2_ag_scatter.hpp` | workspace sizing + wiring the a2av args into the kernel |
| `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc/.h` | `a2av_dispatch` flag, symmetric send/recv/signal buffers, `run_id_`, index math + pack + put loop, forward branch |
| `src/pybind/gemm_grouped_v2_ag_scatter.cc`, `python/flux/cpp_mod.pyi` | `a2av_dispatch=False` / `a2av_ring=False` ctor kwargs |
| `test/python/moe_ag_scatter/test_moe_ag_traffic.py` | `--comm_pattern {allgather,a2av,a2av_ring}`, wire-byte logging, `--gather_input` gating |

## 8. Scheduled a2av (`a2av_ring`): static schedule, sparse wire bytes

The third mode splits the difference between the other two. §4 argued that under
raw a2av "there is no longer any order worth encoding statically" — `a2av_ring`
*creates* one: it fixes the order in which ranks communicate (including across
nodes) by mutual agreement, exactly like the allgather, so the dense path's
static ring-stage tile schedule applies again — while still moving only
`M[s][d]` bytes per pair. Enabled by `a2av_ring=True` (requires `a2av_dispatch`),
`--comm_pattern a2av_ring` in the harness.

**The mirrored send order.** The dense schedule's stage order at receiver `r`
(`shift_rank_to_order`, `sort_util.h`) is: self, then the intra-node ring
`lr+1, lr+2, ...`, then the next node's ranks, and so on. `a2av_ring` keeps
that mapping bit-for-bit unchanged and instead makes every *sender* emit in the
mirror order — the reverse hierarchical ring. For send slot `k = 1..W-1` with
`dn = k / L`, `dl = k % L` (`L` = local world size):

```
d = global_rank(local = (lr_s - dl) mod L,  node = (node_s - dn) mod NN)
```

Receiver `d` then sees source `s` at exactly the stage the dense problem
schedule expects it (`d`'s stage-1 source is `d+1`, which sends to `d` in its
slot 1, etc.). Each slot `k` is a bijection source→destination, so the reverse
ring keeps the original forward ring's no-incast property. Self delivery (the
local copy + self signal) is stage 0, unchanged. Intra-node puts are issued on
`cp_stream`, inter-node puts concurrently on `cp_stream_inter_node` — mirroring
the dense multi-node `all_gather_all2all` split — so NIC transfers start
immediately even though their tiles sit at the end of the schedule.

**What flips back to dense machinery.** One bool on the op arguments
(`a2av_ring_schedule`) makes `get_a2av_ws` return an empty workspace, which
routes both discriminators (`signal_ptr && bucket_tiles`) to the dense branch:
the prepare kernel runs `calc_sorted_problem_schedule_v2` + `fill_problem_info`
(consuming the a2av-produced `sorted_splits_cumsum`, whose semantics were kept
dense-identical for exactly this reason, §4), the GEMM walks the precomputed
ProblemVisitor order, and the bucket workspace regions are never allocated. No
kernel changes: the per-tile wait still takes the uint64 signal branch because
`signal_ptr` is set. The claimer, cursors, masks, and `__nanosleep` polling all
disappear from the hot path.

**Correctness contract, unchanged.** As in both other modes, the schedule is
only an efficiency heuristic; the per-tile system-scope acquire spin on the
per-source epoch signals is the correctness gate. `nbi` puts may complete out
of issue order — that can cost overlap, never correctness. The trade being
measured vs the dynamic claimer: under skewed `M[s][d]`, a large early-stage
chunk stalls the fixed order where the claimer would slide past it; in
exchange, ring mode pays zero claim-loop overhead and its schedule (like the
dense one) is a pure function of the splits.

## 9. `splits_per_source`: the metadata exchange as an explicit input

In a real MoE system each rank, after gating, natively knows only its own row
of `cnt[s][e]` (copies it sends to each expert). One `W x nexperts`-int
allgather (~4 KB, ~10-20 us, latency-bound) gives every rank the full matrix,
and `splits[e] = sum_s cnt[s][e]` is a *derived* column sum. The harness has
always declared `forward()` to be post-exchange — `splits` and `scatter_index`
arrive as untimed inputs — but a2av still re-derived `cnt` *inside* the timed
region, and its wire cannot start without it (message sizes and recv offsets),
while the dense allgather needs no metadata for its wire at all. The optional
`splits_per_source` kwarg (int32 CPU `[W, nexperts]`, identical on all ranks)
completes the contract: what the exchange delivers is now an input, per
iteration, in the same untimed setup that builds `splits` — this is
per-iteration metadata, not cross-iteration caching.

With it, the a2av dispatch derives everything host-side before any device
work: `M_this_ep` (overflow check fires before a single kernel), the chunk
matrix and put offsets (`chunks[s][d]` is a block sum of cnt), and the group
tables `offA/cumA/offR` for my experts — staged in one pinned buffer and
uploaded with a single ~2 KB H2D. The counts kernel histogram, the 1 KB D2H,
and the counts-event wait vanish from the timed path; puts gate only on the
pack. Stage 2 collapses to ONE sort plus an arithmetic identity: because the
Phase-B tie-break is the copy index, the A-order -> recv-order map inside any
`(source, expert)` group is rank-preserving, so
`sorted_gather_index[i] = offR_of_A[g] + i - offA[g]` (g found by a binary
search over `cumA`; tail rows clamped in-bounds), and `sorted_splits_cumsum`
is uploaded directly. The `key_a` sort, the scatter-of-iota inverse, and the
group-boundary searches are all deleted. `FLUX_A2AV_CHECK_IDENTITY=1` rebuilds
the old sort-based indices and asserts equality (bring-up guard).

Fairness: the dense path derives `cnt` too (as `sorted_splits`), fused into
`AgScatterSortOpV2` and hidden under the allgather comm — so it never paid a
positional penalty. With the kwarg it gains exactly one thing: `M_this_ep`
from a host sum, removing its only per-iteration device sync. Its kernels and
schedule are untouched, and omitting the kwarg keeps every path bit-identical
to before. The test passes the matrix for all `--comm_pattern` values;
`--no_metadata_cnt` restores the derive-everything behavior for A/B runs.

## 10. Hierarchical a2av (`a2av_hier`): node-aggregated inter-node messages

`a2av_hier=True` (ctor kwarg, requires `a2av_dispatch`, mutually exclusive with
`a2av_ring`; `--comm_pattern a2av_hier`) mirrors the multi-node dense
allgather's `all_gather_all2all` communication structure, with a2av semantics.
From source rank `s = (node_s, lr_s)`:

- **Intra-node (round 0)**: `s` delivers each local peer's chunk directly —
  exactly the ring mode's `dn == 0` slots (mirror local order, `cp_stream`),
  plus the usual self memcpy + signal.
- **Inter-node**: `s` talks only to its `nnodes - 1` same-local-rank peers.
  To the peer on node `t` (the **gateway** `g = (t, lr_s)`) it sends ONE
  aggregated `putmem_signal` containing everything it has for ALL `L` ranks on
  node `t` — not its whole shard as in allgather, only the node's traffic.
  Because the send buffer is destination-major in *global* rank order and a
  node's ranks are globally contiguous, this aggregate is a contiguous slice of
  the existing send buffer: no repacking. Sends go out in mirror node order
  (`tn = node_s - dn`) on `cp_stream_inter_node`, landing in the gateway's
  symmetric staging buffer with an arrival signal (slot = source node, value =
  epoch `run_id_`, never reset).
- **Gateway forwarding**: for each round `dn = 1..nnodes-1` the gateway's
  `cp_stream` executes a `cuStreamWaitValue64(GEQ, run_id)` on the round's
  arrival signal, then forwards each destination's sub-chunk to its `L - 1`
  local peers via `putmem_signal` into their recv buffers at `RO[s][d]`
  (its own sub-chunk is a local memcpy + signal). This wait is the a2av
  analogue of the allgather's inter-node-getmem -> intra-node-redistribute
  dependency (`fetch_remote_event`), realized per round.

Every row still crosses the network exactly once (inter-node wire bytes equal
the per-node column sums of `M`), but in `nnodes - 1` large messages per source
instead of `W - L` small ones; the extra forwarding hop rides NVLink. Forwarded
sub-chunks are internally (expert, copy)-ordered and land bit-identically to
direct puts, so the recv layout, stage-2 index build, and per-tile signal gate
are untouched, and rounds arrive in receiver stage order — the consumer reuses
the ring mode's static dense schedule (`a2av_ring_schedule` = true). Zero
kernel changes.

Transport is deliberately **push** end-to-end. A pull design (gateway
`getmem_on_stream` per round, the literal allgather structure) would need a
pack-completion handshake (e.g. a `NVSHMEMX_TEAM_SAME_MYPE_NODE` barrier)
before the first get and pays get round-trips on libfabric, for no buffer
savings. The gateway wait uses the raw driver memop rather than
`nvshmemx_signal_wait_until_on_stream`: the NVSHMEM wrapper falls back to an
SM spin kernel when 64-bit stream memops are unavailable, which could deadlock
behind a full-occupancy GEMM; `cuStreamWaitValue64` runs on the GPU front end
with zero SMs (precedent: the layer1 op paces its inter-node puts with
`CUStreamWaitValue` GEQ). The GEMM is gated only on round-0 puts + inter-node
sends being *issued* (`hier_dispatch_event_` + `fetch_remote_event`);
forwarding overlaps the GEMM, whose per-tile signal spin remains the
correctness gate.

Signal ownership per epoch: same-node `(s, d)` slots are set by the source
(round 0 / self path); cross-node slots are set by the gateway `(node_d, lr_s)`
— put-fused, or a bare `signal_op` for empty sub-chunks. Empty node aggregates
still set the arrival signal, so gateways wait uniformly. Single writer per
slot per epoch.

New state (allocated only when `a2av_hier && nnodes > 1`): the symmetric
staging buffer (`FLUX_A2AV_MAX_STAGE_NTOKENS` rows, default = the recv-buffer
formula; expected load ~ one rank's recv since a node's inbound traffic splits
across `L` gateways by source local rank) and the `uint64[nnodes]` arrival
signal array. Staging overflow is FLUX_CHECKed on the host before any wire,
with the max taken over ALL gateways so every rank fails the same iteration
(no one-rank-throws hang). Epoch safety across iterations needs nothing new:
forwarding reads are enqueued before `all_gather_event`, which the main stream
waits before the tail `barrier_all`, and iteration n+1 sends wait on
`ready_event` recorded after that barrier.

`nnodes == 1` degenerates to round-0 only — behaviorally the ring mode's
intra-node slots with the same static schedule (cheap single-node validation
of the branch).

## 11. Layer1 hierarchical a2av combine (`GemmGroupedV2GatherRSOp`, `a2av_hier`)

The combine direction is the transpose of §1-§10: each `(t, j)` copy was
computed on expert-owner rank `s = owner(e(t,j))` and must reach token-home
rank `d = t / tokens_per_rank`, where the only remaining reduction (in the
`T=1, E=W` regime the GEMM rows are complete — no K-partials) is the per-token
topk sum. `a2av_hier=True` on `GemmGroupedV2GatherRSOp` replaces the dense
ring reduce-scatter with a split-pipelined a2av built almost entirely from
machinery this file already had.

**Pipeline per split `sid`** (`n_split` column windows; the split-major GEMM +
tile→problem→split counter cascade are reused byte-for-byte, zero changes to
`cutlass_impls`):

1. **Pack** (`a2av_combine.cu`, persistent kernel on the margin blocks,
   `FLUX_A2AV_RS_PACK_BLOCKS`): waits the split flag — the minimal correct gate,
   since any destination's rows interleave across every local expert — then
   gathers each outgoing copy's `n_per` column window from `gemm_outs` into the
   symmetric send panel, destination-major in global rank order, applying
   `output_vec_scale` per source row during the copy (one extra bf16 rounding vs
   the dense path's fused fp32 accumulation; the destination reduce still
   accumulates fp32). Chunk completion per `(dest_node, sid)` uses the existing
   `group_counters`/`group_flags` handshake — including the OWN node's chunk,
   which the dense ring kernel never flags. Remote-node chunks are packed first
   so NIC-bound flags flip earliest.
2. **Transport** (host-pre-enqueued ladders, zero SMs, all epoch `run_id_`
   signals, never reset, every pair signals every split): intra-node direct
   `putmem_signal` per local peer (CE over NVLink) + self memcpy; inter-node ONE
   aggregated put per `(remote node, sid)` into the same-local-rank gateway's
   staging panel (the send panel's node slice is contiguous — no repacking);
   the gateway ladder paces per `(sid, source node)` on a zero-SM
   `cuStreamWaitValue64(GEQ, run_id)` over the arrival signal and forwards each
   local destination's sub-chunk with the per-source recv signal — forwarded
   sub-chunks land bit-identically to direct puts. Unlike §10's mirror node
   order, the inter-node ladder consumes flags in the pack kernel's production
   order (`node_idx+1` ascending): there is no consumer schedule to satisfy in
   the combine, and matching production order avoids head-of-line blocking under
   `CUDA_DEVICE_MAX_CONNECTIONS=1` (enqueue order across the shared front-end
   channel must be an executable schedule; the pack kernel is always launched
   before any ladder wait, and the reduce waits are enqueued after the gateway
   ladder they depend on).
3. **Reduce** (per split, on its own stream, `FLUX_A2AV_RS_REDUCE_BLOCKS`):
   `W` front-end `cuStreamWaitValue64` waits on the split's per-source recv
   signals (a token's topk copies come from up to topk owners), then one
   memory-bound kernel folds each local token's topk recv rows (fp32) into
   `output[:, sid*n_per : (sid+1)*n_per]`. Deterministic j-order summation —
   bit-stable across runs, unlike the dense ring's arrival-order sums.

**The mirror-layout contract** is what makes the index math nearly free: the
send panel on owner `r` is `(home_rank, expert, dst_row)`-ordered — exactly
§4's recv layout on `r` — so the pack index is the inverse of
`sorted_gather_index`'s arithmetic identity, derived from the SAME
`offA/cumA/offR_of_A` host tables (§9) with no sort; and the recv panel on home
`d` is `(owner_rank, expert, dst_row)`-ordered — exactly §3's send-buffer
layout on `d` — so every copy lands back at its layer0 pack position and the
reduce index is the inverse of the ONE pack-key sort rank `d` already runs as a
layer0 sender. Standalone layer1 therefore pays layer0's index cost (one sort +
identities); a fused layer0+layer1 pipeline passes layer0's tensors via the
`a2av_pack_index`/`a2av_reduce_index` forward kwargs and pays it once.
`FLUX_A2AV_RS_CHECK_IDENTITY=1` asserts identity-path == brute-force-sort;
`test_a2av_combine_sim.py` validates the whole contract on CPU.

**Buffers** (symmetric heap; dense-only buffers — ring reduce buffers, tile
barriers, dense staging, internode signals — are skipped in this mode): send
panel `[n_split, FLUX_A2AV_RS_MAX_SEND_ROWS, n_per]` (routing-dependent hot
owner, collective overflow check), recv panel `[n_split, max_m/W, n_per]`
(EXACT — every token comes home with topk copies, no knob), gateway staging
`[n_split, FLUX_A2AV_RS_MAX_STAGE_ROWS, n_per]` (collective check), recv
signals `uint64[W * n_split]`, arrival signals `uint64[nnodes * n_split]`.
Epoch safety needs nothing new: all four ladder/reduce streams are
event-joined onto the gather-rs stream before `gather_rs_done_event`, so the
existing `barrier_all` close covers panel and staging reuse, and `nnodes == 1`
degenerates to the intra ladder + reduce (cheap single-node validation).
