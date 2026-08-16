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

## 11. Token-dedup hierarchical a2av (`a2av_hier_compress`)

`a2av_hier_compress=True` (ctor kwarg, requires `a2av_dispatch`, mutually
exclusive with both `a2av_ring` and `a2av_hier`; `--comm_pattern
a2av_hier_compress`) changes the WIRE contract of §10 while keeping everything
logical untouched. The traffic matrix now represents **logical** bytes: the
matrix → `choosed_experts` derivation, `splits`, `scatter_index`, the GEMM
problem sizes and the static dense schedule are all exactly as before. The wire
makes a binary send-once decision instead:

- **Intra-node**: a token goes to a destination *rank* at most once, even when
  that rank hosts several of the token's experts.
- **Inter-node**: a token crosses a link at most once — each remote node
  receives ONE union aggregate (tokens needed by *any* rank on that node) at
  the same-local-rank gateway.
- **Receivers handle duplication**, and it is free: the GEMM reads A rows
  through `sorted_gather_index` and `gather_A` is read-only in the kernel
  (it only feeds `IteratorA`), so multiple A rows simply alias one recv row.

### Metadata input: `a2av_unique_counts`

Dedup counts are NOT derivable from `cnt[s][e]` (§9) — they depend on which
tokens overlap across experts/ranks. They ARE derivable locally, though: every
rank already holds the global `choosed_experts`/`scatter_index` (the gateway
forward-index build decodes remote sources' routing from it), so unlike
`splits_per_source` a real system needs no extra exchange — just a local host
computation (or a device kernel + ~1 KB D2H like the no-metadata counts path).
The harness passes it as a second untimed host metadata tensor
(`forward(..., a2av_unique_counts=...)`, same contract as
`splits_per_source`): int32 CPU
contiguous `[W, W + nnodes]`, identical on all ranks. Cols `[0, W)` hold
`u[s][d]` (unique tokens source `s` must deliver to rank `d`), cols
`[W, W + nnodes)` hold `U[s][n]` (unique tokens `s` must deliver to the node-`n`
union). Compress mode FLUX_CHECKs both metadata tensors present; host sanity
checks pin `u ≤ chunks`, `(u > 0) ⟺ (chunks > 0)`, `u[s][d] ≤ U[s][node(d)]`,
and `U[s][n] ≤ Σ_d∈n u[s][d]`.

### Layouts and the one-cumsum consumer identity

All orders derive from the same global `scatter_index` every rank holds:

- **Send buffer**: nodes ascending; my node expanded into `L` per-destination-
  rank segments (ascending global rank), each remote node ONE union segment;
  every segment interior is ascending token index. Intra-node put per rank and
  inter-node aggregate per node each stay ONE contiguous put.
- **Recv buffer**: source-major regions of `u[s][me]` rows, interior ascending
  token index.
- **Consumer index**: with `mine_token[t] ∈ {0,1}` (any copy of global token
  `t` routes to my experts) and `C[t]` its exclusive prefix sum, every copy of
  `t` reads recv row `C[t]` — tokens are source-contiguous, so
  `recv_off_dedup[s] + rank-of-t-within-s == C[t]` exactly. Hence
  `sorted_gather_index = C[copy // topk][perm_a]` with the existing single
  `key_a` argsort; `sorted_scatter_index` and `sorted_splits_cumsum` keep their
  logical semantics, and the per-tile signal waits and epilogue are untouched.
  `FLUX_A2AV_CHECK_COMPRESS=1` (debug, may sync) inverts `C` and asserts every
  A row maps to its own token, and additionally asserts the on-device pack /
  gateway flag counts reproduce the `u`/`U` metadata (catching harness counts
  that pass the host sanity bounds but disagree with the actual routing).

All index builds (producer pack, consumer, gateway forward indices) are
sync-free ATen: scatter-with-garbage-slot + cumsum + index_select, no
`nonzero`/`masked_select`.

### Gateway: exact per-rank subsets

The gateway gathers each local destination's exact subset out of the staged
union (`index_select` with a precomputed per-round index built on the main
stream, gated by `fwd_index_event_`), then forwards `u[s][d]` rows per local
peer — its own subset is gathered straight into the recv region. Two hazards
and their resolutions:

- **Scratch reuse under `nbi`**: forwarded subsets are staged in a local
  scratch refilled every round, and `nbi` puts give no local-completion
  guarantee — round `r+1`'s gather could overwrite scratch mid-put. Fix: the
  scratch-sourced forwards use **non-`nbi`** `nvshmemx_putmem_signal_on_stream`
  (rounds are serialized on `cp_stream` anyway). Double-buffered scratch is a
  perf follow-up.
- **SM-occupancy deadlock**: unlike §10's SM-free forwarding, the gather needs
  SMs while GEMM tiles spin on signals only that gather can produce. Fix:
  compress with `nnodes > 1` FLUX_CHECKs `sm_margin >= 1` (pass e.g.
  `--sm_margin 8`); the ATen gathers run inside a `CUDAStreamGuard` on
  `cp_stream`.

Signal discipline is identical to §10 (per-source epoch signals, per-source-
node arrival signals, empties still signal, single writer per slot per epoch);
only the byte counts and offsets change (`u`/`U` instead of chunk sums), so the
recv/staging overflow checks switch to the dedup sums — still evaluated
identically on every rank. Existing `a2av_hier` remains byte-identical for A/B
comparison.

### Deferred alternatives to the SM gather (decision: strict wire bytes first)

The exact-subset gather is the sole reason compress needs SMs on the forward
path (the `sm_margin >= 1` rule, the scratch, the non-`nbi` puts). Two designs
would remove some or all of that machinery, **deliberately deferred** until the
current strict-wire-bytes design is validated and measured on Perlmutter — the
A/B against `a2av_hier` should isolate the dedup win before any NVLink-byte
trade-off muddies it:

- **Union broadcast** (`a2av_hier_bcast` candidate): the gateway forwards the
  WHOLE staged union contiguously to each local peer — pure CE, no gather, no
  scratch, no `sm_margin` requirement. Receivers alias their subset out of the
  union region via the same one-cumsum consumer identity (flag = "needed by my
  node" for remote-node sources, "needed by me" for same-node ones). Cost:
  intra-node forward bytes rise from `Σ_dl u[s][dl]` to `(L-1)·U[s][n]` and the
  recv regions for remote sources grow to `U` rows; inter-node bytes (the
  compression target) are identical. Mostly a *deletion* of the gateway
  machinery.
- **Receiver pull**: each local peer `index_select`s its own subset directly
  out of the gateway's staging via `nvshmem_ptr` (NVLink read → local write).
  Exact bytes, one hop instead of two (no scratch round-trip through gateway
  HBM), parallel across L receivers instead of serialized on the gateway.
  Needs the gateway to republish arrival to local peers (cheap `signal_op`
  fan-out) and per-receiver subset indices (same one-cumsum machinery); still
  needs `sm_margin` on every rank.

## 12. Balanced inter-node relay (compress default; `FLUX_A2AV_RELAY_IDENTITY=1` restores §11)

§10/§11 fix the inter-node schedule to "relay = self": in round `dn` every rank
sends its own `U[rank][tn]` union rows to the same-local-rank gateway. Rounds
are serialized (sender `cp_stream_inter_node`, gateway `CUStreamWaitValue64`),
so **each round advances at the pace of the largest `U[s][tn]` on the node** —
skewed routings idle the other NICs and concentrate gateway staging on hot
source local ranks. The balanced relay generalizes the fixed assignment into

```
src rank -> local relay rank (NVLink) -> wire -> same-lr cross-node relay -> dst rank(s)
```

with §11's scheme as the degenerate `relay = self` case.

### Partition: canonical stream + contiguous chunks

Per round (source node `n` -> target node `tn`), the L union segments form ONE
canonical stream: ascending source local rank, token-ascending interiors —
exactly the §11 pack order, so no producer change. `chunk_bound()` cuts it into
L near-equal contiguous chunks (`a_k = k*(total/L) + min(k, total mod L)`);
relay local rank `k` owns chunk `[a_k, a_{k+1})` and wire-puts it to gateway
`(tn, k)`. Everything derives from the **replicated** `U` matrix, so sender,
relay, gateway and destination agree with zero new metadata; the helper is the
single source of truth for every offset and capacity check. When the `U`
values are equal, chunk `k == source k`'s segment (zero relocation); in
general at most `2L-1` intra-node pieces move per node per round, and gateway
staging becomes balanced as a side effect (the `FLUX_A2AV_MAX_STAGE_NTOKENS`
hot-lr concentration of §11 disappears).

### Send side: pieces first, then the wire

Phase 1 pushes ALL rounds' pieces (send-buffer sub-ranges cut at chunk
boundaries) into the relays' symmetric `a2av_relay_stage_` via intra-node
`putmem_signal_nbi`, per-(round, src_lr) `a2av_relay_sig_` slots; the
own-relay piece is a local `cudaMemcpyAsync`. **Deadlock rule**: every rank
issues every piece put before its first wire wait — pieces depend only on the
local pack, so the cross-rank wait graph stays acyclic (interleaving pieces
with wire rounds would cycle). Phase 2 then walks the rounds in mirror node
order: front-end waits on the actual contributors (host-known from `U`,
zero-row pairs skip both signal and wait), then ONE contiguous
`putmem_signal_nbi` of `~total/L` rows to the gateway; `node_sig` keeps its
single-writer-per-slot discipline. A chunk fully inside the relay's own
segment wire-puts straight from the send buffer (no staging hop) — the §11
behavior falls out automatically for balanced routings and `L == 1`.

### Receive side: window-generalized forward build + a tiny D2H

Gateway `k` now stages an arbitrary window `[a_k, b_k)` of the canonical
stream, spanning several source local ranks. The forward-index build drops its
`.select(1, local_rank)` and flags `(round, src_lr, token, dst_lr)`; union
positions plus host canonical starts give canonical positions, the window mask
selects my slice, and the stored value is the window-relative staging row.

A window cut inside a source's segment splits its `(s, d)` recv region across
gateways. Each gateway's slice is contiguous (a window cut of token-sorted
rows), but its offset — `cnt_before`, the count of `(s, d)` rows in earlier
windows — is **token-level** information no aggregate metadata can provide.
The build therefore D2Hs `cnt_in`/`cnt_before` (`2·(NN-1)·L²` int32, one
pinned copy) and the host `cudaEventSynchronize`s on it **after** the wire
issue, immediately before the gateway loop (its only consumer); the GEMM
launch gates on `relay_send_event_` (pieces issued) instead of
`fetch_remote_event`, which now contains cross-rank waits. Delivery reuses the
§11 scratch + non-`nbi` machinery, packing scratch exact-sized from the D2H'd
counts and fusing the per-round gateway signal onto the LAST piece per
destination (intra-node on-stream puts to the same peer land in stream order).

### Signal aggregation: per-source epoch signals keep one writer

With slices arriving from several gateways, the per-source signals the GEMM
spins on would have multiple writers. Fix on the destination, on a third
stream `cp_stream_signal` (pure front-end memops, zero SMs): per round, wait
on the L per-(round, gateway) `a2av_gw_round_sig_` slots, then
`CUStreamWriteValue64 signal[s] = run_id` for every source of that node —
zero-traffic sources included, exactly §11's empties-still-signal rule.
`putmem_signal` orders payload before signal, so gateway-slot arrival implies
full delivery. The GEMM kernel is untouched; rounds still complete in mirror
order, so the dense schedule's readiness order is preserved. The stream is
folded into the epoch via `signal_done_event_` before the closing barrier.

### Knobs, capacity, validation

New: `FLUX_A2AV_MAX_RELAY_NTOKENS` (relay staging holds ALL rounds at once,
`~` the node's outbound/L; default mirrors the stage default) and
`FLUX_A2AV_RELAY_IDENTITY=1` (compile the §11 branch verbatim — byte-identical
wire, for A/B; it changes the wire layout, so set it on EVERY rank). Capacity
checks switch to chunk sums, still evaluated identically on every rank
(collective failure). All new signal slots follow the epoch discipline:
init-zero, single writer per iteration, GEQ waits, never memset.

`test/python/moe_ag_scatter/test_relay_balance_math.py` (CPU-only, no GPU/flux
needed) simulates the exact host offset math and ATen index sequences of both
wire modes across 6 topologies × seeds × skews (156 cases): recv buffers
byte-identical to §11 and to a direct dedup reference, every recv row written
exactly once, indices window-bounded, chunks balanced to ≤ 1 row, uniform-`U`
routings relocate zero rows, and the three-stream schedule is deadlock-free
over two epochs under an event-driven executor. Hardware validation on
Perlmutter is still pending, as with §11.

## 13. Layer1 hierarchical a2av combine (`GemmGroupedV2GatherRSOp`, `a2av_hier`)

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

## 14. Layer1 ports of the layer0 optimizations: eager reduce + compress

The two layer0 lessons that survived the starvation and realistic-trace
campaigns transpose onto §13's combine as follows.

### 14.1 Eager (arrival-order) destination reduce — `FLUX_A2AV_RS_EAGER=1`

§13's reduce is the last wait-on-specific-peers gate in the pipeline: per
split, W back-to-back `cuStreamWaitValue64` on one serialized reduce stream,
then a write-mode kernel — the exact shape layer0's H2/H2b starvation analysis
attributed multi-ms spin to. The eager variant (a ctor-time boolean read from
`FLUX_A2AV_RS_EAGER`; knob-off is byte-identical) deletes ALL of it: one
persistent reduce kernel per forward, launched right after the pack kernel
while the conn=1 channel holds no host wait, no front-end reduce waits at all.
Per output element a remaining-mask loop folds in any of the token's topk recv
rows whose source lane's per-split signal has fired (64-bit acquire poll +
nanosleep backoff) — accumulation in arrival order, which is the minimal real
dependency ("all contributions summed before the output row is written", not
"all sources arrived before any addition starts"). The source lane of a recv
row is recovered by binary search over `recv_cum[W+1]` (per-source rows are
contiguous, and splits slice COLUMNS, so the prefix is split-invariant — one
small host array in the args struct). Rides the reserved reduce-block SM
budget. Cost: the sum order becomes arrival-dependent (the dense ring path
already is); correctness checks are tolerance-based.

### 14.2 Token-dedup compress — ctor `a2av_hier_compress`

The transpose of §11's dispatch dedup with the roles flipped: in dispatch all
copies of a token originate on ONE rank, so dedup is local; in the combine the
k' copies of a token owned by one node are SPREAD across its ranks, so dedup
requires convergence before the wire. Design: source rank `(n, lr)` owns all
wire rows destined to rank `(tn, lr)` — same-lr end-to-end.

1. **Convergence** (new NVLink hop): each rank `(n, ls)` puts, per remote node
   `tn` and local peer `lr`, its send-panel sub-chunk destined to `(tn, lr)`
   into peer `(n, lr)`'s conv panel (`putmem_signal`, per-(ls, tn, sid)
   signal slots — nbi puts to one PE are unordered, so per-pair granularity is
   mandatory).
2. **Pre-reduce**: a persistent kernel (grid `FLUX_A2AV_RS_PRERED_BLOCKS`,
   added to `sm_margin`) spins on the L conv signals per (tn, sid), merges
   each wire row's contributing conv rows (CSR `wire_ptr`/`wire_copy`, one
   row per distinct token, token-ascending per segment), and flips a
   `wire_flags[tn * n_split + sid]` via the pack kernel's counter handshake.
3. **Wire**: the inter ladder waits the wire flag and issues ONE
   `putmem_signal` per (remote node, split) straight into the destination's
   recv panel — the §13 destination gateway hop is gone. Wire rows
   `(n,lr) → (tn,lr)` = `U[(tn,lr)][n]`, the layer0 U-matrix consumed
   TRANSPOSED (`a2av_unique_counts`, untimed host metadata).
4. **Destination**: the recv image follows the compress chunk matrix `C'`
   (own-node lanes unchanged; one remote lane per node, at the same-lr rank),
   so `recv_sig` keeps its `[W × n_split]` layout with one writer per slot and
   only L + NN − 1 lanes materialize — §11's `nseg`, transposed. The reduce
   walks the `red_ptr`/`red_row` CSR (own-node copies individually + one
   merged row per contributing remote node, positioned by the transposed
   one-cumsum); composes with 14.1 (CSR eager kernel) or the legacy per-split
   gate restricted to materialized lanes.

Byte accounting: NVLink moves the per-copy hop from the destination node
(§13's gateway forward) to the source node (convergence) — roughly unchanged;
the wire shrinks by the duplication factor `Σ chunk_remote / Σ U_remote`; one
forwarding hop leaves the wire critical path. Arithmetic moves INTO the
transport (pre-reduce costs SMs) — the inverse of dispatch dedup, whose fanout
was free CE copies; whether the wire savings pay for the SM pressure at small
budgets is the open measurement question.

**Kernel-loading constraint (root-caused 2026-08-16, first 2-node GPU runs):**
both 14.1 and 14.2 put a persistent spin kernel on the device before the
epoch's first NVSHMEM on-stream call, and NVSHMEM 3.2.5 delivers every
on-stream signal via a device kernel (`nvshmemi_signal_op_kernel` et al.).
Under `CUDA_MODULE_LOADING=LAZY` (the launch.sh default) a kernel's module
loads at its FIRST launch; a first launch enqueued behind a never-exiting
resident spin kernel never completes the load, no signal is ever produced,
and the epoch deadlocks (the §13 legacy path survives only because its lone
spin kernel, the pack, drains once the GEMM finishes). The combine therefore
preloads every kernel it launches at ctor time — `a2av_combine_preload` +
one priming NVSHMEM op per transport path in `init_buffer_once`, with the
device idle — making the op correct under either loading mode.

Executable specs, validated on CPU before any GPU run:
`test_a2av_combine_sim.py` (`simulate_compress`: set-valued payloads through
conv → pre-reduce → C' wire → CSR reduce; any double-counted or missing copy
fails regardless of accumulation order) and `test_a2av_sched_sim.py` (the full
enqueue order under pessimistic conn=1 semantics — per-rank single host FIFO,
resident kernels on their own program counters — across {hier, compress} ×
{eager, legacy} × n_split grids). On `nnodes == 1` compress degrades to plain
§13 (node-level dedup saves zero wire bytes).
