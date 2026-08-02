# Layer0 baseline data movement — annotated pass

An annotated walkthrough of `layer0_baseline_user_notes.md`, checked against
the actual code in `src/moe_ag_scatter/`. Every mechanism claim below has a
file:line reference; nothing is from memory of "how NVSHMEM code usually looks"
— which matters, because the single biggest correction in this document is
that the baseline does **not** look like typical NVSHMEM put code at all.

## Constants (used consistently throughout)

- `L` — local ranks per node (4 on Perlmutter, 8 on the AWS p4d cluster)
- `N` — number of nodes
- `R = N·L` — total ranks (the code calls this `world_size` / `W`)
- `T` — tokens in one rank's input shard (rows of `inputs_shard`)
- `H` — hidden dimension (columns of a token embedding)
- `S = T·H·sizeof(elem)` — bytes of one rank's shard
- `k` — topk
- `E` — experts owned by this rank (`ep_nexperts`)

The baseline discussed here is the `nnodes > 1` dense path of
`GemmGroupedV2AGScatterOp`: the `all_gather_all2all` routine
(`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:717`), ported from
the sm90/V3 op. (On a single node — e.g. the AWS box — the op takes a
different branch entirely: a CUDA-IPC ring `AllGatherOp`, line 2590. Same
ideas, different plumbing; this doc covers the multi-node hierarchy.)

---

## 1. The producer (input)

> the producer starts off ready, it is a tensor of token embeddings. in our
> profiling and sweep, this is the budget that each rank has. each one of
> these tokens have passed through the gating algorithm and have decided which
> topk it wants to enter/reside on.

**Mostly right, with one framing correction that matters a lot for the
baseline specifically.**

Right: the producer is `inputs_shard`, a `[T, H]` tensor of token embeddings
sitting ready in device memory on each rank when `forward` is called. Gating
has already happened; `splits` and `scatter_index` describe where every
(token, topk-slot) copy belongs.

The correction: **the baseline all-gather does not look at gating at all.**
It replicates every rank's full shard to every other rank — each rank receives
`(R−1)·S` bytes into the symmetric `input_buffer` (`[R·T, H]`), no filtering,
no topk multiplier. Gating only decides, *after* the bytes are everywhere,
which of those gathered rows the local GEMM actually reads (`M_this_ep` rows
through an index — Section 2).

So be careful mapping the sweep's "budget" onto the baseline. In the sweeps,
budget is the **pre-topk send budget** of the a2av variants — bytes that
actually get routed. In the baseline, the wire traffic is just `(R−1)·S` per
rank regardless of what gating decided; budget only shapes how much *compute*
each rank does. This asymmetry (baseline moves everything, a2av variants move
only routed copies) is the entire reason the a2av family exists.

> fundamentally, in our tests, this is passed in as pre-known metadata, but
> actually acquiring it requires a pre-requisite allgather that is not
> separately timed (and relatively cheap, since it doesn't actually move token
> bytes, just metadata of token bytes).

**Correct, and the code says so explicitly.** The harness gives every rank the
identical global `scatter_index` / `splits`, so all offsets are computed
locally; a real system would first exchange per-(source, expert) counts. See
the NOTE at `gemm_grouped_v2_ag_scatter.cc:800`. Right too that it's cheap:
it's `O(R·E)` integers vs `O(T·H)` token bytes.

---

## 2. The consumer (output side / GEMM)

> when enough rows come and fill up this tensor such that a tile can fire,
> some signaling mechanism sets off the tiling multiplication to proceed on
> these rows.

**Right effect, but invert the mental model: nothing "sets off" a tile. Tiles
are launched eagerly and *block themselves* until their rows exist.**

The grouped GEMM is launched once, on the main stream, while communication is
still in flight on the copy streams. The kernel launches more threadblocks
than tiles' worth of SM capacity (oversubscription). Each threadblock, before
touching operand A, runs this gate
(`src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h:513-541`):

1. From its M-range `[m_start, m_end]` and the per-expert, per-source-rank
   cumulative row counts (`split_accum`), it computes which **source-rank
   segments** its rows span. This is a warp ballot: lane `r` of one warp
   checks source rank `r`.
2. Each lane whose source rank overlaps the tile then **spin-waits** on a
   per-source-rank flag: `barrier_ptr[r]`, an `int32` in device memory, read
   with `cuda::atomic_ref<..., thread_scope_device>` acquire loads, until it
   equals 1 (line 537-539). The producer of that flag is the communication
   stream (Section 3). The acquire ordering is what makes the subsequent A
   reads see the shard bytes, not stale memory.
3. `__syncthreads()`, then the tile proceeds normally.

So the "signaling mechanism" is: **one 4-byte flag per source rank, written by
the copy stream when that rank's whole shard has landed, spin-read by any tile
that needs rows from that shard.** Granularity is per-source-shard, not
per-row and not per-tile.

Why doesn't spinning waste the GPU? Two reasons working together:

- The tile *schedule* is sorted so that tiles over early-arriving segments run
  first. `shift_rank_to_order` (`sort_util.h`, used at `sort_util.cu:289`)
  renumbers source ranks into exactly the arrival order of Section 3 (self →
  own-node peers in ring offset → node after mine → ...), and the sorted
  gather index groups each expert's rows in that order. Early tiles rarely
  wait at all.
- Oversubscription: while some threadblocks spin on late segments, the SM
  scheduler runs other threadblocks whose segments are ready. Spinning warps
  stall on memory, so they cost little issue bandwidth.

> we do not need to order the tiles in the way they are consumed, since the
> GEMM API allows an index input ... e.g. if i pass in [1,3,2,0] ... it will
> treat the logical row 0 as physical row 1

**Correct.** Operand A is read through `sorted_gather_index` (gather-on-load,
`gather_tensor.hpp`), so the GEMM's logical row `i` is physical row
`sorted_gather_index[i]` of the gathered buffer — no reshuffle kernel ever
runs. Symmetrically, the epilogue writes output row `i` through
`sorted_scatter_index` so the result lands in per-expert order. One nuance:
the index doesn't "fire based on arrival" — it's a static permutation computed
before the GEMM launches. Arrival-awareness lives entirely in (a) the sort
order chosen to *match* the known arrival order, and (b) the per-segment
spin-wait. There is no dynamism at runtime in the baseline: if a segment is
late, its tiles spin; nothing reorders around it.

---

## 3. The hierarchical all-gather itself

Here is where the notes and the code diverge most, so first the code's actual
timeline (`all_gather_all2all`, `gemm_grouped_v2_ag_scatter.cc:717-783`), then
the annotations. Three streams participate:

- **main stream** — flag reset, own-shard publish, world barrier, metadata
  kernels, then the GEMM
- **`cp_stream_inter_node`** — the N−1 inter-node fetches
- **`cp_stream`** — all intra-node redistribution and all flag writes

**The one non-obvious headline: this is a *pull* (get) design, not a push
(put) design.** Every transfer is `nvshmemx_getmem_on_stream` — the receiver
reaches out and reads the bytes from the source rank's symmetric buffer. There
is no `put_nbi`, no `quiet`, no signal-carrying put anywhere in the baseline.
That single fact answers most of the synchronization questions below, because
a stream-ordered *get* has a property a put never has: **when the get
completes on my stream, the data is already in my local memory.** Completion
and delivery are the same event, so plain stream ordering does almost all the
synchronization work.

Timeline (t0 = `forward` entry, after the a2av-vs-dense branch):

1. **Publish (main stream).** Reset all R flags to 0 (line 2595). Copy my own
   shard into my slot of the symmetric `input_buffer` (line 736). Then
   `nvshmemx_barrier_all_on_stream(main_stream)` (line 742) — the one
   world-wide NVSHMEM barrier. Record `ready_event`.
2. **Round 0: own node (`cp_stream`, gated on `ready_event`).** For `j = 1..L−1`,
   pull local peer `(my_local_rank + j) mod L`'s shard over NVLink
   (`getmem_on_stream`, line 768) and, after each pull, write that source
   rank's flag to 1 with `cuStreamWriteValue` (line 775). My own flag is
   written first with no copy (the `j = 0` iteration skips the get).
3. **Inter-node rounds `i = 1..N−1` (`cp_stream_inter_node`, gated on
   `ready_event` at `i = 1` only, line 746-748).** Pull the shard of the rank
   with **my local rank id** on node `(my_node + i) mod N` (line 752). Then
   `nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, ...)` (line 758) — a
   node-team barrier — and record `fetch_remote_event`.
4. **Redistribution round `i` (`cp_stream`, gated on `fetch_remote_event`).**
   Pull the other L−1 remote shards of node `(my_node + i) mod N` — not from
   the remote node, but **from my local peers over NVLink**, each of whom
   fetched exactly one (line 768 again, source = local peer). Flag each shard
   as it lands; also flag the shard I fetched myself (its data is safe because
   `cp_stream` already waited on `fetch_remote_event`).
5. **GEMM (main stream).** Metadata kernels (`calc_gather_index`, the sorts)
   ran on the main stream in parallel with all of the above. The GEMM launch
   waits on `fetch_remote_event` (line 2771), then tiles gate themselves on
   the flags as in Section 2.

Now the annotations.

> the ordering in which different ranks talk to each other is all
> pre-determined and calculated at the start.

**Right, and even stronger than stated: there is no calculation.** The
schedule is pure rank arithmetic — cyclic offsets from `(my_node,
my_local_rank)` — identical on every rank by symmetry, baked into loop bounds.
Nothing is negotiated, measured, or exchanged.

> in fact, this calculation can be done, in parallel, as the gemm tiles first
> process and sift through what is locally available on the local rank's
> producer.

**Half right; the parallelism is real but it's between the wrong pair of
things.** What runs in parallel with the communication is the *metadata* work
on the main stream — `calc_gather_index_impl`, `ag_scatter_sort_impl_v2`,
building `sorted_gather_index` / `sorted_scatter_index` / the problem
schedules (lines 2618-2672). The GEMM itself cannot "first process local
rows" before that sort finishes — every tile reads A through
`sorted_gather_index`, so the sort is a hard dependency of tile 0. The true
picture: **comm streams move bytes ∥ main stream builds indices; then the
GEMM starts and its early tiles (local + own-node rows) run while late
segments are still landing.**

> then, it schedules L-1 rounds of local transfer, precomputed in ring such
> that there is no incast in the scale-up NVLINK bandwidth.

**Right shape, two refinements.** (1) They're *pulls*: at step j, every rank
reads from peer `(me + j) mod L`, so each step is a perfect matching — every
GPU serves exactly one reader and reads exactly one peer. That's the no-incast
property: it comes from the cyclic offset, and pulling makes it self-pacing
(a slow reader delays only itself). (2) It's not L−1 rounds once — it's L−1
pulls *per node round*, N times total: once for the peers' own shards (round
0) and once per remote node to redistribute what the gateways fetched. Total
intra-node traffic per rank: `N·(L−1)·S` received.

> at the same time, an inter-node transfer is being conducted by ranks with
> their corresponding local rank id remote nodes.

**Exactly right.** Rank `(n, l)` only ever talks across nodes to ranks
`(n', l)`, same local id — L disjoint "rails", matching one NIC-ish path per
gateway pair, N−1 fetches per rank of `S` bytes each. And because
`cp_stream_inter_node` and `cp_stream` are different streams, the fetch for
node round `i+1` genuinely overlaps the intra-node redistribution of round
`i` — that pipelining is real.

> upon receiving, they can start to redistribute these tokens locally (not
> filtering or anything), and so in the ordering in which tokens arrive (all
> tokens arrive), its the local rank first, then local node ranks next, then
> remote ranks after.

**All correct.** No filtering — this is a dense all-gather; gating is applied
only by which rows the GEMM reads. Arrival order is: me, then my L−1 node
peers in ring-offset order, then node `me+1`'s L shards, then node `me+2`'s,
... And crucially, `shift_rank_to_order` gives the GEMM's row sort *exactly
this order*, so the tile schedule consumes segments in the order they land.

---

## 4. The synchronization questions, answered one by one

> if the remote puts are nbi, is there anything stopping the inter-node
> rounds from occurring? after a node issues a put_nbi does it immediately
> issue the next round of nbi?

**The premise doesn't hold in the baseline — there are no puts, nbi or
otherwise.** The question that remains is real though: what paces the
inter-node rounds? Two things:

1. **Stream order.** All N−1 gets are enqueued on the same
   `cp_stream_inter_node`; a stream executes its work in order, so get `i+1`
   cannot begin until get `i` has completed — and for a get, "completed"
   means the bytes are in my memory.
2. **The node-team barrier between rounds** (line 758). This makes the rounds
   *lockstep across the node*: my round-`i+1` fetch can't start until all L
   ranks on my node finished their round-`i` fetch. Why insist on that? The
   redistribution of round `i` reads from my peers' buffers; the barrier is
   what guarantees a peer's round-`i` shard is present before I pull it.
   Without it, `fetch_remote_event` would only prove *my* fetch finished, not
   my neighbor's.

The host CPU, by the way, enqueues the entire N-round program in one shot and
moves on — pacing is enforced entirely by stream order, events, and the team
barriers, never by the CPU waiting.

> i know that the intra-node redistribution depends on the arrival of the
> corresponding inter-node segment, but how is this signaling/synchronization
> dependency held?

By a **CUDA event bridging the two streams**, made sufficient by the team
barrier just described: per round, `cp_stream_inter_node` runs
[get → node-team barrier → record `fetch_remote_event`], and `cp_stream` runs
`cudaStreamWaitEvent(fetch_remote_event)` before its L−1 redistribution pulls
(lines 758-760). Note the layering: NVSHMEM's team barrier creates the
*cross-rank* guarantee ("everyone's segment landed"), and the CUDA event
merely carries that guarantee *across streams within one GPU*. No flags, no
signals — the receiver-side pull model means plain completion ordering is
enough.

> lastly, perhaps theres a nvshmem barrier that restricts all ranks from
> progressing. where is this installed?

Yes — exactly one, and it's at **publish time, not finish time**:
`nvshmemx_barrier_all_on_stream(main_stream)` right after each rank copies its
own shard into its symmetric buffer (line 742). Its job: nobody may *pull*
rank r's shard until rank r has *published* it. In a push design the sender
knows its own data is ready and needs no such barrier — this world barrier is
the toll the pull design pays. Consequences worth internalizing:

- It sits on the **main stream**, upstream of everything — so one straggler
  rank arriving late at `forward` stalls every rank's whole iteration, GEMM
  included.
- There is deliberately **no** world barrier at the end; completion is
  conveyed per-shard by the flags, which is the whole point (tiles start on
  early shards while late ones are still in flight).

**One more finding the notes didn't ask about but should know — where overlap
does *not* happen.** Before launching the GEMM, the main stream waits on
`fetch_remote_event` (line 2771, comment: "do not start the (SM-occupying)
GEMM before the remote fetches are issued"). Because that event was last
recorded *after the final inter-node round's get and team barrier*, on N=2 the
GEMM does not start until the entire inter-node transfer has completed. So in
this baseline, **inter-node time is overlapped only with the metadata/sort
kernels, not with GEMM compute; only the intra-node redistribution truly hides
under the GEMM.** (Contrast the a2av_hier path a few lines up, line 2748,
which gates the GEMM only on puts being *issued* — that gap is precisely one
of the things the hier variants exist to close.)

---

## 5. The timeline on one page

For rank `(n, l)`, N=2, L=4 (Perlmutter shape):

```
main stream:        [reset flags][publish own shard][world barrier]──[sort/index kernels]──────────[wait fetch_remote]──[GEMM: tiles spin on flags[r]]
                                                       │ ready_event
cp_stream:                                             ├─[pull peers l+1,l+2,l+3; flag each]───[wait ev]─[pull 3 remote shards off peers; flag each]
                                                       │                                          │ fetch_remote_event
cp_stream_inter_node:                                  └─[get shard (n+1, l)]──[node-team barrier]┘
```

- Bytes on the wire per rank: receive `S` inter-node (×(N−1) in general) +
  `(N·(L−1))·S` intra-node. Everything moves; gating moves nothing.
- Signals emitted: R flag writes into `barrier_block` (one per source shard),
  `ready_event`, `fetch_remote_event` per round, `all_gather_event` at the
  end of `cp_stream` (the epoch close waits on it, line 2783).
- Signals consumed: every GEMM threadblock acquire-spins on the flags of the
  segments its M-range spans.

---

## Follow-up Q&A (round 2)

> seems like the local nodes first must stage its tokens into the symmetric
> buffer, then enter the barrier. exiting a barrier is the guarantee that all
> nodes have it staged, therefore is ready for the remote pull. that is the
> signalling cost analogous to the put then signal if we were to approach this
> from a push perspective.

Correct, with one property worth naming: the barrier is the *collective,
all-or-nothing* version of put+signal's *incremental, point-to-point*
guarantee. A `putmem_signal` unlocks one (source → dest) payload the moment
that payload lands; the barrier unlocks all R publishes at once and charges
every rank the arrival time of the **slowest** rank. That granularity
difference — incremental unlock vs. bulk unlock priced at the straggler — is
the real cost of expressing "published" as a barrier, and it is paid before a
single inter-node byte moves.

> and then once getmem returns, that is the guarantee that says "ok, now this
> is ready for local redistribution"

This is the one repair. *My* get completing (stream-ordered completion — the
host call returns immediately; "returns" means later ops on that stream see
the data) guarantees only that **my** fetched shard is resident. But round-`i`
redistribution reads my **peers'** buffers — shards *they* fetched. My own
completion says nothing about theirs. The guarantee that actually gates
redistribution is the **node-team barrier** at line 758: "all L gateways on
this node finished round `i`." The `fetch_remote_event` that `cp_stream`
waits on is recorded *after* that barrier, which is why it is sufficient.

> which then sends off a copy write + cuwrite on the stream (analogous to
> put + signal). is this understanding correct?

Structurally yes — payload-then-flag, in that order. One directional twist:
in real put+signal, the **producer** writes both payload and signal into the
**consumer's** memory, and NVSHMEM's delivery semantics order signal after
payload across the wire. Here the **consumer** pulls the payload and then
flags **itself** (`cuStreamWriteValue` into its own `barrier_block`), with
plain stream order providing the payload-before-flag guarantee. Same shape,
but the ordering authority is the CUDA stream, not the NIC.

> on say 4 to 8 nodes, what is the computation really overlapping with?

Blunt answer: **almost none of the inter-node phase — the baseline's
compute overlap is confined to the last intra-node redistribution round plus
tail flag-spins.** Walk the N=4 timeline per rank:

```
inter stream:  [get n+1][bar]──[get n+2][bar]──[get n+3][bar]
cp_stream:     [round0: 3 local pulls]─[redist n+1]─[redist n+2]─────[redist n+3]
main stream:   [publish][world bar][sort/index kernels]──────(wait)──[GEMM      ]
```

- The N−1 inter-node rounds are serialized on one stream and lockstepped by
  the team barriers: total exposed time ≈ `(N−1)·S / BW_inter` plus N−1
  barrier latencies.
- Redistribution of round `i` runs concurrently with inter-node round `i+1`'s
  get. Since NVLink bandwidth ≫ inter-node bandwidth, each redistribution
  finishes well inside the next get. So the hierarchy's overlap is
  **comm-comm**: intra-node traffic hides under inter-node traffic.
- The GEMM waits on the *last* recording of `fetch_remote_event` (line 2771)
  — i.e., after inter-node round N−1 completes. At that moment every flag
  except round N−1's redistribution flags is already set. So the only comm
  left for compute to overlap is one round of NVLink pulls, `(L−1)·S` bytes.
  The elegant per-segment spin machinery of Section 2 is, in the multi-node
  baseline, largely **vestigial** — it earns its keep in the single-node path
  and the a2av variants, which gate the GEMM on *issue* rather than
  completion.
- Scaling doesn't rescue it: exposed inter-node time grows ∝ (N−1)·S, and
  GEMM work grows ∝ N·L·T rows — both linear in N, so the exposed fraction
  of the iteration stays roughly constant as you scale nodes. This fully
  serialized "fetch everything remote, then compute" prefix is precisely the
  gap the a2av_hier family attacks.

What *is* overlapped with the inter-node phase on the main stream: the
metadata/sort kernels (gather/scatter indices, problem schedules) — real but
microsecond-scale work.

## Follow-up Q&A (round 3)

> when you said [get n+1][bar], who participates in this bar, only ranks on
> different nodes that share the same local rank id? or all ranks?

Neither — it's **all L ranks of my own node** (`NVSHMEMX_TEAM_NODE`,
`gemm_grouped_v2_ag_scatter.cc:758`; identical in V3 at
`gemm_grouped_v3_ag_scatter.cc:200`). Picture round `i` on node `n`: all L
gateways of node `n` are fetching from the *same* remote node `n+i` — gateway
`l` grabs shard `(n+i, l)`. The team barrier is node `n` collectively saying
"we now hold all L shards of node `n+i`." Note who is *absent*: node `n+i`'s
ranks never participate — the gets are one-sided RDMA reads served by the
NIC, and the source's safety was established once and for all by the
publish-time world barrier. Each node runs its own team barrier on its own
clock; nodes are not synchronized with each other after publish.

And yes — exiting that barrier is exactly what unlocks this round's
intra-node redistribution: `fetch_remote_event` is recorded immediately after
the barrier, and `cp_stream` waits on it before its L−1 peer pulls.

> what is the relation between sort/index kernels with the streams? its
> consumed by GEMM, right? so it doesn't seem like it needs anything.

Correct: their inputs are `splits_gpu` and `scatter_index` — gating outputs
already resident on device — so they depend on **no communication at all**.
They run on the **main stream**, and their only ordering constraints are
stream-order ones: they sit *behind* the world barrier (which was enqueued on
the main stream first) and *ahead of* the GEMM (same stream), which consumes
their outputs (`sorted_gather_index`, `sorted_scatter_index`, problem
schedules). One wart: in the EP dense path without host metadata, reading
back `M_this_ep` costs a `cudaStreamSynchronize` (line 2664) — a host-side
stall between the sort kernels and the GEMM launch.

> can you draw the timeline for node count = 4 with dependency edges?

One rank's three streams, N=4. Segment lengths are schematic but ordered
(gets ≫ redists, since NVLink ≫ inter-node bandwidth); `R`/`F1..F3` are the
CUDA events; vertical/`└►` edges are `cudaStreamWaitEvent` dependencies.

```
time ─────────────────────────────────────────────────────────────────────────────────────►

main  [pub][█ world barrier █]─[sort/index]···(idle)·······························┌─►[ GEMM ▓▓▓▓▓▓▓▓▓▓ ]
                              │R                                                   │   (only n+3-segment
                              │                                                    │    tiles spin at all)
inter                         ├─[═══ get n+1 ═══][b]─[═══ get n+2 ═══][b]─[═══ get n+3 ═══][b]
                              │                    │F1                  │F2                  │F3
                              │                    │                    │                    │
cp                            └─[r0 pulls][flags]  └►[redist n+1][flags]└►[redist n+2][flags]└►[redist n+3][flags]
                                                                                                    ▲ the only comm
                                                                                                      under the GEMM
```

Reading it: `R` (recorded on main after the world barrier) releases both the
first get and round 0's local pulls. Each `[b]`+`F_i` releases round `i`'s
redistribution on `cp_stream`, while stream order lets get `i+1` begin
immediately on the inter stream — so redistribution hides under the *next*
get. The last `F3` releases the GEMM on the main stream. Everything left of
`F3` on the main stream after the sort kernels is genuinely idle compute.

> if the GEMM only overlaps with the LAST round, wow, is this for real? is
> this what's implemented in the sm90 hopper variant?

**For real, and yes — the sm90/V3 original does exactly the same thing.**
V3: `cudaStreamWaitEvent(stream, fetch_remote_event)` right before the GEMM
launch (`gemm_grouped_v3_ag_scatter.cc:424-426`), with `fetch_remote_event`
re-recorded inside the per-round loop (line 201) — so the wait binds to the
*last* recording. The V2 port mirrors it faithfully. Three observations that
make the waste intelligible rather than mysterious:

1. **It is not needed for correctness.** The per-source flags fully guard the
   GEMM's reads; launching at `R` would compute the right answer. The wait is
   a performance heuristic — keep the SM-occupying GEMM from starving the
   comm streams' kernels (barrier kernels, copy kernels) — that overshoots:
   it could have gated on the *first* round, or on nothing.
2. **The comment betrays the intent gap**: "do not start the (SM-occupying)
   GEMM before the remote fetches are *issued*" — but recycling one event
   object across rounds means the code enforces *completed*, not *issued*.
3. **At N=2 the bug is invisible**: with exactly one inter-node round,
   "first" and "last" coincide, so any 2-node validation (upstream's and
   ours) shows nothing wrong. The exposed serial prefix only opens up at
   N≥3, growing as `(N−1)·S / BW_inter`.

So the indignation is warranted: the baseline's celebrated fine-grained
overlap machinery is real but, multi-node, it is parked behind a
coarse-grained wait. This is precisely the headroom the a2av_hier family
cashes in by gating the GEMM on *issue* (line 2748) and letting the tile
spins carry all the correctness.

## Follow-up Q&A (round 4)

### Grading the round-3 answer on the one-sided read

Correct on both counts: the world barrier guards publish, and `input_buffer`
(yes — the symmetric-heap gather buffer, `[R·T, H]`, one slot per source
rank) is collision-free and immutable for the rest of the iteration. The
missing piece — "not sure what the barrier_all is" — is the **second** world
barrier, the *epoch close*: after the GEMM, the main stream waits
`all_gather_event` (= "my `cp_stream` drained; I finished every pull"), then
runs `nvshmemx_barrier_all_on_stream` (V2 line 2800, V3 line 435). Purpose:
iteration t+1's very first act is overwriting my own slot of `input_buffer`
with new tokens. Without the close barrier, a fast rank could reach t+1's
publish while a slow rank is still pulling my iteration-t bytes. So the full
safety story is: *within* an iteration, slots are write-once-then-read-only;
*across* iterations, the close barrier ensures everyone finished reading
before anyone rewrites. Two world barriers per iteration: publish (open) and
close.

### What CUDA events actually are, and how one object serves many waits

An event is a **bookmark, not a broadcast**. `cudaEventRecord(e, s)` places
the bookmark after everything currently enqueued on stream `s`;
`cudaStreamWaitEvent(s2, e)` makes `s2` block until the bookmark placed by
the **most recent record call in host-enqueue order** completes. Binding
happens when the host *enqueues* the wait, not when anything runs. The host
races through the whole per-iteration program in microseconds:

```
host order:  memset → publish → barrier_all → record R
             → cp: wait R, round-0 pulls + flags
             → for i = 1..N−1:
                 inter: wait R (i==1), get n+i, [b], record F      (re-bookmark!)
                 cp:    wait F, redist n+i pulls + flags
             → main: sort/index kernels → wait F → GEMM → wait AG → close barrier_all
```

Each `cp` wait is enqueued immediately after round `i`'s record, so it binds
to round `i`'s bookmark. The GEMM's wait is enqueued after the loop, so it
binds to the **last** bookmark. Nothing "fires" multiple times and nothing
interleaves at runtime — one event object, re-bookmarked N−1 times, each wait
frozen to whichever bookmark was newest when the host enqueued it. The three
events by name:

- `R` = `ready_event` — main stream, after the publish barrier: "my shard is
  in my slot and every rank's publish is confirmed." Waited by both copy
  streams before they touch anything.
- `F` = `fetch_remote_event` — inter stream, after each round's TEAM_NODE
  barrier: "my node holds all L shards of node n+i." Waited by `cp_stream`
  per round; last binding gates the GEMM.
- `AG` = `all_gather_event` — end of `cp_stream`: "I pulled everything and
  wrote all R flags." Waited by main before the close barrier.

### The timeline, with every event and flag write drawn in

N = 4. The GEMM's start column is aligned under round n+3's `[b]`: the third
(last) recording of `F` is what releases it — same event object as F1/F2.

```
main  [pub][█ open barrier █]R─[sort/index]····(idle — the exposed inter-node phase)·······┌►[ GEMM ▓▓▓▓▓▓▓▓ ]─[█ close barrier █]
                             │                                                             │◄── GEMM gate binds to F3,
                             │                                                             │    the LAST recording of F
inter                        ├─[══ get n+1 ══][b]F1─[══ get n+2 ══][b]F2─[══ get n+3 ══][b]F3
                             │                   │                    │                    │
cp                           └[r0 pulls ·f·f·f]  └►[redist n+1 ·f·f·f]└►[redist n+2 ·f·f·f]└►[redist n+3 ·f·f·f]AG
                                                                                              (the only comm under the GEMM)
```

Legend — code names in `gemm_grouped_v2_ag_scatter.cc`:

| symbol | code | line |
|---|---|---|
| `[█ open barrier █]` | `nvshmemx_barrier_all_on_stream(main_stream)` — publish guarantee | 742 |
| `R` | `ready_event`, recorded on main right after the open barrier | 743 |
| `[══ get ══]` | `nvshmemx_getmem_on_stream(..., cp_stream_inter_node)` — NIC-proxy pull, ~no SMs | 752 |
| `[b]` | `nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, ...)` — **SM kernel**, my node's L ranks only | 758 |
| `F1/F2/F3` | `fetch_remote_event` — ONE event object, re-recorded after every `[b]`; successive recordings | 759 |
| `└►` (inter→cp) | `cudaStreamWaitEvent(cp_stream, F)` enqueued in the same loop iteration → binds to that round | 760 |
| `[redist]` | `nvshmemx_getmem_on_stream(..., cp_stream)` from local peers — NVLink/CE, no SMs | 768 |
| `·f` | `cuStreamWriteValue(barrier_block[src], 1)` — front-end memop, one per landed shard | 775 |
| `AG` | `all_gather_event`, recorded at the end of `cp_stream` | 782 |
| `┌►` (F3→main) | `cudaStreamWaitEvent(stream, fetch_remote_event)` enqueued after the loop → binds to F3 | 2771 |
| `[ GEMM ]` | `op->run(args, ..., stream)`; resident tiles acquire-spin on `barrier_block[r]` | 2778 |
| (before close) | main waits `AG` (2784), then | |
| `[█ close barrier █]` | `nvshmemx_barrier_all_on_stream(stream)` — buffer-reuse guarantee for iteration t+1 | 2800 |

### "What fires the GEMM tiles?" — nothing launches; blocks un-block

The flags after each redistribution pull are the `barrier_block` entries:
one `int32` per source rank, set to 1 by a `cuStreamWriteValue` that stream
order places after that shard's copy. On the GEMM side, the threadblocks are
**already resident** (launched at F3): each block computed which source-rank
segments its M-range spans, and one warp — lane `r` watching source rank `r`
— acquire-spins on `barrier_block[r]`
(`ag_scatter_gemm_grouped_with_absmax.h:537`). "Firing" is nothing more than
a spinning load finally observing 1; the warp passes `__syncthreads()` and
the tile's mainloop begins. Flag-to-compute latency is one poll iteration.

### sm_margin, and why the conservative gate is deadlock-avoidance, not taste

The claim "none of this communication needs SMs" is right for the *copies*
and wrong for the *barriers* — and that asymmetry explains the whole design:

- `cuStreamWriteValue`: front-end stream memop. Zero SMs. ✔
- Inter-node `getmem_on_stream` (Slingshot/EFA): doorbell to the CPU proxy
  thread; NIC does the read. ~Zero SMs. ✔
- Intra-node `getmem_on_stream` (P2P/NVLink): copy-engine transfer. Zero
  SMs. ✔
- **`nvshmemx_barrier_on_stream` / `barrier_all_on_stream`: real kernels**
  that write and poll symmetric flags. They need an SM to run. ✘

Now the hazard. The grouped GEMM runs persistent-style: its blocks occupy
every SM (minus `sm_margin`) and a block spinning on a flag **never yields**
— there is no preemption. Suppose the GEMM launched at `R` with
`sm_margin = 0`: blocks for round-1 segments spin on flags → flags are
written by `cp_stream` after F1 → F1 requires the round-1 `[b]` **kernel** to
run → that kernel needs an SM → every SM is held by spinning GEMM blocks →
**deadlock**. A circular wait through the SM scheduler.

Seen this way, gating on the *last* F is not arbitrary: **F3 is exactly the
point after which no SM-requiring comm operation remains** on any stream —
everything left (redist pulls, flag writes) is copy-engine + front-end work.
After F3 the GEMM may hold every SM with zero progress risk. The price is the
exposed `(N−1)` serial get rounds; the deadlock-free-by-construction property
is what it buys. The alternative — launch early, reserve SMs — is exactly
what the a2av paths do: hier forwards with "front-end waits + CE puts (zero
SMs)", and hier_compress, whose forwarding *does* run SM kernels
(`index_select` gathers), **enforces `sm_margin ≥ 1`** for precisely this
reason (see the comment at `gemm_grouped_v2_ag_scatter.cc:2743-2748`). So
"wasteful" is half the story: the baseline traded overlap for a
no-reserved-SMs progress guarantee; the variants buy the overlap back by
keeping every post-gate comm op off the SMs (or paying margin when they
can't).

## Follow-up Q&A (round 5): is "wait last" real? (challenge + experiment)

> assume the index metadata calculation is extremely fast ... it will post a
> watch/wait for the event before even the first round finishes. by the time
> the first inter-node round finishes and the node_barrier is released, it
> signals F [and the GEMM proceeds].

The challenge is well-aimed: it is exactly the question of whether
`cudaStreamWaitEvent` is a **runtime rendezvous** (a posted watch that
releases when the event next fires — the model above) or a **host-enqueue
snapshot** (the wait binds to whatever the most recent `cudaEventRecord`
*host call* was at the moment the *wait's* host call was made, and later
re-records don't touch it — nor do earlier "firings" release it).

It is the snapshot. Two arguments, one semantic, one measured:

**Semantic.** The host enqueues the entire `all_gather_all2all` loop —
including all N−1 records of `fetch_remote_event` — and *returns* before it
ever reaches the GEMM-gate `cudaStreamWaitEvent`. Host enqueueing takes
microseconds and never waits for the GPU, so by the time the gate is
enqueued, the most recent record is round N−1's. GPU-side execution speed of
the sort kernels cannot change this: binding is decided by host program
order, which is fixed. The same rule explains why per-round redistribution
does NOT wait for the last round: in the loop body, record and wait are
interleaved — record F (line 759), then `cudaStreamWaitEvent(cp_stream, F)`
(line 760), *same iteration* — so each redist wait snapshots its own round's
record. One event object, one rule, different host positions. (Corollary the
challenge itself derived: if the host enqueued all records first and all
redist waits after, every redistribution would collapse behind the final
round. Correct — that is exactly what would happen.)

**Measured** (A100-SXM4-40GB, AWS p4d, 2026-07-30). Stream A: [sleep][record
E][sleep][record E]; the host then immediately — in µs, long before the first
sleep finished in GPU time — enqueued `B.wait_event(E)` + a timing marker on
stream B. If waits were runtime rendezvous, the marker runs after the first
record (~1 sleep). Result: first record fired at 2.4 ms, marker ran at 4.7 ms
= after the *second* sleep + record. The wait ignored the first firing and
bound to the last host-side record. Script:
`event_binding_test.py` (torch streams/events + `torch.cuda._sleep`; ~30
lines, easily re-created from the description above).

So the exposed serial prefix of Section "round 3" stands: the GEMM gate binds
to the completion of the final inter-node round, regardless of how fast the
main stream gets there.

## Comprehension questions

Answer these before moving on to the a2av variants — they test the two ideas
that actually transfer.

1. **Pull vs push.** The a2av_hier variant replaces gets with
   `putmem_signal`-style *pushes*. The baseline needed the world barrier
   because a puller can't know when the source has published. What is the
   analogous "can't know" problem on the *receiver* side of a push design,
   and why does that force the per-source signal to carry an epoch/run-id
   (see `run_id_`, `gemm_grouped_v2_ag_scatter.cc:311`) instead of the
   baseline's memset-to-0-then-set-to-1 flags?

2. **The lockstep cost.** Suppose on Perlmutter one gateway's Slingshot path
   for round 1 is 3× slower than its three node-peers' paths. Trace through
   the round-`i` machinery of Section 4: which specific operations on which
   *other* ranks stall, and would removing the node-team barrier (keeping
   everything else) be correct, incorrect, or correct-but-slower? Why?
