# COMET MoE Layer 0 on SM80 — The Communication Pattern, and Why It Is Shaped That Way

**Scope**: `src/moe_ag_scatter` (AllGather + Scatter + Grouped GEMM), the V2/SM80 code path
that runs on Perlmutter's 4×A100 nodes. Companion doc:
[`comet_layer1_communication_patterns.md`](comet_layer1_communication_patterns.md).

This walkthrough is written distributed-systems-first: before any CUTLASS detail, it pins
down **where the data is born, where it must die, which buffers it passes through, who owns
each buffer, and which signal permits each hop** — and, for every hop, *why the movement is
structured that way* rather than some other way. The COMET paper's two optimizations
(**decompose** the shared tensor, **reschedule** the computation) fall out of this framing:
the shared tensor's decomposition dimension is chosen so that the *communication grain* and
the *computation grain* can interleave, and the reschedule makes the computation order equal
the data **arrival** order.

---

## 1. The goal of the data movement

Setting: `world_size` ranks; layer-0 expert weights `w0` are sharded two ways —
expert-parallel (each rank group owns `nexperts/ep_size` experts) and tensor-parallel
(each expert's `[ffn_hidden, hidden]` weight is **column-sharded**: rank holds
`N = ffn_hidden/ffn_tp_size` columns, but the **full K = hidden** reduction dimension).

Consequence, and the reason this layer is an *all-gather* and not a token-routed
all-to-all: because every TP rank holds a full-K *column slice* of every one of its
experts, **every token routed to an expert must be seen in full by every TP rank holding
that expert**. There is no way to send a token to "the one rank that owns its expert" —
under TP, they all do, for their slice of N. So:

- **Birth of the data**: `inputs_shard` — `[ntokens_per_rank, hidden]` activations produced
  by the previous (attention) layer on each rank. Ordinary local torch CUDA memory. This is
  the only tensor that ever crosses a rank boundary in layer 0.
- **Death of the data**: `outputs[g]` — `[M_this_ep, N]` per-rank grouped-GEMM results
  (`M_this_ep = ntokens * topk` for EP=1), living in ordinary local torch memory,
  row-scattered into the order layer 1 expects. Purely local; no communication on the
  output side.
- **The movement in between**: every rank's shard must land in a **world-readable staging
  buffer** (`input_buffer`, the paper's *shared tensor*) on every rank, and the grouped GEMM
  must be able to start consuming shard *s* the moment shard *s* has landed — not when the
  whole all-gather has finished.

```
rank r:  inputs_shard[r]  ── local copy ──►  input_buffer[segment r]      (on every rank)
         peers' shards    ── AG transport ─► input_buffer[segments ≠ r]   + flag[s] := 1
                                                     │
                              GEMM tile waits flag[s]│ for every segment s the tile touches
                                                     ▼
         input_buffer ── gather_A rows ─► grouped GEMM ── scatter_D rows ─► outputs (local)
```

The **decompose** step of the paper is visible already in this picture: the shared tensor
(`input_buffer`) is decomposed **along M** (the token dimension) into `world_size`
per-source-rank segments, because rows are independent for a GEMM whose reduction dimension
is K — a tile of output rows only needs the token rows it covers, so a *segment* of tokens
is a valid unit of readiness. N or K could not be the decompose dimension here: every
output element needs all of K, and N lives entirely in the (local, resident) weights.

---

## 2. Buffer inventory: who allocates what, where it lives, who can touch it

All allocation happens **once, in the op constructor** (`GemmGroupedV2AGScatterOpImpl`,
`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:283-332`), sized by `max_ntokens` —
forward passes then only carve views. This is deliberate: symmetric/IPC allocations are
collective, slow operations that must not sit on the hot path, and NVSHMEM symmetric
addresses must be identical across ranks, which is only guaranteed if all PEs allocate in
lockstep.

```cpp
// gemm_grouped_v2_ag_scatter.cc:322-328
if (nnodes == 1) {
  ag_op.emplace(this->tp_group, 1, max_ntokens, hidden, input_dtype);
} else {
  FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
  this->input_buffer = nvshmem_create_tensor({max_ntokens, hidden}, input_dtype);
  this->barrier_block.reset(pad_to(world_size * (int64_t)sizeof(int), (int64_t)128));
}
```

| Buffer | Allocated at | Allocator / handle | Resides | Who writes | Who reads | Contents |
|---|---|---|---|---|---|---|
| `inputs_shard` | caller (per fwd) | torch CUDA caching allocator | local HBM | previous layer | AG copy source | this rank's tokens |
| `input_buffer` (single-node) | `AllGatherOp` ctor → `all_gather_op.cc:231` | `flux_create_tensor_list` → NVSHMEM symmetric heap **or** `cudaMalloc`+CUDA IPC (`flux_shm.cc:434-452`) | local HBM, **P2P-mapped into every local peer** | all ranks (mode-dependent) | local GEMM | all ranks' shards, segment-major |
| `input_buffer` (multi-node) | op ctor `:326` | `nvshmem_malloc` via `nvshmem_create_tensor` (`flux_shm.cc:61-92`), wrapped in a torch tensor by `at::from_blob` | NVSHMEM symmetric heap (same offset on every PE) | this rank only (pull model) | local GEMM | all ranks' shards |
| `barrier_buffers` (single-node) | `all_gather_op.cc:242-250` | `flux_create_tensor_list`, 64 ints (`kNumSignals`) | local HBM, P2P-mapped | self (a2a/pull) or peers (ring push) | GEMM spin-wait | per-segment ready flags |
| `barrier_block` (multi-node) | op ctor `:327` | `cutlass::DeviceAllocation` (plain `cudaMalloc`) | local HBM, **private** | this rank (`CUStreamWriteValue`) | local GEMM | per-source-rank ready flags |
| `sync_buffers` | `all_gather_op.cc:294-317` | `flux_create_tensor_list` | P2P-mapped | all | all | rendezvous counters for `flux_barrier_all_on_stream` |
| `gather_index`, `sorted_*` | per fwd, `gemm_grouped_v2_ag_scatter.cc:534-545` | torch CUDA | local | sort kernels | GEMM iterators | index maps (§5) |
| `problem_schedules_gpu`, `workspace_buffer` | per fwd / lazy | torch CUDA | local | prepare kernel | GEMM problem visitor | tile schedule (§6) |
| `outputs[g]` | per fwd `:600-607` | torch CUDA | local | GEMM epilogue | layer 1 | scattered results |

Three allocation regimes appear, and the choice is never arbitrary:

1. **`flux_create_tensor_list`** (`src/ths_op/flux_shm.cc:434-452`) returns a *vector* of
   tensors — index `i` is a view of **rank i's** buffer, usable as a plain pointer because
   intra-node the NVSHMEM heap (or `cudaIpcOpenMemHandle`, `flux_shm.cc:306-337`) maps peer
   memory into the local address space over NVLink. This is what makes the single-node AG a
   set of ordinary `cudaMemcpyAsync` calls: peer HBM is just an address.
2. **`nvshmem_create_tensor`** returns *one* tensor on the symmetric heap. There is no list
   because remote access goes through NVSHMEM verbs (`nvshmemx_getmem_on_stream`) with a
   *(same local address, remote PE)* pair — the symmetric heap's defining property. This is
   the only option across nodes, where no load/store mapping exists (Slingshot, not NVLink).
3. **`cutlass::DeviceAllocation`** for the multi-node flags — not a torch tensor, for a
   documented reason (`gemm_grouped_v2_ag_scatter.cc:248-252`): under
   `expandable_segments:True` the torch allocator hands out virtual addresses that
   `cuStreamWriteValue32` rejects; flags written by the CUDA driver's stream-ordered write
   need a real device VA.

Also note what is **absent**: there is no world-visible output buffer, no gathered-`A`
materialization (`sorted mat A` never exists in memory — §5), and multi-node flags are
private. Every buffer is exactly as shared as its access pattern requires, no more.

---

## 3. The signal architecture, before the transport

Every hop below is gated by exactly one of these mechanisms; it is worth seeing the whole
set at once, because the *choice* of mechanism encodes who-waits-for-whom:

| Mechanism | Granularity | Producer side | Consumer side | Why this one |
|---|---|---|---|---|
| per-segment **ready flag** (int, 0/1) | one source rank's shard | `CUStreamWriteValue32` enqueued *after* the copy on the same stream (`all_gather_op.cc:509-517`, `gemm_grouped_v2_ag_scatter.cc:425-429`) | GEMM warp spin-load acquire (§6) | fine-grained: unblocks *tiles*, not the kernel; stream-ordered driver write costs no SM and cannot reorder ahead of the copy |
| `cudaEvent` between streams | whole phase | `cudaEventRecord` | `cudaStreamWaitEvent` | orders host-enqueued phases (copy stream vs GEMM stream) without touching device flags |
| `flux_barrier_all_on_stream` / `nvshmemx_barrier_all_on_stream` | whole group | everyone | everyone | epoch delimiter: separates *this* iteration's buffer use from the previous/next one (§4.3, §4.4) |
| `nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, …)` | local node | local peers | local peers | closes the network stage of the hierarchical AG before intra-node redistribution (§4.4) |

The division of labor is the classic one: **barriers manage buffer lifetime (epochs), flags
manage data readiness (progress)**. Flags alone cannot be safely reset without a fence that
proves nobody still polls them; barriers alone would serialize compute behind the slowest
rank. Layer 0 uses both, each exactly once per iteration.

---

## 4. The transport: three topologies, one contract

The contract every transport variant must satisfy — this is the interface between the
communication layer and the GEMM, and the concrete form of the paper's *decompose*:

> When `barrier[s] == 1` on rank r, source rank s's full token shard is resident and
> readable in `input_buffer` segment s on rank r, and the write is visible to loads that
> observe the flag with acquire semantics.

Everything else (push vs. pull, ring vs. all-to-all, one level vs. two) is a topology-
dependent strategy for making flags flip **as early and as evenly spaced as possible** —
because each flag flip releases a batch of GEMM tiles, spacing determines overlap quality.

### 4.1 Launch: the forward path forks the streams

`forward_impl`, `gemm_grouped_v2_ag_scatter.cc:507-521` — Step 2 launches communication
*before* any index computation or GEMM setup, on dedicated streams:

```cpp
if (nnodes == 1) {
  CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
  CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream, this->ready_event));
  ag_op->run(inputs_shard, c10::nullopt, opt, this->cp_stream);
} else {
  CUDA_CHECK(cudaMemsetAsync(barrier_block.get(), 0, barrier_block.bytes(), stream));
  this->all_gather_all2all(inputs_shard);
}
```

**Why a separate `cp_stream`**: the copies are executed by the GPU's **copy engines**
(`cudaMemcpyAsync` peer-to-peer), not SMs. On their own stream they proceed in parallel
with every kernel the main stream launches afterwards (sort kernels, workspace prep, then
the GEMM itself). This is the hardware-level foundation of the whole overlap story on
Ampere: *communication consumes DMA engines and NVLink; computation consumes SMs; the two
never contend for the same resource.* The `ready_event` handoff just guarantees the copies
don't read `inputs_shard` before the producer kernel on the main stream finished writing it.

### 4.2 Single node, epoch open: local copy + flag reset

`AllGatherOp::run` (`src/coll/ths_op/all_gather_op.cc:352-371`) first stages the local
shard and re-arms the flags — `copy_local_and_sync_with_cudaMemcpyAsync`
(`all_gather_op.cc:406-453`), whose ordering is a small protocol in itself:

```cpp
barrier_async(stream);                       // (1) join epoch
CUDA_CHECK(cudaMemcpyAsync(                  // (2) my shard -> my segment
    ptr_offset(input_buffer_ptr, this->rank * chunk_size),
    input_ptr, chunk_size, cudaMemcpyDefault, stream));
...
CUDA_CHECK(cudaMemsetAsync(this->barrier_buffer.data_ptr(), 0, ..., stream));  // (3) reset flags
for (int j = 0; j < SPLIT; ++j) {
  set_ready(this->rank, this->rank, j, stream);                                // (4) own flag := 1
}
barrier_async(stream);                       // (5) epoch open
```

Why each step exists:
1. Barrier (1): a peer from the *previous* forward may still be reading my `input_buffer`
   or (in ring mode) still writing my flags. I must not overwrite either until everyone has
   left the previous epoch.
2. Copy (2): local shard into my own segment — after this, my own data is "gathered."
3. Reset (3): flags are reused across iterations; they must return to 0 exactly once per
   epoch, and only inside the barrier-protected window.
4. Set-own (4): my own segment is trivially ready. `set_ready` is `CUStreamWriteValue32`
   on the flag word (`all_gather_op.cc:509-517`) — stream-ordered behind the copy, so the
   flag cannot become visible before the data.
5. Barrier (5): nobody may begin cross-rank copies (which set flags, possibly on *my*
   array in push mode) until *everyone* has finished resetting — otherwise a fast peer's
   `flag := 1` could be erased by my late `memset 0`, deadlocking its consumer.

After this function, `local_prepare_event` is recorded (`all_gather_op.cc:371`). The GEMM
launch waits on precisely this event (`gemm_grouped_v2_ag_scatter.cc:644-645`) — not on the
whole all-gather. **Why**: the GEMM may start as soon as (a) flags are armed and correct,
and (b) at least the local segment is ready; everything later is handled per-tile by flag
waits. Waiting for less would race the flag reset; waiting for more would forfeit overlap.

### 4.3 Single node, steady state: pull-all-to-all or push-ring

Two shapes are provided (`AGRingMode`, selected at `all_gather_op.cc:373-403`), because the
right pattern is a function of the interconnect, not of the algorithm:

**All-to-all pull** (`copy_all_to_all`, `all_gather_op.cc:529-574`) — used on NVLink
full-mesh machines like Perlmutter's A100 nodes:

```cpp
for (int i = rank + 1; i < (world_size + rank); ++i) {
  auto id = i % this->world_size;
  CUDA_CHECK(cudaMemcpyAsync(
      ptr_offset(this->input_ptrs[rank], id * chunk_size + split_offset),  // my buffer, seg id
      ptr_offset(this->input_ptrs[id],   id * chunk_size + split_offset),  // peer id's own shard
      split_chunk_size, cudaMemcpyDefault, stream));
  set_ready(this->rank, id, j, stream);          // flag on MY array
}
```

The pattern: I *pull* each peer's shard out of the peer's buffer into mine, starting with
rank+1 and wrapping. Why pull, and why this order?
- **Single-writer discipline**: only I ever write my `input_buffer` and my flags. No
  cross-rank write ordering to reason about; a flag on my array is stream-ordered behind
  the copy that filled the segment, end of proof.
- **Load spreading**: every rank starts pulling from a *different* peer (`rank+1`), so at
  any instant each source buffer is being read by ~one peer instead of all of them
  hammering rank 0 first. On a full-mesh NVLink fabric all these transfers ride disjoint
  links concurrently.
- **Arrival order is known**: rank r receives segments in the order r+1, r+2, … — exactly
  the order `shift_rank_to_order` (§5) assumes when it schedules computation. The transport
  and the scheduler agree on one clock.

**Ring push** (`copy_ring_push_1d`, `all_gather_op.cc:617-653`) — for PCIe or otherwise
non-full-mesh boxes (plus a NUMA-aware 2D variant, `copy_ring_push_2d_pcie:655-704`):

```cpp
int to_rank = (this->rank - 1 + this->world_size) % this->world_size;
for (int i = 0; i < this->world_size - 1; i++) {
  int send_segment = (this->rank + i) % this->world_size;
  if (i != 0) { wait_ready(this->rank, send_segment, j, stream); }  // wait for it to arrive
  CUDA_CHECK(cudaMemcpyAsync(  /* my copy of send_segment -> to_rank's buffer */ ));
  set_ready(to_rank, send_segment, j, stream);   // flag on the RECEIVER's array
}
```

Here each rank has exactly **one** outgoing neighbor: segments hop around the ring, each
rank forwarding what it received last step (`wait_ready` = `CUStreamWaitValue32` on my own
flag, which my upstream neighbor set). Why a ring when the mesh pull exists? Because on
PCIe every transfer shares one host bridge: an all-to-all would congest it with
`world_size²` flows, while the ring keeps exactly one flow per link direction at all times
— maximum sustained bandwidth on a bandwidth-poor fabric, at the cost of higher latency for
the farthest segment. The flags are the same contract; only who-writes-them changes
(sender writes receiver's flag, again stream-ordered behind the payload).

The trade is the classic one: **all-to-all optimizes time-to-first-byte and evenness of
flag arrivals (best for overlap); ring optimizes link utilization (best when the fabric is
the bottleneck)**. Both flip `world_size` flags spaced roughly one shard-copy apart —
which is exactly the pacing the rescheduled GEMM (§5) is built to consume.

### 4.4 Multi-node: a two-level pattern shaped by the bandwidth cliff

Across nodes there is no load/store path, and per-node network bandwidth (one Slingshot
NIC-set shared by 4 GPUs) is an order of magnitude below intra-node NVLink. The port
(`all_gather_all2all`, `gemm_grouped_v2_ag_scatter.cc:366-433`, brought over from the sm90
V3 op) therefore never sends the same bytes over the wire twice:

> **Each remote shard crosses the network exactly once per destination node** — fetched by
> the one local GPU with the *same local rank* as the shard's owner — and is then
> redistributed to node-mates over NVLink.

```cpp
for (int node_idx = dist_env.node_idx, i = 0; i < dist_env.nnodes;
     ++i, node_idx = (node_idx + 1) % dist_env.nnodes) {
  if (node_idx == dist_env.node_idx) {
    // (a) own node first: local shard -> my symmetric segment, then a global rendezvous
    CUDA_CHECK(cudaMemcpyAsync(shard_input.data(), inputs_shard.data_ptr(), ..., main_stream));
    nvshmemx_barrier_all_on_stream(main_stream);                                   // :392
    ...
  } else {
    // (b) network stage: fetch my same-local-rank counterpart's shard from node `node_idx`
    int src_rank = dist_env.local_rank_to_global_rank(dist_env.local_rank, node_idx);
    nvshmemx_getmem_on_stream(shard_input.data(), shard_input.data(),               // :402
                              shard_input.size(), src_rank, this->cp_stream_inter_node);
    nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, this->cp_stream_inter_node);     // :408
    ...
  }
  // (c) NVLink stage: collect the other local ranks' fetched segments, flag each
  for (int local_rank = dist_env.local_rank, j = 0; j < dist_env.local_world_size; ...) {
    int src_rank = dist_env.local_rank_to_global_rank(local_rank, node_idx);
    if (local_rank != dist_env.local_rank) {
      nvshmemx_getmem_on_stream(..., /*pe=*/local_rank_global, this->cp_stream);    // :418
    }
    CU_CHECK(CUStreamWriteValue(this->cp_stream,
        (CUdeviceptr)(ptr_offset(barrier_block.get(), src_rank * sizeof(int))), 1,  // :425
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }
}
```

Reading the pattern:

- **Same-address `getmem`** (`shard_input.data()` as both src and dst): the symmetric heap
  guarantees segment layouts coincide across PEs, so "fetch rank s's copy of segment s"
  needs no address exchange — the address *is* the protocol. This is why the multi-node
  buffer had to be `nvshmem_malloc`ed.
- **Same-local-rank pairing** (b): GPU ℓ on node A always fetches from GPU ℓ on node B.
  With one flow per GPU, the node's `local_world_size` network flows are all distinct and
  can be spread across NIC rails; no GPU fetches two remote shards for the same node while
  another fetches none.
- **The node-team barrier at `:408`** closes stage (b) before stage (c) may flag anything:
  segment (node g, local ℓ) only exists on *my* node inside GPU ℓ's buffer after ℓ's
  network fetch completes — my NVLink `getmem` from ℓ reads that buffer, so all local peers
  must have finished their network stage first. A *node* team barrier (not global) is the
  minimal sufficient fence: other nodes' progress is irrelevant to this hop.
- **Order of the outer loop: own node first, then node_idx+1, +2, …** and inner loop own
  local rank first — this is `shift_rank_to_order` (§5) enacted by the transport. Again the
  transport defines the arrival clock the scheduler compiles against.
- **Flags stay private** (`barrier_block`, plain `cudaMalloc`): in this pull-only design
  no peer ever needs to signal me — my own `cp_stream` knows when each segment's last copy
  was enqueued. Private flags mean the spin-wait in the GEMM (§6) polls local HBM/L2, never
  generating NVLink or network traffic while waiting.

Two rendezvous bracket the epoch, mirroring §4.2's reasoning at global scale:
`nvshmemx_barrier_all_on_stream` at `:392` (nobody `getmem`s a shard that hasn't been
staged — the *only* global synchronization on the critical path, and it doubles as the
previous-epoch close) and at `:662` after the GEMM (nobody overwrites `input_buffer` for
iteration k+1 while a straggler node still reads it for iteration k).

Finally the GEMM launch gate for this path (`gemm_grouped_v2_ag_scatter.cc:646-649`):

```cpp
// do not start the (SM-occupying) GEMM before the remote fetches are issued
CUDA_CHECK(cudaStreamWaitEvent(stream, this->fetch_remote_event));
```

`fetch_remote_event` is recorded on the inter-node stream right after the *first* remote
node's fetch + team barrier (`:409`). The GEMM is a persistent kernel that will occupy
every available SM and spin; letting it launch before the network stage is en-route risks
the spin-wait shadowing the very work it waits for. Waiting for "first remote fetch
issued" — not "all fetches done" — is again the minimal gate that preserves overlap.

---

## 5. Reschedule: computation order = arrival order

The transport delivers segments in a known order. The *reschedule* optimization makes the
grouped GEMM consume tiles in that same order. Three pieces cooperate, all driven by index
metadata computed on the fly each forward (`gemm_grouped_v2_ag_scatter.cc:523-577`) — cheap
GPU kernels running on the main stream *while the copy engines gather*.

**(i) Sort tokens by (source rank, expert)** — the intent is stated in
`src/moe_ag_scatter/sort_util.h:49-59`:

```
// The original computing flow:
//   input (shard) -> (ag) -> input (full) -> (scatter) -> mat A -> (gemm) -> mat D
// We sort matrix A so that the dependant data from input (shard) is as contiguous as possible.
// The new flow is:
//   input (shard) -> (ag) -> input (full) -> (scatter&sort) -> sorted mat A
//    -> (gemm) -> sorted mat D -> (scatter) -> mat D
// The original gemm is #nexperts problems, sort the tokens by a paired key:
// (the rank it is from, expert id), constructing #nexperts * #tp_size new problems.
```

After the sort, each expert's M range is partitioned into contiguous **per-source-rank
runs** (`sorted_splits_cumsum` holds the run boundaries). A tile of M rows now depends on
one — or at a boundary, two — source segments, instead of a random mix of all of them.
Without the sort, *every* tile would depend on *every* flag and decompose-along-M would buy
nothing: the sort is what turns the physical decomposition of `input_buffer` into a usable
logical decomposition of the GEMM.

Crucially, **`sorted mat A` is never materialized**. The sort produces `sorted_gather_index`
(sorted row → row of `input_buffer`) and `sorted_scatter_index` (sorted row → row of final
`mat D`), and the GEMM's A-iterator and D-iterator apply them on the fly — see the pointer
setup in `prepare_workspace_kernel` (`src/moe_ag_scatter/workspace_util.cu:212-231`):

```cpp
ptr_A[i] = args.input;                       // every problem reads the SAME shared tensor
...
gather_A[i]  = args.gather_A  + M_acc;       // per-problem slice of sorted_gather_index
scatter_D[i] = args.scatter_D + M_acc;       // per-problem slice of sorted_scatter_index
```

Why not materialize? A gathered copy of A would (a) double memory traffic on a
bandwidth-bound layer, and (b) — fatally for overlap — introduce a *serial* gather pass
that could not begin until segments arrive, recreating the dependency the whole design
removes. Indexed iteration lets the arrival wait migrate into the GEMM at tile grain.

**(ii) Rank order → arrival order** (`sort_util.h:172-180`):

```cpp
CUTLASS_HOST_DEVICE int shift_rank_to_order(int rank, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(rank);
  int node_idx_shift  = (node_idx  - dist_env.node_idx  + dist_env.nnodes) % dist_env.nnodes;
  int local_rank_shift = (local_rank - dist_env.local_rank + dist_env.local_world_size)
                         % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_shift, node_idx_shift);
}
```

This maps a source rank to its *arrival stage* on this rank: own rank → 0 (already local),
node-mates next, other nodes in ring order. It is the scheduler-side mirror of the loops in
§4.3/§4.4 — the same circular shifts the transports iterate in. Every rank computes a
*different* schedule, because every rank sees a different arrival order.

**(iii) Stage-tag every tile, then stripe** — `calc_sorted_problem_schedule_v2`
(`workspace_util.cu:35-103`) assigns each M-tile of each expert the stage of the *latest*
segment it touches (`get_stage_for_tile`: ballot over the segments the tile's `[m_start,
m_end]` spans, `max` of their stages — a boundary tile straddling segments from stage 1 and
stage 3 must wait for stage 3, so it *is* stage-3 work). Then `fill_problem_info`
(`workspace_util.cu:105-159`) flattens the stage-ordered tile list and deals it
**round-robin across the persistent threadblocks**:

```cpp
int tid   = i % threadblock_count;
int index = i / threadblock_count;
auto &info = problem_info[tid * tiles_per_tb + index];
```

Why round-robin instead of contiguous chunks: the tile list is sorted by stage, so dealing
it like cards guarantees *every* threadblock's first tile is a stage-0 (local-data) tile,
its next tiles come from the earliest stages, and no threadblock gets stuck holding only
late-stage work while SMs idle. All 108 SMs start computing at t=0 on data that never
crossed a wire; flag waits happen only when a threadblock genuinely runs ahead of the
transport.

---

## 6. The consume moment: a warp-cooperative flag wait per tile

The end of every data path is inside the persistent grouped-GEMM kernel,
`src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h:423-441` — the only
place in layer 0 where compute blocks on communication, scoped to exactly the bytes the
tile needs:

```cpp
int m_start = tile_idx_m * Mma::Shape::kM;
int m_end   = min((tile_idx_m + 1) * Mma::Shape::kM, problem_size.m()) - 1;
int *split_accum = params.split_tp_accum_ptr + params.world_size * (problem_idx % params.nexperts_ep);
int segment_start = __ffs(__ballot_sync(0xffffffff,
    lane_idx < params.world_size ? (m_start < split_accum[lane_idx]) : false)) - 1;
int segment_end   = __ffs(__ballot_sync(0xffffffff,
    lane_idx < params.world_size ? (m_end   < split_accum[lane_idx]) : false)) - 1;
if (lane_idx >= segment_start && lane_idx <= segment_end) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> barrier(params.barrier_ptr[lane_idx]);
  while (barrier.load(cuda::memory_order_acquire) != 1) { }
}
__syncthreads();
```

Mechanics worth naming:
- The two warp ballots binary-search the per-rank cumulative-sum table
  (`sorted_splits_cumsum` from §5) in two instructions: each lane tests one segment
  boundary, `__ffs` of the ballot is the first segment containing the row. One lane then
  polls one flag — a tile spanning segments 2..3 waits on exactly flags 2 and 3, nothing
  else.
- `memory_order_acquire` on the flag load pairs with the stream-ordered flag write
  (§4): once the flag reads 1, the segment's token rows are visible to this SM's
  subsequent gather loads. This pairing is the entire correctness argument for
  flag-gated consumption.
- Because tiles were stage-ordered (§5), in the steady state this loop *spins zero times*
  — the flag flipped while the threadblock was computing earlier tiles. The wait exists as
  a safety valve, not as the expected path.

After the mainloop, the epilogue scatters rows of the result through `scatter_D` directly
into `outputs[g]` — final resting place, local memory, no further movement. Layer 0's
communication is entirely front-loaded on the input side.

---

## 7. The complete timeline (single node, 4×A100)

```
main stream               cp_stream (copy engines)          GEMM view
───────────               ────────────────────────          ─────────
ready_event ─────────────► barrier_all (epoch join)
sort/gather-index kernels   local shard -> seg[r]; flags:=0; flag[r]:=1
prepare_workspace kernel    barrier_all (epoch open)
                            [local_prepare_event] ──────────► GEMM launches (:644)
                            pull seg[r+1]; flag[r+1]:=1        stage-0 tiles (local rows)
                            pull seg[r+2]; flag[r+2]:=1        stage-1 tiles (acquire flag r+1)
                            ...                                ...
wait all_gather_event ◄──── [all_gather_event]                 last-stage tiles
(next op may reuse buffers)                                    epilogue scatter_D -> outputs
```

Every arrow is one of the §3 mechanisms; every box on the left/middle costs no SMs; the GEMM
column never waits except on the acquire loads. That is the paper's layer-0 claim
implemented: decompose made segments independently consumable, reschedule made them
consumed in arrival order, and the transport was *shaped* (pull order, ring order,
same-local-rank hierarchy) so that the arrival order is a schedule worth compiling against.

## TL;DR mapping

| Paper concept | Distributed-systems decision | Code |
|---|---|---|
| shared tensor | world-readable `input_buffer`, allocated once: IPC/NVSHMEM list intra-node, symmetric heap inter-node | `all_gather_op.cc:231`, `gemm_grouped_v2_ag_scatter.cc:326`, `flux_shm.cc:61-92,306-337` |
| decompose along M | ready unit = per-source-rank segment + one flag; flags private to the consumer | `all_gather_op.cc:242-250,509-517`; `gemm_grouped_v2_ag_scatter.cc:327,425-429` |
| comm/comp resource split | copy engines + dedicated streams vs. persistent GEMM on SMs | `gemm_grouped_v2_ag_scatter.cc:511-521,644-649` |
| topology-shaped transport | pull-a2a (NVLink mesh) / push-ring (PCIe) / two-level same-local-rank fetch + NVLink redistribution (multi-node) | `all_gather_op.cc:529-574,617-653`; `gemm_grouped_v2_ag_scatter.cc:366-433` |
| reschedule | token sort by (source rank, expert); stage = arrival order via `shift_rank_to_order`; boundary tiles deferred; round-robin striping | `sort_util.h:49-59,172-180`; `workspace_util.cu:35-103,105-159` |
| fine-grained consume | per-tile warp-ballot flag wait, acquire load; gather_A/scatter_D iterators instead of materialized A/D | `ag_scatter_gemm_grouped_with_absmax.h:423-441`; `workspace_util.cu:212-231` |
| epoch management | barrier_all brackets: reset-flags window at open, buffer-reuse fence at close | `all_gather_op.cc:417,452`; `gemm_grouped_v2_ag_scatter.cc:392,657-663` |

**Run it** (from CLAUDE.md):

```bash
source ./module.sh
salloc --qos interactive -C gpu --account m4243_g            # single node
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py

salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30   # two nodes
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
```
