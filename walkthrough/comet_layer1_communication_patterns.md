# COMET MoE Layer 1 on SM80 — The Communication Pattern, and Why It Is Shaped That Way

**Scope**: `src/moe_gather_rs` (Grouped GEMM + Gather + top-k Reduce + ReduceScatter), the
V2/SM80 path on Perlmutter's 4×A100 nodes, including the multi-node port. Companion docs:
[`comet_layer0_communication_patterns.md`](comet_layer0_communication_patterns.md), and
[`comet_layer1_multinode_design_provenance.md`](comet_layer1_multinode_design_provenance.md)
for where each piece of the multi-node design came from (upstream shipped none for this layer).

Layer 0's communication was on the *input* side: gather tokens, then compute. Layer 1 is
the mirror image — compute first, then a chain of reductions and a scatter of ownership —
and the mirror flips every design decision. The shared tensor is now the **GEMM output**;
the consumer is a **communication kernel** instead of a GEMM; readiness flows
producer→consumer through *counters* instead of transport→GEMM through per-segment flags;
and the decomposition dimension is **N**, not M — forced, as we'll see, by what the
consumer must do with the rows.

---

## 1. The goal of the data movement

Setting after layer 0 + activation: each rank holds `inputs[g]` — `[M_this_ep, K]` expert
FFN intermediates in **expert-sorted row order** (`M_this_ep = ntokens·topk` for EP=1),
where `K = ffn_hidden / ffn_tp_size` is this rank's TP slice. Layer-1 weights `w1` are
`[E, N, K]` with **K row-sharded** across TP ranks and `N = hidden` full.

Because K is the *reduction* dimension and each rank holds only a K-slice, each rank's
grouped GEMM produces a **partial sum** of the true output — every element, on every rank,
is incomplete until summed across all `world_size` ranks. On top of that, three more
transformations stand between the GEMM output and the layer's contract:

1. **Gather**: GEMM rows are expert-sorted; output rows are token-ordered. `routing_idx`
   maps token-order back to expert-order rows.
2. **top-k combine**: each token has `topk` rows (one per expert it visited) that must be
   summed.
3. **Scatter of ownership**: token t belongs to exactly one rank (segment layout inherited
   from layer 0's all-gather); only that rank keeps the final row.

So:

- **Birth**: `gemm_outs[g]` — `[M_this_ep, N]` per-rank partial results, plain local torch
  memory (`gemm_grouped_v2_gather_rs.cc:713-718`). Written by the grouped GEMM epilogue,
  never touched by any peer directly.
- **Death**: `output` — `[ntokens/world_size, N]` on each rank: this rank's token shard,
  fully summed over topk *and* over all ranks (`:719-721`).
- **In between**: a sum-reduction across ranks fused with a permutation (gather) and a fold
  (topk), delivered scattered. In collective vocabulary: a **reduce-scatter whose reduction
  operator includes a per-rank gather+topk preprocess**.

```
rank r:  grouped GEMM ──► gemm_outs (local, expert-order, partial over K)
                                │  gather(routing_idx) + topk-sum        ┐
                                ▼                                        │ fused in one
         intra-node ring: partial sums accumulate hop by hop over NVLink │ consumer kernel
                                ▼                                        ┘
         own-node segment ──► output          remote-node segment ──► staging_send
                                                        │ nvshmem putmem_signal (1 flow/(node,split))
                                                        ▼
                                              staging_recv ──► += into output
```

**Why decompose along N and not M.** The consumer's unit of work is an *output token row*,
and computing it needs `topk` GEMM rows whose indices (`routing_idx`) are scattered
arbitrarily across the entire M extent — token 0's expert copies may be GEMM rows 17, 40961
and 81920. An M-slice of `gemm_outs` is therefore never independently consumable: no output
row is complete until *all* of M is. Columns are different: every output element depends on
exactly one column of each contributing GEMM row. Slice N into `n_split` windows and each
window is a self-contained instance of the entire pipeline (gather → topk → ring → network).
The paper's *decompose* here picks the only dimension the consumer's data dependencies
leave open — communication starts after `1/n_split` of the compute instead of after all
of it.

---

## 2. Buffer inventory: who allocates what, where it lives, who can touch it

Allocation is again front-loaded into constructors (`GemmGroupedV2GatherRSOpImpl`,
`src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc:537-597`, which owns a
`TopkReduceScatterOpImpl`, `:235-293`), sized by `max_m`:

| Buffer | Allocated at | Allocator / handle | Resides | Who writes | Who reads | Contents |
|---|---|---|---|---|---|---|
| `inputs[g]`, `weights[g]` | caller | torch CUDA | local | layer 0 / model | GEMM | expert-sorted activations, w1 shard |
| `gemm_outs[g]` | per fwd `:713-718` | torch CUDA | local, **private** | GEMM epilogue | consumer kernel (gather loads) | partial products, expert order |
| `barrier` (producer→consumer) | ctor→`create_barriers` `:495-505`, sized `:489-493` | `flux_create_tensor_list` (NVSHMEM/IPC list) | local HBM, P2P-mapped | GEMM `set_barrier_ptr` cascade | consumer `wait_eq` | `[n_split flags][n_split counters][problem tile counters]` |
| `reduce_buffers` | `TopkReduceScatterOpImpl::init_buffer_once` `:165-176` | `flux_create_tensor_list`, `[max_m/topk, n_dim]` | each rank's HBM, P2P-mapped into peers | **peers** (ring partial pushes) | owner (next hop reads own buffer) | in-flight ring partial sums |
| `tile_barriers` | `create_rs_barrier` `:200-224` | `flux_create_tensor_list`, one int per (m-tile, n-tile) | P2P-mapped | upstream ring rank (`atomic_store_release_sys`) | downstream rank `wait_eq` | per-tile "partial has landed" |
| `*_dptrs` pointer tables | `:172-176,218-222,284-289` | small `cudaMalloc`/byte tensors | local | host once | kernel | device-side arrays of the peers' buffer pointers |
| `staging_send` / `staging_recv` | `:180-183` (nnodes>1) | `nvshmem_create_tensor` `[nnodes, n_split, staging_rows, n_per]` | NVSHMEM symmetric heap | kernel (send) / remote PE's putmem (recv) | host put ladder / `internode_reduce_gather_rs` | node-level reduced partials |
| `internode_signals` | `:184-185` | `nvshmem_create_tensor` uint64, zero-init | symmetric heap | remote PE's `putmem_signal` | `nvshmemx_signal_wait_until` | arrival signal per (node, split) |
| `group_flags` / `group_counters` | `:186-190` | `cutlass::DeviceAllocation` (plain `cudaMalloc`) | local, **private** | consumer kernel blocks | host ladder `CUStreamWaitValue` | kernel→host "chunk staged" handshake |
| GEMM args workspace | per fwd `:741-752` | torch CUDA char tensor | local | `make_workspace_kernel` (device!) | GEMM problem visitor | per-subproblem sizes/pointers/strides |
| `output` | per fwd `:719-721` | torch CUDA | local | consumer kernel + internode accumulate | caller | final token shard |

The same three-regime allocation logic as layer 0, but the roles moved:

- The **P2P-mapped list** buffers (`reduce_buffers`, `tile_barriers`, `barrier`) are the
  intra-node working set — here peers genuinely *write into each other's memory* from
  inside a kernel (ring pushes, tile flags), which is exactly what load/store-mapped NVLink
  peer memory is for and what a symmetric-heap verb API would make clumsy (a `putmem` call
  per 16-byte pack instead of a `st.global` in the inner loop).
- The **symmetric heap** buffers (`staging_*`, `internode_signals`) exist only for the
  inter-node hop, where verbs are the only option. Note `internode_signals` is
  zero-initialized at allocation (`init_zero=true`) because its very first use is a
  comparison against a monotonically increasing `run_id` (§6) — signals are *never reset*,
  by design.
- `group_flags`/`group_counters` are again `cutlass::DeviceAllocation` for the
  `cuStreamWriteValue/WaitValue` VA restriction (comment at `:147-148`), and again private:
  they synchronize the local kernel with the local host proxy thread of the put ladder —
  no peer involved.
- `gemm_outs` being **private local memory** is the deepest difference from layer 0. The
  shared tensor of layer 1 is never remotely readable! Peers only ever see *reduced
  derivatives* of it (ring partials, staged node sums). Why: shipping raw GEMM output would
  move `world_size×` more bytes than shipping partial sums — the reduction is pushed as
  close to the producer as the dependency structure allows, a classic
  aggregate-before-transmit pattern.

---

## 3. Producer and consumer: two kernels sharing one GPU by static partition

Layer 0 overlapped by using *copy engines* against SMs. Layer 1 cannot: its
"communication" includes gather + summation, which needs SMs. So the overlap resource split
is **SMs vs. SMs, partitioned statically**:

```cpp
// gemm_grouped_v2_gather_rs.cc:99-103
int get_rs_threadblock_count() {
  static int rs_num_blocks = bytedance::flux::get_int_from_env("FLUX_RS_BLOCKS", 3);
  return rs_num_blocks;
}
// :811  — GEMM leaves exactly that many SMs unoccupied
.sm_margin = sm_margin + get_rs_threadblock_count()
```

and the launch sequence (`forward_gather_rs_impl`, `:816-838`):

```cpp
group_barrier.barrier_all(stream);                          // epoch open (all flags zeroed)
CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));
CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));
if (M_this_ep > 0) {
  gemm_op->run(args, ..., stream);                          // producer: persistent grouped GEMM
} else {
  this->barrier.fill_(1);                                   // no work: release consumer manually
}
output = topk_reduce_scatter_op->run(..., get_rs_threadblock_count(),
                                     (intptr_t)gather_rs_stream);   // consumer: 3 blocks
CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));
group_barrier.barrier_all(stream);                          // epoch close
this->barrier.zero_();                                      // re-arm flags inside the fence
this->topk_reduce_scatter_op->reset_buffer();               // tile_barrier.zero_()
```

Why this shape:

- **Concurrent, not fused.** The dense layer-1 op (`src/gemm_rs`) fuses the reduce-scatter
  into the GEMM epilogue — possible there because output tile ↔ communication tile is 1:1.
  Here the consumer must gather rows by `routing_idx` across the whole M extent, so no GEMM
  threadblock's epilogue ever holds a complete output tile. The dependency is
  many-tiles→one-row; the only clean cut is a separate kernel behind a flag protocol.
- **Static partition, tiny consumer.** The consumer is memory/NVLink-bound, not
  compute-bound: 3 threadblocks (of 768 worker + 32 sync threads each,
  `topk_gather_rs_v2.cu:697-700`) are enough to saturate the links, while the GEMM keeps
  105 of 108 SMs. If both kernels floated freely the GPU scheduler could time-slice them
  unpredictably; worse, a persistent GEMM on *all* SMs would starve the consumer whose
  progress the *next* split's buffers depend on. The `sm_margin` handshake makes the
  partition deterministic. `FLUX_RS_BLOCKS` is the single knob trading GEMM throughput
  against drain latency.
- **The barrier/event prologue** is the same epoch logic as layer 0 §4.2: `barrier_all`
  proves every rank's flags are still zero from the previous close (no rank may start
  producing while a peer's consumer could observe stale values), and `gemm_start_event`
  keeps the consumer — a spin-waiting kernel — from launching before the flags it polls are
  in a defined state. The mirrored epilogue (`gather_rs_done_event` → `barrier_all` →
  `zero_()`) guarantees flags are re-armed only after *all* ranks stopped polling them.

---

## 4. Producer side: split-major order and the counter cascade

### 4.1 The GEMM is re-shaped so that communication-order = completion-order

`make_workspace_kernel` (`src/moe_gather_rs/workspace_helper.cu:80-102`) — run **on the
GPU** (splits never leave device memory; a host-side build would force a D2H sync of
`splits` onto the critical path) — expands the `ep_nexperts × num_groups` expert GEMMs into
`n_split ×` that many **column-window subproblems**:

```cpp
for (int i = threadIdx.x; i < problem_count; i += blockDim.x) {
  int sid = i / problem_per_split;        // <— split-major: problems of split 0 come first
  int sr  = i % problem_per_split;
  int gid = sr / args.ep_nexperts;
  int eid = sr % args.ep_nexperts;
  int Mi    = ep_splits[eid];
  int M_acc = ep_splits_acc[eid] - Mi;
  problem_sizes[i] = cutlass::gemm::GemmCoord{Mi, new_N, K};        // new_N = N / n_split
  ptr_A[i] = ptr_with_offset(args.input[gid],  M_acc * K * input_elem_size);
  ptr_B[i] = ptr_with_offset(args.weights[gid], (eid * N + sid * new_N) * K * input_elem_size);
  ptr_D[i] = ptr_with_offset(args.output[gid],  (M_acc * N + sid * new_N) * output_elem_size);
  ldd[i]   = LayoutD::packed({(int)Mi, (int)N}).stride(0);          // stride = FULL N
}
```

Three loads-bearing details:

- **`sid = i / problem_per_split`** is the paper's *reschedule* in one line. CUTLASS's
  grouped-GEMM problem visitor walks problems in index order (the op is built with
  `GroupScheduleMode::kDeviceOnly`, `src/moe_gather_rs/gemm_grouped_v2_gather_rs.hpp:102`,
  so the visitor's device-side schedule follows the array), so all subproblems of split 0
  finish before split 1's bulk begins — the GEMM emits completed *column slices* in
  ascending `sid` order, which is precisely the order the consumer (§5) and the network
  ladder (§6) walk. The paper calls this executing the GroupGEMM "column-wise."
- **`ldd = full N`**: all `n_split` subproblems of an expert write disjoint column windows
  of the *same* `gemm_outs` rows. The decomposition is purely logical — no extra buffer, no
  re-assembly copy; the consumer reads `gemm_outs` as one `[M, N]` matrix and selects
  columns by `sid` arithmetic.
- `n_split` itself is clamped in the ctor so windows stay tile-aligned for the consumer
  (`n_split_fixed`, `gemm_grouped_v2_gather_rs.cc:527-534`: `N / n_split` must be a
  multiple of the consumer's `kTileSizeN = 1024`).

### 4.2 Completion detection: tile → problem → split, in three atomics

The GEMM cannot "send" anything — it just writes local memory. What it *can* do is prove,
cheaply, that a whole column slice is complete. Each threadblock's epilogue ends with
`set_barrier_ptr` (`src/moe_gather_rs/cutlass_impls/gather_rs_gemm_grouped_with_absmax.h:543,550-576`):

```cpp
int *tile_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128) * 2;
int counter = atomicAdd(tile_counter_ptr + problem_idx, 1);
if (counter == problem_tile_count - 1) {                 // last tile of this subproblem
  int problem_per_split   = *params.non_empty_problem_count / params.n_split;
  int group_idx           = problem_idx / problem_per_split;   // = sid
  int *problem_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128);
  int problem_counter = atomicAdd(problem_counter_ptr + group_idx, 1);
  if (problem_counter == problem_per_split - 1) {        // last subproblem of this split
    cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*(params.barrier_ptr + group_idx))
        .fetch_add(1, cuda::memory_order_release);       // split flag := 1
  }
}
```

Why a cascade instead of layer-0-style direct flags: the consumable unit (a full column
slice, §1) is completed *collectively* by hundreds of threadblocks across dozens of
subproblems, in nondeterministic order. Counters convert "unordered many-writer completion"
into "single release flag" with one atomic per tile and one per subproblem — the last
writer, whoever it is, flips the flag. The `release` on the final increment pairs with the
consumer's acquire wait so all epilogue stores to `gemm_outs` are visible once the flag is.
The cascade needs the true subproblem count at flag level, so `make_workspace_kernel`
also counts non-empty experts on device (`workspace_helper.cu:104-115`) — an expert with
zero tokens contributes zero tiles, and dividing by the *static* problem count would leave
split flags permanently one short (deadlock).

Layout of `barrier` (from `get_barrier_size`, `gemm_grouped_v2_gather_rs.cc:489-493`):
`[0, n_split)` split flags — the only words the consumer polls — then padded counter
regions. Flags and counters live in the same P2P-visible allocation so `use_read_mode`
consumers can also poll a *peer's* flags, but in the default write mode each rank polls
only its own — local L2 traffic, like layer 0.

---

## 5. Consumer side: one kernel, three reductions, ring-shaped

`topk_gather_rs_v2_kernel` (`src/moe_gather_rs/topk_gather_rs_v2.cu:625-691`) is launched
with the 3 reserved threadblocks on `gather_rs_stream` and structured as a **split-major
outer loop mirroring the producer**:

```cpp
for (int sid = 0; sid < params.n_split; sid++) {
  Barrier::wait_eq(params.barrier[params.local_rank], threadIdx.x, sid, 1);  // producer gate
  // hierarchical reduce-scatter: one intra-node ring per owner-node segment group.
  // remote groups run first so their inter-node puts overlap the remaining work.
  for (int g_iter = 0; g_iter < params.nnodes; g_iter++) {
    int g = (params.node_idx + 1 + g_iter) % params.nnodes;
    for (int stage = 0; stage < params.local_world_size; stage++) {
      for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
           blk_id += gridDim.x) {
        TopkGatherRsOp<...>{}(params, smem, blk_m, blk_n, sid, stage, g);
      }
    }
    ...
  }
}
```

### 5.1 Why a ring, and why this segment rotation

Within a node, the reduction is an **L-stage ring** (L = `local_world_size` = 4). At each
stage, each rank processes segment `g·L + (stage + local_rank + 1) % L`
(`topk_gather_rs_v2.cu:371`) and pushes its running partial to ring-predecessor
`rank_to = (local_rank + L - 1) % L` (`:372`). Per tile, in `TopkGatherRsOp`
(`:334-463`):

```cpp
// stage 0: produce my own contribution for this segment (gather + topk-sum)
// stage>0: my predecessor's partial has landed in MY reduce_buffer — add mine on top
if (stage != 0) {
  int *tile_barrier_ptr = params.tile_barrier_ptrs[params.local_rank];
  WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, 1);   // :406-410
}
for (row_g = row_start ...) {
  for (int topk = 0; topk < kTopk; topk++) {
    int64_t row_g_from = smem_idx[topk + row_l * kTopk];                // routing_idx (smem)
    pack.data = loadPack((T *)params.input_ptrs[j] + row_g_from * N + col_g);  // gemm_outs
    ... acc[i] += ...                                                   // topk + group sum
  }
  void *output_ptr = last_round
      ? last_round_dst_ptr<T>(...)                       // output shard or staging slot
      : (void *)((T *)params.reduce_ptrs[rank_to] + row_off);   // PUSH to predecessor
  if (stage == 0) storePack(output_ptr, pack.data);
  else { pack_lr.data = loadPack((T *)params.reduce_ptrs[params.local_rank] + row_off);
         storePack(output_ptr, addPack(pack_lr, pack)); }
}
```

The reasoning, piece by piece:

- **Why a ring at all**: a naive gather-then-reduce ("every rank sends its partial for my
  segment to me") delivers L−1 full segment copies to one GPU per segment — an incast, and
  the receiver alone must sum them. The ring moves each byte once per hop but *accumulates
  as it moves*: after L−1 hops the segment's owner receives a single, already-summed
  partial stream. Total NVLink traffic per output byte: (L−1) writes, evenly spread over
  all links, no incast, and the additions are distributed over all ranks.
- **Why the `(stage + local_rank + 1) % L` rotation**: at any stage, the four ranks work on
  four *different* segments — each link carries exactly one segment's traffic, every rank
  computes and pushes simultaneously. A fixed segment order would have all ranks converge
  on the same buffer.
- **The gather+topk is folded into stage 0** (and into every stage's "my contribution")
  rather than materialized first: the rows are read from `gemm_outs` directly via the
  `routing_idx` slice staged in shared memory (`:396-404`). The permutation costs one
  smem-indirected load — it never touches HBM as a standalone pass.
- **Push (write) rather than pull (read)**: each rank *stores* its partial into the
  predecessor's `reduce_buffer` (`reduce_ptrs[rank_to]`). Writes to peer memory are
  fire-and-forget over NVLink (no round trip stalling the pipeline); the matching read is
  then a *local* read by the next-stage owner. `use_read_mode` inverts this (single-node
  only) for fabrics where reads behave better.
- **Per-tile flags, sync-warp trick**: correctness of "predecessor's partial has landed"
  is per (m-tile, n-tile): `tile_barriers` (§2). The flag is set by a dedicated 32-thread
  sync warp per block (`:455-462`): workers hit `FullBarSync` after their stores, then
  thread 0 of the sync warp does `atomic_store_release_sys(tile_barrier_ptr + tile_idx, 1)`
  on the *destination rank's* array. Splitting flag-setting into its own warp lets worker
  threads start the next tile without waiting for the system-scope release to retire.
- **The last hop writes the final destination directly** (`last_round_dst_ptr`,
  `:297-315`): for the segment group owned by *this* node, straight into `output` (minus
  the segment offset — the scatter is implicit in the addressing); for a remote node's
  group, into the `(node, split)` slot of `staging_send`. No copy-out pass exists.

### 5.2 The producer gate, and why the whole design converges here

`Barrier::wait_eq(barrier[local_rank], threadIdx.x, sid, 1)` at `:641` is the *only* place
the consumer touches the producer: one acquire-spin on one local flag word per split. When
it opens, split `sid`'s columns of `gemm_outs` are complete on **this** rank — and because
every rank runs the same split-major producer order, approximately complete on peers too,
which is what makes an immediately-starting ring safe *and* non-blocking: the data each
ring stage reads was produced locally on each participating rank behind its own flag.

This is the layer-1 overlap in one sentence: **while the consumer rings split `sid` around
the node, the GEMM (on its 105 SMs) is producing split `sid+1`** — compute hides the
communication of the previous slice, symmetric to layer 0 hiding transport behind
stage-ordered tiles.

---

## 6. Multi-node: reduce locally, ship once, sum remotely

Across nodes the same bandwidth cliff as layer 0 applies, plus a subtlety: NVSHMEM puts to
a remote PE on Slingshot are proxied — best issued as few, large, contiguous transfers.
The multi-node port therefore adds a second hierarchy level rather than widening the ring:

**Node-level pre-reduction.** The segment groups are walked remote-first
(`g = (node_idx + 1 + g_iter) % nnodes`, `:650`) — so the network transfer for remote
groups is in flight while the kernel still processes the own-node group. Each remote
group's ring ends in `staging_send[g][sid]`: a contiguous `[rows, n_per]` block holding
this **node's total contribution** (already summed over all L local ranks) to node g's
token segment, for this split. Cross-node traffic is thus cut by a factor of L versus
shipping per-GPU partials — same aggregate-before-transmit logic as §2, applied at node
scope.

**Kernel→host handshake.** The kernel cannot issue NVSHMEM puts itself (host-proxied
transport); instead the last block to finish a `(g, sid)` group publishes it
(`topk_gather_rs_v2.cu:667-677`):

```cpp
__threadfence_system();
if (threadIdx.x == 0) {
  int done = atomicAdd(params.group_counters + g * params.n_split + sid, 1) + 1;
  if (done == gridDim.x) {
    atomic_store_release_sys(params.group_flags + g * params.n_split + sid, 1);
  }
}
```

— the same last-writer-flips-the-flag counter idiom as the producer cascade (§4.2), here
with only `gridDim.x` (=3) writers. The `__threadfence_system()` makes the staged data
visible to the host-proxied DMA that the flag will unleash.

**The host put ladder** (`TopkReduceScatterOpImpl::run`,
`gemm_grouped_v2_gather_rs.cc:394-413`), pre-enqueued on a dedicated `internode_stream`:

```cpp
for (int sid = 0; sid < this->n_split; sid++) {
  for (int gi = 0; gi < this->nnodes - 1; gi++) {
    int g = (this->node_idx + 1 + gi) % this->nnodes;      // same order the kernel produces
    CU_CHECK(CUStreamWaitValue(this->internode_stream,
        (CUdeviceptr)(this->group_flags.get() + idx), 1, CU_STREAM_WAIT_VALUE_GEQ));
    nvshmemx_putmem_signal_nbi_on_stream(
        recv_base + (node_idx * n_split + sid) * slot_bytes,   // their staging_recv slot for MY node
        send_base + idx * slot_bytes,                          // my staged chunk
        chunk_bytes,
        sig_base + this->node_idx * this->n_split + sid,       // their signal word for MY node
        this->run_id_, NVSHMEM_SIGNAL_SET,
        /*pe=*/g * this->local_world_size + this->local_rank,  // same-local-rank peer on node g
        this->internode_stream);
  }
}
```

Pattern rationale:

- **`CUStreamWaitValue` + pre-enqueued puts** turn the host ladder into a passive pipeline:
  everything is queued before the kernel produces its first chunk, and each put fires the
  moment its flag flips — no host polling thread, no kernel involvement, and the ladder's
  order matches the kernel's production order so waits never head-of-line-block a ready
  chunk.
- **`putmem_signal` fuses payload and doorbell**: the signal word (`internode_signals`) is
  set by the NIC *after* the payload, with one verb — the receiver needs no separate
  flag-exchange round trip.
- **Signals compare against `run_id_`** (incremented per forward, `:370`; waits use
  `NVSHMEM_CMP_GE`) instead of being reset: resetting a symmetric signal word would itself
  require cross-node synchronization ("has the peer consumed it yet?"). A monotonic epoch
  number needs none. The local `group_flags`/`group_counters`, by contrast, are private and
  simply `cudaMemsetAsync`-ed at the start of each run (`:371-375`), published to the
  internode stream via `staging_reset_event`.
- **Same-local-rank pairing** (`pe = g·L + local_rank`), as in layer 0: node-to-node flows
  are spread one-per-GPU across NIC rails, and the receiving GPU is the one that will
  consume the chunk — no intra-node forwarding hop on the receive side.

**The receive side** stays on the main consumer stream (`:416-437`): per split, wait for
every remote node's signal (`nvshmemx_signal_wait_until_on_stream`, CMP_GE `run_id_`), then
`internode_reduce_gather_rs` adds each `staging_recv[m][sid]` chunk into `output` — the
own-node contribution is already there (the kernel's last ring hop wrote it, §5.1). The
final sum over nodes thus lands with one extra kernel per split, and split `sid`'s
network arrival overlaps split `sid+1`'s intra-node ring.

Constraints this design imposes (checked in ctors): token counts divisible by
`world_size·topk`, `max_m/topk` divisible by `world_size` (fixed-size staging slots),
node-contiguous rank layout (`:272-274` — same-local-rank pairing depends on it), and all
staging/signal buffers on the symmetric heap (`NVSHMEM_SYMMETRIC_SIZE=4G` for large
configs). `do_all_reduce`/`use_read_mode` are single-node-only (FLUX_CHECK at `:269-270`).

---

## 7. The complete timeline (2 nodes × 4 GPUs)

```
main stream (105 SMs)        gather_rs_stream (3 SMs)             internode_stream (host ladder)
─────────────────────        ────────────────────────             ──────────────────────────────
barrier_all; events ───────► (waits gemm_start_event)             (waits staging_reset_event)
GEMM split 0 tiles…          wait_eq flag[0] … spin
  cascade ⇒ flag[0]=1  ────► ring remote group (g=1), split 0
GEMM split 1 tiles…            └ counters ⇒ group_flags[1,0]=1 ──► put chunk(1,0) ⇒ signal
                             ring own group (g=0), split 0 → output
  cascade ⇒ flag[1]=1  ────► wait_eq flag[1]; ring g=1 split 1 ──► put chunk(1,1) ⇒ signal
…                            …
                             signal_wait(node m, split s) ◄──────── remote node's put arrives
                             internode_reduce += staging_recv → output
gather_rs_done_event ◄────── done
barrier_all; flags.zero_()
```

Every column owns a distinct resource (GEMM SMs / consumer SMs + NVLink / NIC), every arrow
is a §3-§6 mechanism, and each split flows through the pipeline while its successor is
still being computed — the paper's layer-1 claim: decompose along N made column slices
independently reducible; the split-major reschedule made the producer emit them in the
consumer's order; and the transport (ring, remote-groups-first, staged node sums,
same-local-rank puts) is shaped so each hop's output is exactly the next hop's minimal
input.

## TL;DR mapping

| Paper concept | Distributed-systems decision | Code |
|---|---|---|
| shared tensor | `gemm_outs` kept **private**; peers only see reduced derivatives | `gemm_grouped_v2_gather_rs.cc:713-718` |
| decompose along N | `n_split` column-window subproblems over the same buffer (`ldd`=full N), tile-aligned via `n_split_fixed` | `workspace_helper.cu:80-102`; `gemm_grouped_v2_gather_rs.cc:527-534` |
| reschedule | split-major problem order (`sid = i / problem_per_split`) + `kDeviceOnly` visitor ⇒ column-wise GroupGEMM | `workspace_helper.cu:81`; `gemm_grouped_v2_gather_rs.hpp:102` |
| readiness signaling | tile→problem→split counter cascade, release/acquire flag per split; non-empty count computed on device | `gather_rs_gemm_grouped_with_absmax.h:550-576`; `workspace_helper.cu:104-115`; `topk_gather_rs_v2.cu:641` |
| comm/comp resource split | concurrent kernels, static SM partition (`sm_margin + FLUX_RS_BLOCKS`) | `gemm_grouped_v2_gather_rs.cc:99-103,811,816-838` |
| intra-node transport | push-mode ring with rotated segments, gather+topk fused into each hop, per-tile P2P flags set by a dedicated sync warp | `topk_gather_rs_v2.cu:371-372,406-462,649-666` |
| inter-node transport | node-level pre-reduction into symmetric staging, kernel→host counter/flag handshake, `CUStreamWaitValue`-gated `putmem_signal` ladder, same-local-rank pairing, monotonic `run_id` signals | `topk_gather_rs_v2.cu:667-677`; `gemm_grouped_v2_gather_rs.cc:367-437` |
| epoch management | `barrier_all` brackets + flag `zero_()` inside the fence; signals never reset | `gemm_grouped_v2_gather_rs.cc:816,840-842` |

**Run it** (from CLAUDE.md):

```bash
source ./module.sh
salloc --qos interactive -C gpu --account m4243_g              # single node (T*E == 4)
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -T 4 -E 1

salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30     # two nodes
export NVSHMEM_SYMMETRIC_SIZE=4G   # staging/signals live on the symmetric heap
srun --nodes=2 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 8 -E 1
```
