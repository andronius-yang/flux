# COMET MoE Layer 0 on SM80: *Decompose* and *Reschedule* — a code walkthrough

This walkthrough maps the two shared-tensor optimizations from the COMET paper (`COMET.pdf`,
§3.1 *Shared Tensor Based Dependency Resolving*) onto the actual SM80 (A100, `--arch 80`)
implementation of **MoE layer 0**: the fused
**AllGather → Scatter → Grouped GEMM** pipeline.

> Paper summary (§3.1): *"the dependency resolving process employs two key optimization
> strategies on the shared tensors: ① decomposing the shared tensors along specific dimensions
> to break the coarse-grained data dependencies and, ② rescheduling the computations to enhance
> efficiency while ensuring effective overlapping."*

On SM80 the layer-0 op is the **V2** code path (`GemmGroupedV2AGScatter*`; the V3 classes are
the Hopper/sm90 warp-specialized design and do not apply on A100):

| Piece | File |
|---|---|
| Torch op driver (host orchestration) | `src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc` |
| Kernel/device op builder | `src/moe_ag_scatter/gemm_grouped_v2_ag_scatter.hpp` |
| Token sort + tile schedule (host & device) | `src/moe_ag_scatter/sort_util.cu`, `src/moe_ag_scatter/sort_util.h` |
| GPU-side workspace/schedule builder | `src/moe_ag_scatter/workspace_util.cu`, `workspace_util.h` |
| Fused CUTLASS grouped-GEMM kernel | `src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h` |
| Custom grouped problem visitor | `src/moe_ag_scatter/cutlass_impls/ag_scatter_grouped_problem_visitor.hpp` |
| Arguments struct | `include/flux/args/moe_ag_scatter.h` (`GemmGroupedV2AGScatterArguments`) |
| Test | `./launch.sh test/python/moe_ag_scatter/test_moe_ag.py` |

---

## 1. The pipeline and its shared tensor

In layer 0, every rank holds a *shard* of the input tokens (`[ntokens/world_size, hidden]`).
The tokens needed by the experts that live on this rank are spread across **all** ranks, so the
producer operator is an **all-gather** of the token shards, and the consumer is a **grouped GEMM**
(one GEMM per local expert) that *gathers* its rows out of the all-gathered buffer and *scatters*
its output rows back to token order.

The **shared tensor** is the all-gathered input buffer. For a single node it lives in the
CUDA-IPC buffer of `AllGatherOp`; for multi-node it is an NVSHMEM symmetric tensor
(`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:322-328`):

```cpp
if (nnodes == 1) {
  ag_op.emplace(this->tp_group, 1, max_ntokens, hidden, input_dtype);
} else {
  FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == dist_env.local_rank);
  this->input_buffer = nvshmem_create_tensor({max_ntokens, hidden}, input_dtype);
  this->barrier_block.reset(pad_to(world_size * (int64_t)sizeof(int), (int64_t)128));
}
```

A naive implementation waits for the whole all-gather before launching the GEMM. COMET instead
lets GEMM tiles start as soon as *the rows that particular tile needs* have arrived.

---

## 2. DECOMPOSE: split the shared tensor along the M (token) dimension

> Paper §3.1.1: *"tokens are independent with each other alongside the M (token) dimension,
> allowing for decomposition of the shared tensor along M. However, since the computation of a
> GEMM tile involves multiplication and reduction along the token embedding dimension …
> decomposing the shared tensor along this dimension is not feasible."*

### 2.1 The decomposition unit = one source rank's shard

The all-gather is decomposed into **per-source-rank shards**, i.e. contiguous row blocks of the
shared tensor. Each shard gets its own **ready flag**: a `world_size`-long array of ints, set to
`1` as each rank's rows land. In the multi-node path this is explicit in `all_gather_all2all`
(`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:412-430`); the single-node
`AllGatherOp` maintains the same per-rank flags in `ag_op->local_barrier_buffer()`:

```cpp
for (int local_rank = dist_env.local_rank, j = 0; j < dist_env.local_world_size;
     ++j, local_rank = (local_rank + 1) % dist_env.local_world_size) {
  int src_rank = dist_env.local_rank_to_global_rank(local_rank, node_idx);
  ...
  if (local_rank != dist_env.local_rank) {
    auto shard_input = full_input(_, _, src_rank);
    nvshmemx_getmem_on_stream(..., local_rank_global, this->cp_stream);
  }
  CU_CHECK(CUStreamWriteValue(                       // <── per-source-rank ready flag
      this->cp_stream,
      (CUdeviceptr)(ptr_offset(barrier_block.get(), src_rank * sizeof(int))),
      1, CU_STREAM_WRITE_VALUE_DEFAULT));
}
```

The copy runs on dedicated copy streams (`cp_stream`, created at
`gemm_grouped_v2_ag_scatter.cc:256-261`) so the communication proceeds **concurrently** with the
GEMM kernel launched right after on the main stream (`forward_impl`, Step 2 at line 507:
*"Launch AG comm as early as possible"*).

### 2.2 The consumer waits per tile, not per tensor

Inside the fused CUTLASS grouped-GEMM kernel, each threadblock computes which row range
`[m_start, m_end]` of its expert's (sorted) input its tile covers, maps that to the range of
**source ranks** it depends on, and spin-waits *only on those flags*
(`src/moe_ag_scatter/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h:423-441`):

```cpp
int tile_idx_m = threadblock_idx / grid_shape.n();
int tile_idx_n = threadblock_idx % grid_shape.n();
// tile_idx_m cross which segments and should wait for which signals
int lane_idx = threadIdx.x % 32;
int m_start = tile_idx_m * Mma::Shape::kM;
int m_end = min((tile_idx_m + 1) * Mma::Shape::kM, problem_size.m()) - 1;
int * split_accum = params.split_tp_accum_ptr + params.world_size * (problem_idx % params.nexperts_ep);
int segment_start =
    __ffs(__ballot_sync(0xffffffff, lane_idx < params.world_size ? (m_start < split_accum[lane_idx]) : false)) - 1;
int segment_end =
    __ffs(__ballot_sync(0xffffffff, lane_idx < params.world_size ? (m_end < split_accum[lane_idx]) : false)) - 1;
if (lane_idx >= segment_start && lane_idx <= segment_end) {
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> barrier(params.barrier_ptr[lane_idx]);
  while (barrier.load(cuda::memory_order_acquire) != 1) { }
}
__syncthreads();
```

This is exactly the paper's "fine-grained data dependency": a tile whose rows all come from
already-arrived ranks never blocks, and while a tile does block, the persistent-kernel
threadblock scheduler (§4/"the threadblock scheduler hides remote-I/O latency by switching among
oversubscribed threadblocks", see `docs/design.md`) lets other threadblocks make progress.

`split_accum` is the per-expert cumulative row count per source rank
(`sorted_splits_cumsum`, computed below) — it is what turns "row range" into "rank range".

### 2.3 Gather-A / Scatter-D make the decomposition transparent

The GEMM never materializes a sorted copy of the tokens. Its A-operand iterator *gathers* rows
by index, and the epilogue *scatters* output rows back — both are enabled in the kernel template
(`src/moe_ag_scatter/cutlass_impls/default_ag_scatter_gemm_grouped_with_absmax.h:284-286`):

```cpp
    true,  /*GatherA*/
    false, /*GatherB*/
    true,  /*ScatterD*/
```

and wired per-problem in the kernel body
(`ag_scatter_gemm_grouped_with_absmax.h:483-489` and `:553-559`):

```cpp
typename Mma::IteratorA iterator_A(
  LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size.k()},
  thread_idx, tb_offset_A,
  params.gather_A_ptr[problem_idx]);      // row gather indices (sorted_gather_index)
...
typename Epilogue::OutputTileIterator iterator_D(
  params_D, ptr_D, problem_size.mn(),
  thread_idx, threadblock_offset.mn(),
  params.scatter_D_ptr[problem_idx]);     // row scatter indices (sorted_scatter_index)
```

Note `ptr_A[i] = args.input` for **every** problem — all experts read the same shared tensor
through different gather indices (`src/moe_ag_scatter/workspace_util.cu:219`).

---

## 3. RESCHEDULE: sort tokens by source rank, compute local tiles first

> Paper §3.1.2 / Figure 5: *"To enable early computation by the experts, tokens are sorted based
> on their source rank … The compute sequence of tiles in the GroupGEMM is then designed to
> minimize dependency on remote data, with computation beginning from tiles containing local
> tokens while the transfer of other remote tokens proceeds concurrently."*

Decomposition alone is not enough: with tokens in arbitrary order, almost every 128-row GEMM
tile would mix rows from many source ranks and would have to wait for the *last* of them.
Rescheduling has two parts in the code.

### 3.1 Reorganize the data: counting-sort each expert's rows by source rank

`AgScatterSortOpV2` (`src/moe_ag_scatter/sort_util.cu:342-498`) runs one threadblock per expert.
It classifies each of the expert's input rows by which rank produced it, and produces the
`sorted_gather_index` / `sorted_scatter_index` permutations consumed by Gather-A/Scatter-D
(`sort_util.cu:441-450`):

```cpp
for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
  int source_row = gather_index[i];
  int source_rank = source_row / params.ntokens_perrank;   // which rank owns this token
  int idx = atomicAdd(&smem.rows_counter[source_rank], 1); // counting sort by source rank
  sorted_gather_index[i] = idx;  // row -> inner index <Expert, Rank>
}
__syncthreads();
for (int i = thread_idx; i < params.tp_size; i += ThreadsInBlock) {
  sorted_splits(i, cur_expert_id) = smem.rows_counter[i];  // per-(rank, expert) row counts
}
```

After this pass, each expert's M range is laid out as `[rows from rank 0][rows from rank 1]…`,
so a tile's rank dependency becomes a *contiguous* segment — this is what makes the per-tile
wait in §2.2 cheap and what makes "local-first" scheduling possible at tile granularity.
`sorted_splits_cumsum` (the per-expert prefix sum over ranks) is passed to the GEMM as
`accum_per_rank_ptr` and becomes `split_accum` in the kernel snippet above.

The host driver invokes this in Step 3 of `forward_impl`
(`ths_op/gemm_grouped_v2_ag_scatter.cc:546-577`): `calc_gather_index_impl` →
`ag_scatter_sort_impl_v2` → `sort_scatter_index_to_per_expert`.

### 3.2 Reorder the computation: stage-based tile schedule, own rank first

The tile execution order is *precomputed* into the workspace and consumed by a custom problem
visitor. The order is organized in `world_size` **stages**: stage 0 = tiles depending only on
the current rank's own (local) tokens, stage 1 = tiles additionally needing the next rank in
the (node-aware) ring, and so on. The rank→stage mapping is a circular shift that puts the
current rank first — and, multi-node, the current *node's* ranks first
(`src/moe_ag_scatter/sort_util.h:166-180`):

```cpp
// we shift the computation rank order to comply with the order of gathering data.
// for ranks of the same nodes, circular shift the ranks to make the current local rank
// to be processed first. ...
//  for rank #1, the order is: (1,2,3,0,5,6,7,4)
//  for rank #6, the order is: (6,7,4,5,2,3,0,1)
CUTLASS_HOST_DEVICE
int shift_rank_to_order(int rank, DistEnv const &dist_env) { ... }
```

The schedule itself is built on the GPU inside the workspace-preparation kernel,
`calc_sorted_problem_schedule_v2` (`src/moe_ag_scatter/workspace_util.cu:35-103`). For every
`(stage, weight-group, expert)` it computes which `tiled_m` range of that expert becomes ready
at that stage. The subtlety is **boundary tiles** that straddle two ranks' row segments: a tile
is charged to the *latest* stage among the ranks it touches, via `get_stage_for_tile`
(`workspace_util.cu:46-68`):

```cpp
auto get_stage_for_tile = [=](int eid, int tiled_m) {
  const int *cumsum_this_rank = args.accum_per_rank_ptr + eid * tp_size;
  ...
  int m_start_this_tile = tiled_m * tile_size_m;
  int m_end_this_tile = std::min(ep_splits[eid], (tiled_m + 1) * tile_size_m) - 1;
  int segment_start_this_tile = get_rank_id(m_start_this_tile);
  int segment_end_this_tile = get_rank_id(m_end_this_tile);
  int stage_max = 0;
  for (int sid = segment_start_this_tile; sid <= segment_end_this_tile; sid++) {
    int stage = shift_rank_to_order(sid, args.dist_env);   // ring distance from this rank
    ...
    stage_max = std::max(stage, stage_max);
  }
  return stage_max;   // tile becomes computable at the LATEST contributing stage
};
```

(the readable host-side reference implementation of the same policy is
`get_sorted_problem_schedule_v2` in `src/moe_ag_scatter/sort_util.cu:647-723`, with the comment
*"start from `rank` segment, and leaves the tile cross bounder to last stage"*).

`fill_problem_info` (`workspace_util.cu:105-159`) then flattens the
`(stage → expert → tile_m)` schedule into a linear array of
`ProblemInfo{problem_idx, problem_tile_idx}` records (`workspace_util.h:21-25`), striping
consecutive tiles round-robin across the persistent threadblocks:

```cpp
int num_tiles = tiled_m * tiled_n * args.num_groups;
int tiles_per_tb = (num_tiles + threadblock_count - 1) / threadblock_count;
for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
  int tile_idx_m = i / tiled_n;
  int tile_idx_n = i % tiled_n;
  int tid = i % threadblock_count;      // round-robin across threadblocks…
  int index = i / threadblock_count;    // …so all TBs advance through stages together
  auto &info = problem_info[tid * tiles_per_tb + index];
  auto &tile = sched_tile[tile_idx_m];
  info.problem_idx = tile.expert_idx;
  info.problem_tile_idx = tile_idx_n + tile.tile_m_idx * tiled_n;
}
```

Because the outer index `i` walks the stage-sorted `sched_tile` order, *every* threadblock's
private work list starts with stage-0 (local-token) tiles: computation begins immediately while
remote shards are still in flight — precisely Figure 5 of the paper.

### 3.3 The kernel consumes the precomputed order

The stock CUTLASS device-side visitor is replaced by `AGScatterGroupedProblemVisitor`, which
just streams the precomputed `ProblemInfo` records through shared memory
(`src/moe_ag_scatter/cutlass_impls/ag_scatter_grouped_problem_visitor.hpp:96-122`):

```cpp
CUTLASS_DEVICE
bool next_tile() {
  if (this->tile_idx >= this->params.tile_count) { return false; }
  ...
  auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
  ++tiles_computed;
  ...
  this->problem_idx = problem_info.problem_idx;          // which expert-GEMM
  this->problem_tile_idx = problem_info.problem_start;   // which (m,n) tile inside it
  return true;
}
```

The whole schedule is built **asynchronously on the GPU** in a single helper kernel launched at
op initialization (`make_workspace_async` → `prepare_workspace_kernel`,
`src/moe_ag_scatter/workspace_util.cu:161-283`), so no host round-trip for the dynamic,
gate-dependent `splits` is needed.

---

## 4. Putting it together: `forward_impl`

`GemmGroupedV2AGScatterOpImpl::forward_impl`
(`src/moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.cc:435-677`) sequences it all:

```text
Step 2 (line 507): launch the all-gather FIRST, on separate copy stream(s)
                   → producer starts filling shared tensor + per-rank flags
Step 3 (line 523): helper kernels: calc_gather_index_impl, ag_scatter_sort_impl_v2
                   → DECOMPOSE-friendly data layout (sort by source rank)
Step 4 (line 589): build GemmGroupedV2AGScatterArguments
                   (gather_A / scatter_D / accum_per_rank_ptr / barrier_ptr,
                    tile_size_m/n taken from the tuned CUTLASS tile shape, lines 503-505)
Step 5 (line 654): op->run(...) — the fused grouped GEMM, whose workspace-prep kernel
                   computes the RESCHEDULEd tile order, and whose tiles spin on
                   per-source-rank flags
```

The arguments struct that carries everything into the kernel is
`GemmGroupedV2AGScatterArguments` (`include/flux/args/moe_ag_scatter.h:37-66`), notably:

```cpp
int32_t *gather_A  = nullptr;   // sorted_gather_index  (A-operand row gather)
int32_t *scatter_D = nullptr;   // sorted_scatter_index (D row scatter)
void *problem_schedules = nullptr;      // stage-sorted ProblemSchedV2 records
int *accum_per_rank_ptr = nullptr;      // per-expert, per-rank row cumsum
int tile_size_m = 0, tile_size_n = 0;
int *barrier_ptr = nullptr;             // per-source-rank ready flags
```

## 5. Try it

```bash
source ./module.sh
# single node (4× A100):
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
# two nodes:
salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
```

The test compares the fused op against a torch reference that performs a *blocking* all-gather
followed by a plain grouped GEMM — the speedup is exactly the overlap won by decompose+reschedule.

---

### TL;DR

| Optimization | Paper | SM80 code |
|---|---|---|
| **Decompose** shared tensor along **M** (tokens are independent; K is a reduction dim) | §3.1.1, Fig. 4 | Per-source-rank shards + ready flags (`gemm_grouped_v2_ag_scatter.cc:425-429`); tiles wait only on their own rank range (`ag_scatter_gemm_grouped_with_absmax.h:423-441`); Gather-A/Scatter-D indices instead of a materialized sorted tensor (`default_ag_scatter_gemm_grouped_with_absmax.h:284-286`) |
| **Reschedule** tiles: local tokens first, ring order after | §3.1.2, Fig. 5 | Counting sort by source rank (`sort_util.cu:441-450`), node-aware ring order (`sort_util.h:172-180`), stage-based tile schedule with boundary tiles deferred (`workspace_util.cu:35-103`, reference `sort_util.cu:647-723`), precomputed order fed to a custom problem visitor (`ag_scatter_grouped_problem_visitor.hpp:96-122`) |
