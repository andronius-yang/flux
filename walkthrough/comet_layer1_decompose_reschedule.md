# COMET MoE Layer 1 on SM80: *Decompose* and *Reschedule* — a code walkthrough

This walkthrough maps the two shared-tensor optimizations from the COMET paper (`COMET.pdf`,
§3.1 *Shared Tensor Based Dependency Resolving*) onto the SM80 (A100, `--arch 80`)
implementation of **MoE layer 1**: the fused
**Grouped GEMM → Gather → top-k Reduce → ReduceScatter** pipeline.

> Paper summary (§3.1): *"① decomposing the shared tensors along specific dimensions to break
> the coarse-grained data dependencies and, ② rescheduling the computations to enhance
> efficiency while ensuring effective overlapping."*

On SM80 the layer-1 op is the **V2** code path (`GemmGroupedV2GatherRS*`; the
`*V3*`/threadblock-specialized classes are the Hopper design):

| Piece | File |
|---|---|
| Torch op driver + TopkReduceScatterOp (host) | `src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc` |
| Kernel/device op builder | `src/moe_gather_rs/gemm_grouped_v2_gather_rs.hpp` |
| GPU workspace builder (N-split subproblems) | `src/moe_gather_rs/workspace_helper.cu` |
| Fused CUTLASS grouped GEMM + split signaling | `src/moe_gather_rs/cutlass_impls/gather_rs_gemm_grouped_with_absmax.h` |
| Consumer kernel: gather + topk-reduce + RS | `src/moe_gather_rs/topk_gather_rs_v2.cu`, `topk_gather_rs.hpp` |
| Arguments structs | `include/flux/args/moe_gather_rs.h` |
| Test | `./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py` |

---

## 1. The pipeline and its shared tensor

Layer 1 is the *computation → communication* direction. The grouped GEMM computes every local
expert's output, `[M_this_ep, N]` rows in expert-sorted order. Afterwards each original token
must (a) **gather** its `topk` expert-output rows, (b) **reduce** them (top-k combine), and
(c) **reduce-scatter** the result so each rank ends up with its own token shard.

The **shared tensor** is the grouped-GEMM output (`gemm_outs` in the code — the `input_ptrs` of
the consumer). Producer = grouped GEMM; consumer = `topk_gather_rs_v2` kernel.

> Paper §3.1.1: *"the consumer operator contains a top-K reduction, which reduces tokens along
> the M dimension, leading to significant interdependencies between tokens along this dimension.
> Thus, the shared tensor can only be decomposed along the N dimension."*

A token's reduction needs rows from *several different experts* (its top-k choices), and those
rows sit at gate-dependent, effectively random M positions. So an M-decomposition would give
the consumer nothing to start on. Along **N**, however, every column slice is independent: once
all experts have produced columns `[0, N/n_split)`, the *complete* top-k reduction and
reduce-scatter for that column slice can run — while the GEMM is still computing the remaining
columns.

---

## 2. DECOMPOSE: split the shared tensor along the N dimension (`n_split`)

### 2.1 Every (expert × N-slice) becomes its own GEMM subproblem

The op is constructed with an `n_split` factor (see `moe_layer1.py` / the test's `--n_split`
flag). The GPU-side workspace builder expands the per-expert problem list by `n_split`,
so problem `i` computes expert `eid`'s slice `sid` of width `new_N = N / n_split`
(`src/moe_gather_rs/workspace_helper.cu:39-98`):

```cpp
int problem_per_split = args.num_groups * args.ep_nexperts;
int problem_count = problem_per_split * args.N_split;
const int new_N = args.N / args.N_split;
...
for (int i = threadIdx.x; i < problem_count; i += blockDim.x) {
  int sid = i / problem_per_split;      // which N slice   <── split-major problem order!
  int sr  = i % problem_per_split;
  int gid = sr / args.ep_nexperts;      // weight group
  int eid = sr % args.ep_nexperts;      // expert

  int Mi = ep_splits[eid];
  int M_acc = ep_splits_acc[eid] - Mi;

  problem_sizes[i] = cutlass::gemm::GemmCoord{Mi, new_N, K};                     // N sliced
  ptr_A[i] = ptr_with_offset(args.input[gid],  M_acc * K * input_elem_size);     // same rows
  ptr_B[i] = ptr_with_offset(args.weights[gid], (eid * N + sid * new_N) * K * input_elem_size);
  ptr_C[i] = nullptr;
  ptr_D[i] = ptr_with_offset(args.output[gid], (M_acc * N + sid * new_N) * output_elem_size);
  ...
  ldd[i] = LayoutD::packed({(int)Mi, (int)N}).stride(0);   // leading dim of the FULL tensor
}
```

Note `ldd` uses the **full** `N` stride: all subproblems write into the *same* shared output
tensor, each into its own column window — the shared tensor is decomposed logically, never
copied. (B is column-major `RCR`, so a `sid * new_N` column window of the weight is a plain
pointer offset.)

`n_split` is clamped so each slice is a whole number of consumer tiles
(`src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc:73` and `:528-533`):

```cpp
static constexpr int kTileSizeM = 128, kTileSizeN = 1024;
...
static int n_split_fixed(int n_split, int n_dim) {
  if (n_dim / n_split % kTileSizeN != 0) {
    n_split = n_dim / kTileSizeN;   // fall back: one split per 1024-wide tile
  }
  return n_split;
}
```

This enforces the paper's first rescheduling principle — *"rescheduled sub-tensors should align
with the original computation tile granularity for computational efficiency"* — the slice width
stays a multiple of both the GEMM tile N and the consumer's `kTiledN = 1024`.

### 2.2 Per-slice ready flags

The dependency between producer and consumer is tracked at N-slice granularity in a small
barrier array whose layout is documented where it is sized
(`ths_op/gemm_grouped_v2_gather_rs.cc:489-494`):

```cpp
int get_barrier_size() const {
  return pad_to(this->n_split, 128) * 2  // 1st: ready flag per tile, 2nd: counter per split
         + this->ep_nexperts * this->n_split * this->max_input_groups;  // per-problem counters
}
```

---

## 3. RESCHEDULE: execute the grouped GEMM column-wise across experts

> Paper §3.1.2 / Figure 6: *"Instead of computing each expert sequentially, GroupGEMM operations
> are executed column-wise. This approach allows the reduction and communicate operations to
> proceed as soon as the first T_N columns of the shared tensors are computed. Without
> rescheduling, tokens could only be reduced after all experts have completed their
> computations."*

### 3.1 Problem order = split-major

The reschedule is encoded in one line of the workspace builder above: `sid = i /
problem_per_split`. Problems `0 … problem_per_split-1` are *all experts'* slice 0, then all
experts' slice 1, etc. CUTLASS's device grouped-GEMM problem visitor walks problems in index
order (`GroupScheduleMode::kDeviceOnly`, selected in
`src/moe_gather_rs/gemm_grouped_v2_gather_rs.hpp:102`), so the persistent threadblocks finish
`(expert 0..E, slice 0)` before moving deep into slice 1 — the column-wise (yellow→green→blue→grey)
execution order of the paper's Figure 6.

### 3.2 The producer signals each finished slice

The fused kernel counts finished tiles per problem, finished problems per split, and finally
releases the per-split flag — the hierarchical counter cascade at the end of every tile
(`src/moe_gather_rs/cutlass_impls/gather_rs_gemm_grouped_with_absmax.h:543` and `:550-576`):

```cpp
set_barrier_ptr(params, problem_idx, grid_shape.m() * grid_shape.n(), thread_idx);
...
CUTLASS_DEVICE
void set_barrier_ptr(const Params &params, int problem_idx, int problem_tile_count, int thread_idx) {
  __syncthreads();
  if (thread_idx == 0) {
    // barrier_ptr:
    //  [0, split_n) + offset=0        : 0/1 ready flag per split
    //  [0, split_n) + offset=split_n  : problem counters per split
    //  [0, problem_count) + offset=2*split_n : tile counters per problem
    int * tile_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128) * 2;
    int counter = atomicAdd(tile_counter_ptr + problem_idx, 1);
    if (counter == problem_tile_count - 1) {                  // this (expert, slice) done
      int problem_per_split = *params.non_empty_problem_count / params.n_split;
      int group_idx = problem_idx / problem_per_split;        // which N slice
      int * problem_counter_ptr = params.barrier_ptr + cutlass::round_nearest(params.n_split, 128);
      int problem_counter = atomicAdd(problem_counter_ptr + group_idx, 1);
      if (problem_counter == problem_per_split - 1) {         // ALL experts' slice done
        cuda::atomic_ref<int32_t, cuda::thread_scope_device>(*(params.barrier_ptr + group_idx))
            .fetch_add(1, cuda::memory_order_release);        // release the split flag
      }
    }
  }
}
```

(`non_empty_problem_count` is computed by the workspace kernel at
`workspace_helper.cu:104-115` so experts with zero tokens don't stall a split forever.)

### 3.3 The consumer starts as soon as the first slice is released

`topk_gather_rs_v2_kernel` (`src/moe_gather_rs/topk_gather_rs_v2.cu:625-691`) is launched
*concurrently* with the GEMM and iterates over N slices, blocking on exactly one flag per slice:

```cpp
CUTLASS_PRAGMA_NO_UNROLL
for (int sid = 0; sid < params.n_split; sid++) {
  Barrier::wait_eq(params.barrier[params.local_rank], threadIdx.x, sid, 1);  // wait split ready
  // hierarchical reduce-scatter: one intra-node ring per owner-node segment group.
  for (int g_iter = 0; g_iter < params.nnodes; g_iter++) {
    int g = (params.node_idx + 1 + g_iter) % params.nnodes;
    for (int stage = 0; stage < params.local_world_size; stage++) {
      for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
           blk_id += gridDim.x) {
        int blk_m = blk_id / n_tiles_per_split;
        int blk_n = blk_id % n_tiles_per_split;
        TopkGatherRsOp<...>{}(params, &smem_buf[0], blk_m, blk_n, sid, stage, g);
      }
    }
    ...
  }
}
```

So while the GEMM is producing slice `sid+1`, the consumer is already gathering, top-k-reducing
and reduce-scattering slice `sid` — the fine-grained overlap of Figure 6.

Inside `TopkGatherRsOp` (`topk_gather_rs_v2.cu:317-465`) each 128×1024 tile of one N slice does
the **gather + top-k reduce** directly out of the shared tensor, using the routing index staged
in shared memory:

```cpp
// load the routing_idx first to the shared memory
smem_idx[row] = params.routing_idx[routing_idx_start + row];
...
for (int row_l = wid, row_g = wid + row_start; row_g < row_end; ...) {
  for (int i = 0; i < kElemsPerPack; i++) { acc[i] = 0.f; }
  for (int topk = 0; topk < kTopk; topk++) {
    int64_t row_g_from = smem_idx[topk + row_l * kTopk];     // gather: token -> expert-output row
    for (int j = 0; j < kNumWeightGroups; j++) {
      pack.data = loadPack((T *)params.input_ptrs[j] + row_g_from * N + col_g);
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] += element_to_float(pack.elems[i]);           // top-k reduce
      }
    }
  }
  ...
  void *output_ptr = last_round ? ... : (void *)((T *)params.reduce_ptrs[rank_to] + row_off);
  if (stage == 0) { storePack(output_ptr, pack.data); }      // ring reduce-scatter:
  else {                                                     // add my partial to neighbor's
    pack_lr.data = loadPack((T *)(params.reduce_ptrs[params.local_rank]) + row_off);
    storePack(output_ptr, addPack(pack_lr, pack));
  }
}
```

The reduce-scatter itself is an intra-node **ring** over `reduce_ptrs` (peer CUDA-IPC buffers)
with per-tile system barriers (`tile_barrier_ptrs`, waited/released at
`topk_gather_rs_v2.cu:407-409` and `:456-460`), again at (128 × 1024)-tile granularity — i.e.
the *communication* is also decomposed along N and pipelined tile-by-tile inside each slice.
For `nnodes > 1` a finished `(remote node, slice)` chunk raises `group_flags`
(`topk_gather_rs_v2.cu:667-677`), and the host pushes it with one NVSHMEM
`putmem_signal` per chunk (`ths_op/gemm_grouped_v2_gather_rs.cc:391-413`) — the same
slice-granular dependency carried across nodes.

The argument struct that carries all of this is `TopKReduceGatherRSV2Arguments`
(`include/flux/args/moe_gather_rs.h:173-211`); the producer-side flags travel in
`GemmGroupedV2GatherRSArguments::{barrier, n_split, non_empty_problem_count}`
(`include/flux/args/moe_gather_rs.h:119-151`), wired to the kernel in
`src/moe_gather_rs/gemm_grouped_v2_gather_rs.hpp:153-173`.

---

## 4. Horizontal fusion: how producer and consumer share the GPU

On SM80, Flux runs the two sides as **two concurrent kernels on two streams** ("horizontal
fusion" — the V2 analogue of the paper's §3.2 adaptive workload assignment). The driver
launches the grouped GEMM on the main stream and the gather-RS kernel on `gather_rs_stream`,
linked by events (`src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc:818-838`):

```cpp
CUDA_CHECK(cudaEventRecord(this->gemm_start_event, stream));
CUDA_CHECK(cudaStreamWaitEvent(gather_rs_stream, this->gemm_start_event));

if (M_this_ep > 0) {
  gemm_op->run(args, ..., stream);                       // producer: grouped GEMM
} else {
  this->barrier.fill_(1);                                // no local tokens: release all splits
}
output = topk_reduce_scatter_op->run(                    // consumer: gather + topk + RS
    gemm_outs, output, ..., get_rs_threadblock_count(), (intptr_t)gather_rs_stream);
CUDA_CHECK(cudaEventRecord(this->gather_rs_done_event, gather_rs_stream));
CUDA_CHECK(cudaStreamWaitEvent(stream, this->gather_rs_done_event));
```

The SM budget is split explicitly: the consumer gets a fixed number of threadblocks
(`FLUX_RS_BLOCKS`, default 3 — `ths_op/gemm_grouped_v2_gather_rs.cc:100-103`), and the GEMM's
persistent grid is shrunk by exactly that amount via `sm_margin`
(`ths_op/gemm_grouped_v2_gather_rs.cc:811` and
`gemm_grouped_v2_gather_rs.hpp:186-198`):

```cpp
.sm_margin = sm_margin + get_rs_threadblock_count()};    // GEMM leaves SMs for the RS kernel
```

```cpp
int get_threadblock_count(int sm_margin) const {
  int num_multiprocessor = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  num_multiprocessor = ... (num_multiprocessor - sm_margin);
  return Gemm::maximum_active_blocks() * num_multiprocessor;
}
```

Tuning `FLUX_RS_BLOCKS` (and `n_split`) is this architecture's knob for balancing communication
vs. computation latency — the role the adaptive workload assignment plays in the paper.

---

## 5. Try it

```bash
source ./module.sh
# single node (4× A100): T * E must equal world size
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -T 4 -E 1
# two nodes:
salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 8 -E 1
```

The reference implementation in the test does grouped GEMM → full topk reduce →
`reduce_scatter_tensor` sequentially; the fused op's win is the per-N-slice pipelining
described above.

---

### TL;DR

| Optimization | Paper | SM80 code |
|---|---|---|
| **Decompose** shared tensor along **N** (top-k reduction couples rows along M) | §3.1.1, Fig. 4 | `n_split` column slices; every (expert × slice) is a separate grouped-GEMM problem writing a column window of the same output (`workspace_helper.cu:80-98`); slice width kept tile-aligned (`gemm_grouped_v2_gather_rs.cc:528-533`) |
| **Reschedule** GroupGEMM column-wise, consumer starts per slice | §3.1.2, Fig. 6 | Split-major problem ordering (`sid = i / problem_per_split`, `workspace_helper.cu:81`); tile→problem→split counter cascade releases a flag per finished slice (`gather_rs_gemm_grouped_with_absmax.h:550-576`); concurrently-running consumer waits per slice, then gathers + topk-reduces + ring reduce-scatters it (`topk_gather_rs_v2.cu:625-691`, `:317-465`) |
