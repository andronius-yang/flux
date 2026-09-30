# Handoff 50: the next integration optimization pass — ours' per-layer fixed cost in SGLang

Opened 2026-09-30 (from handoff 49 rounds 7-9 and the 09-30 discussion). Scope: lopep's MoE layer inside SGLang v0.5.3
(serving path `python/lopep/serving.py` + the fused ops), Qwen3-30B-A3B, 40 GB A100 nodes. Goal: cut the fixed
per-layer cost that loses the 1-2 MiB points, keeping every paper mechanism (placement, capacity-constrained routing,
band-triggered swap, hierarchical all-to-all-v, compute/communication overlap). Code = lopep `sglang-dev` (068e5e4 =
the binary behind every number below); numbers stay in this tree.

## 1. Why the small budgets lose (measured, handoff 49 round 9)

Per-layer MoE block (SGLang bracket, rank 0, includes the communication) fits t = a + b * tokens-per-GPU:

| cell | ours a / b | stock a / b | crossover | measured nearest point |
|---|---|---|---|---|
| 4n decode | 2.70 ms / 2.15 us | 0.34 ms / 4.56 us | ~980 tok (3.8 MiB) | 4 MiB 1.02x |
| 8n decode | 3.12 / 4.18 | 0.64 / 8.74 | ~540 (2.1 MiB) | 2 MiB 0.98x |
| 4n prefill | 3.12 / 1.39 | 1.02 / 5.2 | ~550 (2.2 MiB) | 2 MiB 0.99-1.06x |
| 8n prefill | 3.69 / 2.79 | 1.83 / 10.2 | ~250 (1 MiB) | 1 MiB 1.01x |
| 16n prefill (partial) | layer at 1 MiB: ours 7.05 vs stock 8.16 ms | | | 1 MiB 1.14x (tput), 1.20x (TTFT) |

Worked check (4n decode, 256 tokens): ours saves 256 x (4.56 - 2.15) us = 0.62 ms of per-token work and pays 2.36 ms
more fixed -> 1.74 ms per layer x 48 layers = 84 ms per step; measured 166.9 - 84.5 = 82.4 ms. The whole loss is the
fixed term. Stock's fixed cost is small because its transfer sizes are known before the layer runs (everyone gets
everything: two NCCL collectives + a few launches, no planning); its per-token cost grows with the GPU count. Ours
sends each pick only where needed, which needs a per-layer plan; that plan is the fixed cost. Evidence that the host
is on the critical path: Nsight (round 7) GPU idle 1.27 ms per layer-step waiting on the host, 479 CUDA calls per
layer vs ~82 for stock; turning ours' timing marks off (host-only work) gained ~10 % at 1 MiB. Ours' fixed cost also
grows with the GPU count (1 MiB prefill layer: 3.48 / 4.40 / 7.05 ms at 4n / 8n / 16n): replicated W x W planning,
more remote nodes per layer (NN - 1 puts / wire lanes), wider barriers.

## 2. Walkthrough of one layer (running example: 4n decode, 256 tokens/GPU, 16 GPUs, 128 experts, top-8, 10 slots/GPU)

Costs: clean ledger at S<=256 (80 GB, no-swap steps, 3.08 ms total, LOPEP_TIMING on); calls: Nsight 40 GB capture.
Each GPU must deliver 256 x 8 = 2,048 (token, expert) picks.

| # | step (code) | what it computes | why the system needs it | runs on / host dependency | cost |
|---|---|---|---|---|---|
| 0 | adapter entry (`integrations/sglang/lopep_sglang/layer.py`) | this GPU's tokens, top-k ids, gate weights; exact per-GPU counts gathered once per forward | DP attention gives every GPU a different count | host, once per forward | 0.02 ms |
| 1 | pad + loads (`serving.py:477-516`, `planner.py:92-107`) | copy into S_b-row buckets (next pow2), pad rows aimed at own experts (weight 0); per-expert histogram; NCCL all-gather of loads [R, G] | equal-shaped step for collectives + symmetric buffers; router and swap need global demand | GPU kernels, issued by Python (~230 us host code) | 0.24 ms, 25 calls |
| 2 | swap decision (`src/planner/swap_decide.cu`, one block of 256 threads) | band test (node max > (1+C) x node mean); if out of band, up to 8 exchanges/GPU; rewrites p2l/l2p in place; lane_arm + pad_rebuild kernels | paper's demand-shift mechanism; same step routes on the new placement | GPU; result block read after step 4's sync | ~8 us no swap; 200-350 us on swap steps (serial orbit) |
| 3 | route + exchange (`planner.py:110-153`, `src/planner/routing.cu`) | 4 router kernels: water-fill each pick onto an expert copy (share x (1+C)), vacate remote picks home when the budget allows; NCCL all-gather of routed slots + gate weights (~256 KB); tail graph -> GEMM group ids | load balance across copies + locality; the gather makes planning REPLICATED (every GPU computes the full traffic matrix, identical capacity decisions, no more communication) | GPU; must wait for step 1's gather (router needs global loads) | 0.20 ms, 26 calls |
| 4 | metadata + capacity check (`dispatch_gemm.cc:2283-2355` derive_routed_meta; `sort_util.cu`; `overlap.py:105-136`; `capacity.py`) | GPU meta_counts over all 16 x 2,048 picks: splits, sps[W,E], uc[W,W+NN] (dedup per GPU / per node); stable scatter (row of every pick in the destination's expert-grouped buffer); D2H 12.5 KB + cudaEventSynchronize; host numpy demands vs capacities | every buffer is fixed-capacity on the NVSHMEM symmetric heap: overflow corrupts a PEER; the check must precede any put and agree on every GPU; the host needs the counts because it issues the transfers | GPU kernels + ONE blocking host sync (0.28 ms, mostly waiting for the serial scatter: thread 0 of each block walks 2,048 entries, `sort_util.cu:406-426`, 163 us) + ~250 us host code | 0.45 ms |
| 5 | read swap result (`serving.py:421-475`) | parse the landed result block | GEMM kwargs for moved slots | host | 0.02-0.06 ms |
| 6 | dispatch (`dispatch_gemm.cc:832-1077` tables, forward_impl; design.md) | host tables (chunks, offsets, relay minimal-move water-fill, GEMM arrival gates) -> 1 upload; pack (7 index_select segments); stage1 / consumer_build / prepare_workspace; GEMM 1 (persistent, tiles spin on per-source arrival); wire: 3 intra-node P2P (copy engine), relay pulls (copy engine), 3 BLOCKING inter-node puts, 9 + 3 gateway forwards; NVSHMEM barrier; ALSO (plan_overlap mode 2, `overlap.py:214-223`) the combine's metadata derive: host tables + 4 uploads + compress_plan kernels | contiguous symmetric send regions for RDMA; a token crosses the network once per destination node and every NIC carries an equal share; compute overlaps arrival; the barrier protects buffer reuse | host issue ~180 calls; NVSHMEM lowers each intra-node put_signal to a data copy + an 8-byte signal copy + event pair; blocking puts >= 58 us each; barrier waits for the slowest GPU | 0.96 ms (~1 MB moved per GPU: latency, not bandwidth) |
| 7 | activation | SwiGLU | | GPU | 0.09 ms |
| 8 | combine (`gemm_combine.cc:2373` derive, `:1277` CombineWireImpl, `:1752-1924` ladders) | GEMM 2 in destination-node waves, epilogue writes gate-scaled rows destination-major; convergence (NVLink) to the same-local-rank sender; pre-reduce (one partial per (token, node)); 3 blocking wire puts (one per remote node); intra lanes; bucketed receiver sums arrivals; 2 barriers | Sum gate x expert output back home with minimal network bytes | host issue ~180 calls (publishes to and joins all 16 wire streams, 3 used; 4 memsets for a disabled feature); pre-reduce / pack mostly spin on peers | 0.92 ms |

Fixed-cost ingredients at 1 MiB: (1) host issue (~480 calls/layer); (2) the host round trip of step 4 (D2H, wait,
tables, uploads); (3) serial or chained latency (two dependent collectives, serial scatter, serial swap orbit on swap
steps); (4) network latency floors (blocking puts through the NVSHMEM proxy, 3 barriers per layer).

Why the host is in the loop (only (a) is structural):
- (a) host-issued operations take byte counts as arguments: copy-engine copies (`cudaMemcpyAsync`), the pack's
  `index_select`, `torch::empty` sizes (combine CSRs, `gc:688-691`), the `out_buf[:m]` slice; so the host must wait
  for the counts. (LOPEP_DEVICE_META=1 builds tables on the device but keeps the sync and the host loops for this
  reason — handoff 49 round 8 C4.)
- (b) the capacity check must finish before any put and agree everywhere.
- (c) orchestration is expressed as host API calls (streams, events, stream waits).
The grouped GEMMs are already device-driven (problem sizes read on the device, tiles gate on arrival flags).

## 3. Constraints and positions for this pass (user, 09-30)

1. **Intra-node data movement stays on copy engines, not SMs.** SM copy kernels are rejected as the default (they
   take SMs from the grouped GEMMs; the design deliberately keeps the post-launch wire SM-free, design.md
   "dispatch_gemm" item 4). Section 4 is the research on driving copy engines without the host main thread.
2. **One routing exchange instead of two** (user proposal, to be measured): gather the PRE-routing top-k ids +
   gate weights once, then every GPU computes the loads (histogram) and routes ALL ranks locally. Section 5.
3. Paper mechanisms unchanged; wire-ordering hard rule (CLAUDE.md invariant 5) unchanged; measure at b1/b4/b16
   budgets first (invariant 6) and at 1 / 2 / 4 MiB in serving (plan 6 methodology, 40 GB nodes).

## 4. Copy engines without the host main thread: what CUDA supports (researched 09-30)

CUDA 12.9 headers (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include`) and NVIDIA docs:

| mechanism | what it gives | limits | copy engine? |
|---|---|---|---|
| device-side `cudaMemcpyAsync` (dynamic parallelism, CDP2: `cuda_device_runtime_api.h:226`, `__cudaCDP2MemcpyAsync`) | a kernel issues a D2D copy of any size | needs -rdc + device runtime; the Programming Guide says a device runtime call such as cudaMemcpyAsync "may invoke a kernel" (nesting depth + 1) | NOT guaranteed: likely SM-executed; must be measured, not assumed |
| device graph launch (CUDA 12.0+; `cudaGraphLaunch` in device code, `cuda_device_runtime_api.h:290`; graph instantiated with `cudaGraphInstantiateFlagDeviceLaunch`) | a kernel launches a pre-built graph; graphs may contain kernel, MEMCPY, memset and child-graph nodes; "the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device" (peer copies allowed if accessible at instantiation) | copy sizes and pointers FIXED at instantiation: device-side updates exist only for KERNEL nodes (`cudaGraphKernelNodeSetParam / SetEnabled / SetGridDim`, `driver_types.h:3675-3680`); one pending launch per graph (+ one self-relaunch); graph must be uploaded first | undocumented for device-launched memcpy nodes: must be verified (Nsight: memcpy activity vs kernel) |
| conditional nodes (IF / WHILE 12.3+, SWITCH 12.8+; `cudaGraphSetConditional` from device code, `cuda_device_runtime_api.h:479`) | a planning kernel selects which body runs; bodies may contain memcpy nodes | copy parameters still fixed per node: data-dependent sizes only by selecting among pre-built variants (size buckets, or per-chunk IF nodes at fixed offsets); whether conditional nodes may sit in device-launched graphs is not stated | memcpy nodes of a HOST-launched graph are the ordinary copy-engine path |
| host proxy thread for copy engines (the NVSHMEM-proxy pattern applied to NVLink) | kernels write copy descriptors (exact sizes, offsets, peer) to a pinned ring; a dedicated CPU thread (not the Python main thread) polls and issues `cudaMemcpyAsync` / `cudaMemcpyBatchAsync` (12.8+) + signal writes on per-peer streams; GEMM tiles keep spinning on arrival flags | proxy detection + issue latency per batch (microseconds, to measure); one CPU core per GPU (Perlmutter: 16 cores per GPU); still host API calls, but off the critical path and with exact sizes | YES (ordinary host-issued copies). Precedent: "Multipath Memory Access" (arXiv 2512.16056): CPU worker threads dispatch copy-engine transfers, GPU spin kernels wait on flags |
| `cudaMemcpyBatchAsync` (host API, CUDA 12.8+) | many copies in one call (NCCL 2.28's copy-engine AllGather / AlltoAll use batched copies, "zero-SM") | host-issued, sizes host-known | YES; cheap intermediate step: collapses ~37 + 21 per-copy calls and their event pairs, but keeps the sync |
| inter-node: device-initiated NVSHMEM `putmem_signal` from a kernel | sizes from device memory; data still moves NIC to NIC by RDMA (on Slingshot through NVSHMEM's CPU proxy thread; IBGDA needs InfiniBand) — this is already what the `nvshmemi_proxy_rma_signal_entrypoint_blocking` kernels in the capture are | must pass the wire-ordering payload probe for the DEVICE-side call before any gate relies on it (hard rule) | RDMA offload retained |

Not a copy engine at all: the pack step (gather rows by index into contiguous segments) is SM work in any design
(copy engines do 1D/2D/3D contiguous or strided copies, not index gathers); it is a small kernel today and stays one.

Design consequences:
- Fixed-pointer copy nodes (graphs, conditional or device-launched) cannot target step-dependent offsets. They need a
  receive layout with FIXED per-(source, destination) regions (capacity-sized slots; the GEMM already reads received
  rows through a consumer index, so aliasing into fixed slots is compatible) and variable lengths chosen by size
  bucket (pow2 buckets: <= 2x bytes; finer buckets: more nodes) or per-chunk IF nodes.
- The host proxy keeps exact sizes and today's packed layout; it changes who issues the copies, not the layout.
- Recommended probes before choosing (1 node, 4 GPUs, 0.5 / 1 / 2.4 MB P2P copies, the plan-6 sizes):
  P1 device-launched graph memcpy node to a peer: copy engine or kernel (Nsight activity kind, SM occupancy), launch
     latency; P2 host-launched graph with a SWITCH node selecting size-bucketed peer copies, condition set by a kernel:
     latency from condition write to copy start; P3 proxy thread: device flag -> copy start latency, single copy and
     `cudaMemcpyBatchAsync` of 12-37 copies; P4 device-side `cudaMemcpyAsync` (CDP2): copy engine or kernel.

## 5. One routing exchange instead of two (user proposal)

Today (`planner.py:105-118`): gather loads d[R, G] -> route own picks (needs global d) -> gather routed slots + gate
weights [R, 2 x S x K]. The two collectives are serially dependent. Correction to the proposal's premise: the
pre-routing top-k ids of other GPUs are NOT known today (only their per-expert counts are). Proposal: gather the raw
top-k ids + gate weights ONCE (the same bytes as today's second gather), compute d locally (histogram of all ids,
pads included as today), run the swap decision, then route ALL R ranks locally; derive_routed_meta already consumes
the full [R x S, K] routing, so everything downstream is unchanged.

- Saves: one collective (~94-105 us GPU each at 4n) + ~25 host calls + the serial dependency.
- Costs: router work x R (16n 1 MiB: 64 x 256 x 8 = 131k entries, one thread each — small, to measure).
- BLOCKER (found 09-30): the device router is NOT bitwise deterministic per token. `route_kernel`
  (`routing.cu:231-253`) takes a ticket with `atomicAdd(&cnt[g], 1)`, and the vacate pass consumes release/extra
  budgets with atomics (`routing.cu:416-448`). Per-replica COUNTS are deterministic, but WHICH token lands on which
  replica depends on thread timing. Harmless today (each GPU routes its own picks once and ships the result); fatal
  for replicated routing (two GPUs could disagree on the same token -> wrong rows moved). Required change: a
  deterministic router with the same shares and budgets — ticket = stable ordinal of the entry among the source's
  entries of that expert (the stable-scatter ordinal, flat order), vacate in token-index order (the host
  `reference_route` in `routing.py:228` is already a deterministic all-ranks model to validate against). Paper
  semantics unchanged (shares, budgets, C); only tie-breaking becomes fixed.
- Validation: compare mode (every GPU's local routing of rank r == rank r's own routing, bitwise) over serving
  traffic at 4n / 8n; then measure route-all cost vs the saved collective at 1 / 2 / 4 MiB.

## 6. Proposed order for the pass (each stage measured on its own, plan-6 methodology)

| stage | change | removes | risk |
|---|---|---|---|
| A | parallel stable scatter (bitwise-identical output), shared-memory meta_counts, parallel swap orbit | ~0.15-0.17 ms/layer GPU on the sync's critical path; up to ~0.3 ms on swap steps | low |
| B | deterministic router + one exchange (section 5) | one collective + its host calls per layer | medium (determinism) |
| C | all tables on the device (dispatch + combine + receiver lanes; extend LOPEP_DEVICE_META), drop duplicated / unused host work (combine re-derives dispatch tables, C computed 3x, unused uploads / memsets, 16 joined wire streams for 3 used) | host table loops, most uploads, ~40 event calls | low-medium |
| D | copy issue off the main thread: batched copies (`cudaMemcpyBatchAsync`) first, then the proxy thread or graph-selected copies per the section-4 probes; device-initiated inter-node puts after the wire probe; device capacity verdict (identical on every GPU because planning is replicated), read at the end of the forward, redo the forward on overflow (0 growths so far) | the planning sync and most of the ~480 calls | high (redesign) |
| E | whole-layer CUDA graph per bucket (launch sequence fixed once sizes are device-resident; per-step epoch moved to device memory) | per-layer host issue down to ~1 launch | medium (NVSHMEM ops in graphs) |

What remains after E: collective latency (one gather), put latency through the NVSHMEM proxy, barriers (3 per layer;
round 8 code reading suggests 1 suffices for the write-after-read hazard — unproven, needs a proof + randomized-payload
stress). Measure each stage's fixed term (the a in t = a + b n) at 4n / 8n; success = the crossover moving below 1 MiB.

## 7. Where to look

- Evidence: handoff 49 rounds 7-9 (fits, Nsight deep dive, code map, plan-6 tables); analyzers `49_hostcalls.py`,
  `47_gap_report2.py`; captures `logs/sglang/nsys49/`.
- Drivers for re-measuring: `logs/sglang/jobD6.sh`, `jobP6.sh`, `jobC6.sh`, `run16.sh`, `chainfig2.sh`
  (`GPU_C="gpu&hbm40g"`, regular QOS for 8n / 16n, short pieces).
- Sources (NVIDIA): CUDA Programming Guide, "CUDA Graphs" (device graph launch, conditional nodes) and "CUDA Dynamic
  Parallelism"; blog "Enabling Dynamic Control Flow in CUDA Graphs with Device Graph Launch"; blog "Fusing
  Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28"; arXiv 2512.16056.
