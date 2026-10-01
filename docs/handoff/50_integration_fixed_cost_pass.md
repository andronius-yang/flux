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
2. **One routing exchange instead of two** (user proposal): gather the PRE-routing top-k ids + gate weights once,
   every GPU computes the loads and routes ALL ranks locally. Already tried as "route-global" on 08-29 (handoff 26)
   and CLOSED as a net loss; section 5 has the history and why it stays last in this pass.
3. **Order (user, 09-30): probe device-initiated copies FIRST, then tables on the GPU and the copy path; the routing
   merge only after those, and only if new evidence reopens it.** A device-initiated copy may need an SM (a thread)
   to issue it, but its throughput must not depend on how many SMs are given to copying (section 4a).
4. Paper mechanisms unchanged; wire-ordering hard rule (CLAUDE.md invariant 5) unchanged; measure at b1/b4/b16
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

## 4a. Probe first: are device-initiated intra-node copies independent of SMs? (stage 0, before any redesign)

Acceptance criterion (user, 09-30): a mechanism may use an SM (a thread) to ISSUE a copy, but the copy's throughput
must not be limited by the SMs dedicated to copying, i.e. it must run on a copy engine. Setup: 1 node, 4 A100 40 GB,
peer copies of 0.5 / 1 / 2.4 MB (the plan-6 per-copy sizes) and 16 MB (bandwidth), NVSHMEM symmetric buffers and
plain peer-mapped buffers.

| probe | mechanism | measure |
|---|---|---|
| P0 | host `cudaMemcpyAsync` (today's copy-engine baseline) and an SM copy kernel with k = 1, 2, 4, 8, 16 SMs | reference bandwidth / latency curves; the SM kernel's bandwidth grows with k, a copy engine's does not |
| P1 | device-launched graph (`cudaGraphLaunch` from a kernel, fire-and-forget) with one peer memcpy node | Nsight activity kind (MEMCPY on a copy engine vs a kernel), launch-to-start latency, bandwidth |
| P2 | host-launched graph with a SWITCH / IF node choosing among size-bucketed peer memcpy nodes, condition set by a planning kernel (`cudaGraphSetConditional`) | condition-write-to-copy-start latency, bucket overhead |
| P3 | host proxy thread: kernel writes descriptors to pinned memory, a dedicated CPU thread issues `cudaMemcpyAsync` / `cudaMemcpyBatchAsync` (12-37 copies) | device-flag-to-copy-start latency, CPU cost, bandwidth |
| P4 | device-side `cudaMemcpyAsync` (CDP2, -rdc) | copy engine or kernel (Nsight), latency |

SM-independence test for every probe: run it (a) alone and (b) concurrently with a persistent kernel that occupies
EVERY SM (a spin kernel, and separately the real dispatch GEMM at sm_margin 0). Pass = bandwidth and latency within
noise of (a) AND Nsight shows a MEMCPY activity (no extra kernel doing the movement). A mechanism that slows down
when the SMs are busy, or scales with the SMs it is given, is SM copying in disguise and fails the constraint.
Also record `cudaDevAttrAsyncEngineCount` (how many copy engines can run concurrently) and whether concurrent copies
to 3 peers overlap.

### 4a-R. Stage 0 results (09-30, 1 node nid001553, job 59144224; sources + CSV in `50_copy_probes/`)

Setup: A100-SXM4-40GB x4, driver 580.178 (CUDA 13.0 capable), CUDA 12.9 runtime (the sglang-dev toolchain),
`cudaDevAttrAsyncEngineCount = 5`; one process per GPU, CUDA-IPC peer mappings (what NVSHMEM uses intra-node with
`NVSHMEM_DISABLE_CUDA_VMM=1`) and NVSHMEM 3.2.5 symmetric buffers (`nvshmem_ptr`), identical results on both.
Per-copy sizes 0.5 / 1 / 2.4 / 16 MB; medians of 25 iterations, rank 0; full table `50_copy_probes/results.csv`.

| probe | mechanism | verdict | key numbers (idle SMs) |
|---|---|---|---|
| P0 | host `cudaMemcpyAsync` to a peer | copy engine (reference) | 9.3 / 14.8 / 30.2 / 182 us for 0.5 / 1 / 2.4 / 16 MB = ~4 us + bytes / 95 GB/s; **5.3 us of host time per call**; 3 peers at once from one GPU: 256 GB/s aggregate (3 engines in parallel, no slowdown per copy) |
| P0 | SM copy kernel, k blocks x 1024 threads | SM-bound, FAILS the constraint | 29 GB/s at k = 1, 56 at k = 2, 85 at k >= 4 (saturates NVLink); throughput scales with the SMs it is given |
| P1 | device-launched graph (`cudaGraphLaunch` from a kernel) | **UNSUPPORTED here**: launching the kernel that calls `cudaGraphLaunch` returns `cudaErrorNotSupported` on this A100 / driver, after a successful `cudaGraphInstantiateWithFlags(DeviceLaunch)` + `cudaGraphUpload` | n/a |
| P2 | host-launched graph, planning kernel sets a conditional (IF chain of size buckets / SWITCH, CUDA 12.8) | copy engine; conditional overhead small | condition write -> copy start 6.5 us (SWITCH, 4 buckets), 7.1 us (one IF), 8.5 us (4 IF nodes); the copy itself runs at the P0 rate (0.5 MB 9.5 us, 16 MB 193 us); host launch 5 us (SWITCH) / 8.7 us (4 IFs) per graph |
| P3 | host proxy thread (kernel writes descriptors to pinned memory, a pinned CPU thread issues the copies) | copy engine | device flag -> copy done 20 us for one 0.5 MB copy (= 7 us detection, 5.6 us issue, 9 us copy); one `cudaMemcpyAsync` per copy costs the proxy 3.6 us (12 copies 43 us, 37 copies 127 us); `cudaMemcpyBatchAsync` (12.8+) 2.6 us per copy (12 copies 31 us, 37 copies 91 us); 12 x 2.4 MB: batch 288 us vs loop 471 us |
| P4 | device-side `cudaMemcpyAsync` (CDP2, `-rdc`) | **SM copy kernel (Nsight: `memcpy128`), FAILS**; 88 GB/s at 16 MB like a many-block SM copy, plus ~35 us of fixed latency per call (0.5 MB: 39 us vs 9 us host) | |
| P5 | `cuStreamWriteValue64` to a peer-mapped signal word after the data copy; receiver kernel `ld.acquire.sys` then verifies every payload word (payload = epoch, new every iteration) | **accepted and ordered**: 0 API errors, 0 timeouts, 0 payload violations in 60 iterations x 3 sizes x 2 buffer kinds; the 8-byte `cudaMemcpyAsync` fallback also 0/0/0 | |

Two findings that bind the stage C design:
- **Lazy module loading deadlocks against a spinning kernel.** With `CUDA_MODULE_LOADING=LAZY` (the CUDA 12 default,
  and what `bench/launch.sh` exports), the FIRST launch of any kernel loads its module, and that load waits for the
  GPU to drain; a persistent kernel that spins on a flag never drains, so the launching thread blocks forever (the
  busy-SM passes hung at exactly this point until `CUDA_MODULE_LOADING=EAGER`; the idle passes had loaded everything
  first). The GEMM tiles spin on arrival signals, so any kernel a stage-C issuer launches for the first time while a
  GEMM is resident (descriptor kernel, pack, signal kernels) must be warmed up at init or loaded eagerly.
- **Warp-scheduler starvation is real for co-resident kernels**: a resident block whose warps are always ready (a
  pure FMA spin, or every warp polling one L2 line) stops a younger co-resident block from issuing at all (a
  4 x 1024-thread copy took 845 ms instead of 0.17 ms next to 108 polling blocks, `50_copy_probes/cores.cu`).
  A memory-stalled GEMM does not behave like that, but an SM-based copy mechanism inherits whatever the GEMM leaves.

Decision (unchanged from the plan, now evidence-backed): the stage C issuer is the **host proxy thread** issuing
ordinary `cudaMemcpyAsync` / `cudaMemcpyBatchAsync` (P0 rate, zero SMs, exact sizes from device memory) with
`cuStreamWriteValue64` signals (P5). P1 is unavailable; P2 (conditional graphs) is a valid fixed-shape fallback with
~7 us of extra latency per graph; P4 is a copy engine at the cost of ~35 us per call and keeps a kernel resident.
Batching is worth it: 37 x 0.5 MB copies issue in 91 us batched vs 127 us looped vs ~196 us from today's per-copy
`nvshmemx_putmem_signal_nbi_on_stream` host calls (5.3 us each).

**Nsight activity kinds (`nsys profile -t cuda --cuda-graph-trace=node`, idle SMs, P2 + P3 + P4, 3 iterations;
reports `logs/p50/nsys_p234_<rank>.nsys-rep`):** the P2 graph memcpy nodes and every P3 proxy copy appear ONLY as
`[CUDA memcpy Peer-to-Peer]` activities (780 of them, no copy kernel) = copy engines. The P4 device-side
`cudaMemcpyAsync` appears as SM kernels: `memcpy128` (48 instances, avg 103 us) + `memcpy32_post` (60 instances),
exactly one per P4 call. **P4 is an SM copy in disguise and FAILS the constraint**; its "copy-engine bandwidth" was
the bandwidth of a many-block copy kernel. Verdict column above corrected accordingly.

**Busy-SM pass (job 59150611, 10-01, HBM-saturating busy grid = 212 resident 1024-thread blocks streaming a 1 GB
buffer with read-modify-writes, every SM issuing memory ops; probe warmed up idle first, `CUDA_MODULE_LOADING=EAGER`):**

| mechanism | bytes | blocks | idle us | HBM-saturated us | slowdown |
|---|---|---|---|---|---|
| host_memcpy | 0.5 MB | 1 | 9.3 | 18.7 | 2.0x |
| sm_copy | 0.5 MB | 1 | 22.8 | 174.5 | 7.7x |
| sm_copy | 0.5 MB | 2 | 15.4 | 101.4 | 6.6x |
| sm_copy | 0.5 MB | 4 | 12.8 | 77.0 | 6.0x |
| sm_copy | 0.5 MB | 8 | 12.5 | 67.4 | 5.4x |
| sm_copy | 0.5 MB | 16 | 12.7 | 69.8 | 5.5x |
| sm_copy | 0.5 MB | 32 | 11.8 | 69.6 | 5.9x |
| host_memcpy | 1.0 MB | 1 | 14.8 | 25.3 | 1.7x |
| sm_copy | 1.0 MB | 1 | 39.0 | 398.3 | 10.2x |
| sm_copy | 1.0 MB | 2 | 23.5 | 202.8 | 8.6x |
| sm_copy | 1.0 MB | 4 | 18.9 | 111.4 | 5.9x |
| sm_copy | 1.0 MB | 8 | 18.4 | 132.1 | 7.2x |
| sm_copy | 1.0 MB | 16 | 18.8 | 114.7 | 6.1x |
| sm_copy | 1.0 MB | 32 | 18.2 | 117.7 | 6.5x |
| host_memcpy | 2.4 MB | 1 | 30.2 | 47.5 | 1.6x |
| sm_copy | 2.4 MB | 1 | 84.4 | 874.9 | 10.4x |
| sm_copy | 2.4 MB | 2 | 46.4 | 484.8 | 10.4x |
| sm_copy | 2.4 MB | 4 | 35.5 | 244.4 | 6.9x |
| sm_copy | 2.4 MB | 8 | 35.2 | 245.1 | 7.0x |
| sm_copy | 2.4 MB | 16 | 35.5 | 256.2 | 7.2x |
| sm_copy | 2.4 MB | 32 | 35.4 | 284.5 | 8.0x |
| host_memcpy | 16.0 MB | 1 | 182.2 | 270.1 | 1.5x |
| sm_copy | 16.0 MB | 1 | 570.8 | 5613.7 | 9.8x |
| sm_copy | 16.0 MB | 2 | 297.0 | 2779.1 | 9.4x |
| sm_copy | 16.0 MB | 4 | 198.9 | 1526.9 | 7.7x |
| sm_copy | 16.0 MB | 8 | 197.4 | 1509.2 | 7.6x |
| sm_copy | 16.0 MB | 16 | 196.9 | 1536.5 | 7.8x |
| sm_copy | 16.0 MB | 32 | 196.4 | 1431.5 | 7.3x |

The copy engine slows only by the HBM bandwidth it has to share (1.5 to 2x); the SM copy kernel slows 5 to 10x at
every block count, because it competes for issue slots and residency as well as for bandwidth. With
a resident busy grid that holds every slot but does not touch memory (FMA bursts + nanosleep), both the host memcpy
and the SM copy ran at the idle rate (the sleeping warps leave the schedulers free), which says the SM copy's cost
is set by what the resident GEMM leaves it, while the copy engine's is not. The P3 proxy-thread pass under the
busy grid wedged twice inside the probe's own issue path (the planning kernel + host polling) and is not reported;
its idle numbers and the P0 host-memcpy busy numbers (the proxy issues exactly those calls) stand in for it.
The real-GEMM-at-sm_margin-0 variant was not run: the serving stage check measures that condition directly.

### Stage A1 + B4 landed (10-01; lopep sglang-dev 5cafd2d = metadata chain, 3011c6f = dead work)

- A1: `derive_routed_meta` is three launches (per-source whole-token tiles -> one block scan -> warp-level stable
  scatter) with no memsets and no global atomics, replacing 5 launches + 3 memsets; op-free bindings
  `C.routed_meta` / `C.a2av_demands`; `tests/test_meta_device.py` = 104 routings bitwise identical to
  `routing.meta_from_virtual_route` and `capacity.demands_from_meta` (W 4-64, S 8-1024, K 8 and 6, adversarial
  routings), plus every capacity-violation bit.
- B4 (dispatch): deleted `cp_stream_signal`, `hier_dispatch_event_` (+ its deferred-op kind), `relay_send_event_`,
  the empty `f1_quiet`, the whole-pack `pack_ready` announce (the relay pulls gate on the per-segment signals since
  wave-pack) and the six `t_*` / `issue_put` / `stage_off_u` lambdas nothing called. `ready_event` and the
  `wave_pack_` flag STAY: wave-pack is set only when the relay is built (nnodes > 1); at one node the whole-pack
  event is the live path (found by the 1n gate). B4 (combine): the four per-step memsets of the piece buffers go
  (pieces are off, `a2av_piece_config` want = 0; the buffers stay, zeroed at init). NOT done from the B4 list: the
  lane fan-out/join (the lane choice `(sid * (NN-1) + gi) % S` does use every lane when S > NN-1, so the join is
  live), the `record_stream` and pinned-table items (stage B0).
- Deferred explicitly, not dropped: A1b (expert of a copy = `routing_ids[p]`, deleting the stage-1 binary search
  and the combine `searchsorted` chain) folds into stage B3 where the combine tables move to the device anyway;
  A2 (parallel swap orbit; swap steps only) after stage B.
- Gates (job 59150611, 10-01): unit 4/4 PASS (`test_meta_device`, `test_capacity_host`, `test_swap_decide`,
  `test_lane_device`); 1n serving harness `static`, `staged_ref_dev`, `staged_cmp`, `s4096_swap_dev`,
  `flip_staged` PASS (forced growth via `--caps-scale 0.5` included); `flip_ref` and `devmeta_ref` (reference on,
  popularity flip every 5 steps, forced growth) report ONE bad row of 1.47 M (rank 0, step 50, err 0.0119 vs atol
  1e-2, every rank's max_err 0.012-0.013) — the plan-6 binary 068e5e4 reports the identical row on the same
  allocation, so it is a pre-existing bf16 tolerance edge of that seed, not a metadata change (`devmeta_ref`'s
  device-vs-host assertions all pass). 4n gates (job 59150611): `staged_ref_dev`, `staged_cmp`, `s4096_swap_dev`, `flip_staged` PASS; `devmeta_ref`
  (reference + compare mode, 60 steps x 3 layers at 16 ranks) reached only step 0 inside the 300 s cap with no
  mismatch or error — inconclusive by time, to be rerun with `--steps 10` on the next allocation.

### Stage check after A1 + B4 (10-01, job 59150611, 4 nodes 40 GB, plan-6 protocol with frozen pins; `49_round7_*.csv`)

Decode (LCB t48, OSL 48, 2 waves; step = median decode step over 192 intervals; layer = SGLang bracket of the MoE
block, rank 0, mean over ~4.5k layer-steps). All four arms on ONE allocation; "ours" twice, both runs listed.

| running / rank | plan-6 binary 068e5e4 | A1+B4 3011c6f (run 1 / 2) | cut per step | cut per layer | stock graphs-off | ours vs stock (step) |
|---|---|---|---|---|---|---|
| 256 (1 MiB) | 166.69 ms / 3.209 ms | 154.77, 155.77 / 2.994, 3.014 | -11.4 ms (-6.8 %) | -0.21 ms | 85.49 / 1.510 | 0.55x (plan 6: 0.51x) |
| 512 (2 MiB) | 187.62 / 3.591 | 178.16, 178.54 / 3.391, 3.396 | -9.3 (-4.9 %) | -0.20 | 143.35 / 2.603 | 0.80x (0.77x) |
| 1024 (4 MiB) | 261.22 / 4.933 | 248.33, 249.05 / 4.674, 4.670 | -12.5 (-4.8 %) | -0.26 | 270.49 / 4.986 | 1.09x (1.02x) |

Fixed term (a + b x tokens per GPU over the three points, rank-0 layer): ours a = 2.43 ms, b = 2.19 us/token (plan 6:
2.70 / 2.15); the cut is flat in the token count, i.e. fixed cost, 0.27 ms per layer-step, from the metadata chain
(A1: 5 launches + 3 memsets -> 3 launches, the serial scatter walk gone) and the dead work (B4). The ours-vs-ours
spread is 0.6 % (step) / 0.7 % (layer), the cut 5-7 %: a real effect. Verdict: stage A/B4 PASS (no cell slower;
the drop is inside the A1 + B4 estimate of 0.2-0.45 ms, A2 and A1b still deferred). Distance to the 1 MiB target:
ours 2.99 ms per layer vs 1.37 needed (stock 1.51 / 1.1), i.e. the fixed term must still fall by ~1.6 ms at 4n decode;
the levers left are the host issue (stage C) and the GPU floors (stage D).

Prefill (LCB eval, full prompts, MAXRR 32/rank, OSL 4, concurrency 16/rank, pin 50000; throughput = input tokens/s
of the benchmark; layer = SGLang bracket, rank 0, mean over ~3.7k layer-steps; TTFT = mean). Same allocation.

| SMAX / rank | plan-6 binary (tok/s / layer ms) | A1+B4 run 1, run 2 | throughput vs plan-6 binary | stock graphs-off | ours vs stock (throughput) |
|---|---|---|---|---|---|
| 256 (1 MiB) | 17,654 / 3.331 | 18,749, 19,022 / 3.127, 3.032 | +6.2 %, +7.7 % | 27,225 / 2.212 | 0.69x, 0.70x (plan 6: 0.65x) |
| 512 (2 MiB) | 27,782 / 3.612 | 30,463, 29,438 / 3.342, 3.234 | +9.7 %, +6.0 % | 33,484 / 3.501 | 0.91x, 0.88x (plan 6: 0.99-1.06x, other allocation) |
| 1024 (4 MiB) | 38,375 / 4.344 | 41,248, 41,283 / 4.153, 4.233 | +7.5 %, +7.6 % | 38,005 / 6.082 | 1.085x, 1.086x (plan 6: ~1.0x) |

The prefill layer time fell by 0.2-0.3 ms at every SMAX (same order as decode); throughput +6-10 % over the plan-6
binary on the same allocation, with the ours-vs-ours spread 1.5-3.4 %. TTFT is noisier (mean over few requests).
Stage A1 + B4 verdict stands for prefill too; no cell slower.

### Stage C1 written (10-01, uncommitted until gated): batched copy-engine issue with host-known sizes

Why before stage B: the stage-B maps (`50_stageB_dispatch_map.md`, `50_stageB_combine_map.md`) show the host table
loops are small at 4n (W = 16) and the device arena already exists (LOPEP_DEVICE_META=1, unused in serving); the
fixed-cost levers measured in §1-2 are the host ISSUE (~480 CUDA calls, 5.3 us each in P0) and the planning sync.
C1 attacks the issue count and removes SM copy kernels; it keeps today's sizes, order and gating (L5-L7).

- `include/flux/cuda/ce_batch.h`: `CeBatch` (one `cudaMemcpyBatchAsync` per group of intra-node copies; a loop of
  `cudaMemcpyAsync` below CUDA 12.8) and `CeSignals` (SET signals as `cuStreamWriteValue64` at the peer-mapped
  slot from `nvshmem_ptr`, issued after the batch on the same stream = NVSHMEM's own intra-node lowering, P5-proven);
  a peer without a P2P mapping falls back to the NVSHMEM call; knob `LOPEP_CE_BATCH` (default 1, 0 = today's path).
- Dispatch: (a) `issue_deferred_wire` coalesces the self copy and the round-0 intra-node puts into one batch and
  their signals after it (a front-end wait flushes what precedes it); (b) the gateway forwards: per source node one
  batch of the L windows + L signal writes, replacing L BLOCKING intra-node `putmem_signal_on_stream` calls that ran
  a device kernel each (`flux_rs_put_signal`'s comment); (c) relay pulls: every readiness wait of the round first,
  then the round's pieces as one batch (the round's put needs every piece anyway, so the coarser gate loses nothing).
- Combine: the convergence ladder (L copies + L signals per (tn, sid)) and the intra-node ladder (self + L-1 peers
  per split) each become one batch + signal writes.
- Untouched: every inter-node put (blocking, CLAUDE.md invariant 5), the proxy `getmem` fallback, all waits.
- Expected per layer at 4n: ~40 fewer host calls plus the 12 gateway SM-copy kernels gone; gate = unit tests,
  1n/4n serving harness with the reference on (payload changes every step), then a stage check against the A1+B4
  binary on one allocation.

### Stage C1 measured (10-01, job 59151902, same protocol; C1 binary vs the A1+B4 binary on the same nodes)

| decode, running / rank | A1+B4 (step / layer) | C1 run 1, run 2 (step / layer) | change |
|---|---|---|---|
| 256 (1 MiB) | 156.24 ms / 2.999 ms | 154.51, 154.98 / 3.048, 2.987 | step -1.0 %, layer +0.6 % (noise) |
| 512 (2 MiB) | 179.35 / 3.417 | 181.62, 179.59 / 3.498, 3.419 | +0.7 % / +1.2 % (noise) |
| 1024 (4 MiB) | 249.38 / 4.702 | 250.75, 248.98 / 4.709, 4.697 | +0.2 % / 0.0 % |

C1 is neutral: batching the intra-node copies and replacing the 12 gateway put kernels by copy-engine batches did not
move the layer time. The issue calls it saved were not on the critical path (they run while the GEMM computes), and
the copies were already on the copy engines. Gates passed (unit 4/4; 1n 5/6 + the pre-existing tolerance row; 4n
4/4). By the plan's rule a neutral stage is reverted; C1 is kept in the tree ONLY because C2b issues the same batches
from the proxy, and its fate is decided by the C2b measurement (C2b with `LOPEP_CE_BATCH=1` vs `0`): if C2b gains
and the batches do not contribute, C1 is reverted before any commit. Prefill (same allocation): 1 MiB 19,641 / 19,098
vs 18,566 tok/s (+3-6 %), 2 MiB 29,928 / 29,275 vs 29,125 (+0.5-2.8 %), 4 MiB 42,052 / 40,350 vs 41,953 (0 to -4 %):
neutral within the prefill spread. Rows: `49_round7_*.csv`, alloc_job 59151902 (the "ours_plan6_binary" arm of that
job is the A1+B4 binary, the comparison binary of that check).

### Stage C2b written (10-01, uncommitted until gated): the wire-issue proxy thread

Shape chosen: the main thread keeps every offset and size computation (the host tables of the stage-B maps) and
hands the API CALLS to a dedicated thread, as a program of closures with pointers and sizes captured by value,
executed in posting order. This generalizes the op's own deferred-wire replay (`DeferredWireOp`) and keeps L7:
the program order and every wait are the ones the main thread issued before; only the issuing thread changes.
- `src/core/wire_proxy.{h,cc}`: process-wide `WireProxy` (lazily created on the first post, `cudaSetDevice` of
  the creating thread, pinned to the last core of the task's affinity mask, spin then 1 ms condvar waits after
  2 ms idle); `post(fn)` (inline when `LOPEP_WIRE_PROXY=0`), `quiesce()` (every posted step issued), `shutdown()`
  (quiesce + join; also the static destructor and a Python `atexit` through `C.wire_proxy_shutdown`); a failure
  inside the proxy is stored and rethrown on the main thread at the next post / quiesce.
- Ownership: dispatch `cp_stream`, `cp_stream_inter_node`, `pull_streams_[0]`, the `pack_stream_` tail; combine
  `a2av_intra_stream_`, `internode_stream`, `internode_streams2_[]`, `a2av_conv_stream_`. Main-owned and unchanged:
  the forward streams, `pack_str`, the combine `reduce` and `prered` streams (they carry main-launched kernels).
- Edges: main-before-proxy = events the main thread records before posting (`pack_seg_events_`, `ready_event`,
  `staging_reset_event`); proxy-internal = the existing events (`relay_pull_events_`, `relay_put_events_`,
  `fetch_remote_event`, `a2av_inter*_done_`, `a2av_conv_done_`); main-after-proxy = a device word per op
  (`wire_done_`, `CUStreamWriteValue64(run_id)` by the proxy after its last op, `CUStreamWaitValue64(GEQ run_id)`
  by the main stream at the GEMM-launch join in the dispatch and at the tail join in the combine), so the enqueue
  order between the threads is irrelevant. Growth: `resize_capacities` quiesces first; destructors quiesce before
  their streams go.
- NVSHMEM thread level (found by the first gate run, 10-01 00:25): the Perlmutter NVSHMEM 3.2.5 build grants at
  most `NVSHMEM_THREAD_SERIALIZED` (provided = 2 when MULTIPLE is requested), so a THREAD_MULTIPLE init aborts every
  rank at startup. SERIALIZED allows several threads, one call at a time: the init stays the stock request and every
  NVSHMEM host call runs under one process-wide recursive mutex (`nvshmem_host_mutex()`), held by the proxy for each
  posted step and taken by the main thread at its five in-step call sites (dispatch pack announce, dispatch barrier,
  combine group barriers, the Python barrier binding, peer-pointer lookups).
- Not changed: sizes stay host-known (C3 = device-sized puts needs stage B's plan blocks), the capacity verdict
  stays synchronous (C4), the 3 barriers stay (D1).
- Gate: unit tests; 1n / 4n serving harness with the reference on and forced growth, `LOPEP_WIRE_PROXY=1` and 0;
  then a stage check against the C1 binary on one allocation.

### Stage D1 audit, first pass (10-01; argument only, no change yet)

Per layer-step on the main stream of every rank: dispatch wire -> GEMM 1 -> **B_d** (`dispatch_gemm.cc` end of
forward_impl) -> activation -> **B_c1** (`gemm_combine.cc` before GEMM 2) -> GEMM 2 -> lanes -> join -> **B_c2**.
Cross-rank buffer hazards and which barrier orders them:
- H1 dispatch(k+1) puts into d's recv / relay staging vs d's GEMM 1(k) reads: any barrier after GEMM 1(k) and before
  dispatch(k+1) on d; B_c1(k) and B_c2(k) both qualify, B_d is not needed for it.
- H2 relay pulls of step k read a peer's send buffer vs that peer's pack(k+1): pulls precede GEMM 1(k); same cover.
- H3 combine(k) puts into d's recv panel vs d's bucket reduce(k-1) reads; H4 conv puts vs the gateway's pre-reduce(k-1)
  reads: B_c2(k-1) orders both (d passes it only after its step k-1 combine, join included), so B_c1(k) is not needed.
- H5 send-panel / conv-panel reuse by GEMM 2(k): the readers are this rank's own lanes of step k-1, joined into the
  main stream through the done word before B_c2(k-1): local, no barrier needed.
- "Quiet our outstanding nbi puts" (B_d's comment): after C1 every intra-node transfer is a copy-engine batch,
  complete when its stream passes it; inter-node puts are blocking; the only nbi call left is the getmem fallback
  for a peer without a P2P mapping (never on one node). Epoch signals are never reset.
Conclusion to prove: ONE barrier per layer (B_c2) covers H1-H5; B_d and B_c1 are removable (2 x ~0.27 ms at 4n).
Required before removal (plan, stage D): the argument extended over swap steps (lane pushes / commits touch the
weight staging, not these panels) and forced growth (`resize_capacities` device-syncs on every rank), a
randomized-payload stress at 4n/8n/16n with swaps and `--caps-scale 0.5`, and an epoch-tagged panel check.

## 4b. Where tables on the GPU and the copy path stand without the routing merge

Both are independent of the routing exchange: derive_routed_meta already runs on the gathered routing of all ranks,
so the tables can move to the device with today's relaxed router and two collectives unchanged. Tables on the device
alone (stage B) remove the host table loops and uploads but NOT the planning sync while copies are host-issued with
host-known sizes; the sync goes away only with the copy path of stage C (a mechanism from 4a that takes sizes from
device memory, or the proxy thread) plus the deferred capacity verdict.

## 5. One routing exchange instead of two: already tried, closed, kept last

Today (`planner.py:105-118`): gather loads d[R, G] -> route own picks (needs global d) -> gather routed slots + gate
weights [R, 2 x S x K]; serially dependent. The proposal gathers the raw top-k ids + weights once (the pre-routing
top-k of other GPUs is NOT known today, only their per-expert counts) and routes all R ranks on every GPU.

History (searched 09-30):
- **08-21, handoff 08 (PLACE-lambda port):** the integrated deterministic arm was a GLOBAL torch router
  (`loccap_gpu`, CPU == GPU bit-identical) at 52-67 ms per iteration (launch / sync-bound). User ruling: bit-identity
  RELAXED; sender-local redesign = shared tables as order-independent functions of the d all-gather, per-row decisions
  owned by the sending rank, relaxed atomic tickets -> 0.40-0.45 ms per rank. Premise recorded then: "agreement
  across ranks comes from the phys-row allgather, never from replaying each other's decisions." The current pv3c
  kernel (`routing.cu`: `atomicAdd` ticket at :242, vacate atomics at :416-448) inherits this contract; its tables are
  bit-exact and its ticket invariants order-independent (handoff 40 section 6; network incidence +0.4..+1.3 % above
  the deterministic host reference `reference_route`, `routing.py:228`).
- **08-29, handoff 26 ("route-global", user-directed):** exactly this proposal. One top-k + probs all-gather
  (byte-identical to the phys + probs exchange) replaced the d all-gather + relaxed kernel + decisions all-gather;
  every rank recomputed every rank's assignment. Torch version (`route_global_quota`): correct end to end at 4n,
  2.8 ms (b1) / 13.1 ms (b8) even graph-replayed. Fused deterministic CUDA kernel (`placelambda_route_global`, flux
  ae0dd16: stable ordinals + closed-form quota windows): bitwise-proven, gates green 4n + 8n, 0.67-0.74 ms flat
  across scale. **Perf verdict NEGATIVE at 4n and 8n (+0.07..+0.92 ms everywhere), CLOSED.** Reason (arithmetic, not
  implementation): the exchange bytes are irreducible (raw top-k or routed decisions, same size), so the merge only
  saves the small FLAT d all-gather (~0.08 ms), while the single exchange's latency scales with tokens and the global
  kernel routes R x the entries of the per-rank relaxed route. "The 8/21 sender-local relaxation had already banked
  the available win; route-global re-centralizes the computation it distributed."

What is different today, and why it still stays last:
- Serving is host-bound: the loads gather also costs ~25 host calls and Python work (~0.1-0.2 ms host), not just
  ~0.1 ms of GPU. That could shift the balance slightly, but only once stages B-D have removed the bigger host costs
  would it show; the route-all kernel's extra GPU time lands on the same critical path.
- The router changed (pv3c water-fill + budget + vacate vs the 08-29 LocCap/quota router): the retained kernel is
  not a drop-in; a pv3c route-global needs deterministic tickets AND a deterministic vacate (budgets consumed per
  source in token order), and the vacate is the part most at risk of serializing (the 16n cost of the old serial
  per-expert largest-remainder loops, handoff 38 section 6.9, is the precedent for how such loops scale).
- Reopen only if, after stage D, the loads gather is measurably on the critical path; then measure route-all vs the
  saved collective at 1 / 2 / 4 MiB, 4n / 8n / 16n, with a bitwise cross-rank compare mode.

A different, never-attempted idea from the same history (08-21 next-fusion list): a COUNTS-ONLY exchange instead of
the per-row routing all-gather. Receivers need per-(source, copy) counts and the dedup unions (sps, uc rows) to size
and place their receive regions, not every other rank's per-row decisions; per-row positions are the sender's
business and consumer aliasing could travel in-band. That would shrink both the second exchange (flat in tokens) and
the replicated W x S x K planning work that grows with node count (16n 1 MiB prefill layer: ours 7.05 ms, vs 3.48 at
4n). A redesign of the metadata contract, not a stage of this pass; recorded as a candidate.

## 6. Proposed order for the pass (each stage measured on its own, plan-6 methodology)

| stage | change | removes | risk |
|---|---|---|---|
| 0 | copy-mechanism probes (section 4a): SM-independence and latency of P1-P4 vs P0 | decides stage C's copy path; no serving change | none |
| A | parallel stable scatter (bitwise-identical output), shared-memory meta_counts, parallel swap orbit | ~0.15-0.17 ms/layer GPU on the sync's critical path; up to ~0.3 ms on swap steps | low |
| B | all tables on the device (dispatch + combine + receiver lanes; extend LOPEP_DEVICE_META), drop duplicated / unused host work (combine re-derives dispatch tables, C computed 3x, unused uploads / memsets, 16 joined wire streams for 3 used) | host table loops, most uploads, ~40 event calls (the sync stays, section 4b) | low-medium |
| C | copy issue off the main thread per stage 0: batched copies (`cudaMemcpyBatchAsync`) as the immediate step, then the proxy thread or graph-selected copies; device-initiated inter-node puts after the wire probe; device capacity verdict (identical on every GPU because planning is replicated), read at the end of the forward, redo the forward on overflow (0 growths so far) | the planning sync and most of the ~480 calls | high (redesign) |
| D | whole-layer CUDA graph per bucket (launch sequence fixed once sizes are device-resident; per-step epoch moved to device memory) | per-layer host issue down to ~1 launch | medium (NVSHMEM ops in graphs) |
| E | routing merge (section 5): only if reopened by evidence after D | one flat collective (~0.1 ms) | closed 08-29 |

What remains after D: two collectives, put latency through the NVSHMEM proxy, barriers (3 per layer; round 8 code
reading suggests 1 suffices for the write-after-read hazard — unproven, needs a proof + randomized-payload stress).
Measure each stage's fixed term (the a in t = a + b n) at 4n / 8n / 16n; success = the crossover moving below 1 MiB.

## 7. Where to look

- Evidence: handoff 49 rounds 7-9 (fits, Nsight deep dive, code map, plan-6 tables); analyzers `49_hostcalls.py`,
  `47_gap_report2.py`; captures `logs/sglang/nsys49/`.
- Drivers for re-measuring: `logs/sglang/jobD6.sh`, `jobP6.sh`, `jobC6.sh`, `run16.sh`, `chainfig2.sh`
  (`GPU_C="gpu&hbm40g"`, regular QOS for 8n / 16n, short pieces).
- Sources (NVIDIA): CUDA Programming Guide, "CUDA Graphs" (device graph launch, conditional nodes) and "CUDA Dynamic
  Parallelism"; blog "Enabling Dynamic Control Flow in CUDA Graphs with Device Graph Launch"; blog "Fusing
  Communication and Compute with New Device API and Copy Engine Collectives in NVIDIA NCCL 2.28"; arXiv 2512.16056.
