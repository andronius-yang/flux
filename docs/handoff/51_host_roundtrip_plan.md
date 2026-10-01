# Handoff 51: removing the per-layer host round trip (plan 7, stages B + C3 + C4 + D2) - DRAFT 10-01 (rev 2: mode-1 hang root cause)

Status: design draft, written from measurements of 10-01 (handoff 50 rounds 7-8, the round-8 Nsight capture
`logs/sglang/nsys50/n_ours_b1_d256.*`, the stage-B code maps `50_stageB_{dispatch,combine}_map.md`). Nothing below is
implemented unless marked. Paper semantics (loyalty contract L1-L10 of plan 7) are unchanged by every step: what moves
is WHERE sizes are computed and WHO issues the copies, never what is moved, in which order, or what gates what.

## 1. What the round trip is (measured)

At 4n decode 1 MiB (256 tokens per GPU) one MoE layer-step is ~2.9-3.0 ms (stock 1.5). The capture (instrumented, so
host times are inflated; read proportions) shows the GPU of a rank IDLE for ~49 % of each layer-step's GPU window
(1.87 of 3.80 ms, median), and attributes every idle microsecond to what the main thread was doing:

| idle cause (mean us per layer-step) | us | host work |
|---|---|---|
| dispatch-phase host code | 427 | `a2av_dispatch` host tables (chunks, offsets, water-fill for all NN^2 pairs, gates, pinned staging) + the combine's derive tables (`build_a2av_compress_indices_fast`, `build_a2av_combine_indices`) |
| after routing, before the dispatch | 289 + 132 (sync) | `plan_meta` Python, the host capacity check (numpy), the planning `cudaEventSynchronize` |
| kernel launches / copy issue | 244 + 107 | ~45 launches + ~14 copies in the dispatch phase alone, ~20 + ~7 in the combine |
| pad / loads Python | 123 | buffer copies, pad table, histogram, gather setup |
| everything else | ~550 | event records / queries (partly the ledger's own), memsets, swap arm / parse / commit, activation |

Mechanism: every per-layer host step that needs a COUNT (a copy size, an offset, an allocation size, a launch decision,
the capacity verdict) forces the host to wait for the GPU's planning kernels (the sync), compute on the host, then
issue; the GPU has nothing to do meanwhile. GPU-side work after the host has issued is a separate floor (section 5).

## 2. Every host consumer of the counts (from the stage-B maps), by kind

| kind | examples (file:line in the maps) | what replaces it |
|---|---|---|
| K1 kernel arguments / tables | offA/cumA, expert_base, ssc, seg_off, gate_q, mm tables (dispatch arena); offA/rtab/t64/t32/home_base/conv_base (combine derive) | a device "plan block" written by planning kernels; the device arena already computes most of the dispatch's (mode 1) and `a2av_combine_tables_kernel` the combine's |
| K2 copy-engine copy sizes / offsets | round-0 intra puts, self copy, relay pulls, gateway forwards (dispatch); conv, intra lanes (combine) | the wire proxy issues them from DEVICE DESCRIPTORS (exact sizes, written by a one-block descriptor kernel into pinned mapped memory) |
| K3 inter-node put sizes | the relay's blocking `putmem_signal_on_stream` (dispatch), the wire ladder (combine) | the proxy issues the same BLOCKING calls with descriptor sizes (wire-ordering rule kept; no device-initiated puts) |
| K4 allocation sizes | `pack_index[M]`, combine CSRs `[wire_total+1]`, `[conv_total]`, `[own+rem]`, outputs `[M_this_ep, N]`, `out_buf[:m]` | capacity-sized persistent buffers; consumers read the true counts from the plan block |
| K5 control flow | `if (M_this_ep > 0) GEMM`, msplit wave-adapt decision, per-split skips, zero-row signal-only branches | device decision words read by the kernels (always launch; zero work = early exit), two pre-built variants where the shape differs |
| K6 capacity verdict | host demands vs capacities, `CapacityExceeded` -> grow | device verdict word (sticky per forward) + deferred host check once per forward + redo (plan 7 C4) |
| K7 Python | `_m_this` slicing, swap result parse (`arm_parse`), lane arm, scale buffer | device-resident M; the device lane already exists (`LOPEP_LANE_DEVICE=1`) |

## 3. Target per-layer flow (no host wait on counts)

Main thread enqueues, per layer, a FIXED-SHAPE program that never reads a count: pad / loads -> NCCL gather -> swap
decision -> route -> NCCL gather -> metadata (3 launches, A1) -> plan block (dispatch + combine tables, one or two
launches) -> descriptor kernel (writes the copy / put descriptors of this layer into a pinned mapped ring, bumps the ring
epoch) -> pack (one fixed-grid kernel reading `seg_off` from the plan block) -> consumer build -> GEMM 1 (problem sizes
from the device; tiles spin on the same arrival signals) -> activation (device M) -> combine tables already in the plan
block -> GEMM 2 -> combine pack / pre-reduce / receiver kernels (device sizes) -> one barrier.
The wire proxy (host thread) polls the ring epoch and issues, in the posting order of today, every K2 / K3 transfer
with descriptor sizes on its own streams; ordering edges are device words (stream waits / writes) only. The capacity
verdict accumulates in a device word; the SGLang hook checks it once per forward and redoes the forward on growth.
Then (D2) the main-thread program is captured once per (bucket, layer shape) as a CUDA graph.

## 4. The hang class this design must defeat (root cause, 10-01)

Two hangs today share one root cause: a kernel resident on the GPU spins on work that the HOST must still issue, while
the host thread that would issue it is blocked inside the CUDA driver waiting for that GPU to drain.
- C2b gate hang (10-01 00:37, `gate_c2b1_static_1n`): main thread in its first launch of the activation kernel ->
  lazy module load (`cuLibraryGetModule`, `CUDA_MODULE_LOADING=LAZY` from `bench/launch.sh` and the CUDA 12 default)
  holds the runtime lock and waits for the device; the dispatch GEMM is resident, spinning on arrivals; the proxy
  thread, which must issue those arrivals, is blocked on the runtime mutex in `cudaStreamWaitEvent`. Same class as the
  08-16 "L1 lazy-load hang" and the combine's comment on priming the bare `signal_op` kernel.
- With the round trip removed, GEMM tiles will spin on proxy-issued copies in EVERY layer, so any main-thread call that
  can block in the driver while that is in flight deadlocks: first launches (lazy loading), implicit device syncs
  (`cudaFree`, some `cudaMalloc`, pageable copies, `torch.cuda.synchronize`), and NVSHMEM host calls that take the
  NVSHMEM lock the proxy holds.
Design rules (to be proven by probe P6 below before any C3 code):
- R1 every module the steady state uses is loaded before the first spinning kernel: `CUDA_MODULE_LOADING=EAGER`, or an
  explicit warm-up that touches every kernel (`cudaFuncGetAttributes` per lopep kernel; one full forward per bucket);
- R2 no implicit device sync on the main thread in steady state (pinned buffers only, caching allocator never frees,
  no pageable copies);
- R3 the proxy never needs a lock the main thread can hold while blocked. P6 (below) settled the option: driver-API-only
  issue does NOT help (its work is held on the GPU during a load), so R1 alone carries it;
- R4 NVSHMEM host calls are serialized by one process lock (done: `nvshmem_host_mutex`, NVSHMEM 3.2.5 grants only
  THREAD_SERIALIZED), and no main-thread call holds it across anything that can block.
- P6 (probe, 1 node): thread A launches a kernel that spins on a flag; thread B must write the flag with (a) a runtime
  call, (b) a driver call (`cuStreamWriteValue32`), after thread A starts a first launch of a never-loaded kernel.
  Outcomes decide R3. RESULT 10-01 (job 59155602; `50_copy_probes/p6_lazy.cu`, `p6_cold.cu`, `run_p6.sh`; logs
  `logs/p50/p6_lazy*.log`; A100, driver 580.178, LAZY loading unless noted):

  | cold kernel placement | B releases the spinner with | A's first launch | B's call | spinner |
  |---|---|---|---|---|
  | same module (function-level load) | host store to mapped memory, after 0.2 s / after 3 s | returns at once | (no call) | released |
  | same module | driver stream write / runtime copy | returns at once | returns | NEVER released |
  | separate library (module-level load) | host store, after 0.2 s / after 3 s | returns only when the spinner ends | (no call) | released |
  | separate library | driver stream write | NEVER returns | returns | never released |
  | separate library | runtime copy | NEVER returns | NEVER returns (loader holds the lock) | never released |
  | same module, EAGER | driver / runtime | returns at once | returns | released |
  | separate library (own static cudart), EAGER | driver | NEVER returns | returns | never released |
  | controls without the cold launch | driver / runtime | - | returns | released within 0.5 ms |

  Reading: while code is being loaded with another kernel resident, the GPU executes NO newly submitted work from any
  thread until the resident kernel ends (a load needs the device idle); for a module-level load the launching thread
  also blocks until then, and other threads' runtime calls block behind it. A spinning kernel can then be released
  only by something that needs no GPU work (a host store to memory it polls). So R3 is answered NO: a proxy thread,
  runtime or driver API, cannot rescue a spinner during a lazy load. R1 is mandatory, in its explicit form: every
  kernel of every library the steady state uses is launched once in warm-up (EAGER does not cover a library whose
  runtime instance initializes late, and fails with NCCL in the torch process). Gate for R1: in an Nsight capture of
  the steady state, no `cuKernelGetFunction` / `cudaGetFuncBySymbol` call above ~100 us and no library or module load
  call. Today's serving steady state passes it (round-8 capture: 2740 + 7447 lookups, max 31 us).

Other hangs seen today, root causes:
- C2b in serving: all four GPU workers' proxies pinned to the same core (the last core of the node's shared task mask)
  -> a quarter of the decode rate, then a stall. Fixed in source (no pinning, short spin then condvar), untested.
  The proxy now defaults to OFF (`LOPEP_WIRE_PROXY=1` enables it; uncommitted dev tree, build `logs/p50/bin/dev_p0`).
- `LOPEP_DEVICE_META=1` at 4n (stage-B prerequisite): ROOT-CAUSED AND FIXED 10-01 (lopep 8d3a8a0), section 4a.

### 4a. The device-metadata hang: a GPU block-scheduler deadlock (second hang class)

Symptom (harness 4n and serving 4n decode): some nodes stuck in the dispatch GEMM (cuda-gdb: the stream-K grouped
GEMM `..._agscatter_rcr_gemmgroupedv2_..._streamksk_...`, spinning on arrivals), the others in the next barrier; every
main thread had already issued the whole layer-step. Probe matrix (job 59153644, 4n, qwen3-30b, `--smax 2048
--layers 3`; "varying" = `--token-mode cycle`: equal random counts, per-rank random counts, a zero-row rank, full;
logs `logs/p50/hang_probe.log`, `hang_<name>.log`):

| configuration | result |
|---|---|
| host tables (mode 0), combine derive overlapped with the dispatch GEMM (`plan_overlap 2`), varying | pass |
| mode 1, combine derive before the dispatch (`LOPEP_PLAN_OVERLAP=0`), varying, torch reference on | pass, 0 bad rows |
| mode 2 (device tables compared word for word), derive before the dispatch, varying | pass, 0 mismatches |
| dispatch-only mode 1 (`LOPEP_DEVICE_META_COMBINE=0`), overlapped, varying | pass |
| combine-only mode 1 (`LOPEP_DEVICE_META_DISPATCH=0`), overlapped, varying | HANG at step 0 |
| ... with swaps off / with `LOPEP_SM_MARGIN=32` / with a fresh unshared stream at default priority | HANG |
| ... with the side stream from the high-priority pool (`LOPEP_META_STREAM_PRIO=-1`) | pass |
| plan-6 binary, mode 1, varying / C1 binary (no proxy code), mode 1, varying | HANG (step 1 / step 0) |
| fixed tables kernel, combine-only and both ops, varying, reference on, 4 and 8 steps | pass, 0 bad rows |
| fixed, proxy on, full batches, both ops (hung at step 1 before) | pass, 6 steps |

Mechanism (resource numbers from `cuobjdump --dump-resource-usage`):
1. The dispatch GEMM block is 128 threads x 240 registers (30.7K of an SM's 64K). Its 200 blocks are placed
   breadth-first over all 108 SMs, so `sm_margin` leaves capacity spread thin: no SM is empty while the GEMM is
   resident, and an SM keeps at most ~34.8K registers free.
2. The overlapped combine derive (`plan_overlap 2`, budgets <= 16 MiB, i.e. every serving cell) launches its kernels
   on the side stream while that GEMM spins. In mode 1 that includes `a2av_combine_tables_kernel`, which was one block
   of 512 threads x 84 registers (~45K registers): it fits on NO SM until a GEMM block exits.
3. The work distributor dispatches kernels of one priority in launch order: a pending block that cannot be placed
   holds back every kernel of that priority launched after it. Those include the wire kernels the GEMM still waits
   for (the inter-node blocking put kernels and NVSHMEM signal kernels, released late by their stream waits). The
   GEMM never gets its arrivals, so no GEMM block exits, so the tables block never places: a deadlock, which spreads
   to the other nodes through their missing arrivals and the barriers.
4. Why intermittent: it needs a wire kernel the GEMM still needs to be released AFTER the tables kernel is pending.
   Full batches with inline issue usually release the relay puts first; per-rank-varying counts (serving) and
   proxy-issued wire (issued later) reorder them. Why mode 0 and dispatch-side mode 1 never hung: mode-0 derive
   kernels are <= 256 threads x <= 58 registers (they fit beside a GEMM block), and the dispatch's arena kernel
   (512 x 119 registers) runs before the GEMM launch. Why a high-priority stream "fixed" it: the unplaceable block
   then waits in the other priority's queue and no longer blocks the wire kernels.
5. Fix: the tables kernel is 128 threads with launch bounds (128, 4) (~11K registers per block). The earlier proxy-on
   harness hang (`dm_comb`, full batches) and the serving mode-1 stall of 00:45 (proxy off, varying counts) are this
   same deadlock. Serving re-run 10-01 04:25 (job 59155602, 4n decode, 256 running per rank, timing ledger on, fixed
   binary `bin/dev_ct128`): healthy, the wave completes, 0 growths, 0 tracebacks. Diagnostic ledger, same
   allocation, mode 1 vs mode 0 (never for ratios): layer bracket 3.02 vs 3.27 ms; `meta+check` 0.156 vs 0.273 ms
   (the device demand check replaces the host numpy check); decode step median 168.6 vs 178.7 ms.

### 4b. Why the hangs exist, and the rule set the round-trip removal must follow

Every hang seen in plan 7 has one shape: a resident kernel spins on work whose LAUNCH has not happened yet, and the
launch path waits, directly or indirectly, for that same spinning kernel. Two ways the launch path can wait:
- class H (host): the thread that must issue the work is blocked inside the CUDA driver (a lazy module load, an
  implicit device sync, a lock held across such a call) - the C2b gate hang;
- class G (GPU scheduler): the work is issued, but its kernel sits behind an earlier same-priority kernel that cannot
  be placed beside the spinning kernel - the device-metadata hang.
Removing the host round trip multiplies both: GEMM tiles will spin on proxy-issued or descriptor-driven wire in every
layer, and more planning work (plan blocks, descriptor kernel, device verdict) runs on the GPU around the spinning
GEMMs. So, in addition to R1-R4:
- R5 (class G): every kernel that can be launched while a spinning kernel is resident must fit beside ONE block of
  it on every SM (registers, shared memory, threads), or be launched before the spinning kernel, or on a
  higher-priority stream than the work the spinner waits for. Gate: a resource table (`cuobjdump
  --dump-resource-usage`) of every kernel enqueued between the GEMM launch and its completion (overlapped planning
  chain, swap-lane kernels, NVSHMEM put / signal / barrier kernels, the future descriptor kernel), checked against
  the GEMM's per-SM leftover; the B1 / B3 plan kernels and the descriptor kernel are launched BEFORE GEMM 1 (B3 already
  moves the combine tables between `derive_routed_meta` and the sync, which removes this case by construction).
  R5 baseline (10-01, from the round-8 Nsight capture, mode 0, one rank, every kernel that STARTS while a spinning
  kernel runs, block registers rounded to the allocation unit):

  | spinning kernel | one block | leftover beside one block | kernel kinds starting during it | any not fitting |
  |---|---|---|---|---|
  | dispatch GEMM (stream-K, grid 200) | 128 thr x 240 reg = 30.7K reg, 66.6 KB smem | 34.8K reg, 101 KB smem | 21 (planning chain, torch elementwise / scan / searchsorted, NVSHMEM put and signal kernels) | none |
  | combine GEMM (gather-RS, grid 152) | 128 thr x 254 reg = 32.8K reg, 65.6 KB smem | 32.8K reg | 7 | none |
  | resident pre-reduce (grid 6) | 512 thr x 44 reg = 24.6K reg | 41.0K reg | 7 | none |
  | bucket reduce (grid 8) | 512 thr x 28 reg = 16.4K reg | 49.2K reg | 2 | none |

  Kernels that must stay AHEAD of GEMM 1 (do not fit): the dispatch arena kernel (512 thr x 119 reg = 61.4K reg, mode
  1, today launched before the GEMM) and the old combine tables kernel (45.1K reg, fixed to 11.3K). Budget for any new
  kernel that may run beside a spinning GEMM (plan block, descriptor kernel, device verdict): <= ~32K registers and
  <= ~100 KB shared memory per block; note that on the 92 SMs holding two dispatch GEMM blocks only ~4K registers stay
  free, so such kernels land on the remaining 16 SMs (keep their grids small).
- R1 note: `CUDA_MODULE_LOADING=EAGER` fails in the torch process (NCCL init: `ncclUnhandledCudaError`, 10-01), so R1
  must be the explicit warm-up form, not the environment switch.

## 5. What remains after the round trip (the next floor, measured)

Per layer-step GPU time at 1 MiB (capture, one barrier): the single-block swap decision kernel 214 us on the critical
path (A2 parallelizes it), two NCCL all-gathers 76 + 104 us, planning kernels ~150 us (compress-plan scan 98, the
combine's `searchsorted` 28 = A1b), GEMM 1 697 us of which most is spinning on arrivals, blocking inter-node put kernels
673 us each in the dispatch (2.9 per layer) and 545 us each in the combine (2.7), pre-reduce spin 612, bucket reduce
324, the remaining barrier 318 (waiting for the slowest rank). The inter-node wire at 1 MiB is latency through
NVSHMEM's CPU proxy (1-2 MB per remote node would take < 100 us at NIC speed): after the host round trip it is the
largest term, and it is bound by the wire-ordering rule (blocking puts) and the paper's round schedule; any change
there is a separate decision for the user.

## 6. Proposed order (each step: gates, stage check against the previous binary on one allocation, then commit)

0. Root-cause and fix the mode-1 hang (prerequisite: the device plan block must be trusted). DONE 10-01 (section 4a,
   lopep 8d3a8a0); serving re-run healthy, P6 done (R3: no; R1 explicit warm-up mandatory), R5 baseline table done.
1. B1/B2: the dispatch's plan block on the device; the host reads the plan block (one D2H with the existing sync)
   instead of building tables -> removes most of the 427 us dispatch-phase host code; sync still there.
2. B3 (+A1b): the combine's tables on the device (`a2av_combine_tables_kernel` extended), persistent capacity-sized
   CSRs, `e_of_copy = routing_ids`; removes the combine derive host code and its allocations.
3. A2: the parallel swap decision (bitwise test exists, `tests/test_swap_decide.py`).
4. C2b fixed (pinning) + R1-R4 in place; then C3: descriptor kernel + proxy-issued copies and blocking puts with
   descriptor sizes; gate = the descriptor compare (`LOPEP_WIRE_DESC=2`, every (src, dst, bytes, pe, signal) equal to
   today's) + the randomized-payload wire-ordering stress with the proxy issuing.
5. C4: deferred verdict; the sync is gone; `forward_check` hook in the SGLang patch.
6. D2: per-layer graph.
7. D1 default flip (one barrier) after the 8n / 16n stress (independent; can land any time).
Expected (estimates from the attribution, to be measured at each step): steps 1-2 remove most of the ~0.45 ms of
table code; step 3 ~0.2 ms on swap-heavy windows; steps 4-5 remove the sync and the issue wait (~0.4-0.6 ms); step 6
most of the launch overhead (~0.3 ms). Floor after that: section 5.
