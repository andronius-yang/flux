# Plan 9 text (as approved 2026-10-01, rev 1; preserved at the start of plan 10)

# Plan 9: per-layer fixed cost to parity - device-issued wire + one CUDA graph per layer

DRAFT 2026-10-01, rev 1 (adversarial review folded in). User direction 10-01: graph per layer, compare to stock;
NVSHMEM device-initiated inter-node puts; NVLink pushes fused into the producing kernels, verified for dispatch AND
combine, races / deadlocks first; buckets = protocol default; plan-8 B5 and M2 folded in. Log of record: new handoff
53 (research tree). Code: lopep worktree `lopep_c3` branch p8-c5 (a2b248e) -> new branch p9 (plain commit messages,
no numbers in lopep). Step 0 at execution start: copy the plan-8 text (this file's previous content, from the session
context) into `docs/handoff/52_plan8_text.md`.

## Context

- Plan 8 moved every size / table / verdict to the GPU and removed the per-layer host wait (T1 719 -> 69 us, 0 host
  syncs). Round 13, 4n decode, one allocation: c4b 119.1 / 148.3 / 219.8 ms at 256 / 512 / 1024 per rank vs stock
  86.1 / 144.7 / 273.4 (0.72x / 0.98x / 1.24x); prefill c4b 1.04x stock.
- NC4b capture: the layer-step is HOST-BOUND (host 2.69 ms = GPU window, GPU idle 1.05 ms). Main-thread CPU samples:
  CUDA API 1005 us, torch dispatcher 488, Python 421, libc 290, drain waits 364, lopep C++ 42. The host computes
  nothing; it ISSUES ~350 GPU operations per layer (main 214 API calls + proxy ~130), and its drains wait for the
  proxy, which waits for the GPU plan block.
- Target: >= 1.02x stock at the 1 MiB decode point (256 per rank, 4n): step <= 84.4 ms, i.e. MoE layer 2.17 ->
  <= ~1.45 ms; no regression vs c4b at 512 / 1024 / prefill. Host removal is necessary, not sufficient: GPU-occupied
  time per layer is 1.65 ms (prologue 393 us = planning kernels 257 + two NCCL all-gathers 151; first wire -> GEMM1 93;
  GEMM1 360, mostly spinning because the first payload lands 245 us after planning ends = proxy issue lag; GEMM1 ->
  GEMM2 117; GEMM2 103; combine tail 545 = proxied puts, pre-reduce, bucket reduce, 196 us barrier). Review tally if
  A-D land with green probes: ~1.2-1.3 ms.
- Why a device wire: copy engines are programmable only from the host with sizes fixed at issue (stage-0: device
  memcpy = SM kernel, device graph launch unsupported), and a host thread issuing producers after in-graph consumers
  are enqueued is the P8 deadlock with no drain possible in a graph.

## Loyalty contract (plan-8 L1-L10, unchanged) + L11, L12

Outputs = fp32 reference within bf16 tolerance (bad rows 0) and token agreement vs stock; routing, BOTH exchanges
(user: required), calibrations, swap decision, split values, put sizes / counts / union layout, issue order and tile
gating unchanged; no per-step allocation; every inter-node put a consumer gates on has its signal provably after its
data (rule 5), payload randomized per step in every correctness cell; capacity verdict before every write.
L11 the same bytes land in the same places (only the issuing agent changes); L12 graph replay output bitwise equal to
the eager step on the same inputs.

## Design

**A. Graphs.** One CUDA graph per (layer, bucket, generation); buckets = protocol default (`LOPEP_EXACT_BUCKETS=0`,
power-of-two `token_buckets`). Graph = the layer-step from the bucket buffers on (loads all-gather, swap decision +
lane kernels, routing + all-gather, planning, dispatch, activation, combine, barrier). Eager per layer: copy-in of
x / ids / w + pad fill (`serving.py:876-884`), `replay()`, copy-out of `y[:n]` from a static output buffer.
- Captured in a SEPARATE capture pass after the eager warm-up (warm-up steps are host-checked: `_sd_ev.synchronize`
  SV:775, `.tolist` SV:799, CapacityExceeded SV:546-556), per bucket with `begin_forward` (verdict armed),
  `LOPEP_TIMING=0`, nested planner-tail / scale graphs computed eagerly inside the capture (`planner.py:123`,
  `overlap.py:238`), ONE shared mempool (`pool=`) for all layer graphs, `capture_error_mode="thread_local"` (NVSHMEM
  proxy and NCCL watchdog threads must not invalidate it).
- The set of captured (layer, bucket) pairs is agreed by an all-reduce; eager paths (redo `kept` passes, uncaptured
  buckets) run the same code on the same dedicated lopep communicator, so ranks never mix graph and eager NCCL.
- Growth: drop all graphs (key `comm.generation`), recapture collectively in `recover` after its warm-up.
- Capture-time checks (code, every capture): every cross-rank reader is an ancestor of the barrier node; the K2 rule
  below holds for every spinner.
- Scale check P11d first: all 48 x ~10 real graphs instantiated (device + host memory, instantiate time, replay CPU
  cost vs preset limits); fallback = one layer-agnostic graph per bucket (layer weights via the GEMMs' per-expert
  `weight_ptr_override`, DG:2972-2982; route / swap tables via a device pointer table).

**B. Device step state + static buffers.**
- A device step block with PER-LAYER slots: the head kernel of each layer writes that layer's slot (dispatch run id,
  combine run id, swap epoch, GEMM start-mark epoch, layer ordinal, force flag); every consumer reads its own slot
  through a pointer, so a later head kernel can never move the value a resident persistent GEMM re-reads per tile.
  Values identical on every rank, only increasing (fresh ids on redo), excluded from the warm-up snapshot / restore
  (SV:747-762 and `swap.py:538-558` rewind the lane epoch today).
- Consumers converted: GEMM1 tile gate (`ag_scatter_gemm_grouped_with_absmax.h:510-532`, today `signal_expected =
  run_id_` DG:2931), both weight gates, the start mark (DG:2690-2705), pack / pre-reduce `run_id` args (GC:1759, 1851),
  lane kernels (`lane_device.cu` epoch args), demands ordinal (DG:2528), swap band C (1e30 in warm-up / kept passes,
  SV:586-589) and the dry-gate epoch (`swap.py:532-535`) as device words or per-graph constants.
- Persistent buffers: `problem_schedules` (DG:2873), dispatch GEMM workspace (DG:3003-3004), combine `gemm_outs` /
  `output` / workspace (GC:3454-3460, 3782), `c_excl`, route `phys` / `stats` (routing.cu:501-503), `set_routing`
  `.long()` (PL:96), scale compute (OV:221-225), pad `torch.where` (SV:160).
- Dedicated lopep NCCL process group (same ranks) for both all-gathers, eager and captured.
- Asserted unreachable on the deferred path (host syncs / D2H): DG:1393, 2270, 2537-2575, 2709; GC:3159, 3586, 3841.

**C. Device wire - dispatch** (new rdc object library `src/dwire/`, build pattern of `src/direct/CMakeLists.txt:9-11`;
device peer-base table from PeerTable `ce_batch.h:93-135`; kill-word spin helper):
- C1 pack-and-push (replaces `a2av_pack_rows_kernel` for the own-node segments + self copy + round-0 CE puts + their
  signals, DG:877-912): segments outer, grid-stride inner; rows go straight to peer d's recv buffer at `round0[dlg]`
  dst (self: `kSelf*`). Per segment every block does `__threadfence_system`, `__syncthreads`, then thread 0 arrives on
  a per-destination counter (`atomicInc` wrap, the `last_block_arrival` pattern CK:94-97); EVERY block arrives once
  per destination, also with zero rows and on degenerate (aborted) layers; the last arrival release-stores
  `signal[rank]` at d = run. One own-node segment per destination + the self segment = exactly one signal per source
  per destination (lane semantics unchanged). Remote segments packed locally as today; the announce becomes a release
  store into the peers' `seg_sig` (replaces DG:1984-1996). Graph edge pack-push -> GEMM1 (A/B in P9: own-node tiles
  cannot start before the pack anyway; the edge stops GEMM1 taking the SMs the pack needs).
- C2 relay gather: per round dn (in order, S-slot reuse), acquire-spin on own pack word / peers' `seg_sig`, SM-gather
  the pieces from the peers' send buffers into the relay slot; per-slot last-block counter releases a local "slot
  ready" word.
- C3 gateway forward: per source node ns, spin `node_sig[ns] >= run`, SM-copy `kWinA..kWinB` to every local d's recv
  buffer at `kFwd[dlg]` (ring-rotated order kept), per-destination last-block release of `signal[ns*L+my_lr]` at d.
- C4 device wire warp: ONE persistent 1-warp kernel per op issues the inter-node puts in schedule order (waits on the
  "slot ready" / own pack words, then put-with-signal to (tn, my_lr) `stage+kWireDst`, `node_sig[my_node]` = run; zero
  rows -> signal only). Put form chosen by P10: device blocking `putmem_signal` only if P10 proves data-before-signal
  inside a graph; else `putmem_nbi` -> `nvshmem_quiet` -> `signal_op` (correct on any transport). NVSHMEM code stays
  out of the data kernels (register pressure, divergent-branch block calls).
- No kernel mixes producer and consumer roles (non-cooperative grids may be partly resident -> cross-rank deadlock);
  C1, C2, C3, C4 are separate kernels, all fitting beside GEMM1 at the same time (measured occupancy, see K1).

**D. Device wire - combine:**
- D1 pack-and-push: `a2av_combine_pack_kernel` already waits per wave on the GEMM2 cascade (CK:110-169). It pushes each
  wave's rows from BOTH sources: the send panel when the waves run, `gemm_out` x vec_scale when the wave-adapt decision
  collapses them (`msplit_dec==0`, CK:144-146, 174-191; vec_scale always passed in deferred mode, GC:1751). Own-node
  homes d -> peer d `recv_panel[dst_off[d] + p - send_off[d]]`; remote homes -> gateway (my_node, dl)
  `conv_panel[conv_dst[tn*L+dl] + ...]`; per-destination last-block release of `recv_sig[rank*n_split+sid]` at d /
  `conv_sig[(my_lr*NN+tn)*n_split+sid]` at the gateway. 512 threads, `__launch_bounds__(512,2)` (64-register cap; a
  GEMM2 block allocates 256 x 128 regs). Replaces the conv and intra lanes (GC:1281-1373). Which branch runs at the
  1 MiB point is logged.
- D2 pre-reduce unchanged (flips `wire_flags`); the combine's device wire warp (C4 pattern) waits on `wire_flags[tn]`
  and puts wire panel (symmetric, GC:912) -> (tn, my_lr) `recv_panel[dst_off]`, `recv_sig[rank*n_split+sid]` = run.
- D3 receivers: ONE persistent receiver kernel walks the chain positions (map / scan / scatter, then per lane an
  acquire spin on `recv_sig[sq]` + kill word, then the bucket reduce), replacing the 16 front-end waits + launches
  (GC:2247-2273).
- Swap lane: `commit_after(1)` moves before the end-of-combine barrier (today after it, SV:1005-1011 vs GC:4010; a
  peer's next `lane_push` into my W2 staging is otherwise ordered only by the next layer's all-gathers).

**E. GPU chain shortening** (after A-D; each a same-binary knob A/B, kept only if it wins):
- E1 planning fusion: route (memset + 4 kernels), meta (stage1, pass1, scan, pass2, arena), plan + demands, consumer
  build x2 + gating cumsum + pack scan -> as few kernels as the dependencies allow (257 us chain).
- E2 barrier -> release words: per cross-rank-written buffer (recv buffer after GEMM1, gateway stage after forwards,
  send buffer / relay after the peers' pulls, conv panel after pre-reduce, recv panel after receivers, swap W1 / W2
  staging after commits) the reader releases layer l, the writer of layer l+1 waits for exactly those; proof table in
  handoff 53 before code.
- E3 faster transport for the two (kept) exchanges: NCCL all-gather (72-82 us each on CXI) vs an NVSHMEM on-stream /
  device fcollect of the same data; same bytes, same semantics; probe first.

## Deadlock and race model (design input; rule + gate per class)

| class | risk in plan 9 | rule | gate |
|---|---|---|---|
| K1 HOL (G1) | graph branches carry no launch-order guarantee; GEMM1 / GEMM2 persistent grids resident first | every kernel concurrent with a spinner fits beside one block of it; C1-C4 and D1-D3 must fit beside GEMM1 / GEMM2 SIMULTANEOUSLY | R5 test with measured occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` with the GEMM block's regs + smem); Nsight T7 |
| K2 queue head (P8 / G3) in a graph | a successor of a spinner submitted ahead of that spinner's producer in a shared hardware queue | BY CONSTRUCTION: every in-graph producer of a spinner's words is an ancestor of every successor of that spinner (today's `wire_done_` join, made a graph edge); costs nothing, the spinner cannot finish earlier | capture-time graph check; probe P11a; harness at CDMC=1 with graphs |
| K3 cross-rank cycle (NH-1 class) | A's spinner waits on B's push ordered on B after a wait on A; mixed-role kernels | per-layer cross-rank wait table proved acyclic in handoff 53 before code; no producer / consumer roles in one non-cooperative kernel; W2_EARLY kept | table review; harness cycle / flips / starve-rank; `LOPEP_DWIRE_DELAY` per kernel per rank |
| K4 cross-rank buffer reuse | pushes write PEER buffers | the end-of-combine barrier kept through S4, with every reader (C2, C3, C4, GEMM1, pre-reduce, receivers, D2 warp, swap commits) an ancestor; E2 only with its proof | capture check; randomized payload + delay injection + `--layers 3` varying counts; W2 commit-vs-push delay cell |
| K5 signal before data | many-block NVLink stores; device puts on CXI (`nvshmemi_proxy_quiet` is GPU-wide, `proxy_device.cuh:62-76`) | per-block fence + last-block counter + release store, consumers acquire; inter-node: blocking device form only if P10 proves it, else nbi -> quiet -> signal_op | P10 (both forms, inside a graph, 2 + 4 nodes); payload randomized in every cell |
| K6 counter / epoch reuse (E1) | step slots, done counters, signals across replays, redo, growth | per-layer slots, only increasing, never restored; done counters self-reset by the last arrival; signals never memset after allocation | growth + redo cells, graph and eager; head-kernel-overlap cell |
| K7 lazy load (H1) | eager paths | instantiation loads every module; existing warm-up covers eager | Nsight T6 |
| K8 NCCL in graphs / rank divergence | captured vs eager NCCL; one rank falling back to eager | dedicated communicator for both; captured set agreed by all-reduce | P11b; forced capture failure on one rank -> all ranks agree on eager |
| K9 NVSHMEM proxy | GPU-wide quiet couples rounds and ops | one wire warp per op, puts in schedule order; measure k concurrent puts | P10 latency vs concurrency |
| K10 kill word | every new spin | polls the host-mapped kill word | unit test |
| K11 aborted layer (V1) | zero-row / degenerate layers | every push / put / forward arrives and raises its signals with zero rows | `LOPEP_FORCE_ABORT` cells, graph and eager |

## Stages (each: gates -> stage check where marked -> lopep commit -> handoff 53)

- S0 probes (flux `docs/handoff/50_copy_probes/`, ~5 node-hours; a red result changes the design before code):
  P9 pack-push vs CE batch at the real per-layer volumes (15 / ~30 / ~60 MiB P2P per layer), 3 peers, beside a REAL
  GEMM1 and GEMM2 (incl. the collapse path), with / without the pack-push -> GEMM1 edge, incl. C3 forward throughput
  in the SMs GEMM1 leaves; P10 device puts on CXI: blocking `putmem_signal` and nbi -> quiet -> signal_op, issued from
  a kernel inside a graph, payload ordering (0 violations), latency, k concurrent puts, 2 + 4 nodes; P11 (a) K2 in a
  graph at CDMC 1 / 8 / 24, (b) NCCL capture on a dedicated communicator next to eager NCCL, (c) capture of
  `nvshmemx_barrier_all_on_stream` + memops, (d) all 48 x ~10 real graphs (memory, instantiate, replay CPU).
- S1 step state + static buffers + dedicated communicator (B), eager, proxy still on: bitwise vs c4b, perf neutral.
- S1.5 `src/dwire/` skeleton: CMake rdc library, device peer table, the put wrapper chosen by P10, kill-word spin
  helper, the R5 occupancy test; `lopep_layer_barriers()` moved out of `wire_proxy.h`.
- S2 device wire dispatch (C) behind `LOPEP_DWIRE=1`; then S3 device wire combine (D) (merged after S2; may develop in
  parallel on top of S1.5).
- S3.5 delete the proxy, plan ring, staging word, `wire_done_`, drains, host builders and legacy D2H copies (B5).
- S4 graphs (A) behind `LOPEP_LAYER_GRAPH=1`: capture pass, replay, growth recapture, eager redo, capture checks.
  Stage check round 14 + Nsight capture NG (`--cuda-graph-trace=node`).
- S5 chain shortening E1, E2, E3 (knob A/B each). Stage check round 15 + capture.
- S6 M2 milestone on the plan-9 binary: 4 / 8 / 16n, decode + prefill, 1 / 2 / 4 MiB (regular QOS, 12-20 min pieces);
  default flips; lopep commits to sglang-dev; handoff 53 final.

## Gates, measurement, verification

- Unit (1 GPU): existing tests + R5 occupancy test over every kernel + step-slot test.
- Harness (`examples/serving_check.py` via `logs/p50/hang_probe2.sh`, 4n; 8n at milestones): `--token-mode cycle` and
  `full`, `--swap 1`, `--skew-flip 1`, `--caps-scale 0.5` (growth + redo), `LOPEP_FORCE_ABORT`, `--ref 1`, fresh
  payload every step, CDMC 1 / 8 / 24, `LOPEP_DWIRE_DELAY` per rank, knob off vs on; graph cells: replay vs eager
  `--out-hash 1` bitwise; forced capture failure on one rank.
- Cheap perf A/B: harness layer-step medians (`--ref 0 --token-mode full`, `logs/p50/g11_ab.sh` pattern).
- Serving (protocol frozen = lib49 OURS_ENV incl. `LOPEP_REBALANCE=0 LOPEP_EXACT_BUCKETS=0`, KV pins, 40 GB nodes,
  CDMC 24, all arms on one allocation, ours twice, one driver per allocation): rounds 14 / 15 decode 256 / 512 / 1024 +
  prefill SMAX 256, c4b vs plan-9 vs stock (`49_plan6_csv.py`); success = 256-per-rank step >= 1.02x stock, no
  regression vs c4b at 512 / 1024 / prefill.
- Nsight per stage (`logs/sglang/jobN11.sh` single-step recipe + graph node trace; `52_timeline.py` + the CPU-sample
  split, extended for graph nodes): host range vs GPU window, GPU idle, T1 / T1p, T8 segments, T7.
- Allocations: account m5350_g, 40 GB nodes, 4n interactive pieces, release at once, job ids from our own salloc logs.
- Node-hours: S0 ~5, S1-S3.5 gates ~12, S4 round 14 + capture ~8, S5 ~8, M2 ~20: ~53.

## Critical files

lopep: `src/dispatch/ths_op/dispatch_gemm.cc`, `src/dispatch/sort_util.cu`,
`src/dispatch/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h`, `src/combine/ths_op/gemm_combine.cc`,
`src/combine/combine_kernels.cu`, `src/combine/cutlass_impls/gather_rs_gemm_grouped_with_absmax.h`, new `src/dwire/`,
`src/CMakeLists.txt`, `src/planner/lane_device.cu`, `src/core/wire_proxy.*` + `src/core/plan_ring.h` (deleted in S3.5),
`python/lopep/{serving,planner,swap}.py`, `python/lopep/comm/overlap.py`, `integrations/sglang/lopep_sglang/runtime.py`.
Reuse: `lane_push_kernel` (push + fence + last-block signal, `lane_device.cu:79-130`), `last_block_arrival`
(CK:94-97), `wait_geq_kernel` (`cuda_common.cu:91`), the rdc build of `src/direct/`, PeerTable, planner tail-graph
capture (`planner.py:138-159`), NCCL capture (`bench/replay.py:156-183`), the GEMMs' `weight_ptr_override`.
Research tree: `docs/handoff/53_plan9_*.md`, new probes, `52_timeline.py` graph support.
