# Handoff 53: plan 9 - per-layer fixed cost to parity (device-issued wire + one CUDA graph per layer)

Log of record for plan 9 (plan file `~/.claude/plans/the-kv-cache-size-cozy-hamster.md`, approved 2026-10-01). Plan 8
text: `52_plan8_text.md`; plan 8 log: `52_plan8_roundtrip_removal.md`. Code: lopep worktree
`$PSCRATCH/workspace/andrewy/lopep_c3`, branch p9 (from p8-c5 a2b248e).

## Starting point (round 13, NC4b capture, 10-01)
- 4n decode step (ms) 256 / 512 / 1024 per rank: c4b 119.1 / 148.3 / 219.8, stock 86.1 / 144.7 / 273.4.
- Layer-step host-bound: host 2.69 ms = GPU window (profiled; 2.17 ms unprofiled). Main-thread CPU samples per layer:
  CUDA API 1005 us, torch dispatcher 488, Python 421, libc 290, drains 364, lopep C++ 42. GPU-occupied 1.65 ms.
- Target: >= 1.02x stock at 256 per rank (step <= 84.4 ms, MoE layer <= ~1.45 ms).

## K3: cross-rank wait table of one layer (plan 9 design, draft rev 0)
Every spin on a word written by ANOTHER rank, its writer, and what precedes the write on the writer's rank. A cycle
would need some writer to be preceded by a wait on the waiting rank in the same layer; none is.

| # | spinner (rank r) | word | writer (rank w) | preceded on w by (same layer) | waits on r? |
|---|---|---|---|---|---|
| 1 | GEMM1 tile gate, own-node lane s | `signal[s]` at r | s: C1 pack-push last block | s's planning, the two all-gathers (collective), previous barrier | no |
| 2 | GEMM1 tile gate, remote window lane ns*L+g | `signal[ns*L+g]` at r | g (r's node): C3 forward | g's C3 waits `node_sig[ns]` <- (ns,g) C4 wire warp <- (ns,g) C2 relay <- node ns packs (no waits) | no |
| 3 | GEMM1 / GEMM2 weight gates | swap gate words | node peer: `lane_push` W1 + W2 (W2_EARLY, before its dispatch GEMM) | peer's swap decision (after the loads all-gather) | no |
| 4 | C2 relay gather | `seg_sig[sl*NN+tn]` at r | node peer sl: pack of remote segment (announce) | sl's planning | no |
| 5 | C3 gateway forward | `node_sig[ns]` at r | (ns, my_lr): C4 wire warp put | its C2 <- node ns packs | no |
| 6 | combine pre-reduce (r = gateway) | `conv_sig` at r | node peer: D1 pack-push | its GEMM2 <- its GEMM1 <- dispatch producers (rows 1-5, none waits on r's combine) | no |
| 7 | receivers | `recv_sig[w*n_split+sid]` at r | node peer: D1 (intra); remote gateway (tn, my_lr): D2 wire warp | D2 <- its pre-reduce <- row 6 chain | no |
| 8 | end-of-combine barrier | all ranks | all ranks | each rank's whole layer (all readers are ancestors, K4) | collective |
Cross-layer: layer l+1 producers start after the barrier; every reader of a cross-rank-written buffer in layer l is
an ancestor of the barrier (incl. swap `commit_after(1)`, moved before it). Same-rank spins (GEMM1 on own C1 via the
graph edge, C4 on own C2 slot words, D2 warp on own pre-reduce flags, D1 on own GEMM2 cascade) are covered by K1
(co-residency) and K2 (producers are ancestors of the spinner's successors).

## S0 probes (job 59186345, 4n 40 GB, 10-01 21:27; sources `50_copy_probes/p9_push.cu`, `p10_dev_put.cu`,
## `p11a_graph_k2.cu`, `p11b_nccl_graph.py`, build `build_p9.sh`, driver `logs/p50/s0_probes.sh`; logs `logs/p50/s0_*.log`)
- P11a (K2 inside a graph, 1 GPU): a spinner S, a successor D of S, and S's producer P (kernel or captured memop),
  captured on two streams in every order (P after D, P before S, P as an ancestor of D) and with P DEEP in its branch
  (6 independent kernels before it) while D is shallow; 1-block and GEMM1-shaped (200 x 128, 66 KB smem) spinners;
  CUDA_DEVICE_MAX_CONNECTIONS 1 / 8 / 24; 0 / 40 extra streams; 200 replays each: 120 / 120 cells ok, no deadlock.
  The P8 queue-head deadlock does not occur inside a graph. K2 by construction stays as a free safety rule.
- P10 (device-initiated NVSHMEM puts on CXI, inside a CUDA graph, payload changing every iteration, receiver checks
  the last word the moment the signal is visible and then every word; 8 and 16 PEs, partner = same local rank on the
  next node; 300 iterations per cell; 4 KiB - 4 MiB, 1 and 4 concurrent puts):
  - `nvshmem_putmem_signal` (thread, blocking), `nvshmemx_putmem_signal_warp`, `_block`: 0 violations everywhere.
  - `nvshmem_putmem_nbi` -> `nvshmem_quiet` -> `nvshmemx_signal_op`: 0 violations.
  - `nvshmem_putmem_signal_nbi` (control, the form rule 5 forbids): violations at >= 1 MiB in every cell (e.g. 4 MiB
    conc 4: 162 last-word / 46 M word violations at 4n): the probe detects the hazard.
  - host on-stream blocking put (today's wire, eager): 0 violations.
  - cost: per-put device time 17-23 us at 4 KiB, 72-81 us at 1 MiB, 240-250 us at 4 MiB (13-17 GB/s per GPU, all 4
    GPUs of a node putting at once); 4 concurrent puts take 4x (serialized by NIC / proxy); iteration times equal to
    the host on-stream put within noise (1 MiB: 109 vs 115 us at 2n).
  - Decision: wire warps use the blocking device `nvshmem_putmem_signal` (thread form, one put at a time in schedule
    order); nbi -> quiet -> signal_op is the equivalent fallback.
- P11c: `nvshmemx_barrier_all_on_stream` + device puts captured in one graph (2n): ok, ~35 us per barrier.
- P11b (2n, 8 ranks): the two all-gathers on a dedicated communicator captured in a graph with ~150 small kernels,
  200 replays interleaved with eager all-reduces on the default group: 0 mismatches.
- P9 (1 node, 4 GPUs at once, each pushing to its 3 peers, gather by random token index, 4 KiB rows; payload checked,
  bad 0 everywhere): fused push vs today's gather + copy-engine copies + signal writes (us, max over GPUs, best
  block count):

| per-peer | idle: CE / push | cuBLAS bf16 GEMM running: CE / push |
|---|---|---|
| 0.5 MiB | 60 / 45 | 61 / 47 |
| 1 MiB | 82 / 60 | 91 / 63 |
| 2 MiB | 118 / 88 | 139 / 96 |
| 4 MiB | 207 / 143 | 241 / 135 |

  The push wins at every size (it removes the local pack pass); 32 blocks x 512 threads (32 regs) suffice. The first
  GEMM-shaped occupant cells were VOID (10-180 ms for both paths: every occupant thread polled a host-mapped word, a
  PCIe storm no real kernel makes). Rerun (job 59186725, `logs/p50/s0_p9_occ2.log`) with the occupant fixed (a
  GEMM1-shaped resident grid spinning on a DEVICE word with acquire loads, released by `cuStreamWriteValue64` after the
  timed region), best block count:

| per-peer | GEMM1-shaped spinning occupant: CE / push |
|---|---|
| 0.5 MiB | 68.7 / 56.7 |
| 1 MiB | 91.7 / 71.0 |
| 2 MiB | 143.6 / 102.1 |
| 4 MiB | 247.7 / 166.9 |

  Push wins beside a resident spinner as well (-17 to -33 %); 32 blocks best up to 2 MiB, 32 = 64 at 4 MiB.
- P11d (4n, 16 ranks, `logs/p50/s0_p11d2.log`): 48 layers x 10 power-of-two buckets = 480 graphs, each the two
  all-gathers + 150 small kernels, one shared pool: capture + instantiate 13.2-14.5 s, device memory 1340 MiB (pool
  reserved 424 MiB), replay() host cost 32 us median (p90 39) with an idle GPU; back-to-back replays 357 us host per
  call = backpressure (the host waits on the GPU queue, i.e. the host is no longer the bottleneck). Per-layer graphs
  are affordable; the layer-agnostic fallback is not needed.

## S1 + S2 gates (job 59186725, 4n 40 GB; `logs/p50/s1_gate.log`)
- S1 (step state: per-layer device slots, GEMM gates / weight gates / pack / pre-reduce / lane kernels read the slot;
  binary 49c9aa03a991): ref twice, ss, ss_flip (880 swap moves), ss_grow (3 growths + 3 redos, forced aborts), ss_c1
  (CDMC 1), ss_nodefer: all PASS, bad rows 0. Perf neutral: harness layer-step median 256: c4b 1.703, ss 1.710 ms;
  1024: 3.420 / 3.421 ms.
- S2 (device-issued dispatch wire, `LOPEP_DWIRE=1`, proxy still on for the combine; binary 84ee54322f3f): dw,
  dw_flip, dw_grow, dw_c1, dw_full: all PASS. Perf REGRESSED: 256: 2.048 ms (+20 % vs c4b), 1024: 3.980 (+16 %).
  Hypothesis: SM shared-memory carveout - the small-smem dwire kernels (relay / forward 128 x 16 per SM) are resident
  when GEMM1 launches and their SMs keep the default carveout, so GEMM1 (66 KB smem per block) cannot co-reside on
  those SMs. Fix (lopep cff855e): every dwire / combine side kernel prefers the max shared carveout
  (`cudaFuncAttributePreferredSharedMemoryCarveout`, `LOPEP_DWIRE_CARVEOUT=0` disables); A/B in S3b.

## S3 gates (device-issued combine wire; job 59187432; `logs/p50/s3_gate.log`)
- First pass (binary 7709dafb85ef): every dw3 cell failed the host check at `gemm_combine.cc:1464` (the deferred plan
  check did not know the device wire needs no host plan / ring) - fixed (lopep dfb2cb5).
- S3b (binary d66bc9765ea6; dispatch + combine device wire, proxy thread still running for nothing else):
  dw3, dw3_flip (880 moves), dw3_grow (3 growths, 3 redos, forced aborts), dw3_c1 (CDMC 1), dw3_full: all PASS,
  bad rows 0. "bare" (no proxy thread, no plan ring: `LOPEP_WIRE_PLAN=0 LOPEP_WIRE_PROXY=0`, every wire op issued by
  kernels): bare (flips, 880 moves), bare_grow (3 growths, 3 redos), bare_c1: all PASS.
- Harness A/B, same allocation (layer-step median ms, 256 / 1024 per rank): c4b 1.694 / 3.430; bare (carveout on)
  1.953 / 3.917; bare carveout off 1.999 / 4.063; step state + proxy wire (binary control) 1.706 / 3.415. The
  carveout fix buys 2-4 %; the device wire still costs +15 % (256) / +14 % (1024) GPU-side vs the copy-engine wire.
- Serving decode at 256 per rank (one allocation, job 59187432, report `logs/sglang/j4nDd14b_report.txt`): c4b
  117.60 ms, device wire (bare, eager) 124.77 ms (+6 %). Per-layer MoE timing: device wire FASTER at the smallest
  batches (n_pad <= 8: 1.86 vs 2.22 ms: less fixed cost), slower at 256 (mean ~2.27 vs ~2.18): a bandwidth loss.
  Suspect: the relay pulls (NVLink loads, one 16-byte load in flight per thread, 16 x 128-thread blocks) and the
  gateway forwards (32 x 128) vs the copy engine; Nsight capture + copy knobs (lopep p9-dw2) next.

## S4 gates (one CUDA graph per (layer, power-of-two bucket); lopep_b3 branch p9-s4; job 59188021; `logs/p50/s4_gate.log`)
- g_full, g_cycle (per-rank counts 0 .. 1903 in one step, graphs captured at the bucket size), g_flip (880 swap
  moves, 72 of 72 layer-steps replayed from graphs), g_c1 (CDMC 1): all PASS, bad rows 0. 27 graphs (3 layers x 9
  buckets) captured in 0.7-1.0 s.
- g_grow FAILED: the recapture after a growth reused the released graph pool (PyTorch caching allocator assert
  `use_count > 0` at capture_begin). Fix lopep 4dcf8e9 (a new pool per capture pass); rerun g_grow2 (job 59188614):
  PASS, 3 growths, 2 redos, 76 graph replays.
- Teardown: a process whose NCCL communicator was captured in graphs hangs in `ncclCommFinalize` (as in P11b); the
  harness drops the graphs and skips the process-group teardown (b222b0b).
- Harness A/B (layer-step median ms, 256 / 1024): device wire eager 1.952 / 3.863, + layer graphs 1.887 / 3.720
  (-3.3 % / -3.7 %). The harness loop is close to GPU-bound already; the serving layer (host-bound, 2.69 ms host
  for 1.65 ms GPU in the NC4b capture) is where graphs should pay. Parity needs the GPU side too: the device wire's
  copies must at least match the copy engine.

## Log
- 10-01: plan 9 approved; plan-8 text preserved; handoff opened.
