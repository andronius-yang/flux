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

## Nsight of the harness layer-step (job 59188611, `logs/p50/nsys9/h_{c4b,bare,graph}`, analyzer `53_harness_kernels.py`)
- Recipe: node 0's torchrun process tree under nsys, `-t cuda,nvtx --cuda-graph-trace=node`, capture range =
  `serving_check --nsys 1` (isolated layer-steps from step 20: a sync before and after each). `-t ...,osrt`
  crashes NVSHMEM init ("double free detected in tcache 2" in every traced process): never trace osrt here.
- Isolated layer-step (host range / GPU window, us): c4b 2440 / 2350, device wire eager 2357 / 2272, device wire +
  layer graphs 1958 / 1873. Pipelined harness medians (same binaries): 1.69 / 1.95 / 1.89 ms.
- Why c4b looks better pipelined: the harness layers are independent (every layer reads the same input), so the
  eager c4b code overlaps a layer's prologue (exchanges, planning) with the previous layer's combine tail. A model
  cannot (layer l+1's routing needs layer l's output), and a layer graph cannot either (graphs serialize). Serving is
  the arbiter; the isolated number is the GPU chain of the plan-9 layer.
- Graph layer-step, main stream timeline (us from the first copy-in): copy-in + graph launch 0-79; loads all-gather
  110-199 (89); swap decide, lane arm, pad rebuild, route tables (23), budget (14), route, vacate (34) 199-294;
  routing all-gather 299-365 (66); planning chain 365-528 (torch elementwise 14, meta pass1/scan/pass2 32, dispatch
  plan 38, demands 11, 4 D2H mirror copies 13, meta arena 27); pack-push 528-655 (127); GEMM1 prep 655-694; relay /
  wire warp / forward 694-1087 (3 serialized puts ~105 us each, forward done 1087); GEMM1 719-1189; GEMM2 1228-1363;
  combine pack / pre-reduce / wire warp 1229-1640, bucket reduce done 1661; barrier 1664-1858 (195).
  Combine-side planning (tables 47, plan block 31, compress plan 50) runs on the meta stream, off the critical path.
- Barrier: even the latest of node 0's four ranks waits ~177 us (median): mostly ranks on other nodes finishing
  later (rank skew), plus the barrier latency (P11c: ~35 us at 2n).
- Planning kernels are single-block: dispatch plan (1 x 256, 38 us), meta arena (1 x 512, 27), route tables (1 x
  128 threads, one per expert, 255 registers, 23), combine tables (1 x 128, 48), combine plan block (1 x 128, 32),
  compress plan scan (1 x 256, 23), demands (1 x 512, 11), meta scan (1 x 256, 12): the E1 target.

## Copy knobs and the early wire fork (jobs 59188611, 59189027; `logs/p50/dw2_ab.log`, `dw3_ab.log`)
- lopep p9-dw2 570aa53 (LOPEP_DWIRE_UNROLL[_PACK|_RELAY|_FWD], LOPEP_DWIRE_{PACK,RELAY,FWD}_BLOCKS), harness with
  graphs, layer-step median ms at 256 / 1024: base 1.877 / 3.707; relay x4 loads in flight 1.897 / 3.722; + 32 relay
  blocks 1.892 / 3.652; pack x4 1.886 / 3.722; 64 forward blocks 2.026 / 3.761 (spinner blocks take GEMM 1's SM
  room); all 2.045. No gain: defaults kept. All knobs on: correctness PASS (flips).
- lopep p9-dw3 04a6bbc (LOPEP_DWIRE_EARLY_FORK): the pack-push is split, remote segments first, then the relay /
  wire / forward kernels are enqueued, then the own-node push (no spinner waits on own-node data, so a spinner
  ahead of the own-node push in an aliased hardware queue only delays it). Correctness: graph flips, eager flips,
  eager CDMC 1, graph growth: PASS. Harness: 256: 1.833 vs base 1.896 / 1.874 (-2.8 %); 1024: 3.608 vs 3.748
  (-3.7 %). Fewer spinner blocks on top: no further gain (16 forward: 1.877; 16 forward + 8 relay: 1.834).

## Round 14 (serving, job 59188614, `logs/sglang/j4nD14_report.txt`)
- Decode step median ms at 256 / 512 / 1024 per rank: stock 86.45 / 144.29 / 274.27, c4b 116.73 / 149.24 / 220.84.
- Plan-9 graph arm (g9, g9b): 480 graphs (48 layers x 10 buckets) captured in 104-105 s, server healthy, then an
  illegal memory access at the first wave's forward (reported at the replay's output copy), both runs. Under
  investigation (`logs/p50/g9dbg.sh`: 4096-bucket harness cell, GPU core dumps, graphs capped at 1024).
- Latent bug found on the way (not this crash: the device lane never marks the pad table dirty): `_refresh_pads`
  rebound `pad_table` to a new tensor while captured graphs write it by address; fixed in place (lopep 4438740).

## P12: barrier latency (job 59189979, `50_copy_probes/p12_barrier.cu`, `logs/p50/p12_n{2,4}.log`)
- `nvshmemx_barrier_all_on_stream`, one PE per GPU, idle: 4n (16 PEs) 34.0 us back-to-back, 33.6 us inside a CUDA
  graph, 38 us single (host skew included); 2n 25 / 24.5 / 29 us. With 100 us injected on PE 0, the others wait
  140 us (4n): skew + latency, additive.
- So the ~180-200 us end-of-layer barrier of the harness layer-step is ~34 us of barrier and ~150 us of waiting
  for slower ranks on other nodes (node 0's four ranks finish their combines within 35 us of each other, 1308-1342
  us after the routing all-gather, which all ranks leave within 5 us). E2 alone buys ~34 us; the rest is rank skew.

## Serving illegal access of the layer graphs (round 14, then jobs 59189479, 59190067, 59190590)
- Every serving arm with graphs crashed at the first real wave (decode and prefill, all buckets or graphs capped
  at the 1024 bucket), one rank each time (DP6 x4, DP2), inside a graph replay; health-check forwards before it
  replayed fine. The harness never reproduced it: 4096 bucket (3 layers), 48 layers x 8 buckets (384 graphs, shared
  weights, 11200 swap moves, 480 replays): no access error; 3-4 out-of-tolerance rows of millions with max error
  0.0133-0.0136 = the known bf16 tolerance edge.
- Tools that failed: GPU core dumps (truncated, unreadable by cuda-gdb 12.9 on driver 580: the process dies while
  dumping); CUDA_DEVICE_WAITS_ON_EXCEPTION + cuda-gdb 12.9 attach ("Selected thread is running", probes timed out);
  a private pool per graph (OOM at 48 layers; the follow-on "capturing stream has unjoined work" is that OOM).
- Leading root cause (being verified, round 14b): both GEMM workspaces grow on demand (dispatch
  `lazy_init_buffer_tensor`, combine `create_workspace_or_expand`): a growth replaces the member and FREES the old
  buffer. A layer graph captured before the growth keeps the old address. Freed outside a capture, the old buffer
  returns to the general caching allocator; SGLang allocates its KV cache and index pools right after lopep's
  warm-up and capture and takes that memory, and the graphs keep writing GEMM workspace into it. The harness
  allocates nothing after the capture, so the stale writes hit nothing. Fix (lopep 6bf04e5): a superseded workspace
  stays allocated (scratch: a graph using its own older buffer is correct), and every growth is logged.
- Root cause CONFIRMED (job 59190751, `logs/p50/hang_r14b_g.log`): with growth logging, the dispatch GEMM workspace
  grows DURING the capture on every rank (16384 -> 29184 -> 38784 bytes, 32 growths with `capturing 1`): the
  small-bucket graphs were captured against the 16 KB warm-up workspace, which the old code freed to the general
  allocator. With superseded workspaces kept (lopep 6bf04e5), round 14b served both plan-9 arms with zero tracebacks.

## Round 14b (job 59190751; `52_round14b_arms.csv`, `logs/sglang/j4n{D,P}14b_report.txt`)
- lopep p9-dw3 6bf04e5 (tree lopep_b1): device-issued wire, layer graphs (480 per process, all buckets), early
  wire fork, workspace fix. Same allocation, decode step median ms (per-layer bracket ms) at 256 / 512 per rank:
  stock 88.83 (1.596) / 151.16 (2.762); c4b 119.49 (2.172) / 153.28 (2.769); plan 9 130.26 (2.311) and 135.08
  (2.381) / 164.23 (2.954) and 166.65 (2.973). 1024: stock 286.98, c4b 228.27 (1.26x); plan 9 not measured (the
  graphs' memory shrank the KV pool: 111k tokens vs 139k, KV usage 0.50 at 512).
- Prefill SMAX 256 input tok/s: stock 25655, c4b 25026 (0.98x), plan 9 24649 (0.96x).
- Plan 9 is SLOWER than c4b in serving (+6-13 % decode at 256) although its isolated harness layer (1.87 ms GPU
  window) beats c4b's (2.35). The per-layer bracket (SGLang patch 47: prepare_mlp + mlp + postprocess; for lopep
  only the mlp is non-empty) is 2.31-2.38 ms vs the harness 1.87: Nsight serving captures NG (plan 9) and NC4b2
  (`logs/sglang/jobN14.sh`, `--cuda-graph-trace=node`) to find the 0.45 ms.

## Serving Nsight: NG (plan 9) vs NC4b2 (c4b) (job 59192086; `logs/sglang/nsys50/n_ours_{g9e,c4b}_d256`, jobN14.sh)
- 52_timeline.py: plan 9 host range 561 us per layer-step, host lead ~29 ms (the host runs far ahead: GPU-bound),
  GPU window 2557 us with 131 us idle; c4b host range 2653 us = GPU window 2687 us with 707 us idle (host-bound).
- Per-layer period (MoE window start -> next window start), profiled: plan 9 2876 us (window 2557 + 321 gap, 188
  busy in the gap = attention etc.), c4b 4072 us (window 2687 + 1354 gap, 186 busy: 1.17 ms of host-induced idle,
  heavier under the profiler than in the clean run, where c4b's period is ~2.49 ms and plan 9's ~2.71 ms).
- So plan 9 removed the host bottleneck; what limits it is its GPU chain (~2.43 ms busy per layer in serving vs
  1.87 ms in the harness). Serving-only costs on the critical path (graph node timeline, us from window start):
  swap_decide 205-448 (210 us; 8 us in the harness: real loads trigger rounds), lane_push x2 819-915 (140 us, the
  expert-weight pushes of real moves, IN LINE on the forward stream between the derive and the dispatch planning),
  routing all-gather ends 696 (+100 us skew), pack-push 213 us, combine GEMM2 end -> bucket reduce end 483 us
  (harness 298), barrier 270 us. c4b pays the same swap_decide (212) and lane_push (136).
- Fixes (lopep p9-dw3, tree lopep_b1): LOPEP_LANE_PUSH_SIDE=1 (a1e6113, python): the pushes run on a side stream
  forked where they are issued (ahead of GEMM 1 in any shared hardware queue: no K2), joined before the first
  commit (a commit may overwrite a slot a push reads) and at the end of the layer-step. LOPEP_SWAP_DECIDE_WARP=1
  (e541763): one warp per heavy/light pair (slot ranks, memberships, the 64 exchange candidates in parallel, warp
  max of the unique selection key) instead of one thread per pair (8 of 256 threads busy at 4n): bitwise identical
  (test_swap_decide 840 cases: 401 swap decisions, 8683 moves, both variants PASS), 479 -> 91 us at 5 rounds,
  708 -> 117 us at 8 rounds (R=16, L=4, G=128, nlp=10, strongly skewed loads; ~85-95 -> ~13-15 us per round).

## Round 15 (job 59192670; `52_round15_arms.csv`, `logs/sglang/j4nD15_report.txt`) - swap fixes, one allocation
- Decode step median ms at 256 / 512 / 1024 per rank (x stock): stock 84.49 / 144.09 / 271.92; c4b 116.05 (0.73) /
  146.87 (0.98) / 218.88 (1.24); **c4s** = c4b config on the p9-dw3 e541763 binary + LOPEP_LANE_PUSH_SIDE=1 +
  LOPEP_SWAP_DECIDE_WARP=1: **113.83 (0.74) / 141.02 (1.02) / 215.99 (1.26)**; plan 9 as round 14b 127.73 / 159.50
  / 240.43; plan 9 + both swap fixes (g9s) 120.17 and 117.47 / 153.44 and 152.56 / 234.60 and 233.84. Gates (r15_*):
  graph flips, eager flips, eager CDMC 1, graph growth, c4s flips: all PASS.
- The swap fixes help both paths (-4..-8 ms plan 9, -2..-6 ms c4b); c4s clears stock at 512 per rank. Plan 9 stays
  3-8 % behind the eager path: GPU-bound, its deficit is GPU-chain time. It remains the route to parity at 256
  (c4b runs into its host floor; plan 9's host is ~0.56 ms per layer): the per-layer period must go from ~2.45 to
  ~1.76 ms (stock 84.49 / 48).

## Fence flattening (S5)
- Serving combine detail (NG capture, relative to GEMM2 end): the combine pack-push (20 blocks x 512) ends +213 us,
  the pre-reduce +211, the three combine puts +368, the barrier +780. Cause: per (node, destination) the push ends
  with `__threadfence_system()` + `__syncthreads()` before the last-block signal: 16 serial round-trip-latency-bound
  push rounds per block. LOPEP_COMBINE_PUSH_FLAT (lopep cd87b8a): a node's rows of all its L destinations in one
  pass, one fence, then the per-destination arrivals (signals still after their data, every block still arrives
  once per destination). The dispatch pack-push has the same pattern (a fence per segment): LOPEP_DWIRE_PACK_FLAT
  (lopep p9-e1, tree lopep_c4). Gates: graph flips, eager CDMC 1, graph growth PASS (combine); graph flips PASS
  (both). Harness A/B (graphs, layer-step ms): combine flat 1.891 vs 1.869 / 1.871 at 256 (neutral), 3.653 vs
  3.724 / 3.731 at 1024 (-2 %); the harness never had serving's long combine tail: serving A/B decides.
- Not parallelizable as hoped: the meta arena reads the step's counts, which the demands kernel zeroes on a
  degenerate (aborted) layer, so the arena must follow the demands (K11).
