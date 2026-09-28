# Handoff 45: SGLang end-to-end integration of the moe_ep layer

Lane opened 2026-09-26. Plan of record: `~/.claude/plans/mellow-doodling-clock.md` (approved
2026-09-26); this file is the progress record. Code lives on branch `sglang-dev` of the moe_ep
working repo (`$PSCRATCH/workspace/andrewy/moe_ep`); `main` there stays the published tree.

## Purpose and rulings
- Show the paper's mechanisms integrate simply and CORRECTLY into a serving system with a speedup;
  no figures required, stock-SGLang comparison desired (user + postdoc). DP attention (attention
  TP = 1). "Runs end to end without bugs" is the bar. Only the paper optimizations are ported.
- No fallback path: capacity is guaranteed by an exact pre-launch check plus collective growth.
- Stack: SGLang v0.5.3 (torch 2.8.0+cu129), moe_ep rebuilt with nvcc 12.9, NVSHMEM 3.2.5 module.
- Models: Qwen3-30B-A3B for bring-up, Qwen3-235B-A22B bf16 (hbm80g) as the paper-scale model.

## Log (newest last)
- 09-26 19:55 S0: branch `sglang-dev`; conda env `$PSCRATCH/conda_envs/andrewy-sglang` (py3.11,
  torch 2.8.0+cu129, sglang 0.5.3, sgl-kernel 0.3.14.post1, flashinfer 0.4.0rc3); SGLang v0.5.3
  cloned to `$PSCRATCH/workspace/andrewy/sglang`; checkpoints downloaded to
  `$PSCRATCH/workspace/andrewy/models/{Qwen3-30B-A3B (57 GB), Qwen3-235B-A22B (438 GB)}`; all
  caches under `$PSCRATCH/workspace/andrewy/caches` (home is 98 % full). Logs:
  `$PSCRATCH/workspace/andrewy/logs/sglang/`.
- 09-26 20:00 S1 code (moe_ep 96d6ea9 + next commit): `capacity.py` (exact demands, T0 host test
  PASS vs routing.py loops), `serving.py` (SharedComm / LayerState, buckets K·2^i, pads to own
  non-replicated slots, pad histogram subtracted before the band test, growth protocol),
  SwiGLU (`ModelShape.act`, `ffn1`), per-call weights, shared swap lane (`bind`), direct path
  per-step gate weights (bug fix) + per-bucket views. Torch 2.8 build of the branch in worktree
  `$PSCRATCH/workspace/andrewy/moe_ep_t28` (CUTLASS symlinked) — build complete.
- 09-26 20:10 T1 on 1 node (job 58931804, old torch 2.6 build): direct PASS (2 growths handled,
  one at prime: pad rows of a zero-token rank exceed calibrated pair/recv caps — calibration
  must add a pad allowance S_max·K); overlap PASS (1 growth mid-run); swap + load shift PASS
  (48 moves, correct); tiny shape PASS. Forced growth (`--caps-scale 0.1`) HANGS at the 4th
  growth (first in-step growth at layer 2, step 8; deterministic); GPUs 1,2 spin at 100 %,
  0,3 idle. Under investigation with a gdb watchdog (`logs/sglang/hang_probe.sh`).
- 09-26 20:45 Forced-growth hang ROOT-CAUSED (gdb + cuda-gdb watchdog, `logs/sglang/probe_grow{3,5}.log`):
  after the 4th op rebuild the first forward deadlocks — two GPUs spin in kernels, two have NO
  resident kernel yet never drain (streams blocked on memop waits), all hosts in the post-step
  `torch.cuda.synchronize()`. Rebuilding whole ops re-creates dedicated streams (16 combine wire
  lanes + dispatch streams per instance), re-primes transports and restarts the signal epoch,
  which reshuffles the stream→hardware-queue mapping under the CUDA_DEVICE_MAX_CONNECTIONS=24
  pin (the same class of stall the pin was chosen against, handoff 42). Decision: growth =
  in-place `resize_capacities` on both ops (reallocates only the capacity-sized symmetric panels;
  streams, events, signals, run epoch untouched; no re-priming) — C++ methods added to
  DispatchGemmOp / CombineWire / GemmCombineOp + bindings; Python `OverlapComm.resize`,
  `DirectComm.resize` (wires have no stream state; rebuild kept), `SharedComm._grow` no longer
  detaches/attaches the swap lane. Rebuilding the torch 2.6 tree to re-test.
- 09-26 20:49 S1 DONE: in-place resize verified — one-node matrix on the resize build (moe_ep 7a7550e,
  torch 2.6 tree; job 58931804) 7/7 PASS, 0 bad rows everywhere: forced growth overlap (4 growths incl.
  the formerly deadlocking mid-step one), direct (2), overlap (1), forced growth direct (6), swap +
  load shift (48 moves, 1 growth), swap + growth (3 growths, 48 moves), tiny shape with swap.
  `logs/sglang/matrix_r1_status.txt`. Same commit built for torch 2.8 in `moe_ep_t28`.
  SGLang v0.5.3 patch (`integrations/sglang/patches/sglang-v0.5.3.patch`, 4 files) applied to the
  clone and installed editable with the adapter (`moe_ep_sglang`) into the sglang env; launch
  scripts (`integrations/sglang/launch/{env_launch.sh,server.sh}`), `calibrate.py`, README written.
  Rule learned: never rebuild the fused ops inside a process; grow panels in place.
- 09-26 20:49 S0/S2 start: stock SGLang baseline smoke on 1n with Qwen3-30B-A3B (tp=dp=ep=4, DP
  attention, flashinfer, graphs off) + per-token expert-routing recording + gsm8k 50q
  (`logs/sglang/smoke_b30_status.txt`).
- 09-26 21:15 S0 DONE / S2 in progress (1 node, Qwen3-30B-A3B, tp=dp=ep=4, DP attention, flashinfer,
  graphs off): stock SGLang baseline serves (load 100 s, KV 82k tokens), gsm8k 50q accuracy 0.920
  (`logs/sglang/gsm8k_b30.log`). moe_ep arm (`direct`, uniform calibration `calib_30b_uni_direct`,
  heap 6 GiB, `--mem-fraction-static 0.85`): serves; first greedy generation token-identical to the
  baseline. Adapter fixes on the way (moe_ep 25c1114..833e727): slot-view shape assert; construct
  moe_ep state under `torch.device("cpu")` (SGLang builds the model with CUDA default device ->
  pinned host buffers landed on GPU); rebind `param.data` to the slot views instead of new
  Parameters (the replaced storage stayed reserved, ~15 GB, KV sizing failed); strip DP-attention
  padding rows (SGLang pads every rank's batch to the longest one, padded rows carry top-k id -1 ->
  device assert in the router) using `global_num_tokens_cpu` / `num_token_non_padded_cpu`, pad the
  output back. SGLang v0.5.3 bug found: `ExpertDistributionReq` lacks `@dataclass` (recorder
  endpoints 500) — fixed in our patch. Chain c1 (ours gsm8k -> token agreement ours vs baseline)
  running on job 58931804.
- 09-26 21:27 S2 GATE (1 node, 30B, moe_ep `direct`, uniform calibration): gsm8k 50q accuracy 0.960
  (baseline 0.920 on the same questions; 50-question noise), first generation token-identical, 0
  capacity growths (`logs/sglang/smoke_smoke_c2_status.txt`, adapter fix b924fd9: SGLang's
  synchronized `global_num_tokens_cpu` are padded batch sizes, `num_token_non_padded_cpu` is the exact
  local count; the layer buckets from the synchronized bound and gathers exact counts only for swap).
  Latency of the direct arm at this stage 107 s vs 38 s baseline for the gsm8k batch (eager, plain
  all-to-all, no placement data yet) — performance is S3's job. Token-agreement check (48 prompts)
  made concurrent; second 1-node allocation 58935447 for the calibration recording (recorder fixed).
- 09-26 21:45 Calibration recorded on 1n (baseline, gsm8k 100q as traffic, accuracy 0.950; recorder
  fixed by our patch): 4 per-rank per_token dumps (`logs/sglang/dumps_b30cal`), 1.1 M token rows over
  48 layers; `calibrate.py` now attributes tokens per rank (each dump = that rank's attention DP
  shard). Configs `calib_30b_{overlap_s0,overlap_s1,direct_s0}` (caps recv 52684 incl. pad allowance
  16384). On ONE node the placement cannot replicate (one instance per node by design), so the
  redundant slots stay empty; replication is exercised from 4 nodes on. Chain v2 on job 58935447:
  ours overlap, ours overlap+swap (gsm8k 100 + 48-prompt greedy token run each), two baseline token
  runs (noise floor). 4-node interactive allocation requested.
- 09-26 21:50 1n moe_ep `overlap` (real-routing calibration, swap off): gsm8k 100q accuracy 0.950 =
  baseline 0.950 (same questions); gsm8k batch latency 62 s (direct arm 107 s, baseline 32 s at this
  one-node, 30B, tiny-batch scale — the fixed per-layer planning cost dominates; performance is read at
  4 nodes+). ShareGPT prompt file cached at `caches/hf/ShareGPT_V3_unfiltered_cleaned_split.json`.
- 09-26 21:58 1n TOKEN AGREEMENT (48 prompts, greedy, 64 new tokens, 16 concurrent; bf16 batching
  makes greedy outputs vary between runs): baseline vs baseline 24/48 identical, 67.1 % agreeing prefix
  (the noise floor); baseline vs moe_ep overlap 26/48 identical, 70.8 % — the moe_ep arm is at the noise
  floor (`logs/sglang/chain_v2_report.txt`, `tok_v2_*.json`). moe_ep overlap: 0 growths. The swap arm
  (ov1) server died during gsm8k — under investigation.
- 09-26 22:10 Swap arm HANG root-caused (1n, warm-up request; every rank stuck at the first host sync of
  the next layer): the dispatch op wrote the GEMM-start mark only when it had rows to compute
  (`if (M_this_ep > 0)`); a rank that receives no rows in a step (possible in serving when the bucket has
  no pad rows) never released the swap lane's movement streams, which had been armed by
  `before_dispatch` -> `ev_done` never recorded -> the next collective deadlocked. The combine op already
  wrote its mark unconditionally. Fix: dispatch writes the mark whenever armed (moe_ep commit above);
  harness gets `--starve-rank` (zero-row rank + swap). 4n baseline: the v0.5.3 per-token recorder buffer
  (`chunked_prefill_size * 8`) overflowed on the DP-gathered batch (20649 > 16384 rows) -> patched to
  `* dp_size`; calibration rerun on job 58936312.
- 09-26 22:20 Mark fix verified at layer level (both trees rebuilt, moe_ep 8316a38): harness `--starve-rank`
  with swap (rank 0 receives no rows, 16 moves) PASS, without swap PASS. 4n baseline (Qwen3-30B, tp=dp=ep=16):
  healthy in 180 s, gsm8k 100q 0.950 in 38 s, 16 per-rank routing dumps (`dumps_b30cal4n2`), 48 baseline
  generations (`tok_b30cal4n2.json`). Cleanup rule: a crashed multi-node server leaves ~36-70 processes per
  node that block the next srun step ("Requested nodes are busy") — kill by pid on every node first.
- 09-26 22:25 16-rank calibration from the 4n recording (`calib_30b4n_{overlap_s0,overlap_s1,direct_s0}`):
  2.0 M token rows, replicas max 4 per expert (the node-level replication is live), caps recv 57531 /
  dispatch_recv 66424 / stage 18431 / relay 9612 / combine_send 60322 / conv 30857 / wire 18710 / pair 33375
  (x1.5 + pad allowance 16384). Chain n4 on job 58936312: ours overlap, ours overlap+swap, baseline A/B,
  baseline-graph; gsm8k 100q, 48-prompt token agreement, bench_serving (ShareGPT 256 prompts, c=16 and 64).
  Swap-arm serving re-test (fixed build) running on 1n (job 58935447).
- 09-26 22:35 Swap-arm hang #2 ROOT-CAUSED (SIGUSR1/USR2 probes + cuda-gdb, `logs/sglang/repro_sw3`):
  after the first layer with moves, the receiving rank's combine GEMM stays resident (its moved-slot
  weight gate never opens), the other GPUs idle, all hosts at the next layer's first host sync. Cause:
  SGLang's `engine.py::_set_envs_and_config` unconditionally sets `CUDA_DEVICE_MAX_CONNECTIONS=4`
  (and `CUDA_MODULE_LOADING=AUTO`) in every worker, overriding the launcher's 24 — with 4 hardware
  queues the swap lane's `cuStreamWaitValue64` waits deadlock (the false-dependency class behind the
  conn=24 pin, handoff 42); the overlap arm survived but ran throttled. Fix in our patch: `setdefault`.
  The 4n chain (run under 4 connections) was stopped after its overlap arm (gsm8k 0.950, 48 tokens
  saved, bench c=16: 108 tok/s, TTFT 261 ms, ITL 127 ms; `chain_n4_conn4_report.txt`) and restarts.
- 09-26 22:55 4n moe_ep OVERLAP (conn=24, real-routing calibration, 0 growths): gsm8k 100q 0.960;
  bench_serving ShareGPT 256 prompts: c=16 106 tok/s out, median TTFT 268 ms, mean ITL 130 ms;
  c=64 212 tok/s out, median TTFT 297 ms, mean ITL 140 ms (`chain_n4b_report.txt`). Baseline arms
  (graphs off A/B, graphs on) running now on the same nodes for the comparison. The swap arm still hangs
  under conn=24 (first layer-step with moves; receiving rank's combine GEMM waits on its moved-slot
  gate) — the connection count was a real but not the only cause; device-memory probe of the lane's
  signal words in flight (`repro_sw7`). SGLang's scheduler watchdog (300 s) kills hung workers, so probes
  run with `--watchdog-timeout 3600`.
- 09-26 23:15 4n COMPARISON (Qwen3-30B-A3B, tp=dp=ep=16, ShareGPT 256 prompts; `chain_n4b_report.txt`):
  | arm | c | out tok/s | median TTFT ms | mean ITL ms | median E2E ms |
  | moe_ep overlap (graphs off) | 16 | 106 | 268 | 130 | 16954 |
  | baseline none (graphs off) | 16 | 190 | 145 | 72 | 9467 |
  | baseline none (graphs on) | 16 | 408 | 105 | 34 | 4442 |
  | moe_ep overlap (graphs off) | 64 | 212 | 297 | 140 | 19401 |
  | baseline none (graphs off) | 64 | 403 | 146 | 69 | 9350 |
  | baseline none (graphs on) | 64 | 307 | 272 | 163 | 22258 |
  Correctness at 4n: moe_ep gsm8k 0.960 vs baseline 0.940; token agreement baseA-vs-ours 22/48 identical,
  71.9 % prefix = baseA-vs-baseB 22/48, 66.0 % (noise floor). Performance: the moe_ep arm is ~2x slower
  than the matched baseline at this scale (30B, H 2048, 48 layers, per-rank batches of a few tokens: the
  per-layer planning cost, host syncs and eager launches dominate; the paper's regime starts at 128
  tokens per rank). This is the S3 baseline reading; the performance question moves to the 235B model
  and larger batches. Swap arm: still hangs (see below).
- 09-26 23:16 Swap hang #3 narrowed (stream-state dump, `repro_sw10`): on both exchange partners phase 0
  completed, all four movement streams are stuck before phase 1's pushes, whose only gate is the COMBINE
  op's GEMM-start mark; the combine GEMMs are resident (launched) — so the combine mark was not written
  (or not to the tensor the lane waits on) in the serving process, while the dispatch op's mark works.
- 09-26 23:36 4n direct arm: gsm8k 0.920; bench c=16 86 tok/s, TTFT 415 ms, ITL 161 ms (slower than
  overlap, as in the paper's regime ordering). Swap hang #3: memop primitive self-test PASSES in the serving
  process (kernel and H2D releases); writing the marks from the forward stream does not help -> the
  exchange partners block at phase-1's PUSH copies, not the mark. Hypothesis: first-launch module load of a
  copy kernel behind the resident gated GEMM (lazy loading; the harness never swaps before warming up).
  CUDA_MODULE_LOADING=EAGER breaks SGLang startup (NCCL unhandled cuda error), so the fix is a lane warm-up
  (every push/pull primitive once to self at first use) — under test (`repro_sw16`).
- 09-27 00:15 Swap hang ROOT CAUSE (final): hardware work-queue aliasing of stream MEMORY-OPERATION waits.
  Evidence: memop primitive self-test passes in the process; marks force-written from the forward stream
  change nothing; lane warm-up (module preloading) changes nothing; 32 connections and high-priority lane
  streams change nothing; issuing the phases BEFORE the op forward flips which ranks spin (receivers ->
  pushers) = FIFO channel semantics. A `cuStreamWaitValue64` blocks its channel until satisfied; with more
  streams than CUDA_DEVICE_MAX_CONNECTIONS (SGLang's own streams + ~22 op streams + 4 movement streams)
  a movement stream shares a channel with a combine wire lane whose queued kernels finish only after the
  GEMM, which waits for the movement -> deadlock. The benchmark process has a different (benign) layout,
  which is also why the conn=24 pin was ever needed. Fix (moe_ep 3ca62cd): the lane's waits become a
  one-warp spin KERNEL (`stream_wait_geq`, new binding); kernels never block a channel. Memop waits stay
  available as `MOE_EP_LANE_WAIT=memop`. Rebuilding; then the swap arm re-test.
- 09-27 00:30 SWAP ARM SERVES. Checkpoint events showed both exchange partners blocked at phase 1's mark
  wait itself, and the spin-wait kernels never became resident: their launches sat behind blocking
  entries of other streams in shared hardware channels (the serving process's stream layout; not
  controllable from Python). Decision: serving uses an INLINE movement mode (moe_ep commit "inline swap
  movement"): on the forward stream right before each GEMM, push outgoing slots into the peers' staging
  + flag, spin-wait (one-warp kernel) for incoming flags, copy staging -> slot; no side streams, marks or
  tile gates. Deviation from the paper's 3D schedule: in serving the movement is not hidden under the
  GEMM (cost ~ the NVLink copy of <= 8 experts per swapping layer-step); the overlapped lane remains the
  benchmark's mechanism (`MOE_EP_LANE_MODE=overlap`). Follow-up for full fidelity: issue the movement from
  inside the ops on their own streams after the GEMM launch (C++), where the channel ordering is known.
  Reproduction sw22: swap arm answers correctly (token-identical first generation), 1632 layer-steps.
  Full 1n gate (gsm8k 100 + token run) running; 4n allocation requested for the swap arm's gate + bench.
- 09-27 00:33 1n SWAP ARM GATE (inline movement, real-routing calibration `calib_30b_overlap_s1`): gsm8k
  100q accuracy 0.950 = baseline, 0 growths (`smoke_sw_full_status.txt`). 4n chain n4c (swap arm gsm8k +
  token run + bench, fresh baseline) running on job 58942368.
- 09-27 00:40 Layer-level harness on the final code (main tree build): inline swap PASS (48 moves), inline
  swap with a starved rank PASS (16 moves), overlapped lane (memop waits, benchmark path) PASS (48 moves).
  Published bench path on the branch: `bench/replay.py --check 1` Qwen b1 overlap+swap 1n PASS (2.79 ms).
  Experiment switches removed from the code; `MOE_EP_TRACE` debug aids (per-layer trace, dump thread,
  checkpoints, memop self-test) kept. 1n allocation released.
- 09-27 01:05 USER GOAL RESTATED: performance at least on par with the baseline, verified at larger
  deployments; small-model slowness must be attributed (overhead vs regime). 4n swap arm (inline
  movement): gsm8k 0.940 = baseline; bench c=16 60 tok/s / ITL 230 ms (`chain_n4c_report.txt`).
  DIAGNOSIS: under DP attention the per-rank token counts are wildly uneven per step (one rank prefills
  2048 while 15 decode a few tokens); the fused ops need equal shards, so the adapter padded every rank to
  the bucket and every pad token cost its K GEMM rows + local wire work (15 x 2048 pads vs 2048 real:
  ~1.2 ms/layer, ~60 ms/step = the observed ITL gap 130 vs 72 ms). The stock path gathers the exact token
  sum. FIX (moe_ep 2fe1b96): the adapter REBALANCES the step's real tokens evenly across ranks before the
  layer (all-to-all of hidden + packed top-k; exact counts gathered once per pass) and sends outputs back
  (T*H bytes vs the stock all-gather's W*T*H); the layer uses EXACT buckets (multiples of K, lazy planners,
  eager tail) so residual padding is < K per rank. Re-measure on 4n 30B next (job 58942368); 235B stock
  baseline + recording loading on hbm80g job 58942974.
- 09-27 01:10 S4 START: Qwen3-235B-A22B bf16 stock baseline on 4 hbm80g nodes (tp=dp=ep=16): weights 40.3 GB
  per rank, KV pool 155k tokens, healthy in 150 s, gsm8k 100q 0.960 in 57 s, 16 per-token routing dumps
  (`dumps_b235cal4n`), 48 baseline generations (`tok_b235cal4n.json`). Calibration (16 ranks, 94 layers,
  overlap s0/s1) running; then the moe_ep arms + benchmarks on the same allocation (job 58942974).
- 09-27 01:33 235B moe_ep OVERLAP (rebalanced batches, exact buckets, real-routing calibration, 4 hbm80g
  nodes): weights+heap 61.7 GB per rank, KV 38k tokens, healthy in 190 s, gsm8k 100q **0.970** (stock 0.960).
  Benchmarks running (ours overlap, ours swap, baseline graphs off/on). 30B rebalanced overlap: gsm8k 0.940
  (= baseline on these nodes); bench pending.
- 09-27 01:40 NEGATIVE RESULT: the rebalanced 30B overlap arm is SLOWER (4n, c=16: 92 tok/s, TTFT 307 ms,
  ITL 149 ms vs 106 / 268 / 130 unrebalanced; c=64: 184 / 324 / 158 vs 212 / 297 / 140). Padding was not the
  dominant cost; the three extra all-to-alls and the exact-bucket planners cost more than the pads saved
  at this scale. Attribution now measured, not inferred: `MOE_EP_TIMING=1` per-phase CUDA-event
  breakdown runs (rebalance on/off) on the 30B 4n allocation (`logs/sglang/server_timing_*.log`).
- 09-27 01:55 ATTRIBUTION MEASURED (30B, 4n, ShareGPT c=16 load, `MOE_EP_TIMING`, mean ms per layer-step on
  the forward stream): rebalance ON total 2.18 = pads+loads 0.19, route+xchg 0.16, meta+check 0.34,
  dispatch 0.84, act 0.07, combine 0.60; rebalance OFF total 2.37 (pads+loads 0.40, rest equal) but
  end-to-end faster (ITL 138 vs 156 ms): the adapter's three all-to-alls cost more than the pads at this
  scale. Conclusion: the gap is the fused design's FIXED per-step latency (~2 ms/layer: wire handshakes,
  device barriers, persistent-kernel launches, planning collectives, one host sync), not padding and not
  a bug; token-proportional work is negligible at 1-4 tokens per rank. Decode under DP attention on 16
  GPUs holds concurrency/16 tokens per rank, far below the paper's regime (>= 128/rank); larger
  deployments shrink per-rank batches further. Parity is expected only for prefill-heavy traffic or very
  high concurrency -> prefill-heavy benchmark (ISL 2048, OSL 32) on the 235B next (`chain_prefill_nn.sh`).
- 09-27 02:15 235B PREFILL-HEAVY (4 hbm80g nodes, random ISL 2048 +-50 %, OSL 32, 128 prompts, c=32;
  `prefill_p235_report.txt`):
  | arm | req/s | mean TTFT ms | median TTFT ms | mean ITL ms | median E2E ms |
  | moe_ep overlap, rebalance OFF | 1.66 | 2255 | 2087 | 682 | 18705 |
  | moe_ep overlap, rebalance ON | 2.37 | 1667 | 1181 | 455 | 12221 |
  | baseline none, graphs off | 3.35 | 1717 | 1001 | 312 | 8988 |
  Reading: in the prefill phase (per-rank batches of ~128 tokens after rebalancing = the paper's b1 regime)
  the moe_ep arm reaches TTFT parity (mean better, median 18 % worse); without rebalancing one rank's
  2048-token chunk makes the other 15 pad 16k rows each (TTFT 2.1 s). Decode remains the gap (ITL 455 vs
  312 ms, ~1.5x): the fixed per-step latency at 2 tokens per rank. Rebalance stays ON by default.
- 09-27 02:20 235B prefill-heavy, baseline graphs ON: 3.52 req/s, mean TTFT 1717, median 945, ITL 300 ms
  (prefill-bound: graphs change little). 235B per-layer timing under the prefill-heavy load (decode-
  dominated average, H 4096): total 2.5 ms = pads+loads 0.23, route+xchg 0.19, meta+check 0.36, dispatch
  0.92, act 0.07, combine 0.72 (`server_timing_p235prefill.log`). Prefill-only (OSL 1) timing split by
  step size running to isolate the large-batch layer-step.
- 09-27 02:30 SESSION CLOSE. The prefill-only timing run (OSL 1) logged only decode-sized steps: the
  collector dropped every step whose GPU work was still running at the next step (the prefill steps) —
  fixed (steps queued until complete). The prefill-regime breakdown is therefore still to be measured.
  All allocations released. State: moe_ep `sglang-dev` at the commit above (main untouched); patched
  SGLang clone + adapter installed in `andrewy-sglang`; torch-2.8 build in `moe_ep_t28`; calibrations
  `calib_30b4n_*`, `calib_235b4n_*`; all runs under `logs/sglang/`.
  STATUS vs the plan: S0-S3 done (1n + 4n, 30B, all three settings correct at the noise floor, swap via
  inline movement); S4 done for overlap (235B 4 hbm80g nodes: gsm8k 0.97, ShareGPT + prefill-heavy
  benchmarks); S4 swap arm and 8n/16n (S5) not run. PERFORMANCE VERDICT so far: correctness and
  integration goals met; prefill (large per-rank batches) at TTFT parity with the stock server on the 235B
  once batches are rebalanced; decode (1-4 tokens per rank under DP attention) 1.5-2x slower because of
  the fused design's ~2.2-2.5 ms fixed per-layer-step latency (dispatch 0.9, combine 0.7, metadata+check
  0.36, plan collectives 0.4) vs the stock path's ~0.5 ms. NEXT: (1) measured prefill-step breakdown
  (fixed collector); (2) trim the fixed cost: fold the loads exchange into the routing exchange, drop the
  per-step host sync where the check can run on device, skip the per-layer combine metadata derive when
  the routing is unchanged in shape...; (3) ops-side fixed latency (barriers, persistent launches) is C++
  work; (4) decide with the user whether the e2e claim is "TTFT parity + correct integration" or requires
  decode parity (likely needs a decode-specialized small-batch path).
- 09-27 (session 2) DEVIATION REVIEW with the user, verified in code, decides the scope of the second plan.
  (a) SWAP IS ON THE CRITICAL PATH: `SwapLane.inline_phase(k)` runs on the forward stream right before
  the GEMM that reads matrix k (push outgoing slots into the peers' staging + flag, spin for every
  incoming flag, copy staging->slot, then the GEMM). The paper's 3D schedule hid the movement (side
  streams, moved-last GEMM order, tile gates). Measured: swap arm 60 tok/s / ITL 230 ms vs 106 / 130
  (30B 4n ShareGPT c=16). MECHANISM deviation; the swap arm must not be quoted until fixed. Restoration
  design (no side streams, which hang in the SGLang process by hardware-channel aliasing): sender pushes
  into the destination staging + flag on its forward stream (exists); the receiver's GEMM reads the moved
  expert FROM THE STAGING through a per-group weight pointer table, moved-last with the existing tile
  gates; staging->slot copy after the GEMM off the gated path. Residual: the sender's push-copy issue time.
  (b) THE LAST HOST SYNC is not the capacity check (free numpy on landed arrays) but host-side schedule
  metadata derivation in BOTH ops: dispatch `derive_routed_meta` (event sync on sps/uc D2H, host region /
  staging / relay layout), combine `derive_combine_meta` (host loops over splits_per_source, index builds,
  piece-table D2H). Removing it = device-side metadata derivation + device demand check with a trap on
  violation (growth becomes a restart; zero growths observed, consistent with no-fallback). Payoff beyond
  the ~0.3 ms: the layer-step becomes CUDA-graph capturable, opening SGLang's decode graphs (graphs-on
  baseline = 2x graphs-off in decode). Audits needed: host-side NVSHMEM barriers on the ops' forward path;
  the host-side swap decision (`decide_swaps` on `loads.cpu()`).
  (c) Plan tail graph (`use_graph=False` for lazy exact-bucket planners) is a small pure-perf toggle
  (4 elementwise kernels, ~20-40 us), not the 0.3 ms guessed earlier. (d) Rebalance, padding, growth path
  = integration deviations (accepted). (e) The per-pass count gather is one int per rank for rebalance
  splits + pad histogram; NOT the paper's loads exchange (unchanged, per layer) and exact (pads are
  closed-form). Branch state: sglang-dev f7716ea (42 commits over master); stray bench outputs moved to
  `logs/sglang/results/`. NEXT: plan 2 = (1) swap off the critical path per (a), (2) remove the host sync
  per (b); then the regime tests (30B 4n 40G concurrency sweep; 16n 40G prefill-heavy, -q regular).
- 09-27 (session 2) PLAN 2 APPROVED (plan file; sequencing: swap fix -> S-A device groundwork immediately,
  gate + regime tests in parallel, S-B/S-C/S-D on GO). STAGE 1 DONE in code: moe_ep 9d14e35 "staged"
  swap lane (default MOE_EP_LANE_MODE=staged): sender pushes into the peer's staging + raises the
  peer's per-slot GATE word on its forward stream; the GEMM reads the moved expert FROM THE STAGING via a
  per-expert weight pointer override (both workspace kernels, `weight_ptr_override`), moved-last, tile
  gates spin only until the push lands; gated commit after the GEMM. No side streams.
  1n (torch-2.8 build, small/tiny shapes): PASS 0 bad rows + every hosted slot == its expert after each
  swap step; late-push test (rank 0 pushes ~3 ms late via MOE_EP_LANE_DELAY): receivers spin on the gate,
  output exact; starved rank with swaps PASS (gated commit on a rank that skips its GEMM); inline and
  overlap modes still PASS; bench replay --check PASS swap 0/1. Per-phase (swap layer-steps, small shape,
  ~4.6 moves/rank): lane GPU phases push0 0.18 commit0 0.08 push1 0.08 commit1 0.03 = 0.37 ms; the
  HOST swap decision (orbit) 0.96 ms in every mode.
  4n 30B swap arm (chain s1a, ov1 = calib_30b4n_overlap_s1): gsm8k 0.960 (baseline 0.94-0.97), tokens
  saved; per-phase decode: decode+swap class 199 of 200 layer-steps (!), swap_decide 1.53 ms, lane
  phases 0.27 ms (push0 0.16), dispatch 0.83, combine 0.70 -> the band test trips on nearly every
  decode step at 2-4 tokens/rank (routing noise); the paper's regime (>=128 tok/rank) never saw this.
  OPEN for the user: a minimum-load floor / hysteresis on the band trigger in the decode regime.
  S-A started: `a2av_meta_arena_impl` (device rebuild of the dispatch pinned arena incl. the minmove
  water-fill, target = my node) + `a2av_arena_compare_impl`; `a2av_demands_impl` (device demand check,
  7 demands + violation mask, returned by `derive_routed_meta(topk, caps, direct)` on the same sync);
  knob MOE_EP_DEVICE_META 0/1/2 (dispatch_gemm.cc + overlap.py). Built (build_t28_devmeta/_demands).
- 09-27 12:02 STAGE 1 4n GATE (chain s1a, 30B, MOE_EP_TIMING=1 on both arms, so absolute numbers carry the
  collector's overhead; compare arms within the chain): no-swap ov0 c=16 88 tok/s ITL 156 ms, c=64 178 /
  166, gsm8k 0.930; swap ov1 (staged lane) c=16 53 / 260, c=64 115 / 250, gsm8k 0.960, growths 0 both.
  Per layer-step (decode): ov0 2.21 ms; ov1 3.25 (no-swap steps) / 4.31 (swap steps, 73 of 200 in the
  bench window). ov1 lane phases push0 0.12 commit0 0.02 push1 0.02 commit1 0.004 = 0.17 ms in swap steps
  -> the MOVEMENT is off the critical path (stage 1 goal met); the remaining gap is entirely the host
  decision path: swap_decide 1.39 ms in swap steps (orbit) and 0.49 ms in every other step (the
  `loads.cpu()` sync = lost host run-ahead). USER DECISION NEEDED: (a) a minimum-load floor / hysteresis
  on the band trigger for the decode regime (routing noise at 2-4 tok/rank trips it constantly); (b) the
  per-step loads sync is inherent to a same-step host decision (paper semantics); a one-step-deferred
  decision would remove it but deviates. Logs: server_s1a_ov{0,1}.log, bench_s1a_*.
- 09-27 13:20 S-A ACCEPTED + COMMITTED (moe_ep sglang-dev, "S-A: device-side derivation ..."): device
  dispatch arena / demands / combine tables agree bitwise with the host on 1n and 4n (mode 2), device-
  authoritative mode 1 passes serving checks + bench replay --check on 4n. One real fix found by the
  compare: stale minmove piece entries beyond the step's piece count (host re-zeroes; kernel now
  zeroes the tail). Mode 2 hangs in the bench path (its syncs ride the late combine derive's side
  stream) - acceptance runs use serving_check. NOT done in S-A: fixed shapes (S-A.3) and device epochs
  (S-A.5), which only pay off under graph capture (see the gate).
  DECISION GATE (nsys, 2 nodes, qwen3 shape, replay --isolated, logs/sglang/nsys/): per layer-step at
  8 tokens/rank: span 2.47 ms = GPU busy 1.27 + host-induced gap 1.19, 132 launches; at 128 tok/rank:
  2.85 = 1.83 + 1.02. GPU busy at 8 tok/rank breaks down (per step): CUTLASS GEMMs 0.41, NVSHMEM
  on-stream barriers 0.35 (3 per step), the two NCCL all-gathers 0.33 (LL, inter-node), combine
  prereduce 0.25 + pack 0.16, proxy signal 0.13, memcpy 0.11. Serving check at 8 tok/rank: 2.86 =
  1.62 busy + 1.26 gap (the loads all-gather dominates there). VERDICT: capture removes at most the gap
  (~1.2 ms) -> best case ~1.3 ms per layer-step vs ~0.5 ms for the stock path: graphs alone CANNOT
  reach decode parity; the device-side fixed cost (barriers, all-gathers, persistent kernels at tiny M)
  is structural. S-B/S-C/S-D = NO-GO as the route to decode parity unless paired with a small-M path
  (fewer barriers, one exchange, no relay rounds for tiny payloads). User decision pending.
- 09-27 14:23 REGIME SWEEP 30B 4n (40 GB nodes, job 58968164, ShareGPT 4096 prompts, MAXRR 256/rank, no timing
  collector; regime_reg30b_report.txt / bench_rreg30b_*):
  | conc | ours tok/s / mean ITL | baseline graphs off | baseline graphs on |
  | 256 | 948 / 181 ms | 2021 / 83 | 2508 / 77 |
  | 512 | 1032 / 180 | 2176 / 85 | 2565 / 81 |
  | 1024 | (rerun below) | 2233 / 78 | 2776 / 72 |
  | 2048 | (rerun below) | 2215 / 81 | 2715 / 75 |
  Reading: ITL is FLAT with concurrency for every arm (fixed-cost / launch-bound regime for all of them);
  the gap ours vs graphs-off stays ~2.1-2.2x from 16 to 512 concurrent (2 -> 32 tokens/rank). ours at
  1024/2048 timed out under the 1800 s bench cap on the first pass; rerun in flight.
- 09-27 14:45 regime rerun: ours c=1024: 990 tok/s, mean ITL 182 ms (achieved concurrency 649 ~ 40 tokens/rank),
  growths 0; c=2048 timed out again under the 1500 s bench cap (ours needs ~2x the baseline's wall time per
  concurrency level). ITL for ours is flat at 180 ms from 2 to 40 tokens/rank: the per-layer fixed cost is not
  amortized in this range; the baseline's ITL is flat at ~80 ms (also fixed-cost bound). All allocations
  released (58963326, 58965213, 58968164). Session state: moe_ep sglang-dev ae703df (stage 1 + S-A), research
  tree handoff up to date. USER DECISIONS: (1) swap band floor / decision semantics in the decode regime;
  (2) stage 3 (S-B/S-C/S-D, 24-36 days) is NO-GO for decode parity by the gate unless paired with a small-M
  path; (3) whether the e2e claim scope becomes correctness + prefill parity + mechanism fidelity.
- 09-28 RENAME moe_ep -> LoPEP (Locality-Preferred Expert Parallelism; user decision after an arXiv/GitHub/PyPI
  collision check: LoPEP is unclaimed; LOCALE collides with the Python stdlib module and ~10k i18n repos).
  Working repo `$PSCRATCH/workspace/andrewy/lopep` (was moe_ep; branches master 9f24fb8, sglang-dev dd67a16;
  package `lopep`, adapter `lopep_sglang`, library `liblopep_cuda`, env prefix `LOPEP_*`, class `LoPEPFusedMoE`,
  SGLang backend string "lopep"; calibrate writes `lopep_config.json`, old calib dirs got symlinks); build
  worktree `lopep_t28` (linked worktree repaired, full rebuild); release snapshot `lopep_release` (single
  commit 303b586 "Initial release" by "LoPEP authors", file list identical to the old snapshot modulo the
  rename, CUTLASS gitlink df8a550); the SGLang clone re-patched; ops scripts under logs/ renamed.
  HISTORY REWRITE (user request): all Claude co-authorship trailers removed from every commit on every ref of
  the working repo (filter-branch msg-filter, refs/original dropped, reflog expired, gc); pre-rewrite backup
  `logs/moe_ep_prerename_20260928_0034.bundle`. No attribution lines are added to that repo from now on.
  The research tree (this repo) is untouched by the rewrite.
