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
