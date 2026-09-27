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
