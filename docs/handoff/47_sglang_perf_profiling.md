# Handoff 47: SGLang performance profiling (plan 3, phase A) — log

Opened 2026-09-28 (session 3). Plan of record: `~/.claude/plans/the-kv-cache-size-cozy-hamster.md` (approved
2026-09-28). Analysis that motivated it: `46_sglang_perf_gap_analysis.md`. Goal (user ruling): the MoE-layer
speedup must show inside SGLang at 128–2048 tokens per rank per layer-step with overlap ON, everything
outside the MoE layer held constant; profile first (4n, 8n), port paper-consistent fixes second.

## Instruments (A0), all off by default
- SGLang clone, LOCAL patch `47_layer_bracket_sglang.patch` (applied on top of the shipped lopep patch,
  never shipped): `layers/layer_step_timing.py` + brackets in `Qwen3MoeDecoderLayer.forward` around
  `prepare_mlp` / `mlp` / `postprocess_layer` (phases `pre`, `moe`, `post`, identical span for both arms),
  keyed by (forward mode, DP padding mode, tokens-per-rank bin) with a per-phase `a + b·n_pad` fit; NVTX
  range per layer-step + mark per phase; per-forward NVTX range in `model_runner.forward`.
  Env: `SGLANG_LAYER_TIMING=1`, `_RANK`, `_EVERY`, `_DIR` (raw CSV), `SGLANG_LAYER_NVTX=1`.
- lopep `sglang-dev` (uncommitted): `Timing` keyed by `S<=bucket[+swap]`, raw samples + fit,
  `LOPEP_TIMING_RANK`, `LOPEP_NVTX=1` (range `lopep.step` + marks); the adapter opens the span before
  `exact_counts` (marks `counts`, `rebal_in`, `rebal_out`, `pad_out`) and the step is `timed_by_caller`;
  `bench/replay.py`: NVTX marks per phase, `--act {gelu,swiglu}`, `--comm-strategy stock`
  (`--stock-mode gather|sum`, `--graph 1`). Synced to `lopep_t28` by copy.
- `47_gap_report2.py`: per-phase attribution by runtime-call correlation (host span, GPU busy, launches,
  host sync time, kernel classes). Validated on the 09-27 gate captures: identical totals (128 tok/rank:
  span 2848.9 us, busy 1830.5, gap 1017.0, 132 launches) plus the new sync-time column: 767 us of
  host-side synchronize/query calls per step at 128 tok/rank, 368 us at 8.
- `logs/sglang/chain_prof.sh <jobid> <tag> <calib>`: arms ours / ours_rb / baseA / baseG (+ `_nt`
  timing-off twins), SMAXES 128 512 2048, workloads psat (ISL 4096, OSL 1, c=4·ranks), dec (ISL 128,
  OSL 256, c=128·ranks), sgpt; `XARGS` appended to every server (KV pool pin), CONTROL line per arm.

## Log (newest last)
- 09-28 13:10 First 4n interactive allocation (59035298, `-C gpu`) came MIXED (1 hbm40g + 3 hbm80g): SGLang refuses
  to start ("memory capacity is unbalanced", model_runner.init_torch_distributed). Rule: request `-C "gpu&hbm80g"`
  (or hbm40g) explicitly; mixed allocations cannot serve. Both allocations released and re-requested homogeneous.
  KV pin for 80 GB nodes: XARGS="--max-total-tokens 400000" (ours pool 524k, baseline 664k on the 30B).
- 09-28 13:40 smoke2 (job 59036000, hbm80g x4): instruments confirmed live in the serving process (stock bracket
  `[layer timing rank 0] <MODE> <PAD> n_pad<=b: ...` and lopep `[lopep timing rank 0] S<=b[+swap]: ...` lines;
  CONTROL line prints KV pool 400000 / MAXRR 256 / chunk 512). Then a HANG on the first regime step
  (ours arm: swap ON, rebalance OFF, exact buckets OFF, LOPEP_DEVICE_META=1; psat SMAX 512, c=64): every DP
  rank had served one prefill batch, all 16 scheduler main threads at `serving.py:394` (`allc.tolist()` after
  the per-step count all-gather) -> the GPU queues are blocked by the previous layer-step's work; SGLang
  watchdog (300 s) killed the workers. Chain bug found: `wait` waited on the server srun too (fixed: `wait $PW`).
  Bisect launched (`bisect_smoke.sh`): A swap off / device meta 1; B swap on / device meta 0 / trace;
  C swap on / rebalance+exact buckets on (the 09-27 validated config), watchdog 120 s.
- 09-28 14:10 BISECT VERDICT (job 59036000, 30B, SMAX 512, prefill-saturated ISL 4096 / OSL 1): the hang is the
  S-A device-metadata path in serving. B (swap ON, `LOPEP_DEVICE_META=0`): no hang, bench OK (32 prompts, 13.7k
  input tok/s). D (swap ON, DM=1, 256 prompts c=128): hang at the first regime step (trace: step 480 layer 0,
  n = [512 x8, 0 x8], "done"; step 481 never starts; all 16 ranks at serving.py:394 = host waits for the count
  all-gather behind the stuck GPU). E (swap OFF, DM=1): hang at step 433 inside `plan_meta` ->
  `derive_routed_meta` (overlap.py:108, device demands/arena + its sync), after a step where one rank prefilled
  512 and the others had 0 tokens. The 09-27 S-A acceptance (serving_check zero-token modes, --smax 8) did not
  cover this regime. Action: PROFILE WITH THE HOST PATH (DM=0; same mechanism, host-side derive), fix S-A in
  phase B (B2) with a serving_check case at --smax 512 and single-rank-token patterns. Chain bug fixed: bench
  output now goes through `bench1.sh` (full log per bench). Load-shape lesson (trace): 32-64 prompts at c=64
  is a burst that leaves 15 ranks idle; R0 uses 256 prompts at c=128 (16 ranks).
- 09-28 14:12 R0 launched: arms ours (swap ON, rebalance OFF, exact buckets OFF, DM=0), ours_rb (rebalance ON,
  exact ON), baseA (graphs off, stock bracket), baseG (graphs on); SMAX 128/512/2048; psat 256 prompts c=128;
  XARGS `--max-total-tokens 400000 --watchdog-timeout 180`; report `prof_r0_report.txt`.
- 09-28 14:40 R0 SMAX 128 (b1 regime, 30B, 4n hbm80g, psat 256 prompts c=128, KV 400k pinned, same nodes):
  | arm | input tok/s | per layer-step (EXTEND SUM, n_pad<=128): pre / moe / post = total | lopep sub-ledger |
  | ours (swap on, rebal off, DM=0) | 6927 | 0.044 / 4.239 / 0.002 = 4.285 (11376 steps) | S<=128 (11437): 3.414 = counts 0.06 pad+loads 0.50 swap_decide 0.23 route+xchg 0.25 meta+check 0.44 push0 0.11 dispatch 0.91 act 0.07 combine 0.82; S<=128+swap (6323 = 36 % of steps): 4.980 (swap_decide 1.44) |
  | ours_rb (rebalance on, exact on) | 7204 | (see ledger) | |
  | baseA (stock, graphs off) | 16332 | 0.845 / 1.166 / 0.009 = 2.019 (12768 steps) | pre = the SUM_LEN DP all-reduce (16 x 128 x 2048 x 2 B = 8 MB), moe = gate+topk on 2048 rows + fused_experts + second all-reduce |
  Reading: at b1 inside SGLang ours is 2.1x slower per layer-step and 2.4x slower in prefill throughput; idle
  ranks pay the full step (lockstep). The swap band trigger fires on 36 % of the regime steps (host decision
  1.44 ms) even at 128 tok/rank with real Qwen3-30B routing. The stock-span "moe" for ours (4.24) exceeds the lopep
  sub-ledger (3.41) by ~0.8 ms = gate/topk + adapter/python + the lopep collector's own flush (read-backs)
  inside the outer span; the `_nt` twins quantify the collector. Fits are invalid (raw ring filled by warm-up
  samples; to be made per-bin) - use the bin means.
- 09-28 15:00 R0 SMAX 512 (b4 regime, 30B, 4n, same controls): input tok/s ours 24623 / ours_rb 23759 / baseA 37843.
  Per layer-step (EXTEND SUM n_pad<=512): ours 4.715 (moe 4.702; 3600 steps), ours_rb 5.404, baseA 3.450 = pre 1.238
  (32 MB DP all-reduce) + moe 2.200 + post 0.011 (2640 steps). lopep sub-ledger ours: no-swap steps (2139) 4.215 =
  counts 0.22 pad+loads 0.45 swap_decide 0.20 route+xchg 0.30 meta+check 0.46 push0 0.11 dispatch 1.13 act 0.04
  combine 1.30; SWAP steps (1605 = 43 %) 5.752 with swap_decide 1.42. Readings: (1) the layer inside SGLang runs
  at the paper-bench speed (replay 4n b4 = 4.43 ms vs 4.2 here on a smaller-H model), i.e. the integration is
  not the loss - the STOCK path is faster than the paper's baselines at 4n on the 30B (its two all-reduces are
  1.2 ms at 32 MB); (2) the band trigger fires on 43 % of regime steps with real routing -> +1.4 ms host decision
  on those steps (~0.8 ms/step averaged), far above the paper's "~7 % of iteration"; (3) rebalance costs ~0.7 ms
  per layer-step here and buys nothing under saturation (ours_rb 5.40 vs ours 4.72).
- 09-28 15:40 R0 SMAX 2048 (b16 regime, 30B, 4n, same controls): input tok/s ours 46615 / ours_rb 49394 / baseA 44848.
  Per layer-step (EXTEND SUM n_pad<=2048): ours 9.517 (816 steps; lopep no-swap 7.884 = pad+loads 0.60 swap_decide
  0.21 route+xchg 0.37 meta+check 0.58 push0 0.11 dispatch 2.50 act 0.13 combine 3.36; swap steps 52 %: 10.19),
  ours_rb 10.688, baseA 12.505 = pre 4.378 (134 MB DP all-reduce) + moe 8.106 + post 0.021 (768 steps).
  R0 SUMMARY (30B, 4n, per layer-step ours / stock; input tok/s ours / stock):
  | SMAX (tok/rank) | 128 | 512 | 2048 |
  | ms per layer-step | 4.29 / 2.02 (2.1x slower) | 4.72 / 3.45 (1.4x slower) | 9.52 / 12.51 (1.3x FASTER) |
  | input tok/s | 6927 / 16332 | 24623 / 37843 | 46615 / 44848 (49394 with rebalance) |
  The crossover on the 30B (H 2048) at 4n lies between 512 and 2048 tokens per rank. The stock cost is the two
  SUM_LEN all-reduces (pre 0.85 -> 1.24 -> 4.38 ms) + fused_experts on W x T rows; ours' fixed part (~3 ms:
  pad+loads, plan, meta+check, swap decision, barriers) does not amortize before b4. The swap band trigger fires
  on 36 / 43 / 52 % of regime steps (128 / 512 / 2048) with real routing, +1.4 ms host decision each.
- 09-28 16:20 R5 235B (paper shape, H 4096 / ffn 1536, 4n hbm80g, KV 36k pinned, psat 256 prompts c=64) SMAX 512:
  input tok/s ours 9340 / baseA 9452 (parity). Per layer-step (EXTEND SUM n_pad<=512): ours 7.275 (5828 steps;
  lopep no-swap 6.051 = counts 0.24 pad+loads 0.63 swap_decide 0.19 route+xchg 0.29 meta+check 0.45 push0 0.20
  dispatch 1.79 act 0.09 combine 2.17; SWAP steps 60 % (4537/7606) 8.079 with swap_decide 1.43), baseA 6.534 = pre
  1.751 (64 MB all-reduce) + moe 4.769 + post 0.014 (7426 steps). Against the paper bench at the same shape and
  tokens/rank (replay 4n b4 ours = 4.43 ms isolated, host path): the serving layer-step is ~1.6 ms slower even on
  no-swap steps (dispatch+combine 3.96 vs the whole bench layer 4.43; plan phases 1.6 on top). Candidates: real
  per-step routing vs the bench's pool-matched routing (locality), rank skew absorbed in pad+loads (bench syncs
  before every iteration), pads. R2 (replay on these nodes, stock + overlap arms, swiglu) separates bench vs serving.
- 09-28 16:45 R5 235B SMAX 2048 ours: capacity growth #1 at step 1275 layer 53 (combine_conv 30992 > 29061; the 4n
  calibration used gsm8k traffic x1.5, the psat random prompts route differently), then the server HUNG after the
  in-place resize (19 watchdog dumps; bench stalled at 11 %). Second serving bug for phase B: growth path hangs at
  4n in the regime (the 09-26 verification was 1n). Pre-hang ledger (S<=2048, 200 no-swap steps): 14.60 = pad+loads
  1.40 swap_decide 0.21 route+xchg 0.34 meta+check 0.55 push0 0.21 dispatch 5.03 act 0.31 combine 6.52; swap steps
  16.34. Action: re-solve the 235B calibration with factor 2.5 (no growth in the regime) and re-measure 2048.
- 09-28 17:05 R5 235B SMAX 2048 baseA: 9964 input tok/s; per layer-step (EXTEND SUM n_pad<=2048, 1658 steps) 28.534 =
  pre 7.231 (268 MB all-reduce) + moe 21.268 + post 0.035. Against ours' pre-hang 14.60 (no-swap) / 16.34 (swap
  steps): the layer is ~1.8x faster than stock at b16 on the paper shape; the throughput row needs the r5b
  re-measure (factor-2.5 calibration, no growth). R5 SUMMARY so far (235B, 4n): ms per layer-step ours / stock =
  7.28 / 6.53 @512 (parity, -10 %), ~14.6-16.3 / 28.5 @2048 (win); tok/s 9340 / 9452 @512, (r5b) / 9964 @2048.
- 09-28 17:35 r5b (235B, SMAX 2048, factor-2.5 calibration, DM=0): NO growth; per layer-step S<=2048 15.0-15.4 ms
  (windows of 200-216 steps), EXTEND SUM n_pad<=2048 16.25 (386 steps) vs stock 28.53 -> the layer is 1.75x faster
  at b16 on the paper shape; the bench reached 202/256 prompts at 8.2 req/s (~29k input tok/s vs stock 9964,
  estimate: the jsonl was not written) and then the server HUNG (19 watchdog dumps, every dumped rank at
  serving.py:395 = the count all-gather behind a stuck GPU; no growth, host metadata). HANG-CLASS CORRECTION: the
  earlier attribution to LOPEP_DEVICE_META=1 is not proven - D/E (DM=1, 30B, 512) hung at the first regime step,
  r5b (DM=0, 235B, 2048) hung near the end of the load, while every 30B DM=0 run (R0: 3 x 2 arms, thousands of
  regime steps) never hung. Common factor to probe: swap ON + zero-token ranks + large buckets (lockstep device
  hang: all hosts reach the next step, one GPU never drains). Phase-B debugging item with the LOPEP_TRACE dump thread
  (side-stream lane state) + cuda-gdb, as in 09-26/27.
- 09-28 18:05 R2 replay head-to-head (4n hbm80g, job 59036000, lopep_t28 binary, qwen3 shape, --act swiglu, host
  metadata, --isolated, 20 iters, every arm's --check PASS on fresh payloads; total_ms iter_max_median):
  | arm | b1 (128/rank) | b4 (512) | b16 (2048) |
  | stock (all-gather + fused_experts + reduce-scatter) | 2.132 | 6.840 | 26.932 |
  | stock sum-mode (the prefill all-reduce pair) | 2.730 | 8.344 | 33.831 |
  | overlap | 3.601 | 5.266 | 12.015 |
  | overlap + swap | 3.204 | 4.816 | 11.592 |
  Reading: vs production kernels at 4n the layer LOSES b1 (1.5-1.7x), wins b4 (1.3-1.6x) and b16 (2.2-2.8x) in the
  isolated bench. Serving vs bench for OUR arm: 235B b4 7.28 vs 5.27 (+2.0 ms), b16 16.3 vs 12.0 (+4.3 ms); for
  stock: b4 6.53 vs 6.84/8.34 (none), b16 28.5 vs 26.9/33.8 (within). The serving penalty is ours alone: swap-decision
  thrash (36-60 % of steps x 1.4 ms host), plan-phase skew (plan_comm/pad+loads 0.6-1.4 in serving), pads, and the
  per-layer state of 94 layers; phase B attacks exactly these. Note the plan phases even in the bench (plan_comm
  0.6 + plan 0.8-0.9 ms at b4/b16) are ~1.5 ms of host-issued planning per step.
- 09-28 18:30 HANG SNAPSHOT (probe r5c: 235B, SMAX 2048, swap ON, DM=0, LOPEP_TRACE + dump thread, watchdog off):
  reproduced during the bench warm-up at step 1037 (layer 3, S_b 256, n = [0 x8, 236 on rank 8, 0 x7], swap moves
  on 12 of 16 ranks in each of steps 1034-1037). Device state at the stall: ranks 4, 8, 12 (the first rank of nodes
  1-3; rank 0's dump line missing) have default_stream_done=False, the other 13 ranks' default streams are done
  (idle at the host count gather). The swap lane is NOT the blocker: on the stuck ranks every movement stream,
  push event and phase is done (rank 4 has no moves at all). So the blocked work is inside the op on the NODE
  LEADERS, in a step where 15 ranks carry only pad rows (pads route locally, so some (source node, destination
  node) pairs have zero inter-node rows): candidate = the relay/gateway wire's zero-row signalling (bare signal vs
  put, dispatch_gemm.cc relay put / gateway forward) or a node-level combine lane. Probe r5d re-runs with a
  cuda-gdb attach on stall to name the resident kernels.
- 09-28 18:50 r5d (same config as r5c + cuda-gdb attach armed): NO hang; full bench: 256 prompts (541k input tokens:
  the random dataset with range-ratio 0 draws lengths uniformly up to 4096, mean ~2114 - so the earlier r5b
  progress estimate of ~29k tok/s was wrong; corrected to ~15k), input 15046 tok/s, mean TTFT 7547 ms, per layer-step
  EXTEND SUM n_pad<=2048 16.05-16.44 (500-step windows), lopep S<=2048 no-swap 14.8-15.0 / swap 16.8-17.4. 235B
  2048 row: ours 15.0k vs stock 9.96k input tok/s (1.5x; ours traced, so a floor), 16.3 vs 28.5 ms per layer-step
  (1.75x). The hang is INTERMITTENT (r5c at warm-up step 1037, r5b at 202/256, r5d never) -> a race on the node
  leaders in zero-inter-node-row steps; r5e = one more attempt under the attach. KV note: the factor-2.5 heap
  capped ours' pool at 25998 < the 36000 pin (stock had 36000); not binding at c=64, equalize next time.
- 09-28 19:05 r5e (third attempt under the attach): no hang, 14941 input tok/s (repeat of r5d within 1 %). Hang
  reproduction rate today on the 235B/2048/swap-on configuration: 2 of 5 (r5 = after growth, r5b late, r5c at warm-up;
  r5d/r5e clean). The device-state snapshot from r5c (node leaders' default streams blocked, swap lane idle) is the
  evidence to carry into the phase-B root cause; next reproduction needs the cuda-gdb attach to fire (armed in
  probe_hang.sh). Decode-b1 row (30B, ISL 128 / OSL 256, c=2048, ours vs stock) running as the last 4n item.
- 09-28 19:40 DECODE ROW (30B, 4n hbm80g, ISL 128 / OSL 256, 2048 prompts, requested c=2048, achieved ~975 both arms
  = ~60 tokens/rank average; per-rank decode batches 8-128, mostly <=16 on the stock ledger): ours 833 output tok/s,
  ITL mean/median/p99 259 / 232 / 578 ms; stock 2441 tok/s, ITL 84 / 77 / 167 ms (3.1x). Per layer-step: stock DECODE
  SUM n_pad<=16 1.211 = pre 0.518 + moe 0.684 + post 0.009 (37920 steps), MAX<=16 0.948; ours DECODE SUM<=16 4.759,
  MAX<=16 4.077; lopep S<=128 no-swap 3.316 (pad+loads 0.51 swap_decide 0.29 route+xchg 0.25 meta+check 0.43 push0
  0.11 dispatch 0.88 act 0.07 combine 0.73), swap steps 72 % (18661/25968) 4.749 with swap_decide 1.42. Decode in the
  16-128 tok/rank band is a 3-4x per-layer-step loss: ours' fixed ~3.3 ms vs stock's 1.2 ms total; consistent with
  the isolated bench (b1 stock 2.13 vs overlap 3.60) - not an integration artifact.

## Decision brief for the user: the swap band trigger in serving (phase-B item B4)
Evidence (today's ledgers, real Qwen3 routing, calibrated placement, router_c 0.25):
| run | tokens/rank | swap-step fraction | cost on a swap step |
| 30B psat 128 / 512 / 2048 | 128 / 512 / 2048 | 37 % / 49 % / 63 % | +1.4 ms host decision (`swap_decide`, orbit search on `loads.cpu()`), +0.1-0.3 ms lane push/commit, dispatch/combine +0.1-0.4 (moved-last order) |
| 235B psat 512 / 2048 | 512 / 2048 | 63 % / 75 % | same; mean 9.4 moves per swap step across the 16 ranks (r5d trace) |
| 30B decode | ~60 | 56 % | same |
Averaged over all regime steps this is ~0.7-1.1 ms per layer-step, i.e. 15-25 % of our layer-step at b4 and the
single largest item of the serving penalty (bench-vs-serving gap: ~2 ms at b4, ~4 ms at b16). The non-swap steps
still pay 0.2-0.3 ms for the `loads.cpu()` sync of the decision. The paper (§4.3, §5) triggers a swap "only when the
current GPU-load imbalance exceeds the configured load constraint" and reports routing + swap planning + metadata
at ~7 % of iteration latency on trace replays where swaps are rare; in serving the per-step load noise exceeds
c = 0.25 on most steps, so the mechanism runs at its worst case every step.
Options (all keep the swap mechanism; only the trigger policy or its cost changes):
(a) keep as is (paper-faithful, ~1 ms/step);
(b) tighten the trigger: swap only when the estimated max-load reduction exceeds the swap's own cost, or after the
    imbalance persists k steps (hysteresis) - still "the configured load constraint", no new mechanism;
(c) make the decision cheap: the orbit search on device (or vectorised), same semantics, same step (engineering);
(d) defer the decision one step (decide on step t, apply at t+1): removes the sync and the host cost from the
    critical path but deviates from the paper's same-step semantics.
Recommendation: (b)+(c); the A/B is one chain with a twin knob (`LOPEP_SWAP_MIN_GAIN` / hysteresis) and the
per-step ledger; success = swap-step fraction < 10 % in the regime with the same routing, no loss of the
placement balance (gsm8k at the noise floor, loads imbalance statistic in the trace).
- 09-28 18:15 8n (32 ranks, hbm80g, debug QOS). CONTROL CAVEAT: the two arms ran in PARALLEL on two different
  8-node debug jobs (A 59036012 = calibration + stock, B 59043125 = ours; the debug QOS caps at 2 x 30 min), so
  8n compares arms across node sets, not inside one chain; KV pool 400k / MAXRR 256 / chunk pinned equal.
  8n calibration recorded on the stock server (gsm8k 100q as traffic, accuracy 0.940, 3.23 M token rows, 32 dumps):
  `calib_30b8n_overlap_s{0,1}` (replicas max 8, redundant 64).
  30B SMAX 512 psat (512 prompts c=256): input tok/s ours 33405 vs stock 41217 (0.81x). Per layer-step (EXTEND
  SUM n_pad<=512): ours 7.759 (3600 steps) vs stock 6.283 = pre 2.380 (64 MB all-reduce over 32 ranks) + moe 3.892
  (3408 steps). lopep sub-ledger: no-swap S<=512 6.487 (468 steps: pad+loads 0.62 swap_decide 0.22 route+xchg 0.45
  meta+check 0.54 dispatch 1.82 combine 1.89 counts 0.78), SWAP steps 7.899 = 88 % of regime steps (3372/3840),
  swap_decide 1.62. Reading: at 8n the no-swap layer-step is at parity with stock (6.49 vs 6.28); the swap band
  trigger now fires on 88 % of steps (4n: 49 %) and alone turns parity into a 1.24x loss. Scaling 4n -> 8n at
  512: stock 3.45 -> 6.28 (+82 %, the all-reduce grows with W), ours no-swap 4.22 -> 6.49 (+54 %).
- 09-28 18:20 8n SMAX 2048 first pass (512 prompts c=256): input tok/s ours 36698 vs stock 48804; per layer-step
  EXTEND SUM n_pad<=2048 ours 29.33 (624 steps = 13 forward passes) vs stock 22.88 = pre 8.35 + moe 14.51 (768 steps).
  NOT a valid reading for ours: capacity growth #1 at step 725 (combine_conv 38213 > 37131; the gsm8k-traffic
  calibration x1.5 under-sizes psat routing at 32 ranks) with its re-prime of every bucket inside the short regime
  window, and the lopep 2048-bucket sub-ledger never printed. Rerun (r3b) on both jobs with 1024 prompts and ours on
  `calib_30b8n_f25_overlap_s1` (factor 2.5). Stock scaling 4n -> 8n at 2048: 12.51 -> 22.88 ms.
- 09-28 18:45 8n SMAX 2048 RERUN (r3b, 1024 prompts c=256; ours on calib_30b8n_f25, 0 growths, no hang): input tok/s
  ours 47370 vs stock 51898 (0.91x). Per layer-step EXTEND SUM n_pad<=2048: ours 24.015 (1584 steps) vs stock 20.476
  = pre 7.06 + moe 13.40 (1824 steps). lopep sub-ledger: NO-SWAP steps (280) 13.485 = pad+loads 0.79 swap_decide 0.21
  route+xchg 0.59 meta+check 0.77 dispatch 4.51 act 0.12 combine 6.30; SWAP steps (1400 = 83 %) 25.122 = swap_decide
  1.63, route+xchg 0.84, dispatch 4.54, COMBINE 15.30 (+9.0 ms vs no-swap). The combine inflation on swap steps
  appears ONLY here (4n 30B/235B b16 and 8n b4: combine on swap steps within +0.2 ms). Unexplained; prime suspect =
  the staged lane's W2 push, issued on the sender's forward stream right before the sender's combine GEMM, so a
  receiver's moved-last combine tiles wait on the sender's dispatch progress (the benchmark lane starts W2 on side
  streams at the combine-GEMM start). Needs one nsys capture of a swap step at 8n b16.
- 09-28 18:50 SWAP ANALYSIS (for the user's trigger decision). Mechanism vs paper: trigger (node-level band test on
  reference loads, C = router_c 0.25), greedy heaviest<->lightest intra-node pairing, accept only max-reducing,
  same-step decision on the host = paper §4.3/§5 as reconciled in handoff 42. Evaluation difference: bench/replay.py
  (the paper figure's source) replays ONE sampled batch per cell and the placement persists (layer.py prepare:
  self.placement = new_pl), so the swap fires in warm-up and the timed iterations are in-band (place 0.21-0.28 ms =
  loads D2H + band test); serving sees a new batch every step. Serving traffic in today's rows = random-token prompts,
  calibration = gsm8k: the paper's "rotating" (unseen demand) scenario, its worst case for swaps (hypothesis,
  testable by calibrating on the served traffic). Implementation deviation: staged lane (no side streams) vs the
  bench overlap lane. Missing measurement: a swap-OFF serving arm in the same chain (the "no-swap steps" above are
  selection-biased: in-band batches).
