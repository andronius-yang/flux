# Handoff 49: SGLang serving section for the paper (prefill and decode figures at 1 / 4 / 16 MiB)

Opened 2026-09-29 (session 3, after handoff 48). Plan 5 (approved 09-29):
`~/.claude/plans/the-kv-cache-size-cozy-hamster.md`. Code on lopep `sglang-dev` (paper mechanisms only, no
performance hacks); numbers only in this tree.

## Goal and rulings (user, 09-28)
- Two figures, prefill and decode: latency / throughput vs per-rank input budget 1 / 4 / 16 MiB (the main-perf x axis),
  ours vs stock SGLang, at the topology that shows the system best; everything outside the MoE layer held equal.
- Prerequisites first: step 1 (arm the swap lane from the GPU decision), step 2 (the capacity-growth fault).
- Prefill figure on Qwen3-235B; decode figure with SHORTENED contexts so 1/4/16 MiB fit; C stays main perf's
  (1/4 at 4n and 8n, 1/2 at 16n), a high swap rate is reported, not tuned away; swap trigger unchanged.

## Budget semantics
Bytes per rank per layer-step = tokens in the step x hidden x 2 B. Prefill: tokens = chunk per rank (`SMAX`). Decode:
one token per running request, so the budget = running requests per rank. Qwen3-235B (hidden 4096): 1/4/16 MiB =
128/512/2048 tokens; Qwen3-30B (hidden 2048): 256/1024/4096.

## Phase 0
- Launcher bug (every serving row before 09-29): SGLang divides `--max-running-requests` by the DP size under DP
  attention (tp_worker.py:144-152); server.sh passed MAXRR (documented per rank) unchanged, so the chains' MAXRR 256
  gave 16 running requests per rank at 4n and 8 at 8n. The old CONTROL lines grepped the GLOBAL value from the
  ServerArgs dump (256) and hid it; control lines now print the DP0 scheduler line (per-rank values).
  Fixed in lopep sglang-dev f82f02a: `--max-running-requests $((MAXRR * TP))`, default MAXRR 128 per rank,
  `CTX_LEN` -> `--context-length`. SGLang sizes each rank's request-to-token table for the GLOBAL count
  (model_runner.py:438-441), (MAXRR x TP) x (CTX_LEN + 4) int32 per rank, so CTX_LEN is bounded whenever MAXRR is large.
- Audit note: `scripts/oss_audit.sh` on sglang-dev fails on pre-existing site paths (env/perlmutter_sglang.sh,
  integrations/sglang/README.md) and on the snapshot-only checks (single commit, author); the f82f02a diff adds none.
  sglang-dev needs that cleanup before any merge toward master / the snapshot.
- Decode workload: `48_lcb_workload.py --truncate T` keeps the LAST T prompt tokens (the head is the instruction and
  the two worked examples, identical for every problem; the tail is the problem's code and input and ends at
  [ANSWER]). T = 48 (57 with the chat template), OSL 48: 4096 requests per rank x ~105 tokens = ~430k KV tokens per
  rank (ours' 4n pool ~524k). Files `caches/hf/lcb/lcb_exec_{history,eval}_t48_x{1,16,1024}.json`; bench1.sh works
  lcbtd (eval) / lcbhtd (history).
- Decode client: one connection per request cannot reach the 16 MiB point (65k requests at 4n, 131k at 8n; 28k
  ephemeral ports per client host and server port), so `49_decode_wave.py run` sends waves of exactly C requests
  through batched /generate calls (512 prompts per call, input_ids with the chat template, ignore_eos, temperature 0).
  Metric = the scheduler's per-rank "Decode batch" intervals (--decode-log-interval 8): decode step time =
  running / gen throughput, median over ranks and intervals (`49_decode_wave.py parse`). Continuous-arrival decode
  rows (handoff 47/48 lcbd) mixed prefill steps into those intervals; waves do not, after each wave's first interval.
- Probe driver `logs/sglang/probe49.sh <jobid>` (4n, one job): decode calibration (truncated history, SMAX 4096),
  stock and ours unpinned decode waves at 256/1024/4096 per rank, ours prefill ledger (arming breakdown), 235B pools.

## Log (newest last)
- 09-29 00:20 plan 5 approved. f82f02a (launcher), --truncate + t48 files, bench1 lcbtd/lcbhtd, 49_decode_wave.py,
  serving.py arming sub-marks (arm_sync / arm_parse / arm_mirror before swap_arm; uncommitted, synced to lopep_t28),
  probe49.sh. 4n hbm80g interactive allocation requested (2 h cap, driver scancels at the end).
- 09-29 00:28 job 59066150 (4n hbm80g interactive, m5350_g). Stage 1: decode calibration from the truncated history
  half (3824 requests, stock + recorder, SMAX 4096, MAXRR 512/rank, CTX_LEN 256), solved s1 (C 0.25, factor 2.5,
  max_tokens_per_rank 4096) -> logs/sglang/calib_30b4n_lcbtd_overlap_s1 (32 redundant experts; caps recv 162718,
  dispatch_recv 188976, combine_send 170178).
- 09-29 00:36 stage 2 stock (graphs off, unpinned): CONTROL per rank max_total_num_tokens 664381, max_running_requests
  4096, context_len 256 (launcher fix confirmed). 256 running/rank: decode step median 102.9 ms (IQR 101.5-107.9, 192
  intervals over 16 ranks; the first interval of each wave, which holds the prefill, is the outlier as designed).
  1024/rank: every scheduler died with CUDA OOM in the LOGITS all-gather, not in the MoE: under DP attention the
  vocab-parallel lm_head gathers the full-vocabulary logits of every rank's tokens, 16384 x 151936 x 2 B = 4.64 GiB per
  rank at 1024/rank (~18.6 GiB at 4096/rank, more if upcast). Fix for BOTH arms: SGLang's `--enable-dp-lm-head`
  (logits per DP rank with the full vocabulary; ~0.6 GB of extra weight per rank on the 30B), outside the MoE layer,
  recorded as a control. The client had no watchdog and waited on the dead schedulers (18 min lost); probe49.sh now
  kills the client as soon as the server logs a traceback. Restarted stages 2-5 at 00:56.
- 09-29 00:57-01:02 stage 2 rerun, stock with --enable-dp-lm-head (graphs off, 4n, 30B, unpinned pool 658451/rank):
  decode step median (scheduler intervals, 16 ranks) 256/rank 81.74 ms (IQR 80.5-86.7), 1024/rank 258.52 ms
  (IQR 254.9-277.0), 4096/rank 1044.70 ms (IQR 1000.2-1181.1, one wave: the second died on a client connection reset,
  idle keep-alive connections closed by the server between waves; client now uses force_close). KV usage max 0.63 at
  4096/rank. Stock per-layer brackets (rank 0, GPU time, DECODE MAX): 256/rank moe 0.557 of 1.403 ms/layer;
  1024/rank moe 1.993 of 4.563; 4096/rank moe 7.936 of 17.880 -> the MoE layer is ~40-45 % of stock's decode layer.
  dp-lm-head also removed ~21 ms per step at 256/rank (102.9 -> 81.7 ms): the logits all-gather.
- 09-29 01:04 stage 3 ours decode (decode calibration, device decision, unpinned pool 491065/rank): ILLEGAL MEMORY
  ACCESS on DP1 and DP11 in the first 4096-token-per-rank prefill step of the first wave (surfaced at the pad-table
  H2D in _refresh_pads, i.e. an earlier kernel), no growth. 4096 tokens per rank had never run on ours before.
- 09-29 01:06 stage 4 ours prefill ledger (SMAX 2048, MAXRR 32/rank, CTX_LEN 2048, lcbp calibration = yesterday's
  clean 4n configuration except MAXRR 16 -> 32 per rank): ILLEGAL MEMORY ACCESS on rank 1 after ~190 clean
  2048-token layer-steps (surfaced at dispatch_gemm.cc:2355 cudaEventSynchronize = the planning sync), the other
  ranks then waited in an all-gather until the 600 s watchdog. Same LOPEP environment as the clean 09-28 runs; both
  faults predate the 09-29 python edits reaching the runtime. Step 2 (correctness) now gates BOTH figures.
- ARMING BREAKDOWN (step 1, stages 3-4 ledgers, every swap class S<=8..2048): arm_sync 0.022, arm_parse 0.047,
  arm_mirror 0.645-0.648, lane prepare (swap_arm) 0.034-0.095 ms. arm_mirror = rebuild_l2p's python loop (one torch
  element store per physical slot). Vectorized (numpy stable argsort + searchsorted, identical tables on 1500 random
  placements): 0.715 -> 0.029 ms per call; also used once per host decision (swap.py). Remaining arming ~0.13-0.17 ms.
  Swap share at tiny S during warm-up: S<=8 395 of 432 layer-steps (noise), S<=2048 11 of 192 (6 %).
- 01:19 growth49 (harness, 4n): s4096 / s4096_swap first; then ima49.sh (serving localization: stage-4 config with
  CUDA_DEVICE_WAITS_ON_EXCEPTION=1, cuda-gdb attach to every scheduler worker when batches stop for 60 s), then the
  forced-growth runs.
- 09-29 01:19-01:29 STEP 2 ROOT CAUSE (the illegal access behind stages 3/4, and the likely common cause of the
  handoff-48 8n 2048 fault and the 235B 2048 intermittent hang): harness reproducer at 4n, serving_check qwen3-30b,
  every rank 4096 tokens, full capacities, static popularity, swap off: faults deterministically at step 1 (no
  growth; the "growths" count in the report was a grep of the word). ima49h.sh (CUDA_DEVICE_WAITS_ON_EXCEPTION=1 +
  cuda-gdb attach to every worker) named the kernel: a2av_bucket_map_kernel, device 1 of node 1 (rank 5), Warp
  Illegal Address inside the per-row lane search. Cause: GemmCombineOp::forward runs the receiver's plan-time
  kernels (bucket map / scan / scatter, wait-all reduce) on its reduce stream, which had NO edge to the caller's
  stream (intra, internode, wire, conv and prered streams all wait on staging_reset_event; reduce did not). The
  reduce CSR / reduce index they read are written by the combine-metadata derive on the caller's stream, or in
  plan-overlap mode 2 (<= 16 MiB per rank, i.e. every serving point) on a side stream joined into the caller's
  stream via _meta_ev. So the bucket map could read a fresh caching-allocator block before the derive wrote it:
  garbage red_ptr -> index past red_row -> illegal address (or, with in-bounds garbage, wrong buckets and lane waits
  that never fire: a hang). Larger steps = longer derive = wider window. Fix (lopep 6d06801): reduce stream waits on
  staging_reset_event (recorded on the caller's stream at forward start, before the combine GEMM is enqueued).
  Rebuilt lopep_t28 01:27. Reproducer after the fix: s4096 PASS (3 layers x 60 steps), s4096_swap PASS (staged
  lane, popularity shifting every step, 1 growth, 52 swap moves); before the fix both faulted at step 1.
  Research tree: the same edge is missing in flux src/moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.cc's bucketed
  receiver (~2770-2825; present only in the lane-chain branch at 2700). Not changed (main-perf binary frozen);
  the research harness derives the metadata on the main stream well ahead of the combine, so the window is small,
  but it is the same latent race.
- lopep c2267c5: vectorized rebuild_l2p + arming sub-marks. Runtime lopep_t28 = c2267c5 sources.
- 09-29 01:30-01:31 forced growth at 4n on the fixed binary (serving_check qwen3-30b, 2048/rank full, caps x0.5, 3 layers
  x 60 steps): static PASS (2 growths), popularity shifting every step PASS (3 growths), shifting + staged swap lane
  PASS (2 growths, 63 swap moves). Remaining for step 2: the same at 8n (debug) and the 235B 2048 re-test.
- 09-29 01:33-01:39 stage 3 rerun, ours decode on the FIXED binary (4n, 30B, decode calibration, device decision,
  --enable-dp-lm-head, unpinned pool 491065/rank vs stock 658451): every wave completed (2 per point), 0 growths,
  KV usage max 0.85 at 4096/rank. PROBE numbers (unpinned pools, stock 4096 point from one wave), not figure rows:
    running/rank   decode step median (IQR)            per-layer bracket (rank 0)      ratio (stock/ours)
    256 (1 MiB)    ours 205.93 (184-227) vs 81.74 ms   ours 3.636 vs stock 1.403 ms    step 0.40x, layer 0.39x
    1024 (4 MiB)   ours 285.61 (268-329) vs 258.52     ours 4.823 vs stock 4.563       step 0.91x, layer 0.95x
    4096 (16 MiB)  ours 749.66 (725-898) vs 1044.70    ours 12.512 vs stock 17.880     step 1.39x, layer 1.43x
  The like-for-like layer comparison is the bracket TOTAL: stock's DP gather / scatter sit outside its mlp call
  (pre 3.80 + post 6.14 ms at 4096/rank), ours' communication is inside its MoE (pre/post ~0). Ours' swap share in
  decode: S<=256 17/144 (12 %), S<=1024 24/336 (7 %), S<=4096 32/480 (7 %) of layer-steps. Arming on swap steps after
  the l2p fix: arm_sync 0.021 + arm_parse 0.046-0.062 + arm_mirror 0.079-0.096 + prepare 0.039-0.126 ms (was ~0.75).
  1 MiB loss = ours' fixed per-layer-step cost (~3 ms: meta+check 0.44, dispatch 0.75, combine 0.94, pad+loads 0.26,
  comm.prep in push0 0.26, routing 0.2) against stock's 1.4 ms all-gather layer at 256 tokens/rank.
- 09-29 01:41 stage 4 rerun on the fixed binary (ours prefill, SMAX 2048, MAXRR 32/rank, lcbp eval x8): 1920/1920
  requests, 0 growths, no fault (the pre-fix run faulted after ~190 heavy layer-steps). Ledger S<=2048 6.269 ms per
  layer-step (237 steps), +swap 6.817 (3 steps: 1.3 % swap share). 01:42 job 59066150 scancelled by the driver.
  Open after this job: step 2 at 8n (forced growth, debug QOS) + the 235B 2048 re-test (5 runs); step 1 arming
  ~0.2 ms left (target < 0.1: parse + prepare on device); Phase 3 (8n re-diagnosis); Phase 4 figure chains with the
  KV pin (smaller pool, e.g. 491065/rank at 4n decode) on both arms.
- ROUND 2 (user: "go ahead with all your proposed next steps"; report timestamps of this round are UTC).
  Arming, rest of step 1 (python, lopep working tree, synced to lopep_t28 before any round-2 job; to commit after
  the checks): the decision parse reads only this node's move lists (the lane needs its own pulls and its node
  peers' pulls from it; compare mode still parses and checks every rank's); the host mirror no longer rebuilds l2p
  (the device tables are authoritative, nothing reads the mirror's l2p on the device path); SwapLane.prepare uploads
  the unchanged-slot index from a pinned buffer (was a pageable torch.tensor(..., device=cuda)), fills both gate
  blocks with one index_fill_ on a [2, gpe] view, and scans only this node's lists; the staged lane skips the
  side-stream ev_pre record. Checks: harness staged_ref_dev (torch reference on, device decision, swaps every 2
  steps) and staged_cmp (host vs device decisions every step), at 4n and 8n.
  Jobs: 59080319 4n interactive 4 h (job4nB.sh: arming checks, 235B LCB calibration, 235B 2048 re-test x5, 235B
  prefill figure data 128/512/2048 x ours/stock graphs off/on pinned to ours' pool, 30B decode figure data 256/1024/4096
  x stock graphs on/ours/stock graphs off pinned at 480000); 59080285 8n debug (job8nA.sh: handoff-48 8n fault
  configuration + arming checks + forced growth, then 30B prefill at 8n SMAX 2048/1024 ours vs stock); 59080328 8n
  debug (job8nD.sh: 8n decode calibration + unpinned decode probe). Shared functions logs/sglang/lib49.sh.
- (UTC) 08:20 arming checks at 4n on e268463 sources: staged_ref_dev PASS (torch reference, device decision,
  122880 rows/rank, 0 bad, max_err 0.0127, 60 swap moves); staged_cmp PASS (host vs device decision every step, 63
  moves). lopep e268463 committed (the arming change).
- 08:27 235B LiveCodeBench history calibration at 4n (239 requests, SMAX 2048, MAXRR 32/rank): calib_235b4n_lcbp_
  overlap_s1 (32 redundant experts, recv_cap 85717). Ours' KV pool with it: 12949 tokens/rank (the factor-2.5
  capacities enlarge the symmetric heap; the gsm8k calibration left 38493). Enough for prefill (32 requests x ~340
  tokens/rank), and the prefill arms are pinned to it, but it is one more reason for 8n/16n in the prefill figure.
- 08:31-08:35 235B RE-TEST at 2048 tokens/rank (the pre-fix 2-in-5 intermittent hang): 5/5 benchmark runs clean
  (960 requests each, 0 growths, 0 tracebacks); input throughput 14.1-15.8k tok/s, mean TTFT 2.84-3.14 s.
  Ledger S<=2048 10.5-11.3 ms per layer-step; swap share S<=2048 7/244 (3 %), S<=512 24/94, S<=16 ~82 %.
  ARMING on swap steps now: arm_sync 0.021-0.025 + arm_parse 0.031-0.033 + arm_mirror 0.021-0.022 + prepare
  0.022-0.064 = 0.10-0.14 ms (0.72-0.76 before the round-1 l2p fix).
- 08:35-09:27 235B PREFILL at 4n (LCB eval, MAXRR 32/rank, KV pinned 12949/rank on every arm, ours = device decision):
    SMAX/rank      ours in tok/s  stock (graphs off)  ratio   mean TTFT ours/stock   MoE layer-step ours / stock layer   swap %
    128 (1 MiB)    4700           6927                0.68x   8.59 / 5.05 s          3.628 / 2.421 ms                    29.7
    512 (4 MiB)    11004          9690                1.14x   2.75 / 3.70 s          4.550 / 6.863 ms                    19.1
    2048 (16 MiB)  no data: bench1 needed lcb_exec_eval_x32.json, never generated (now generated); rerun queued after
                   stage 4 on the same nodes with the same pin (job8nP.sh 59080319 "2048", tag j4nP).
  Stock graphs on (baseline-graph) cannot start on the 235B at 4n: CUDA graph capture overflows flashinfer's
  workspace (batch_prefill_tmp_v 536 MB, graph batch sizes up to 512) and the server does not exit; SGLang graphs only
  decode steps, so the prefill figure's stock arm is graphs off (the graphs-on starts were cancelled on sight).
  (layer-step: ours = lopep ledger class S<=SMAX weighted over swap / no-swap; stock = SGLang decoder bracket total
  of the EXTEND class, i.e. gather + MoE + scatter.)
- 09:27-09:52 30B DECODE FIGURE DATA at 4n (LCB eval truncated t48, OSL 48, --enable-dp-lm-head, KV pinned 480000
  tokens/rank on every arm, 2 waves per point, scheduler decode intervals 16 ranks x 12 each):
    run/rank       stock graphs on        stock graphs off        ours (device decision)   ours vs best stock
    256 (1 MiB)    78.23 (77.1-83.3) G    81.45 (80.2-86.1)       203.73 (187.3-220.9)     0.38x
    1024 (4 MiB)   258.24 (254.0-278.1)   257.60 (254.4-279.1)    284.29 (264.4-320.3)     0.91x
    4096 (16 MiB)  990.43 (982.6-1067.0)  1001.18 (981.5-1131.4)  746.24 (722.8-845.9)     1.33x
  (decode step median, IQR, ms; G = graphed: stock graphs cover up to 512 requests per rank, so 1024/4096 run eager on
  both stock arms.) Output tok/s per GPU: stock on 3273/3965/4136, off 3143/3975/4091, ours 1260/3608/5490.
  Per-layer bracket (rank 0): stock off 1.384/4.551/17.999 ms, ours 3.578/4.761/12.552 ms. Ours' swap share 5.2 /
  9.4 / 7.5 % of layer-steps. 0 growths, 0 tracebacks on every arm. Ours pinned = ours unpinned (probe) within 1 %.
- 09:52-10:09 235B 2048 prefill pair rerun on the same 4n nodes (job8nP.sh, tag j4nP, PIN 12949): ours 16521 in tok/s,
  mean TTFT 2.82 s, MoE layer-step 10.84 ms, swap share 0.4 %; stock graphs off 9986 in tok/s, mean TTFT 5.52 s,
  decoder layer bracket 30.69 ms. 10:09 job 59080319 scancelled by the driver.
  235B PREFILL AT 4n, COMPLETE (KV 12949/rank on every arm, LCB eval x2/x8/x32, ~80 prefill steps each):
    SMAX/rank      in tok/s ours / stock   ratio   mean TTFT ours / stock   layer ours / stock (ms)   swap %
    128 (1 MiB)    4700 / 6927             0.68x   8.59 / 5.05 s            3.63 / 2.42               29.7
    512 (4 MiB)    11004 / 9690            1.14x   2.75 / 3.70 s            4.55 / 6.86               19.1
    2048 (16 MiB)  16521 / 9986            1.65x   2.82 / 5.52 s            10.84 / 30.69             0.4
  Queue at 10:10: 8n debug jobs estimated ~14:10 / ~14:40 (drivers start on grant), 16n regular no estimate.
- ROUND 3 (user, local ~12:30: "why did you not implement the device kernel? ... host to device sync ... detrimental
  ... at smaller batches"; "queue with small walltimes for better backfill"; "is the 8n 95 % swap cause solved?").
  Answer given: the kernel was skipped on an average-swap-share estimate (arming ~0.1 ms x 3-12 %), which was the
  wrong basis: swaps concentrate at small steps (S<=16 80-90 %, S<=128 30 %, S<=256 5-12 %, S<=2048 0.4-6 % of
  layer-steps), and every swap also forced a full stream sync on the next step of that layer (host pad-table rebuild
  uploaded from pageable memory), which I had not counted. The per-layer planning sync (host dispatch metadata +
  capacity check) exists on every layer-step and is NOT removed by the lane kernel. Also found: comm.prep zeroed the
  whole GEMM output buffer every layer-step (recv_cap x ffn1: ~500 MB on the 30B decode calibration, ~0.26 ms).
  8n 95 %: mechanism (small, noisy steps trip the band) supported by the 4n S-dependence; the 95 % was measured at 8
  running requests per rank (launcher bug); 8n re-measure queued (job8nS). USER DECISION (AskUserQuestion): "Swap
  path first" (device lane + device pad table + zero only used rows; the planning sync stays for now).
  Queue: cancelled the 30 min / 1.5 h requests; resubmitted short: 8nH 12 min (started within ~20 min), 8nS 15 min.
- 12:46-12:50 STEP 2 AT 8n PASS (job8nH, e268463 binary): f2048_flip (the handoff-48 8n fault configuration) PASS
  1 growth; f2048_flip_staged PASS 447 swap moves; staged_cmp (host vs device decision every step) PASS 259 moves;
  flip_staged (caps x0.5) PASS 3 growths 447 moves; 0 illegal, 0 tracebacks. Job 4 minutes, scancelled by driver.
- DEVICE LANE (lopep working tree, built into lopep_t28 at ~13:00, cmake re-run): src/planner/lane_device.cu
  lane_arm (gate words of unchanged slots, moved-last encoding, overrides), lane_push (matrix k: P2P copy of every
  slot given to a node peer into its staging, system fence, last block raises the peer gate words), lane_commit
  (after the GEMM: wait gate, staging -> slot, clear override), pad_rebuild (next step's pad table from the new
  p2l), lane_device_preload (lazy-loading guard); all return at once when the block reports no swap. serving.py:
  arm + pad rebuild launched right after swap_decide; after the planning sync the host reads only its own pulls
  (GEMM kwargs); no host mirror / host pad refresh in device mode; compare mode rebuilds the device pad table into a
  scratch and checks it against the host rebuild. LOPEP_LANE_DEVICE=0 keeps the python lane (A/B). OverlapComm.prep
  (used_only=True) zeroes only out_buf[:m] in serving; the benchmark keeps the full zero outside its timed window.
  tests/test_lane_device.py (1 GPU). Validation job jobV.sh (4n, 30 min) queued; the 8n swap-rate job waits on the
  .validating lock.
- 12:55-13:08 validation try 1 (jobV, 4n): unit tests PASS (test_lane_device: 149 swap steps, 2636 rank checks, 1924
  quiet checks; test_swap_decide PASS); harness: staged_cmp PASS, staged_ref_pylane PASS, but staged_ref_dev and
  s4096_swap_dev FAILED on the harness's own slot check, which compared slot weights with the HOST mirror placement
  (st_l.placement.p2l) that the device lane deliberately no longer updates. Fix: the harness checks against the
  device table (st_l.p2l), valid in every mode (lopep examples/serving_check.py).
- 13:08-13:21 validation try 2 (jobV, 4n): all PASS: staged_ref_dev (torch reference, device lane, 122880 rows/rank,
  0 bad, max_err 0.012, 60 moves, slot checks vs the device table), staged_cmp (host vs device decision + device vs
  host pad table at every refresh, 63 moves), s4096_swap_dev (52 moves), staged_ref_pylane. Lock removed. lopep
  068e5e4 committed (device lane) = runtime lopep_t28 sources.
  SAME-BINARY A/B, 30B decode 4n, KV 480000/rank, dp-lm-head, 2 waves (decode step median, IQR, ms):
    run/rank   device lane             python lane (LOPEP_LANE_DEVICE=0)   earlier figure row (e268463)   stock (best)
    256        187.25 (180.2-195.9)    208.54 (188.2-227.9)                203.73                         78.23
    1024       266.31 (260.5-295.8)    279.04 (263.8-316.0)                284.29                         257.60
  Ledger (S<=1024): pad+loads 0.09 (device) vs 0.11-0.38 ms (python; the post-swap pageable pad upload), no-swap
  layer-step 4.42-4.47 vs 4.47-4.74 ms; push0 0.046 ms in both (was 0.265: the full out_buf memset, fixed for both
  arms). Remaining per-layer host cost at small S: the planning sync (meta+check 0.46-0.52 ms). Swap share in decode
  varies strongly by phase of a wave (windows from 1 % to 99 % of S<=1024 layer-steps).
- 13:21 job8nS started on 068e5e4 (8n swap rate, 30B LCB prefill).
- 13:21-13:30 8n SWAP RATE RE-MEASURED (job8nS, 068e5e4, 30B LCB prefill eval x8, MAXRR 32/rank fixed, 8n
  calibration C 1/4). Swap share of layer-steps by bucket, summed over every ledger window of the server run:
    8n SMAX 2048: all 66.5 % | S<=8 91 %, S<=16 93 %, S<=512 54 %, S<=1024 68 %, S<=2048 17 %
    8n SMAX 512:  all 49.1 % | S<=8..256 95-100 %, S<=512 27 %
    4n for comparison: 30B SMAX 2048 S<=2048 2 %, S<=512 29 %; 235B S<=2048 1 %, S<=512 18-26 %, S<=128 18 %;
    30B decode S<=256 50 %, S<=1024 50-54 %, S<=4096 5-21 %.
  Reading: the handoff-48 95 % was dominated by small steps (8 running requests per rank under the launcher bug);
  with it fixed, full 2048-token steps swap on 17 % at 8n vs 1-2 % at 4n, and partial steps (S<=512-1024) on
  27-68 % vs 18-54 % at 4n. Decisions are the paper's (8n compare mode PASS: device = host every step); C stays 1/4
  per ruling, so this is reported, not tuned. The handoff-47 8n combine growth on swap steps is gone: steady-state
  S<=2048 swap layer-steps 7.97 ms vs no-swap 8.64-9.75 ms (first windows include start-up effects).
  8n prefill point on the way: 30B SMAX 2048 ours 46123 vs stock 38894 in tok/s (1.19x), mean TTFT 1.67 vs 2.65 s
  (unpinned pools: ours 540132, stock 683133 per rank). SMAX 512: ours 35354 in tok/s (no stock arm in this job).
- 13:33-13:51 4n refresh piece 1 on 068e5e4 (job8nP.sh "2048", PIN 12949): 235B SMAX 2048 ours 16775 vs stock
  9827 in tok/s (1.71x; e268463 run 1.65x), mean TTFT 2.81 vs 5.62 s, layer-step 11.76 vs 31.02 ms, swap 2.7 %.
  chainfig.sh BUG: the spec was read on stdin inside the loop and salloc/srun consumed it, so every chain stopped
  after its first piece. Fixed (spec on fd 3, commands on /dev/null). The three 8n/16n chains still waiting for
  their first allocation were stopped (by PID, pending jobs cancelled) and relaunched on the fixed runner; the 4n
  remainder relaunched as c4nb.
- 13:33-14:40 4n FIGURE DATA ON 068e5e4 (device lane), same 4 nodes per piece, every arm KV-pinned, 0 growths,
  0 tracebacks. PREFILL Qwen3-235B (LCB eval x2/x8/x32, MAXRR 32/rank, KV 12949/rank):
    SMAX/rank      in tok/s ours / stock   ratio   mean TTFT ours / stock   layer ms ours / stock   swap %
    128 (1 MiB)    4807 / 6714             0.72x   6.79 / 4.90 s            3.23 / 2.50              13.1
    512 (4 MiB)    11276 / 9544            1.18x   2.47 / 3.36 s            4.51 / 6.31              7.4
    2048 (16 MiB)  16775 / 9827            1.71x   2.81 / 5.62 s            11.76 / 31.02            2.7
  DECODE Qwen3-30B (truncated LCB eval, OSL 48, dp-lm-head, KV 480000/rank, 2 waves; step median, IQR, ms):
    run/rank       stock graphs on        stock graphs off       ours                   ours vs best stock
    256 (1 MiB)    77.08 (76.1-82.2) G    80.95 (79.2-85.7)      186.47 (179.9-190.5)   0.41x
    1024 (4 MiB)   256.89 (253.6-281.1)   257.19 (252.6-277.3)   265.67 (260.5-294.2)   0.97x
    4096 (16 MiB)  985.52 (975.6-1065.5)  996.03 (974.6-1085.3)  724.91 (718.5-857.9)   1.36x
  Per-layer bracket: stock off 1.385 / 4.53 / 17.89 ms, ours 3.60 / 4.72 / 12.43 ms; ours' swap share 27 / 9.5 / 7.7 %
  (last window). Versus the e268463 rows: prefill 0.68/1.14/1.65x -> 0.72/1.18/1.71x, decode 0.38/0.91/1.33x ->
  0.41/0.97/1.36x (device lane + used-rows zeroing).
  8n / 16n chains: first pieces (calibrations) still pending at 14:40.
- ROUND 4 (user, 15:4x: "how have we not secured allocations for 8n/16n yet? how long have we been queueing? ... what
  are the sequence length and batch size of prefill / decode? ... how much time is spent on attention (and is DP
  attention properly configured, with what params) and how much on MoE?").
  QUEUE: 8n debug waits so far 19 / 53 / 52 / 109 min per piece (hbm80g, reason Priority; fair-share 0.23 on
  m5350_g; debug QOS: 2 running / 5 submitted per user); 16n regular never started (08:58-12:30 and 13:33 onward).
  Perlmutter has ~256 general hbm80g GPU nodes (241 allocated, 4 idle at 15:5x) vs ~1530 hbm40g. The chains ran
  pieces SEQUENTIALLY (each piece paid a full queue wait): restructured to parallel pieces with fixed per-topology
  KV pins (prefill 12949, decode 480000 = the 4n pins; pin_j*.txt): 8n prefill "2048" (kept queue position) +
  "512 128", 8n decode ours (kept) + baseA + baseG, 16n calib (kept) + "2048" + "512 128" with
  --dependency=afterany:<calib job>. Done so far at 8n: both calibrations (235B prefill 14:51, 30B decode 15:47).
- ATTENTION VS MoE (jobA.sh, 4n, 068e5e4, SGLang bracket with SGLANG_LAYER_TIMING_ATTN=1 = 49_attn_bracket_sglang.patch;
  49_breakdown.py; rank 0 CUDA events; per layer ms, per step = x layers):
  30B decode (48 layers), attn | MoE block (stock = gather + MoE + scatter) per layer, shares of the step:
    256/rank   ours attn 0.171 | MoE 3.846   step 186.7 ms: attn 4 %, MoE 99 %*
               stock attn 0.166 | MoE 1.488 (0.431 + 0.548 + 0.510)   step 82.1 ms: attn 10 %, MoE 87 %
    1024/rank  ours attn 0.477 | MoE 5.091   step 264.4: attn 9 %, MoE 92 %*
               stock attn 0.477 | MoE 4.774 (0.987 + 1.984 + 1.803)   step 261.0: attn 9 %, MoE 88 %
    4096/rank  ours attn 1.636 | MoE 13.322  step 729.6: attn 11 %, MoE 88 %*
               stock attn 1.647 | MoE 18.869 (3.924 + 8.036 + 6.909)  step 1040.5: attn 8 %, MoE 87 %
    (* bracket means include slow outlier steps, scheduler step times are medians: ours' shares sum slightly
    above 100 %; rest of forward 3-24 ms, time outside the forward ~0-24 ms.)
  235B prefill (94 layers), full 2048-token steps: ours attn 1.740 | MoE 11.350 -> forward 1233 ms/step: attn 13 %,
    MoE 87 %; stock attn 1.591 | MoE 27.418 (gather 6.775 + MoE 20.610) -> forward 2730 ms: attn 5 %, MoE 94 %.
  Attention per layer is the same in both arms (control holds); stock's MoE block at 16 MiB decode is more
  communication (gather + scatter 10.8 ms) than expert compute (8.0 ms).
  CAVEAT: decode contexts are 56 -> 104 tokens (truncated prompts, so 4096 requests per rank fit in KV); attention
  per decode step grows with context, so at realistic contexts its share would be larger than 4-11 %.
  SEQUENCE LENGTH / BATCH: prefill 235B: prompts 273-424 tokens (median 337, chat template), output 4; batch = the
  per-rank chunk SMAX 128 / 512 / 2048 tokens per step (0.4 / 1.5 / 6 prompts per rank per step; chunked prefill
  splits long prompts), running cap 32 requests/rank, client concurrency 16/rank. Decode 30B: prompt 55-56 tokens
  (last 48 of the LCB prompt + template), output 48 (ignore_eos) -> context 56 -> 104; batch = running requests per
  rank 256 / 1024 / 4096 = decode tokens per rank per step.
  DP ATTENTION CONFIG (ServerArgs): tp = dp = ep = 16 at 4n (32 at 8n), enable_dp_attention (attention TP 1: every
  GPU holds the full attention weights and runs attention for its own requests), attention_backend flashinfer (prefill
  and decode), load_balance round_robin, schedule fcfs, page_size 1, radix cache off, overlap schedule off, CUDA
  graphs off except the stock graphs-on arm (cuda_graph_max_bs 512), enable_dp_lm_head on decode runs,
  mem_fraction_static 0.85, chunked_prefill_size = SMAX per rank, max_prefill_tokens 16384, context_length 2048
  (prefill) / 256 (decode), max_total_tokens pinned; DP padding: DECODE MAX, EXTEND SUM; moe_a2a_backend none (stock:
  all-gather -> local experts -> scatter) vs lopep (ours, 32 redundant experts).
- ROUND 5 (user: "switch to 40g so we can actually get 16n runs"; can 16 MiB decode run at 16n on 40 GB with shorter
  sequences?). 40 GB budget ~33.5 GB for weights + heap + KV (mem_fraction_static 0.85 of ~39.4). Heap
  (lopep.heap, 6 GiB floor): 235B 4n 9 / 8n 12 GiB at 3x headroom, 6-7 GiB at 1.5x; 30B decode 4n 9 / 8n 11 at 3x,
  6 at 1.5x. After weight load at 4n (80 GB): 235B stock 40.3 GB, ours 64.0 GB (+9 heap, +7 redundant slots, +8
  buffers); 30B decode ours 21.3 GB. ESTIMATES at 16n on 40 GB: 235B ours ~36+ GB before KV (attention weights are
  replicated by DP attention, ~12 GB) -> does not fit; 30B ours 15-22 GB -> KV ~115-190k tokens/rank -> 16 MiB decode
  (4096/rank) fits only with ~28-46 tokens per request (vs 56 -> 104 now); 1 and 4 MiB fit.
  USER DECISIONS: 16n prefill = BOTH (30B prefill at 16n on 40 GB now + keep the 235B 16n pieces on 80 GB); decode
  context = short ctx ONLY at 16n (4n / 8n keep 56 -> 104).
  16n 40 GB pieces (regular QOS, chainfig2.sh GPU_C=gpu&hbm40g): 59103871 p49_n16g40p1_1 (p1_16n.sh: 30B prefill
  calibration SMAX 4096 + decode calibration, both C 1/2, then jobD16 probe: ours unpinned with heap at 1.5x
  headroom -> pin, heap and context T/OSL files), then with --dependency=afterany:59103871: 59103899 p49_n16g40p2_1
  (jobP30 30B prefill SMAX 4096 / 1024 / 256, LCB_REP 128 / 32 / 8, ours pinned at min(pool, 50000), stock at ours'
  pool), 59103896 p49_n16g40p3_1 (jobD16 ours + stock graphs off), 59103895 p49_n16g40p4_1 (jobD16 stock graphs on).
  Other sessions of the same Unix user have jobs named pz* in the queue (not ours; pzd_async1n shares our debug
  per-user limit).
- 17:10-17:24 8n DECODE ours (job 59092316 p49_c8nD_2, 30B, KV 480000, 2 waves): 256/rank 219.38 ms (IQR 218.6-226.5),
  1024/rank 378.16 (366.4-422.1), 4096/rank 1056.31 (1025.6-1232.4); 1 growth, 0 tracebacks. Stock arms at 8n queued.
- ROUND 6 (user, ~20:00: "why is 8n decode slower than 4n?"; "are these allocations blocking other users?").
  8n vs 4n ours decode, per layer-step, same per-rank batch (ledger over all windows): S<=256 3.27 -> 4.10 ms,
  S<=1024 4.84 -> 6.60, S<=4096 12.26 -> 18.08; dispatch + combine grow the most (4096: 10.39 -> 15.14 ms: 7/8 of every
  token's copies leave the node vs 3/4, fewer copies merged per destination node, fixed per-GPU inter-node bandwidth,
  more peers), planning (route+xchg + meta+check) 1.12 -> 1.70 ms (load exchange over 32 ranks, 32 x 32 host tables),
  swap share 21 -> 34 % (256: 51 -> 81 %). Stock's all-gather volume per GPU also doubles at 8n (its 8n arms were queued).
  QUEUE CONTENTION: account m5350_g is shared; user yuetu had 37 pending 1-node 24 h hbm80g jobs (fe4-espin-*), and
  yufeid is 71 % of the account's decayed usage (every user on the account shares fair-share 0.228; 25.6 node-hours
  today). Our 80 GB 8n / 16n requests competed for the same ~256-node hbm80g pool. USER RULING: "cancel ours only and
  let them get priority over us, dont change ANY of their jobs". 20:07:33-34 cancelled all 11 of ours (p49_*, IDs from
  our own salloc logs, names verified before each scancel); yuetu's 37 untouched. (The other session's pz_verl1n /
  pz_naive1n / pz_overlap_decode were cancelled at 19:53:34 by that session, not by us; pzd_async1n still pending.)
  NOT DONE (cancelled): 8n 235B prefill arms, 8n 30B decode stock arms, 16n 235B prefill, 16n 30B (40 GB) prefill and
  decode. Done at 8n: both calibrations (235B prefill, 30B decode) and the 30B decode ours arm. Scripts ready to
  resubmit when the user allows: spec_p8a.txt / job8nP.sh, jobD.sh (8n), p1_16n.sh + spec_16n40_p*.txt (16n 40 GB,
  chainfig2.sh GPU_C=gpu&hbm40g), spec_p16a/b.txt (16n 80 GB).
- ROUND 7 (user: root-cause the 1 MiB loss; nsys evidence?; deep dive of each step: remaining host work, is the
  metadata computation fused or repeated host calls; "the mechanisms are there ... not sure the best way to achieve
  and plan them is in place"). FIT DECOMPOSITION (4n, 30B decode, SGLang bracket = MoE block incl. communication):
  stock 0.62 ms + 4.5 us/token, ours 3.14 ms + 2.5 us/token per layer-step -> crossover ~1300 tokens/rank (~5 MiB);
  predicts 0.47x / 0.91x / 1.40x at 256 / 1024 / 4096 (measured 0.41 / 0.94 / 1.42). Ours' per-token cost is 44 %
  lower, its fixed cost 5x higher. Old nsys (09-27, 2n, harness, pre-fix) supported it; no capture of the current path.
  NSYS CAPTURE (user OK; job 59112026, 4 x 40 GB nodes, 21:54-22:04, jobN.sh): SGLang serving, 30B decode 256/rank,
  KV 100000 both arms, node 0, 20 forward steps, NVTX (SGLang layer + lopep phases, LOPEP_TIMING on = ~17 event marks
  per layer-step). Reports logs/sglang/nsys49/n_{ours,stock}_d256.{nsys-rep,sqlite}; analyzers 47_gap_report2.py and
  49_hostcalls.py (per phase: host / in-API / host-code / sync us, API calls, copies by direction and memory kind,
  kernels). Caveats: 40 GB GPUs; ours' instrumentation heavier than stock's (17 vs 4 marks per layer); nsys API tracing
  inflates host time in proportion to call count (forward in capture ours 266 ms vs 187 unprofiled, stock 76 vs 78).
  OURS per layer-step (median of 912): span 3.87 ms, GPU busy 2.60, GPU IDLE 1.27 (host-bound), 479 CUDA API calls,
  212 GPU activities. STOCK MoE layer incl. attention: span 1.73 ms, GPU busy 1.61, idle 0.13, ~82 calls, 25 activities.
    phase        host us (API/code/sync)   GPU us   calls  content
    pad+loads    333 (98/231/0)            115      25     Python pads, 6 D2D copies, NCCL all-gather of loads (94)
    swap_decide  126 (45/81/0)             233      10     swap_decide kernel 216 us (single block), pad_rebuild, lane_arm
    route+xchg   277 (94/179/0)            138      26     router kernels 66, a SECOND NCCL all-gather (105)
    meta+check   594 (337/250/276)         211      16     scatter/count kernels, 2 D2H (pinned 6 KB) + the planning SYNC,
                                                           host table derive + capacity check
    dispatch     1426 (630/797/7)          783      181    51 cudaMemcpyAsync (37 intra-node peer copies ~0.84 MB, 5 pinned
                                                           H2D metadata uploads ~6.8 KB, 4 pageable 8 B), 46 launches,
                                                           31 event records, 15 stream waits, 12 event polls, 6 memsets;
                                                           kernels: GEMM1 711 (incl. gate waits), blocking NVSHMEM proxy
                                                           puts 655 (x2.9), NVSHMEM barrier 387, compress-plan scan/conv/red
                                                           156, gathers / searchsorted / workspace prep ~120
    act          99                        25-68    5
    combine      576 (381/191/8)           871      183    55 stream waits, 36 event records, 31 copies (21 peer ~1.1 MB),
                                                           21 launches, 9 memsets, 14 stream memops; kernels: blocking
                                                           proxy puts 1512 (x2.7, summed), pre-reduce 551, barriers 454,
                                                           bucket reduce 317 (x6.7), GEMM2 204, pack 175
    arming/pushes/commits/marks ~300 host (mostly instrumentation; lane kernels 56 + 33 + 13 + 8 us GPU)
  FINDINGS: (1) at 1 MiB ours is host-bound (GPU idle 1.27 ms/layer-step vs stock 0.13); (2) the host work left is the
  dispatch / combine ORCHESTRATION (364 of 479 calls: per-peer intra-node copies issued one cudaMemcpyAsync each,
  multi-stream event record / wait choreography, event polling, per-round metadata uploads) plus the planning sync with
  host table building (meta+check); the swap path is no longer a host cost; (3) metadata is NOT fused: counts / scatter
  on GPU -> D2H + sync -> host tables + capacity check -> 5 H2D uploads -> more plan kernels in dispatch (compress plan,
  searchsorted, gathers, consumer build, workspace prep) -> the combine derives its own metadata again (meta derive,
  bucket map / scan / scatter); two NCCL all-gathers per layer (loads, routing exchange); (4) GPU fixed latency: the
  single-block swap decision kernel 216 us every layer-step on the critical path; blocking NVSHMEM proxy puts (CXI
  wire-ordering rule) and 3 barriers per layer. The ~41.5k small pageable D2H copies per rank in the capture fall
  outside the steady-state window (between-forward work is equal in both arms: ~55 vs 50 calls, 3.6 vs 16.9 ms).
- ROUND 8 (user: recreate the evidence behind the 1 MiB root cause, to plan fixed-cost campaigns next). Re-derived
  from the capture (49_hostcalls.py rerun -> logs/sglang/nsys49/deep_ours_v2.txt), the clean ledger and a code map of
  lopep 068e5e4 (= the captured source; lopep_t28 src + python identical). CORRECTIONS to round 7 marked (C).
  FIT SOURCE: SGLang bracket least squares over 4788 layer-steps of j4nD (80 GB, 4n decode): stock pre 0.128 + moe
  0.356 + post 0.136 = 0.62 ms + 4.51 us/token; ours 3.128 + 2.54 us/token. Two-point fits from jA give stock
  0.33-0.39 ms: stock's fixed cost is 0.3-0.6 ms; crossover 1300-1400 tokens/rank either way.
  CLEAN LEDGER at S<=256 (j4nD, unprofiled, LOPEP_TIMING on): no-swap 3.08 ms = dispatch 0.96, combine 0.92,
  meta+check 0.45, pad+loads 0.24, route+xchg 0.20, swap_decide 0.09, act 0.09, rest 0.13; swap steps 3.45 (+0.37:
  swap_decide +0.21, push0 +0.06, route +0.05, push1 +0.04, arm_parse +0.03); swap share 51 % at 256 and 1024, 21 %
  at 4096. The nsys window was 671 of 672 swap steps at S<=256 (whole jN run: 61 %).
  (C1) The two all-gathers are DEPENDENT (planner.py:105-118): loads d[R,G] -> route kernel -> all-gather of this
  rank's routed slots + gate weights. One collective is possible only by gathering the raw top-k ids + weights once
  (same bytes as today's second gather) and routing every rank's tokens locally (the route tables are identical on
  every rank; verify the route kernel is a pure function of (d, placement, rank, ids)).
  (C2) swap_decide_kernel: p10 8 us (no swap), p25-p90 199-351 us (orbit rounds, <<<1,256>>>): a swap-step cost.
  (C3) The dispatch phase's host window holds THREE things: the dispatch host tables (dispatch_gemm.cc:832-1077), the
  dispatch issue, and the COMBINE metadata derive (plan_overlap 2 at <= 16 MiB, overlap.py:42/214-223). Its 5 pinned
  H2D = 1 dispatch arena (dg:1082, 8.7 KB) + 4 combine tables (gc:439/440/666/667); the compress_plan_* kernels,
  searchsorted and most direct_copy kernels are the combine's derive.
  (C4) LOPEP_DEVICE_META=1 keeps the D2H + sync and the host table loops (host values size every copy, put,
  index_select, out_buf slice and torch::empty); it replaces only the uploads and the numpy check. Removing the
  planning sync needs device-sized data movement.
  (C5) cudaEventQuery x12 is not a poll: PyTorch pinned-allocator event checks (per-step pinned allocations at
  gc:334/367/574/584). (C6) Peer copies: NVSHMEM lowers each intra-node put_signal on stream to a data copy + an
  8-byte signal copy (measured: dispatch 12.8 data of 2.40 MB + 24 tiny, combine 9.4 data of 2.47 MB + 12 tiny); the
  code issues 12 puts per phase (dispatch: 3 round-0 + 9 gateway forwards; combine: 9 conv + 3 intra). The 8-byte
  pageable H2D x4 per phase are likely NVSHMEM self-signals (inferred from counts).
  PLANNING KERNELS (critical path: the planning sync waits on them; replicated over all ranks' routing, 32768 pairs):
  a2av_stable_scatter_pass2 = thread 0 of each of 16 blocks walks 2048 entries serially (sort_util.cu:406-426),
  constant 163 us; a2av_meta_counts = 16 blocks, per-token global atomics, no shared-memory aggregation (p50 28, p90
  280 us). Host after the sync: ~250 us numpy demands + capacity check (overlap.py:113-130, capacity.py:29-55).
  DUPLICATION: combine re-derives cumA/offA/offR_of_A/expert_base (gc:340-362/592 vs dg:861-899); C per layer 3x
  (dg:839, gc:559, gc:1318); expert-of-copy 2x (stage1 serial prefix per block, gc:655 searchsorted);
  prepare_workspace <<<1,768>>> recomputes the splits prefix. UNUSED WORK: cumA/offR_of_A uploaded but read only by
  unreachable non-fused branches; ssc unused at NN>1; 4 piece_* memsets for the disabled pieces feature (gc:1463);
  combine publishes to and joins all 16 wire streams (20 waits + 20 records + 20 waits, gc:1475-1491/2069-2086) while
  3 carry puts; ready_event / hier_dispatch_event_ / relay_send_event_ recorded, never waited.
  GPU FIXED LATENCY: barriers x3 (dg:2710, gc:3417, gc:3453): min 14, p10 137, p50 273, p90 603 us = mostly waiting
  for the slowest rank; code reading says the only cross-rank hazard is write-after-read on destination panels by the
  next layer's puts, which one barrier per layer covers (UNPROVEN: needs a proof + randomized-payload stress).
  Blocking inter-node puts: min 58, p10 87, p50 197 us each; combine's 3 are already one per remote node on separate
  streams. prereduce (551 us) and pack spin on peers' conv signals / GEMM flags (waiting, not work); bucket_reduce
  x6.7 = 317 us real work on 8 CTAs. GPU busy 2.60 ms in the capture therefore includes spin-waits.
  OPEN: intra-node data bytes at 256 tokens/rank look large (dispatch ~31 MB, combine ~23 MB per layer per GPU):
  check exact vs capacity-sized copies; ~0.4 ms per layer-step sits outside the lopep step (bracket vs ledger).
- ROUND 9 = PLAN 6 (user 09-30: 4n / 8n x prefill / decode x 1 / 2 / 4 MiB per rank on 40 GB NODES ONLY, current
  implementation, goal = positive speedups showing correct and efficient integration). Plan file rewritten (plan 6).
  Decisions: stock graphs-on at SGLang's default (cap 160 on 40 GB -> no decode point graphed); ours repeated at 2 / 4
  MiB (and at every decode point, one server serves all three). Measurement settings: ours LOPEP_TIMING=0; SGLang
  bracket on ours + stock graphs-off; ours heap 6G (lopep.heap --headroom 1.5; 9G / 11G at 3x for the 4n / 8n decode
  calibrations, 6G either way for prefill). Existing calibrations (history half): lcbtd (decode), lcbp (prefill).
  Drivers logs/sglang/jobD6.sh, jobP6.sh; specs spec_p6_4n.txt (interactive), spec_p6_8nD / 8nP.txt (8n moved from
  debug to REGULAR QOS, same short walltimes: the other session's 5 pzd_* jobs fill the per-user debug submit limit of
  5; not touched). Chains p6n4 (59115359, 59115686), p6n8d (59115360), p6n8p (59115361, 59115722...).
  4n DECODE (40 GB, KV pin = ours' pool 143794/rank, MAXRR 1024/rank, 0 growths / 0 tracebacks on all 4 arms; step
  median ms, IQR; per-layer MoE-block bracket ms):
    run/rank   stock off       stock on (eager)   ours / ours again          best stock / ours   layer ours / stock off
    256        84.45           84.29              166.85 / 164.32            0.51x              3.25 / 1.51  (0.46x)
    512        142.97          143.32             186.91 / 188.78            0.76x              3.59 / 2.59  (0.72x)
    1024       271.43          270.67             259.97 / 260.73            1.04x (WIN)        4.90 / 5.01  (1.02x)
  Ours' repeat within 1.5 % (0.3 % at 1024). Stock on = stock off within 0.3 % (SGLang bracket costs stock nothing;
  no graphs at >= 256/rank on 40 GB). Ours at 256: 166.9 ms vs 186.5 on 80 GB with LOPEP_TIMING=1 (not like-for-like).
  4n PREFILL (40 GB, pin 50000/rank, LCB eval x4 / x8 / x16, 0 growths / 0 tracebacks on all 8 arms):
    SMAX   in tok/s ours (again) / stock      ratio          mean TTFT ms ours (again) / stock   ratio         layer ours (again) / stock
    256    16807 / 25662                      0.65x          1514 / 1234                         0.81x         3.48 / 2.35 (0.68x)
    512    27727 (27998) / 33365              0.83-0.84x     1227 (1013) / 857                   0.70-0.85x    3.57 (3.31) / 3.52 (0.99-1.06x)
    1024   40520 (39007) / 38600              1.01-1.05x     940 (916) / 1194                    1.27-1.30x    4.59 (4.50) / 6.34 (1.38-1.41x)
  (layer = SGLang bracket EXTEND SUM class n_pad <= SMAX, full-chunk steps, rank 0.) At 4 MiB the MoE layer is 1.4x
  faster but throughput only 1.01-1.05x: part of ours' step time lies outside the bracket (hypothesis: host gaps
  between layers, the round-7 host-bound finding; CUDA-event brackets do not see GPU idle between layers).
  8n DECODE (40 GB, KV pin = ours' pool 160242/rank, MAXRR 1024/rank; D1 = ours + stock off on job 59115360, D2 =
  stock on + ours again on job 59117189 (different nodes); 0 growths / 0 tracebacks on all 4 arms):
    run/rank   stock off (D1)   stock on, eager (D2)   ours D1 / ours D2    ratio D1 / D2      layer ours / stock off
    256        148.59           147.71                 213.07 / 212.07      0.70x / 0.70x      4.19 / 2.88  (0.69x)
    512        255.33           254.85                 260.51 / 260.05      0.98x / 0.98x      5.11 / 4.99  (0.98x)
    1024       482.75           481.25                 373.03 / 370.92      1.29x / 1.30x WIN  7.40 / 9.59  (1.30x)
  Ours repeats across different node sets within 0.6 %. Output tok/s per GPU at 1024: ours 2746-2761 vs stock
  2121-2128. From 4n to 8n at 1024/rank stock's step grows 271 -> 482 ms, ours' 260 -> 373 ms.
  8n PREFILL (40 GB, pin 50000/rank, eval x8 / x16 / x32, one piece per SMAX: jobs 59115361 / 59115722 / 59117102;
  0 growths / 0 tracebacks on all 8 arms):
    SMAX   in tok/s ours (again) / stock      ratio          mean TTFT ms ours (again) / stock   ratio          layer ours (again) / stock
    256    26836 / 29425                      0.91x          2010 / 1985                         0.99x          4.40 / 4.43 (1.01x)
    512    41353 (42385) / 37859              1.09-1.12x WIN 1441 (1310) / 2045                  1.42-1.56x     4.88 (5.01) / 6.49 (1.30-1.33x)
    1024   57224 (58977) / 42806              1.34-1.38x WIN 1237 (1390) / 1576                  1.13-1.27x     6.60 (6.48) / 12.24 (1.86-1.89x)
  PLAN 6 SUMMARY (stock / ours; > 1 = ours faster; all 40 GB, one binary, equal KV, 0 growths / 0 tracebacks on all
  24 arms; ~9.7 node-hours, 40 GB only):
    cell                 1 MiB           2 MiB                  4 MiB
    4n decode step       0.51x           0.76x                  1.04x (win, drift 0.3 %)
    8n decode step       0.70x           0.98x (parity)         1.29-1.30x (win)
    4n prefill tput      0.65x           0.83-0.84x             1.01-1.05x (parity; TTFT 1.27-1.30x, layer 1.38-1.41x)
    8n prefill tput      0.91x (TTFT     1.09-1.12x (win;       1.34-1.38x (win; TTFT 1.13-1.27x, layer 1.86-1.89x)
                         0.99x)          TTFT 1.42-1.56x)
  Reading: ours wins at 4 MiB everywhere and at 2 MiB prefill on 8n; the advantage grows with node count (stock's
  all-gather volume per GPU grows with ranks); 1 MiB remains a loss except near-parity 8n prefill (fixed per-layer
  cost, rounds 7-8). Stock graphs-on = stock graphs-off within 0.6 % everywhere (no graphs at >= 256/rank on 40 GB).
  16n (09-30, 40 GB, regular QOS, short pieces; plan-6 methodology). CALIBRATION recorded at 16n on stock (record-only
  pieces jobC6.sh, ~4 min each: decode = truncated history t48 x16, 3824 requests, 64 dumps; prefill = history x4, 956
  requests, 64 dumps), solved on the LOGIN node (run16.sh; no nodes held), C = 1/2 (main-perf rule), 128 redundant
  experts: calib_30b16n_lcbtd_overlap_s1 (heap at 1.5x = 8G: the hard-coded 6G would have been too small; jobD6 /
  jobP6 now compute the heap per calibration), calib_30b16n_lcbp_overlap_s1 (6G).
  16n DECODE (KV pin = ours' pool 144285/rank; D1 = ours + stock off on job 59129175, D2 = stock on + ours again on
  job 59132691, different nodes; 0 growths / 0 tracebacks on all 4 arms):
    run/rank   stock off (D1)   stock on, eager (D2)   ours D1 / ours D2    ratio D1 / D2        layer ours / stock off
    256        274.97           274.69                 315.71 / 317.16      0.87x / 0.87x        6.43 / 5.48  (0.85x)
    512        468.67           469.69                 381.56 / 381.61      1.23x / 1.23x WIN    7.78 / 9.51  (1.22x)
    1024       883.37           880.54                 529.67 / 527.71      1.67x / 1.67x WIN    10.98 / 18.06 (1.65x)
  16n PREFILL (pin 50000/rank, eval x16 / x32 / x64; jobs 59129121 / 59129660 / 59132633; 0 growths / 0 tracebacks):
    SMAX   in tok/s ours (again) / stock      ratio          mean TTFT ms ours (again) / stock   ratio          layer ours (again) / stock
    256    35755 / 31320                      1.14x WIN      3563 / 4286                         1.20x          7.05 / 8.16 (1.16x)
    512    55806 (55626) / 40504              1.37-1.38x WIN 2188 (2057) / 3610                  1.65-1.76x     7.77 (7.91) / 13.06 (1.65-1.68x)
    1024   75675 (73882) / 45156              1.64-1.68x WIN 2250 (1866) / 4250                  1.89-2.28x     10.40 (9.76) / 23.93 (2.30-2.45x)
  PLAN 6 COMPLETE (4n / 8n / 16n x prefill / decode x 1 / 2 / 4 MiB, 40 GB only; stock / ours for decode steps, ours /
  stock for prefill throughput; > 1 = ours faster):
    decode    4n 0.51 / 0.76 / 1.04    8n 0.70 / 0.98 / 1.29    16n 0.87 / 1.23 / 1.67
    prefill   4n 0.65 / 0.83 / 1.01-1.05    8n 0.91 / 1.09-1.12 / 1.34-1.38    16n 1.14 / 1.37-1.38 / 1.64-1.68
  Stock graphs-on (SGLang default, cap 160 on 40 GB) = graphs-off within 0.3-0.6 % at every node count.
  MAIN PERF vs PRODUCTION (09-30): the same layer's fixed cost shows in both (235B 4n 1 MiB: ours 2.80 ms/layer in the
  harness, 3.63 in SGLang); the difference is the baseline: main-perf baselines at 1 MiB cost 3.75-10 ms/layer
  (COMET+EPLB 3.75, COMET 4.46, NVSHMEM a2av 5.14, EPIC 6.98), SGLang's production all-gather path 2.42 ms (235B 4n
  1 MiB prefill) and 1.51 ms (30B 4n 1 MiB decode). USER + POSTDOC DECISION (09-30): run the fixed-cost reduction
  campaign (handoff 50) with the goal of >= 1.1x at 1 MiB, paper semantics kept.
