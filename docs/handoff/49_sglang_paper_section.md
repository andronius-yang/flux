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
