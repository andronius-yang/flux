# Handoff 54: plan 10 - per-layer fixed cost down to a 4-node win at 1 MiB (decode and prefill)

Log of record for plan 10 (plan text `54_plan10_text.md`, approved 2026-10-02; plan 9 text `53_plan9_text.md`, plan 9
log `53_plan9_graph_device_wire.md`). Code: lopep worktree `$PSCRATCH/workspace/andrewy/lopep_p10`, branch p10 from
p9-f 288abfe (= sglang-dev).

## Starting point (round 17 + the k17 serving capture, 10-02)
- 4n decode x stock at 256 / 512 / 1024 per rank: 0.867 / 1.123 / 1.355 (k17 99.5 ms vs stock 86.25 at 256); 8n 1.11,
  16n 1.37 at 1 MiB; 4n prefill at 1 MiB last measured on the M2 binary (0.97x).
- Goal: 4n at 1 MiB decode step and prefill (SMAX 256) input tok/s both >= 1.02x stock, placement reflecting the
  requests' expert popularity. Needs -14.9 ms per step = -0.31 ms per layer at 256.
- Per-layer chain of the k17 capture (`n_ours_k17_d256`, rank 0, median of 889 decode layers, us from the graph's
  staging kernel): head + idle 0-66; AG1 66-181; swap + routing 181-303 (swap decide 44, lane arm 3, pad rebuild 13,
  route tables 25, budget 12, route 3, vacate 21, pack send 2); AG2 303-391; planning 391-536 (tail 3, pass1 8, scan
  11, pass2 10, dispatch plan 35, demands 12, zero 6, arena 39, stage1 4, pack scan 2); remote pack-push, relay, wire,
  forward 536-1007; GEMM 1 after the last forward 1007-1098; silu / msplit / fold / invert / workspace 1098-1139;
  GEMM 2 + combine pack / pre-reduce / wire 1139-1527; bucket reduces after the last combine signal 1527-1648 (8
  blocks each, serialized); lane commit 1648-1672.
- Combine waves are already node-aligned (`a2av_msplit_tables_kernel`: wave w = rows of source nodes
  [wave_lo[w], wave_hi[w]) of every expert), so the paper's "pre-reduce each return group, transfer when ready" holds.

## Log
- 10-02: plan 10 approved; plan-9 text preserved (`53_plan9_text.md`); worktree lopep_p10 (p10 from p9-f) created.

## L6 K4 proof: the routing exchange over HAG with LOPEP_LAYER_SYNC=2 (LOPEP_HAG=2; written before the code)
- HAG (src/dwire/hag.cu) writes the PEERS' copy of the gathered routing buffer (step 1: NVLink stores into every node
  peer's `out` at this rank's offset; step 2: one put of the node block to the same local rank of each remote node;
  step 3: the receiver forwards it to its node peers) and raises epoch signals. The only reader of that buffer is
  the planner tail of the same layer-step (it copies it into `_vce_buf` / `_probs_all_buf`).
- With LOPEP_HAG=2 the loads exchange stays NCCL, and each layer-step runs the loads all-gather (AG1) before the
  routing exchange on every rank (stream / graph order; warm-up, redo and growth steps included).
- Claim: rank w's HAG writes of layer-step k+1 into rank p's buffer happen after p's planner tail of layer-step k.
  w issues HAG(k+1) only after its AG1(k+1) completed; AG1(k+1) completes on w only once every rank, p included, has
  run its AG1(k+1) kernel far enough to send its contribution; p's AG1(k+1) is stream-ordered after p's whole
  layer-step k (its graph / stream), which contains its tail of k. Steps 2 and 3 write remote buffers under the same
  condition (the sender passed AG1(k+1)).
- Epoch signals: every rank calls HAG once per layer-step in the same sequence; a rank can be at call c+1 only after
  every rank left call c (the same AG1 argument), so a waiter of call c never sees a call-(c+1) payload.
- LOPEP_HAG=1 (both exchanges) stays incompatible with LAYER_SYNC=2: its loads exchange has no NCCL collective before
  its first peer write.

## S0 instruments (lopep p10 92223d5 + fe23470, tree lopep_p10; every knob off by default)
- LOPEP_TILE_TRACE: GEMM 1 per-tile records (run, layer slot, problem, tile m / n, first / last source lane, CTA,
  SM, rows, t_enter / t_fire (gate passed) / t_done, %globaltimer) in a device ring keyed by the step's run id
  (graph-safe: no atomics, no reset; header from / stride in device memory). C.dispatch_tile_trace() /
  dispatch_tile_trace_arm().
- LOPEP_REDUCE_TRACE: the combine receivers (bucket reduce) per chain position and block: lane, bucket tokens,
  t_enter / t_waited (lane signal seen) / t_exit. C.combine_reduce_trace() / combine_reduce_trace_arm().
- LOPEP_LOAD_STATS (serving.py): per (layer slot, log2 bucket) GPU rows from the gathered routing (pad rows
  excluded): sum of max / total / node max / min, swap steps, moves, rounds; one 1-block kernel on a side branch
  joined at the end of the layer-step; pinned copy behind each forward, rank 0 writes .npy every N forwards.
- Harness: serving_check --routing-dumps (SGLang per_token recordings, decode forwards, consecutive rows per rank)
  --calib (serving calibration placements + capacities) --trace-out; serving: LOPEP_TRACE_OUT + LOPEP_TRACE_DUMP_FWD.
- calibrate.py --forward-mode decode|prefill|all; LOPEP_HAG=2 (routing exchange only over HAG, K4 proof above).
- Analyzers: `54_tile_trace.py` (tile / reduce rings), `54_placement_fidelity.py` (offline replay of a recording
  through the router's water-fill stage with the serving swap dynamics under several placements).

## P3: placement fidelity (offline; eval recording job 59214795, `logs/sglang/dumps_30b4n_lcbtdEVAL`)
- The decode calibration in use (`calib_30b4n_lcbtd_overlap_s1`) was solved from the history recording with prompt
  and generated tokens mixed (rank 0: 13384 prefill vs 11233 decode rows). Decode-only history calibration:
  `calib_30b4n_lcbtdDEC_overlap_s1` (same solver, factor 2.5, C 0.25, 2 redundant).
- Eval half, decode forwards (69 steps x 16 ranks), 9 layers, serving swap dynamics (band C 0.25, the placement
  carried across steps), mean GPU max / mean (p90), node max / mean, swap trigger rate, moves per step:
  lcbtd 1.652 (2.05) / 1.385 / 0.702 / 8.1; decode-only 1.621 (2.00) / 1.373 / 0.680 / 7.7; oracle (solved on the
  eval rows themselves) 1.620 (2.04) / 1.360 / 0.671 / 7.7; round robin 2.929 (4.0) / 1.510.
- History vs eval expert popularity correlation per layer 0.985-0.994. The serving placement captures ~98 % of the
  round-robin -> oracle reduction: it reflects the requests' popularity. The band trigger fires on ~70 % of decode
  steps under EVERY placement (also the oracle): at 256 tokens per rank the per-step popularity moves more than the
  calibration error, so the swap path (decision rounds, lane pushes, weight gates) is on most decode steps. Model
  caveat: the router's vacate pass is not modelled; the live counters (R0) check the offline numbers.
- Prefill (eval lcbp at SMAX 256, 75 steps, 5 layers; serving calibration `calib_30b4n_lcbp_overlap_s1` from 239
  history requests): lcbp 1.338 (p90 1.53) / node 1.186 / trigger 0.168; oracle 1.294 / 1.152 / 0.181; round robin
  2.117 / 1.372. Popularity correlation 0.998-0.999; ~95 % of the round-robin -> oracle reduction (weakest layer 47:
  1.50 vs 1.25). Larger steps fluctuate less: the band fires on ~17 % of prefill steps.

## S0 harness (job 59215456 "s0a1", 4n 40 GB; lopep p10 fe23470 = binary 56360e054656; `logs/p50/p10.log`, `hang_s0a_*`)
- Slurm: the first two salloc attempts of this phase died with "Connection timed out" (10:50, 10:57; no orphaned job);
  drivers now retry salloc 4 times and cancel their own pending job id on failure.
- Gates with every new knob on (tile + reduce traces, load stats, LOPEP_HAG=2): graph flips, CDMC 1, growth + forced
  abort (3 growths, 2 redos), full 1024, eager: PASS, 0 bad rows.
- Recorded-routing reference cell (LCB decode history routing, serving calibration lcbtd, 256 per rank, swap on, 16
  steps, 823 swap moves): 1 bad row on rank 1, max err 0.0125 (atol 1e-2; every rank's max 0.010-0.0125). The SAME row
  (rank 1, 0.0125) on the L1a binary without HAG: deterministic numerics (the bf16 edge of plan 9: 0.012-0.014), not a
  race; confirmation cell with LOPEP_COMBINE_WAVE_ADAPT=0 (no collapsed combine) next.
- Harness layer-step ms, recorded routing (no-move median n=6 / move median n=24; 80 % of layer-steps move experts):
  defaults 1.608 / 1.534 and 1.456 / 1.585; k17 knobs set explicitly 1.465 / 1.537 and 1.557 / 1.599 (the defaults
  equal the round-17 set); LOPEP_LOAD_STATS=1 1.467 / 1.529 and 1.523 / 1.553 (no cost); LOPEP_HAG=2 1.541 / 1.603 and
  1.509 / 1.572; 1024: defaults 3.270 / 3.414, HAG=2 3.320 / 3.553.
- **P4: the routing exchange over HAG is no faster (256) and slower (1024): L6 dropped.**
- **P1 (GEMM 1 tiles, 256, grouped wire)**: GEMM 1 span ~315 us; the last tile passes its gate ~250 us after the first
  tile entry; the tail after it = 63 us = ONE tile's compute (a tile takes ~65 us: two CTAs share an SM); 45 % of the
  tiles depend on the last wire round, never two per CTA (the schedule spread, option (a), has nothing left). 1024:
  tail 56 us, 33 % late tiles, up to 2 per CTA. Delivery-order rows would cut the late tiles to ~1/4 (about one per SM:
  ~35 us) -> L3 worth ~25-35 us at 256.
- **P2 (receivers, 256)**: own-node lanes complete 0 tokens (every token has a remote contribution); the three remote
  lanes complete ~2-17 / 20-42 / ~205-219 tokens; the last fold takes 33-43 us on 8 blocks; tail after the last lane
  signal 35-46 us. 1024: last bucket ~800 tokens, fold 176 us, tail 182 us. L5 worth ~30-38 us at 256, ~150 at 1024.
- **Grouped vs per-round wire signals (the open decision)**: NBI_GROUP=0 + FWD_FLAT=0 (per round, paper form) 1.570 /
  1.635 vs grouped 1.486 / 1.559 ms: +80 us per layer (+5.5 %) at 256; GEMM 1 span 354 vs 315 us; GEMM 1 tail unchanged.

## S1 step 1 = L1a (LOPEP_PLAN_BRANCHES; lopep p10-s1 f9bdb6e, tree lopep_p10s1, binary 7d34b8a87b82; same job)
- Gates (knob on): graph flips, growth + forced abort (degenerate steps: the arena re-run), CDMC 1, full, eager: PASS;
  recorded-routing cell: the same deterministic row as above.
- Harness layer-step ms (recorded routing; no-move / move): 256 off 1.490 / 1.564, 1.492 / 1.566; on 1.465 / 1.531,
  1.436 / 1.519 -> **-40 us per layer**; 1024 off 3.308 / 3.539, on 3.210 / 3.476 (-98 / -63 us).

## R0 (job 59214795, serving, 4n 40 GB, every arm on one allocation, KV pool pinned 110000 for every arm)
- Decode step ms at 256 / 512 / 1024 per rank: stock 86.50 / 144.10 / 273.60 and 86.50 / 145.51 / 273.45; plan-9
  defaults (lopep p10 fe23470, load stats on) 97.55 / 128.79 / 205.61 and 98.92 / 129.41 / 205.51 = **0.881x / 1.122x
  / 1.330x**. 1.02x at 256 needs a step <= 84.8 ms: -13.4 ms per step.
- Per-layer bracket at 256 (SGLang layer timing, rank 0): stock total ~1.51 ms (pre = DP all-gather 0.40, moe 0.63,
  post = reduce-scatter 0.48); ours ~1.64 ms (all in the lopep layer-step). 48 x 0.13 = 6.2 ms of the 11.7 ms step gap;
  **~5.5 ms per step is outside the MoE bracket** (stock 14.0, ours 19.5). Both arms run with the overlap scheduler off
  (DP attention), so the per-forward verdict wait is not an extra serialization. P5 (attention + whole-forward brackets,
  SGLANG_LAYER_TIMING_ATTN=1, both arms) decides where it is.
- Live load statistics (both ours runs; bucket 256 = full decode steps, ~4600 layer-steps each): GPU max / mean 1.55-1.56,
  node 1.34, swap on 50 % of the decode layer-steps, 5.2 moves per step, 1.55 orbit rounds per swap step (the offline
  water-fill model: 1.65 / 1.39 / 70 %: the router's vacate pass balances somewhat better; same conclusion).
- Prefill SMAX 256 (1 MiB), input tok/s, same allocation: stock 26681.75 / 26863.66; plan-9 defaults 31487.58 /
  30601.57 = **1.16x (worst ours / best stock 1.14x)**: the 4n prefill target is met on the plan-9 stack (first 4n
  prefill measurement since M2's 0.97x); decode at 1 MiB (0.88x) is what plan 10 still has to win.
- Serving trace arm (256, rings after forwards 40 / 80): GEMM 1 tail 59-60 us, late tiles 38-54 %; receivers' tail
  37-39 us, last bucket ~200 tokens folded in ~36 us: the harness on recorded routing reproduces serving.

## Recorded-routing row: numerics, closed (jobs 59215456, 59218660, 59220100)
- The 1 bad row (rank 1, max err 0.0125 vs atol 1e-2) is identical on: the S0 binary with HAG=2 and every instrument,
  L1a, L1a + L5 (spinning and v2), the collapsed combine disabled (LOPEP_COMBINE_WAVE_ADAPT=0), and the LEGACY path
  (eager, host-issued wire + proxy, end-of-layer sync, no graphs). Deterministic arithmetic of that input at the bf16
  tolerance edge (plan 9: 0.012-0.014), present before plan 9: not a race, not plan-10 code.

## L5 v1 (spinning wide receivers) rejected; v2 (job 59218660 data, lopep p10-s1 df23874 -> 5a20294)
- v1: the remote-lane receivers at 108 blocks, each block spinning on its lane signal from GEMM 2's end. Gates PASS,
  but harness 256: 1.559 / 1.632 vs L1a 1.508 / 1.492 ms; 1024: 5.369 / 5.409 vs 3.198 / 3.530 ms. Rings: the fold of
  the last bucket 33-36 -> 4.3 us and the tail after the last signal 35-46 -> 5 us, BUT every lane signal arrives
  50-60 us later (first remote 367 -> ~410 us, last ~450 -> 500-516 us): ~100 spinning blocks resident from GEMM 2's
  end slow the combine pack / pre-reduce / wire on every GPU. 64 blocks: 1.477 / 1.504 (about neutral).
- v2: a one-thread lane wait (device run slot, kill word) on the reduce stream, then the same fold with no wait and the
  wide grid: wide blocks exist only once their lane has landed. Gates (flips, growth + abort, CDMC 1, full, eager)
  PASS (job 59220100); A/B running.

## P5: where the step time outside the MoE bracket goes (job 59218660; SGLANG_LAYER_TIMING_ATTN=1, rank 0, 256)
- Stock: per layer attn 0.172 | pre 0.396 | moe 0.62 | post 0.47 ms = 1.67; forward 83.0 ms; step 85.90 ms.
- Ours (plan-9 defaults): per layer attn 0.230 | pre 0.014 | moe 1.65 | post 0.002 = 1.90; forward 94.2-95.1 ms; step
  99.98 ms.
- Per step: attention part +2.8 ms (48 x 58 us), MoE bracket +8.2 ms, rest of the forward ~+0.5, between forwards
  ~+2.1 ms (ours ~5.0 vs stock 2.9).
- The attention part runs the same kernels (~166 us busy per layer in both captures). Ours' k17 capture: the kernels
  were launched a median 7 ms before they ran (the host is throttled in cudaGraphLaunch, ~290 us per call, and waits
  ~14.6 ms per forward in the verdict sync), yet ~64 us per layer are gaps: 17 us from the MoE graph's last kernel to
  the next eager kernel, then 3-7 us between each of ~10 small eager kernels. Stock is launched with
  CUDA_DEVICE_MAX_CONNECTIONS unset (8), ours with 24 (env_launch.sh). P5b isolates: ours 24 / 8 connections, ours
  with an eager MoE, stock 24 / unset.
- L5 v2 A/B (job 59220100, harness, recorded routing; no-move / move median ms): 256 L1a 1.504 / 1.491, 1.526 / 1.512;
  + v2 at 64 blocks 1.431 / 1.490, 1.450 / 1.511; + v2 at 108 blocks 1.422 / 1.495, 1.388 / 1.485 (-110 us no-move on
  n = 6, -12 us on swap steps); 1024 L1a 3.209 / 3.450, 64 blocks 3.144 / 3.236, **108 blocks 3.054 / 3.141 (-155 /
  -309 us)**. Rings (256, 108 blocks): last fold 4.3 us, tail after the last lane signal 5.5 us (was 35-46), lane
  signals NOT delayed (last 431-442 us vs 444-460 without L5). On swap steps the layer ends behind the W2 lane commit
  rather than the receivers (to check in serving). L5 v2 at 108 blocks goes into R1.

## P5b: the root cause of the attention-part gap (job 59220389; 256 per rank, SGLANG_LAYER_TIMING_ATTN=1, rank 0)
- Step ms: ours (plan-9 defaults: layer graphs, CUDA_DEVICE_MAX_CONNECTIONS 24) 98.02; ours with 8 connections 97.27;
  **ours with the MoE eager (LOPEP_LAYER_GRAPH=0) 95.43**; stock 84.46; stock with 24 connections 84.59.
- Per layer attn / moe bracket / total ms: ours graphs 0.225-0.229 / 1.61-1.65 / 1.86-1.89 (forward 92.7-93.9 ms); 8
  connections 0.200-0.203 / 1.64-1.67 / 1.85-1.88 (92.0-93.3); eager MoE **0.176-0.181** / 1.65-1.66 / 1.84-1.85
  (91.2-91.8); stock 0.174-0.177 / 1.47 / 1.64 (81.6-81.7); stock 24 connections 0.175 / 1.46 / 1.64.
- **Root cause 1: the per-layer CUDA graph of the MoE step makes the eager attention kernels after it slower (+50 us
  per layer, 2.4 ms per step); without the graphs our attention part equals stock's.** The 24 connections explain about
  half (8: -26 us), the same 24 cost stock nothing. Mechanism (open): the host blocks ~290 us inside each
  cudaGraphLaunch and every small eager kernel after a graph starts 3-7 us after its predecessor instead of 1-2.
  At 256 per rank the eager MoE pays +20-30 us in its own bracket (host launches) and wins overall (one sample).
- **Root cause 2: between forwards ours spends 3.6-4.2 ms (eager) / 4.1-5.3 ms (graphs) vs stock 2.9 ms**: the
  per-forward lopep work (verdict all-reduce + D2H + host wait, counts upload, begin / end forward).
- Consequence: the default serving path (graphs on) needs a decision on 512 / 1024 data (plan 9: graphs helped
  there); 1.02x at 256 from the eager baseline needs -11.3 ms per step: the MoE levers (L1a, L5, L2, L3: ~6-9 ms) plus
  the per-forward boundary trims (~1-2 ms).

## L3 delivery-order rows (lopep p10-s2 39ddf8e, LOPEP_ROW_ORDER=1; job 59221842): correct, but not a lever
- Gates with L1a + L5 v2 (flips, growth + abort, CDMC 1, full, eager): PASS. LOPEP_DWIRE_DELAY_US=300 (new test knob,
  9baa90b) on rank 4 (gate) and rank 13 (recorded routing, reference): no new bad row (only the known deterministic
  rank-1 row); that the delay took effect is still to be shown with the wire trace (the gate cells print no timing).
- Harness (recorded routing; no-move / move ms): 256 base 1.387 / 1.464, 1.376 / 1.476; L3 1.418 / 1.451, 1.376 / 1.423
  (move -33 us, no-move neutral); 1024 base 2.983 / 3.149, L3 3.039 / 3.262 (+56 / +113 us, one sample).
- Tile rings (POS=1): GEMM 1 tail 62 us (63 without L3), late-round tiles still 45 %: with ~2 tiles per expert at 256
  one of the two holds the last node's rows in either order (P1's "~1/4" expectation was wrong; only straddles go).
  A last-round tile takes ~56 us alone on its SM and 67-68 us when the SM holds another late tile: the tail is one
  128x128 (K 2048) tile's compute. What would shrink it: narrower tiles (128x64) or split-K for GEMM 1 (a second
  CUTLASS config; ~-30 us at 256). LOPEP_ROW_ORDER stays as a knob, off.

## R1 (job 59221309, serving, one allocation, KV pinned 110000; decode step ms at 256 / 512 / 1024 per rank)
- stock 84.71 / 143.64 / 271.37 and 84.49 / 143.31 / 272.43; plan-9 defaults (lopep_p10) 98.43 / 127.26 / 204.71;
  **L1a + L5 v2 (lopep p10-s1 5a20294; LOPEP_PLAN_BRANCHES=1, LOPEP_TAIL_REDUCE_BLOCKS=108), layer graphs: 93.81 /
  122.60 / 192.76 = 0.902x / 1.170x / 1.411x stock** (-4.6 / -4.7 / -12.0 ms vs the defaults); the same with the MoE
  eager 93.73 / 125.80 / 197.87; with 8 connections 93.95 / 122.93 / 194.34.
- Graphs stay (equal at 256, better at 512 / 1024); 8 connections do not help in this round. 1.02x at 256 needs a step
  <= 82.9 ms: -10.9 ms from here.
- LOPEP_DWIRE_DELAY_US verified (job 59223227, harness 256, LOPEP_WIRE_TRACE): rank 4's dispatch puts 119.6 / 75.3 /
  107.8 us without the knob, 652.8 / 349.4 / 347.6 us with 300 us; layer-step 1.505 / 1.468 -> 2.086 / 2.137 ms. So
  the L3 delay gates ran with node 1's (rank 4) and node 3's (rank 13) windows landing late: no under-gated tile.

## P5c: the root cause of the layer-graph penalty is the host running far ahead (job 59223312; 256 per rank, ATTN timing)
- LOPEP_HOST_AHEAD=k (lopep p10-s3 2a50a04): before replaying a layer graph the host waits for the replay k layers back.
  Step ms: stock 85.83; L1a + L5 v2 unthrottled 93.92; **k = 2: 87.75 (0.978x)**; k = 8: 92.83.
- Per layer (rank 0) attn / moe bracket / total ms: stock 0.172-0.176 / ~1.49 / 1.664-1.667 (forward 82.6-83.0);
  unthrottled 0.224-0.228 / 1.54-1.56 / 1.78-1.80 (88.7-89.4); **k = 2: 0.188-0.191 / 1.46-1.48 / 1.665-1.685
  (82.9-83.6): the per-layer total equals stock's**; k = 8 as unthrottled.
- A deep launch queue (the host ~7 ms ahead, blocked in cudaGraphLaunch) slows BOTH the eager attention kernels and the
  MoE graph itself (-36 and -75 us per layer when throttled). This also explains why the eager MoE never paid the
  penalty (its host is never far ahead). The hardware / driver mechanism is not identified (GPU front-end processing
  of a deep queue, or driver polling while blocked in the launch); the throttle is the empirical fix.
- Left at 256 with k = 2: between forwards ~4.5 ms vs stock 3.0, the forward +0.5 ms.

## S2b (job 59223227): LOPEP_SAMPLE_BEFORE_CHECK has no measurable effect
- Decode ms 256 / 512 / 1024: stock 84.47 / 143.76 / 270.87; L1a + L5 (tree lopep_p10s2) 95.66 / 121.89 / 193.06;
  + LOPEP_SAMPLE_BEFORE_CHECK=1 95.51 / 122.05 / 192.03 (the boundary cost is not the sampling launch). Kept as a knob,
  off.
- The repeat arm (ns2) failed to START: a segmentation fault in torch.cuda.graph capture_begin during lopep's layer
  graph capture at warm-up (before any forward; not the knob's path). The same segfault: 3 of the last 60 ours server
  starts (plan-9 rounds ls and cf too): a pre-existing intermittent startup failure of the graph capture, open.

## The intermittent capture_begin segfault: a harness artifact (stall watchdog of a concurrent job)
- `logs/p50/stall_watch.sh` finds "the" wave client with a user-wide pgrep. With two serving jobs running (S2b and P5c,
  13:00-13:17), S2b's watchdog took P5c's client for its own, declared a stall on S2b's FINISHED arm (its server log
  idle, 13:16:05), launched its probes into S2b's allocation (an overlapping srun step) while the next arm's server was
  capturing its layer graphs (segfault in capture_begin 13:16:28), and at 13:16:54 killed the user's wave clients
  (P5c's last arm a8: its 92.83 ms is not reliable). Plan-9 round ls: the same sequence (probe 02:43:16, segfault
  02:43:32). Round cf's segfault (02:01:50) has no watchdog event: unexplained.
- Fix: `stall_watch2.sh` matches only the job's own wave client (its node-0 URL) for the stall test and the kill; every
  later driver uses it. Not a lopep defect as far as the evidence goes; recheck if a capture segfault recurs without a
  concurrent watchdog.

## S4: B1 (LOPEP_META_FUSED) and the faster dispatch plan kernel (lopep p10-s4 632c363, 9590b43, ed35f19; tree lopep_p10s4)
- B1: meta pass1 and the meta scan in one launch; every block fences its rows and arrives on a self-resetting counter,
  the last-arriving block scans (rows staged in shared memory up to 64 KB). Gates (flips, growth + abort, CDMC 1, full,
  eager) PASS (job 59224947); test_meta_device with LOPEP_META_FUSED=1: 104 cases bitwise identical. Harness (recorded
  routing, no-move / move ms): 256 base 1.490 / 1.493, 1.413 / 1.472; B1 1.373 / 1.448, 1.382 / 1.452 (**-32 us on
  swap steps**, both runs below both); 1024 base 2.998 / 3.241, B1 3.001 / 3.302 (neutral; the staging reserves ~50 KB
  of shared memory for every pass1 block there: cap it if 1024 confirms).
- Dispatch plan kernel phases (LOPEP_PLAN_TRACE, with per-phase barriers, 256 and 1024 alike): stage 2.6, zero 1.3,
  checks 12.8, water-fill + segments 13.8, tables 19.7 = 50.7 us. Tables loops ran on NN = 4 threads with O(W L) region
  sums each. Fix (ed35f19): warp-reduced row sum, the receive-offset and peer segment-base prefix tables built once in
  parallel, the water-fill's division hoisted: tables 10.5, water-fill 11.8, total 39.2 us (job 59225619).
  test_dispatch_plan_device: 90 cases, every rank's plan word-for-word equal to the reference (1-16 nodes); gates PASS.
- tests/test_r5_resources.py failed on the combine pack kernel: its table still said 1024 threads (the kernel launches
  512 since plan 9's 9669b14; 512 x 64 registers = the 32768 budget). Table fixed; rerun with the registry test --gpu.

## R2 (job 59224548, serving, one allocation, KV pinned 110000): the throttle depth; k = 1 wins
- Tree lopep_p10s3 (p10-s3 2a50a04), L1a + L5 v2 (LOPEP_DEVICE_META=1, LOPEP_PLAN_BRANCHES=1,
  LOPEP_TAIL_REDUCE_BLOCKS=108), layer graphs, no ATTN timing. Decode step median ms at 256 / 512 / 1024 per rank
  (192 intervals; zero tracebacks, zero growths in every arm):

| arm | 256 | 512 | 1024 | x stock (256 / 512 / 1024) |
|---|---|---|---|---|
| stock | 84.81 | 143.67 | 272.94 | |
| HOST_AHEAD=2 | 88.74 | 116.93 | 184.02 | 0.952 / 1.227 / 1.481 |
| **HOST_AHEAD=1** | **84.07** | **116.10** | **183.53** | **1.005 / 1.236 / 1.485** |
| HOST_AHEAD=3 | 88.32 | 119.29 | 190.64 | 0.957 / 1.202 / 1.430 |
| unthrottled | 93.70 | 122.40 | 190.47 | 0.902 / 1.171 / 1.432 |
| stock | 84.19 | 143.42 | 272.30 | |

  (x stock against the mean of the two stock runs, 84.50 / 143.55 / 272.62.)
- k = 1 (the host replays layer i's graph only after layer i-1's graph has finished; attention i is already queued
  behind it) is best at every size, and better than k = 2 by 4.7 ms at 256. **First ours arm at parity with stock at
  256** (84.07 vs 84.81 / 84.19, inside the stock spread). 1.02x needs <= 82.8 ms: -1.3 ms per step = -27 us per
  layer. B1 (-32 us per layer on swap steps in the harness) and the faster dispatch plan (-11 us) are not in this tree.
- Default for the next rounds: LOPEP_HOST_AHEAD=1. Prefill (SMAX 256) has not run with the throttle yet: R3 checks it
  (the prefill host is never far ahead, so the throttle should be neutral there).

## P5d (job 59226326): the step boundary with the throttle is +0.34 ms per step (nsys, node 0, rank 0, 256 per rank)
- Capture `logs/sglang/nsys50/n_ours_h2_d256` (lopep_p10s3, L1a + L5 v2, LOPEP_HOST_AHEAD=2) against
  `n_stock_base_d256`; analysis scripts `54_nsys_boundary.py` (windows from layer 47's rope kernel to the
  next forward's layer-0 rope; `54_nsys_boundary_seq.py`: the boundary's kernel sequence from the lm head GEMM; `54_nsys_host_gap.py`: host API
  calls between two launches). Under nsys stock's layers are inflated (2.01 ms per layer window vs ours 1.69), so only
  the boundary is compared.
- lm head start -> next layer-0 rope: stock 5748 us, ours 6084 us = **+336 us per step**:
  - +189 us: the capacity verdict all-reduce (u64, 60 us) and then 112 us of GPU idle before the sampling argmax: the
    host launches the all-reduce, blocks in cudaEventSynchronize until the verdict is in (3.37 ms on the thread), and
    only then launches the sampling;
  - +108 us: embedding -> layer-0 RMSNorm gap 358 vs 250 us (host Python between the two launches; 82 us of it in
    cudaEventQuery);
  - +71 / +44 us: scheduler batch preparation gaps (kv indices -> embedding 1325 vs 1254, clamp 354 vs 310).
- So with the throttle the outside-bracket cost is almost gone: the +1.5 ms "between forwards" from the P5c layer
  timing is mostly the timing brackets' own host waits, not GPU idle. What is left at the boundary is small
  (LOPEP_SAMPLE_BEFORE_CHECK would take back at most the 112 us idle); the remaining 1.02x gap is per-layer.

## S5: L2-lite LOPEP_ROUTE_FUSED (lopep p10-s5, tree lopep_p10s5; job 59226539, binary bf7be4bbf683)
- The router in three launches instead of six (memset, tables, budgets on G x 32 blocks, route, vacate, pack send):
  the tables kernel zeroes the ticket counters and the stats; the budgets run one block per expert with a warp per
  replica (warp-level largest remainder, the same integer arithmetic); one kernel takes each entry's ticket, runs the
  vacate pass and writes the routing exchange's send row (phys, then the gate weight bits). Tickets and vacate budgets
  stay relaxed atomics (the router's contract), so routing is as non-deterministic as before.
- tests/test_route_fused.py (437 cases, 4-64 ranks, K 8 and 16, skewed and shifted demand): release + extra per
  (expert, replica) is invariant under the vacate pass, so it is checked against the host reference on both paths
  (equal); send rows exact; every replica's global fill inside [Lb, U]; remote (token, node) pairs 1612716 fused vs
  1612595 (+0.008 %), vacate moves 108286 vs 108448. Gates with every S4 knob + ROUTE_FUSED (flip, growth + forced
  abort, CDMC 1, full, eager) PASS; R5 PASS. test_kernel_registry --gpu failed on names only (the two existing vacate
  instantiations are now `<16, false>` / `<8, false>`; the same functions were registered and preloaded); fixed in
  source, rebuilt after R3.
- Harness on recorded routing (no-move / move layer-step ms): 256 base 1.365 / 1.401, 1.402 / 1.467; fused
  1.501 / 1.430, 1.415 / 1.436; 1024 base 3.044 / 3.328, fused 3.019 / 3.340: **neutral within the harness spread**
  (three launches less inside a graph is ~10 us). R3 measures it in serving (f vs fr, two runs each).

## R3 (job 59227268, serving, one allocation, KV pinned 110000): 1.012-1.019x at 1 MiB decode, prefill 1.32x
- Arms (decode step median ms, 192 intervals; zero tracebacks and growths everywhere): h1 = R2's best (lopep_p10s3,
  LOPEP_DEVICE_META=1, LOPEP_PLAN_BRANCHES=1, LOPEP_TAIL_REDUCE_BLOCKS=108, LOPEP_HOST_AHEAD=1); f = h1's knobs on
  lopep_p10s5 (binary bf7be4bbf683: + the faster dispatch plan kernel) + LOPEP_META_FUSED=1 (B1); fr = f +
  LOPEP_ROUTE_FUSED=1 (L2-lite). Order stock, h1, f, fr, fr, f, h1, stock.

| arm | 256 | 512 | 1024 | x stock 256 (vs the stock mean 84.57) |
|---|---|---|---|---|
| stock | 84.60 / 84.53 | 143.72 / 143.67 | 272.02 / 271.27 | |
| h1 | 84.41 / 85.12 | 116.08 / 117.11 | 185.44 / 186.05 | 1.002 / 0.994 |
| f | 83.11 / 83.55 | 115.51 / 116.44 | 184.88 / 186.03 | 1.018 / 1.012 |
| fr | 83.01 / 83.56 | 114.41 / 116.41 | 184.53 / 183.83 | 1.019 / 1.012 |

- B1 + the plan kernel (f vs h1): both f runs below both h1 runs at 256 (-1.3 ms per step = -27 us per layer, matching
  the harness's -32 us on swap steps + -11 us); neutral at 512 / 1024. ROUTE_FUSED (fr vs f): neutral at 256 / 512,
  both fr runs below both f runs at 1024 (-1.3 ms). Keep both.
- **1 MiB decode is now 1.012-1.019x stock; 1.02x on every run needs the worst ours run <= 82.87 ms (84.53 / 1.02):
  -0.7 ms per step = ~15 us per layer.** 512 / 1024 stay 1.24x / 1.47x.
- Prefill SMAX 256, input tok/s: stock 27121.22 / 27041.39, fr 35781.80 = **1.32x** (R0 plan-9 stack 1.16x): the
  throttle and the per-layer levers carry to prefill.

## P5e (job 59231907): with LOPEP_HOST_AHEAD=1 the host has (almost) no slack under nsys
- lopep_p10s5 rebuilt (registry-name fix only, binary d2ad7dab7e72); test_kernel_registry --gpu PASS (98 kernels
  preloaded). Capture `logs/sglang/nsys50/n_ours_fr1_d256` (fr stack, k = 1), analysis scripts 54_nsys_graph_launch.py /
  54_nsys_launch_timeline.py / 54_nsys_host_cycle.py.
- GPU idle right before each layer graph's first kernel: k = 2 3.9 us median; **k = 1 39.5 us median (p90 80.6)**,
  ~1.9 ms per forward under nsys.
- Why (host clock, between consecutive graph launch calls, median): k = 1 1447 us = CUDA API 267 (event wait 6) +
  **non-API host time (Python) 1179 us**; with the launch call (~220 us) the host's per-layer cycle (~1.67 ms) equals
  the GPU's (~1.68 ms). At k = 1 the throttle wait returns at once and layer i's 18 eager attention kernels are still
  being launched 280 us after graph i-1 ended: under nsys the host is the bottleneck. nsys (CUPTI + Python sampling at
  2 kHz) inflates host time, and R2's k-dependence (4.7 ms between k = 1 and 2) shows the host is ahead without it, so
  the real slack is measured next without the profiler (LOPEP_HOST_STATS, R4).
- Consequence for the remaining 0.7 ms: per-layer host time is now close to critical; the fixed cost to cut is ours
  in Python between graph replays (stage-in, pads, count upload, the post-replay bookkeeping) as much as GPU time.

## R4 so far (job 59232564, tree lopep_p10s6 = p10-s5 + python knobs, binary 52e4de2f6c22)
- LOPEP_HOST_STATS (no profiler; fr arm, 256 per rank, rank 0, median per layer replay): host cycle 1560 us = ours
  before the wait 19 + **throttle wait 671 (p90 844)** + replay call 49 + ours after 23 + outside ours (SGLang
  attention layer + adapter) 797. Without nsys the host has ~670 us of slack per layer at k = 1: the k = 1 launch
  exposure seen in P5e is a profiler artifact, and our own per-layer Python is ~91 us.
- fr reproduces R3: 83.06 / 114.33 / 182.25 ms (stock 84.70 / 143.07 / 271.89).
- **New hang: LOPEP_HOST_AHEAD_MARK=gemm2 (m2).** The knob records an external event node after the activation inside
  every layer graph; the host waits for the previous graph's mark (not its end) before replaying the next one, so a
  layer graph is launched while the previous graph's GEMM 2 / combine wire / reduces are in flight. m2 served the 256
  point, then all ranks stopped at ~15:32:18 (512 point): every dumped rank (one per node) blocked in
  `_last_mark.synchronize()` (serving.py:1111), i.e. the previous graph never reached its activation end; the gate
  probe found no GEMM / pack / lane kernel resident (the stuck kernel, if any, is not one it inspects); SGLang's
  watchdog fired at 15:46:40. stall_watch2 caught it at 15:33:51 (probes + client kill: hang_probe.log,
  threads_j4nDr4_d30_km2.txt); the 1024 client on the dead server was killed by hand at 15:53:56.
- Status of the mechanism: not root-caused. The default paths are not in the hazard window as far as the launch
  timing goes: k = 1 launches graph i only after graph i-1 has completed; k = 0 / 2 / 3 launch it before graph i-1
  starts (the host is >= 1 layer ahead), and none of R2-R4's k arms or plan 9's unthrottled rounds hung. The m2-only
  ingredients are the external event record node in the graph and a launch that lands mid-combine of the previous
  graph. The mark lever is dropped (no exposed launch time to win, per HOST_STATS); a harness repro (k = 1 + mark,
  cuda-gdb `info cuda kernels` on the hang) would pin the mechanism if it is ever needed.
- **R4 final** (decode step median ms, 256 / 512 / 1024; x stock against the stock mean 84.55 at 256):

| arm | 256 | 512 | 1024 | x stock 256 |
|---|---|---|---|---|
| stock | 84.70 / 84.40 | 143.07 / 143.09 | 271.89 / 271.72 | |
| fr (R3 full stack) | 83.06 / 82.59 | 114.33 / 114.08 | 182.25 / 182.22 | 1.018 / 1.024 |
| **sb = fr + LOPEP_SAMPLE_BEFORE_CHECK=1** | **82.46 / 82.23** | 114.77 / 114.04 | 182.18 / 182.17 | **1.025 / 1.028** |
| m2 = fr + LOPEP_HOST_AHEAD_MARK=gemm2 | 85.08 / 85.04 | hung / 114.77 | - / 183.86 | 0.994 / 0.994 |

  - **sb: every sb run >= 1.02x every stock run at 256 (worst pair 84.40 / 82.46 = 1.024x) on this allocation**; both
    sb runs below both fr runs at 256 (-0.5 ms: with the host held one layer ahead, enqueueing the sampling before
    the verdict wait removes the 112 us idle P5d found; S2b without the throttle saw nothing); neutral at 512 / 1024.
  - m2: slower at 256 and the hang did not recur in its second run (intermittent); dropped.
  - The headline rule needs a second allocation: R6 = stock x2, sb x2 (decode 256 / 512 / 1024 + prefill SMAX 256).

## R5 (job 59236826): clocks are not it; the QKV GEMM slowdown is interference from our graph's tail
- nvidia-smi on node 0 every 250 ms (`logs/p50/p10/smi_r5.csv`, `54_smi_clocks.py`; per-position kernel durations: `54_nsys_attention_pos.py`, `54_nsys_attention.py`), decode-wave samples only: stock
  and fr both hold **1410 MHz SM** the whole time, memory 1215 MHz, no throttle reason set; power at 256 stock
  207-213 W, fr 188-192 W (p90 ~232-246 W); at 1024 ~220-225 W both. Decode ms: stock 84.23 / 84.25, fr 83.27 /
  83.27 (1.012x) at 256; 1024 stock 271.44 / 272.28, fr 183.29 / 184.38.
- Same kernels and grids at every attention position in both captures. Split by layer (nsys, rank 0, median us):

| kernel | stock layer 0 | stock layers 4-47 | ours layer 0 | ours layers 4-47 |
|---|---|---|---|---|
| input FusedAddRMSNorm | 3.97 | 4.58 | 4.29 | 5.15 |
| QKV GEMM (grid 40x2) | 30.94 | 32.48 | 31.01 | 35.77 |
| attention core (grid 256x1x4) | 53.25 | 53.66 | 58.91 | 59.33 |

  - QKV GEMM: equal after the step boundary's idle (31.0), +1.5 us after stock's MoE, +4.8 us after our graph:
    transient interference from our layer's tail, ~3.3 us per layer more than stock's. Leading hypothesis: dirty L2
    lines of consumed receive / staging buffers (MBs written by peers over NVLink and the NIC) are written back while
    the GEMM streams its weights. Candidate lever: `discard.global.L2` (sm_80) on consumed receive rows in the bucket
    reduce / GEMM 1 input after the last read (safe under the same cross-rank reuse ordering as the reads).
  - Attention core: +5.7 us in ours at layer 0 too, so not tail interference; the two captures are hours apart and
    not at the same decode position (KV lengths), so this needs a same-allocation stock / ours capture before it
    counts.

## R6 (job 59237998): the second allocation is 1.016-1.021x at 256, so 1.02x is not robust yet
- Decode ms (256 / 512 / 1024): stock 84.17 / 142.46 / 271.63 and 84.45 / 143.62 / 271.22; sb 82.71 / 114.00 /
  181.97 and 82.88 / 114.48 / 182.51 (x stock mean at 256: 1.019 / 1.017; worst pair 84.17 / 82.88 = 1.016x; 512 /
  1024 1.25x / 1.49x); sbls (sb + LOPEP_LOAD_STATS) 83.73 / 115.21 / 182.65 (the stats kernel costs ~1 ms per step).
- Across R4 and R6 the final stack is 1.016-1.028x at 1 MiB decode depending on the allocation (stock moves ~0.5 ms
  between allocations): ~0.5 ms per step more is needed for every pair on every allocation -> L4 (S7).
- Live load statistics, final stack (bucket 256, 4564 layer-steps): GPU max / mean 1.555, node 1.343, swap on 50.6 %
  of the layer-steps, 5.2 moves per step, 1.54 orbit rounds per swap step; R0 (plan-9 stack) 1.558 / 1.343 / 49.6 % /
  5.15 / 1.55: the router fusion leaves the placement fidelity unchanged (offline model 1.65 / 1.39).
- R6 prefill SMAX 256, input tok/s: stock 27311.03 / 27242.76, sb 35253.58 = **1.29x** (R3: 1.32x): the prefill
  target holds on a second allocation.

## 8n / 16n gates (S6 milestone, binary 52e4de2f6c22 = lopep_p10s6): one tolerance-edge element, not a defect
- 8n (job 59238203, every plan-10 knob): flip, growth + forced abort, CDMC 1, eager PASS; full FAIL with 2 bad rows of
  1.57 M (ranks 10 / 13, max err 0.0111 / 0.0124; max over ranks 0.0133).
- 16n (job 59238205): flip FAIL, 1 bad row (rank 30, max err 0.0116). Re-check on one allocation (job 59238455,
  LOPEP_CHECK_DETAIL=1, a harness-only print of each bad row's worst element): every knob 1 bad row (rank 30, step 4,
  layer 2, row 510, col 1943: y 0.02344, ref 0.01278, err 0.01066 vs tol 0.01026, the only element over in the row;
  max err over ranks 0.0136); no plan-10 knob 0 bad rows (max 0.0136); every knob again **0 bad rows** (max 0.0139).
- So the same configuration flags the element in one run and not the next: the error distribution is unchanged (max
  0.0133-0.0139 everywhere, the bf16 rounding of O(1) partial sums), and which near-zero element lands just over
  atol + rtol |ref| depends on the summation order, which the relaxed router (plan 9's as well) varies run to run.
  Same class as the 4n recorded-routing row closed above. The serving pieces follow (decision rule: max err < 0.02 in
  every run).
- 8n full gate re-check (job 59240057, LOPEP_CHECK_DETAIL=1): every knob and **no plan-10 knob flag the same two
  elements with identical values** (rank 13 step 3 layer 1 row 508 col 1917: y -0.01636 ref -0.00563 err 0.01073 tol
  0.01011; rank 10 step 15 layer 2 row 95 col 133: y -0.01172 ref -0.00125 err 0.01046 tol 0.01003; max err 0.0133):
  deterministic arithmetic of the synthetic full-mode input at the tolerance edge, present on the plan-9 paths, not
  plan-10 code.
- **16n decode** (job 59240195, regular QOS, sb = the final stack on lopep_p10s6): stock 273.76 / 470.78 ms at 256 /
  512 per rank, sb 188.45 / 246.56 = **1.453x / 1.909x** (plan 9 k17 on its own allocation: 1.367x / 1.805x): no
  regression, better.

## R7 (job 59240168): S7 (L4 part 1) wins at every size; 1 MiB decode 1.040x+ on a slow allocation
- One binary (lopep_p10s7 d588f725e2b1): b7 = the R4/R6 stack with LOPEP_FOLD_VEC8=0; s7 = + the vectorized scale
  fold (default) + LOPEP_COMBINE_PREP_SIDE=1. Decode ms (256 / 512 / 1024): stock 88.62 / 151.19 / 287.30 and 88.81
  / 151.22 / 287.08 (this allocation's stock is ~5 % slower than R4-R6's); b7 86.40 / 118.55 / 190.71 and 85.77 /
  118.29 / 190.07; **s7 85.18 / 116.97 / 186.35 and 85.15 / 117.33 / 188.36**.
- s7 vs b7: both s7 runs below both b7 runs at 256 (-0.9 ms), 512 (-1.3 ms) and 1024 (-3 ms). Headline: worst s7 /
  best stock at 256 = 88.62 / 85.18 = **1.040x** (s7 1.042x vs the stock mean); 512 / 1024 1.29x / 1.53x.
- S7a (job 59239856): R5 PASS with the msplit tables listed beside the dispatch GEMM; registry --gpu, deferred verdict
  (fold 6 cases) PASS; gates (flip, growth + forced abort, CDMC 1, full, eager) PASS with PREP_SIDE; harness 1024
  base 2.882 / 3.115 vs prep 2.780 / 3.068 ms (256 inside the harness noise).
- **8n decode** (job 59240717, sb on lopep_p10s6): stock 149.91 / 256.06 ms at 256 / 512 per rank, sb 123.44 /
  166.69 = **1.214x / 1.536x** (plan 9 defaults on their own allocation: 1.111x / 1.386x): no regression, better.
- Queued: S7 at 8n / 16n (logs/p50/p10_scale7.sh: gates with PREP_SIDE accepted on PASS or a max err < 0.02 FAIL,
  then decode stock vs s7); R8 = the S7 headline on a second 4n allocation.
- **16n prefill** SMAX 256 (job 59241122, sb on lopep_p10s6): stock 30931.04, sb 55094.14 input tok/s = **1.781x**
  (plan 9 k17: 1.611x).
- **8n prefill** SMAX 256 (job 59242114, sb on lopep_p10s6): stock 29598.86, sb 44323.16 input tok/s = **1.497x**
  (plan 9 defaults: 1.416x).
- S7 at 16n (job 59242118): flip gate with PREP_SIDE PASS (0 bad rows, max err 0.0139, 96512 swap moves).

## R8 (job 59242113): the 4n 1 MiB decode goal holds on a second allocation
- s7 (lopep_p10s7, every knob incl. PREP_SIDE): decode ms 256 / 512 / 1024: stock 88.45 / 151.06 / 284.40 and 88.66 /
  150.68 / 284.57; s7 85.02 / 116.82 / 186.51 and 85.12 / 117.44 / 186.73. **Worst s7 / best stock at 256 = 1.039x**
  (mean 1.041x); 512 1.29x, 1024 1.52x.
- **Goal met: every ours run >= 1.02x every stock run at 4n 1 MiB decode on two allocations (R7 1.040x, R8 1.039x)**;
  prefill SMAX 256 1.32x / 1.29x (R3 / R6; R8 prefill pending).
- S7 at 8n (job 59242119): flip, growth + forced abort, CDMC 1 PASS with PREP_SIDE (max err 0.0136).
- S7 at 16n (job 59242118): decode stock 272.43 / 472.16 ms, s7 187.79 / 245.50 = **1.451x / 1.923x** (S6 sb on
  another allocation 1.453x / 1.909x): no regression. S7 at 8n: gates acceptable (flip / grow / CDMC 1 / eager PASS,
  full = the deterministic tolerance-edge pair, max err 0.0133); p10_scale7.sh skipped its decode because torchrun's
  ChildFailedError traceback (exit code 1 of the FAIL report) tripped the script's traceback test: decode re-queued
  (p10_s7_8nd.sh).
- 16 MiB (4096 running per rank at 30B) needs the 80 GB nodes (KV ~480k tokens per rank with t48 contexts; handoff 49
  ran it on hbm80g 09-29: decode 1.33x, 235B prefill 2048 1.65x on the plan-5-era binary); every plan-10 round ran on
  40 GB nodes (KV pinned 110000, usage 0.95 at 1024).
- R8 prefill SMAX 256 (S7): stock 26213.04 / 26773.84, s7 35356.85 input tok/s = **1.32-1.35x**.

## Freeze: defaults branch p10-f and what was SKIPPED (user ruling 10-02 ~18:50)
- lopep p10-f (tree lopep_p10f, from p10-s7 ae6f25a; binary built 18:48): every plan-10 knob on by default
  (LOPEP_PLAN_BRANCHES, LOPEP_META_FUSED, LOPEP_TAIL_REDUCE_BLOCKS = the device SM count capped at 128, LOPEP_ROUTE_FUSED,
  LOPEP_HOST_AHEAD=1, LOPEP_SAMPLE_BEFORE_CHECK, LOPEP_COMBINE_PREP_SIDE, LOPEP_FOLD_VEC8; each env var still overrides,
  0 restores the old path); the LOPEP_HOST_AHEAD_MARK experiment removed. Uncommitted until the 4n validation passes.
- 4n validation running (logs/p50/p10_fv.sh, job 59243240): unit tests (r5, registry --gpu, route_fused, deferred
  verdict, meta, dispatch plan), gates with an EMPTY environment (flip, growth + forced abort, CDMC 1, full, eager),
  harness defaults vs the knobs set explicitly vs every knob off at 256, greedy token agreement + gsm8k (stock, stock,
  defaults; logs/sglang/tokS10.sh).
- **SKIPPED for now (user: "as long as the 4n tests are complete"); rerun these exact checks if the campaign hits an
  error at 8n / 16n / 16 MiB:**
  - 8n / 16n on the default binary: `logs/p50/p10_scalef.sh 8` and `p10_scalef.sh 16` (regular QOS; gates with an
    empty environment accepted on a check result with max err < 0.02, then decode 256 / 512 stock vs defaults). The
    same code with the knobs set explicitly already passed: S7 gates at 8n (all acceptable) and 16n (flip PASS), S7
    16n decode 1.451x / 1.923x, S6 8n / 16n decode and prefill above.
  - 16 MiB readiness smoke on 4n 80 GB nodes: `logs/p50/p10_smoke80.sh` (30B decode at 4096 running per rank, KV
    pinned 480000, t48 contexts; and 235B prefill at SMAX 2048, unpinned to read ours' pool). Not run: the hbm80g
    nodes are heavily contended (10-02 ~19:00: ~1060 pending jobs asking for ~10.7k hbm80g node-slots, 371 nodes
    allocated); the user also ruled not to jump ahead to 235B. (The driver obtained job 59243239 at 18:48:46 just as it
    was being stopped; released within a minute, before any server started.)
  - p10.log's "smoke80: d30_kf: SERVER FAILED / p235_kf_s2048: SERVER FAILED" (18:49-18:50) are NOT results: the stop
    killed the launching subshell, not the script, which then tried to serve on the already-cancelled job 59243239.

## Freeze validation (job 59243240, 4n, binary 59bb9af580b7 = lopep p10-f) and state: READY FOR THE CAMPAIGN
- Unit tests PASS: R5 resources, kernel registry --gpu, route_fused (437 cases), deferred verdict, meta (104 cases bitwise),
  dispatch plan (90 cases).
- Gates with an EMPTY environment (the defaults) PASS: flip, growth + forced abort (3 growths, 2 redos), CDMC 1, full,
  eager; all 0 bad rows.
- Harness at 256 (recorded routing, no-move / move ms): defaults 1.308 / 1.330; the same knobs set explicitly 1.351 /
  1.391 (identical configuration: harness noise); every plan-10 knob off 1.454 / 1.503 (the old path still works).
- Greedy token agreement (logs/sglang/tokS10.sh, 48 prompts, 64 new tokens, report j4nTf_report.txt): stock vs stock
  15/48 identical (31 %), mean agreeing prefix 61.6 %; **stock vs ours 27/48 (56 %), prefix 76.0 %**: within the bf16
  serving noise floor (batching / DP attention make stock itself non-deterministic). Zero tracebacks on every server.
  gsm8k did NOT run: the decode serving configuration has a 256-token context and the few-shot prompts are 789 tokens
  (every arm rejected them); an accuracy check needs a server with CTX 2048 (optional follow-up).
- Committed: lopep p10-s5 2fb844f (ROUTE_FUSED), p10-s6 c1d9149 (HOST_STATS + the MARK experiment), p10-s7 ae6f25a
  (vec8 fold + PREP_SIDE), **p10-f 3d83066 (defaults; MARK removed)**; **sglang-dev fast-forwarded to 3d83066 (local,
  not pushed)**. oss_audit.sh on lopep_p10f: only the findings handoff 53 already recorded (site paths in
  env/perlmutter_sglang.sh and the integration README, both pre-plan-9 and absent from master; history / author apply to
  the release snapshot); no new content findings from plan 10.
- Final 4n numbers (S7 = p10-f code): decode 1 / 2 / 4 MiB 1.04x / 1.29x / 1.52x (R7, R8), prefill 1 MiB 1.33x (R8);
  8n decode 1.21x / 1.54x, prefill 1.50x; 16n decode 1.45x / 1.92x, prefill 1.78x (8n rows and 8n / 16n prefill on S6).
- Open user decisions for the campaign: figure grid (handoff 49: 1 / 4 / 16 MiB, decode 30B, prefill 235B; 16 MiB and
  235B need hbm80g nodes, contended), metrics (prefill TTFT is recorded by bench_serving; decode TPOT = the step time;
  request-rate sweep or not), low-load points, whether 8n / 16n enter the figure. Skipped checks to rerun if the
  campaign errors: p10_scalef.sh 8 | 16, p10_smoke80.sh (see "Freeze" above).
