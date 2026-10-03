# Plan 10: per-layer fixed cost down to a 4-node win at 1 MiB (decode and prefill)

DRAFT 2026-10-02, rev 1 (adversarial code review folded in). Log of record: new research-tree handoff
`docs/handoff/54_plan10_fixed_cost.md`. Code: new lopep worktree `lopep_p10`, branch p10 from p9-f 288abfe
(= sglang-dev); never build in a tree a running job uses; plain commit messages, no numbers in lopep.
Step 0 at execution start: write the plan-9 text (this file's previous content, in the session transcript) to
`docs/handoff/53_plan9_text.md`; open handoff 54.

## Context

- Plan 9 closed at 4n decode 0.867x / 1.123x / 1.355x stock at 256 / 512 / 1024 per rank (round 17: k17 99.5 ms vs
  stock 86.25 at 256); 8n 1.11x, 16n 1.37x at 1 MiB. 4n prefill at 1 MiB was last measured on the M2 binary (0.97x);
  never on the round-17 stack.
- Goal: **4n, 1 MiB (256 tokens per rank): decode step and prefill (SMAX 256) input tok/s both >= 1.02x stock**,
  with placement that reflects the requests' expert popularity; no regression at 512 / 1024, 8n, 16n.
- Arithmetic: 1.02x at 256 = step <= 84.6 ms = -14.9 ms per step = **-0.31 ms per layer** (48 layers): MoE bracket
  1.72 -> ~1.41 ms (stock 1.54), less if part of the 4.6 ms per step outside the bracket (k17 16.9 vs stock 12.3) goes.
- Our data path already beats stock's all-gather + experts + reduce-scatter; the gap is the fixed cost around it.

### Where one layer's time goes (k17 serving capture `n_ours_k17_d256`, rank 0, median of 889 decode layers, us)

| segment | span | us | lever |
|---|---|---|---|
| graph head (step head, 3 adds, zero_, index_add_) then 44 us idle | 0-66 | 66 | (P5: matters only on the slowest rank) |
| loads all-gather AG1 incl. rank skew | 66-181 | 115 | - (skew) |
| swap decide 44, lane arm 3, pad rebuild 13, route tables 25, budget 12, route 3, vacate 21, pack send 2, gaps | 181-303 | 122 | **L2** |
| routing all-gather AG2 | 303-391 | 88 | L6 (probe-gated) |
| planner tail 3, meta pass1 8 / scan 11 / pass2 10, dispatch plan 35, demands 12, zero rows 6, arena 39, stage1 4, pack scan 2, gaps | 391-536 | 145 | **L1** |
| remote pack-push 61, relay, 3 wire rounds, gateway forward | 536-1007 | 471 | - (wire-bound; user ruling); L7 reserve |
| GEMM 1 after the last forward lands | 1007-1098 | 91 | **L3** |
| silu 8, msplit tables 8, fold scales 11, invert index 2, make workspace 6 | 1098-1139 | 41 | L4 |
| GEMM 2, combine pack, pre-reduce, combine wire | 1139-1527 | 388 | - (wire-bound; waves already node-aligned) |
| bucket reduces after the last combine signal (3 remote lanes, 8 blocks each, serialized) | 1527-1648 | 121 | **L5** |
| lane commit, end of graph | 1648-1672 | 24 | - |

Estimates per layer (code review applied): L1 -70..-90, L2 -30..-50, L3 -10..-50, L4 -10..-25, L5 -60..-80,
L6 0..-40 = **-180..-335 us = 8.6-16 ms per step = ~0.94x-1.03x**. Reaching 1.02x needs most levers near their upper
bound or the reserves: P5 (outside the bracket), L7 (relay by push). Protocol-level items (one-step-stale swap
decision, one exchange, fp8 wire) are not in this plan without explicit approval. S0 refines every estimate before code.

### What earlier plans already tried (do not repeat)
- Wins: plan-7 A1 metadata derive 5 -> 3 launches + warp scatter (-0.21..-0.27 ms per layer: launch fusion pays);
  FUSED_STAGE2 consumer build; swap decide warp-per-pair; side-stream lane pushes; PLAN_SMEM; LAYER_SYNC=2; row-warp
  pushes; flat forward; grouped nbi wire.
- Null / negative: route-global / one exchange (closed 08-29); HAG for both exchanges (E3, sync still on: AG1 is
  skew); ROUTE_SMEM; PLANNER_FUSED at 256; 40 combine pack SMs; 3 relay slots; forced collapse;
  NCCL_GRAPH_MIXING_SUPPORT=0; C1 batched copies.
- Constraints learned: K11 (arena after the demands' dead-step zeroing); R5 (kernels beside a resident GEMM fit beside
  one of its blocks); NH-1 (W2 with W1 before the dispatch GEMM); graph workspaces never freed; inside graphs every
  run id comes from the device step slot; PLAN_SMEM > 48 KB needs the opt-in (8n).

## Levers (design)

**L1 (1b): planning chain after AG2 -> 3 launches instead of ~11** (`src/dispatch/sort_util.cu`). Grids stay
parametric (W x tiles_per_src, 2048 copies per tile: 16 blocks only at 4n / 256), per-phase thread counts kept
(demands / arena loops assume 512, pass2 8 warps).
- B1 `plan_front`: planner tail + meta pass1 per tile; the last-arriving block (`last_block_arrival`,
  combine_kernels.cu:95) runs the meta scan (sps, splits, uc, expert_base).
- B2 `plan_core`, wait-free, one role per block: block 0 = dispatch plan then the demands verdict (plan -> demands
  order kept: the dead decision reads the plan's error bit) without in-place zeroing, `s_dead` published to global
  with a fence; block 1 = meta arena from the counts (it reads only sps / uc); pass2 blocks (scatter_index; also zero
  `pack_flag` / `mine_token` for B3, which they never touch); the last-arriving block runs the dead-step path after
  every reader is done: zero sps / uc / splits / plan except kSeq, recompute the arena on zero counts, zero the staged
  shared-memory copies = today's K11 result. Readers of the zeroed state (zero rows, meta stream after `_plan_ev`,
  mirrors when enabled) are stream-ordered after B2.
- B3 `plan_tail`: stage1 + zero rows in one grid; the last-arriving block runs pack scan.
- Then make the serial single-block loops warp-parallel (plan per (s,d) / (n,m), arena scans), only after B1-B3 are
  bitwise equal.
- `LOPEP_PLAN_FUSED_B=0|1|compare`: compare runs the old chain into shadow buffers against a SNAPSHOT of the verdict
  block taken before the step (verdict words are sticky: kMask / kLatch / kDem), compares every word incl. forced-abort
  steps; mismatch -> verdict error + log.

**L2 (1a): swap decision + routing between the exchanges in one block** (`swap_decide.cu`, `routing.cu`,
`lane_device.cu`).
- 512 threads, phases separated by barriers; named barriers (`bar.sync id, n`) where warps run different phases (lane
  arm on warp 0 beside pad rebuild; pad rebuild's own `__syncthreads`, lane_device.cu:187, becomes a named barrier).
  Swap decide's per-thread `ws_h` arrays (sized for 256 threads) re-indexed for 512.
- mtab [G,32,R] stays in global memory (256 KB at 4n, 1 MB at 16n); `rel` / `ext` compacted per physical slot
  (W x nlp entries) so the budget / vacate state fits shared memory; register cap 128 per thread at 512 threads:
  route tables' per-thread `[kMaxRep]` arrays move to shared memory (no local-stack reservation, routing.cu:31-37).
  The route-tables visit loop is serial across (p, jj): the gain is launches, gaps and memory traffic, not lanes.
- Semantics unchanged by default: tickets and vacate budgets keep atomic claim + rollback (now shared-memory atomics),
  so routing stays non-deterministic exactly as today. A deterministic entry-order ticket variant is NOT neutral
  (own-node replicas are visited first, so late tokens drift to remote replicas, changing uc / union / wire bytes):
  A/B knob only.
- Checks: swap decision, lane arm, pad rebuild, tables, budgets bitwise vs the current kernels; route per-(expert,
  replica) multisets before vacate equal; after vacate the invariants (budgets >= 0, fill <= U, remote-node count never
  rises, every copy routed) + the distribution of uc / union size / wire bytes over many steps equal to the current
  kernels within noise.
- `LOPEP_ROUTE_FUSED=0|1|compare`; lane-push side-stream fork stays after the arm; W2 stays with W1 (NH-1).

**L3 (3): GEMM 1 tail after the last arrival (91 us).** Delivery order at a receiver with the defaults (NBI_GROUP=3
capped at 2 relay slots, dispatch_wire.cu:759): own-node lanes (pushed long before any window), then source nodes
m+1 and m+2 together, then m+3; the L lanes of one source node come from L gateways in parallel (no order among them).
Arithmetic (~190-205 rows per expert slot = 2 tiles of 128): in rank order and in delivery order alike, about one
tile per expert depends on the last round, unless the last node's rows straddle the tile boundary (1 of 4 receivers at
4n). So row order alone is a small lever; P1 (per-tile trace) decides which of these applies, in this order:
- (a) schedule: spread the last-stage tiles so no CTA holds two of them (static round-robin `fill_problem_info`,
  workspace_util.cu:126-186), the cheapest change;
- (b) delivery-order rows: lane order by the existing closed form `shift_rank_to_order` / `revert_order_to_rank`
  (sort_util.h:265-283) in `a2av_gating_cumsum_kernel` (sort_util.cu:335); the gate maps position -> slot with
  `revert_order_to_rank`; the schedule uses positions directly (no second shift, workspace_util.cu:61,76). Safe: rows
  inside a lane are already in arbitrary order (sort_util.cu:303-308); only the gate and the schedule read the cumsum;
  the moved-last remap is keyed by stage and expert; `scatter_D` (sort_util.cu:319) restores logical order, so the
  combine and its ascending-home-node assumption are untouched; per-step GEMM 1 output hash equal between modes;
- (c) finer tiles for the last stage (64x128 config, gen_dispatch.cc:20 + selector) if the tail is tile-granular.
- GEMM 1 prep (cub scan, consumer build x2, cumsum, prepare workspace, ~75 us) needs stage1's outputs and today sits on
  the pack stream behind the own-node push (`pack_str == stream`, dispatch_gemm.cc:1487): move it to a side stream
  forked after B3 (parallel to the own push), so GEMM 1 is ready when own rows land; the pack-push -> GEMM 1 edge
  (plan-9 C1) becomes an A/B knob.
- Knobs `LOPEP_ROW_ORDER=rank|delivery`, `LOPEP_TAIL_SPREAD`.

**L4: GEMM 1 -> GEMM 2 gap (41 us).** Prerequisites first (review): msplit writes `dec`, read by make workspace,
fold scales and the tail push, so msplit stays first; msplit needs the weight-gate state that serving builds after
the activation (serving.py:1228; one-shot reset gemm_combine.cc:4123-4125) -> build it right after the swap
decision; make workspace allocates a workspace every forward and needs the activation buffer pointer -> persistent
workspace (graph-safe) before it can move to the meta stream; invert index grid capped (today up to 4096 blocks)
to fit R5 beside GEMM 1. Then: msplit + invert + make workspace in `issue_combine_meta_late` (overlap.py:278) beside
GEMM 1; fold scales into `silu_mul_bounded` with the same two roundings (bitwise), the dec==0 skip and the
kErrCombineRows check (combine_kernels.cu:1012-1017); silu then waits on `_meta_ev`. Knob `LOPEP_GEMM2_PREP_EARLY`.
Lowest priority of L1-L5.

**L5: wide tail reduce.** Confirmed in code: bucket reduces run 8 blocks x 512 (`kCombineReduceBlocks`, part of GEMM
2's sm_margin, gemm_combine.cc:4103-4106), one launch per lane in sequence on the reduce stream, every block spinning
on its lane signal. Change for the remote lanes only: same reduce stream (the bucket-prefix guarantee relies on the
sequential waits, gemm_combine.cc:2312-2314), an event wait on GEMM 2's end (recorded between `gemm_op->run` and
`combine_wire->run`, ~:4142) before the first remote-lane kernel, then the same kernel with its in-kernel wait on the
DEVICE run slot (`run_ptr`, as today :2418; never a host run id: a graph would bake the capture-time value and the
wait would pass instantly on replay) and a wide grid (64-108 blocks; GEMM 2 has ended, so nothing resident needs the
SMs). Own-node lanes unchanged. Knob `LOPEP_TAIL_REDUCE_BLOCKS`; P2 sizes it first.

**L6 (probe-gated): routing exchange only over HAG** (`LOPEP_HAG=2`, src/dwire/hag.cu). AG2 has little skew (ranks
leave AG1 within ~5 us), so its ~88 us is mostly ring latency. K4 proof under LAYER_SYNC=2 written into handoff 54
before code: HAG(k+1) writes peers' buffers only after this rank's NCCL AG1(k+1) completes, which needs every rank's
AG1(k+1) contribution, stream-ordered after that rank's layer-k reads of its AG2 output. serving.py assert relaxed for
mode 2 only. Dropped if P4 shows < 20 us.

**L7 (reserve, design only if S0 shows > 60 us from planning end to the first put): relay by push**: node peers push
their remote rows straight into the gateway's relay slot instead of packing locally (61 us) for the relay to pull.
Needs a K3 / K4 table extension (slot reuse in round 3 waits on slot-free), so it is not started without the S0 data
and a written proof.

## Deadlock and race model (new risks only; plan-9 K1-K11 rules all stay)

| risk | where | rule | gate |
|---|---|---|---|
| grid-wide wait inside a fused kernel -> partial-residency deadlock (K1) | B1-B3, L2, L5 | no inter-block spins; hand-offs only via `last_block_arrival` (fence + self-resetting counter); serial phases in one block; the only new spinners are L5's existing per-lane waits, launched after GEMM 2 ends | review, R5 test, CDMC 1 cells |
| phase memory ordering across blocks | B1-B3 | `__threadfence` before arrival; readers only after the counter; `s_dead` published to global | compare modes over thousands of steps, randomized inputs |
| dead / aborted layer (K11) | B2 | zeroing in the last block after all readers; arena recomputed on zero counts; staged smem zeroed | compare mode on `LOPEP_FORCE_ABORT` steps; growth + redo |
| compare mode perturbing sticky verdict words | L1 / L2 compare | shadow chains run on a pre-step snapshot of the verdict block | compare cells across a forced violation followed by normal layers |
| capture-time constants in a graph (a host run id baked at capture) | L5, any new wait | every wait in a graph reads the device step slot | graph replay cells with `LOPEP_DWIRE_DELAY` on the combine wire |
| counter reuse across replays / redo (K6) | every new counter | distinct per kernel, self-reset by the last arrival, never memset after allocation | growth + redo; graph replay vs eager |
| under-gated GEMM 1 tile reads rows not yet landed | L3 | gate covers every position of the tile through `revert_order_to_rank`; unit test against the writers' slots (dispatch_wire.cu 267/295/497/511) | randomized payload + `LOPEP_DWIRE_DELAY` on one remote node, separately on the own push; GEMM 1 output hash |
| co-residency changes (K1 / R5) | L3 (prep side stream, GEMM 1 earlier, 64-row tile), L4 (prep beside GEMM 1), L5 | anything launched while a GEMM is resident fits beside one of its blocks; wide reduce only after GEMM 2 ends | R5 test incl. the new kernels, CDMC 1, Nsight T7 |
| cross-stream lifetime | L4 (workspace on the meta stream) | persistent graph-safe workspace, no per-forward allocation | graph growth cells; serving with KV allocation after capture |
| cross-rank buffer reuse (K4) | L6, L7 | written proof first | delay injected between AG1 and HAG step 1 on one rank |
| swap-lane ordering (NH-1) | L2 | push fork after the arm; W2_EARLY kept | swap-flip cells (880 moves); harness on real routing (real swap rate) |
| lazy module load beside a spinner (K7) | every new kernel | kernel registry + warm-up | test_kernel_registry, Nsight T6 |
| sizes scaling with W and tokens | B grids, B2 smem, L2 smem at W=32/64 | parametric grids; smem accounting incl. static bytes, opt-in above 48 KB | 4n at 1024 per rank; 8n gates; 16n smoke before default flips |

## Verification under real load

1. **Placement reflects popularity (S0 P3, before any number is trusted).** Record expert routing on the stock server
   serving the LiveCodeBench EVAL half (decode t48 workload; the recorder path of `chain_calib_lcb.sh`; evaluation
   only, never calibration). Offline, per layer and per step: GPU load max / mean and the predicted band-trigger rate
   under (a) the serving calibration `calib_30b4n_lcbtd_overlap_s1` (history half, prompt and generated tokens mixed
   ~1:1, recorded at concurrency 4096), (b) a decode-only history calibration (same solver: factor 2.5, C 0.25,
   2 redundant), (c) the oracle placement solved on the eval routing itself, (d) round robin; plus history-vs-eval
   popularity correlation per layer. Same for prefill with `lcbp` (recorded from only 239 requests). Decision for the
   user at the S0 report: keep the calibration or re-record from decode steps of the history half (trigger and C
   unchanged, ruling); the chosen one is fixed for every ours arm in every round. Live check in every ours arm:
   `LOPEP_LOAD_STATS=1` device counters per layer (sum of max-GPU rows, sum of mean rows, swap-fired steps, moves),
   read once at shutdown, no per-step host sync; the live imbalance must match the offline prediction and is reported
   beside every latency.
2. **Harness on real routing:** `examples/serving_check.py --routing-dumps <eval recording> --calib <serving calib>`
   replays consecutive recorded steps per layer (today: synthetic only). Every harness A/B and tile trace in plan 10
   uses it, so swap rounds, replica counts, tile composition and wire sizes match serving.
3. **Serving rounds:** one 4n 40 GB allocation per round, every arm on it, fresh server per arm, order stock, base,
   candidate, candidate, base, stock; KV pool pinned equal for every arm (round 17 was not: 111k vs 144k); decode LCB
   eval t48 at 256 / 512 / 1024 per rank (2 waves, median of 192 scheduler step intervals); prefill lcbp SMAX 256
   (512 / 1024 as no-regression); `LOPEP_TIMING=0`; `stall_watch.sh`; one driver per allocation.
4. **Statistics:** a knob is a win only if both candidate runs beat both base runs by more than the same-allocation
   spread (~3 ms at 256); the headline needs every ours run >= 1.02x every stock run (decode step and prefill tok/s)
   on two separate allocations.
5. **Correctness:** unit (test_meta_device, test_dispatch_plan_device, test_deferred_verdict, test_swap_decide,
   test_lane_device, test_r5_resources, test_kernel_registry + new test_plan_fused, test_route_fused,
   test_row_order); harness cells 4n and 8n (graph flips, CDMC 1, growth + forced abort, full, eager; randomized
   payload, per-rank `LOPEP_DWIRE_DELAY`, compare modes on; ~60 s per iteration watch budget); serving: zero
   tracebacks, stall_watch clean, token agreement vs stock on the existing greedy check for the final binary.

## Stages (each: unit -> harness gates -> harness A/B on real routing -> lopep commit -> handoff 54)

- **S0 probes and baselines (~12 node-hours).** Knob-only branch: `LOPEP_TILE_TRACE` (port of flux
  `A2AVTileRecord`; per tile stage, gate pass, done; hooks at ag_scatter...h:481/563/729, static buffer + per-launch
  cursor), bucket-reduce per-lane timestamps (PACK_TRACE pattern), `LOPEP_LOAD_STATS`, harness `--routing-dumps` /
  `--calib`. P1 GEMM 1 tile trace, harness + one serving capture, NBI_GROUP 3 vs 0 (L3 option choice; data for the
  grouped-signal decision); P2 per-lane reduce timestamps (L5 size); P3 placement fidelity; P4 AG2 NCCL vs HAG in the
  graph; P5 outside-bracket attribution (clean step - 48 x bracket, step-boundary idle, all-node capture +
  `53_rank_skew.py`: is the pre-AG1 idle on the slowest rank, forward_check host block, staging, first graph
  launch); WIRE_TRACE planning-end -> first put (L7 trigger); harness check that the p9-f defaults equal the k17 knob
  set (1.485 vs 1.465 ms at the flip). R0 serving round on p9-f defaults: stock x2, base x2, decode + the first 4n
  prefill on the plan-9 stack, load stats on. **Report to the user**: refined budget, calibration decision,
  grouped-signal data, lever order.
- **S1 L1**, **S2 L2** -> serving round **R1** (base vs S1 + S2; each alone if the sum wins).
- **S3 L5 (+ L4 if prerequisites are cheap)**, **S4 L3** (option from P1), **S5 L6** (if P4 positive) -> round **R2**.
- **S6 milestone:** 4n final on two allocations (decode 256 / 512 / 1024, prefill SMAX 256 / 512 / 1024), 8n + 16n
  decode + prefill (regular QOS, 12-20 min pieces), default flips (8n gates + 16n smoke first), sglang-dev
  fast-forward (no push without asking), handoff 54 closed.
- If R2 is still short of 1.02x: report the remaining gap with P5 / L7 data and propose; no protocol change without
  approval.

Node-hours: S0 ~12, gates ~10, R1 + R2 ~14, S6 ~16: **~52**. Allocations: salloc + srun only, released at once, job
ids from our own salloc logs; logs under `$PSCRATCH/workspace/andrewy/logs/p50` and `logs/sglang`.

## Critical files

lopep: `src/dispatch/sort_util.cu` (+ `sort_util.h` order helpers), `src/dispatch/ths_op/dispatch_gemm.cc`,
`src/dispatch/workspace_util.cu`, `src/dispatch/cutlass_impls/ag_scatter_gemm_grouped_with_absmax.h`,
`src/dispatch/gen_dispatch.cc`, `src/planner/{swap_decide,routing,lane_device}.cu`, `src/combine/combine_kernels.cu`,
`src/combine/ths_op/gemm_combine.cc`, `src/combine/workspace_helper.cu`, `src/dwire/{hag,dispatch_wire}.cu`,
`python/lopep/{serving,planner,swap}.py`, `python/lopep/comm/overlap.py`, `examples/serving_check.py`, `tests/`.
Reuse: `last_block_arrival` (combine_kernels.cu:95), `arena_block_exscan` / `ct_block_exscan` / cub BlockScan,
`plan_stage_counts` / `plan_dyn_smem_optin` (sort_util.cu:576-611), `shift_rank_to_order` / `revert_order_to_rank`,
`kill_word_dev`, old-flux `A2AVTileRecord`, `bench/traces.py` sampling, `chain_calib_lcb.sh` recorder, python
`reference_route` (routing.py:228). Research tree: handoff 54, `53_serving_segments.py`, `53_rank_skew.py`, new
`54_placement_fidelity.py`.

## Open user decisions carried
- Grouped nbi signals + flat forward: keep or restore per-round (decide on P1 data).
- Calibration: keep lcbtd or a decode-only history recording (decide on P3 data).
- Push sglang-dev; S3.5 deletion; any protocol-level item.
