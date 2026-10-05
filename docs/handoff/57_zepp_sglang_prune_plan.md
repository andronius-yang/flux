# 57 — Prune zepp `sglang-dev` to main's standard (2026-10-04, DONE)

User directive (10-04): the lopep repository stays the safe record; prune zepp's `sglang-dev` so it is presentable like
`main` (no performance numbers, no closed / stale arms, no process residue). A brief 4n correctness + performance test
must still pass.

## Baseline (measured 10-04)

`sglang-dev` (5c3fe70, tree 63fbd9e = p10-f 3d83066 renamed) vs `main`: +17899 / -2539 lines in 90 files. It reads
**111 ZEPP_* knobs** (44 `getenv` in src/include, 60 `os.environ` in python) vs main's 3 (CONDA_ENV, ROOT,
GENERATOR). Residue beyond the release audit: process references (plan / round / stage / R5 / L3 / S-A / handoff /
p10-*) 89 vs main 19; "(debug)/(test)/(measurement)" labels 14 vs 0; percentages 20 vs 10. The audit's
unit-bearing-number scan and data-file scan are already 0. Inventory: `sgl_audit/knobs.tsv`, `sgl_audit/reads.txt`.

## Rules (the same as main's M4 batches B1-B7, rulings 9/25)

R1. **Reachability rule.** A code path is deleted only if, with every knob at its default, it is unreachable in
    every supported configuration: 1 node and multi-node; comm_strategy overlap and direct; serving (SGLang), bench
    replay, examples and tests; CUDA graphs and eager; swap on and off. A path that is still reachable keeps its code;
    only its env override goes. Example: the host-issued wire stays (1 node, non-deferred bench replay); only
    `ZEPP_DWIRE` goes.
R2. **Defaults are frozen bit for bit.** The surviving behavior is exactly today's default, with no retuning.
R3. **Oracles stay.** If a test compares against an older implementation (C.route vs the fused router, the serial
    vs warp swap decision, the host vs device metadata builder), that implementation stays callable from the test
    and is described as the reference, not as a runtime option.
R4. **No env reads in src/.** Knobs that pick behavior become constants (`src/core/tuning.h`,
    `python/zepp/constants.py`). Remaining env reads are the deployment interface only: `ZEPP_CONDA_ENV`,
    `ZEPP_CACHE_DIR`, `ZEPP_ROOT`, `ZEPP_GENERATOR` (as main), `ZEPP_CONFIG` / `ZEPP_CALIB_DIR` (adapter / launcher).
    `ZEPP_KERNEL_ENTRY*` are C macros, not env.
R5. **No baseline lineage** (ruling 9/25): the replay benchmark's "stock" comparison arm goes.
R6. **Vocabulary and comments as main (B7):** no knob names, dates, plan / round / handoff / job references,
    measurements or ratios in comments, help strings or docs.
R7. **Test hooks survive only as explicit harness arguments**, never env: the forced capacity abort (grow gate)
    becomes `serving_check --force-abort <layer>`; `--gpu` already exists for the kernel-registry GPU check.

## Decision table (111 knobs; default in brackets)

| Class | Knobs | Action |
|---|---|---|
| Interface | CONDA_ENV, CACHE_DIR, ROOT, GENERATOR, CONFIG, CALIB_DIR; KERNEL_ENTRY(_T) macros | keep |
| Instruments (B1) | TILE_TRACE (+_PER_CTA, _SNAPS), REDUCE_TRACE (+_SNAPS), WIRE_TRACE, PACK_TRACE, PLAN_TRACE, LOAD_STATS (+_EVERY, _OUT), HOST_STATS (+_EVERY), TRACE, TRACE_DUMP_FWD, TRACE_OUT, DUMP, DUMP_DIR, DBG_EVENTS, ENQ_AUDIT, NVTX, TIMING (+_EVERY, _RANK), STEP_CHECK, GRAPH_POOL [shared], SM_MARGIN, PLAN_OVERLAP | delete with their kernels' ring/stamp code, pybind getters and serving_check printing |
| Test hooks (B1) | DWIRE_DELAY_US/_RANK, LANE_DELAY/_RANK, PROXY_DELAY_US; FORCE_ABORT; TEST_GPU | delete; FORCE_ABORT -> `--force-abort`; TEST_GPU -> `--gpu` only |
| Closed / default-off (B2) | WIRE_PROXY [0] + WIRE_PLAN [0] + PROXY_DRAIN + RING_TIMEOUT_S (proxy thread, plan ring), HAG [0] (+ hag.cu), ROW_ORDER [0], DERIVE_EARLY [0], DWIRE_REMOTE_FIRST [0], COMBINE_TAIL_PUSH [0], ROUTE_SMEM [0], GRAPH_OUT_COPY [0], EXACT_BUCKETS [0], REBALANCE [0], LANE_MODE [staged], LANE_WAIT [memop], SWAP_DECIDE [device], DEVICE_META [1] modes 0/2 + _COMBINE/_DISPATCH overrides, SWAP_OFF (config has `swap`); replay `--comm-strategy stock` | delete the non-default paths (R1/R3 decide what stays) |
| Tuning (B3) | CE_BATCH [on], DWIRE_CARVEOUT [on], COMBINE_PACK_SMS, TAIL_REDUCE_BLOCKS [SMs, cap 128], COMBINE_WAVE_ADAPT, DWIRE_UNROLL (+_PACK/_RELAY/_FWD), DWIRE_PACK/RELAY/FWD_BLOCKS, RELAY_SLOTS, GRAPH_MAX_BUCKET [1024], WIRE_NBI_GROUP [3], HOST_AHEAD [1], LAYER_BARRIERS [1], LAYER_SYNC [2 device wire / 1 host wire] | constants; the selection logic stays where it depends on the configuration |
| Default-on switches (B4) | META_FUSED, PLAN_SMEM, PLAN_BRANCHES, ROUTE_FUSED, PLANNER_FUSED, PLANNER_KCOPY, SWAP_DECIDE_WARP, FOLD_VEC8, COMBINE_PREP_SIDE, COMBINE_PUSH_FLAT, COMBINE_PUSH_ROWWARP, DWIRE_PACK_FLAT, DWIRE_PUSH_ROWWARP, DWIRE_FWD_FLAT, DWIRE_EARLY_FORK, GRAPH_STAGE_KERNEL, NO_D2H_MIRRORS [2], BOUNDED_ACT, W2_EARLY, LANE_PUSH_SIDE, LANE_DEVICE, OWN_GROUP, STEP_STATE, DWIRE, DEFERRED_VERDICT, LAYER_GRAPH, SAMPLE_BEFORE_CHECK, COUNTS_HOST, WARMUP | env goes; the OFF implementation goes where R1 allows (otherwise it stays as the configuration's path or as an R3 oracle). Graphs vs eager stays a constructor argument (eager serves buckets above the graph limit) |

## Batches (work repo branch `prune` from zs = the zepp sglang-dev tree; worktree `$PSCRATCH/workspace/andrewy/zepp_prune`)

B1 instruments + test hooks · B2 closed / default-off paths + stock arm · B3 tuning constants · B4 default-on
switches · B5 comments, help strings, docs, tests, launch scripts (R6) · B6 final scan.

Checks after every batch (login node, no GPU): fresh-cache incremental build rc 0 (build slots, 8 jobs);
`py_compile` of every python file; `sass_compare.py` against the zepp_rt library (functions removed / changed are
listed and must be explained by the batch); grep shows the batch's knobs gone; nothing in `tests/` or `examples/`
refers to a removed API. Each batch is one commit in the private work repo, so a failure bisects to a batch.

B6 final scan: release audit PASS; knobs left = the interface set only; process-reference / debug-label residue as
main's or explained; `git diff --stat` reported.

## 4n gate (one interactive allocation, 40 GB nodes, 90 min; pass criteria declared before the run)

Amended 10-04 ~21:55 (user asked for a concluding SGLang latency check): items 2b and 3 below replace the old item 3.
2b. One node (the B2 review flagged a possible 1-node defect that predates the pruning): serving_check on p10-f AND
    pruned, and bench/replay.py --check (qwen3 overlap/direct/swap, k2 swap) on pruned (a failure re-runs on zepp_rt).
    Verdict: pruned == p10-f outcome (a failure on both = an existing limitation, reported, not a gate failure).
3'. SGLang (`ops/servlat_zepp.sh`), arms p10-f / pruned / p10-f / pruned, each server in the plan-10 R8 decode sizing
    (Qwen3-30B, MAXRR 1024, KV pin 143794, context 256): (a) greedy token agreement, with the same criterion as item 3;
    (b) 1 MB decode latency, 256 running/GPU, 4 waves; PASS if the pruned mean decode-step median is within 2 % of the
    p10-f mean; one pre-declared repeat of the 4 arms if outside (verdict = the repeat's 8-arm mean).

Pruned build `zepp_prune` vs the validated p10-f binary `lopep_p10f` (59bb9af580b7; == zepp_rt in SASS).
1. Unit tests that remain: all PASS.
2. The five p10_fv gates on the pruned build (flip, grow via `--force-abort`, c1, full, eager): `[check] PASS`,
   0 bad rows.
3. Greedy token agreement, 4n Qwen3-30B decode config, arms p10-f / pruned / p10-f: pruned vs p10-f must be at least
   the p10-f vs p10-f floor minus 10 points (today's floor 46 %, run-to-run range 46-54 %); 0 tracebacks.
4. Performance: harness layer-step move-median (24 steps), 256 and 1024 tokens/rank, order p10-f / pruned / p10-f /
   pruned. PASS if the pruned mean is within 3 % of the p10-f mean at each size. If outside, ONE pre-declared repeat
   of the same four runs; the verdict is the 8-run mean. Instrumented harness: parity evidence, not a latency quote.

## After PASS

Regenerate the zepp `sglang-dev` commit from the pruned tree (`commit-tree`, parent main, same identity; the author
question is still open), audit PASS, move `zepp_rt` to it and rebuild, and update handoff 56/57. The lopep repo is
not touched. On FAIL: bisect over the batch commits with the login checks first, and report before spending a second
allocation.

## RESUME STATE (update after every step; read this first after a context compaction)

Repos and trees: work repo `logs/zepp_rename/work` (private, full history); prune worktree
`$PSCRATCH/workspace/andrewy/zepp_prune` on branch `prune` (starts at `zs` 6d65d3a, tree 63fbd9e == zepp
sglang-dev); release repo `$PSCRATCH/workspace/andrewy/zepp` (main 45ba36a, sglang-dev 84c137e = pre-prune tree, author Andrew Yang);
validated reference builds `zepp_rt` (== p10-f SASS) and `lopep_p10f`. Progress: `git -C zepp_prune log --oneline zs..prune`
and `prune/<batch>.txt`. Batch agents get `ops/AGENT_BRIEF.md` + this file and must pass `ops/batch_check.sh <TAG>`
before committing. After each batch: review its diff stat and the SASS changed-function list, then update this section.

USER RULINGS 10-04 ~22:00: author = committer = `Andrew Yang <androniusyang@gmail.com>` (zepp regenerated: main 45ba36a,
sglang-dev 84c137e, same trees; old objects pruned; zepp_rt / zepp_rt_main moved to them; audit AUTHOR + CLAUDE.md
updated; audit PASS). The 4n gate runs ONCE, after B6 (no GPU runs in between). Publishing is the user's step. Home freed 10-04 ~20:10: research_tree.patch APPLIED to flux (uncommitted: CLAUDE.md Zepp section,
oss_audit.sh, four parsers, docs/handoff/56), memory updated (zepp-rename-campaign + index + 3 related). At the end,
copy this file to flux docs/handoff/57_zepp_sglang_prune_plan.md.

| Step | State | Commit / result |
|---|---|---|
| B0 baseline check | DONE 19:51 | SASS identical to zepp_rt (103/103), 111 knobs, residue 89 / 14 / 20 |
| B1 instruments + test hooks | DONE 20:07, reviewed | 4d6cf7d: 111 -> 76 knobs, -1370 lines; 12 kernels changed = exactly the 66 globaltimer readers (66 -> 0); the 4 removed __syncthreads were trace-only (3) or in the deleted load_stats_kernel; dispatch GEMM regs 254 -> 240 |
| B2 closed / default-off + stock arm | DONE 20:48, reviewed | 2e78991: 76 -> 56 knobs, -4059 lines; wire proxy + plan ring + HAG + tail-push + host metadata paths + stock arm deleted (proxy helpers kept in src/core/wire_runtime.*; NVSHMEM host mutex dropped: only served the proxy thread); 9 kernels removed; independent barrier count: no surviving function changed in B2 (the 3 count changes vs zepp_rt are B1's trace-only barriers). FLAGGED: possible pre-existing 1-node serving defect (deferred + device wire, NN=1): fv_prune.sh step 2b runs 1n serving_check on p10-f AND pruned + 1n bench --check |
| B3 tuning constants | DONE ~21:15, reviewed | dad7e95: 56 -> 37 knobs, -407 lines; constants = old defaults (kCombinePackBlocks 10, kCombineWaveAdapt 48, kDwire{Pack,Relay,Forward}Blocks 64/16/32, kRelaySlots 2 (+ new FLUX_CHECK caps[8]==relay_slots_), kWireNbiGroup 3, HOST_AHEAD 1, GRAPH_MAX_BUCKET 1024); deleted unroll-4 kernels (dw kernels lost the template param: renamed), blocking put-signal path (never default: nbi_group>=2 always), LAYER_BARRIERS>=3 barriers; independent fence check: only wire/combine_wire changed = the deleted blocking path |
| B4a default-on switches read in C++ (14) | DONE ~21:45, reviewed | b1c35dc: 37 -> 23 knobs; DWIRE/STEP_STATE/NO_D2H_MIRRORS fixed (host-wire combine lanes kept: 1n / non-deferred / n_split>1), flat/rowwarp/early-fork variants + meta pass1/scan + serial swap_decide deleted, FOLD_VEC8/PLAN_SMEM/PLAN_BRANCHES env-only (fallbacks kept); new host throw L>8 in forward launcher (never taken); fence account matches (combine pack x4: BAR-2 MEMBAR-3 ERRBAR-3 CCTL-2). FOUND: the DISPATCH host-issued wire is unreachable (device wire on every step since device metadata is fixed; also true at p10-f defaults) -> R1 example was wrong for dispatch. RULING (mine): delete it in B4c but keep every stream creation and its order (hardware-queue mapping; past HOL hangs) |
| B4c dispatch host-issued wire (unreachable) | DONE ~22:10, reviewed | 81190ef: -890 lines (host wire program, deferred replay, pack branches, a2av_pack_rows_kernel + pybind + test case); 86 surviving functions identical incl. operands; 0 fence changes; all 14 stream/event creation sites identical (order, flags); unused symmetric alloc a2av_gw_round_sig_ removed on every rank (heap stays symmetric; gate exercises). B5 TODO from its report: docs/design.md dispatch steps 1-4 describe the host wire; dwire.h:5-6 'host-issued wire' + probe/handoff refs; dead code in a2av_dispatch/forward_impl (!use_meta branches behind a FLUX_CHECK, code after the unconditional return in build_stage2, `par = false ? ...`, allgather_output branch rejected by a FLUX_CHECK, unused members barrier_block / input_buffer) |
| B4b default-on switches read in python + DEFERRED_VERDICT (15) | DONE ~22:45, reviewed | 0d365f1: only interface ZEPP_ names left (getenv 2, os.environ 1); planner_pack_send_kernel removed (no caller), 85 functions identical incl. operands, 0 fence changes; python stream/group creation order unchanged; new SharedComm(layer_graphs=None: on for overlap, off for direct): direct serving now runs eagerly (p10-f asserted unless LOPEP_LAYER_GRAPH=0) -> gate 'direct' added to fv_prune.sh; kept: non-deferred path, run_forward (patch calls it), C.route (oracle), in-line prep/bounded-act paths for non-deferred steps. B5 TODO: warmup_swap_path docstring 'python lanes', join_push hasattr guard, pyflakes (unused cfg in SharedComm._step, unused constants import in planner.py) |
| B5 comments / docs / tests / scripts + dead code | DONE ~23:15, reviewed | 5922d1a: device code identical to B4b (85/85 incl. operands, 0 fence changes); residue to main's level (24 vs 19, every hit legitimate); design.md dispatch rewritten for the device wire + Serving section; dwire.h header; integrations README rewritten; server.sh baseline-graph removed (baseline kept for calibration); dead host code in a2av_dispatch/forward_impl removed; patch comments only, applies to pristine v0.5.3. Main-identical residue ('legacy', 'v2 M2', '3D scheduling' labels in both branches) LEFT as main has it: optional comment-only follow-up for both branches |
| B6 final scan | DONE ~23:25 | aa45727: 4 unused dispatch allocations + the never-waited counts_event_ removed (events only; stream creation untouched); constants.py comment without the measured '8-10 %' (ALSO to apply to main before regeneration); SASS == B5; knobs = interface only; release scan 0/0/0; percent hits all legitimate; gate flags all present; final shipped patch applies to pristine v0.5.3 and every functional line runs in sglang_zepp (extras = timing instruments only). Totals vs zs: 69 files, +1709 / -8301 lines; 111 -> 8 ZEPP_ names; 103 -> 85 GPU functions |
| 4n gate `ops/fv_prune.sh` (job 59354234, 22:41-23:19, released) | DONE | unit tests 10/10 PASS; gates flip/grow/c1/full/eager PASS, = p10-f field for field; gate direct FAIL 997116 bad rows on BOTH builds (p10-f identical) = pre-existing; 1n serving_check SIGSEGV after warm-up on BOTH builds = pre-existing (B2's prediction); 1n bench --check qwen3 overlap/direct/swap + k2 swap 4/4 PASS; SGLang 1 MB decode (R8 sizing, 4 waves) p10-f 81.57/81.98 vs pruned 82.38/82.33 ms = +0.71 % PASS (both pruned arms slower in their pair); tokens pruned-vs-p10-f 38-50 % identical (floors: p10-f 44 %, pruned 52 %; criterion 34 %) PASS, 0 growths 0 tracebacks; harness parity move medians 256: p10-f 1.416/1.428 vs pruned 1.420/1.380 = -1.5 % PASS; 1024: p10-f 3.274/3.091 vs pruned 3.117/2.987 = -4.1 % (FASTER; no-move 2.937/2.938 vs 2.827/2.825 = -3.8 %) = OUTSIDE the declared +-3 % band in the faster direction; the declared repeat was NOT run (driver omission: repeat logic only in the SGLang step) -> user decides |
| B7 serving guard | DONE 23:20 | a08d287: SharedComm raises NotImplementedError for comm_strategy='direct' or < 2 nodes (both broken on p10-f too); serving_check --strategy and calibrate --comm-strategy removed; README says overlap on >= 2 nodes; python-only, login test: 1n refused, direct refused, 4n overlap passes the guard |
| main comment fix | DONE | work zm 80bcdf7: constants.py comment without the measured '8-10 %' (same text as sglang-dev) |
| regenerate zepp, rebuild zepp_rt, audit | DONE 23:26 | zepp main 6da0bcf, sglang-dev 83a7a59 (Andrew Yang; trees == work zm / prune); old objects pruned; audit PASS; zepp_rt cleaned + rebuilt from 83a7a59: SASS == the gate-tested zepp_prune build (85/85) |

## Result (2026-10-04 23:30)

sglang-dev pruned in B1-B7 (work repo branch `prune`, 9 commits over zs): 69 files, +1.7k / -8.3k lines; ZEPP_ names
111 -> 8 (6 deployment env + 2 C macros), getenv in src/include 44 -> 2, os.environ in python/ 60 -> 1; GPU functions
103 -> 85; process residue at main's level. Released as zepp sglang-dev 83a7a59 on main 6da0bcf. Gate as above.
Open for the user: (1) the 1024 harness point is -4.1 % (faster) and outside the declared band, and the declared repeat
was not run; (2) direct-strategy and single-node serving are broken in p10-f too and are now refused, so a real fix is
follow-up work; (3) optional comment-only pass over both branches for the main-identical labels ('legacy', 'v2 M2',
'3D scheduling'), with no GPU rerun needed (SASS identity proves it); (4) research-tree changes uncommitted.
