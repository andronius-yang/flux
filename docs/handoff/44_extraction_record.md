# 44 — Open-source extraction record (moe_ep)

Execution record of the plan approved 2026-09-25 (rulings: handoff 43 §8–§9). The new
repository lives at `$PSCRATCH/workspace/andrewy/moe_ep` (placeholder name; local git). This
file is the only process log; the new repo carries no process artifacts.

Rules in force: no baseline lineage (Flux attribution excepted); public knobs
`comm_strategy=overlap|direct`, `swap`, `router_c`; tuning constants frozen; pruning in batches
checked by quick 4n reads only; the cross-binary remeasure rule is WAIVED for this operation —
only the final binary is compared, against `figs/main_perf_v4`.

## Milestone log

| date | milestone | new-repo commit | verified |
|---|---|---|---|
| 2026-09-25 | M0 skeleton + env + traces + target | c3c35cb | `env/perlmutter.sh` sources on a login node (py 3.11, CUDA 12.4, NVSHMEM 3.2.5 module); `bench/traces.py --self-test` regenerates the six published 4n batches (qwen3/k2 × b1/b4/b16) byte-for-byte (routing sha256 match vs final3 capsules 20260925-200150/-200638); `results/expected_main_perf.csv` = 52 v4 Ours cells + plotted min + reference ceiling (min plotted baseline, numbers only) |
| 2026-09-25 | M1 copy-first build | 9afb954 | `./build.sh` (login node, 8 jobs, CUDA 12.4 pinned) builds `libmoe_ep_cuda.so` (14.4 MB) + `moe_ep._C`; import exports the 13 expected symbols; `from moe_ep import EPMoE` works |
| 2026-09-25 | M2+M3 python port + capacity/swap simplifications | 9afb954 (+gen fix) | 4n Qwen read 1: 7/9 cells within 5 %, swap cells 6-8 % faster than published, best-of-three below the ceiling everywhere (table below) |

## Trace slice provenance
`data/traces/{qwen3,k2}/{eval,pool}.txt` = `pool_cache/layer5_decode_d{64_96,32_64}.txt` of the
LCB execution pools (headers stripped; pool_sha in manifest.json). Sampler seed key kept
verbatim from `sweeps/gen_matrix.canonical_string` (params incl. `pool=decode` and the 12-hex
dataset fingerprint), so benchmark batches equal the capsules' routing files.

## M1 notes (2026-09-25, in progress)

- Copy-first (batch B0) done in the new repo: `include/flux` (26 headers), `src/{core,dispatch,combine,direct,swap,planner}`,
  generators rewritten to bf16 x sm80 x A100 only (1 register TU per op), single library
  `libmoe_ep_cuda.so` + pybind module `moe_ep._C` (module.cc / dispatch.cc / combine.cc / direct.cc /
  swap.cc / routing.cc). Compile-driven cuts (anchored script, all cuts asserted): dispatch source
  5988 -> ~4670 lines (all-gather fallback, in-kernel swap, flat fan-out, triton, profiling,
  multi-weight, dispatch-only entries, `prepare_moe_ag_scatter_args`), combine 5111 -> 4690 (triton,
  profiling, multi-input), headers trimmed to the used surface.
- **Site drift found while building**: the `cudatoolkit/12.4` module now exports hpc_sdk 26.5 /
  CUDA 13.2 on CPATH, PATH and LD_LIBRARY_PATH (nvcc itself still resolved to 12.4 through the
  pinned CUDA_HOME). cutlass and the NVSHMEM device headers do not compile against 13.2.
  `env/perlmutter.sh` now filters every `hpc_sdk/Linux_x86_64/2[5-9].*` and `/usr/local/cuda-13+`
  entry and pins CUDA 12.4 + math_libs 12.4 (torch needs cusparse.h from there). The research tree's
  fab120 env script does not filter these paths; the research build of 9/16-9/25 predates the drift.
- Python package written (config, constants, placement, routing incl. capacities, planner, swap,
  comm/{overlap,direct}, layer, heap) + bench (replay.py, launch.sh, traces.py). Capacities are
  computed from the routing's own provable bounds (simplification 1 of the plan, applied at M2
  rather than M3 so no LocCap code is ever ported); the swap decision runs the band test first
  (simplification 2). Constants that the research driver read from the environment are frozen in
  `python/moe_ep/constants.py`; the fused ops still read their tuning/capacity values from the
  environment (exported by `comm/overlap.py`) until batch B5.

## M2 smoke (2026-09-25 23:26-23:30, 1 node, job 58890762, qwen3 b1, 2 warmup + 3 timed, --check)

| strategy | swap | correctness (4 ranks) | total_ms iter_max_median | note |
|---|---|---|---|---|
| overlap | off | PASS 0/128 bad rows each | 1.899 | first run of the fresh repo end to end |
| overlap | on | PASS | 2.105 (means: plan_comm 1.25, place 0.61, plan 2.05) | band swaps still executing in the timed window (only 2 warmup); 4n read uses 5+10 |
| direct | off | -- | -- | `No registered hparams found for GemmMeta(... c=Void ...)`: the bias-free GEMM uses the void-C dtype variant, dropped by the pruned generator; restored (gen_gemm.cc), rebuilt |

Environment fixes on the way (all in `env/perlmutter.sh`): CUDA 12.4 pin + path filter (13.2 drift),
`math_libs/12.4` include (cusparse.h for torch headers), `nccl/2.24.3` module (Slingshot NCCL
plugin: without it torch's NCCL fails with "network AWS Libfabric not found").

## M2 + M3 read 1 (2026-09-25 23:32-23:37, 4 nodes, job 58890820, qwen3, 5 warmup + 10 timed, isolated)

New-repo build 9afb954 + generator fix; capacities from the routing bounds (no LocCap floor), swap
decision band-first. `results/compare.py` vs `figs/main_perf_v4`:

| MiB | strategy | swap | new repo | published | delta |
|---|---|---|---|---|---|
| 1 | overlap | off | 2.776 | 2.817 | -1.5 % |
| 1 | direct | off | 4.721 | 4.660 | +1.3 % |
| 1 | overlap | on | 3.091 | 3.362 | **-8.1 %** |
| 4 | overlap | off | 4.475 | 4.366 | +2.5 % |
| 4 | direct | off | 10.168 | 10.139 | +0.3 % |
| 4 | overlap | on | 4.582 | 4.893 | **-6.4 %** |
| 16 | overlap | off | 11.629 | 11.401 | +2.0 % |
| 16 | direct | off | 32.831 | 32.404 | +1.3 % |
| 16 | overlap | on | 11.948 | 12.072 | -1.0 % |

7/9 within the 5 % band; the two outside are the swap strategy being FASTER (band test first,
orbit only for out-of-band nodes: place bracket 0.23 ms vs ~0.5-0.7 in the research arm). Best of
three below the reference ceiling at every budget (2.776/4.475/11.629 vs 4.457/5.932/15.018).
Verdict: M2 and M3 pass on Qwen 4n; the port is faithful, the two simplifications are neutral or
better. K2 joins at M6 (full 4n grid on the final binary).

## M4 read 2 (2026-09-25 23:52-23:57, 4n, job 58891082, qwen3): batches B1 (env knobs frozen) + B2 (debug/trace removed)

Build 3b7af6e. 8/9 within 5 % of published (swap b1 -11 %, faster); every cell within ~3 % of read 1
-> no step change. Dispatch source 4670 -> 4094 lines, combine 4690 -> 4337. Env reads left in
src/: the 7 capacity values (B5).
| MiB | overlap | direct | swap |
|---|---|---|---|
| 1 | 2.733 | 4.651 | 2.993 |
| 4 | 4.385 | 10.252 | 4.670 |
| 16 | 11.764 | 33.025 | 12.080 |

## M4 read 3 (2026-09-26 00:06-00:11, 4n, job 58891348, qwen3): batch B3 (dead-branch deletion + ctor flags)

Build d1b16a2. Dispatch 4094 -> 3104 lines, combine 4337 -> 3883. All cells match read 2 except the
FIRST cell (b1 overlap 3.419, +25 %, with GELU/l0/l1 all inflated); a dedicated re-run of that cell
on the same build (job 58891438, two repetitions) gave 2.797 and 2.843 -> one-off, no step change.
| MiB | overlap | direct | swap |
|---|---|---|---|
| 1 | 3.419 (re-run 2.797 / 2.843) | 4.701 | 3.025 |
| 4 | 4.386 | 10.229 | 4.621 |
| 16 | 11.414 | 32.313 | 11.785 |
B3 tooling: `scratchpad/b3_simplify.py` constant-folds flags line by line (declarations, ctor
initializers and parameter lines protected; string literals were NOT protected — messages that got
a literal substituted were reworded by hand), deletes `if (false)` blocks (else bodies become bare
scopes), unwraps `if (true)`, folds `FLUX_CHECK(true)`. Lesson: never pass a flag that shares its
name with a method (`a2av_dispatch` = ctor flag AND the dispatch method) — it deleted the method.

## M4 read 4 (2026-09-26 00:20-00:26, 4n, jobs 58891525 + rerun, qwen3): batch B4 (legacy Flux helpers)

Build e6ff180. Dispatch 3104 lines, combine ~3745, a2av_combine.cu 1445 -> 956, sort_util.cu 1426 -> 455;
src/swap, topk_gather_rs{,_v2}.cu, system_barrier.hpp removed. 8/9 within 5 % (swap b1 -9 %, faster);
every cell within ~2 % of read 3 -> no step change. The b4 swap cell was re-run in a second
allocation because I edited the Python tree (B5) while the read was still importing it (operator
error: the read uses the working tree; never edit `python/` during a read).
| MiB | overlap | direct | swap |
|---|---|---|---|
| 1 | 2.772 | 4.849 | 3.053 |
| 4 | 4.431 | 10.217 | 4.609 (rerun) |
| 16 | 11.482 | 32.661 | 11.844 |
Best of three below the reference ceiling at every budget (2.772 / 4.431 / 11.482 vs 4.457 / 5.932 / 15.018).

B5 committed c1eaf6c after read 4: capacities are `DispatchOptions{max_recv_rows, max_stage_rows,
max_relay_rows}` / `CombineOptions{max_send_rows, max_conv_rows, max_wire_rows}` constructor
arguments (pybind kwargs), computed by `moe_ep.routing.compute_capacities`; `_export_op_env` and
`constants.OP_ENV` deleted; `grep -rn get_int_from_env src/` = 1 hit (op_registry_proto_utils RANK
print gate, goes with B6). The combine's gateway (non-compress) path is dead code (compress is on
whenever nnodes > 1) but its ~45 references are still compiled; deleting it is folded into B6.

## M4 read 5 (2026-09-26 00:28-00:37, 4n, jobs 58891589 + rerun, qwen3): batch B5 (capacities -> constructor options)

Build c1eaf6c. Pure plumbing (same capacity numbers through `DispatchOptions` / `CombineOptions`
instead of the environment), yet the overlap-off arm read +5-6 % over published at b4/b16 in the
main pass (4.638 / 12.068) with direct and swap unchanged; a two-repetition re-run of those two
cells on the same build gave b4 4.391 / 4.536 and b16 11.892 / 11.626 (final compare 8/9 within 5 %). So
the B5 binary sits ~2-4 % above read 4 on the overlap arm at b4/b16 — inside twin noise, no
correctness change, not a step; flagged here so the final-binary read can tell drift from noise.
| MiB | overlap | direct | swap |
|---|---|---|---|
| 1 | 2.789 | 4.730 | 3.157 |
| 4 | 4.638 (re-run 4.391 / 4.536) | 10.251 | 4.710 |
| 16 | 12.068 (re-run 11.892 / 11.626) | 32.779 | 12.152 |
Best of three below the reference ceiling at every budget.

## M4 read 6 (2026-09-26 00:43-00:48, 4n, job 58891774, qwen3): batch B6 (kernel-argument leftovers + combine gateway path)

Builds 59aacb4 (B6a: dynamic tile claimer + bucket workspace, progress slots / tile trace, ballot
segment gate, prefetch-last remap, rot-align removed from the dispatch kernel Params — the one
kernel-visible batch) and 298097b (B6b: the combine's dead destination-side gateway path; last
environment read in src/ gone). 8/9 within 5 % (swap b1 -10 %, faster); the overlap arm is back
on read 4's numbers (b4 4.404, b16 11.713), so read 5's +2-4 % was noise. No step from the smem /
Params layout change. Dispatch kernel header 1000 -> 732 lines, workspace_util.cu 496 -> 341,
dispatch .cc 3104 -> 3035, combine .cc 3745 -> 3574.
| MiB | overlap | direct | swap |
|---|---|---|---|
| 1 | 2.773 | 4.622 | 3.010 |
| 4 | 4.404 | 9.798 | 4.758 |
| 16 | 11.713 | 31.278 | 11.878 |
Best of three below the reference ceiling at every budget. Next: B7 (rename + purge) -> read 7 = FINAL binary.

## M4/M5 batch B7 (2026-09-26 00:50-01:05): rename + purge, three commits

- 6a2011f B7a/B7b: public identifiers by mechanism — `DispatchGemmOp` / `GemmCombineOp` /
  `CombineWire` (the combine's wire class), files `src/dispatch/ths_op/dispatch_gemm.{cc,h}`,
  `src/dispatch/dispatch_gemm_kernel.hpp`, `src/combine/ths_op/gemm_combine.{cc,h}`,
  `src/combine/gemm_combine_kernel.hpp`, `src/combine/combine_kernels.cu`,
  `include/flux/args/{dispatch_gemm,gemm_combine}.h`; entry methods `forward()` on both ops,
  `combine()` inside the wire class; forward kwargs lose the `a2av_` prefix (`unique_counts`,
  `pack_index`, `reduce_index`, `wire_csr`, `reduce_csr`); router `route()` /
  `route_workspace_ints()` with kernels `route_{tables,budget,vacate}_kernel` / `route_kernel`
  (the un-covered variant deleted); Python handles `dispatch_op` / `combine_op`; bench columns
  `dispatch_ms` / `combine_ms`. Ban grep (`epic|eplb|moonep|ultraep|\bfast\b|comet|loccap|
  placelambda|slipstream|pv2|pv3|dual3|\bs1\b|\bs2\b|\beps\b` over src python bench scripts
  include) = 0 hits; loop variables `s1`/`s2` renamed so the gate is literal.
  Internal identifiers keep their names per the public-only ruling: `a2av_*` buffers/kernels,
  `A2AVStage1Arguments`, `GemmGroupedV2AGScatter_Kernel` (Flux kernel-builder pattern),
  `relay_*`, `lb_minmove_`.
- B7c: every comment that named a removed knob (`FLUX_A2AV_*`), a date, a capsule id, a
  handoff, a SCHEMA rule, a "Tier B" / "gen-N" / "M4-Cn" campaign label or a user decision was
  rewritten to describe the shipped mechanism (dispatch wire narrative, minimal-move relay,
  per-round staging, P2P pulls, wave-pack, blocking-wire rule, fused pack, wave collapse, wire
  lanes, bucketed receiver). Dead members those comments described went with them: flat
  fan-out tables, fan-out streams/events, in-kernel swap scratch + timing pool, NVTX window
  sidecar, forward-count/forward-index/pack-overlap events; `flux_rs_blocking_wire()` inlined.
  `grep -rnE 'FLUX_A2AV_|20[0-9]{2}-[01][0-9]-|capsule|handoff|Tier B|gen-[0-9]'` = 0 in src/
  include/ python/ bench/. The word "legacy" remains where it names Flux's dense fallback paths.
- Sizes after B7: dispatch .cc 2765 lines (research tree: 5988), combine .cc 3499 (5111),
  combine_kernels.cu 954 (1445), dispatch kernel header 730 (1000), sort_util.cu 453 (1426),
  workspace_util.cu 340, routing.cu 549; src/+include/ 23.5k lines total, python+bench 2.3k;
  repo 3.0 MB excluding 3rdparty/build/.so. Build B7 = FINAL binary -> read 7.

## M5 read 7 = FINAL binary (2026-09-26 01:01-01:06, 4n, job 58891949, qwen3)

Build 98a867f (B7a/b/c) + docs 81e8ac0 + b5559e3. 8/9 within 5 %; the ninth is the swap arm
10.8 % FASTER than published (band-first decision, as in every read). Best of three below the
reference ceiling at every budget (2.781 / 4.475 / 11.624 vs 4.457 / 5.932 / 15.018).
| MiB | overlap | direct | swap | published (overlap / direct / swap) |
|---|---|---|---|---|
| 1 | 2.781 | 4.716 | 3.000 | 2.817 / 4.660 / 3.362 |
| 4 | 4.475 | 10.127 | 4.861 | 4.366 / 10.139 / 4.893 |
| 16 | 11.624 | 32.273 | 11.867 | 11.401 / 32.404 / 12.072 |
Seven reads across the batches, every cell of every read within twin noise of its predecessor;
the pruning (5988 -> 2765 dispatch lines, 5111 -> 3499 combine, 106 env knobs -> 0, 7 baseline
drivers -> 0) changed nothing the figure can see. M6 (full 4n grid incl. K2) and M7 (8n/16n,
regular QOS) launched on this binary via `scripts/reproduce.sh`; 1-node `--check 1` + layer demo
via `logs/moe_ep/final1n.sh`.

## SGLang integration scoping (M8 note, against the `EPMoE` surface)

Target experiment (handoff 43 §7): Perlmutter, Qwen3-235B-A22B only, TTFT/TPOT with SGLang's EP
MoE layer replaced by `moe_ep.EPMoE`. What the layer already offers and what the integration must
add:

| need | status in `moe_ep` | integration work |
|---|---|---|
| construction per layer | `EPMoE(cfg, group, sizing_routing, pool_routing)` is collective and allocates symmetric buffers from provable routing bounds | one instance per MoE layer (58 for Qwen3-235B) or one shared wire with per-layer weights — the symmetric heap (6-16 GiB per rank) cannot be paid 58x, so the ops must be shared across layers and `load_weights` becomes per-layer weight *slots*; `sizing_routing` from a calibration prompt set |
| weights | `load_weights(w1_of, w2_of)` fills `[ffn, H]` / `[H, ffn]` per expert | map SGLang's fused `w13` (gate+up) layout: the dispatch GEMM computes one `[*, ffn]` intermediate followed by GELU; Qwen3 uses SwiGLU (`silu(x W_gate) * (x W_up)`), so either the dispatch GEMM grows to `2*ffn` with a SwiGLU activation between the ops (cheap: `act` is 0.1 ms) or the kernels gain a gated epilogue — decide first |
| routing in | `prepare(topk_ids, topk_weights)` per step | SGLang's router gives exactly these; loads exchange is the one collective and must run before the swap decision |
| one step ahead | `prepare` and `forward` are separate; the plan is a plain object | for decode TPOT at tiny batches, run `prepare(step t+1)` on a side stream while `forward(step t)` runs — the planner's graphs are already captured; only the D2H of the loads for the swap decision is a host sync |
| CUDA graphs | `prime()` captures the plan and scale graphs; forward is eager with host-issued NVSHMEM puts | SGLang's decode CUDA graph must treat the layer as a graph break (eager fallback), or the wire issue moves into a captured region — the host-issued blocking `putmem_signal` cannot be captured; measure the graph-break cost first |
| process groups | the layer uses one `torch.distributed` group for its collectives and NVSHMEM for the wire | NVSHMEM bootstrap = UID over the same ranks as SGLang's EP group; `bench/launch.sh` env (heap size, `CUDA_DEVICE_MAX_CONNECTIONS=24`, libfabric/CXI) must be set before SGLang initialises CUDA |
| shapes | `max_tokens_per_rank` sizes everything; Qwen3 1 MiB = 128 tokens per rank | decode batches are far below 1 MiB; prefill is far above (TTFT) — size for the prefill chunk and accept the idle buffer, or build two configurations |
| memory | Qwen3-235B bf16 = ~470 GB of experts over 16 nodes x 4 x 40 GB = 2.5 TB; with 2 redundant slots per rank (+25 %) | fits at 16 nodes; the hbm80g pool relaxes KV-cache pressure |

The first integration milestone is a single-layer harness inside SGLang's process (no model):
construct `EPMoE` from the running EP group, run `prepare/forward` on the router's real
`topk_ids`, and compare against SGLang's own MoE output — the same check `bench/replay.py
--check 1` does against the torch reference.

## M6 first pass + two single-node defects (2026-09-26 01:06-01:20)

**4n grid on 98a867f (pre-fix binary, job 58892081, `scripts/reproduce.sh --nodes 4`):** 13/18
within 5 %, best of three below the reference ceiling in all six (model, budget) cells. The five
"out" cells: four are the swap arm 5-9 % FASTER than published (band-first decision), one is
qwen3 b4 overlap at +5.6 % (4.609 vs 4.366; the same cell read 4.404-4.475 in reads 6/7 — noise
band). K2 is new here and lands +2.0 / -0.9 / +1.7 % on the overlap arm.
| model | MiB | overlap | direct | swap | published (overlap / direct / swap) |
|---|---|---|---|---|---|
| qwen3 | 1 | 2.774 | 4.741 | 3.058 | 2.817 / 4.660 / 3.362 |
| qwen3 | 4 | 4.609 | 10.105 | 4.610 | 4.366 / 10.139 / 4.893 |
| qwen3 | 16 | 11.612 | 32.632 | 12.004 | 11.401 / 32.404 / 12.072 |
| k2 | 1 | 3.872 | 5.994 | 4.063 | 3.797 / 6.059 / 4.369 |
| k2 | 4 | 5.715 | 11.735 | 6.106 | 5.765 / 11.614 / 6.431 |
| k2 | 16 | 14.102 | 33.953 | 14.277 | 13.864 / 33.859 / 14.911 |

**Single-node checks (`final1n.sh`, job 58892051) found two defects the 4n reads could not see:**
1. `--check 1` overlap/swap on ONE node tripped `gemm_combine.cc: Check failed: a2av_compress_`.
   Cause: B3 folded the receiver selection to the bucketed receiver (`a2av_bucket_ = true`), but
   in the research tree the bucket receiver was `compress && nnodes > 1 && ...`, and the
   single-node path used the per-split wait-all top-k reduce that B3/B4 then deleted as dead.
   Fix 66efe95: `CombineReduceArguments` + `a2av_combine_reduce` kernel restored (35 lines),
   receiver = `if (!compress) wait-all reduce else bucketed`. Multi-node path unchanged.
   The direct strategy passed on both models (bad rows 0/128, 0/72).
2. `examples/layer_demo.py` failed at construction: NVSHMEM was never initialised (the bench
   called `C.init_shm` itself). Fix fa07db0: `EPMoE` calls `moe_ep._ext.ensure_shm(group)`
   (idempotent, once per process); the bench no longer initialises NVSHMEM.
Lesson for the record: the per-batch 4n perf read (ruling 3) gates performance, not the
single-node configuration; a 1-node `--check` belongs in the final gate. Final binary v2 =
fa07db0 build; `final1n.sh` + the 4n grid re-run on it; 8n/16n (held while rebuilding) run on it.

## M6 DONE on the final binary v2 (2026-09-26 01:20-01:36, 4n job 58892326; 1n job 58892324)

Binary = fa07db0 build (98a867f + single-node receiver 66efe95 + layer-owned NVSHMEM init).
**1-node `--check 1`: 6/6 PASS** (qwen3 + k2 x overlap / direct / swap, bad rows 0/128 and
0/72). **4n grid: 16/18 within 5 %**, the two "out" cells are the swap arm 7-11 % FASTER than
published (qwen3 b1 3.000 vs 3.362, b4 4.567 vs 4.893); best of three below the reference
ceiling in all six (model, budget) cells. Copied into the repo as
`results/measured/main_perf_4n.csv` + `compare_4n.txt`.
| model | MiB | overlap | direct | swap | published (overlap / direct / swap) |
|---|---|---|---|---|---|
| qwen3 | 1 | 2.768 | 4.686 | 3.000 | 2.817 / 4.660 / 3.362 |
| qwen3 | 4 | 4.424 | 10.149 | 4.567 | 4.366 / 10.139 / 4.893 |
| qwen3 | 16 | 11.806 | 32.409 | 11.756 | 11.401 / 32.404 / 12.072 |
| k2 | 1 | 3.818 | 6.004 | 4.454 | 3.797 / 6.059 / 4.369 |
| k2 | 4 | 5.776 | 11.838 | 6.402 | 5.765 / 11.614 / 6.431 |
| k2 | 16 | 14.134 | 34.075 | 14.177 | 13.864 / 33.859 / 14.911 |
Third small defect from the layer demo: `load_weights` asked the caller for expert -1 on an
unassigned redundant slot (the bench's synthetic weight generator happened to accept it). Fix
3719c05: unassigned slots stay zero in both `EPMoE.load_weights` and `WeightSlots.fill`
(python-only; the bench path is unchanged). Demo re-run in flight; 8n/16n grids queued
(regular QOS, jobs 58892055 / 58892056).

## Layer demo PASS + M7 status (2026-09-26 01:33)

`examples/layer_demo.py` (job 58892432): overlap / direct / overlap+swap all PASS against the
dense PyTorch reference (max |err| 2.4e-4 at ref max 4.8e-2, bf16). The final repo state is
3719c05 (+ results 1f3ca13); the binary in `python/moe_ep/lib` is the fa07db0 build (the
python-only 3719c05 change does not touch it). M7 = `scripts/reproduce.sh --nodes 8|16`
queued in the regular QOS (jobs 58892055 / 58892056, no start estimate); their outputs land in
`$PSCRATCH/workspace/andrewy/logs/moe_ep/grid/{8n,16n}/` (`compare_<N>n.txt`) and must be
copied into `results/measured/` and read against the acceptance rule (every cell within 5 % of
the v4 Ours row, best of three below the reference ceiling). 16n runs `--router-c 0.5`
(published setting); 16n direct b16 has no published value (heap-sizing failure in the paper
runs) and is expected to fail the same way.

## M7 DONE (2026-09-26 06:42-07:05, regular QOS): 8n job 58892055 (11 min), 16n job 58892056 (12 min)

Queue wait ~5.5 h after submission at 01:07. Final binary (fa07db0 build), `router_c` 0.25 at
8n / 0.5 at 16n. **8n: 14/17 within 5 %; 16n: 12/16 within 5 %; best of three below the
reference ceiling in all twelve (nodes, model, budget) cells.** Copied into the repo as
`results/measured/main_perf_{8,16}n.csv` + `compare_*.txt`.

8n (ms, overlap / direct / swap; published in parentheses):
| model | MiB | overlap | direct | swap |
|---|---|---|---|---|
| qwen3 | 1 | 3.825 (4.462) | 5.225 (5.173) | 3.991 (4.193) |
| qwen3 | 4 | 6.234 (6.004) | 12.039 (12.106) | 6.417 (6.832) |
| qwen3 | 16 | FAILED, see below (15.580) | 39.002 (40.072) | 15.211 (15.629) |
| k2 | 1 | 4.463 (4.371) | 5.707 (5.737) | 4.875 (5.377) |
| k2 | 4 | 7.350 (7.271) | 11.891 (12.079) | 7.765 (8.064) |
| k2 | 16 | 17.987 (17.774) | 38.216 (38.712) | 17.631 (17.613) |
16n:
| model | MiB | overlap | direct | swap |
|---|---|---|---|---|
| qwen3 | 1 | 6.040 (5.998) | 5.912 (5.828) | 6.132 (6.732) |
| qwen3 | 4 | 9.188 (9.122) | 14.389 (14.246) | 9.324 (10.121) |
| qwen3 | 16 | 23.833 (23.209) | heap failure (no published value) | 23.485 (24.557) |
| k2 | 1 | 6.633 (6.566) | 6.040 (6.088) | 6.870 (7.536) |
| k2 | 4 | 10.321 (9.877) | 14.936 (14.164) | 10.922 (10.750) |
| k2 | 16 | 25.452 (26.294) | heap failure (no published value) | 26.150 (26.239) |

Out-of-band cells: six are the swap arm 6-9 % FASTER than published (as at 4n); qwen3 8n b1
overlap is 14 % FASTER (3.825 vs 4.462; the published 8n b1 plotted min was the swap arm at
4.193, and the fresh overlap arm beats it); k2 16n b4 direct is +5.5 % (14.936 vs 14.164, the
direct arm's plan_comm 0.88 ms was the largest of the grid — noise on the loads all_gather).
Failures: (1) 16n b16 direct on BOTH models: `NVSHMEM_MALLOC failed` — the same symmetric-heap
wall the paper runs hit (16n direct b16 has no published value; the acceptance CSV carries
none). (2) 8n qwen3 b16 overlap: CUDA illegal memory access on rank 17 in the combine = the
OPEN intermittent class recorded in memory as "Combine IMA 8n Qwen b16" (~2/10 cells in the
research tree), carried over unchanged by the extraction — not a pruning regression. A
two-repetition re-run of that cell is queued (`logs/moe_ep/rerun8n.sh`, output
`grid/8n/rerun_qwen3_b16_overlap_swap0.csv`); the swap arm of the same cell ran clean (15.211).
Acceptance at 8n/16n: rule (ii) holds everywhere; rule (i) holds for every cell that ran except
the one +5.5 % direct cell and the swap cells that are faster.

## Combine IMA root-cause hunt (2026-09-26 08:40-, user directive: the repo must ship without bugs)

Setup: scratch clones `$PSCRATCH/workspace/andrewy/moe_ep_debug` (bench instrumented: iteration
+ phase-sync localization via `MOE_EP_DEBUG_SYNC=1`) and `moe_ep_debug2` (same + `-lineinfo`);
cell = 8n qwen3 b16 overlap swap=0; chains in `logs/moe_ep/ima/` (debug QOS, 8 nodes, 30 min).
- Rate: 5 faults / 64 plain reps (~8 %); ranks 4, 5, 9, 15 (+17 in the grid); iterations 2, 9,
  13, last -> not first-iteration, not rank-specific. Always detected at the next device sync.
- With a device sync after every phase (plan / dispatch / act / combine): 0 / 8 -> the fault
  needs cross-stream concurrency that the phase syncs remove (side-stream combine-meta derive
  under the dispatch GEMM, or dispatch side-stream stragglers under the combine).
- Tooling: launch-blocking / sanitizer deadlock the spin kernels (known). GPU core dumps
  (`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`) are written (full ones truncated by torchrun's teardown;
  `CUDA_COREDUMP_GENERATION_FLAGS=skip_*` gives complete 82 KB dumps) but NEITHER cuda-gdb 12.4
  nor 13.2 can open them under driver 580.178 (`m_num_devices > 0` assertion) — dead end.
- Live attach works: `CUDA_DEVICE_WAITS_ON_EXCEPTION=1` parks the faulting rank; cuda-gdb 12.4
  attaches to the workers (no yama ptrace restriction). Chain `chain8n_wait.sh` (w2) = watchdog
  at 100 s + parallel per-node `probe_node.sh` (kernels, device bt, lanes). In flight.

## Combine IMA ROOT-CAUSED (2026-09-26 10:11, chain w3 rep 3, live cuda-gdb attach)

Not the combine at all. The faulting kernel is the dispatch op's `prepare_workspace_kernel`
(`src/dispatch/workspace_util.cu`, upstream Flux code, one block of 768 threads): warp 17 (threads
576-607) with `Warp Illegal Address` at the `problem_info[...]` store, line 174 of
`fill_problem_info` (SASS offset 0x8070 mapped through the -lineinfo build).
Mechanism: after `aligned_block_prefix_sum_and_sync`, every thread reads
`ep_splits_acc[ep_nexperts-1]` from shared memory to derive `tiled_m` / `num_tiles`; then
`fill_problem_info` lets warp 0 overwrite the SAME shared buffer with the per-tile schedule
table (`sched_tile[m]`, int16 pairs) with no barrier in between. A warp that is scheduled late
reads a schedule pair instead of the padded row count, gets a garbage tile count, and its
`problem_info` writes run past the workspace (crash) or land at wrong positions (silent
corruption risk). Intra-block warp-timing race: identical batch every rep, ~8 % at 8n Qwen b16
where per-expert tile counts make warp 0 reach entry E-1 fastest. The same race exists in the
research tree (`src/moe_ag_scatter/workspace_util.cu`, dense + a2av static paths) and in
upstream Flux's dense path.
Why the earlier evidence misled: the fault is asynchronous and surfaced at the next host sync,
which in the research harness was the combine's msplit event sync (hence "combine IMA").
Fix (one barrier at the top of `fill_problem_info`): release repo e252095, debug copy, and the
research tree file (uncommitted there; needs its own rebuild before any research-tree run).
Validation in flight: 46 plain reps on the fixed debug build (chain fix1; pre-fix rate 5/64),
then the release binary rebuilt and re-verified (4n `--check` grid, 4n perf grid; 8n/16n grids
re-queued in the regular QOS so every published number comes from the final binary).
Also today: `bench --check` tightened (every timed iteration, fresh payload, real route path):
18/18 cells PASS on 4n (both models x b1/b4/b16 x overlap/direct/swap) on the pre-fix binary.

## Fix validated (2026-09-26 10:15-10:50)

- Rate: **46/46 plain reps clean** on the fixed debug build (8n qwen b16 overlap; pre-fix 5/64,
  P(46 clean | 8 %) ~ 2 %). Chain `logs/moe_ep/ima/chain_fix1_status.txt`.
- Correctness: `bench --check` (every timed iteration, fresh payload, real route) **18/18 PASS**
  on the fixed release binary, 4n, both models x b1/b4/b16 x overlap/direct/swap (verify v3).
- 8n perf grid on the final binary (debug QOS, 12 min): **18/18 cells ran** (the former faulting
  cell 15.262 vs 15.580 published); 11/18 within 5 %, all seven "out" cells FASTER than published
  (swap -5..-11 %, qwen b1 overlap -15 %, direct b16 -6/-8 %); best of three below the reference
  ceiling in all six cells. Committed as `results/measured/main_perf_8n.csv`.
| model | MiB | overlap | direct | swap | published (overlap / direct / swap) |
|---|---|---|---|---|---|
| qwen3 | 1 | 3.789 | 5.120 | 3.982 | 4.462 / 5.173 / 4.193 |
| qwen3 | 4 | 6.158 | 11.912 | 6.171 | 6.004 / 12.106 / 6.832 |
| qwen3 | 16 | 15.262 | 37.751 | 15.424 | 15.580 / 40.072 / 15.629 |
| k2 | 1 | 4.394 | 5.650 | 4.766 | 4.371 / 5.737 / 5.377 |
| k2 | 4 | 7.508 | 11.748 | 7.483 | 7.271 / 12.079 / 8.064 |
| k2 | 16 | 17.678 | 35.502 | 18.287 | 17.774 / 38.712 / 17.613 |
- Final binary = build of e252095 (+ b740cc2 script change: time limits 45/15/30 min by node
  count, 300 s per cell, 16n direct b16 cells skipped as the known heap wall). 4n grid re-run
  and 16n grid (regular QOS, 30 min) in flight on it.

## 4n grid on the final binary (2026-09-26 10:34-10:50, job 58906878) + re-run

13/18 within 5 %; the four swap cells are 5-10 % FASTER than published; the one slow cell,
qwen3 b16 overlap 12.122 (+6.3 %), re-ran at 11.429 / 11.366 on the same binary (published
11.401) -> a high draw, not a shift. Best of three below the reference ceiling in all six cells.
Committed as `results/measured/main_perf_4n.csv` + `rerun_4n_qwen3_b16_overlap.csv`.
| model | MiB | overlap | direct | swap | published (overlap / direct / swap) |
|---|---|---|---|---|---|
| qwen3 | 1 | 2.795 | 4.725 | 3.030 | 2.817 / 4.660 / 3.362 |
| qwen3 | 4 | 4.425 | 10.213 | 4.616 | 4.366 / 10.139 / 4.893 |
| qwen3 | 16 | 12.122 (re-run 11.429 / 11.366) | 33.019 | 12.008 | 11.401 / 32.404 / 12.072 |
| k2 | 1 | 3.803 | 6.012 | 3.997 | 3.797 / 6.059 / 4.369 |
| k2 | 4 | 5.796 | 11.837 | 6.068 | 5.765 / 11.614 / 6.431 |
| k2 | 16 | 14.178 | 35.101 | 14.217 | 13.864 / 33.859 / 14.911 |
Remaining: 16n grid on the final binary (regular QOS, 30 min, job 58906248, queued 10:29).

## Memory budget of the defaults (for the SGLang lane; 2026-09-26, user question)

Symmetric heap is exact and known at construction (per rank, overlap, Qwen3, T = 2048 tokens/rank
= 16 MiB, 8n capacities from the grid logs): dispatch send 128 MB, recv/stage/relay 363/147/44 MB,
combine send/recv/conv/wire 305/134/235/148 MB, swap staging 201 MB (if on), plus a LEFTOVER dense
inter-node staging pair in the combine (`staging_send`/`staging_recv`, 2·NN·T·H·2 B = 268 MB at
8n, 537 MB at 16n) that nothing reads since the gateway path went (B6b) — delete in a later
allocation-only batch. Total ~1.8-2.0 GB against the 6 GiB heuristic minimum (`heap.py`); the
overlap path scales with T only, the direct path with ranks x pair capacity (the 16n b16 wall,
direct only, accepted). At the 16 GiB cap overlap fits T ~ 16k tokens/rank.
SGLang implications: one shared op instance (94 MoE layers x 2 GB is impossible), size
`max_tokens_per_rank` for the prefill chunk (TTFT) not decode, SwiGLU doubles the dispatch
intermediate, and on 40 GB A100 at 16n the per-rank weights (2 home + 2 redundant slots x 94
layers ~ 14 GB experts + replicated non-expert params) leave too little for KV cache at
prefill-sized T -> run on the hbm80g pool (or 32 nodes) with the heap set from the chunk size.

## CLOSED (2026-09-26 12:25): M0-M8 complete

Final state: open-source repository `$PSCRATCH/workspace/andrewy/moe_ep`, binary = build of
e252095 (source HEAD d1abfe5 adds only results/docs). Gates: ban grep 0, env reads in src 0
(three rank-0 log gates in Flux's op_registry.h remain), process-artifact grep 0, working tree
clean, fresh-clone build verified (cutlass from NVIDIA's public GitHub), layer demo 3/3,
`bench --check` 18/18 on 4n (every iteration, fresh payload, real route path), workspace-kernel
race fixed and validated 46/46.
Final-binary grids (`results/measured/`, README table, `results/main_perf.png`):
| nodes | within 5 % of v4 | cells outside | verdict |
|---|---|---|---|
| 4 | 13/18 | 4 swap cells faster (-5..-10 %); qwen b16 overlap +6.3 % -> re-run 11.43/11.37 (+0.2/-0.3 %) | all 6 cells below ceiling |
| 8 | 11/18 | all 7 faster than published (swap -5..-11, qwen b1 overlap -15, direct b16 -6/-8) | all 6 below ceiling; the former IMA cell ran (15.26 vs 15.58) |
| 16 | 13/16 | 3 swap cells faster (-6..-12 %); direct b16 skipped by design | all 6 below ceiling |
No cell of any grid is slower than published beyond twin noise (max +4.7 %, 16n K2 b4 overlap).
Research tree: `figs/main_perf_v5` = v4 baselines + Ours rows from the open-source build (commit
54fbd65); plotted Ours moves within +-3 % in 15/18 cells (8n Qwen b1 -9.6 %, 16n K2 b4 +4.7 %,
8n K2 b4 +2.9 %). The workspace-kernel fix is committed in the research tree (79ae81b) but its
binary is NOT rebuilt (item for whoever next runs it).
Open, optional: (a) `fill_problem_info` shared-buffer aliasing refactor (bug class, not
instance); (b) leftover dense inter-node staging allocation in the combine (268-537 MB of heap);
(c) residual dead code ("pieces" mode, constant getters, Flux dense remnants, kernel-builder
names); (d) the three rank-0 log env gates. Each = one small batch + one 4n read.
Next lane: SGLang integration per the scoping table above (one shared op instance, SwiGLU,
prefill-sized `max_tokens_per_rank`, hbm80g pool).

## Post-close batch (2026-09-26 12:40-12:58, user directive): dense staging removed

moe_ep 5027029: the combine's unused dense inter-node staging buffers (`staging_send`,
`staging_recv`, `internode_signals`; declared + allocated, never read) are gone — 2*NN*T*H*2 bytes
of symmetric heap per rank freed (268 MB at 8n, 537 MB at 16n for Qwen b16). Allocation-only
change; shipped binary = build of 5027029. Gate read 8 (4n qwen, job 58913250): 7/9 within 5 %,
the two outside = swap arm faster (3.055 / 4.628 vs 3.362 / 4.893), best of three below the
reference ceiling at every budget (2.792 / 4.435 / 11.203). The other residuals (pieces mode,
constant getters, dispatch dense remnants, kernel-builder names, rank-0 log env gates) stay by
user decision (main perf holds and runs; the repo is open-sourced as is).
