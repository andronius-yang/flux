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
