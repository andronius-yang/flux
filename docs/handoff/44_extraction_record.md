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
