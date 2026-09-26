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

## Trace slice provenance
`data/traces/{qwen3,k2}/{eval,pool}.txt` = `pool_cache/layer5_decode_d{64_96,32_64}.txt` of the
LCB execution pools (headers stripped; pool_sha in manifest.json). Sampler seed key kept
verbatim from `sweeps/gen_matrix.canonical_string` (params incl. `pool=decode` and the 12-hex
dataset fingerprint), so benchmark batches equal the capsules' routing files.
