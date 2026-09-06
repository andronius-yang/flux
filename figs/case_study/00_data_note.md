# Case-study figure lane — data note (2026-09-05)

Two 4n case studies of OURS (one-round overlapped expert swap,
`ours_l01_s2_swap_p2p_t1_r2`) showing where GPU compute, NVLink and NIC
RDMA time went, plus two twins per case: `_noov` (sequential swap = overlap
effect) and `_esplit` (locality-oblivious equal-split routing = local-routing
effect). All from ONE capsule / ONE binary / ONE process per arm:

* capsule `sweeps/results/runs/20260905-121506_perlmutter_7a584851` (6/6 ok,
  nsys mode, K2 b64, 16 ranks). nsys mode = per-iteration synced timelines
  and byte ledgers only — NEVER latency (SCHEMA rule 3).
* spec `sweeps/specs/casestudy_sc_d4_k2_4n_nsys.yaml`: family 1 = the S-C
  dwell-4 schedule of handoff 34 (LOO 7-pool basis, 8 topics, placement
  carried over, 32+5 iterations all profiled); family 2 = plain lcb homog
  (main-figure pool, fresh oracle placement, 3+3 iterations).
* iteration -> topic (family 1, run index incl. 5 warmups; NVTX label
  `iterN`): proLaw 0-3 (warmup), lcb 4-7, clinical 8-11, colmath 12-15,
  elecEng 16-19, hsPsych 20-23, hsWorldHist 24-27, philosophy 28-31,
  proLaw 32-35, lcb 36. Sidecar epoch index k == run iteration k.

Selected cells / iterations:

| case | cell | iteration | why |
|---|---|---|---|
| EFFICIENT | plain lcb `..._trace-610042_b64_k8_nsys` | `iter4` (2nd timed) | fresh oracle placement, 46.4 ms, rows/rank spread 1.17x, no swap fires |
| SKEWED | schedule `..._trace-b549f7_b64_k8_nsys` | `iter33` (2nd of the proLaw block; `iter32` = block entry) | LOO basis, 57.1 ms, rows/rank spread 1.53x, 14/16 ranks swap 58 MB every iteration |

Derived data (PSCRATCH, regenerate with the two scripts here):
`$PSCRATCH/workspace/andrewy/figs_data/case_study/{timeline_20260905-121506.json,
timeline_summary_20260905-121506.csv, byte_ledger_20260905-121506.csv, sqlite/}`
— `extract_timeline.py` (per rank x iteration device events classed into
GPU / NIC / NVLink / wait lanes + host `plan.*`/`swap.*` ranges + a2av proxy
per-source wait/compute ranges), `byte_ledger.py` (receive-side rows/bytes
per rank per epoch split self / NVLink / NIC from the tile-trace sidecar).

Code changes for this lane (ALL case-study-only, default-off, no effect on
existing arms/specs): `--route_rule equal_split` (driver + planner +
`equal_split_route_all` sizing fold), gated `swap.*` NVTX ranges
(`FLUX_OURS_NVTX=1`), runner keeps spec iters for schedule families under
profiling modes, arm `ablation_l01_s2_swap_t1_esplit_p2p_r2`.

## Capture 2 (2026-09-05 pm) — overlapped COMET baseline, one rebuilt binary

User ruling: the figure needs an "overlapped" worse-performing baseline to
compare the OURS timeline against = COMET with its pre-GEMM completion gate
dropped and `CUDA_DEVICE_MAX_CONNECTIONS=8` (`l01_allgather_dense_nogate_c8`,
handoff 30 rebuttal cell). The knob lived only on the motif branch, so
`36dfd8d` (TILE_TRACE_DENSE) and `af76971` (DENSE_NO_GEMM_GATE + arms +
specs) were cherry-picked onto main as `b2e9998` / `bb4a637` and the .so
rebuilt (9/5 07:06, tags present, python/flux/lib + lib64 synced).
**Capsule 20260905-121506 is the pre-knob binary — never mix.**

* capsule `sweeps/results/runs/20260905-141104_perlmutter_fdd1c4af` (20/20
  ok): spec `casestudy2_baseline_sc_d4_k2_4n_nsys.yaml` = the same two
  families x 5 arms (COMET no-gate c8, COMET gated, OURS ovl / noov /
  equal-split) x modes [nsys, isolated]. THE FIGURE IS NOW DRAWN FROM THIS
  CAPSULE (all rows one binary); the isolated cells answer the latency
  question on the same binary.
* Isolated, max rank per iteration, median over block, total / l0 / l1 ms:

| arm | plain lcb (efficient) | sched proLaw block (skewed) | sched whole |
|---|---|---|---|
| COMET overlapped (no-gate c8) | 55.2 / 21.7 / 32.6 | 58.2 / 23.6 / 33.7 | 54.0 / 20.9 / 31.3 |
| COMET gated | 60.2 / 29.2 / 29.4 | 64.7 / 33.6 / 30.1 | 58.0 / 29.2 / 27.9 |
| OURS overlapped swap | 46.7 / 19.0 / 25.0 | 56.6 / 23.1 / 29.8 | 49.6 / 20.7 / 25.9 |
| OURS sequential swap | 46.3 / 19.3 / 24.1 | 57.2 / 22.8 / 30.6 | 51.6 / 20.7 / 26.8 |
| OURS equal-split routing | 50.0 / 21.2 / 26.2 | 57.2 / 23.1 / 30.2 | 52.2 / 22.0 / 26.8 |

  Reading: dropping COMET's gate buys 7-10 ms of layer 0 but costs 3-4 ms of
  layer 1 (8 connections hurt the dense reduce-scatter); at b64 4n its l0 does
  NOT beat OURS in the efficient case (21.7 vs 19.0) and ties in the skewed
  block (23.6 vs 23.1). On the schedule's lcb block right after proLaw
  (stale carried placement) overlapped COMET is the faster arm (51.0 vs 53.5
  total) — quote with that caveat only.
* Derived data: `timeline_20260905-141104.json`, `timeline_summary_20260905-141104.csv`,
  `byte_ledger_20260905-141104.csv`. CAVEAT for the COMET rows: the sidecar's
  per-source row counts are the rows this rank COMPUTES from each source
  (post-scatter GEMM rows), which for the dense allgather is not what
  crossed the wire (the allgather moves the fixed W x budget shard set,
  ~6.4 GB NIC total, spread ~1.0x by construction). So `bytes_inter` for
  COMET = compute rows attributed to remote sources, and `rows_total` spread
  = COMET's expert-compute imbalance (3.0x efficient, 3.8x skewed vs OURS
  1.17x / 1.53x). The byte table in the review page labels COMET rows
  accordingly.
* Extractor classes added for the dense path: `nvshmemi_proxy_rma_entrypoint_blocking`
  (getmem fetch) -> NIC token comm; `ep_topk_gather_rs_kernel_v2` +
  `internode_reduce_kernel` -> Top-k Reduce (GPU side lane);
  `nvshmemi_signal_wait_until_on_stream_kernel` -> wait; index/workspace
  kernels -> plan compute. 67 MB P2P copies on NVLink = the intra-node allgather.
