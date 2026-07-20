# Launching the COMET traffic-matrix tests (Perlmutter)

Profiles MoE layer0 (`moe_ag_scatter`) and layer1 (`moe_gather_rs`) with token
routing prescribed by an a2av traffic matrix (line 1 = nranks, then an NxN byte
matrix, row = src rank, col = dst rank; entries are multiples of 8192 B = one
4096-dim 16-bit token; equal row sums; zero diagonal). Tokens homed on rank `s`
route exactly `M[s][d] / 8192` (token, topk-slot) copies to experts owned by
rank `d`. EP is fixed so each expert's full FFN weight resides on one rank
(layer0: `ep_size = world_size`; layer1: `T=1, E=world_size`).

Matrices: `~/workspace/changchen/andrewy/profiling/traffic/matrices/a2av/4n_16r/<budget>mib/`.

## Single node (4 GPUs)

```bash
source ./module.sh
salloc -A m4243_g -q interactive -C gpu -N 1 --gpus-per-node=4 -t 30
srun --nodes=1 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_ag_scatter/test_moe_ag_traffic.py --traffic_matrix <4-rank-matrix.txt>
srun --nodes=1 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_gather_rs/test_moe_gather_rs_traffic.py --traffic_matrix <4-rank-matrix.txt>
```

The matrix rank count must equal the world size (4-node matrices need 16 ranks).

## Multi node (4 nodes x 4 GPUs, 16-rank matrices)

```bash
source ./module.sh
export NVSHMEM_SYMMETRIC_SIZE=4G   # needed for the 32/64 MiB budgets
salloc -A m4243_g -q interactive -C gpu -N 4 --gpus-per-node=4 -t 30
MTX=.../matrices/a2av/4n_16r/64mib/a2av_4n_16r_dist_001.txt
srun --nodes=4 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_ag_scatter/test_moe_ag_traffic.py --traffic_matrix $MTX --iters 10 --warmup_iters 5
srun --nodes=4 --ntasks-per-node=1 ./launch.sh \
    test/python/moe_gather_rs/test_moe_gather_rs_traffic.py --traffic_matrix $MTX --iters 10 --warmup_iters 5
```

## Key arguments and constraints

- Defaults: `--G 32 --topk 4`, hidden `--H 4096` (layer0) / `-N 4096` (layer1),
  bf16. `H * dtype_size` (layer0) / `N * dtype_size` (layer1) must equal the
  matrix chunk granularity (`--chunk_bytes`, default 8192).
- `G % world_size == 0`; `topk` must divide each row's chunk count; no expert
  may receive more chunks from one source rank than that rank has tokens
  (distinct topk experts per token) — violations fail loudly.
- `--profile` dumps merged chrome traces under `prof/`. Both tests verify flux
  against a torch reference (allclose) every run.
- Sanity-check a matrix without a job:
  `python3 python/flux/testing/traffic_matrix.py <matrix.txt> [G] [topk]`.

Note on semantics: layer0 physically performs a dense all-gather then a local
scatter — the matrix shapes logical dispatch and per-rank GEMM load, not layer0
wire bytes. Layer1's gather-RS payload realizes the matrix transpose
(expert-owner -> token-home).

## Layer0 a2av dispatch mode (`--comm_pattern a2av`, sm80/V2 only)

Design walkthrough (what changed and why, from the communication-patterns
narrative): `comet_traffic_matrix_a2av.md`.

`test_moe_ag_traffic.py --comm_pattern a2av` replaces the dense all-gather with a
raw alltoallv: each (token, topk-slot) copy is sent directly producer ->
expert-owner rank via host-issued NVSHMEM `putmem_signal`, so wire bytes s->d
equal exactly `M[s][d]`. The grouped GEMM claims tiles dynamically in
signal-arrival order (per-source buckets + atomic cursors; the per-tile signal
spin remains the correctness backstop). Signals are epoch-valued (`run_id`),
never reset.

- Requires `ep_size == world_size`, single weight group, no `--gather_input`.
- Recv buffer capacity defaults to 2x the balanced per-rank load; skewed
  matrices (the real a2av sets have ~3x hot columns) need
  `FLUX_A2AV_MAX_RECV_NTOKENS=<rows>` (rows = received (token, slot) copies;
  4x average, i.e. `ntokens * topk / world * 4`, is a safe choice). The op
  prints its symmetric-heap need at construction; keep
  `NVSHMEM_SYMMETRIC_SIZE=4G`.
- Validated 2026-07-17 on 1 node (4r, synthetic + skewed matrices) and 4 nodes
  (`4n_16r/{16mib,64mib}/dist_001`), 16/16 ranks allclose vs torch in every
  configuration.

## Layer0 hierarchical a2av (`--comm_pattern a2av_hier`, sm80/V2 only)

Design: `comet_traffic_matrix_a2av.md` §10. Same wire semantics as a2av
(`M[s][d]` bytes end to end), but inter-node data travels as ONE aggregated
`putmem_signal` per remote node, addressed to the same-local-rank "gateway"
rank there, which forwards each destination's sub-chunk intra-node (paced per
round by a front-end `cuStreamWaitValue64` on the arrival signal). Consumer
uses the static ring schedule; per-tile signal spin is unchanged.

```bash
# single node (degenerates to intra-node direct puts, validates the branch)
srun --nodes=1 --ntasks-per-node=1 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_ag_traffic.py <matrix> --comm_pattern a2av_hier

# 4 nodes x 4 GPUs
NVSHMEM_SYMMETRIC_SIZE=4G srun --nodes=4 --ntasks-per-node=1 ./launch.sh \
  test/python/moe_ag_scatter/test_moe_ag_traffic.py \
  <matrices>/4n_16r/16mib/..._dist_001.txt --comm_pattern a2av_hier
```

- Same constraints/knobs as a2av (`ep_size == world_size`, one weight group,
  no `--gather_input`, `FLUX_A2AV_MAX_RECV_NTOKENS` for skewed matrices).
- Additionally `FLUX_A2AV_MAX_STAGE_NTOKENS` sizes the gateway staging buffer
  (rows; default = the recv-buffer formula). Overflow aborts collectively
  before any communication; raise the knob for matrices whose inbound node
  traffic concentrates on one source local rank.
- The harness prints per-rank aggregated inter-node send bytes
  (`a2av_hier inter-node wire bytes`) next to the dense a2av wire bytes.
- Validated 2026-07-19: 1 node (4r uniform/skewed x metadata/derive + 50-iter
  epoch stress) and 4 nodes (`4n_16r/{2,16,64}mib/dist_001`, metadata + derive),
  16/16 ranks allclose in every cell. Same-allocation sweep (mean ms over 16
  ranks): 2mib AG 1.19 / a2av 1.30 / ring 1.07 / hier 0.93; 16mib AG 2.73 /
  a2av 2.36 / ring 2.42 / hier 2.20; 64mib AG 8.44 / a2av 8.07 / ring 8.66 /
  hier 6.70 — a2av_hier is the fastest pattern at every budget, including the
  previously AG-favored hot-rank-bound 64mib (~20% under AG).
- topk sweep 2026-07-19 (`dist_001`, topk in {1,2,4}, same allocation, mean ms
  over 16 ranks, AG / a2av / ring / hier): 2mib tk1 1.47/1.08/1.07/0.91,
  tk2 1.27/1.08/1.08/0.92, tk4 1.15/1.07/1.06/0.92; 16mib tk1
  5.27/2.43/2.38/2.30, tk2 3.64/2.39/3.34/2.29, tk4 2.71/2.47/2.35/2.25;
  64mib tk1 20.28/8.52/8.34/7.89, tk2 12.66/8.43/8.30/7.85, tk4
  8.52/8.46/8.57/7.94. For a fixed matrix the a2av-family wire is the budget
  (constant in topk) while allgather's is budget/topk, so allgather degrades
  ~linearly as topk shrinks and the a2av modes stay flat: the a2av advantage
  is largest at topk=1 (hier 2.6x under AG at 64mib) and nearly vanishes at
  topk=4 (7%). a2av_hier is fastest in all 9 cells. (16mib tk2 ring 3.34 is
  an off-trend outlier vs tk1/tk4 ~2.35 — likely a contention blip.)

## Layer0 FAST baseline (un-overlapped: FAST alltoallv + separate grouped GEMM)

`test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py` measures the
un-overlapped counterpart of the fused patterns above: the load-balancing
alltoallv from the FAST submodule (`3rdparty/FAST`; BvN decomposition into
balanced inter-node permutation steps, intra-node load-balance/redistribute
over NVLink/IPC, inter-node NVSHMEM) moves exactly `M[s][d]` wire bytes, then a
comm-free `flux.GemmGroupedV2` consumes the landed tokens.

Build the submodule extension once (login node OK):

```bash
source ./module.sh
git submodule update --init 3rdparty/FAST
./scripts/build_fast.sh          # -> 3rdparty/FAST/nvidia/libflash.so
```

Run (multi-node ONLY — FAST asserts `server_n > 1`; use `launch_fast.sh`, NOT
`launch.sh`: FAST performs the only NVSHMEM init in this process and needs its
own validated env, host lib preloaded ahead of torch's bundled NVSHMEM):

```bash
salloc -A m4243_g -q interactive -C gpu -N 4 --gpus-per-node=4 -t 30
srun --nodes=4 --ntasks-per-node=1 ./launch_fast.sh \
  test/python/moe_ag_scatter/test_moe_ag_fast_baseline.py \
  --traffic_matrix <matrices>/4n_16r/16mib/a2av_4n_16r_dist_001.txt
```

- PRIMARY METRIC: `e2e` = one window from communication start (BvN schedule +
  send staging + wire) to computation finish (unpack + grouped GEMM), CUDA
  events, directly comparable to the fused `op.forward` numbers of
  `test_moe_ag_traffic.py`. The BvN schedule is recomputed inside the window
  every iteration (one-shot methodology — never amortized); the send-side pack
  `index_select` is reported separately outside the window. The printed
  `schedule/fill/wire/unpack/gemm` split is diagnostic.
- Correctness: the unpacked receive buffer must be **bitwise** equal to the
  reference `scatter_inputs` block (send-side stable argsort by expert id is
  simultaneously destination-major and expert-grouped; FAST's receive layout is
  deterministically source-major with within-flow order preserved), plus
  allclose of the GEMM output vs the torch per-expert loop.
- `--capacity_mib` sizes FAST's persistent buffers (default 4x the max
  row/column sum of the matrix); undersized capacity fails loudly pre-wire.
- Index-math unit test (no GPU/dist; login node):
  `python3 test/python/moe_ag_scatter/test_fast_index_math.py`.
- The test never calls `flux.init_flux_shm` — do not add fused-op timing to
  this harness; run `test_moe_ag_traffic.py` separately for those numbers.
- Validated 2026-07-20 (4 nodes, `4n_16r/{2,16,64}mib/dist_001`, 10+10 iters):
  48/48 rank-checks pass (wire+unpack bitwise vs the reference scatter block,
  same-op GEMM bitwise, allclose vs the torch loop). e2e mean over 16 ranks
  (ms): 2mib 4.06, 16mib 5.85, 64mib 13.24 — vs the fused patterns' recorded
  AG 1.19/2.73/8.44 and a2av_hier 0.93/2.20/6.70. The un-overlapped baseline
  is dominated by the per-iteration BvN schedule (~1.3 ms) and FAST's per-call
  signal/credit reset incl. two nvshmem_barrier_all (1.5/2.2/4.7 ms); wire
  0.84/1.76/5.15, grouped GEMM 0.15/0.33/1.16.
