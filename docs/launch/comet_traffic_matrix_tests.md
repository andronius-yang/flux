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
