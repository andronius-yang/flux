# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Deployment context: NERSC Perlmutter

This is ByteDance's Flux/Comet repository (fine-grained computation-communication overlapping GPU kernels), **deployed on the NERSC Perlmutter platform**. Compute nodes have **4x A100 GPUs each** (sm80, 108 SM cores — hence `--arch 80 --sm-cores 108`). The working tree contains **local edits made specifically to get Flux/Comet to compile on Perlmutter** — do not blindly revert them to match upstream:

- `module.sh` (untracked, Perlmutter-specific): restores Cray lmod defaults, loads `PrgEnv-gnu`, `gcc/12.2.0`, `cray-mpich`, `cudatoolkit/12.4`, `nvshmem/3.2.5-1`, `nccl/2.24.3`, and activates the conda env at `$PSCRATCH/conda_envs/andrewy-comet`. Sets `CC/CXX/CUDAHOSTCXX`, `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.0`, `FLUX_ROOT`, and puts the bundled NCCL headers on `CPATH`.
- `build.sh`: python bindings installed via `pip install --no-build-isolation --editable .` instead of upstream's `python3 setup.py develop --user`.
- `setup.py` / `python/flux/cpp_mod.py`: `NVSHMEM_HOME` from the environment (the Perlmutter module) takes strict precedence over the pip-installed nvshmem; the eager import of pip nvshmem is avoided. The `nvshmem_transport_ibrc.so.3` preload is commented out (Perlmutter is Slingshot, not InfiniBand).

## Build

Always load the environment first (must be sourced, not executed):

```bash
source ./module.sh
```

Then build (this is the known-working Perlmutter compile invocation):

```bash
nproc=16 ./build.sh \
  --arch 80 \
  --sm-cores 108 \
  --nvshmem \
  --no_test \
  --jobs 16
```

Notes:
- `--nvshmem` is required for the MoE (Comet) kernels.
- Set `FLUX_BUILD_SKIP_CMAKE=1` to skip re-running cmake when `build/CMakeCache.txt` already exists (incremental rebuilds).
- `./build.sh --clean-py` removes only python build artifacts; `./build.sh --clean-all` also removes `build/` and the 3rdparty NCCL/protobuf builds.
- `--debug` enables FLUX_DEBUG; `--package` produces a wheel under `dist/`.
- Dependencies: NCCL and CUTLASS 4.0 are git submodules under `3rdparty/` (NCCL is built as a static lib by build.sh); NVSHMEM comes from the Perlmutter module (`NVSHMEM_HOME`).

## Running experiments

Get an interactive GPU allocation first — jobs must run on compute nodes, not login nodes. Always use `salloc` + `srun` (never `sbatch` — user preference, and the interactive QOS rejects batch jobs):

```bash
salloc --qos interactive -C gpu --account m4243_g
```

Keep all test logs under `/tmp` (e.g. a per-session scratch dir), never inside the
repository tree. Redirect job output at the login-side shell (the one running
`salloc`/`srun`), since a compute node's `/tmp` is node-local and discarded when
the job ends. The inverse holds for job *inputs* the compute nodes must read
(e.g. traffic matrices): those cannot live in login-node `/tmp` — put them on a
shared filesystem such as `$PSCRATCH` (synthetic 4-rank test matrices live in
`$PSCRATCH/a2av_test_matrices/`), still not in the repo.

Single-node multi-GPU tests go through `launch.sh`, which wraps `torchrun` with `--nproc_per_node` set to the number of visible GPUs (4 on Perlmutter) and sets required env vars (`NVSHMEM_BOOTSTRAP=UID`, `NVSHMEM_DISABLE_CUDA_VMM=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, ...):

```bash
# gemm only (single GPU, no launcher needed)
python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 --dtype=float16

# all-gather fused with gemm (dense MLP layer0)
./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10

# gemm fused with reduce-scatter (dense MLP layer1)
./launch.sh test/python/gemm_rs/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10

# Comet MoE layer0 (all-gather + scatter + grouped gemm)
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py

# Comet MoE layer1 (grouped gemm + gather + topk-reduce + reduce-scatter)
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py
```

MoE examples also live in `examples/` (`bash examples/run_moe.sh`); `examples/moe_flux_only.py` is a minimal MoE layer built on Flux. Note `moe_layer1.py` defaults to `-T 8` (8 ranks); on a single 4-GPU node pass `-T 4 -E 1`, and the gather_rs tests require `T * E == world size`.

### Multi-node (Comet multi-node port, validated on 2 nodes)

The sm80/V2 MoE ops support multi-node: layer0 via a port of the sm90/V3 NVSHMEM all-gather into `GemmGroupedV2AGScatterOp` (honors `DistEnvTPWithEP.nnodes`), layer1 via a hierarchical intra-node-ring + host-staged NVSHMEM `putmem_signal` design in `GemmGroupedV2GatherRSOp` (new `nnodes` ctor kwarg, wired through the tests/examples via `flux.testing.NNODES()`). `launch.sh` is Slurm-aware: with `SLURM_NNODES > 1` it derives `nnodes`/`node_rank`/master from Slurm and selects the NVSHMEM libfabric/CXI (Slingshot) transport. Launch one `launch.sh` per node:

```bash
salloc -A m4243_g -q interactive -C gpu -N 2 --gpus-per-node=4 -t 30
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
srun --nodes=2 --ntasks-per-node=1 ./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 8 -E 1
```

Multi-node constraints: token counts divisible by `world_size * topk`; `max_m/topk` divisible by `world_size`; for large configs export `NVSHMEM_SYMMETRIC_SIZE=4G` (all reduce/staging buffers live on the symmetric heap). `do_all_reduce`, `use_read_mode`, and the triton/int8 paths are single-node only (FLUX_CHECK-guarded).

## Formatting

`./code-format.sh` runs clang-format/black over the diff (`--format-all` for everything, `--fail-on-diff` for CI-style checks). Style follows Google style guides via clang-format.

## Architecture

Flux fuses communication into CUTLASS GEMM kernels so tile-level compute overlaps with comm I/O. Comet (the MoE work) is implemented in this same repo.

Layering, bottom to top:

1. **CUDA/C++ kernels — `src/`**: one subdirectory per fused op, each generating CUTLASS-based kernels:
   - `src/ag_gemm` — AllGather+GEMM (dense MLP layer0)
   - `src/gemm_rs` — GEMM+ReduceScatter fused into the GEMM epilogue (dense MLP layer1)
   - `src/moe_ag_scatter` — **Comet MoE layer0** (all-gather + scatter + grouped GEMM)
   - `src/moe_gather_rs` — **Comet MoE layer1** (grouped GEMM + gather + topk-reduce + reduce-scatter)
   - `src/a2a_transpose_gemm`, `src/gemm_a2a_transpose` — all-to-all variants; `src/coll` — collectives; `src/generator` — kernel/config generation (GEMM configs are enumerated per arch and tuned; see `docs/tuning_guide.md`)
2. **Torch op layer — `src/ths_op/`**: C++ torch wrappers around the kernels.
3. **Python bindings — `src/pybind/` → `python/flux/`**: the `flux` python package. `python/flux/cpp_mod.py` preloads `libflux_cuda.so` and the NVSHMEM host libs before importing the pybind module. `libflux_cuda.so` is installed into `python/flux/lib/`.
4. **Tests — `test/python/<op_name>/`**: mirror the `src/` op layout; multi-GPU tests are launched via `./launch.sh` (torchrun). C++ unit tests under `test/` are gated by `BUILD_TEST` (disabled here via `--no_test`).

Key design points (details in `docs/design.md`): on Ampere (this deployment), communication is fused into the GEMM epilogue and the threadblock scheduler hides remote-I/O latency by switching among oversubscribed threadblocks; kernels reschedule tile computation along the independent dimension (Dim-M for MoE layer0, Dim-N for layer1) to start overlap early. Hopper uses a different warp-specialization design that does not apply on A100.

Inter-GPU data movement for the MoE kernels goes through **NVSHMEM** (hence the hard requirement on `--nvshmem` and a correct `NVSHMEM_HOME`).
