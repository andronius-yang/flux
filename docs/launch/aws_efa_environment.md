# AWS ParallelCluster environment (p4d/A100 + EFA)

How the Flux/Comet environment is built and run on the AWS ParallelCluster
deployment (`andrewy-coll-comm`, us-west-2), and — critically — the
NVSHMEM-over-EFA fixes that any future environment rebuild must repeat.
This is the AWS analogue of the Perlmutter instructions in `CLAUDE.md`;
where they conflict, this document wins on this cluster.

Validated 2026-07-28: full ladder up to 2-node / 16-rank MoE layer0+layer1
with NVSHMEM on libfabric/EFA and NCCL on aws-ofi-nccl/EFA.

## Cluster facts that shape everything

| Fact | Consequence |
|---|---|
| Head node is m7i.large (2 vCPU, no GPU, no gdrcopy headers) | Never compile there. All builds run inside a Slurm allocation on a p4d node via `srun`. |
| Compute: 2× `p4d.24xlarge` (partition `a100`), **8× A100-SXM4-40GB** each, 4× 100 Gbps EFA, 96 vCPU | `--arch 80 --sm-cores 108` carries over from Perlmutter unchanged. 8 GPUs/node (not 4) changes MoE test args: gather_rs needs `T*E == world_size` → `-T 8 -E 1` (1 node), `-T 16 -E 1` (2 nodes). |
| **`/home` is the only shared filesystem** (NFS export of the head node's root volume; no FSx, no `$PSCRATCH`) | Everything persistent lives under `/home/ubuntu/sw/`. Compute nodes are dynamic — node-local installs (apt, `/tmp`) evaporate at scale-down. Job inputs (e.g. traffic matrices) must be under `/home`. |
| AMI ships **CUDA 13.0** at `/usr/local/cuda`, gcc/g++ 11.4, no NCCL lib, no NVSHMEM, no python venv module | We install CUDA 12.4 ourselves (validated stack) and must keep `/usr/local/cuda` out of the build's PATH. |
| EFA stack on every node: libfabric 2.4.0amzn at `/opt/amazon/efa`, aws-ofi-nccl 1.18 at `/opt/amazon/ofi-nccl` (on ldconfig path), gdrcopy 2.5.2 (**compute nodes only**) | NCCL-over-EFA is zero-config. NVSHMEM-over-EFA is not — see below. |

## Layout under `/home/ubuntu/sw`

```
cuda-12.4/            CUDA 12.4.1 toolkit (runfile --toolkit install, no sudo)
venvs/flux/           python3.10 venv: torch 2.6.0+cu124, nvidia-nvshmem-cu12==3.3.9,
                      numpy, ninja, cmake, packaging, wheel, pynvml, cuda-python==12.4.0
nvshmem-3.3.9/        our NVSHMEM source build (only the libfabric transport is used)
src/                  kept sources: cuda runfile, nvshmem_src (3.3.9 tarball),
                      libfabric-1.22.0 (built but NOT used at runtime — see pitfall 3)
```

Entry point: `source ./env_aws.sh` at the repo root (sibling of the Perlmutter
`module.sh`, which stays untouched). It selects CUDA 12.4, activates the venv,
sets `NVSHMEM_HOME` to the pip wheel, and exports the EFA transport env.

## The NVSHMEM-over-EFA fixes (the part rebuilds get wrong)

Three independent issues, discovered in order; all three fixes are required.
Symptom of getting any of them wrong: SIGSEGV inside `flux.init_flux_shm` →
`nvshmemt_init` (NVSHMEM's transport error path itself segfaults on a NULL
mem-handle cache, so init *failures* present as crashes, not error messages —
debug with `NVSHMEM_DEBUG=INFO` and gdb, not just the Python traceback).

1. **The pip wheel's libfabric transport is broken.** The
   `nvshmem_transport_libfabric.so.3` shipped in `nvidia-nvshmem-cu12==3.3.9`
   was linked against an unversioned custom libfabric (`readelf -V` shows no
   `FABRIC_*` version needs) and segfaults inside `fi_getinfo`/`fi_dupinfo` of
   any real libfabric. Fix: rebuild the transport from NVIDIA's source tarball
   and overlay it into the wheel (recipe below). Only the transport `.so` is
   swapped; the wheel's host lib, device lib, headers, and the flux build all
   stay as-is (transport plugin ABI is version-locked to 3.3.9, which is why
   the source version must match the wheel version).

2. **The EFA provider requires GDRCopy.** Both at NVSHMEM compile time
   (`-DNVSHMEM_USE_GDRCOPY=1`) and at runtime (`NVSHMEM_DISABLE_GDRCOPY=0` —
   the Perlmutter/Slingshot scripts set `=1`, which EFA rejects with
   "EFA Provider requires GDRCopy"). gdrcopy 2.5.2 (gdrdrv module,
   `/dev/gdrdrv`, `libgdrapi`, `/usr/include/gdrapi.h`) is preinstalled on the
   **compute** AMI but absent on the head node — hence the transport build must
   run on a compute node.

3. **Exactly one libfabric per process, and it must be Amazon's.** The
   transport must be compiled against `/opt/amazon/efa` headers
   (`-DLIBFABRIC_HOME=/opt/amazon/efa`). Two failure modes bracket this:
   - A transport built against libfabric **1.x** headers segfaults in Amazon
     libfabric 2.4's `fi_getinfo` 1.x-compat shims.
   - Putting a self-built libfabric 1.x ahead of Amazon's in `LD_LIBRARY_PATH`
     silently breaks NCCL: aws-ofi-nccl needs 2.4's `FABRIC_1.8` symbols, its
     dlopen fails, and NCCL falls back to **sockets** (400 Gbps → ~10 Gbps with
     no error). `env_aws.sh` therefore puts only `/opt/amazon/efa/lib` on the
     path; `sw/libfabric-1.22` is a debugging leftover, not a runtime component.

### Transport rebuild recipe (inside `salloc`, on a compute node)

```bash
# source tarball (kept in sw/src/nvshmem_src; original URL:)
# https://developer.download.nvidia.com/compute/redist/nvshmem/3.3.9/source/nvshmem_src_cuda12-all-all-3.3.9.tar.gz
srun --jobid=<id> -N1 -n1 -c 96 bash -c '
  source /home/ubuntu/workspace/flux/env_aws.sh
  cd /home/ubuntu/sw/src/nvshmem_src && rm -rf build
  cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/home/ubuntu/sw/nvshmem-3.3.9 \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 -DLIBFABRIC_HOME=/opt/amazon/efa \
    -DNVSHMEM_USE_GDRCOPY=1 -DGDRCOPY_HOME=/usr \
    -DNVSHMEM_IBGDA_SUPPORT=0 -DNVSHMEM_IBRC_SUPPORT=0 -DNVSHMEM_IBDEVX_SUPPORT=0 \
    -DNVSHMEM_UCX_SUPPORT=0 -DNVSHMEM_MPI_SUPPORT=0 -DNVSHMEM_SHMEM_SUPPORT=0 \
    -DNVSHMEM_PMIX_SUPPORT=0 -DNVSHMEM_BUILD_TESTS=0 -DNVSHMEM_BUILD_EXAMPLES=0 \
    -DNVSHMEM_BUILD_PYTHON_LIB=0
  cmake --build build -j 32 --target install'

# overlay into the wheel (the ONLY file swapped):
NV=/home/ubuntu/sw/venvs/flux/lib/python3.10/site-packages/nvidia/nvshmem
cp $NV/lib/nvshmem_transport_libfabric.so.3 $NV/lib/nvshmem_transport_libfabric.so.3.wheel-orig  # once
cp /home/ubuntu/sw/nvshmem-3.3.9/lib/nvshmem_transport_libfabric.so.3.0.0 \
   $NV/lib/nvshmem_transport_libfabric.so.3

# sanity: the overlaid transport must show FABRIC_1.8 among its version needs
readelf -V $NV/lib/nvshmem_transport_libfabric.so.3 | grep -A4 libfabric.so
```

**⚠ Any `pip install`/upgrade of `nvidia-nvshmem-cu12` silently clobbers the
overlay.** Re-run the two `cp` lines afterwards (no rebuild needed while the
wheel stays 3.3.9; a different wheel version needs a matching source rebuild).

### Runtime env (already in `env_aws.sh`; listed for auditability)

```bash
NVSHMEM_REMOTE_TRANSPORT=libfabric   # launch.sh default, kept
NVSHMEM_LIBFABRIC_PROVIDER=efa       # overrides launch.sh's ${...:-cxi} Slingshot default
FI_EFA_ENABLE_SHM_TRANSFER=0         # required by NVSHMEM on EFA
NVSHMEM_IB_ENABLE_IBGDA=0            # no IBGDA on EFA
NVSHMEM_DISABLE_GDRCOPY=0            # EFA requires gdrcopy (Perlmutter scripts say 1 — wrong here)
LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:/opt/amazon/efa/lib:...   # single (Amazon) libfabric
NVSHMEM_SYMMETRIC_SIZE=4G            # per-run, for large multi-node MoE configs
```

`launch.sh` needs no edits: its multi-node defaults are `${VAR:-...}` and the
exports above override them. `srun` propagates the sourced environment.

## Full environment rebuild from scratch

```bash
# 0. submodules (head node; FAST is private-SSH but the deploy key works)
git submodule update --init 3rdparty/nccl 3rdparty/cutlass
git submodule update --init 3rdparty/FAST   # optional, FAST baseline only

# 1. CUDA 12.4.1 toolkit (head node, no sudo)
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sh cuda_12.4.1_550.54.15_linux.run --silent --toolkit \
   --toolkitpath=/home/ubuntu/sw/cuda-12.4 --no-man-page

# 2. venv (head node; python3.10-venv was apt-installed on the head once)
python3 -m venv /home/ubuntu/sw/venvs/flux
source /home/ubuntu/sw/venvs/flux/bin/activate
pip install --upgrade pip setuptools wheel packaging
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy ninja cmake pynvml "cuda-python==12.4.0" nvidia-nvshmem-cu12==3.3.9
NV=$(python -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")
ln -sf libnvshmem_host.so.3 $NV/lib/libnvshmem_host.so   # setup.py links -lnvshmem_host

# 3. NVSHMEM transport rebuild + overlay  →  recipe above (compute node!)

# 4. flux build (compute node, writes to shared /home; ~20 min)
salloc --partition=a100 --nodes=2 --exclusive --time=04:00:00 --no-shell   # note job id
srun --jobid=<id> -N1 -n1 -c 96 bash -c \
  'source ./env_aws.sh && nproc=32 ./build.sh --arch 80 --sm-cores 108 --nvshmem --no_test --jobs 32'
```

Build notes:
- `build.sh` was patched to `PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH`;
  without `CUDA_HOME` exported it would silently use the AMI's nvcc 13.0.
- `nproc=32` matters: `build_nccl` runs `make -j${nproc}` (an env var, not
  `$(nproc)`) — unset means unbounded parallelism.
- The build can exit 1 *after* succeeding: the `merge_compile_commands` EXIT
  trap runs `ninja -f -t compdb`, which ninja ≥1.12 rejects. Check for
  `Successfully installed byte_flux` instead of trusting the exit code.
- gcc/g++ 11.4 (stock) is the host compiler; CUDA 12.4 supports gcc ≤ 13.

## Validation ladder (each step gates the next)

```bash
# 1. EFA fabric visible:            expect 4 'provider: efa' RDM endpoints
srun --jobid=<id> -N1 -n1 /opt/amazon/efa/bin/fi_info -p efa
# 2. single GPU (note: --input_dtype/--weight_dtype, not --dtype)
srun --jobid=<id> -N1 -n1 bash -c 'source ./env_aws.sh; CUDA_VISIBLE_DEVICES=0 \
  python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 --input_dtype float16 --weight_dtype float16'
# 3. 1 node, 8 GPUs, dense
srun --jobid=<id> -N1 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; \
  ./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10'
srun --jobid=<id> -N1 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; \
  ./launch.sh test/python/gemm_rs/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10'
# 4. 1 node MoE (8 ranks: T*E == 8)
srun --jobid=<id> -N1 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py'
srun --jobid=<id> -N1 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; ./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -T 8 -E 1'
# 5. NCCL-over-EFA: must log "NET/OFI Selected provider is efa" and
#    "Using network Libfabric" — "Using network Socket" means the plugin failed to load
NCCL_DEBUG=INFO ... any 2-node run
# 6. 2 nodes, 16 ranks, MoE over EFA (the goal)
srun --jobid=<id> -N2 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; \
  NVSHMEM_SYMMETRIC_SIZE=4G ./launch.sh test/python/moe_ag_scatter/test_moe_ag.py'
srun --jobid=<id> -N2 --ntasks-per-node=1 bash -c 'source ./env_aws.sh; \
  NVSHMEM_SYMMETRIC_SIZE=4G ./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -M 40960 -T 16 -E 1'
```

Expected: `✅ flux check passed` / `✅ flux and torch matches` on every rank
("not bitwise match" alongside "all close" is the normal outcome).
Multi-node constraints (16 ranks): token counts divisible by `16 * topk`;
`max_m / topk` divisible by 16.

## Slurm usage on this cluster

```bash
salloc --partition=a100 --nodes=2 --exclusive --time=04:00:00 --no-shell
squeue                      # note job id; nodes power up on demand (minutes)
srun --jobid=<id> ...       # all work, including builds
scancel <id>                # release (nodes scale down ~5 min later)
```

No `sbatch` (user preference). Redirect logs on the login side to somewhere
outside the repo; compute-node `/tmp` is discarded at scale-down.
