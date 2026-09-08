#!/usr/bin/env bash
# CSCS ALPS (GH200: Grace aarch64 + H100, Slingshot-11/CXI) analogue of
# module.sh / env_aws.sh for the H100 weak-scaling lane
# (docs/handoff/35_h100_alps_weak_scaling.md). TEMPLATE: every line marked
# TODO(alps) is filled in by the session that brings the port up, from what it
# discovers on the cluster (handoff 35 §3). Keep the CONTRACT below intact:
#   * one CUDA toolkit whose nvcc major.minor == torch.version.cuda
#   * NVSHMEM 3.x with the libfabric transport (nvshmem_transport_libfabric.so.3)
#   * gcc <= the newest the toolkit supports (CUDA 12.4: gcc <= 13)
#   * arch: build --arch 90 --gen-arch 80 --sm-cores 132; run FLUX_ARCH_OVERRIDE=80

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with: source $0"
    exit 1
fi

# The clone this file lives in (never hardcode: the collaborator's path differs).
export FLUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- software stack ---------------------------------------------------------
# TODO(alps): pick ONE of the site mechanisms and delete the others.
#   (a) uenv:      uenv start --view=default <image>   (e.g. a pytorch or prgenv-gnu image)
#                  NOTE: `uenv start` opens a subshell; source this file INSIDE it, and
#                  put the same image on the allocation (`#SBATCH --uenv=` / `srun --uenv=`).
#   (b) modules:   module load <cuda> <gcc> <cray-mpich> <nvshmem> ...
#   (c) container: not covered here (the runner needs srun+torchrun on the host side).
# module load ...                                   # TODO(alps)

# Python environment with torch (CUDA build for aarch64), numpy, matplotlib.
# source "$FLUX_ROOT/figs/weak_scaling/h100/venv/bin/activate"   # TODO(alps) (venv INSIDE the clone, gitignored)

# CUDA toolkit (nvcc) — must match torch.version.cuda major.minor.
# export CUDA_HOME=/path/to/cuda                   # TODO(alps)
export CUDACXX="${CUDA_HOME:?TODO(alps): export CUDA_HOME first}/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

# NVSHMEM (NVSHMEM_HOME takes strict precedence over the pip wheel in setup.py/cpp_mod.py).
# export NVSHMEM_HOME=/path/to/nvshmem             # TODO(alps)  (module prefix, or
#   "$(python -c 'import nvidia.nvshmem;print(nvidia.nvshmem.__path__[0])')" for the pip wheel)
export LD_LIBRARY_PATH="${NVSHMEM_HOME:?TODO(alps): export NVSHMEM_HOME first}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Compilers.
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="9.0"
export CPATH="$FLUX_ROOT/3rdparty/nccl/build/local/include${CPATH:+:$CPATH}"

# ---- H100 port: run the sm80/V2 kernel space (natively compiled for sm_90a) ----
export FLUX_ARCH_OVERRIDE=80

# ---- transport: Slingshot-11 (same as Perlmutter; launch.sh defaults to these when SLURM_NNODES>1) ----
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-libfabric}
export NVSHMEM_LIBFABRIC_PROVIDER=${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}
# torch.distributed (NCCL) over Slingshot needs the libfabric NCCL plugin (aws-ofi-nccl)
# or it silently falls back to TCP sockets (slow topk allgather = timed plan_comm).
# export NCCL_NET_PLUGIN=ofi                        # TODO(alps): whatever the site documents
# export NCCL_DEBUG=INFO                            # first 2n run only: confirm "NET/OFI"/"Libfabric"

# ---- checks (fail loudly; the handoff's gate ladder depends on these) --------
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "WARNING: expected aarch64 (GH200); got $(uname -m) — is this really an ALPS GH200 node/login?"
fi
nvcc_ver=$(nvcc --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | awk '{print $2}')
torch_cuda=$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)
if [[ -z "$nvcc_ver" || -z "$torch_cuda" ]]; then
    echo "ERROR: nvcc ($nvcc_ver) or torch ($torch_cuda) not found on PATH"
    return 1
fi
if [[ "$nvcc_ver" != "$torch_cuda" ]]; then
    echo "ERROR: nvcc $nvcc_ver != torch.version.cuda $torch_cuda (cmake configure fails on the minor mismatch — Perlmutter drift lesson)"
    return 1
fi
if [[ ! -f "$NVSHMEM_HOME/include/nvshmem.h" ]]; then
    echo "ERROR: NVSHMEM header missing under NVSHMEM_HOME=$NVSHMEM_HOME"
    return 1
fi
if [[ ! -f "$NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3" ]]; then
    echo "ERROR: $NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3 missing — inter-node NVSHMEM over CXI will NOT work"
    return 1
fi
if [[ ! -f "$FLUX_ROOT/3rdparty/nccl/build/local/include/nccl.h" ]]; then
    echo "WARNING: bundled NCCL not built yet ($FLUX_ROOT/3rdparty/nccl/build/local); build.sh builds it"
fi

echo "COMET/Flux environment ready (ALPS / H100 / CXI)"
echo "  FLUX_ROOT:    $FLUX_ROOT"
echo "  Python:       $(command -v python)"
echo "  GCC:          $($CC --version | head -n 1)"
echo "  NVCC:         $(command -v nvcc) ($nvcc_ver)"
echo "  CUDA_HOME:    $CUDA_HOME"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
echo "  arch:         device sm90, FLUX_ARCH_OVERRIDE=$FLUX_ARCH_OVERRIDE (sm80/V2 kernel space)"
python -c 'import torch; print(f"  PyTorch:      {torch.__version__}, CUDA {torch.version.cuda}")'
