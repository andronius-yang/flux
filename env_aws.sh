#!/usr/bin/env bash
# AWS ParallelCluster (p4d.24xlarge, 8x A100 + EFA) analogue of module.sh.
# Keeps the validated Perlmutter software contract (CUDA 12.4, torch 2.6.0+cu124,
# NVSHMEM 3.3.9, sm80) and swaps only the transport layer: Slingshot/CXI -> EFA.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with: source $0"
    exit 1
fi

SW=/home/ubuntu/sw
export FLUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUDA 12.4 (validated stack; the AMI's /usr/local/cuda is 13.0 -- do not use).
export CUDA_HOME=$SW/cuda-12.4
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH

# Python environment.
source $SW/venvs/flux/bin/activate || return 1

# NVSHMEM (pip wheel; NVSHMEM_HOME takes strict precedence in setup.py/cpp_mod.py).
export NVSHMEM_HOME="$(python -c 'import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])')" || return 1
# Single libfabric per process: Amazon's 2.4 (aws-ofi-nccl requires its FABRIC_1.8
# symbols). The NVSHMEM transport overlaid into the wheel is built against these
# same headers — 1.x-header clients segfault in 2.4's fi_getinfo compat path.
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Compilers (CUDA 12.4 requires GCC <= 13; stock 11.4 is supported).
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="8.0"
export CPATH="$FLUX_ROOT/3rdparty/nccl/build/local/include${CPATH:+:$CPATH}"

# EFA / libfabric (replaces Perlmutter Slingshot/CXI). launch.sh's multi-node
# defaults are ${VAR:-...}, so these exports override its cxi default untouched.
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export FI_EFA_ENABLE_SHM_TRANSFER=0     # required by NVSHMEM on EFA
export NVSHMEM_IB_ENABLE_IBGDA=0        # no IBGDA on EFA
export NVSHMEM_DISABLE_GDRCOPY=0        # EFA provider REQUIRES gdrcopy (present on compute AMI)
# export NVSHMEM_SYMMETRIC_SIZE=4G      # per-run, for large multi-node MoE configs
# export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=<iface>  # first knob if 2-node bootstrap hangs

# Verify the important dependencies.
if [[ "$(nvcc --version 2>/dev/null)" != *"release 12.4"* ]]; then
    echo "ERROR: nvcc is not CUDA 12.4 (got: $(command -v nvcc))"
    return 1
fi

if [[ ! -f "$NVSHMEM_HOME/include/nvshmem.h" ]]; then
    echo "ERROR: NVSHMEM header is missing under NVSHMEM_HOME=$NVSHMEM_HOME"
    return 1
fi

if [[ ! -f "$NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3" ]]; then
    echo "WARNING: $NVSHMEM_HOME/lib/nvshmem_transport_libfabric.so.3 is missing;"
    echo "         NVSHMEM internode over EFA will NOT work."
fi

if [[ ! -f "$FLUX_ROOT/3rdparty/nccl/build/local/include/nccl.h" ]]; then
    echo "WARNING: bundled NCCL not built yet ($FLUX_ROOT/3rdparty/nccl/build/local);"
    echo "         run build.sh before compiling against it."
fi

echo "COMET/Flux environment ready (AWS/EFA)"
echo "  Python:       $(command -v python)"
echo "  GCC:          $($CC --version | head -n 1)"
echo "  NVCC:         $(command -v nvcc)"
echo "  CUDA_HOME:    $CUDA_HOME"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
echo "  NCCL CPATH:   $FLUX_ROOT/3rdparty/nccl/build/local/include"

python -c 'import torch; print(f"  PyTorch:      {torch.__version__}, CUDA {torch.version.cuda}")'
