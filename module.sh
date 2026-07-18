#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with: source $0"
    exit 1
fi

# Restore Perlmutter's standard module environment.
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh || return 1

# Remove compiler-dependent modules before changing GCC.
module unload nccl nvshmem cray-mpich cray-libsci cudatoolkit 2>/dev/null || true

# CUDA 12.4 requires GCC <= 13.
module load PrgEnv-gnu || return 1
module load gcc/12.2.0 || return 1
module load craype-x86-milan || return 1

# Perlmutter communication and CUDA stack.
module load cray-mpich || return 1
module load cudatoolkit/12.4 || return 1
module load nvshmem/3.2.5-1 || return 1
module load nccl/2.24.3 || return 1

# Python environment.
module load conda/Miniforge3-25.11.0-1 || return 1
conda activate "$PSCRATCH/conda_envs/andrewy-comet" || return 1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Flux build configuration.
export FLUX_ROOT="$HOME/workspace/changchen/andrewy/flux"
export CPATH="$FLUX_ROOT/3rdparty/nccl/build/local/include${CPATH:+:$CPATH}"

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export CUDA_HOME="$CUDATOOLKIT_HOME"
export TORCH_CUDA_ARCH_LIST="8.0"

# Verify the important dependencies.
if [[ ! -f "$FLUX_ROOT/3rdparty/nccl/build/local/include/nccl.h" ]]; then
    echo "ERROR: bundled NCCL header is missing:"
    echo "  $FLUX_ROOT/3rdparty/nccl/build/local/include/nccl.h"
    return 1
fi

if [[ ! -f "$NVSHMEM_HOME/include/nvshmem.h" ]]; then
    echo "ERROR: NVSHMEM header is missing under NVSHMEM_HOME=$NVSHMEM_HOME"
    return 1
fi

echo "COMET/Flux environment ready"
echo "  Python:       $(command -v python)"
echo "  GCC:          $($CC --version | head -n 1)"
echo "  NVCC:         $(command -v nvcc)"
echo "  CUDA_HOME:    $CUDA_HOME"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
echo "  NCCL CPATH:   $FLUX_ROOT/3rdparty/nccl/build/local/include"

python -c 'import torch; print(f"  PyTorch:      {torch.__version__}, CUDA {torch.version.cuda}")'

module list
