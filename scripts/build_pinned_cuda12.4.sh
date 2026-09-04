#!/bin/bash
# Pinned CUDA 12.4 incremental rebuild (perlmutter cudatoolkit drift workaround).
# Run on a compute node: login-node arbiter kills heavy nvcc units. See
# scripts/sbatch_build_cpu.sh. FLUX_BUILD_SKIP_CMAKE=1 => build/CMakeCache must exist.
cd /global/u1/y/yufeid/workspace/changchen/andrewy/flux
source ./module.sh

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDACXX=$CUDA_HOME/bin/nvcc
PATH_FILTERED=$(echo "$PATH" | tr ':' '\n' | grep -v 'hpc_sdk/Linux_x86_64/25.5' | paste -sd:)
export PATH="$CUDA_HOME/bin:$PATH_FILTERED"
CPATH_FILTERED=$(echo "${CPATH:-}" | tr ':' '\n' | grep -v 'hpc_sdk/Linux_x86_64/25.5' | paste -sd:)
export CPATH="$CUDA_HOME/include:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/include:$CPATH_FILTERED"
export LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/lib64:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"

echo "=== rebuild start $(date -u +%FT%TZ), nvcc: $(command -v nvcc)"
nvcc --version | tail -2
FLUX_BUILD_SKIP_CMAKE=1 nproc=4 ./build.sh --arch 80 --sm-cores 108 --nvshmem --no_test --jobs 4
echo "=== rebuild done $(date -u +%FT%TZ)"
ls -la python/flux/lib/libflux_cuda.so
