#!/bin/bash
# Build the stage-0 probe with the sglang-dev toolchain (CUDA 12.9, gcc 12, Cray MPI, NVSHMEM 3.2.5).
set +e
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1
module unload nccl nvshmem cudatoolkit 2>/dev/null
module load libfabric/1.20.1 gcc-native/12.3 cudatoolkit/12.9 nccl/2.24.3 nvshmem/3.2.5-1 >/dev/null 2>&1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
MPI_CFLAGS="-I$CRAY_MPICH_DIR/include"; MPI_LIBS="-L$CRAY_MPICH_DIR/lib -lmpi_gnu -Xlinker -rpath=$CRAY_MPICH_DIR/lib"
OUT=${OUT:-$PSCRATCH/workspace/andrewy/logs/p50/bin}; mkdir -p $OUT
cd "$(dirname "$0")"
$CUDA_HOME/bin/nvcc -O2 -std=c++17 -arch=sm_80 -rdc=true -ccbin g++ $MPI_CFLAGS -DUSE_NVSHMEM -I$NVSHMEM_HOME/include \
  probe_ce.cu -o $OUT/probe_ce -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lcudadevrt -lcuda $MPI_LIBS \
  -Xlinker -rpath=$NVSHMEM_HOME/lib -Xlinker -rpath=$CUDA_HOME/lib64 2>&1 | grep -v "warning #\|Remark\|^$" || true
rc=${PIPESTATUS[0]}; ls -la $OUT/probe_ce && echo "BUILD OK" || echo "BUILD FAILED rc=$rc"
