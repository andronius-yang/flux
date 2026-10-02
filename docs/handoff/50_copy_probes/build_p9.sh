#!/bin/bash
# build_p9.sh: plan-9 S0 probes (P9, P10, P11a) with the stage-0 toolchain (CUDA 12.9, NVSHMEM 3.2.5, Cray MPI).
set +e
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh >/dev/null 2>&1
module unload nccl nvshmem cudatoolkit 2>/dev/null
module load libfabric/1.20.1 gcc-native/12.3 cudatoolkit/12.9 nccl/2.24.3 nvshmem/3.2.5-1 >/dev/null 2>&1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
MATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9
MPI_CFLAGS="-I$CRAY_MPICH_DIR/include"; MPI_LIBS="-L$CRAY_MPICH_DIR/lib -lmpi_gnu -Xlinker -rpath=$CRAY_MPICH_DIR/lib"
OUT=${OUT:-$PSCRATCH/workspace/andrewy/logs/p50/bin}; mkdir -p $OUT
cd "$(dirname "$0")"
NV="$CUDA_HOME/bin/nvcc -O2 -std=c++17 -arch=sm_80 -ccbin g++"
$NV p11a_graph_k2.cu -o $OUT/p11a_graph_k2 -lcuda && echo "p11a OK"
$NV p9_push.cu -o $OUT/p9_push -I$MATH/targets/x86_64-linux/include -L$MATH/lib64 -lcublas -lcuda \
  -Xlinker -rpath=$MATH/lib64 -Xlinker -rpath=$CUDA_HOME/lib64 && echo "p9 OK"
$NV -rdc=true $MPI_CFLAGS -I$NVSHMEM_HOME/include p10_dev_put.cu -o $OUT/p10_dev_put -L$NVSHMEM_HOME/lib \
  -lnvshmem_host -lnvshmem_device -lcudadevrt -lcuda $MPI_LIBS -Xlinker -rpath=$NVSHMEM_HOME/lib \
  -Xlinker -rpath=$CUDA_HOME/lib64 2>&1 | grep -v "warning #\|Remark\|^$"; ls $OUT/p10_dev_put >/dev/null 2>&1 && echo "p10 OK"
$NV -rdc=true $MPI_CFLAGS -I$NVSHMEM_HOME/include p12_barrier.cu -o $OUT/p12_barrier -L$NVSHMEM_HOME/lib \
  -lnvshmem_host -lnvshmem_device -lcudadevrt -lcuda $MPI_LIBS -Xlinker -rpath=$NVSHMEM_HOME/lib \
  -Xlinker -rpath=$CUDA_HOME/lib64 2>&1 | grep -v "warning #\|Remark\|^$"; ls $OUT/p12_barrier >/dev/null 2>&1 && echo "p12 OK"
NCCLD=${NCCL_HOME:-$NCCL_DIR}
$NV -rdc=true $MPI_CFLAGS -I$NVSHMEM_HOME/include -I$NCCLD/include p13_allgather.cu -o $OUT/p13_allgather -L$NVSHMEM_HOME/lib \
  -L$NCCLD/lib -lnccl -lnvshmem_host -lnvshmem_device -lcudadevrt -lcuda $MPI_LIBS -Xlinker -rpath=$NVSHMEM_HOME/lib \
  -Xlinker -rpath=$NCCLD/lib -Xlinker -rpath=$CUDA_HOME/lib64 2>&1 | grep -v "warning #\|Remark\|^$"; ls $OUT/p13_allgather >/dev/null 2>&1 && echo "p13 OK"
$NV -rdc=true $MPI_CFLAGS -I$NVSHMEM_HOME/include -I$NCCLD/include p14_p2p_bw.cu -o $OUT/p14_p2p_bw -L$NVSHMEM_HOME/lib \
  -L$NCCLD/lib -lnccl -lnvshmem_host -lnvshmem_device -lcudadevrt -lcuda $MPI_LIBS -Xlinker -rpath=$NVSHMEM_HOME/lib \
  -Xlinker -rpath=$NCCLD/lib -Xlinker -rpath=$CUDA_HOME/lib64 2>&1 | grep -v "warning #\|Remark\|^$"; ls $OUT/p14_p2p_bw >/dev/null 2>&1 && echo "p14 OK"
