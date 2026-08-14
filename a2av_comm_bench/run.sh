#!/bin/bash
# Run inside a Slurm allocation:
#   salloc --qos interactive -C gpu --account m4243_g -N 4 --gpus-per-node=4 \
#     bash a2av_comm_bench/run.sh
# Compares the two transports AS IMPLEMENTED in src: a2av_ring dispatch
# (traffic-matrix alltoallv) vs all_gather_all2all (two-level allgather).
set -e
cd "$(dirname "$0")"
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=cxi
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-2G}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# libnvshmem_host.so needs a newer libstdc++ than the Cray default; LD_PRELOAD
# outranks the binary's RPATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

MDIR=${MDIR:-$HOME/workspace/changchen/andrewy/profiling/traffic/matrices/a2av/4n_16r}
UIDDIR=${UIDDIR:-$PSCRATCH/a2av_comm_bench_uids}   # must be a shared FS
mkdir -p "$UIDDIR"
ITERS=${ITERS:-20}
# allgather shard bytes per rank = tokens_per_rank * hidden * 2 (bf16)
declare -A SHARD=( [2mib]=524288 [16mib]=4194304 [64mib]=16777216 )

i=0
for B in 2mib 16mib 64mib; do
  for CASE in "a2av $MDIR/$B/a2av_4n_16r_dist_001.txt" "ag ${SHARD[$B]}"; do
    i=$((i + 1))
    rm -f "$UIDDIR/uid_$i"
    echo "=== BUDGET=$B CASE=$CASE ==="
    srun --nodes=$SLURM_NNODES --ntasks-per-node=$GPUS_PER_NODE \
      --gpus-per-node=$GPUS_PER_NODE --export=ALL \
      ./comm_bench "$UIDDIR/uid_$i" $CASE $ITERS
  done
done
