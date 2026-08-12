#!/bin/bash
# Phase 0 saturation check for the getmem weight pull (WeightPrefetchGetmem).
# Run inside a Slurm allocation spanning >= 2 nodes:
#   salloc --qos interactive -C gpu --account m4243_g -N 2 --gpus-per-node=4 \
#     bash a2av_comm_bench/run_prefetch.sh
# Every rank pulls <msg> bytes from the next node's same-local rank (1 ingress
# + 1 egress per NIC). Targets: NCCL batch_isend_irecv prefetch ~12 GB/s
# (capsule 20260808-032217: 32 MiB in ~2.7 ms); Slingshot NIC line rate
# ~25 GB/s/GPU. Derive GB/s as wire_MB / med_ms (RESULT lines).
set -e
cd "$(dirname "$0")"
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=cxi
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-2G}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# libnvshmem_host.so needs a newer libstdc++ than the Cray default
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

UIDDIR=${UIDDIR:-$PSCRATCH/a2av_comm_bench_uids}   # must be a shared FS
mkdir -p "$UIDDIR"
ITERS=${ITERS:-20}

i=100
run_case() {  # msg impl chunks intra
  local MSG=$1 IMPL=$2 CHUNKS=$3 INTRA=$4
  i=$((i + 1))
  rm -f "$UIDDIR/uid_p$i"
  local CHUNK=$((MSG / CHUNKS))
  echo "=== PREFETCH msg=$MSG impl=$IMPL chunks=$CHUNKS intra=$INTRA ==="
  PREFETCH_IMPL=$IMPL PREFETCH_CHUNK_BYTES=$CHUNK PREFETCH_NBLOCKS=$CHUNKS \
  PREFETCH_INTRA=$INTRA \
  srun --nodes=$SLURM_NNODES --ntasks-per-node=$GPUS_PER_NODE \
    --gpus-per-node=$GPUS_PER_NODE --export=ALL \
    ./comm_bench "$UIDDIR/uid_p$i" prefetch "$MSG" "$ITERS"
}

# cross-node: message size x impl x chunking (32 MiB = one fc1 expert matrix)
for MSG in 4194304 16777216 33554432 67108864; do
  for IMPL in kernel stream; do
    for CHUNKS in 1 4 16 64; do
      run_case $MSG $IMPL $CHUNKS 0
    done
  done
done
# NVLink intra-node baseline at the real expert size
run_case 33554432 kernel 16 1
run_case 33554432 stream 1 1
