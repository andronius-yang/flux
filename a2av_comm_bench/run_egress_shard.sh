#!/bin/bash
# Egress NIC-sharding physics check for the WeightPushMulticast sharding design.
# Run inside a Slurm allocation spanning >= 2 nodes:
#   salloc --qos interactive -C gpu --account m5350_g -N 2 --gpus-per-node=4 \
#     bash a2av_comm_bench/run_egress_shard.sh
# One hot-expert home per node pushes <msg> bytes to the next node's same-lr
# rank. SHARD_DIRECT=1 is the single-NIC baseline; sharded cases split the leg
# across SHARD_L same-local-rank wires with dest-side NVLink reassembly.
# Questions: does SL=4 approach ~4x the single-NIC push; where is the
# shard-size knee (threshold knob); does per-chunk pipelining of the NVLink
# stage vs the NIC push pay? Derive leg GB/s as msg_MB / med_ms of the
# slowest rank (the dest's arrival wait dominates).
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

i=200
run_case() {  # msg SL chunks_per_shard nhomes direct
  local MSG=$1 SL=$2 CHUNKS=$3 NHOMES=$4 DIRECT=$5
  i=$((i + 1))
  rm -f "$UIDDIR/uid_es$i"
  local SHARD=$(((MSG + SL - 1) / SL))
  local CHUNK=$(((SHARD + CHUNKS - 1) / CHUNKS))
  echo "=== EGRESS_SHARD msg=$MSG SL=$SL chunks/shard=$CHUNKS nhomes=$NHOMES direct=$DIRECT ==="
  SHARD_L=$SL SHARD_CHUNK_BYTES=$CHUNK SHARD_NHOMES=$NHOMES SHARD_DIRECT=$DIRECT \
  srun --nodes=$SLURM_NNODES --ntasks-per-node=$GPUS_PER_NODE \
    --gpus-per-node=$GPUS_PER_NODE --export=ALL \
    ./comm_bench "$UIDDIR/uid_es$i" egress_shard "$MSG" "$ITERS"
}

# msg axis: 2/8/32/64 MiB (32 MiB = one fc1 expert shard; 64 = w1+w2)
for MSG in 2097152 8388608 33554432 67108864; do
  run_case $MSG 4 1 1 1     # single-NIC direct baseline
  run_case $MSG 1 1 1 0     # SL=1 sanity (fast-path only, ~= direct)
  run_case $MSG 2 1 1 0     # 2-NIC
  run_case $MSG 4 1 1 0     # 4-NIC, whole-shard (no pipelining)
  run_case $MSG 4 4 1 0     # 4-NIC, 4 chunks/shard (stage/push pipelined)
  run_case $MSG 4 16 1 0    # 4-NIC, 16 chunks/shard
done
# contention case: every rank is a home (saturation — sharding should be ~flat)
run_case 33554432 4 4 4 0
run_case 33554432 4 1 4 1
