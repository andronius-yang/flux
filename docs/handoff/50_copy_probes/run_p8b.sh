#!/bin/bash
# run_p8b.sh: probe P8 with the producer streams at the highest priority (P8_PRIO=hp), with and without stream pools.
B=$PSCRATCH/workspace/andrewy/logs/p50/bin
export CUDA_VISIBLE_DEVICES=${CVD:-0}
for pool in 0 40; do
  for n in 1 8 24 32; do
    for m in dep_kernel dep_wgeq fe_wait full_dep_wgeq; do
      P8_PRIO=hp P8_NSTREAMS=$pool CUDA_DEVICE_MAX_CONNECTIONS=$n timeout 30 $B/p8_deporder $m 2>&1 | grep -E "^P8|error|failed"
    done
  done
done
for n in 8 24 32; do
  for m in dep_kernel dep_wgeq fe_wait; do
    P8_NSTREAMS=40 CUDA_DEVICE_MAX_CONNECTIONS=$n timeout 30 $B/p8_deporder $m 2>&1 | grep -E "^P8|error|failed"
  done
done
