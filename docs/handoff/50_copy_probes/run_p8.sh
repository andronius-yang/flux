#!/bin/bash
# run_p8.sh: probe P8 (p8_deporder.cu) on one GPU of the current node, every case, one and 32 hardware queues.
B=$PSCRATCH/workspace/andrewy/logs/p50/bin
export CUDA_VISIBLE_DEVICES=${CVD:-0}
for n in 1 32; do
  for m in base dep_kernel dep_event dep_wgeq fe_wait pre12_dep write_only \
           full_base full_dep_kernel full_dep_event full_dep_wgeq full_pre12_dep; do
    CUDA_DEVICE_MAX_CONNECTIONS=$n timeout 30 $B/p8_deporder $m 2>&1 | grep -E "^P8|error|failed"
  done
done
