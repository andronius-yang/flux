#!/bin/bash
# run_p6.sh: probe P6 on one GPU of the current node, LAZY loading, every mode, both cold-kernel placements.
B=$PSCRATCH/workspace/andrewy/logs/p50/bin
export CUDA_MODULE_LOADING=LAZY CUDA_VISIBLE_DEVICES=${CVD:-0}
for exe in p6_lazy p6_lazy_so; do
  for m in host late drv rt; do
    timeout 30 $B/$exe $m 2>&1 | grep -E "^P6|RESULT|error|failed"
  done
done
