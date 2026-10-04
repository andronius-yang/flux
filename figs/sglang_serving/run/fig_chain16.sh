#!/bin/bash
# fig_chain16.sh: on the 16n 80 GB allocation of job 59250973 (the 235B job, time limit raised to 2 h): run the 235B
# dataset, then the 30B dataset (Job A ported to 16n, for a fair 30B 4n-vs-16n comparison on 80 GB), then release.
J=59250973; EV=$PSCRATCH/workspace/andrewy/logs/p55/fig.log; HERE=$(cd "$(dirname "$0")" && pwd)
while true; do
  st=$(squeue -j $J -h -o %T 2>/dev/null)
  [ -z "$st" ] && { echo "$(date +%T) chain16: job $J left the queue before running" >> $EV; exit 1; }
  [ "$st" = RUNNING ] && break; sleep 30
done
echo "$(date +%T) chain16: job $J running; 235B first, then 30B" >> $EV
$HERE/fig_run_on.sh 235b16n $J
$HERE/fig_run_on.sh 30b16n_80g $J
scancel $J; echo "$(date +%T) chain16: scancel $J" >> $EV
