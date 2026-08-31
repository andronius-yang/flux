#!/bin/bash
# self-driving 16n dov A/B lane: salloc (blocks until grant) -> run both
# models -> scancel. Jobid parsed from THIS salloc's own output only
# (slurm-jobid-capture convention).
cd /pscratch/sd/y/yufeid/workspace/andrewy/flux-dov || exit 1
source ./module.sh >/dev/null 2>&1
# the sweep's capability probe must grep THIS worktree's .so, not
# whatever tree the shared conda editable install currently points at
export PYTHONPATH="/pscratch/sd/y/yufeid/workspace/andrewy/flux-dov/python${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$DOV_JOBID" ]; then
  JOBID="$DOV_JOBID"
else
  salloc -q regular -C gpu -N 16 --gpus-per-node=4 -t 75 -A m5350_g --no-shell > salloc_16n.log 2>&1
  JOBID=$(grep -oE "Granted job allocation [0-9]+" salloc_16n.log | grep -oE "[0-9]+")
  if [ -z "$JOBID" ]; then echo "16N_SALLOC_FAILED"; cat salloc_16n.log; exit 1; fi
fi
echo "16N_GRANTED $JOBID"
python sweeps/sweep.py run --spec sweeps/specs/dov_ab_16n_k2.yaml --jobid $JOBID
python sweeps/sweep.py run --spec sweeps/specs/dov_ab_16n_qwen.yaml --jobid $JOBID
echo "AB16N_DONE"
scancel $JOBID
echo "16N_RELEASED $JOBID"
