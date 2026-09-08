#!/bin/bash
# sbatch wrapper for ONE weak-scaling rung on CSCS ALPS (docs/handoff/35 §7).
# Use only if interactive salloc is impractical at this node count; the capsule
# is identical either way (the runner srun's inside this job via --jobid).
# Fill the TODO(alps) #SBATCH lines, then from the CLONE ROOT:
#   sbatch --export=ALL,FLUX_CLONE=$PWD[,SPECS="..."] figs/weak_scaling/h100/sbatch_rung.template.sh
#SBATCH -A TODO_alps_account
#SBATCH -p TODO_alps_partition        # or -q <qos>, -C gpu, --reservation=...
#SBATCH -N 2                          # TODO(alps): 2 | 4 | 8 | 16 | 32 (== the spec's nodes:)
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:45:00                   # TODO(alps): 32n -> 01:15:00 (smoke + perf)
#SBATCH -J h100_weak_rung
#SBATCH -o figs/weak_scaling/h100/logs/%x-%j.out   # relative to the submit dir = clone root
# #SBATCH --uenv=TODO_image:TODO_view # TODO(alps) if the site uses uenv

set -euo pipefail
cd "${FLUX_CLONE:?export FLUX_CLONE=<absolute clone path> (sbatch --export=ALL,FLUX_CLONE=\$PWD)}"
source ./env_alps.sh
# default: the perf spec for this job's node count; 32n runs the smoke FIRST
if [[ -z "${SPECS:-}" ]]; then
  if [[ "${SLURM_NNODES}" == "32" ]]; then
    SPECS="sweeps/specs/h100_weak32_smoke_k2.yaml sweeps/specs/h100_weak_32n_k2.yaml"
  else
    SPECS="sweeps/specs/h100_weak_${SLURM_NNODES}n_k2.yaml"
  fi
fi
for spec in $SPECS; do
  echo "=== $(date -u +%FT%TZ) rung start: $spec (job $SLURM_JOB_ID, $SLURM_NNODES nodes)"
  python sweeps/sweep.py run --spec "$spec" --jobid "$SLURM_JOB_ID"
  echo "=== $(date -u +%FT%TZ) rung done: $spec"
done
# capsules land under sweeps/results/runs/<run_id>_alps_*; commit them from the login node.
