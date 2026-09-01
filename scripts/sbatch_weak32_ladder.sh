#!/bin/bash
# 32n weak-scaling LADDER job (2026-08-31): canon OURS s1 arm (pv2_r2), K2,
# b1-b64 (6 budgets), perf lane (correctness carried by the smoke capsule).
# Queued concurrently with sbatch_weak32_smoke.sh — scancel this job if the
# smoke lands red. -t 40 caps a full-job wedge at ~21 nh; per-cell
# timeout_s=420 + retries=1 in the spec.
# Submit from the repo root: sbatch scripts/sbatch_weak32_ladder.sh
#SBATCH -A m5350_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 32
#SBATCH --gpus-per-node=4
#SBATCH -t 15
#SBATCH -J weak32_ladder
#SBATCH -o /pscratch/sd/y/yufeid/workspace/andrewy/sweep_data/slurm_32n/%x-%j.out

set -euo pipefail
cd /global/u1/y/yufeid/workspace/changchen/andrewy/flux
source ./module.sh
echo "=== weak32_ladder job $SLURM_JOB_ID on $(hostname), $(date -u +%FT%TZ) ==="
python3 sweeps/sweep.py run \
    --spec sweeps/specs/weak32_ladder_k2.yaml \
    --jobid="$SLURM_JOB_ID"
echo "=== weak32_ladder done, $(date -u +%FT%TZ) ==="
