#!/bin/bash
# 32n weak-scaling SMOKE job (2026-08-31): first-ever 32-node run. Canon OURS
# s1 arm (pv2_r2), K2, b1+b64 extremes, correctness ON. -t 20 caps a full-job
# wedge at ~10.7 nh; per-cell timeout_s=420 in the spec keeps one wedged cell
# from eating its sibling. Deliberately sbatch (user ruling this campaign):
# multi-day 32n queue wait must not hold an interactive session open.
# Submit from the repo root: sbatch scripts/sbatch_weak32_smoke.sh
#SBATCH -A m5350_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -N 32
#SBATCH --gpus-per-node=4
#SBATCH -t 5
#SBATCH -J weak32_dwire_b1
#SBATCH -o /pscratch/sd/y/yufeid/workspace/andrewy/logs/slurm/%x-%j.out

set -euo pipefail
cd /global/u1/y/yufeid/workspace/changchen/andrewy/flux
source ./module.sh
echo "=== weak32_dwire_b1 job $SLURM_JOB_ID on $(hostname), $(date -u +%FT%TZ) ==="
python3 sweeps/sweep.py run \
    --spec sweeps/specs/weak32_dwire_b1_k2.yaml \
    --jobid="$SLURM_JOB_ID"
echo "=== weak32_dwire_b1 done, $(date -u +%FT%TZ) ==="
