#!/bin/bash
# Self-contained CPU-node build (no GPU needed to compile). Logs go to
# $PSCRATCH/workspace/andrewy/logs/slurm (project log root; note: Lustre, unreachable during outages).
#SBATCH -A m5350
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=120G
#SBATCH -t 40
#SBATCH -J flux_build
#SBATCH -o /pscratch/sd/y/yufeid/workspace/andrewy/logs/slurm/%x-%j.out
set -euo pipefail
bash /global/u1/y/yufeid/workspace/changchen/andrewy/flux/scripts/build_pinned_cuda12.4.sh
