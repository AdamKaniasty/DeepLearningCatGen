#!/bin/bash
#SBATCH --job-name=catgen-partc
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/partc-%j.out
#SBATCH --error=slurm-logs/partc-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh
bash scripts/slurm/run_partc.sh
echo "=== Part C done at $(date) ==="
