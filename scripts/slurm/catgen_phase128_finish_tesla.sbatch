#!/bin/bash
#SBATCH --job-name=catgen-p128-finish
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/phase128-finish-%j.out
#SBATCH --error=slurm-logs/phase128-finish-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh
bash scripts/slurm/run_phase128_finish.sh
echo "=== Phase128 finish done at $(date) ==="
