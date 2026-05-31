#!/bin/bash
#SBATCH --job-name=catgen-finalize
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/finalize-%j.out
#SBATCH --error=slurm-logs/finalize-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh

echo "=== Job Information ==="
echo "Node: $(hostname)"
echo "PWD: $PWD"
echo "Date: $(date)"

bash scripts/slurm/run_finalize.sh
echo "=== Finalize completed at $(date) ==="
