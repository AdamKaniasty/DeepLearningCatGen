#!/bin/bash
#SBATCH --job-name=catgen-ext
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/ext-%j.out
#SBATCH --error=slurm-logs/ext-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh

echo "=== Job Information ==="
echo "Node: $(hostname)"
echo "PWD: $PWD"
echo "Date: $(date)"
nvidia-smi -L || true

bash scripts/slurm/run_extension.sh
echo "=== Extension job completed at $(date) ==="
