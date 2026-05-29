#!/bin/bash
#SBATCH --job-name=catgen-smoke
#SBATCH --account=mi2lab-normal
#SBATCH --partition=mi2lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/smoke-%j.out
#SBATCH --error=slurm-logs/smoke-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
export CATGEN_SMOKE=1
source scripts/slurm/job_env.sh

echo "=== Job Information ==="
echo "Node: $(hostname)"
echo "PWD: $PWD"
echo "Date: $(date)"
nvidia-smi -L || true

bash scripts/slurm/run_sweep.sh
echo "=== Job completed at $(date) ==="
