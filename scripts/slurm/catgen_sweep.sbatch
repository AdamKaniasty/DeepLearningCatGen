#!/bin/bash
#SBATCH --job-name=catgen-sweep
#SBATCH --account=mi2lab-normal
#SBATCH --partition=mi2lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/sweep-%j.out
#SBATCH --error=slurm-logs/sweep-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh

echo "=== Job Information ==="
echo "Node: $(hostname)"
echo "PWD: $PWD"
echo "Date: $(date)"
nvidia-smi -L || true

bash scripts/slurm/run_sweep.sh
echo "=== Job completed at $(date) ==="
