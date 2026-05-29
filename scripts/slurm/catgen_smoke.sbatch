#!/bin/bash
#SBATCH --job-name=catgen-smoke
#SBATCH --partition=mi2lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm_logs/smoke-%j.out
#SBATCH --error=slurm_logs/smoke-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
export CATGEN_SMOKE=1
source .venv/bin/activate
echo "node=$(hostname) date=$(date -Is)"
nvidia-smi -L || true
bash scripts/slurm/run_sweep.sh
