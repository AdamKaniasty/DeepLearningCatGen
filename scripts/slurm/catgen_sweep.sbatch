#!/bin/bash
#SBATCH --job-name=catgen-sweep
#SBATCH --partition=mi2lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm_logs/sweep-%j.out
#SBATCH --error=slurm_logs/sweep-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source .venv/bin/activate
echo "node=$(hostname) date=$(date -Is) cuda=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true
bash scripts/slurm/run_sweep.sh
