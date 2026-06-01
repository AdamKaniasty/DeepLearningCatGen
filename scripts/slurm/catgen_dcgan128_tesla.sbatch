#!/bin/bash
#SBATCH --job-name=catgen-dcgan128
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/dcgan128-%j.out
#SBATCH --error=slurm-logs/dcgan128-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
source scripts/slurm/job_env.sh
bash scripts/slurm/run_dcgan128.sh
echo "=== DCGAN 128 done at $(date) ==="
