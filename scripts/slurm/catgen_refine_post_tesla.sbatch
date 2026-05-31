#!/bin/bash
#SBATCH --job-name=catgen-refine-post
#SBATCH --account=mi2lab-normal
#SBATCH --partition=experimental
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen
#SBATCH --output=slurm-logs/refine-post-%j.out
#SBATCH --error=slurm-logs/refine-post-%j.err

set -euo pipefail
export CATGEN_ROOT="$PWD"
export PRESENTATION_PHASE=refine
source scripts/slurm/job_env.sh
bash scripts/slurm/run_refine_post.sh
echo "=== Refine post done at $(date) ==="
