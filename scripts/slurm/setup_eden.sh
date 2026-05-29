#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/evafs/groups/mi2lab/akaniasty
REPO=DeepLearningCatGen
URL="${GIT_REPO_URL:-https://github.com/AdamKaniasty/DeepLearningCatGen.git}"
cd "$BASE"
if [ ! -d "$REPO" ]; then
  git clone "$URL"
fi
cd "$REPO"
git pull --ff-only
mkdir -p slurm-logs "$BASE/tmp"
export CATGEN_ROOT="$PWD"
bash scripts/slurm/link_data.sh
echo "setup done (deps install on first sbatch via uv sync): $PWD"
