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
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
mkdir -p slurm_logs reports presentation/figures
export CATGEN_ROOT="$PWD"
bash scripts/slurm/link_data.sh
echo "setup done: $PWD"
