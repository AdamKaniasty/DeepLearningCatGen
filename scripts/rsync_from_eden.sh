#!/usr/bin/env bash
# Pull runs + reports + figures from eden (skip heavy per-epoch checkpoints).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${CATGEN_EDEN_REMOTE:-eden:/mnt/evafs/groups/mi2lab/akaniasty/DeepLearningCatGen}"
EXCLUDE=(--exclude='checkpoints/epoch_*.ckpt')

rsync -avz "${EXCLUDE[@]}" "$REMOTE/runs/" "$ROOT/runs/"
rsync -avz "$REMOTE/reports/" "$ROOT/reports/"
rsync -avz "$REMOTE/presentation/figures/" "$ROOT/presentation/figures/"
echo "synced runs/ (no epoch_*.ckpt), reports/, presentation/figures/"
