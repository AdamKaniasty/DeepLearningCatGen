#!/usr/bin/env bash
# Build all presentation artifacts (PLAN.md) after training + eval jobs complete.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

MIXED_TRAIN_N=$((2 * CATGEN_MIXED_TRAIN_PER_CLASS))

bash scripts/slurm/link_data.sh

echo "=== dataset preview ==="
python scripts/preview_dataset.py \
  --cat-split "train_${CATGEN_TRAIN_N}.txt" \
  --dog-split "mixed_train_${MIXED_TRAIN_N}.txt" \
  --n-cats 16 \
  --n-dogs 8

echo "=== per-run curves ==="
python scripts/plot_metrics.py

echo "=== cross-run rollup ==="
python -m catgen.leaderboard
python scripts/build_figures.py
python scripts/build_report.py

echo "=== verify presentation checklist ==="
bash scripts/slurm/check_presentation_artifacts.sh

echo "=== finalize complete ==="
