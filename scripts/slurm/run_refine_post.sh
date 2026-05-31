#!/usr/bin/env bash
# After run_refine.sh: eval refine runs + presentation figures (refine-only).
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

bash scripts/run_eval_refine.sh "$CATGEN_FID_N" "fid_ref_${CATGEN_REF_N}.txt" "mixed_ref_$((2 * CATGEN_MIXED_REF_PER_CLASS)).txt" "${CATGEN_DEVICE:-cuda}"
python scripts/plot_metrics.py
PRESENTATION_PHASE=refine python scripts/build_figures.py
python scripts/build_report.py
PRESENTATION_PHASE=refine bash scripts/slurm/check_presentation_artifacts.sh
