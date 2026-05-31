#!/usr/bin/env bash
# Eval a single run (sample + FID + quality + interpolate).
set -euo pipefail
RID="${1:?usage: run_eval_run.sh <run_id> [n_samples] [ref_split] [device]}"
N="${2:-500}"
REF="${3:-fid_ref_500.txt}"
DEVICE="${4:-auto}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi 2>/dev/null || true)}"
echo "=== eval $RID n=$N ref=$REF ==="
python -m catgen.sample --run-id "$RID" --n "$N" --device "$DEVICE"
python -m catgen.eval_fid --run-id "$RID" --ref-split "$REF"
python -m catgen.eval_quality --run-id "$RID" --device "$DEVICE" || true
python -m catgen.interpolate --run-id "$RID" --device "$DEVICE" || true
