#!/usr/bin/env bash
set -euo pipefail
N="${1:-1000}"
REF="${2:-fid_ref_1000.txt}"
DEVICE="${3:-auto}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi 2>/dev/null || true)}"
for d in runs/*/; do
  rid="$(basename "$d")"
  case "$rid" in _*) continue;; esac
  [ -f "$d/manifest.json" ] || continue
  status=$(python -c "import json,sys;print(json.load(open('$d/manifest.json'))['status'])")
  [ "$status" = "done" ] || continue
  echo "=== $rid ==="
  python -m catgen.sample --run-id "$rid" --n "$N" --device "$DEVICE"
  python -m catgen.eval_fid --run-id "$rid" --ref-split "$REF"
  python -m catgen.eval_quality --run-id "$rid" --device "$DEVICE" || true
  python -m catgen.interpolate --run-id "$rid" --device "$DEVICE" || true
done
python -m catgen.leaderboard
