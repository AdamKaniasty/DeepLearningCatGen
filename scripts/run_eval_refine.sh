#!/usr/bin/env bash
# Eval only runs whose config has tags: [refine].
set -euo pipefail
N="${1:-500}"
REF="${2:-fid_ref_500.txt}"
EXT_REF="${3:-mixed_ref_400.txt}"
DEVICE="${4:-auto}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi 2>/dev/null || true)}"

is_refine() {
  python3 -c "import json,sys; m=json.load(open('$1')); sys.exit(0 if 'refine' in (m.get('config') or {}).get('tags',[]) else 1)"
}

for d in runs/*/; do
  rid="$(basename "$d")"
  case "$rid" in _*) continue;; esac
  [ -f "$d/manifest.json" ] || continue
  is_refine "$d/manifest.json" || continue
  status=$(python3 -c "import json; print(json.load(open('$d/manifest.json'))['status'])")
  [ "$status" = "done" ] || continue
  ref_split="$REF"
  if python3 -c "import json; print('mixed' in json.load(open('$d/manifest.json')).get('config',{}).get('data',{}).get('split',''))" | grep -q True; then
    ref_split="$EXT_REF"
  fi
  echo "=== eval refine $rid ref=$ref_split ==="
  python -m catgen.sample --run-id "$rid" --n "$N" --device "$DEVICE"
  python -m catgen.eval_fid --run-id "$rid" --ref-split "$ref_split"
  python -m catgen.eval_quality --run-id "$rid" --device "$DEVICE" || true
  python -m catgen.interpolate --run-id "$rid" --device "$DEVICE" || true
done
python -m catgen.leaderboard
