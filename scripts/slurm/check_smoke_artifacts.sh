#!/usr/bin/env bash
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
n=0
ok=0
for d in runs/*/; do
  rid="$(basename "$d")"
  case "$rid" in _*) continue;; esac
  m="$d/manifest.json"
  [ -f "$m" ] || continue
  st=$(python -c "import json; print(json.load(open('$m'))['status'])")
  [ "$st" = "done" ] || continue
  n=$((n + 1))
  miss=0
  for f in manifest.json metrics.csv events.jsonl summary.md; do
    [ -f "$d/$f" ] || { echo "MISSING $rid/$f"; miss=1; }
  done
  [ -d "$d/samples" ] || { echo "MISSING $rid/samples"; miss=1; }
  [ -d "$d/checkpoints" ] || { echo "MISSING $rid/checkpoints"; miss=1; }
  [ -f "$d/eval/fid.json" ] || echo "WARN $rid/eval/fid.json"
  [ "$miss" -eq 0 ] && ok=$((ok + 1)) && echo "OK $rid"
done
echo "done runs: $n (artifacts ok: $ok)"
[ -f reports/leaderboard.md ] && echo "OK reports/leaderboard.md" || echo "MISSING reports/leaderboard.md"
[ "$n" -ge 3 ] && [ "$ok" -ge 3 ]
