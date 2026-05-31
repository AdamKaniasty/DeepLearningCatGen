#!/usr/bin/env bash
# Exit 0 iff all artifacts required by presentation/PLAN.md are present.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
# shellcheck source=sizes.env
source scripts/slurm/sizes.env

PHASE="${PRESENTATION_PHASE:-sweep}"
if [ "$PHASE" = "refine" ]; then
  MIN_CAT_RUNS="${CATGEN_MIN_CAT_RUNS:-3}"
  MIN_MIXED_RUNS="${CATGEN_MIN_MIXED_RUNS:-1}"
  REFINE_ONLY=1
else
  MIN_CAT_RUNS="${CATGEN_MIN_CAT_RUNS:-10}"
  MIN_MIXED_RUNS="${CATGEN_MIN_MIXED_RUNS:-1}"
  REFINE_ONLY=0
fi
export REFINE_ONLY

fail=0

require_file() {
  local path="$1"
  local label="${2:-$path}"
  if [ -f "$path" ]; then
    echo "OK $label"
    return 0
  fi
  echo "MISSING $label"
  return 1
}

echo "=== global presentation artifacts ==="
require_file data/splits/manifest.json || fail=$((fail + 1))
if [ -f data/splits/manifest.json ]; then
  python3 -c "
import json, sys
m = json.load(open('data/splits/manifest.json'))
if 'mixed' not in m:
    print('MISSING data/splits/manifest.json mixed section')
    sys.exit(1)
print('OK data/splits/manifest.json has mixed splits')
" || fail=$((fail + 1))
fi
echo "=== check phase=$PHASE ==="
require_file presentation/figures/dataset_preview.png || fail=$((fail + 1))
require_file presentation/figures/fid_bar.png || fail=$((fail + 1))
require_file presentation/figures/compare_grid.png || fail=$((fail + 1))
require_file presentation/figures/ext_compare.png || fail=$((fail + 1))
if [ "$PHASE" = "refine" ]; then
  require_file presentation/figures/interp_dcgan.png || fail=$((fail + 1))
  require_file presentation/figures/interp_aae.png || fail=$((fail + 1))
fi
require_file reports/leaderboard.md || fail=$((fail + 1))
require_file reports/leaderboard.csv || fail=$((fail + 1))
require_file reports/report_bundle.md || fail=$((fail + 1))

echo "=== per-run artifacts ==="
cat_done=0
mixed_done=0
total_done=0

while read -r rid model is_mixed; do
  [ -n "$rid" ] || continue
  d="runs/$rid"
  total_done=$((total_done + 1))
  if [ "$is_mixed" = "1" ]; then
    mixed_done=$((mixed_done + 1))
  else
    cat_done=$((cat_done + 1))
  fi

  run_fail=0
  for f in manifest.json metrics.csv events.jsonl summary.md; do
    require_file "$d/$f" "$rid/$f" || run_fail=1
  done
  if compgen -G "$d/samples/epoch_*.png" > /dev/null; then
    echo "OK $rid/samples/epoch_*.png"
  else
    echo "MISSING $rid/samples/epoch_*.png"
    run_fail=1
  fi
  if [ -f "$d/checkpoints/last.ckpt" ] || compgen -G "$d/checkpoints/*.ckpt" > /dev/null; then
    echo "OK $rid/checkpoints"
  else
    echo "MISSING $rid/checkpoints"
    run_fail=1
  fi
  require_file "$d/eval/fid.json" "$rid/eval/fid.json" || run_fail=1
  require_file "$d/eval/quality.json" "$rid/eval/quality.json" || run_fail=1
  require_file "$d/eval/curves.png" "$rid/eval/curves.png" || run_fail=1

  case "$model" in
    dcgan)
      require_file "$d/eval/interpolation.png" "$rid/eval/interpolation.png" || run_fail=1
      ;;
    aae)
      if compgen -G "$d/samples/recon_epoch_*.png" > /dev/null; then
        echo "OK $rid/samples/recon_epoch_*.png"
      else
        echo "MISSING $rid/samples/recon_epoch_*.png"
        run_fail=1
      fi
      require_file "$d/eval/interpolation.png" "$rid/eval/interpolation.png" || run_fail=1
      ;;
    vqvae)
      require_file "$d/eval/codebook_hist.png" "$rid/eval/codebook_hist.png" || run_fail=1
      ;;
  esac

  if [ "$run_fail" -eq 0 ]; then
    echo "OK run $rid ($model, mixed=$is_mixed)"
  else
    fail=$((fail + 1))
  fi
done < <(python3 -c "
import json, os
from pathlib import Path
refine_only = os.environ.get('REFINE_ONLY', '0') == '1'
for d in sorted(Path('runs').iterdir()):
    if not d.is_dir() or d.name.startswith('_'):
        continue
    mp = d / 'manifest.json'
    if not mp.exists():
        continue
    m = json.loads(mp.read_text())
    if m.get('status') != 'done':
        continue
    tags = (m.get('config') or {}).get('tags', [])
    is_refine = 'refine' in tags
    if refine_only and not is_refine:
        continue
    if not refine_only and is_refine:
        continue
    split = (m.get('config') or {}).get('data', {}).get('split', '')
    mixed = 1 if 'mixed_train' in split else 0
    print(d.name, m.get('model', '?'), mixed)
")

echo "=== run counts ==="
echo "done total=$total_done cat=$cat_done mixed=$mixed_done (need cat>=$MIN_CAT_RUNS mixed>=$MIN_MIXED_RUNS)"
if [ "$cat_done" -lt "$MIN_CAT_RUNS" ]; then
  echo "MISSING cat-only runs: have $cat_done need $MIN_CAT_RUNS"
  fail=$((fail + 1))
fi
if [ "$mixed_done" -lt "$MIN_MIXED_RUNS" ]; then
  echo "MISSING mixed extension runs: have $mixed_done need $MIN_MIXED_RUNS"
  fail=$((fail + 1))
fi

if [ "$fail" -gt 0 ]; then
  echo "CHECK FAILED ($fail problems)"
  exit 1
fi
echo "CHECK PASSED ($total_done runs, presentation bundle complete)"
