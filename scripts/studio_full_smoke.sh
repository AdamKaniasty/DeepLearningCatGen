#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi 2>/dev/null || true)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

echo "==== 0. wipe + fake data ===="
rm -rf runs reports presentation/figures/{compare_grid.png,fid_bar.png,ext_compare.png} data/raw data/splits
python scripts/prepare_data.py --fake 100 --fake-dogs 60 \
  --train-n 60 --ref-n 20 --mixed-train 20 --mixed-ref 10

echo "==== 1. preview dataset ===="
python scripts/preview_dataset.py --cat-split train_60.txt --dog-split mixed_train_40.txt \
  --n-cats 16 --n-dogs 8

echo "==== 2. train 3 cats-only smoke runs ===="
OVR="--max-epochs 1 --limit-batches 4 --device $DEVICE --set data.split=train_60.txt --set data.batch_size=8 --set data.num_workers=0"
python -m catgen.train --config src/catgen/configs/dcgan_smoke.yaml $OVR
python -m catgen.train --config src/catgen/configs/aae_smoke.yaml   $OVR
python -m catgen.train --config src/catgen/configs/vqvae_smoke.yaml $OVR

echo "==== 3. train 1 extension run (mixed cats+dogs) ===="
python -m catgen.train --config src/catgen/configs/dcgan_smoke.yaml \
  --max-epochs 1 --limit-batches 4 --device $DEVICE \
  --set data.split=mixed_train_40.txt --set data.batch_size=8 --set data.num_workers=0

echo "==== 4. per-run eval (sample + fid + quality + interpolate + curves) ===="
for d in runs/*/ ; do
  rid="$(basename "$d")"
  case "$rid" in _*) continue;; esac
  echo "--- $rid ---"
  python -m catgen.sample --run-id "$rid" --n 20 --device "$DEVICE"
  python -m catgen.eval_fid --run-id "$rid" --ref-split fid_ref_20.txt || true
  python -m catgen.eval_quality --run-id "$rid" --device "$DEVICE" --n-train-ref 30 || true
  python -m catgen.interpolate --run-id "$rid" --device "$DEVICE" || true
done
python scripts/plot_metrics.py

echo "==== 5. cross-run figures + leaderboard + report ===="
python -m catgen.leaderboard
python scripts/build_figures.py
python scripts/build_report.py

echo "==== 6. summary ===="
echo "runs:" && ls runs
echo "reports:" && ls reports
echo "presentation/figures:" && ls presentation/figures
echo "leaderboard:" && head -15 reports/leaderboard.md
echo "==== DONE ===="
