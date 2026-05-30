#!/usr/bin/env bash
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

if [ "${CATGEN_SMOKE:-0}" = "1" ]; then
  rm -f data/raw/cats
  mkdir -p data/raw/cats
  python scripts/prepare_data.py --fake 200 --train-n 120 --ref-n 40 --no-mixed
else
  bash scripts/slurm/link_data.sh
fi

if [ "${CATGEN_SMOKE:-0}" = "1" ]; then
  DEVICE="${CATGEN_DEVICE:-cuda}"
  OVR="--max-epochs 2 --limit-batches 8 --device $DEVICE --set data.split=train_120.txt --set data.batch_size=16 --set data.num_workers=4"
  python -m catgen.train --config src/catgen/configs/dcgan_smoke.yaml $OVR
  python -m catgen.train --config src/catgen/configs/aae_smoke.yaml $OVR
  python -m catgen.train --config src/catgen/configs/vqvae_smoke.yaml $OVR
  bash scripts/run_eval.sh 100 fid_ref_40.txt "$DEVICE"
  python scripts/plot_metrics.py
  python -m catgen.leaderboard
  exit 0
fi

python scripts/gen_configs.py
python scripts/prepare_data.py --train-n 1500 --ref-n 500 --no-mixed

DEVICE="${CATGEN_DEVICE:-cuda}"
bash scripts/run_phase.sh dcgan "$DEVICE"
bash scripts/run_phase.sh aae "$DEVICE"
bash scripts/run_phase.sh vqvae "$DEVICE"
bash scripts/run_eval.sh 500 fid_ref_500.txt "$DEVICE"
python scripts/plot_metrics.py
python scripts/build_figures.py
python scripts/build_report.py
