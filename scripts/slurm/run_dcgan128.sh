#!/usr/bin/env bash
# DCGAN 128x128 on full scope cat split (3000 train / 1000 FID ref).
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export CATGEN_TRAIN_N=3000
export CATGEN_REF_N=1000
export CATGEN_FID_N=1000
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/slurm/link_data.sh
python scripts/prepare_data.py \
  --train-n "$CATGEN_TRAIN_N" \
  --ref-n "$CATGEN_REF_N" \
  --mixed-train "$CATGEN_MIXED_TRAIN_PER_CLASS" \
  --mixed-ref "$CATGEN_MIXED_REF_PER_CLASS"

DEVICE="${CATGEN_DEVICE:-cuda}"
CFG=src/catgen/configs/dcgan_128_t3000_z128_bs32.yaml
echo "=== train $CFG (train_${CATGEN_TRAIN_N}, ${CATGEN_REF_N} ref) ==="
python -m catgen.train --config "$CFG" --device "$DEVICE"
python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"

RUN_ID=$(python3 -c "
import json, yaml
from pathlib import Path
from catgen import artifacts
cfg = yaml.safe_load(Path('$CFG').read_text())
rid = artifacts.run_id('dcgan', cfg, int(cfg.get('seed', 42)))
print(rid)
")
echo "=== eval $RUN_ID ==="
python -m catgen.sample --run-id "$RUN_ID" --n "$CATGEN_FID_N" --device "$DEVICE"
python -m catgen.eval_fid --run-id "$RUN_ID" --ref-split "fid_ref_${CATGEN_REF_N}.txt"
python -m catgen.eval_quality --run-id "$RUN_ID" --device "$DEVICE" || true
python -m catgen.interpolate --run-id "$RUN_ID" --device "$DEVICE" || true
python scripts/plot_metrics.py --run-id "$RUN_ID"
python -m catgen.leaderboard
echo "=== dcgan128 done run_id=$RUN_ID ==="
