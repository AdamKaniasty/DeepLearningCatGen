#!/usr/bin/env bash
# Cats-and-dogs extension (scope § extension): DCGAN on mixed split, eval vs mixed ref.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/slurm/link_data.sh
if [ ! -d data/raw/dogs ] && [ ! -L data/raw/dogs ]; then
  echo "ERROR: data/raw/dogs missing; set CATGEN_DATA_DOGS or fix link_data.sh"
  exit 1
fi

python scripts/gen_configs.py
python scripts/prepare_data.py \
  --train-n "$CATGEN_TRAIN_N" \
  --ref-n "$CATGEN_REF_N" \
  --mixed-train "$CATGEN_MIXED_TRAIN_PER_CLASS" \
  --mixed-ref "$CATGEN_MIXED_REF_PER_CLASS"

MIXED_TRAIN_N=$((2 * CATGEN_MIXED_TRAIN_PER_CLASS))
MIXED_REF_N=$((2 * CATGEN_MIXED_REF_PER_CLASS))
EXT_CFG="src/catgen/configs/dcgan_ext_mixed_z128_lr2e-04_bs32.yaml"

DEVICE="${CATGEN_DEVICE:-cuda}"
clear_gpu() {
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

echo "=== extension train (mixed_train_${MIXED_TRAIN_N}) ==="
python -m catgen.train --config "$EXT_CFG" --device "$DEVICE"
clear_gpu

RID=$(python -c "
import json, yaml, hashlib
from pathlib import Path
cfg = yaml.safe_load(Path('$EXT_CFG').read_text())
seed = int(cfg.get('seed', 42))
h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
print(f\"{cfg['model']}_{h}_{seed}\")
")

echo "=== extension eval (mixed_ref_${MIXED_REF_N}) run_id=$RID ==="
bash scripts/run_eval_run.sh "$RID" "$CATGEN_EXT_FID_N" "mixed_ref_${MIXED_REF_N}.txt" "$DEVICE"
clear_gpu
echo "=== extension train+eval done (presentation bundle in finalize job) ==="
