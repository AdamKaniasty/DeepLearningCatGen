#!/usr/bin/env bash
# Strategy B: train refined models (upsample-conv DCGAN, longer training, tags: refine).
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
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
python scripts/gen_refine_configs.py

DEVICE="${CATGEN_DEVICE:-cuda}"
clear_gpu() {
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

for cfg in \
  src/catgen/configs/dcgan_refine_z128_bs64.yaml \
  src/catgen/configs/aae_refine_z128_bs64.yaml \
  src/catgen/configs/vqvae_refine_K128_bs32.yaml \
  src/catgen/configs/dcgan_ext_refine_mixed_bs64.yaml
do
  echo "=== train $cfg ==="
  python -m catgen.train --config "$cfg" --device "$DEVICE"
  clear_gpu
done

echo "=== refine training complete ==="
