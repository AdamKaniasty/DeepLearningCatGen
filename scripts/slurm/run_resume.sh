#!/usr/bin/env bash
# Finish VQ-VAE after job 1694964 OOM: resume failed run + train missing config + eval.
# Does not wipe runs/ or retrain DCGAN/AAE.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/slurm/link_data.sh

DEVICE="${CATGEN_DEVICE:-cuda}"
clear_gpu() {
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

# Job 1694964: K512 lr1e-4 OOM at epoch 21 (glob order: after K128, before K512 lr2e-4).
FAILED_ID="vqvae_a0e4bdc2_42"
FAILED_CFG="src/catgen/configs/vqvae_K512_D64_lr1e-04.yaml"

if compgen -G "runs/${FAILED_ID}/checkpoints/*.ckpt" > /dev/null; then
  echo "=== resume ${FAILED_ID} from checkpoint (batch_size=16) ==="
  python -m catgen.train \
    --config "$FAILED_CFG" \
    --run-id "$FAILED_ID" \
    --resume \
    --device "$DEVICE" \
    --set data.batch_size=16
else
  echo "=== no checkpoint for ${FAILED_ID}; restart from scratch (batch_size=16) ==="
  python -m catgen.train \
    --config "$FAILED_CFG" \
    --run-id "$FAILED_ID" \
    --force \
    --device "$DEVICE" \
    --set data.batch_size=16
fi
clear_gpu

echo "=== train missing vqvae_K512_D64_lr2e-04 (batch_size=16) ==="
python -m catgen.train \
  --config src/catgen/configs/vqvae_K512_D64_lr2e-04.yaml \
  --device "$DEVICE" \
  --set data.batch_size=16
clear_gpu

echo "=== eval all done runs (figures/report in finalize job) ==="
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
bash scripts/run_eval.sh "$CATGEN_FID_N" "fid_ref_${CATGEN_REF_N}.txt" "$DEVICE"
