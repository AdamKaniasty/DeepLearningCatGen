#!/usr/bin/env bash
# Train missing VQ-VAE configs (K256; K512 dropped for P100 memory) + eval all done runs.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/slurm/link_data.sh
python scripts/gen_configs.py

DEVICE="${CATGEN_DEVICE:-cuda}"
clear_gpu() {
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

for cfg in src/catgen/configs/vqvae_K256_*.yaml; do
  echo "=== $cfg ==="
  python -m catgen.train --config "$cfg" --device "$DEVICE"
  clear_gpu
done

echo "=== eval all done runs (figures/report in finalize job) ==="
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
bash scripts/run_eval.sh "$CATGEN_FID_N" "fid_ref_${CATGEN_REF_N}.txt" "$DEVICE"
