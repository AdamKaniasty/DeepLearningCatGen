#!/usr/bin/env bash
# Finish phase128 after training done: eval VQ, extension, figures (no re-train).
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export CATGEN_FID_N=1000
export CATGEN_EXT_FID_N=400
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
# shellcheck source=phase128_lib.sh
source scripts/slurm/phase128_lib.sh
export CATGEN_DEVICE="${CATGEN_DEVICE:-cuda}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

DCGAN_RID="${PHASE128_DCGAN_RID:-dcgan_e9574605_42}"
AAE_RID="${PHASE128_AAE_RID:-aae_7ad02727_42}"
VQ_RID="${PHASE128_VQ_RID:-vqvae_28e76f2a_42}"
CAT_REF="fid_ref_1000.txt"
MIXED_REF="mixed_ref_$((2 * CATGEN_MIXED_REF_PER_CLASS)).txt"

phase128_log "=== phase128 finish (eval + extension) ==="

for rid in "$DCGAN_RID" "$AAE_RID" "$VQ_RID"; do
  eval_run_if_needed "$rid" "$CAT_REF"
done

BEST_MODEL=$(pick_best_phase128_model)
phase128_log "=== best phase128 cats model: $BEST_MODEL ==="

case "$BEST_MODEL" in
  dcgan) EXT_CFG=src/catgen/configs/dcgan_ext_128_mixed_bs32.yaml ;;
  aae)   EXT_CFG=src/catgen/configs/aae_ext_128_mixed_bs32.yaml ;;
  vqvae) EXT_CFG=src/catgen/configs/vqvae_ext_128_mixed_bs8.yaml ;;
  *)     EXT_CFG=src/catgen/configs/dcgan_ext_128_mixed_bs32.yaml ;;
esac

EXT_RID=$(train_if_needed "$EXT_CFG")
eval_ext_run_if_needed "$EXT_RID" "$MIXED_REF" "$CATGEN_EXT_FID_N"

write_phase128_report "$BEST_MODEL" "$EXT_CFG" "$EXT_RID" "$DCGAN_RID" "$AAE_RID" "$VQ_RID"
finish_phase128_artifacts
phase128_log "=== phase128 finish complete best=$BEST_MODEL ext=$EXT_RID ==="
