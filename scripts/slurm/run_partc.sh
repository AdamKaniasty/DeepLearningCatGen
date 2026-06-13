#!/usr/bin/env bash
# Part C: DCGAN @128 G/D rebalance (same 3k cats as phase128 Part B).
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export CATGEN_TRAIN_N=3000
export CATGEN_REF_N=1000
export CATGEN_FID_N=1000
# shellcheck source=sizes.env
source scripts/slurm/sizes.env
# shellcheck source=phase128_lib.sh
source scripts/slurm/phase128_lib.sh
export CATGEN_DEVICE="${CATGEN_DEVICE:-cuda}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -m certifi)}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASELINE_RID="${PARTC_BASELINE_RID:-dcgan_e9574605_42}"
CFG=src/catgen/configs/dcgan_128_partc_gd_rebalance.yaml
CAT_REF="fid_ref_${CATGEN_REF_N}.txt"

phase128_log "=== Part C: DCGAN G/D rebalance @128 ==="
phase128_log "baseline (Phase B): $BASELINE_RID"

bash scripts/slurm/link_data.sh
python scripts/prepare_data.py \
  --train-n "$CATGEN_TRAIN_N" \
  --ref-n "$CATGEN_REF_N" \
  --mixed-train "$CATGEN_MIXED_TRAIN_PER_CLASS" \
  --mixed-ref "$CATGEN_MIXED_REF_PER_CLASS"
python scripts/gen_partc_configs.py

PARTC_RID=$(train_exact "$CFG")
eval_run_if_needed "$PARTC_RID" "$CAT_REF"

python3 << PY
import json
from pathlib import Path

def fid(rid):
    p = Path("runs") / rid / "eval" / "fid.json"
    return json.loads(p.read_text()).get("fid") if p.exists() else None

baseline = "$BASELINE_RID"
partc = "$PARTC_RID"
report = {
    "baseline_run_id": baseline,
    "baseline_fid": fid(baseline),
    "partc_run_id": partc,
    "partc_fid": fid(partc),
    "config": "src/catgen/configs/dcgan_128_partc_gd_rebalance.yaml",
    "changes": {
        "lr": "2e-4 (was 1e-4)",
        "lr_d": "1e-4 (was 2e-4)",
        "label_smooth": "0.05 (was 0.1)",
    },
}
Path("reports").mkdir(exist_ok=True)
Path("reports/partc_best.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

export PRESENTATION_PHASE=partc
python scripts/build_figures.py
python -m catgen.leaderboard
phase128_log "=== Part C complete partc=$PARTC_RID baseline=$BASELINE_RID ==="
