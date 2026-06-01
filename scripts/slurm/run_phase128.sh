#!/usr/bin/env bash
# Phase 128: train AAE+VQ (DCGAN cats if missing), pick best FID, train mixed ext for winner, eval + figures.
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
export CATGEN_TRAIN_N=3000
export CATGEN_REF_N=1000
export CATGEN_FID_N=1000
export CATGEN_EXT_FID_N=400
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
python scripts/gen_phase128_configs.py

DEVICE="${CATGEN_DEVICE:-cuda}"
MIXED_REF="mixed_ref_$((2 * CATGEN_MIXED_REF_PER_CLASS)).txt"

find_done_run() {
  local model="$1"
  local split_pat="$2"
  python3 -c "
import json
from pathlib import Path
model, split_pat = '$model', '$split_pat'
for d in Path('runs').iterdir():
    mp = d / 'manifest.json'
    if not mp.exists(): continue
    m = json.loads(mp.read_text())
    if m.get('status') != 'done' or m.get('model') != model: continue
    data = (m.get('config') or {}).get('data', {})
    if data.get('image_size') != 128: continue
    if split_pat not in data.get('split', ''): continue
    print(d.name)
    break
"
}

train_if_needed() {
  local cfg="$1"
  local rid
  local model split_pat
  model=$(python3 -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('$cfg').read_text())['model'])")
  split_pat=$(python3 -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('$cfg').read_text())['data']['split'].replace('.txt',''))")
  rid=$(find_done_run "$model" "$split_pat")
  if [ -n "$rid" ]; then
    echo "[skip] $rid done ($cfg)"
    echo "$rid"
    return 0
  fi
  rid=$(python3 -c "
import yaml
from pathlib import Path
from catgen import artifacts
cfg = yaml.safe_load(Path('$cfg').read_text())
print(artifacts.run_id(cfg['model'], cfg, int(cfg.get('seed', 42))))
")
  echo "=== train $cfg -> $rid ==="
  python -m catgen.train --config "$cfg" --device "$DEVICE"
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
  echo "$rid"
}

eval_run() {
  local rid="$1"
  local ref_split="$2"
  echo "=== eval $rid ref=$ref_split ==="
  python -m catgen.sample --run-id "$rid" --n "$CATGEN_FID_N" --device "$DEVICE"
  python -m catgen.eval_fid --run-id "$rid" --ref-split "$ref_split"
  python -m catgen.eval_quality --run-id "$rid" --device "$DEVICE" || true
  if python3 -c "import json; print(json.load(open('runs/$rid/manifest.json'))['model'])" | grep -qv vqvae; then
    python -m catgen.interpolate --run-id "$rid" --device "$DEVICE" || true
  fi
  python scripts/plot_metrics.py --run-id "$rid"
}

# Cats-only 128 models
DCGAN_CFG=src/catgen/configs/dcgan_128_t3000_z128_bs32.yaml
AAE_CFG=src/catgen/configs/aae_128_t3000_z128_bs32.yaml
VQ_CFG=src/catgen/configs/vqvae_128_t3000_K128_bs16.yaml

DCGAN_RID=$(train_if_needed "$DCGAN_CFG")
AAE_RID=$(train_if_needed "$AAE_CFG")
VQ_RID=$(train_if_needed "$VQ_CFG")

CAT_REF="fid_ref_${CATGEN_REF_N}.txt"
for rid in "$DCGAN_RID" "$AAE_RID" "$VQ_RID"; do
  eval_run "$rid" "$CAT_REF"
done

# Pick best cats-only phase128 by FID
BEST_MODEL=$(python3 << 'PY'
import json
from pathlib import Path
best = (1e9, None)
for d in Path("runs").iterdir():
    mp = d / "manifest.json"
    if not mp.exists():
        continue
    m = json.loads(mp.read_text())
    if m.get("status") != "done":
        continue
    tags = (m.get("config") or {}).get("tags", [])
    data_cfg = (m.get("config") or {}).get("data", {})
    is_p128 = (
        "phase128" in tags
        or "dcgan128" in tags
        or (data_cfg.get("image_size") == 128 and "train_3000" in data_cfg.get("split", ""))
    )
    if not is_p128:
        continue
    data_cfg = (m.get("config") or {}).get("data", {})
    split = data_cfg.get("split", "")
    if "mixed_train" in split:
        continue
    if data_cfg.get("image_size") != 128:
        continue
    fp = d / "eval" / "fid.json"
    if not fp.exists():
        continue
    fid = json.loads(fp.read_text()).get("fid")
    if fid is None:
        continue
    if fid < best[0]:
        best = (fid, m.get("model"))
print(best[1] or "dcgan")
PY
)
echo "=== best phase128 cats model: $BEST_MODEL ==="

case "$BEST_MODEL" in
  dcgan) EXT_CFG=src/catgen/configs/dcgan_ext_128_mixed_bs32.yaml ;;
  aae)   EXT_CFG=src/catgen/configs/aae_ext_128_mixed_bs32.yaml ;;
  vqvae) EXT_CFG=src/catgen/configs/vqvae_ext_128_mixed_bs16.yaml ;;
  *)     EXT_CFG=src/catgen/configs/dcgan_ext_128_mixed_bs32.yaml ;;
esac

EXT_RID=$(train_if_needed "$EXT_CFG")
python -m catgen.sample --run-id "$EXT_RID" --n "$CATGEN_EXT_FID_N" --device "$DEVICE"
python -m catgen.eval_fid --run-id "$EXT_RID" --ref-split "$MIXED_REF"
python -m catgen.eval_quality --run-id "$EXT_RID" --device "$DEVICE" || true
if [ "$BEST_MODEL" != "vqvae" ]; then
  python -m catgen.interpolate --run-id "$EXT_RID" --device "$DEVICE" || true
fi
python scripts/plot_metrics.py --run-id "$EXT_RID"

python3 -c "
import json
from pathlib import Path
Path('reports/phase128_best.json').write_text(json.dumps({
    'best_model': '$BEST_MODEL',
    'ext_cfg': '$EXT_CFG',
    'ext_run_id': '$EXT_RID',
    'dcgan_cats': '$DCGAN_RID',
    'aae_cats': '$AAE_RID',
    'vqvae_cats': '$VQ_RID',
}, indent=2))
"

export PRESENTATION_PHASE=phase128
python scripts/build_figures.py
python -m catgen.leaderboard
echo "=== phase128 complete best=$BEST_MODEL ext=$EXT_RID ==="
