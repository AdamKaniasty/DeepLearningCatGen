# Shared helpers for phase128 (source from run_phase128*.sh; do not execute directly).
# Log messages go to stderr; only run_id is printed to stdout from train_if_needed.

phase128_log() {
  echo "$@" >&2
}

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

train_exact() {
  local cfg="$1"
  local rid
  rid=$(python3 -c "
import yaml
from pathlib import Path
from catgen import artifacts
cfg = yaml.safe_load(Path('$cfg').read_text())
print(artifacts.run_id(cfg['model'], cfg, int(cfg.get('seed', 42))))
")
  if [ -f "runs/$rid/manifest.json" ] && python3 -c "import json; exit(0 if json.load(open('runs/$rid/manifest.json')).get('status')=='done' else 1)"; then
    phase128_log "[skip] $rid done ($cfg)"
    printf '%s\n' "$rid"
    return 0
  fi
  phase128_log "=== train $cfg -> $rid ==="
  python -m catgen.train --config "$cfg" --device "${CATGEN_DEVICE:-cuda}" >&2
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()" >&2
  printf '%s\n' "$rid"
}

train_if_needed() {
  local cfg="$1"
  local rid
  local model split_pat
  model=$(python3 -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('$cfg').read_text())['model'])")
  split_pat=$(python3 -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('$cfg').read_text())['data']['split'].replace('.txt',''))")
  rid=$(find_done_run "$model" "$split_pat")
  if [ -n "$rid" ]; then
    phase128_log "[skip] $rid done ($cfg)"
    printf '%s\n' "$rid"
    return 0
  fi
  rid=$(python3 -c "
import yaml
from pathlib import Path
from catgen import artifacts
cfg = yaml.safe_load(Path('$cfg').read_text())
print(artifacts.run_id(cfg['model'], cfg, int(cfg.get('seed', 42))))
")
  phase128_log "=== train $cfg -> $rid ==="
  python -m catgen.train --config "$cfg" --device "${CATGEN_DEVICE:-cuda}" >&2
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()" >&2
  printf '%s\n' "$rid"
}

resume_vqvae_phase128() {
  local cfg="${1:-src/catgen/configs/vqvae_128_t3000_K128_bs8.yaml}"
  local rid="${2:-vqvae_28e76f2a_42}"
  local bs="${3:-8}"
  local done_rid
  done_rid=$(find_done_run vqvae train_3000)
  if [ -n "$done_rid" ]; then
    printf '%s\n' "$done_rid"
    return 0
  fi
  if [ ! -f "runs/$rid/checkpoints/last.ckpt" ]; then
    phase128_log "=== train VQ from scratch (bs=$bs) -> $rid ==="
    python -m catgen.train --config "$cfg" --device "${CATGEN_DEVICE:-cuda}" --set "data.batch_size=$bs" >&2
  else
    phase128_log "=== resume VQ $rid from checkpoint (bs=$bs) ==="
    python -m catgen.train \
      --config "$cfg" \
      --device "${CATGEN_DEVICE:-cuda}" \
      --run-id "$rid" \
      --resume \
      --set "data.batch_size=$bs" >&2
  fi
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()" >&2
  printf '%s\n' "$rid"
}

eval_run() {
  local rid="$1"
  local ref_split="$2"
  phase128_log "=== eval $rid ref=$ref_split ==="
  python -m catgen.sample --run-id "$rid" --n "$CATGEN_FID_N" --device "${CATGEN_DEVICE:-cuda}"
  python -m catgen.eval_fid --run-id "$rid" --ref-split "$ref_split"
  python -m catgen.eval_quality --run-id "$rid" --device "${CATGEN_DEVICE:-cuda}" || true
  if python3 -c "import json; print(json.load(open('runs/$rid/manifest.json'))['model'])" | grep -qv vqvae; then
    python -m catgen.interpolate --run-id "$rid" --device "${CATGEN_DEVICE:-cuda}" || true
  fi
  python scripts/plot_metrics.py --run-id "$rid"
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

eval_run_if_needed() {
  local rid="$1"
  local ref_split="$2"
  if [ -f "runs/$rid/eval/fid.json" ]; then
    phase128_log "[skip eval] $rid (fid.json exists)"
    return 0
  fi
  eval_run "$rid" "$ref_split"
}

eval_ext_run_if_needed() {
  local rid="$1"
  local ref_split="$2"
  local n="${3:-$CATGEN_EXT_FID_N}"
  if [ -f "runs/$rid/eval/fid.json" ]; then
    phase128_log "[skip ext eval] $rid (fid.json exists)"
    return 0
  fi
  phase128_log "=== ext eval $rid ref=$ref_split n=$n ==="
  python -m catgen.sample --run-id "$rid" --n "$n" --device "${CATGEN_DEVICE:-cuda}"
  python -m catgen.eval_fid --run-id "$rid" --ref-split "$ref_split"
  python -m catgen.eval_quality --run-id "$rid" --device "${CATGEN_DEVICE:-cuda}" || true
  local model
  model=$(python3 -c "import json; print(json.load(open('runs/$rid/manifest.json'))['model'])")
  if [ "$model" != "vqvae" ]; then
    python -m catgen.interpolate --run-id "$rid" --device "${CATGEN_DEVICE:-cuda}" || true
  fi
  python scripts/plot_metrics.py --run-id "$rid"
  python -c "from catgen.cuda_util import clear_cuda_cache; clear_cuda_cache()"
}

pick_best_phase128_model() {
  python3 << 'PY'
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
}

write_phase128_report() {
  local best_model="$1"
  local ext_cfg="$2"
  local ext_rid="$3"
  local dcgan_rid="$4"
  local aae_rid="$5"
  local vq_rid="$6"
  python3 -c "
import json
from pathlib import Path
Path('reports/phase128_best.json').write_text(json.dumps({
    'best_model': '$best_model',
    'ext_cfg': '$ext_cfg',
    'ext_run_id': '$ext_rid',
    'dcgan_cats': '$dcgan_rid',
    'aae_cats': '$aae_rid',
    'vqvae_cats': '$vq_rid',
}, indent=2))
"
  phase128_log "wrote reports/phase128_best.json"
}

finish_phase128_artifacts() {
  export PRESENTATION_PHASE=phase128
  python scripts/build_figures.py
  python -m catgen.leaderboard
}
