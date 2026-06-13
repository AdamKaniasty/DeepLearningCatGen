#!/usr/bin/env bash
# Copy presentation assets into presentation/assets/ for stable MARP paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ASSETS="$ROOT/presentation/assets"
mkdir -p "$ASSETS/dataset" "$ASSETS/part64" "$ASSETS/part128" "$ASSETS/partc"

copy() {
  local src="$1" dst="$2"
  if [ ! -f "$src" ]; then
    echo "MISSING $src"
    return 1
  fi
  cp "$src" "$dst"
  echo "OK $dst"
}

# Dataset (regenerate if smoke placeholders)
if ! python3 scripts/build_dataset_preview.py 2>/dev/null; then
  echo "warn: dataset preview build failed; using existing figures"
fi
copy presentation/figures/dataset_preview.png "$ASSETS/dataset/dataset_preview.png" || \
  copy presentation/figures/dataset_preview_cats.png "$ASSETS/dataset/dataset_preview.png"

# Part 64
copy presentation/figures/fid_bar.png "$ASSETS/part64/fid_bar.png"
copy presentation/figures/compare_grid.png "$ASSETS/part64/compare_grid.png"
copy presentation/figures/ext_compare.png "$ASSETS/part64/ext_compare.png"
copy presentation/figures/interp_dcgan.png "$ASSETS/part64/interp_dcgan.png"
copy presentation/figures/interp_aae.png "$ASSETS/part64/interp_aae.png"
copy runs/dcgan_5150b7dc_42/samples/epoch_079.png "$ASSETS/part64/dcgan_samples.png"

# Part 128
copy presentation/figures/phase128/fid_bar.png "$ASSETS/part128/fid_bar.png"
copy presentation/figures/phase128/compare_grid.png "$ASSETS/part128/compare_grid.png"
copy runs/dcgan_e9574605_42/eval/interpolation.png "$ASSETS/part128/interp_dcgan.png"
if [ -f presentation/figures/phase128/interp_aae.png ]; then
  copy presentation/figures/phase128/interp_aae.png "$ASSETS/part128/interp_aae.png"
else
  copy presentation/figures/interp_aae.png "$ASSETS/part128/interp_aae.png"
fi
if [ -f presentation/figures/phase128/ext_compare.png ]; then
  copy presentation/figures/phase128/ext_compare.png "$ASSETS/part128/ext_compare.png"
else
  copy presentation/figures/ext_compare.png "$ASSETS/part128/ext_compare.png"
fi

# Hero 4x4 from eval samples
HERO_DIR="runs/dcgan_e9574605_42/eval/samples"
python3 << 'PY'
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image
import torch
from torchvision import transforms

root = Path("runs/dcgan_e9574605_42/eval/samples")
paths = sorted(root.glob("*.png"))[:16]
if len(paths) < 4:
    raise SystemExit(f"need eval samples in {root}")
tf = transforms.ToTensor()
tiles = [tf(Image.open(p).convert("RGB")) for p in paths]
from torchvision.utils import make_grid
grid = make_grid(tiles, nrow=4, padding=2)
out = Path("presentation/assets/part128/hero_dcgan.png")
out.parent.mkdir(parents=True, exist_ok=True)
save_image(grid, out)
print(f"OK {out} ({len(paths)} tiles)")
PY

# Part C (G/D rebalance @128)
copy presentation/figures/partc/fid_bar.png "$ASSETS/partc/fid_bar.png"
copy presentation/figures/partc/gd_compare.png "$ASSETS/partc/gd_compare.png"
copy presentation/figures/partc/compare_grid.png "$ASSETS/partc/compare_grid.png"
copy presentation/figures/partc/interp_dcgan.png "$ASSETS/partc/interp_dcgan.png"

python3 scripts/validate_presentation_assets.py || true
echo "=== staged presentation/assets ==="
