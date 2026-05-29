#!/usr/bin/env bash
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
BASE="$(dirname "$ROOT")"
export CATGEN_ROOT="$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
export TMPDIR="${TMPDIR:-$BASE/tmp}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$ROOT/slurm-logs"

UV_BIN="$ROOT/.uv-bin/uv"
if [ ! -x "$UV_BIN" ]; then
  echo "Installing uv to $ROOT/.uv-bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.uv-bin" UV_NO_MODIFY_PATH=1 sh
fi
export UV="$UV_BIN"

cd "$ROOT"
echo "Using uv: $UV_BIN"
"$UV_BIN" --version
echo "Setting up Python environment..."
echo "Syncing dependencies with uv..."
"$UV_BIN" sync

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -c "import torch, catgen; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
