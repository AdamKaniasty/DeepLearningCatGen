#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?usage: run_phase.sh <dcgan|aae|vqvae> [device]}"
DEVICE="${2:-auto}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for cfg in src/catgen/configs/${MODEL}_*.yaml; do
  case "$cfg" in *_smoke.yaml) continue;; esac
  echo "=== $cfg ==="
  python -m catgen.train --config "$cfg" --device "$DEVICE"
done
