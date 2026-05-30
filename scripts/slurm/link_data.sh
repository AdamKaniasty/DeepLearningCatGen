#!/usr/bin/env bash
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CATS="${CATGEN_DATA_CATS:-/mnt/evafs/faculty/home/kbokhan/data/cats}"
mkdir -p "$ROOT/data/raw"
rm -rf "$ROOT/data/raw/cats"
ln -sfn "$CATS" "$ROOT/data/raw/cats"
echo "linked $ROOT/data/raw/cats -> $CATS"
