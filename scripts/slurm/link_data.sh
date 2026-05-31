#!/usr/bin/env bash
set -euo pipefail
ROOT="${CATGEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CATS="${CATGEN_DATA_CATS:-/mnt/evafs/faculty/home/kbokhan/data/cats}"
DOGS="${CATGEN_DATA_DOGS:-/mnt/evafs/faculty/home/kbokhan/data/cats_dogs}"
mkdir -p "$ROOT/data/raw"
rm -rf "$ROOT/data/raw/cats"
ln -sfn "$CATS" "$ROOT/data/raw/cats"
echo "linked $ROOT/data/raw/cats -> $CATS"
if [ -d "$DOGS" ] || [ -L "$DOGS" ]; then
  rm -rf "$ROOT/data/raw/dogs"
  ln -sfn "$DOGS" "$ROOT/data/raw/dogs"
  echo "linked $ROOT/data/raw/dogs -> $DOGS"
else
  echo "WARN: dogs data not found at $DOGS (cats-and-dogs extension will be skipped)"
fi
