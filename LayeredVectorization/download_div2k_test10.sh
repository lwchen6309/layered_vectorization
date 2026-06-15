#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/lwchen/layered_vectorization/LayeredVectorization"
DATA_ROOT="$ROOT/dataset"
DIV2K_DIR="$DATA_ROOT/DIV2K_train_HR"
DIV2K_LINK="$DATA_ROOT/DIV2K_HR"
TEST10_DIR="$DATA_ROOT/DIV2K_test10"
ZIP_PATH="$DATA_ROOT/DIV2K_train_HR.zip"
URL="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"

mkdir -p "$DATA_ROOT"

if [ ! -f "$ZIP_PATH" ]; then
  echo "[Info] Downloading DIV2K_train_HR.zip"
  wget -c -O "$ZIP_PATH" "$URL"
else
  echo "[Skip] ZIP exists: $ZIP_PATH"
fi

if [ ! -d "$DIV2K_DIR" ]; then
  echo "[Info] Unzipping DIV2K_train_HR.zip"
  unzip -q -n "$ZIP_PATH" -d "$DATA_ROOT"
else
  echo "[Skip] Unzipped dir exists: $DIV2K_DIR"
fi

if [ ! -e "$DIV2K_LINK" ]; then
  ln -s DIV2K_train_HR "$DIV2K_LINK"
fi

mkdir -p "$TEST10_DIR"
python3 - <<'PY'
from pathlib import Path
src = Path('/home/lwchen/layered_vectorization/LayeredVectorization/dataset/DIV2K_HR')
dst = Path('/home/lwchen/layered_vectorization/LayeredVectorization/dataset/DIV2K_test10')
files = sorted([p for p in src.iterdir() if p.is_file()])[:10]
for p in files:
    link = dst / p.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(p)
print('selected:')
for p in files:
    print(p.name)
PY

echo "[Done] DIV2K_test10 prepared at $TEST10_DIR"
