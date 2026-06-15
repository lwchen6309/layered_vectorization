#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/lwchen/layered_vectorization/LayeredVectorization"
DATASET_DIR="$ROOT/dataset/DIV2K_test10"
OUTPUT_ROOT="$ROOT/div2k_test10_runs"
CONFIG="config/256_config.yaml"
MAIN_PY="main.py"

source /home/lwchen/miniconda3/etc/profile.d/conda.sh
conda activate lv_svg
set -u

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

find "$DATASET_DIR" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort | while read -r IMG_PATH; do
  IMG_NAME="$(basename "$IMG_PATH")"
  STEM="${IMG_NAME%.*}"
  SAVE_NAME="div2k_${STEM}"
  RUN_DIR="$OUTPUT_ROOT/$SAVE_NAME"
  FINAL_SVG="$RUN_DIR/final.svg"

  echo "============================================================"
  echo "Running LayerVec on $IMG_NAME"
  echo "============================================================"

  if [ -f "$FINAL_SVG" ]; then
    echo "[Skip] final.svg exists: $FINAL_SVG"
    continue
  fi

  if [ -d "$RUN_DIR" ]; then
    echo "[Info] Removing incomplete run dir: $RUN_DIR"
    rm -rf "$RUN_DIR"
  fi

  python "$MAIN_PY" \
    --config "$CONFIG" \
    --target_image "$IMG_PATH" \
    --file_save_name "$SAVE_NAME" \
    --output_root "$OUTPUT_ROOT"

done

echo "ALL_DONE"
