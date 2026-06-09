#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TARGET_IMAGE="target_imgs/scene.jpg"
OUTPUT_ROOT="./workdir"
CONFIG="config/base_config.yaml"

for npath in 16 32 64 128 256; do
  echo "=== Running npath=${npath} ==="
  python main.py \
    --config "${CONFIG}" \
    --target_image "${TARGET_IMAGE}" \
    --file_save_name "scene_${npath}" \
    --output_root "${OUTPUT_ROOT}" \
    --max_path_num_limit "${npath}"
done
