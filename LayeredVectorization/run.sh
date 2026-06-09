#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TARGET_IMAGE="target_imgs/scene.jpg"
OUTPUT_ROOT="./workdir"
BASE_CONFIG="config/base_config.yaml"
TMP_CONFIG_DIR="./config/_generated"
mkdir -p "${TMP_CONFIG_DIR}"

for npath in 16 32 64 128 256; do
  echo "=== Running npath=${npath} ==="
  CFG="${TMP_CONFIG_DIR}/${npath}_config.yaml"
  RUN_NAME="scene_${npath}"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  python - <<PY
from pathlib import Path
import yaml
base = Path("${BASE_CONFIG}")
out = Path("${CFG}")
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["max_path_num_limit"] = ${npath}
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
PY

  python main.py \
    --config "${CFG}" \
    --target_image "${TARGET_IMAGE}" \
    --file_save_name "${RUN_NAME}" \
    --output_root "${OUTPUT_ROOT}"

  if [ -f "${RUN_DIR}/final.svg" ]; then
    python svg_aspect_ratio_fix.py \
      --input-svg "${RUN_DIR}/final.svg" \
      --reference-image "${TARGET_IMAGE}" \
      --output-svg "${RUN_DIR}/final_aspect_fixed.svg" \
      --square-size 512
  else
    echo "[warn] ${RUN_DIR}/final.svg not found; skip aspect-ratio fix"
  fi
done
