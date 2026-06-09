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
    --file_save_name "scene_${npath}" \
    --output_root "${OUTPUT_ROOT}"
done
