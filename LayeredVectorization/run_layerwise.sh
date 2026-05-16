#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lwchen/layered_vectorization/LayeredVectorization"
PY="/home/lwchen/miniconda3/envs/lv_svg/bin/python"
SCRIPT="$ROOT/layerwise_images.py"
IMAGE="$ROOT/target_imgs/scene.jpg"
OUTDIR="$ROOT/outputs/layerwise_scene"

cd "$ROOT"

if [ ! -f "$IMAGE" ]; then
  echo "[ERROR] Missing input image: $IMAGE" >&2
  exit 1
fi

if [ ! -f "$ROOT/sam_vit_h_4b8939.pth" ]; then
  echo "[ERROR] Missing SAM checkpoint: $ROOT/sam_vit_h_4b8939.pth" >&2
  exit 1
fi

"$PY" "$SCRIPT" \
  --image "$IMAGE" \
  --sam-checkpoint "$ROOT/sam_vit_h_4b8939.pth" \
  --output-dir "$OUTDIR"
