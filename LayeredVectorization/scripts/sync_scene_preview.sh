#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-kuhpc}"
REMOTE_REPO="${REMOTE_REPO:-/home/lwchen/layered_vectorization/LayeredVectorization}"
SCENE="${1:-workdir_depth_order/scene_depth_layer_256}"
DEST="${2:-./scene_previews/$(basename "$SCENE")}"

if [[ "$SCENE" = /* ]]; then
  REMOTE_SCENE="$SCENE"
elif [[ "$SCENE" == scenes/* ]]; then
  REMOTE_SCENE="$REMOTE_REPO/$SCENE"
else
  REMOTE_SCENE="$REMOTE_REPO/scenes/$SCENE"
fi

mkdir -p "$DEST"

echo "Syncing preview from ${REMOTE}:${REMOTE_SCENE}/"
echo "Destination: ${DEST}/"

rsync -avP --prune-empty-dirs \
  --include='*/' \
  --include='final.svg' \
  --include='final_render.png' \
  --include='final_before_depth_sort.svg' \
  --include='cluster_img.png' \
  --include='initial_depth_mask_order.json' \
  --include='depth_draw_order.json' \
  --include='depth_ordering/***' \
  --include='*.txt' \
  --include='*.json' \
  --exclude='*' \
  "${REMOTE}:${REMOTE_SCENE}/" \
  "${DEST}/"

echo "Done."
