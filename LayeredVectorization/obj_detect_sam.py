# obj_detect_sam_only.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

BBox = List[float]  # [x0,y0,x1,y1] in ref_size coord system


def init_sam_mask_generator(
    device: torch.device,
    ckpt_path: str,
    model_type: str = "vit_h",
    *,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.92,
    crop_n_layers: int = 1,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 200,
):
    """
    Returns a SegmentAnything AutomaticMaskGenerator.
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    sam.to(device)

    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )
    return gen


def _bbox_from_xywh(x: int, y: int, w: int, h: int) -> List[int]:
    return [int(x), int(y), int(x + w), int(y + h)]


def _make_square_bbox_xyxy(b: List[int], W: int, H: int, enlarge: float = 1.0) -> List[int]:
    x0, y0, x1, y1 = map(float, b)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * enlarge, (y1 - y0) * enlarge
    s = max(w, h)
    x0 = int(round(cx - s / 2))
    x1 = int(round(cx + s / 2))
    y0 = int(round(cy - s / 2))
    y1 = int(round(cy + s / 2))
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(1, min(W, x1))
    y1 = max(1, min(H, y1))
    return [x0, y0, x1, y1]


def _box_iou(a: List[int], b: List[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return float(inter) / float(area_a + area_b - inter + 1e-8)


def _nms_boxes(boxes: List[List[int]], scores: List[float], iou_thr: float = 0.6) -> List[int]:
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    for i in idxs:
        ok = True
        for j in keep:
            if _box_iou(boxes[i], boxes[j]) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


@dataclass
class SamOnlyConfig:
    ref_size: int = 512
    topk: int = 8
    square: bool = True
    enlarge: float = 1.05

    # heuristics to filter masks
    min_area_frac: float = 0.002     # relative to image area
    max_area_frac: float = 0.20      # remove giant background masks
    min_iou_score: float = 0.0       # optional gate using mask["predicted_iou"]
    min_stability: float = 0.0       # optional gate using mask["stability_score"]

    nms_iou: float = 0.65


@torch.inference_mode()
def detect_bboxes_sam_only(
    *,
    mask_generator,            # SamAutomaticMaskGenerator
    base_img_pil: Image.Image,
    cfg: SamOnlyConfig = SamOnlyConfig(),
    debug_dir: Optional[str] = None,
) -> List[BBox]:
    """
    Return top-K bboxes in cfg.ref_size coord system (default 512): [x0,y0,x1,y1] floats.
    Saves debug crops/masks if debug_dir is set.
    """
    base_img_pil = base_img_pil.convert("RGB")
    base_rgb = np.array(base_img_pil).astype(np.uint8)
    H, W = base_rgb.shape[:2]
    img_area = float(H * W)

    masks = mask_generator.generate(base_rgb)

    boxes_px: List[List[int]] = []
    scores: List[float] = []
    segs: List[np.ndarray] = []

    for m in masks:
        # expected keys: 'segmentation' (bool HxW), 'area', 'bbox' (x,y,w,h),
        # 'predicted_iou', 'stability_score'
        area = float(m.get("area", 0.0))
        if area <= 0:
            continue
        area_frac = area / img_area
        if area_frac < cfg.min_area_frac:
            continue
        if area_frac > cfg.max_area_frac:
            continue

        if m.get("predicted_iou", 1.0) < cfg.min_iou_score:
            continue
        if m.get("stability_score", 1.0) < cfg.min_stability:
            continue

        x, y, bw, bh = m["bbox"]
        bb = _bbox_from_xywh(x, y, bw, bh)

        # simple shape heuristic: ignore extremely thin shapes
        wbb = bb[2] - bb[0]
        hbb = bb[3] - bb[1]
        if min(wbb, hbb) < 8:
            continue

        if cfg.square:
            bb = _make_square_bbox_xyxy(bb, W, H, enlarge=cfg.enlarge)

        # score: prefer larger + stable
        sc = float(area_frac)
        sc += 0.2 * float(m.get("predicted_iou", 0.0))
        sc += 0.2 * float(m.get("stability_score", 0.0))

        boxes_px.append(bb)
        scores.append(sc)
        segs.append((m["segmentation"].astype(np.uint8) * 255))

    if not boxes_px:
        return []

    keep = _nms_boxes(boxes_px, scores, iou_thr=cfg.nms_iou)
    keep = keep[: cfg.topk]

    boxes_px = [boxes_px[i] for i in keep]
    segs = [segs[i] for i in keep]

    # sort final by score desc (again)
    paired = list(zip(boxes_px, segs, [scores[i] for i in keep]))
    paired.sort(key=lambda t: t[2], reverse=True)
    boxes_px = [p[0] for p in paired]
    segs = [p[1] for p in paired]

    # map px -> ref coord system
    sx = float(cfg.ref_size) / float(W)
    sy = float(cfg.ref_size) / float(H)
    boxes_ref: List[BBox] = []
    for x0, y0, x1, y1 in boxes_px:
        boxes_ref.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy])

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        for i, (bb, seg) in enumerate(zip(boxes_px, segs)):
            x0, y0, x1, y1 = bb
            crop = base_img_pil.crop((x0, y0, x1, y1))
            crop.save(os.path.join(debug_dir, f"bbox_{i:02d}_crop.png"))
            Image.fromarray(seg).save(os.path.join(debug_dir, f"bbox_{i:02d}_mask.png"))

        with open(os.path.join(debug_dir, "bboxes_ref.txt"), "w") as f:
            for b in boxes_ref:
                f.write(f"{b}\n")

    return boxes_ref
