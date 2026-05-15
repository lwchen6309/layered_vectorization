#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFilter
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from obj_detect_sam import init_sam_mask_generator


DEFAULT_DEPTH_MODELS: Sequence[str] = (
    "LiheYoung/depth-anything-small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
)


@dataclass
class MaskRecord:
    mask_id: int
    area: int
    bbox_xyxy: List[int]
    center_xy: List[float]
    sam_score: float
    stability_score: float
    mean_depth: float
    median_depth: float
    depth_score: float
    depth_bucket: str


@dataclass
class LayerBucket:
    name: str
    masks: List[MaskRecord]


# ---------- SAM helpers ----------
def _bbox_xywh_to_xyxy(bbox_xywh: Sequence[float]) -> List[int]:
    x, y, w, h = bbox_xywh
    return [int(x), int(y), int(x + w), int(y + h)]


def _box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter + 1e-8)


def _nms_candidates(candidates: List[Dict], iou_thresh: float) -> List[Dict]:
    ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept: List[Dict] = []
    for cand in ordered:
        if all(_box_iou(cand["bbox_xyxy"], prev["bbox_xyxy"]) < iou_thresh for prev in kept):
            kept.append(cand)
    return kept


def collect_sam_masks(
    image_rgb: np.ndarray,
    mask_generator,
    *,
    min_area_frac: float,
    max_area_frac: float,
    max_masks: int,
    iou_thresh: float,
) -> List[Dict]:
    h, w = image_rgb.shape[:2]
    img_area = h * w
    raw_masks = mask_generator.generate(image_rgb)
    candidates: List[Dict] = []

    for idx, entry in enumerate(raw_masks):
        seg = entry.get("segmentation")
        if seg is None:
            continue
        seg = seg.astype(bool)
        area = int(seg.sum())
        if area <= 0:
            continue

        area_frac = area / float(img_area)
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        bbox_xyxy = _bbox_xywh_to_xyxy(entry["bbox"])
        x0, y0, x1, y1 = bbox_xyxy
        if min(x1 - x0, y1 - y0) < 8:
            continue

        sam_score = float(entry.get("predicted_iou", 0.0))
        stability = float(entry.get("stability_score", 0.0))
        score = area_frac + 0.2 * sam_score + 0.2 * stability
        candidates.append(
            {
                "orig_id": idx,
                "mask": seg,
                "area": area,
                "bbox_xyxy": bbox_xyxy,
                "sam_score": sam_score,
                "stability_score": stability,
                "score": score,
            }
        )

    kept = _nms_candidates(candidates, iou_thresh=iou_thresh)
    return kept[:max_masks]


# ---------- Depth helpers ----------
def load_depth_estimator(model_name: str | None = None, device: torch.device | None = None):
    tried: List[str] = []
    model_names = [model_name] if model_name else list(DEFAULT_DEPTH_MODELS)

    last_err = None
    for name in model_names:
        tried.append(name)
        try:
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModelForDepthEstimation.from_pretrained(name)
            if device is not None:
                model = model.to(device)
            model.eval()
            return (processor, model), name
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_err = exc

    names = ", ".join(tried)
    raise RuntimeError(f"Failed to load any depth-estimation model: {names}\nLast error: {last_err}")


def run_depth_estimation(depth_bundle, image_pil: Image.Image, device: torch.device | None = None) -> np.ndarray:
    processor, model = depth_bundle
    inputs = processor(images=image_pil, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = predicted_depth.squeeze().detach().float().cpu().numpy()
    depth = cv2.resize(depth, image_pil.size, interpolation=cv2.INTER_CUBIC)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth -= depth.min()
    maxv = float(depth.max())
    if maxv > 1e-8:
        depth /= maxv
    return depth


def build_depth_scores(
    masks: List[Dict],
    depth_map: np.ndarray,
    *,
    near_mode: str,
) -> Tuple[List[MaskRecord], List[float]]:
    records: List[MaskRecord] = []
    raw_scores: List[float] = []

    for local_id, item in enumerate(masks):
        seg = item["mask"]
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        values = depth_map[seg]
        mean_depth = float(values.mean())
        median_depth = float(np.median(values))
        raw_score = median_depth
        if near_mode == "small":
            depth_score = 1.0 - raw_score
        else:
            depth_score = raw_score
        raw_scores.append(depth_score)

        x0, y0, x1, y1 = item["bbox_xyxy"]
        records.append(
            MaskRecord(
                mask_id=local_id,
                area=int(item["area"]),
                bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
                center_xy=[float(xs.mean()), float(ys.mean())],
                sam_score=float(item["sam_score"]),
                stability_score=float(item["stability_score"]),
                mean_depth=mean_depth,
                median_depth=median_depth,
                depth_score=float(depth_score),
                depth_bucket="",
            )
        )

    if not records:
        return [], []

    scores_np = np.asarray(raw_scores, dtype=np.float32)
    if len(scores_np) == 1:
        thresholds = [scores_np[0], scores_np[0]]
    elif len(scores_np) == 2:
        thresholds = [float(np.min(scores_np)), float(np.max(scores_np))]
    else:
        thresholds = [float(np.quantile(scores_np, 1.0 / 3.0)), float(np.quantile(scores_np, 2.0 / 3.0))]

    for rec in records:
        s = rec.depth_score
        if s <= thresholds[0]:
            rec.depth_bucket = "background"
        elif s <= thresholds[1]:
            rec.depth_bucket = "midground"
        else:
            rec.depth_bucket = "foreground"

    return records, thresholds


# ---------- Visualization ----------
def save_depth_visual(depth_map: np.ndarray, path: str) -> None:
    depth_u8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    Image.fromarray(colorized).save(path)


def _alpha_sprite_from_mask(image_rgba: np.ndarray, mask: np.ndarray, bbox_xyxy: Sequence[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox_xyxy
    crop_rgb = image_rgba[y0:y1, x0:x1, :3].copy()
    crop_mask = (mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, crop_mask])
    return Image.fromarray(rgba, mode="RGBA")


def save_mask_overlay(
    image_pil: Image.Image,
    mask_records: List[MaskRecord],
    mask_items: List[Dict],
    out_path: str,
) -> None:
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "foreground": "#ff5f56",
        "midground": "#ffbd2e",
        "background": "#27c93f",
    }

    for rec in sorted(mask_records, key=lambda r: r.area, reverse=True):
        seg = mask_items[rec.mask_id]["mask"]
        rgb = ImageColor.getrgb(colors[rec.depth_bucket])
        tint = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8)
        tint[seg] = (*rgb, 90)
        overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
        x0, y0, x1, y1 = rec.bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (255,), width=2)
        draw.text((x0 + 4, y0 + 4), f"{rec.mask_id}:{rec.depth_bucket[0].upper()}", fill=rgb + (255,))

    Image.alpha_composite(base, overlay).save(out_path)


def save_layered_visualization(
    image_pil: Image.Image,
    mask_records: List[MaskRecord],
    mask_items: List[Dict],
    out_path: str,
    *,
    layer_step_x: int,
    layer_step_y: int,
    scale_step: float,
    background_blur: float,
) -> None:
    w, h = image_pil.size
    pad_x = max(80, layer_step_x * 5)
    pad_y = max(60, layer_step_y * 5)
    canvas = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (250, 250, 252, 255))

    bg = image_pil.convert("RGBA")
    if background_blur > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=background_blur))
    canvas.alpha_composite(bg, (pad_x, pad_y))

    layer_rank = {"background": 0, "midground": 1, "foreground": 2}
    sorted_records = sorted(
        mask_records,
        key=lambda r: (layer_rank[r.depth_bucket], r.depth_score, r.area),
    )

    image_rgba = np.array(image_pil.convert("RGBA"))

    for rec in sorted_records:
        item = mask_items[rec.mask_id]
        sprite = _alpha_sprite_from_mask(image_rgba, item["mask"], rec.bbox_xyxy)
        x0, y0, x1, y1 = rec.bbox_xyxy
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        bucket_idx = layer_rank[rec.depth_bucket]
        shift_x = int((bucket_idx - 1) * layer_step_x)
        shift_y = int((1 - bucket_idx) * layer_step_y)
        scale = 1.0 + bucket_idx * scale_step

        new_w = max(2, int(sprite.size[0] * scale))
        new_h = max(2, int(sprite.size[1] * scale))
        sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

        paste_x = int(pad_x + cx - new_w / 2 + shift_x)
        paste_y = int(pad_y + cy - new_h / 2 + shift_y)

        shadow = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
        shadow_alpha = np.array(sprite.getchannel("A"), dtype=np.uint8)
        shadow_rgba = np.zeros((sprite.size[1], sprite.size[0], 4), dtype=np.uint8)
        shadow_rgba[..., 3] = (shadow_alpha * 0.28).astype(np.uint8)
        shadow = Image.fromarray(shadow_rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=7))
        canvas.alpha_composite(shadow, (paste_x + 10, paste_y + 10))
        canvas.alpha_composite(sprite, (paste_x, paste_y))

    legend = ImageDraw.Draw(canvas)
    legend.text((24, 20), "Layerwise view: background → midground → foreground", fill=(30, 30, 30, 255))
    canvas.save(out_path)


# ---------- Main ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SAM + depth layer visualization.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--sam-checkpoint", default="sam_vit_h_4b8939.pth", help="Path to SAM checkpoint (.pth).")
    parser.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--depth-model", default=None, help="Optional HF depth model override.")
    parser.add_argument("--output-dir", default="outputs/layerwise_images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-masks", type=int, default=18)
    parser.add_argument("--min-area-frac", type=float, default=0.003)
    parser.add_argument("--max-area-frac", type=float, default=0.45)
    parser.add_argument("--mask-nms-iou", type=float, default=0.65)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=200)
    parser.add_argument(
        "--near-mode",
        choices=["large", "small"],
        default="large",
        help="Interpret larger depth values as nearer (default, often right for DPT/Depth Anything) or smaller as nearer.",
    )
    parser.add_argument("--layer-step-x", type=int, default=40)
    parser.add_argument("--layer-step-y", type=int, default=22)
    parser.add_argument("--scale-step", type=float, default=0.06)
    parser.add_argument("--background-blur", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    image_pil = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image_pil)

    sam_generator = init_sam_mask_generator(
        device=device,
        ckpt_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )

    mask_items = collect_sam_masks(
        image_rgb,
        sam_generator,
        min_area_frac=args.min_area_frac,
        max_area_frac=args.max_area_frac,
        max_masks=args.max_masks,
        iou_thresh=args.mask_nms_iou,
    )
    if not mask_items:
        raise RuntimeError("No usable SAM masks found. Try lowering --min-area-frac or SAM thresholds.")

    depth_pipe, used_depth_model = load_depth_estimator(args.depth_model, device=device)
    depth_map = run_depth_estimation(depth_pipe, image_pil, device=device)
    mask_records, thresholds = build_depth_scores(mask_items, depth_map, near_mode=args.near_mode)
    if not mask_records:
        raise RuntimeError("Depth scoring produced no valid masks.")

    depth_path = os.path.join(args.output_dir, "depth_map.png")
    overlay_path = os.path.join(args.output_dir, "mask_depth_overlay.png")
    layered_path = os.path.join(args.output_dir, "layered_visualization.png")
    metadata_path = os.path.join(args.output_dir, "mask_depth_metadata.json")

    save_depth_visual(depth_map, depth_path)
    save_mask_overlay(image_pil, mask_records, mask_items, overlay_path)
    save_layered_visualization(
        image_pil,
        mask_records,
        mask_items,
        layered_path,
        layer_step_x=args.layer_step_x,
        layer_step_y=args.layer_step_y,
        scale_step=args.scale_step,
        background_blur=args.background_blur,
    )

    grouped = {
        "foreground": [],
        "midground": [],
        "background": [],
    }
    for rec in sorted(mask_records, key=lambda r: r.depth_score, reverse=True):
        grouped[rec.depth_bucket].append(asdict(rec))

    metadata = {
        "input_image": os.path.abspath(args.image),
        "sam_checkpoint": os.path.abspath(args.sam_checkpoint),
        "sam_model_type": args.sam_model_type,
        "depth_model": used_depth_model,
        "near_mode": args.near_mode,
        "num_masks": len(mask_records),
        "depth_thresholds": {
            "background_to_midground": float(thresholds[0]),
            "midground_to_foreground": float(thresholds[1]),
        },
        "layers": grouped,
        "artifacts": {
            "depth_map": os.path.abspath(depth_path),
            "mask_depth_overlay": os.path.abspath(overlay_path),
            "layered_visualization": os.path.abspath(layered_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"\nSaved outputs to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
