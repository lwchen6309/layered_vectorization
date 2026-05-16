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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel, AutoTokenizer
from diffusers import AutoPipelineForInpainting

from obj_detect_sam import init_sam_mask_generator


DEFAULT_DEPTH_MODELS: Sequence[str] = (
    "LiheYoung/depth-anything-small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
)

DEFAULT_INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_SEMANTIC_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_SEMANTIC_LABELS: Sequence[str] = (
    "person",
    "face",
    "hand",
    "animal",
    "vehicle",
    "bicycle",
    "building",
    "window",
    "door",
    "tree",
    "plant",
    "flower",
    "sky",
    "cloud",
    "mountain",
    "water",
    "road",
    "ground",
    "furniture",
    "table",
    "chair",
    "screen",
    "book",
    "food",
    "background texture",
    "foreground object",
)
DEFAULT_INPAINT_NEGATIVE_PROMPT = (
    "extra objects, duplicated objects, floating objects, warped geometry, blurry, distorted,"
    " low quality, text, watermark"
)


@dataclass
class RegionRecord:
    region_id: int
    source_mask_id: int
    area: int
    bbox_xyxy: List[int]
    center_xy: List[float]
    sam_score: float
    stability_score: float
    mean_depth: float
    median_depth: float
    depth_score: float
    normalized_depth_score: float
    depth_rank: int
    layer_id: int
    layer_name: str
    semantic_label: str
    semantic_confidence: float


@dataclass
class LayerRecord:
    layer_id: int
    name: str
    region_ids: List[int]
    num_regions: int
    total_area: int
    cluster_center_depth_score: float
    mean_depth_score: float
    min_depth_score: float
    max_depth_score: float


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


def _mask_to_bbox(seg: np.ndarray) -> List[int]:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def collect_mask_regions(
    masks: List[Dict],
    *,
    min_region_area: int,
) -> List[Dict]:
    regions: List[Dict] = []

    for mask_idx, item in enumerate(masks):
        seg_u8 = item["mask"].astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_u8, connectivity=8)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_region_area:
                continue
            region_mask = labels == label
            if not region_mask.any():
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            regions.append(
                {
                    "source_mask_id": mask_idx,
                    "mask": region_mask,
                    "area": area,
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "sam_score": float(item["sam_score"]),
                    "stability_score": float(item["stability_score"]),
                }
            )

    if not regions:
        return []

    regions.sort(key=lambda item: item["area"], reverse=True)
    for region_id, region in enumerate(regions):
        region["region_id"] = region_id
    return regions


def _kmeans_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    unique = np.unique(x)
    k_eff = max(1, min(int(k), len(unique), n))
    if k_eff == 1:
        return np.zeros(n, dtype=np.int32)

    percentiles = np.linspace(0, 100, k_eff + 2)[1:-1]
    centers = np.percentile(x, percentiles).astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(50):
        dist = np.abs(x[:, None] - centers[None, :])
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for idx in range(k_eff):
            mask = new_labels == idx
            if np.any(mask):
                new_centers[idx] = float(x[mask].mean())
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order.tolist())}
    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
    return labels


def load_semantic_encoder(model_name: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return processor, model, tokenizer


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


def build_region_semantic_features(
    image_pil: Image.Image,
    regions: List[Dict],
    semantic_bundle,
    *,
    semantic_labels: Sequence[str],
    device: torch.device,
) -> List[Dict[str, object]]:
    processor, model, tokenizer = semantic_bundle
    image_rgb = image_pil.convert("RGB")
    prompts = [f"a photo of {label}" for label in semantic_labels]

    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**tokenized)
    text_features = text_features.detach().float().cpu().numpy()
    text_features = _normalize_rows(text_features)

    crops: List[Image.Image] = []
    for item in regions:
        x0, y0, x1, y1 = item["bbox_xyxy"]
        crop = image_rgb.crop((x0, y0, x1, y1))
        mask_crop = item["mask"][y0:y1, x0:x1]
        crop_np = np.array(crop)
        crop_np[~mask_crop] = crop_np[~mask_crop] // 3
        crops.append(Image.fromarray(crop_np))

    inputs = processor(images=crops, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features.detach().float().cpu().numpy()
    image_features = _normalize_rows(image_features)

    similarity = image_features @ text_features.T
    results: List[Dict[str, object]] = []
    for idx in range(len(regions)):
        sim_row = similarity[idx]
        best_idx = int(np.argmax(sim_row))
        conf = float(sim_row[best_idx])
        semantic_vector = sim_row.astype(np.float32)
        norm = float(np.linalg.norm(semantic_vector))
        if norm > 1e-8:
            semantic_vector = semantic_vector / norm
        results.append(
            {
                "semantic_label": semantic_labels[best_idx],
                "semantic_confidence": conf,
                "semantic_embedding": semantic_vector,
            }
        )
    return results


def assign_depth_layers(
    regions: List[Dict],
    depth_map: np.ndarray,
    *,
    near_mode: str,
    num_depth_layers: int,
) -> Tuple[List[RegionRecord], List[LayerRecord]]:
    records: List[RegionRecord] = []

    for item in regions:
        seg = item["mask"]
        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue
        values = depth_map[seg]
        mean_depth = float(values.mean())
        median_depth = float(np.median(values))
        raw_score = mean_depth
        depth_score = 1.0 - raw_score if near_mode == "small" else raw_score

        records.append(
            RegionRecord(
                region_id=int(item["region_id"]),
                source_mask_id=int(item["source_mask_id"]),
                area=int(item["area"]),
                bbox_xyxy=[int(v) for v in item["bbox_xyxy"]],
                center_xy=[float(xs.mean()), float(ys.mean())],
                sam_score=float(item["sam_score"]),
                stability_score=float(item["stability_score"]),
                mean_depth=mean_depth,
                median_depth=median_depth,
                depth_score=float(depth_score),
                normalized_depth_score=0.0,
                depth_rank=-1,
                layer_id=-1,
                layer_name="",
                semantic_label=str(item.get("semantic_label", "unknown")),
                semantic_confidence=float(item.get("semantic_confidence", 0.0)),
            )
        )

    if not records:
        return [], []

    scores = np.asarray([rec.depth_score for rec in records], dtype=np.float32)
    min_score = float(scores.min())
    max_score = float(scores.max())
    span = max(max_score - min_score, 1e-8)
    for rec in records:
        rec.normalized_depth_score = float((rec.depth_score - min_score) / span)

    norm_scores = np.asarray([rec.normalized_depth_score for rec in records], dtype=np.float32)
    labels = _kmeans_1d(norm_scores, num_depth_layers)

    grouped_by_label: Dict[int, List[RegionRecord]] = {}
    for rec, label in zip(records, labels):
        grouped_by_label.setdefault(int(label), []).append(rec)

    grouped_layers_unsorted = list(grouped_by_label.values())
    grouped_layers_unsorted.sort(key=lambda layer_regions: float(np.mean([rec.depth_score for rec in layer_regions])))

    total_layers = len(grouped_layers_unsorted)

    def semantic_layer_name(layer_idx: int, total: int) -> str:
        if total <= 1:
            return "midground"
        if total == 2:
            return ["background", "foreground"][layer_idx]
        if total == 3:
            return ["background", "midground", "foreground"][layer_idx]
        if layer_idx == 0:
            return "background"
        if layer_idx == total - 1:
            return "foreground"
        if layer_idx == total // 2:
            return "midground"
        if layer_idx < total // 2:
            return f"background_plus_{layer_idx}"
        return f"foreground_minus_{total - 1 - layer_idx}"

    layer_records: List[LayerRecord] = []
    for layer_idx, layer_regions in enumerate(grouped_layers_unsorted):
        layer_regions.sort(key=lambda rec: (rec.depth_score, -rec.area))
        semantic_name = semantic_layer_name(layer_idx, total_layers)
        layer_name = f"layer_{layer_idx:02d}_{semantic_name}"
        for depth_rank, rec in enumerate(layer_regions):
            rec.layer_id = layer_idx
            rec.layer_name = layer_name
            rec.depth_rank = depth_rank
        scores_layer = [rec.depth_score for rec in layer_regions]
        layer_records.append(
            LayerRecord(
                layer_id=layer_idx,
                name=layer_name,
                region_ids=[rec.region_id for rec in layer_regions],
                num_regions=len(layer_regions),
                total_area=int(sum(rec.area for rec in layer_regions)),
                cluster_center_depth_score=float(np.mean(scores_layer)),
                mean_depth_score=float(np.mean(scores_layer)),
                min_depth_score=float(np.min(scores_layer)),
                max_depth_score=float(np.max(scores_layer)),
            )
        )

    records.sort(key=lambda rec: (rec.layer_id, rec.depth_score, -rec.area))
    return records, layer_records


# ---------- Inpainting helpers ----------
def load_inpaint_pipeline(model_name: str, device: torch.device):
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(model_name, torch_dtype=torch_dtype)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    pipe = pipe.to(device)
    return pipe


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1)
    return dilated > 0


def run_background_inpainting(
    image_pil: Image.Image,
    inpaint_mask: np.ndarray,
    pipe,
    *,
    prompt: str,
    negative_prompt: str,
    strength: float,
    guidance_scale: float,
    steps: int,
) -> Image.Image:
    mask_bool = inpaint_mask.astype(bool)
    if not mask_bool.any():
        return image_pil.copy()

    image_rgb = image_pil.convert("RGB")
    mask_image = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")

    width, height = image_rgb.size
    out_w = max(8, (width // 8) * 8)
    out_h = max(8, (height // 8) * 8)
    if (out_w, out_h) != (width, height):
        image_input = image_rgb.resize((out_w, out_h), Image.Resampling.LANCZOS)
        mask_input = mask_image.resize((out_w, out_h), Image.Resampling.NEAREST)
    else:
        image_input = image_rgb
        mask_input = mask_image

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_input,
        mask_image=mask_input,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=steps,
    ).images[0].convert("RGB")

    if result.size != (width, height):
        result = result.resize((width, height), Image.Resampling.LANCZOS)
    return result


# ---------- Visualization ----------
def save_depth_visual(depth_map: np.ndarray, path: str) -> None:
    depth_u8 = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    Image.fromarray(colorized).save(path)


def _alpha_sprite_from_mask(image_rgba: np.ndarray, mask: np.ndarray, bbox_xyxy: Sequence[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox_xyxy
    crop_rgb = image_rgba[y0:y1, x0:x1, :3].copy()
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    rgba = np.dstack([crop_rgb, crop_mask])
    return Image.fromarray(rgba, mode="RGBA")


def _layer_palette(num_layers: int) -> Dict[int, Tuple[int, int, int]]:
    colors: Dict[int, Tuple[int, int, int]] = {}
    denom = max(num_layers - 1, 1)
    for layer_id in range(num_layers):
        hue = layer_id / denom
        hsv = np.uint8([[[int(179 * hue), 200, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors[layer_id] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return colors


def save_mask_overlay(
    image_pil: Image.Image,
    region_records: List[RegionRecord],
    region_items: List[Dict],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = image_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = _layer_palette(max((rec.layer_id for rec in region_records), default=-1) + 1)

    for rec in sorted(region_records, key=lambda r: r.area, reverse=True):
        seg = region_items[rec.region_id]["mask"]
        rgb = colors[rec.layer_id]
        tint = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8)
        tint[seg] = (*rgb, 90)
        overlay = Image.alpha_composite(overlay, Image.fromarray(tint, mode="RGBA"))
        x0, y0, x1, y1 = rec.bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=rgb + (255,), width=2)
        draw.text((x0 + 4, y0 + 4), f"R{rec.region_id}:L{rec.layer_id}", fill=rgb + (255,))

    Image.alpha_composite(base, overlay).save(out_path)


def save_layered_visualization(
    image_pil: Image.Image,
    region_records: List[RegionRecord],
    region_items: List[Dict],
    out_path: str,
    *,
    layer_step_x: int,
    layer_step_y: int,
    scale_step: float,
    background_blur: float,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    w, h = image_pil.size
    max_layer_id = max((rec.layer_id for rec in region_records), default=0)
    num_layers = max_layer_id + 1
    center_idx = (num_layers - 1) / 2.0

    pad_x = max(80, layer_step_x * max(num_layers + 2, 5))
    pad_y = max(60, layer_step_y * max(num_layers + 2, 5))
    canvas = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (250, 250, 252, 255))

    bg = image_pil.convert("RGBA")
    if background_blur > 0:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=background_blur))
    canvas.alpha_composite(bg, (pad_x, pad_y))

    sorted_records = sorted(region_records, key=lambda r: (r.layer_id, r.depth_score, r.area))
    image_rgba = np.array(image_pil.convert("RGBA"))

    for rec in sorted_records:
        item = region_items[rec.region_id]
        sprite = _alpha_sprite_from_mask(image_rgba, item["mask"], rec.bbox_xyxy)
        x0, y0, x1, y1 = rec.bbox_xyxy
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        offset = rec.layer_id - center_idx
        shift_x = int(offset * layer_step_x)
        shift_y = int(-offset * layer_step_y)
        scale = max(0.2, 1.0 + offset * scale_step)

        new_w = max(2, int(sprite.size[0] * scale))
        new_h = max(2, int(sprite.size[1] * scale))
        sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

        paste_x = int(pad_x + cx - new_w / 2 + shift_x)
        paste_y = int(pad_y + cy - new_h / 2 + shift_y)

        shadow_alpha = np.array(sprite.getchannel("A"), dtype=np.uint8)
        shadow_rgba = np.zeros((sprite.size[1], sprite.size[0], 4), dtype=np.uint8)
        shadow_rgba[..., 3] = (shadow_alpha * 0.28).astype(np.uint8)
        shadow = Image.fromarray(shadow_rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=7))
        canvas.alpha_composite(shadow, (paste_x + 10, paste_y + 10))
        canvas.alpha_composite(sprite, (paste_x, paste_y))

    legend = ImageDraw.Draw(canvas)
    legend.text((24, 20), "Layerwise view by connected regions (far → near)", fill=(30, 30, 30, 255))
    canvas.save(out_path)


def _layer_union_mask(region_records: List[RegionRecord], region_items: List[Dict], layer_id: int) -> np.ndarray:
    if not region_items:
        raise ValueError("region_items must not be empty")

    union = np.zeros(region_items[0]["mask"].shape, dtype=bool)
    for rec in region_records:
        if rec.layer_id != layer_id:
            continue
        union |= region_items[rec.region_id]["mask"]
    return union


def save_layer_exports(
    image_pil: Image.Image,
    region_records: List[RegionRecord],
    region_items: List[Dict],
    layer_records: List[LayerRecord],
    output_dir: str,
    *,
    inpaint_pipe=None,
    inpaint_prompt: str = "",
    inpaint_negative_prompt: str = DEFAULT_INPAINT_NEGATIVE_PROMPT,
    inpaint_strength: float = 0.99,
    inpaint_guidance_scale: float = 7.5,
    inpaint_steps: int = 30,
    inpaint_mask_dilate: int = 9,
) -> Dict[str, str]:
    image_rgb = np.array(image_pil.convert("RGB"))
    layer_artifacts: Dict[str, str] = {}
    layers_dir = os.path.join(output_dir, "layers")
    masks_dir = os.path.join(output_dir, "masks")
    overlays_dir = os.path.join(output_dir, "overlays")
    os.makedirs(layers_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    layer_masks = {
        layer.layer_id: _layer_union_mask(region_records, region_items, layer.layer_id)
        for layer in layer_records
    }

    for layer in layer_records:
        layer_mask = layer_masks[layer.layer_id]
        alpha = layer_mask.astype(np.uint8) * 255
        rgba = np.dstack([image_rgb, alpha])

        layer_path = os.path.join(layers_dir, f"{layer.name}_layer.png")
        mask_path = os.path.join(masks_dir, f"{layer.name}_mask.png")
        Image.fromarray(rgba, mode="RGBA").save(layer_path)
        Image.fromarray(alpha, mode="L").save(mask_path)

        layer_artifacts[f"{layer.name}_layer"] = os.path.abspath(layer_path)
        layer_artifacts[f"{layer.name}_mask"] = os.path.abspath(mask_path)

    non_background_layers = [layer.layer_id for layer in layer_records[1:]]
    non_background_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for layer_id in non_background_layers:
        non_background_mask |= layer_masks[layer_id]

    inpaint_mask = _dilate_mask(non_background_mask, inpaint_mask_dilate)
    inpaint_mask_u8 = inpaint_mask.astype(np.uint8) * 255

    if inpaint_pipe is not None:
        inpainted_rgb = np.array(
            run_background_inpainting(
                image_pil,
                inpaint_mask,
                inpaint_pipe,
                prompt=inpaint_prompt,
                negative_prompt=inpaint_negative_prompt,
                strength=inpaint_strength,
                guidance_scale=inpaint_guidance_scale,
                steps=inpaint_steps,
            )
        )
    else:
        inpainted_bgr = cv2.inpaint(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            inpaint_mask_u8,
            5,
            cv2.INPAINT_TELEA,
        )
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    inpaint_path = os.path.join(layers_dir, "background_inpainted.png")
    holes_mask_path = os.path.join(masks_dir, "background_inpaint_mask.png")
    Image.fromarray(inpainted_rgb).save(inpaint_path)
    Image.fromarray(inpaint_mask_u8, mode="L").save(holes_mask_path)

    layer_artifacts["background_inpainted"] = os.path.abspath(inpaint_path)
    layer_artifacts["background_inpaint_mask"] = os.path.abspath(holes_mask_path)
    return layer_artifacts


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
    parser.add_argument(
        "--num-depth-layers",
        type=int,
        default=2,
        help="Number of depth layers for 1D k-means grouping.",
    )
    parser.add_argument(
        "--semantic-model",
        default=DEFAULT_SEMANTIC_MODEL,
        help="HF CLIP-like model used to derive auxiliary semantic/text embeddings per region.",
    )
    parser.add_argument(
        "--semantic-labels",
        default=",".join(DEFAULT_SEMANTIC_LABELS),
        help="Comma-separated label vocabulary used for auxiliary text-guided semantic region descriptors.",
    )
    parser.add_argument(
        "--semantic-aux-weight",
        type=float,
        default=0.035,
        help="How much semantic distance nudges clustering relative to depth distance. Keep small so depth dominates.",
    )
    parser.add_argument(
        "--semantic-depth-gate",
        type=float,
        default=0.18,
        help="Semantic cue is faded out as depth difference grows; smaller means semantics only disambiguate very similar depths.",
    )
    parser.add_argument("--layer-step-x", type=int, default=40)
    parser.add_argument("--layer-step-y", type=int, default=22)
    parser.add_argument("--scale-step", type=float, default=0.06)
    parser.add_argument("--background-blur", type=float, default=4.0)
    parser.add_argument(
        "--inpaint-model",
        default=DEFAULT_INPAINT_MODEL,
        help="HF diffusers inpainting model id/path for background fill.",
    )
    parser.add_argument(
        "--inpaint-prompt",
        default="clean coherent background, natural continuation, no foreground objects",
        help="Prompt used for the background inpainting model.",
    )
    parser.add_argument(
        "--inpaint-negative-prompt",
        default=DEFAULT_INPAINT_NEGATIVE_PROMPT,
        help="Negative prompt used for the background inpainting model.",
    )
    parser.add_argument("--inpaint-steps", type=int, default=30)
    parser.add_argument("--inpaint-guidance-scale", type=float, default=7.5)
    parser.add_argument("--inpaint-strength", type=float, default=0.99)
    parser.add_argument(
        "--inpaint-mask-dilate",
        type=int,
        default=9,
        help="Dilate the removed foreground/midground mask before inpainting.",
    )
    parser.add_argument(
        "--fallback-opencv-inpaint",
        action="store_true",
        help="If set, skip diffusers and use the old OpenCV TELEA inpaint fallback.",
    )
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
    region_items = collect_mask_regions(mask_items, min_region_area=args.min_mask_region_area)
    if not region_items:
        raise RuntimeError("No usable connected regions found after splitting SAM masks. Try lowering --min-mask-region-area.")

    semantic_labels = [label.strip() for label in args.semantic_labels.split(",") if label.strip()]
    if not semantic_labels:
        semantic_labels = list(DEFAULT_SEMANTIC_LABELS)
    semantic_bundle = load_semantic_encoder(args.semantic_model, device=device)
    semantic_features = build_region_semantic_features(
        image_pil,
        region_items,
        semantic_bundle,
        semantic_labels=semantic_labels,
        device=device,
    )
    for item, semantic in zip(region_items, semantic_features):
        item.update(semantic)

    region_records, layer_records = assign_depth_layers(
        region_items,
        depth_map,
        near_mode=args.near_mode,
        num_depth_layers=args.num_depth_layers,
    )
    if not region_records:
        raise RuntimeError("Depth scoring produced no valid regions.")

    inpaint_pipe = None
    used_inpaint_backend = "opencv-telea"
    if not args.fallback_opencv_inpaint:
        inpaint_pipe = load_inpaint_pipeline(args.inpaint_model, device=device)
        used_inpaint_backend = "diffusers-sdxl"

    overlays_dir = os.path.join(args.output_dir, "overlays")
    masks_dir = os.path.join(args.output_dir, "masks")
    layers_dir = os.path.join(args.output_dir, "layers")
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(layers_dir, exist_ok=True)

    depth_path = os.path.join(overlays_dir, "depth_map.png")
    overlay_path = os.path.join(overlays_dir, "mask_depth_overlay.png")
    layered_path = os.path.join(overlays_dir, "layered_visualization.png")
    metadata_path = os.path.join(args.output_dir, "mask_depth_metadata.json")

    save_depth_visual(depth_map, depth_path)
    save_mask_overlay(image_pil, region_records, region_items, overlay_path)
    save_layered_visualization(
        image_pil,
        region_records,
        region_items,
        layered_path,
        layer_step_x=args.layer_step_x,
        layer_step_y=args.layer_step_y,
        scale_step=args.scale_step,
        background_blur=args.background_blur,
    )
    layer_artifacts = save_layer_exports(
        image_pil,
        region_records,
        region_items,
        layer_records,
        args.output_dir,
        inpaint_pipe=inpaint_pipe,
        inpaint_prompt=args.inpaint_prompt,
        inpaint_negative_prompt=args.inpaint_negative_prompt,
        inpaint_strength=args.inpaint_strength,
        inpaint_guidance_scale=args.inpaint_guidance_scale,
        inpaint_steps=args.inpaint_steps,
        inpaint_mask_dilate=args.inpaint_mask_dilate,
    )

    grouped_layers = {
        layer.name: [asdict(rec) for rec in sorted(region_records, key=lambda r: (r.layer_id, r.depth_score, -r.area)) if rec.layer_id == layer.layer_id]
        for layer in layer_records
    }

    metadata = {
        "input_image": os.path.abspath(args.image),
        "sam_checkpoint": os.path.abspath(args.sam_checkpoint),
        "sam_model_type": args.sam_model_type,
        "depth_model": used_depth_model,
        "near_mode": args.near_mode,
        "semantic_model": args.semantic_model,
        "semantic_labels": semantic_labels,
        "semantic_aux_weight": float(args.semantic_aux_weight),
        "semantic_depth_gate": float(args.semantic_depth_gate),
        "inpaint_backend": used_inpaint_backend,
        "inpaint_model": None if args.fallback_opencv_inpaint else args.inpaint_model,
        "inpaint_prompt": args.inpaint_prompt,
        "inpaint_negative_prompt": args.inpaint_negative_prompt,
        "inpaint_steps": int(args.inpaint_steps),
        "inpaint_guidance_scale": float(args.inpaint_guidance_scale),
        "inpaint_strength": float(args.inpaint_strength),
        "inpaint_mask_dilate": int(args.inpaint_mask_dilate),
        "num_masks": len(mask_items),
        "num_regions": len(region_records),
        "num_layers": len(layer_records),
        "num_depth_layers_requested": int(args.num_depth_layers),
        "num_depth_layers_effective": len(layer_records),
        "depth_grouping": "kmeans_1d_mean_depth",
        "layers": [asdict(layer) for layer in layer_records],
        "regions_by_layer": grouped_layers,
        "artifacts": {
            "depth_map": os.path.abspath(depth_path),
            "mask_depth_overlay": os.path.abspath(overlay_path),
            "layered_visualization": os.path.abspath(layered_path),
            **layer_artifacts,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    print("\nNew inspection artifacts:")
    for key, value in metadata["artifacts"].items():
        print(f"- {key}: {value}")
    print(f"\nSaved outputs to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
