import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from main import init_diffvg, init_optimizer, load_config
from utils.img_process import (
    connect_mask_interior_exterior,
    get_mean_color,
    init_path_by_mask,
    rgba_to_rgb,
    svg_to_img,
)


@dataclass
class LayerAsset:
    layer_id: int
    name: str
    image_path: str
    mask_path: str
    rgba: np.ndarray
    rgb: np.ndarray
    mask_u8: np.ndarray
    mask_bool: np.ndarray
    area: int


@dataclass
class LayerwiseBundle:
    width: int
    height: int
    layers: List[LayerAsset]
    background_image: np.ndarray
    metadata: Dict


def _load_rgba(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGBA"))
    return arr


def _load_mask(path: str, size: Tuple[int, int] | None = None) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    if size is not None and (mask.shape[1], mask.shape[0]) != size:
        mask = np.array(Image.fromarray(mask).resize(size, Image.NEAREST))
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _composite_rgba(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    base = base_rgb.astype(np.float32) / 255.0
    over_rgb = overlay_rgba[..., :3].astype(np.float32) / 255.0
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    out = over_rgb * alpha + base * (1.0 - alpha)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _component_masks(mask_u8: np.ndarray, limit: Optional[int] = None) -> List[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    comps: List[Tuple[int, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = np.where(labels == label, 255, 0).astype(np.uint8)
        comps.append((area, comp))
    comps.sort(key=lambda x: x[0], reverse=True)
    if limit is not None:
        comps = comps[:limit]
    return [mask for _, mask in comps]


def _residual_component_masks(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    allowed_mask_u8: np.ndarray,
    count: int,
    diff_threshold: float = 0.02,
) -> List[np.ndarray]:
    if count <= 0:
        return []
    pred = pred_rgb.astype(np.float32) / 255.0
    target = target_rgb.astype(np.float32) / 255.0
    diff = np.mean((pred - target) ** 2, axis=2)
    allowed = allowed_mask_u8 > 0
    diff = np.where(allowed, diff, 0.0)
    binary = (diff > diff_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps: List[Tuple[int, np.ndarray]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = np.where(labels == label, 255, 0).astype(np.uint8)
        comp = cv2.bitwise_and(comp, allowed_mask_u8)
        if comp.sum() <= 0:
            continue
        comps.append((area, comp))
    comps.sort(key=lambda x: x[0], reverse=True)
    return [mask for _, mask in comps[:count]]


def load_layerwise_bundle(layer_dir: str) -> LayerwiseBundle:
    root = Path(layer_dir)
    meta_path = root / "mask_depth_metadata.json"
    metadata: Dict = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    artifacts = metadata.get("artifacts", {})
    layers_meta = metadata.get("layers", [])

    layers: List[LayerAsset] = []
    for idx, layer_meta in enumerate(layers_meta):
        layer_id = int(layer_meta["layer_id"])
        name = str(layer_meta["name"])
        image_path = artifacts.get(f"{name}_layer") or str(root / "layers" / f"{name}_layer.png")
        mask_path = artifacts.get(f"{name}_mask") or str(root / "masks" / f"{name}_mask.png")
        if not (Path(image_path).exists() and Path(mask_path).exists()):
            continue
        rgba = _load_rgba(image_path)
        mask_u8 = _load_mask(mask_path, size=(rgba.shape[1], rgba.shape[0]))
        rgba[..., 3] = mask_u8
        rgb = rgba[..., :3]
        layers.append(
            LayerAsset(
                layer_id=layer_id,
                name=name,
                image_path=str(image_path),
                mask_path=str(mask_path),
                rgba=rgba,
                rgb=rgb,
                mask_u8=mask_u8,
                mask_bool=mask_u8 > 0,
                area=int((mask_u8 > 0).sum()),
            )
        )

    if not layers:
        layers_dir = root / "layers"
        masks_dir = root / "masks"
        layer_image_paths = sorted(layers_dir.glob("layer_*_layer.png"))
        for i, image_path in enumerate(layer_image_paths):
            name = image_path.stem[:-6] if image_path.stem.endswith("_layer") else image_path.stem
            mask_path = masks_dir / f"{name}_mask.png"
            if not mask_path.exists():
                continue
            rgba = _load_rgba(str(image_path))
            mask_u8 = _load_mask(str(mask_path), size=(rgba.shape[1], rgba.shape[0]))
            rgba[..., 3] = mask_u8
            layers.append(
                LayerAsset(
                    layer_id=i,
                    name=name,
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    rgba=rgba,
                    rgb=rgba[..., :3],
                    mask_u8=mask_u8,
                    mask_bool=mask_u8 > 0,
                    area=int((mask_u8 > 0).sum()),
                )
            )

    if not layers:
        raise RuntimeError(f"No layer assets found under {layer_dir}")

    layers.sort(key=lambda x: x.layer_id)
    background_path = artifacts.get("background_inpainted") or str(root / "layers" / "background_inpainted.png")
    if Path(background_path).exists():
        background_image = np.array(Image.open(background_path).convert("RGB"))
    else:
        background_image = layers[0].rgb.copy()

    height, width = background_image.shape[:2]
    return LayerwiseBundle(width=width, height=height, layers=layers, background_image=background_image, metadata=metadata)


def build_composite_targets(bundle: LayerwiseBundle) -> List[np.ndarray]:
    composites: List[np.ndarray] = []
    current = bundle.background_image.copy()
    for layer in bundle.layers:
        current = _composite_rgba(current, layer.rgba)
        composites.append(current.copy())
    return composites


def parse_layer_budgets(layers: Sequence[LayerAsset], total_budget: int, args) -> Dict[str, int]:
    budget_map: Dict[str, int] = {}
    if getattr(args, "layer_budgets_json", None):
        src = args.layer_budgets_json
        if os.path.exists(src):
            budget_map.update(json.loads(Path(src).read_text()))
        else:
            budget_map.update(json.loads(src))

    for item in getattr(args, "layer_budget", []) or []:
        name, value = item.split("=", 1)
        budget_map[name] = int(value)

    resolved: Dict[str, int] = {}
    area_sum = max(1, sum(layer.area for layer in layers))
    unspecified = [layer for layer in layers if layer.name not in budget_map and str(layer.layer_id) not in budget_map]

    used = 0
    for layer in layers:
        key_id = str(layer.layer_id)
        if layer.name in budget_map:
            val = int(budget_map[layer.name])
        elif key_id in budget_map:
            val = int(budget_map[key_id])
        else:
            val = -1
        if val >= 0:
            resolved[layer.name] = val
            used += val

    remaining = max(0, total_budget - used)
    if unspecified:
        raw = [remaining * (layer.area / area_sum) for layer in unspecified]
        assigned = [int(x) for x in raw]
        remainder = remaining - sum(assigned)
        ranked = sorted(range(len(unspecified)), key=lambda i: raw[i] - assigned[i], reverse=True)
        for k in range(remainder):
            assigned[ranked[k]] += 1
        for layer, value in zip(unspecified, assigned):
            resolved[layer.name] = max(1 if layer.area > 0 else 0, value)

    return resolved


def build_layer_shapes(layer: LayerAsset, target_rgb: np.ndarray, budget: int, epsilon: int) -> Tuple[List, List]:
    if budget <= 0:
        return [], []

    component_masks = _component_masks(layer.mask_u8, limit=budget)
    shapes = []
    shape_groups = []
    for mask in component_masks:
        usable = connect_mask_interior_exterior(mask)
        path = init_path_by_mask(usable, epsilon=epsilon)
        color = get_mean_color(target_rgb, mask)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([len(shapes)]),
            fill_color=torch.FloatTensor(list(color) + [255]) / 255.0,
            stroke_color=torch.FloatTensor([0, 0, 0, 1]),
        )
        shapes.append(path)
        shape_groups.append(group)

    return shapes, shape_groups


def add_residual_shapes_for_layer(
    device,
    canvas_wh: Tuple[int, int],
    all_shapes: List,
    all_shape_groups: List,
    layer: LayerAsset,
    target_rgb: np.ndarray,
    budget_left: int,
    epsilon: int,
) -> Tuple[List, List]:
    if budget_left <= 0:
        return [], []

    width, height = canvas_wh
    if all_shapes:
        pred = svg_to_img(width, height, all_shapes, all_shape_groups, device)
        pred = rgba_to_rgb(pred, device).permute(1, 2, 0).detach().cpu().numpy()
        pred_rgb = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    else:
        pred_rgb = np.full((height, width, 3), 255, dtype=np.uint8)

    residual_masks = _residual_component_masks(pred_rgb, target_rgb, layer.mask_u8, budget_left)
    shapes = []
    groups = []
    for mask in residual_masks:
        usable = connect_mask_interior_exterior(mask)
        path = init_path_by_mask(usable, epsilon=epsilon)
        color = get_mean_color(target_rgb, mask)
        groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([len(shapes)]),
                fill_color=torch.FloatTensor(list(color) + [255]) / 255.0,
                stroke_color=torch.FloatTensor([0, 0, 0, 1]),
            )
        )
        shapes.append(path)
    return shapes, groups


def _reindex_shape_groups(shape_groups: List) -> None:
    for i, group in enumerate(shape_groups):
        group.shape_ids = torch.LongTensor([i])


def _render_rgb(device, width: int, height: int, shapes: List, shape_groups: List, bg: Optional[torch.Tensor] = None):
    img = svg_to_img(width, height, shapes, shape_groups, device)
    return rgba_to_rgb(img, device, para_bg=bg)


def layerwise_optimize(
    device,
    width: int,
    height: int,
    shapes: List,
    shape_groups: List,
    target_rgb: np.ndarray,
    train_conf: Dict,
    base_lr_conf: Dict,
    *,
    is_opt_list: List[int],
    active_layer_indices: List[int],
    active_layer_mask_u8: np.ndarray,
    containment_weight: float,
    output_dir: str,
    start_count: int = 0,
) -> int:
    target = torch.tensor(target_rgb, device=device).float() / 255.0
    target = target.permute(2, 0, 1)
    active_mask = torch.tensor((active_layer_mask_u8 > 0).astype(np.float32), device=device)
    outside_mask = 1.0 - active_mask
    black_bg = torch.tensor([0.0, 0.0, 0.0], requires_grad=False, device=device)

    optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_visual_color"],
        is_opt_list,
        lr_base=base_lr_conf,
    )

    white_groups = []
    for idx in active_layer_indices:
        white_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([len(white_groups)]),
                fill_color=torch.FloatTensor([1, 1, 1, 1]),
                stroke_color=torch.FloatTensor([1, 1, 1, 1]),
            )
        )

    active_shapes = [shapes[idx] for idx in active_layer_indices]

    count = start_count
    num_iters = int(train_conf["visual_opt_num_iters"])
    with tqdm(total=num_iters, desc="Layerwise optimization", unit="iter") as pbar:
        for _ in range(num_iters):
            img = _render_rgb(device, width, height, shapes, shape_groups)
            loss_rgb = F.mse_loss(img, target)

            active_render = _render_rgb(device, width, height, active_shapes, white_groups, bg=black_bg)
            active_alpha = active_render.mean(dim=0)
            loss_outside = torch.mean(active_alpha * outside_mask)

            loss = loss_rgb + containment_weight * loss_outside

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pydiffvg.save_svg(os.path.join(output_dir, f"{count:04d}.svg"), width, height, shapes, shape_groups)
            count += 1
            pbar.update(1)

    return count


def save_path_metadata(output_dir: str, layers: Sequence[LayerAsset], path_to_layer: Sequence[int]) -> None:
    layer_by_id = {layer.layer_id: layer for layer in layers}
    payload = {
        "num_paths": len(path_to_layer),
        "paths": [
            {
                "path_index": idx,
                "layer_id": int(layer_id),
                "layer_name": layer_by_id[layer_id].name if layer_id in layer_by_id else f"layer_{layer_id}",
            }
            for idx, layer_id in enumerate(path_to_layer)
        ],
    }
    Path(output_dir, "path_layer_map.json").write_text(json.dumps(payload, indent=2))


def layered_vectorization_from_layerwise(args, device):
    bundle = load_layerwise_bundle(args.layer_input_dir)
    composites = build_composite_targets(bundle)

    exp_dir = os.path.join(args.output_root, args.file_save_name)
    os.makedirs(exp_dir, exist_ok=True)
    stage_dir = os.path.join(exp_dir, "layer_steps")
    os.makedirs(stage_dir, exist_ok=True)

    budgets = parse_layer_budgets(bundle.layers, args.max_path_num_limit, args)

    all_shapes: List = []
    all_shape_groups: List = []
    path_to_layer: List[int] = []
    save_count = 0

    for layer_idx, layer in enumerate(bundle.layers):
        target_rgb = composites[layer_idx]
        budget = int(budgets.get(layer.name, 0))
        current_shapes, current_groups = build_layer_shapes(layer, target_rgb, budget, args.approxpolydp_epsilon)
        spent = len(current_shapes)

        _reindex_shape_groups(current_groups)
        base_index = len(all_shapes)
        all_shapes.extend(current_shapes)
        all_shape_groups.extend(current_groups)
        path_to_layer.extend([layer.layer_id] * len(current_shapes))

        extra_shapes, extra_groups = add_residual_shapes_for_layer(
            device,
            (bundle.width, bundle.height),
            all_shapes,
            all_shape_groups,
            layer,
            target_rgb,
            max(0, budget - spent),
            args.approxpolydp_epsilon,
        )
        _reindex_shape_groups(extra_groups)
        all_shapes.extend(extra_shapes)
        all_shape_groups.extend(extra_groups)
        path_to_layer.extend([layer.layer_id] * len(extra_shapes))

        _reindex_shape_groups(all_shape_groups)
        active_indices = [i for i, lid in enumerate(path_to_layer) if lid == layer.layer_id]
        if not active_indices:
            continue

        is_opt_list = [0] * len(all_shapes)
        for i in active_indices:
            is_opt_list[i] = 1

        layer_step_dir = os.path.join(stage_dir, f"{layer.layer_id:02d}_{layer.name}")
        os.makedirs(layer_step_dir, exist_ok=True)
        save_count = layerwise_optimize(
            device,
            bundle.width,
            bundle.height,
            all_shapes,
            all_shape_groups,
            target_rgb,
            args.train,
            args.base_lr,
            is_opt_list=is_opt_list,
            active_layer_indices=active_indices,
            active_layer_mask_u8=layer.mask_u8,
            containment_weight=args.outside_mask_loss_weight,
            output_dir=layer_step_dir,
            start_count=save_count,
        )

        pydiffvg.save_svg(
            os.path.join(exp_dir, f"after_{layer.name}.svg"),
            bundle.width,
            bundle.height,
            all_shapes,
            all_shape_groups,
        )

    pydiffvg.save_svg(
        os.path.join(exp_dir, "final.svg"),
        bundle.width,
        bundle.height,
        all_shapes,
        all_shape_groups,
    )
    save_path_metadata(exp_dir, bundle.layers, path_to_layer)

    summary = {
        "layer_input_dir": os.path.abspath(args.layer_input_dir),
        "budgets": budgets,
        "num_layers": len(bundle.layers),
        "num_paths": len(all_shapes),
        "outside_mask_loss_weight": args.outside_mask_loss_weight,
    }
    Path(exp_dir, "layerwise_run_summary.json").write_text(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental layerwise LayerVec prototype.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/base_config.yaml",
        help="YAML/YML file for configuration.",
    )
    parser.add_argument("--layer-input-dir", type=str, required=True, help="Directory produced by layerwise_images.py.")
    parser.add_argument("-fsn", "--file_save_name", type=str, default="layerwise_proto", help="Output experiment name.")
    parser.add_argument("--output_root", type=str, default="./workdir", help="Root directory for experiment outputs.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer-budgets-json", type=str, default=None, help="JSON object or path to JSON mapping layer name/id to budget.")
    parser.add_argument("--layer-budget", action="append", default=[], help="Repeated override like layer_08_midground=12 or 8=12.")
    parser.add_argument("--outside-mask-loss-weight", type=float, default=0.25, help="Penalty weight for rendered layer coverage outside its mask.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = load_config(args.config, args)

    device = torch.device(args.device)
    init_diffvg(device=device)
    layered_vectorization_from_layerwise(args, device)
