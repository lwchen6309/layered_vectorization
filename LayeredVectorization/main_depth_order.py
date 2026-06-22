import os
import glob
import shutil
import argparse
import time
import json
from typing import Optional, List

import yaml
import torch
import torch.nn.functional as F
import pydiffvg
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

from utils.img_process import *
from sds_image_simplicity import sds_based_simplification
from depth_main import _build_depth_layered_masks


def init_diffvg(
    device: torch.device,
    use_gpu: bool = torch.cuda.is_available(),
    print_timing: bool = False,
):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)


def get_exp_dir(args) -> str:
    return os.path.join(args.output_root, args.file_save_name)


def init_optimizer(
    shapes,
    shape_groups,
    is_train_stroke: bool = False,
    is_train_color: bool = True,
    is_opt_list: List[int] = [],
    lr_base: dict = {},
):
    points_vars = []
    color_vars = []
    stroke_width_vars = []
    stroke_color_vars = []

    if len(is_opt_list) == 0:
        is_opt_list = [1 for _ in range(len(shapes))]

    for i, path in enumerate(shapes):
        is_active = is_opt_list[i] == 1
        path.points.requires_grad = is_active
        if is_train_stroke:
            path.stroke_width.requires_grad = is_active
        if is_active:
            points_vars.append(path.points)
            if is_train_stroke:
                stroke_width_vars.append(path.stroke_width)

    if is_train_color:
        for i, group in enumerate(shape_groups):
            is_active = is_opt_list[i] == 1
            group.fill_color.requires_grad = is_active
            if is_train_stroke:
                group.stroke_color.requires_grad = is_active
            if is_active:
                color_vars.append(group.fill_color)
                if is_train_stroke:
                    stroke_color_vars.append(group.stroke_color)

    params = {"point": points_vars}
    if is_train_color:
        params["color"] = color_vars
    if is_train_stroke:
        params["stroke_width"] = stroke_width_vars
        params["stroke_color"] = stroke_color_vars

    learnable_params = [
        {"params": params[k], "lr": lr_base[k], "_id": str(k)}
        for k in sorted(params.keys())
        if len(params[k]) > 0
    ]

    svg_optimizer = torch.optim.Adam(
        learnable_params,
        betas=(0.9, 0.9),
        eps=1e-6,
    )
    return svg_optimizer


def exclude_loss(raster_img, scale=1):
    img = F.relu(178 / 255 - raster_img)
    loss = torch.sum(img) * scale
    return loss


def mask_to_torch(mask: np.ndarray, device) -> torch.Tensor:
    mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
    return torch.tensor(mask, device=device).unsqueeze(0)


def masked_mse_loss(img: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum() * img.shape[0], min=1.0)
    return torch.sum((img - target) ** 2 * mask) / denom


def _path_depth_score(path, depth_map: np.ndarray, near_mode: str = "large") -> float:
    points = path.points.detach().float().cpu().numpy()
    if points.size == 0:
        return 0.0

    h, w = depth_map.shape[:2]
    xs = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    raw_depth = float(np.median(depth_map[ys, xs]))
    return raw_depth if near_mode == "large" else (1.0 - raw_depth)


def sort_shapes_back_to_front_by_depth(
    shapes,
    shape_groups,
    depth_map: np.ndarray,
    near_mode: str = "large",
):
    scored = []
    for index, (shape, group) in enumerate(zip(shapes, shape_groups)):
        scored.append(
            {
                "old_index": index,
                "score": _path_depth_score(shape, depth_map, near_mode=near_mode),
                "shape": shape,
                "group": group,
            }
        )

    # pydiffvg draws later paths on top, so far-to-near gives correct occlusion.
    scored.sort(key=lambda item: (item["score"], item["old_index"]))
    sorted_shapes = [item["shape"] for item in scored]
    sorted_groups = [item["group"] for item in scored]
    for new_index, group in enumerate(sorted_groups):
        group.shape_ids = torch.LongTensor([new_index])

    order = [
        {
            "new_index": new_index,
            "old_index": item["old_index"],
            "depth_score": item["score"],
        }
        for new_index, item in enumerate(scored)
    ]
    return sorted_shapes, sorted_groups, order


def sanitize_layered_masks(layerd_struct_masks: list) -> list:
    sanitized = []
    for layer in layerd_struct_masks:
        clean_layer = []
        for mask in layer:
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            clean_layer.append(np.ascontiguousarray(mask))
        if clean_layer:
            sanitized.append(clean_layer)
    return sanitized


def sort_layered_masks_back_to_front_by_depth(
    layerd_struct_masks: list,
    depth_map: np.ndarray,
    near_mode: str = "large",
):
    flat_masks = []
    for layer_index, layer in enumerate(layerd_struct_masks):
        for mask_index, mask in enumerate(layer):
            seg = mask > 0
            if np.any(seg):
                raw_depth = float(np.median(depth_map[seg]))
            else:
                raw_depth = 0.0
            depth_score = raw_depth if near_mode == "large" else (1.0 - raw_depth)
            flat_masks.append(
                {
                    "old_layer": layer_index,
                    "old_index": mask_index,
                    "depth_score": depth_score,
                    "area": int(np.sum(seg)),
                    "mask": mask,
                }
            )

    if not flat_masks:
        return layerd_struct_masks, []

    flat_masks.sort(key=lambda item: (item["depth_score"], -item["area"]))
    layer_count = max(1, len(layerd_struct_masks))
    sorted_layers = [[] for _ in range(layer_count)]
    order_meta = []

    for rank, item in enumerate(flat_masks):
        layer_index = min(layer_count - 1, rank * layer_count // len(flat_masks))
        sorted_layers[layer_index].append(item["mask"])
        order_meta.append(
            {
                "new_rank": rank,
                "new_layer": layer_index,
                "old_layer": item["old_layer"],
                "old_index": item["old_index"],
                "depth_score": item["depth_score"],
                "area": item["area"],
            }
        )

    sorted_layers = [layer for layer in sorted_layers if layer]
    return sorted_layers, order_meta


def _kmeans_1d(values: np.ndarray, k: int, num_iters: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)

    k = max(1, min(k, len(values)))
    centers = np.quantile(values, np.linspace(0.0, 1.0, k)).astype(np.float32)
    labels = np.zeros(len(values), dtype=np.int64)

    for _ in range(num_iters):
        distances = np.abs(values[:, None] - centers[None, :])
        next_labels = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for label in range(k):
            assigned = values[next_labels == label]
            if len(assigned) > 0:
                next_centers[label] = float(np.mean(assigned))

        if np.array_equal(next_labels, labels) and np.allclose(next_centers, centers):
            break
        labels = next_labels
        centers = next_centers

    center_order = np.argsort(centers)
    remap = {int(old_label): int(new_label) for new_label, old_label in enumerate(center_order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int64)


def kmeans_layered_masks_by_depth(
    layerd_struct_masks: list,
    depth_map: np.ndarray,
    near_mode: str = "large",
    num_depth_layers: int = 3,
):
    records = []
    for layer_index, layer in enumerate(layerd_struct_masks):
        for mask_index, mask in enumerate(layer):
            seg = mask > 0
            if np.any(seg):
                raw_depth = float(np.median(depth_map[seg]))
            else:
                raw_depth = 0.0
            depth_score = raw_depth if near_mode == "large" else (1.0 - raw_depth)
            records.append(
                {
                    "old_layer": layer_index,
                    "old_index": mask_index,
                    "raw_depth": raw_depth,
                    "depth_score": depth_score,
                    "area": int(np.sum(seg)),
                    "mask": mask,
                }
            )

    if not records:
        return layerd_struct_masks, []

    labels = _kmeans_1d(
        np.asarray([record["depth_score"] for record in records], dtype=np.float32),
        k=num_depth_layers,
    )
    layer_count = int(labels.max()) + 1
    sorted_layers = [[] for _ in range(layer_count)]
    order_meta = []

    for record, label in sorted(zip(records, labels), key=lambda item: (int(item[1]), item[0]["depth_score"], -item[0]["area"])):
        layer_index = int(label)
        sorted_layers[layer_index].append(record["mask"])
        order_meta.append(
            {
                "new_layer": layer_index,
                "old_layer": record["old_layer"],
                "old_index": record["old_index"],
                "raw_depth": record["raw_depth"],
                "depth_score": record["depth_score"],
                "area": record["area"],
            }
        )

    sorted_layers = [layer for layer in sorted_layers if layer]
    return sorted_layers, order_meta


def pixel_depth_cluster_layers(
    layerd_struct_masks: list,
    depth_map: np.ndarray,
    near_mode: str = "large",
    num_depth_layers: int = 3,
):
    """Partition all pixels by depth, then clip SAM masks into those layers."""
    depth_score = depth_map if near_mode == "large" else (1.0 - depth_map)
    labels = _kmeans_1d(depth_score.reshape(-1), k=num_depth_layers).reshape(depth_map.shape)
    layer_count = int(labels.max()) + 1
    depth_layers = [(labels == layer_index).astype(np.uint8) * 255 for layer_index in range(layer_count)]

    path_layers = [[] for _ in range(layer_count)]
    owner_layers = [[depth_layer] for depth_layer in depth_layers]
    order_meta = [
        {
            "new_layer": layer_index,
            "old_layer": None,
            "old_index": None,
            "raw_depth": None,
            "depth_score": None,
            "area": int(np.sum(depth_layer > 0)),
            "source": "pixel_depth_cluster_base",
        }
        for layer_index, depth_layer in enumerate(depth_layers)
    ]
    flat_masks = []
    for old_layer, masks in enumerate(layerd_struct_masks):
        for old_index, mask in enumerate(masks):
            flat_masks.append((old_layer, old_index, mask))

    for old_layer, old_index, mask in flat_masks:
        mask_bool = mask > 0
        if not np.any(mask_bool):
            continue
        raw_depth = float(np.median(depth_map[mask_bool]))
        mask_depth_score = raw_depth if near_mode == "large" else (1.0 - raw_depth)
        for layer_index, depth_layer in enumerate(depth_layers):
            clipped = np.logical_and(mask_bool, depth_layer > 0).astype(np.uint8) * 255
            area = int(np.sum(clipped > 0))
            if area <= 0:
                continue
            path_layers[layer_index].append(np.ascontiguousarray(clipped))
            order_meta.append(
                {
                    "new_layer": layer_index,
                    "old_layer": old_layer,
                    "old_index": old_index,
                    "raw_depth": raw_depth,
                    "depth_score": mask_depth_score,
                    "area": area,
                    "source": "sam_mask_clipped_to_pixel_depth_cluster",
                }
            )
    for layer_index, depth_layer in enumerate(depth_layers):
        if path_layers[layer_index]:
            geometry_union = combine_binary_masks(path_layers[layer_index])
        else:
            geometry_union = np.zeros_like(depth_layer, dtype=np.uint8)
        residual = np.logical_and(depth_layer > 0, geometry_union == 0).astype(np.uint8) * 255
        for component in split_binary_mask_components(residual):
            path_layers[layer_index].append(component)
            order_meta.append(
                {
                    "new_layer": layer_index,
                    "old_layer": None,
                    "old_index": None,
                    "raw_depth": None,
                    "depth_score": None,
                    "area": int(np.sum(component > 0)),
                    "source": "owner_residual_fallback",
                }
            )
    for layer_index, masks in enumerate(path_layers):
        if masks:
            continue
        path_layers[layer_index].append(np.ascontiguousarray(depth_layers[layer_index]))
        order_meta.append(
            {
                "new_layer": layer_index,
                "old_layer": None,
                "old_index": None,
                "raw_depth": None,
                "depth_score": None,
                "area": int(np.sum(depth_layers[layer_index] > 0)),
                "source": "empty_layer_depth_cluster_fallback",
            }
        )
    return path_layers, owner_layers, order_meta


def save_depth_layer_masks(layerd_struct_masks: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for layer_index, masks in enumerate(layerd_struct_masks):
        if not masks:
            continue
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined = np.maximum(combined, np.where(mask > 0, 255, 0).astype(np.uint8))
        Image.fromarray(combined).save(os.path.join(out_dir, f"depth_layer_{layer_index:02d}.png"))


def get_layer_shape_ranges(layerd_struct_masks: list) -> list:
    ranges = []
    start = 0
    for masks in layerd_struct_masks:
        end = start + len(masks)
        ranges.append((start, end))
        start = end
    return ranges


def combine_binary_masks(masks: list) -> np.ndarray:
    if not masks:
        raise ValueError("Cannot combine an empty mask list.")
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, np.where(mask > 0, 255, 0).astype(np.uint8))
    return combined


def split_binary_mask_components(mask: np.ndarray, min_area: int = 16) -> list:
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        components.append(np.ascontiguousarray(component))
    return components


def ensure_depth_layers_cover_canvas(layerd_struct_masks: list, out_dir: str):
    if not layerd_struct_masks:
        return layerd_struct_masks, 0

    union_mask = np.zeros_like(layerd_struct_masks[0][0], dtype=np.uint8)
    for masks in layerd_struct_masks:
        union_mask = np.maximum(union_mask, combine_binary_masks(masks))

    uncovered = np.where(union_mask > 0, 0, 255).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(uncovered).save(os.path.join(out_dir, "uncovered_pixels.png"))
    uncovered_area = int(np.sum(uncovered > 0))
    with open(os.path.join(out_dir, "coverage.txt"), "w", encoding="utf-8") as f:
        total_area = int(uncovered.shape[0] * uncovered.shape[1])
        f.write(f"total_pixels={total_area}\n")
        f.write(f"uncovered_pixels={uncovered_area}\n")
        f.write(f"uncovered_ratio={uncovered_area / max(1, total_area):.8f}\n")

    if uncovered_area > 0:
        canvas_mask = np.ones_like(uncovered, dtype=np.uint8) * 255
        layerd_struct_masks[0] = [canvas_mask] + layerd_struct_masks[0]
    return layerd_struct_masks, uncovered_area


def svg_optimize_img_struct(
    device,
    shapes,
    shape_groups,
    target_img: np.ndarray,
    layerd_struct_masks: list,
    file_save_path: str,
    train_conf: dict,
    base_lr_conf: dict,
    layer_loss_masks: Optional[list] = None,
):
    struct_target_imgs, struct_colors_list = init_struct_target_imgs(layerd_struct_masks)
    struct_target_imgs = [x.to(device) for x in struct_target_imgs]

    struct_shape_groups_list = []
    for struct_colors in struct_colors_list:
        struct_shape_groups = []
        for i, color in enumerate(struct_colors):
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=torch.FloatTensor(color + [1]),
                stroke_color=torch.FloatTensor([0, 0, 0, 1]),
            )
            struct_shape_groups.append(path_group)
        struct_shape_groups_list.append(struct_shape_groups)

    transparent_shape_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
            stroke_color=torch.FloatTensor([0, 0, 0, 0.3]),
        )
        transparent_shape_groups.append(path_group)

    black_bg = torch.tensor([0.0, 0.0, 0.0], requires_grad=False, device=device)
    white_bg = torch.tensor([1.0, 1.0, 1.0], requires_grad=False, device=device)

    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img, device=device) / 255.0
    target_img = target_img.permute(2, 0, 1)

    if train_conf.get("three_pass_depth_layervec", train_conf.get("depth_layer_sequential_fit", False)):
        layer_ranges = get_layer_shape_ranges(layerd_struct_masks)
        loss_mask_layers = layer_loss_masks if layer_loss_masks is not None else layerd_struct_masks
        layer_masks = [mask_to_torch(combine_binary_masks(masks), device) for masks in loss_mask_layers]
        layer_num_iters = int(
            train_conf.get(
                "three_pass_struct_num_iters",
                train_conf.get(
                    "depth_layer_struct_num_iters",
                    train_conf["struct_opt_num_iters"],
                ),
            )
        )
        global_iter = 0
        for struct_i, (start, end) in enumerate(layer_ranges):
            if start == end:
                continue

            active_list = [0 for _ in range(len(shapes))]
            for index in range(start, end):
                active_list[index] = 1

            svg_optimizer = init_optimizer(
                shapes,
                shape_groups,
                train_conf["is_train_stroke"],
                train_conf["is_train_struct_color"],
                active_list,
                lr_base=base_lr_conf,
            )

            cur_masks = layerd_struct_masks[struct_i]
            layer_mask = layer_masks[struct_i]
            with tqdm(
                total=layer_num_iters,
                desc=f"LayerVec pass {struct_i}",
                unit="iter",
            ) as pbar:
                for _ in range(layer_num_iters):
                    visible_shapes = shapes[:end]
                    visible_shape_groups = shape_groups[:end]
                    img = svg_to_img(
                        img_width,
                        img_height,
                        visible_shapes,
                        visible_shape_groups,
                        device,
                    )
                    img = rgba_to_rgb(img, device, white_bg)
                    loss_mse = masked_mse_loss(img, target_img, layer_mask)

                    struct_img = svg_to_img(
                        img_width,
                        img_height,
                        shapes[start:end],
                        struct_shape_groups_list[struct_i],
                        device,
                    )
                    struct_img = rgba_to_rgb(struct_img, device, black_bg)
                    loss_struct = F.mse_loss(struct_img, struct_target_imgs[struct_i])

                    if train_conf.get("disable_exclude_loss", False):
                        loss_exclude = 0
                    else:
                        transparent_img = svg_to_img(
                            img_width,
                            img_height,
                            shapes[start:end],
                            transparent_shape_groups[:len(cur_masks)],
                            device,
                        )
                        transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                        loss_exclude = exclude_loss(transparent_img, scale=2e-7)

                    loss = loss_mse * 0.02 + loss_exclude + loss_struct
                    svg_optimizer.zero_grad()
                    loss.backward()
                    svg_optimizer.step()

                    pydiffvg.save_svg(
                        os.path.join(file_save_path, f"{global_iter}.svg"),
                        img_width,
                        img_height,
                        shapes,
                        shape_groups,
                    )
                    global_iter += 1
                    pbar.update(1)

        return shapes, shape_groups

    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_struct_color"],
        lr_base=base_lr_conf,
    )

    with tqdm(
        total=train_conf["struct_opt_num_iters"],
        desc="Struct optimization",
        unit="iter",
    ) as pbar:
        for i in range(train_conf["struct_opt_num_iters"]):
            loss_struct = 0
            loss_exclude = 0
            shape_index = 0

            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device, white_bg)
            loss_mse = F.mse_loss(img, target_img)

            if not train_conf.get("single_render_struct_loss", False):
                for struct_i, struct_target_img in enumerate(struct_target_imgs):
                    cur_masks = layerd_struct_masks[struct_i]
                    shape_index += len(cur_masks)

                    struct_img = svg_to_img(
                        img_width,
                        img_height,
                        shapes[shape_index - len(cur_masks):shape_index],
                        struct_shape_groups_list[struct_i],
                        device,
                    )
                    struct_img = rgba_to_rgb(struct_img, device, black_bg)
                    loss_struct += F.mse_loss(struct_img, struct_target_img)

                    if not train_conf.get("disable_exclude_loss", False):
                        transparent_img = svg_to_img(
                            img_width,
                            img_height,
                            shapes[shape_index - len(cur_masks):shape_index],
                            transparent_shape_groups[:len(cur_masks)],
                            device,
                        )
                        transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                        loss_exclude += exclude_loss(transparent_img, scale=2e-7)
                loss = loss_mse * 0.02 + loss_exclude + loss_struct
            else:
                loss = loss_mse

            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()

            pydiffvg.save_svg(
                os.path.join(file_save_path, f"{i}.svg"),
                img_width,
                img_height,
                shapes,
                shape_groups,
            )
            pbar.update(1)

    return shapes, shape_groups


def svg_optimize_img_visual(
    device,
    shapes,
    shape_groups,
    target_img: np.ndarray,
    file_save_path: str,
    is_opt_list: List[int],
    train_conf: dict,
    base_lr_conf: dict,
    count: int = 0,
    struct_path_num: int = 0,
    is_path_merging_phase: bool = False,
):
    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img, device=device) / 255.0
    target_img = target_img.permute(2, 0, 1)

    transparent_shape_groups = []
    for i in range(len(shapes) - struct_path_num):
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
            stroke_color=torch.FloatTensor([0, 0, 0, 0.3]),
        )
        transparent_shape_groups.append(path_group)

    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_visual_color"],
        is_opt_list,
        lr_base=base_lr_conf,
    )

    num_iters = train_conf["visual_opt_num_iters"]
    if is_path_merging_phase:
        num_iters = 50

    with tqdm(total=num_iters, desc="Visual optimization", unit="iter") as pbar:
        for _ in range(num_iters):
            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device)
            loss = F.mse_loss(img, target_img)

            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()

            pydiffvg.save_svg(
                os.path.join(file_save_path, f"{count}.svg"),
                img_width,
                img_height,
                shapes,
                shape_groups,
            )
            count += 1
            pbar.update(1)

    return shapes, shape_groups, count


def add_visual_paths_in_depth_layer(
    shapes,
    shape_groups,
    device,
    target_img: np.ndarray,
    allowed_mask: np.ndarray,
    insert_index: int,
    epsilon: int = 5,
    N: int = 50,
):
    img_height, img_width = target_img.shape[:2]
    visible_shapes = shapes[:insert_index]
    visible_shape_groups = shape_groups[:insert_index]
    raster_img = svg_to_img(img_width, img_height, visible_shapes, visible_shape_groups, device)
    raster_img = rgba_to_rgb(raster_img, device=device)
    raster_img = raster_img.detach().cpu().numpy()
    target_img1 = np.transpose((target_img / 255).astype(np.float16), (2, 0, 1))

    candidate_masks = select_mask_by_conn_area(raster_img, target_img1, N)
    if len(candidate_masks) == 0:
        return shapes, shape_groups, []

    allowed_mask = allowed_mask > 0
    new_indices = []
    for mask in candidate_masks:
        mask = np.logical_and(mask > 0, allowed_mask).astype(np.uint8) * 255
        if int(np.sum(mask > 0)) <= 0:
            continue

        color = get_mean_color(target_img, mask)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([0]),
            fill_color=torch.FloatTensor(list(color) + [255]) / 255,
            stroke_color=torch.FloatTensor([0, 0, 0, 1]),
        )
        mask = connect_mask_interior_exterior(mask)
        path = init_path_by_mask(mask, epsilon)

        shapes.insert(insert_index, path)
        shape_groups.insert(insert_index, path_group)
        new_indices.append(insert_index)
        insert_index += 1

    for index, shape_group in enumerate(shape_groups):
        shape_group.shape_ids = torch.LongTensor([index])
    return shapes, shape_groups, new_indices


def svg_optimize_img_visual_prefix(
    device,
    shapes,
    shape_groups,
    target_img: np.ndarray,
    file_save_path: str,
    active_indices: List[int],
    visible_end: int,
    loss_mask: np.ndarray,
    train_conf: dict,
    base_lr_conf: dict,
    count: int = 0,
):
    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img, device=device) / 255.0
    target_img = target_img.permute(2, 0, 1)
    loss_mask = mask_to_torch(loss_mask, device)

    active_set = set(active_indices)
    is_opt_list = [1 if index in active_set else 0 for index in range(len(shapes))]
    svg_optimizer = init_optimizer(
        shapes,
        shape_groups,
        train_conf["is_train_stroke"],
        train_conf["is_train_visual_color"],
        is_opt_list,
        lr_base=base_lr_conf,
    )

    num_iters = train_conf["visual_opt_num_iters"]
    with tqdm(total=num_iters, desc="Visual LayerVec pass", unit="iter") as pbar:
        for _ in range(num_iters):
            img = svg_to_img(
                img_width,
                img_height,
                shapes[:visible_end],
                shape_groups[:visible_end],
                device,
            )
            img = rgba_to_rgb(img, device)
            loss = masked_mse_loss(img, target_img, loss_mask)

            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()

            pydiffvg.save_svg(
                os.path.join(file_save_path, f"{count}.svg"),
                img_width,
                img_height,
                shapes,
                shape_groups,
            )
            count += 1
            pbar.update(1)

    return shapes, shape_groups, count


def layered_vectorization(
    args,
    device=None,
    pipe: Optional[StableDiffusionPipeline] = None,
):
    exp_dir = get_exp_dir(args)
    os.makedirs(exp_dir, exist_ok=True)

    simp_img_seq_save_path = os.path.join(exp_dir, "simplified_image_sequence")
    os.makedirs(simp_img_seq_save_path, exist_ok=True)

    all_simp_img_seq_save_path = "-1"
    if args.is_save_all_simp_img_seq:
        all_simp_img_seq_save_path = os.path.join(
            exp_dir, "all_simplified_image_sequence"
        )
        os.makedirs(all_simp_img_seq_save_path, exist_ok=True)

    masks_save_path = -1
    if args.is_save_masks:
        masks_save_path = os.path.join(exp_dir, "masks")
        os.makedirs(masks_save_path, exist_ok=True)

    struct_svgs_save_path = os.path.join(exp_dir, "struct_svgs")
    os.makedirs(struct_svgs_save_path, exist_ok=True)

    visual_svgs_save_path = os.path.join(exp_dir, "struct&visual_svgs")
    os.makedirs(visual_svgs_save_path, exist_ok=True)

    layerd_struct_save_path = os.path.join(exp_dir, "layerd_struct")
    os.makedirs(layerd_struct_save_path, exist_ok=True)

    print("SDS-based Simplification...")
    simp_img_seq = sds_based_simplification(
        device,
        args.target_image,
        args.simp_img_seq_indexs,
        simp_img_seq_save_path,
        all_simp_img_seq_save_path,
        pipe=pipe,
    )

    target_img = simp_img_seq[0]
    img_height, img_width = target_img.shape[:2]

    if getattr(args, "use_depth_ordering", True):
        print("SAM + Depth Ordering...")
        layerd_struct_masks, depth_map = _build_depth_layered_masks(
            target_img,
            args,
            device,
            exp_dir,
        )
        layerd_struct_masks = sanitize_layered_masks(layerd_struct_masks)
        depth_layer_grouping = getattr(args, "depth_layer_grouping", "kmeans")
        if depth_layer_grouping == "pixel_cluster":
            layerd_struct_masks, layer_loss_masks, initial_depth_order = pixel_depth_cluster_layers(
                layerd_struct_masks,
                depth_map,
                near_mode=getattr(args, "depth_near_mode", "large"),
                num_depth_layers=int(getattr(args, "depth_num_layers", 3)),
            )
        elif depth_layer_grouping == "kmeans":
            layerd_struct_masks, initial_depth_order = kmeans_layered_masks_by_depth(
                layerd_struct_masks,
                depth_map,
                near_mode=getattr(args, "depth_near_mode", "large"),
                num_depth_layers=int(getattr(args, "depth_num_layers", 3)),
            )
            layer_loss_masks = layerd_struct_masks
        else:
            layerd_struct_masks, initial_depth_order = sort_layered_masks_back_to_front_by_depth(
                layerd_struct_masks,
                depth_map,
                near_mode=getattr(args, "depth_near_mode", "large"),
            )
            layer_loss_masks = layerd_struct_masks
        layer_loss_masks, uncovered_area = ensure_depth_layers_cover_canvas(
            layer_loss_masks,
            os.path.join(exp_dir, "depth_ordering"),
        )
        if uncovered_area > 0:
            initial_depth_order.insert(
                0,
                {
                    "new_rank": -1,
                    "new_layer": 0,
                    "old_layer": None,
                    "old_index": None,
                    "depth_score": None,
                    "area": int(img_height * img_width),
                    "uncovered_pixels": uncovered_area,
                    "source": "canvas_background_for_uncovered_pixels",
                },
            )
        save_depth_layer_masks(
            layer_loss_masks,
            os.path.join(exp_dir, "depth_ordering", f"{depth_layer_grouping}_layer_masks"),
        )
        if depth_layer_grouping == "pixel_cluster":
            save_depth_layer_masks(
                layerd_struct_masks,
                os.path.join(exp_dir, "depth_ordering", "sam_clipped_layer_masks"),
            )
        with open(os.path.join(exp_dir, "initial_depth_mask_order.json"), "w", encoding="utf-8") as f:
            json.dump(initial_depth_order, f, indent=2)
    else:
        print("SAM...")
        masks = sam_img_seq(device, simp_img_seq, masks_save_path, args.sam)

        print("Layered Structure Reconstruction...")
        layerd_struct_masks = layer_segmented_masks([[masks[0]]], masks[1:])
        layerd_struct_masks = get_struct_masks_by_area(
            layerd_struct_masks,
            int(args.max_path_num_limit * 0.4),
        )
        depth_map = None
        layer_loss_masks = layerd_struct_masks

    shapes, shape_groups = init_svg_by_mask(
        layerd_struct_masks,
        target_img,
        args.approxpolydp_epsilon,
    )

    shapes, shape_groups = svg_optimize_img_struct(
        device,
        shapes,
        shape_groups,
        target_img,
        layerd_struct_masks,
        struct_svgs_save_path,
        args.train,
        args.base_lr,
        layer_loss_masks=layer_loss_masks,
    )

    if args.color_fitting_type not in ["dominan", "mse"]:
        raise ValueError(
            f"args.color_fitting_type can only be 'dominan' or 'mse', "
            f"but got {args.color_fitting_type}"
        )

    target_img_cluster = target_img
    if args.color_fitting_type == "dominan":
        shape_groups, target_img_cluster = color_fitting(
            shape_groups,
            target_img,
            layerd_struct_masks,
            args.is_cluster_target_img,
            args.kmeas_k,
        )
        Image.fromarray(target_img_cluster).save(os.path.join(exp_dir, "cluster_img.png"))
        pydiffvg.save_svg(
            os.path.join(exp_dir, "color-adjusted.svg"),
            img_width,
            img_height,
            shapes,
            shape_groups,
        )

    print("Visual Refinement...")
    count = 0
    if args.train.get("three_pass_visual_refinement", args.train.get("depth_layer_visual_refinement", False)) and depth_map is not None:
        layer_ranges = get_layer_shape_ranges(layerd_struct_masks)
        layer_masks = [combine_binary_masks(masks) for masks in layer_loss_masks]

        for i in range(args.add_visual_path_num_iters):
            any_added = False
            for layer_index, allowed_mask in enumerate(layer_masks):
                remaining_path_num = args.max_path_num_limit - len(shapes)
                if remaining_path_num <= 0:
                    break

                remaining_slots = max(
                    1,
                    (args.add_visual_path_num_iters - i) * (len(layer_masks) - layer_index),
                )
                add_num = max(1, int(np.ceil(remaining_path_num / remaining_slots)))
                start, end = layer_ranges[layer_index]
                add_dir = os.path.join(
                    visual_svgs_save_path,
                    f"{i}_layervec_pass_{layer_index}_add_paths",
                )
                os.makedirs(add_dir, exist_ok=True)

                shapes, shape_groups, new_indices = add_visual_paths_in_depth_layer(
                    shapes,
                    shape_groups,
                    device,
                    target_img_cluster,
                    allowed_mask,
                    insert_index=end,
                    epsilon=args.approxpolydp_epsilon,
                    N=add_num,
                )
                if not new_indices:
                    continue

                any_added = True
                added_count = len(new_indices)
                layer_ranges[layer_index] = (start, end + added_count)
                for later_index in range(layer_index + 1, len(layer_ranges)):
                    later_start, later_end = layer_ranges[later_index]
                    layer_ranges[later_index] = (
                        later_start + added_count,
                        later_end + added_count,
                    )

                _, visible_end = layer_ranges[layer_index]
                shapes, shape_groups, count = svg_optimize_img_visual_prefix(
                    device,
                    shapes,
                    shape_groups,
                    target_img,
                    add_dir,
                    new_indices,
                    visible_end,
                    allowed_mask,
                    args.train,
                    args.base_lr,
                    count,
                )

            if not any_added:
                print("There are no new depth-layer visual paths to add.")
                break
    else:
        pseudo_struct_masks = [mask for sublist in layerd_struct_masks for mask in sublist]
        is_opt_list = []
        struct_path_num = len(shapes)

        for i in range(args.add_visual_path_num_iters):
            add_dir = os.path.join(visual_svgs_save_path, f"{i}_add_paths")
            os.makedirs(add_dir, exist_ok=True)

            if i == args.add_visual_path_num_iters - 1:
                remaining_path_num = args.max_path_num_limit - len(shapes)
            else:
                remaining_path_num = int((args.max_path_num_limit - len(shapes)) * 0.6)

            shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = add_visual_paths(
                shapes,
                shape_groups,
                device,
                struct_path_num,
                target_img_cluster,
                pseudo_struct_masks,
                is_opt_list,
                epsilon=args.approxpolydp_epsilon,
                N=remaining_path_num,
            )

            if struct_path_num == -1:
                print("There are no new paths to add.")
                break

            print("Add new path")
            shapes, shape_groups, count = svg_optimize_img_visual(
                device,
                shapes,
                shape_groups,
                target_img,
                add_dir,
                is_opt_list,
                args.train,
                args.base_lr,
                count,
                struct_path_num,
            )

            if i == args.add_visual_path_num_iters - 1:
                break

            shapes, shape_groups = remove_lowquality_paths(
                shapes,
                shape_groups,
                device,
                img_width,
                img_height,
                visual_difference_threshold=args.paths_remove_visual_threshold,
                struct_path_num=struct_path_num,
            )

            print("Path merging")
            merge_dir = os.path.join(visual_svgs_save_path, f"{i}_merge_paths")
            os.makedirs(merge_dir, exist_ok=True)

            shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = merge_path(
                shapes,
                shape_groups,
                device,
                img_width,
                img_height,
                struct_path_num,
                pseudo_struct_masks,
                is_opt_list,
                color_threshold=args.paths_merge_color_threshold,
                overlapping_area_threshold=args.paths_merge_distance_threshold,
            )

            shapes, shape_groups, count = svg_optimize_img_visual(
                device,
                shapes,
                shape_groups,
                target_img,
                merge_dir,
                is_opt_list,
                args.train,
                args.base_lr,
                count,
                struct_path_num,
                is_path_merging_phase=True,
            )

    uses_three_pass_depth_layervec = args.train.get(
        "three_pass_depth_layervec",
        args.train.get("depth_layer_sequential_fit", False),
    )
    if depth_map is not None and getattr(args, "depth_sort_final_svg", True) and not uses_three_pass_depth_layervec:
        pydiffvg.save_svg(
            os.path.join(exp_dir, "final_before_depth_sort.svg"),
            img_width,
            img_height,
            shapes,
            shape_groups,
        )
        shapes, shape_groups, depth_order = sort_shapes_back_to_front_by_depth(
            shapes,
            shape_groups,
            depth_map,
            near_mode=getattr(args, "depth_near_mode", "large"),
        )
        with open(os.path.join(exp_dir, "depth_draw_order.json"), "w", encoding="utf-8") as f:
            json.dump(depth_order, f, indent=2)

    pydiffvg.save_svg(
        os.path.join(exp_dir, "final.svg"),
        img_width,
        img_height,
        shapes,
        shape_groups,
    )


def load_config(file_path, args):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="layered_image_vectorization")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/base_config.yaml",
        help="YAML/YML file for configuration.",
    )
    parser.add_argument(
        "-timg",
        "--target_image",
        type=str,
        default="./target_imgs/Snipaste_2024-11-19_16-31-12.png",
        help="Path to target image or directory of images.",
    )
    parser.add_argument(
        "-fsn",
        "--file_save_name",
        type=str,
        default="man",
        help="Files save name (used if input_type=image).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./workdir",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--input_type",
        choices=["image", "dir"],
        default="image",
        help="Whether --target_image is a single image file or a directory.",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=0,
        help="Index of split for parallel jobs (0-based).",
    )
    parser.add_argument(
        "--n_split",
        type=int,
        default=1,
        help="Total number of splits (parallel jobs).",
    )
    parser.add_argument(
        "--use_depth_ordering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SAM masks ordered by monocular depth instead of simplification-order layers.",
    )
    parser.add_argument(
        "--depth_sort_final_svg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resort final SVG paths from far to near using the estimated depth map.",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default=None,
        help="Optional Hugging Face depth model override.",
    )
    parser.add_argument(
        "--depth_near_mode",
        choices=["large", "small"],
        default="large",
        help="Interpret larger depth values as nearer (default) or smaller as nearer.",
    )
    parser.add_argument(
        "--depth_layer_grouping",
        choices=["pixel_cluster", "kmeans", "sort"],
        default="pixel_cluster",
        help="Build depth layers from pixel-depth clusters, then assign SAM masks to those layers.",
    )
    parser.add_argument(
        "--depth_num_layers",
        type=int,
        default=3,
        help="Number of depth layers for k-means grouping, ordered far to near.",
    )

    args = parser.parse_args()
    args = load_config(args.config, args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    if args.input_type == "image":
        t = time.time()
        init_diffvg(device=device)
        layered_vectorization(args, device, pipe=pipe)
        print(f"Elapsed time: {time.time() - t:.2f} sec")

    elif args.input_type == "dir":
        folder_path = args.target_image
        image_files = glob.glob(os.path.join(folder_path, "*.png")) + glob.glob(
            os.path.join(folder_path, "*.jpg")
        )
        image_files.sort()
        total_files = len(image_files)

        files_per_split = (total_files + args.n_split - 1) // args.n_split
        start_idx = args.split_index * files_per_split
        end_idx = min((args.split_index + 1) * files_per_split, total_files)
        assigned_files = image_files[start_idx:end_idx]

        filtered_files = []
        for file_path in assigned_files:
            save_name = os.path.splitext(os.path.basename(file_path))[0]
            exp_dir = os.path.join(args.output_root, "batch", save_name)
            final_svg = os.path.join(exp_dir, "final.svg")

            if not os.path.exists(final_svg):
                if os.path.exists(exp_dir):
                    print(f"Remove {exp_dir} for reinitialization")
                    shutil.rmtree(exp_dir)
                filtered_files.append(file_path)

        print(
            f"[Split {args.split_index}/{args.n_split}] "
            f"Assigned {len(assigned_files)}, processing {len(filtered_files)} "
            f"(after filtering)"
        )

        for idx, file_path in enumerate(filtered_files, start=1):
            save_name = os.path.splitext(os.path.basename(file_path))[0]
            sub_args = argparse.Namespace(**vars(args))
            sub_args.target_image = file_path
            sub_args.file_save_name = f"batch/{save_name}"

            t = time()
            init_diffvg(device=device)
            layered_vectorization(sub_args, device, pipe=pipe)
            print(f"({idx}/{len(filtered_files)}) {file_path} done in {time() - t:.2f} sec")
