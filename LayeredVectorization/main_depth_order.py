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


def svg_optimize_img_struct(
    device,
    shapes,
    shape_groups,
    target_img: np.ndarray,
    layerd_struct_masks: list,
    file_save_path: str,
    train_conf: dict,
    base_lr_conf: dict,
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

    if train_conf.get("depth_layer_sequential_fit", False):
        layer_ranges = get_layer_shape_ranges(layerd_struct_masks)
        layer_num_iters = int(
            train_conf.get(
                "depth_layer_struct_num_iters",
                train_conf["struct_opt_num_iters"],
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
            with tqdm(
                total=layer_num_iters,
                desc=f"Struct depth layer {struct_i}",
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
                    loss_mse = F.mse_loss(img, target_img)

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
        if depth_layer_grouping == "kmeans":
            layerd_struct_masks, initial_depth_order = kmeans_layered_masks_by_depth(
                layerd_struct_masks,
                depth_map,
                near_mode=getattr(args, "depth_near_mode", "large"),
                num_depth_layers=int(getattr(args, "depth_num_layers", 3)),
            )
        else:
            layerd_struct_masks, initial_depth_order = sort_layered_masks_back_to_front_by_depth(
                layerd_struct_masks,
                depth_map,
                near_mode=getattr(args, "depth_near_mode", "large"),
            )
        save_depth_layer_masks(
            layerd_struct_masks,
            os.path.join(exp_dir, "depth_ordering", f"{depth_layer_grouping}_layer_masks"),
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
    pseudo_struct_masks = [mask for sublist in layerd_struct_masks for mask in sublist]
    is_opt_list = []
    count = 0
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

    if depth_map is not None and getattr(args, "depth_sort_final_svg", True):
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
        choices=["kmeans", "sort"],
        default="sort",
        help="Sort initial masks by depth. The kmeans option is kept only for ablation.",
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
