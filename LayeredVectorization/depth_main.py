import os
import glob
import shutil
import argparse
import time
from typing import Optional, List, Dict, Sequence, Tuple

import yaml
import torch
import torch.nn.functional as F
import pydiffvg
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from utils.img_process import *
from sds_image_simplicity import sds_based_simplification
from obj_detect_sam import init_sam_mask_generator


DEFAULT_DEPTH_MODELS: Sequence[str] = (
    "LiheYoung/depth-anything-small-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
)


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
        if is_opt_list[i] == 1:
            path.id = i
            path.points.requires_grad = True
            points_vars.append(path.points)
            if is_train_stroke:
                path.stroke_width.requires_grad = True
                stroke_width_vars.append(path.stroke_width)

    if is_train_color:
        for i, group in enumerate(shape_groups):
            if is_opt_list[i] == 1:
                group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
                if is_train_stroke:
                    group.stroke_color.requires_grad = True
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

                transparent_img = svg_to_img(
                    img_width,
                    img_height,
                    shapes[shape_index - len(cur_masks):shape_index],
                    transparent_shape_groups[:len(cur_masks)],
                    device,
                )
                transparent_img = rgba_to_rgb(transparent_img, device, white_bg)
                loss_exclude += exclude_loss(transparent_img, scale=2e-7)

            img = svg_to_img(img_width, img_height, shapes, shape_groups, device)
            img = rgba_to_rgb(img, device, white_bg)
            loss_mse = F.mse_loss(img, target_img)

            loss = loss_mse * 0.02 + loss_exclude + loss_struct

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


# ---------- depth-aware mask ordering helpers ----------
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


def _load_depth_estimator(model_name: Optional[str] = None, device: Optional[torch.device] = None):
    model_names = [model_name] if model_name else list(DEFAULT_DEPTH_MODELS)
    last_err = None
    for name in model_names:
        try:
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModelForDepthEstimation.from_pretrained(name)
            if device is not None:
                model = model.to(device)
            model.eval()
            return (processor, model), name
        except Exception as exc:
            last_err = exc
    tried = ", ".join(model_names)
    raise RuntimeError(f"Failed to load any depth-estimation model: {tried}\nLast error: {last_err}")


def _run_depth_estimation(depth_bundle, image_pil: Image.Image, device: Optional[torch.device] = None) -> np.ndarray:
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


def _collect_sam_masks_for_depth(image_rgb: np.ndarray, sam_conf: dict, device) -> List[Dict]:
    mask_generator = init_sam_mask_generator(
        device=device,
        ckpt_path=sam_conf["sam_checkpoint"],
        model_type=sam_conf["model_type"],
        points_per_side=sam_conf["points_per_side"],
        pred_iou_thresh=sam_conf["pred_iou_thresh"],
        stability_score_thresh=sam_conf["stability_score_thresh"],
        crop_n_layers=sam_conf["crop_n_layers"],
        crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
        min_mask_region_area=sam_conf["min_mask_region_area"],
    )

    h, w = image_rgb.shape[:2]
    img_area = h * w
    raw_masks = mask_generator.generate(image_rgb)
    candidates: List[Dict] = []

    min_area_frac = float(sam_conf.get("depth_min_area_frac", 0.003))
    max_area_frac = float(sam_conf.get("depth_max_area_frac", 0.45))
    max_masks = int(sam_conf.get("depth_max_masks", max(8, int(sam_conf.get("points_per_side", 32)))))
    iou_thresh = float(sam_conf.get("depth_mask_nms_iou", sam_conf.get("box_nms_thresh", 0.65)))

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
                "mask": (seg.astype(np.uint8) * 255),
                "area": area,
                "bbox_xyxy": bbox_xyxy,
                "sam_score": sam_score,
                "stability_score": stability,
                "score": score,
            }
        )

    kept = _nms_candidates(candidates, iou_thresh=iou_thresh)
    return kept[:max_masks]


def _build_depth_layered_masks(target_img: np.ndarray, args, device, exp_dir: str) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    image_pil = Image.fromarray(target_img).convert("RGB")
    image_rgb = np.array(image_pil)
    mask_items = _collect_sam_masks_for_depth(image_rgb, args.sam, device)
    if not mask_items:
        raise RuntimeError("No usable SAM masks found for depth-aware ordering.")

    depth_bundle, depth_model_name = _load_depth_estimator(
        getattr(args, "depth_model", None),
        device=device,
    )
    depth_map = _run_depth_estimation(depth_bundle, image_pil, device=device)

    near_mode = getattr(args, "depth_near_mode", "large")
    scored_masks = []
    for item in mask_items:
        seg = item["mask"] > 0
        if not np.any(seg):
            continue
        values = depth_map[seg]
        median_depth = float(np.median(values))
        depth_score = median_depth if near_mode == "large" else (1.0 - median_depth)
        scored_masks.append({**item, "depth_score": depth_score, "median_depth": median_depth})

    if not scored_masks:
        raise RuntimeError("Depth-aware ordering produced no valid masks.")

    scores = np.asarray([x["depth_score"] for x in scored_masks], dtype=np.float32)
    if len(scores) == 1:
        thresholds = [scores[0], scores[0]]
    elif len(scores) == 2:
        thresholds = [float(np.min(scores)), float(np.max(scores))]
    else:
        thresholds = [float(np.quantile(scores, 1.0 / 3.0)), float(np.quantile(scores, 2.0 / 3.0))]

    buckets = {"background": [], "midground": [], "foreground": []}
    for item in sorted(scored_masks, key=lambda x: x["depth_score"]):
        score = item["depth_score"]
        if score <= thresholds[0]:
            bucket = "background"
        elif score <= thresholds[1]:
            bucket = "midground"
        else:
            bucket = "foreground"
        buckets[bucket].append(item)

    depth_debug_dir = os.path.join(exp_dir, "depth_ordering")
    os.makedirs(depth_debug_dir, exist_ok=True)
    Image.fromarray(np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)).save(
        os.path.join(depth_debug_dir, "depth_map_gray.png")
    )
    with open(os.path.join(depth_debug_dir, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"depth_model={depth_model_name}\n")
        f.write(f"near_mode={near_mode}\n")
        f.write(f"thresholds={thresholds}\n")
        for bucket_name in ["background", "midground", "foreground"]:
            f.write(f"{bucket_name}={len(buckets[bucket_name])}\n")

    layerd_struct_masks: List[List[np.ndarray]] = []
    for bucket_name in ["background", "midground", "foreground"]:
        bucket_masks = [x["mask"] for x in sorted(buckets[bucket_name], key=lambda y: y["area"], reverse=True)]
        if bucket_masks:
            layerd_struct_masks.append(bucket_masks)

    layer_limit = max(1, int(args.max_path_num_limit * 0.4))
    layerd_struct_masks = get_struct_masks_by_area(layerd_struct_masks, layer_limit)
    return layerd_struct_masks, depth_map


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

    print("SAM + Depth Ordering...")
    layerd_struct_masks, _ = _build_depth_layered_masks(target_img, args, device, exp_dir)

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
            img_height,
            img_width,
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

    pydiffvg.save_svg(
        os.path.join(exp_dir, "final.svg"),
        img_height,
        img_width,
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

            t = time.time()
            init_diffvg(device=device)
            layered_vectorization(sub_args, device, pipe=pipe)
            print(f"({idx}/{len(filtered_files)}) {file_path} done in {time.time() - t:.2f} sec")
