# post_local.py
import os
import argparse
import yaml
import numpy as np
from PIL import Image
import cv2

import torch
import pydiffvg
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline

# ---------------------------------------------------------------------
# 1) IMPORT YOUR ORIGINAL LAYERVEC (UNCHANGED)
# ---------------------------------------------------------------------
from main import layered_vectorization, init_diffvg
from main import add_visual_paths, svg_optimize_img_visual, merge_path, remove_lowquality_paths  # originals
from obj_detect_sam import (
    init_sam_mask_generator,
    detect_bboxes_sam_only,
    SamOnlyConfig,
)

# ---------------------------------------------------------------------
# 2) Minimal zoom helper (post stage only)
# ---------------------------------------------------------------------
class LocalZoomHelper:
    """
    In-place zoom helper using forward + inverse transforms (NO reset).

    zoomin:
      - Compute scale s so bbox fills canvas of size im_size
      - Apply: p' = (p - c_bbox) * s + c_canvas
      - Freeze outside bbox / unfreeze inside bbox (based on ORIGINAL coords at zoomin time)

    zoomout:
      - Apply inverse to ALL paths (including paths added during zoom-in):
            p = (p' - c_canvas) / s + c_bbox
      - Restore requires_grad flags to what they were at zoomin time for paths that existed then
        (new paths keep their current requires_grad)
    """

    def __init__(self, im_size: int):
        self.im_size = int(im_size)
        self._st = None

    @staticmethod
    def _any_point_in_bbox(points: torch.Tensor, bbox) -> bool:
        x0, y0, x1, y1 = bbox
        inside = (
            (points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 1] >= y0) & (points[:, 1] <= y1)
        )
        return bool(inside.any().item())

    def _compute_scale(self, bbox):
        x0, y0, x1, y1 = bbox
        bw = max(float(x1 - x0), 1.0)
        bh = max(float(y1 - y0), 1.0)
        return float(self.im_size) / float(max(bw, bh))

    def zoomin(self, bbox, shapes, scale_width: bool = True):
        if self._st is not None:
            return

        x0, y0, x1, y1 = map(float, bbox)
        bbox_f = (x0, y0, x1, y1)

        paths = [p for p in shapes if hasattr(p, "points") and isinstance(p.points, torch.Tensor)]
        if not paths:
            self._st = {}
            return

        device = paths[0].points.device
        c_canvas = torch.tensor([self.im_size * 0.5, self.im_size * 0.5], device=device, dtype=torch.float32)
        c_bbox   = torch.tensor([(x0 + x1) * 0.5, (y0 + y1) * 0.5], device=device, dtype=torch.float32)
        s = float(self._compute_scale(bbox_f))

        st = {
            "ids": set(),
            "reqp0": {},
            "reqw0": {},
            "c_canvas": c_canvas,
            "c_bbox": c_bbox,
            "s": s,
            "scale_width": bool(scale_width),
        }

        # freeze/unfreeze (based on current/original coords)
        for p in paths:
            pid = id(p)
            st["ids"].add(pid)
            st["reqp0"][pid] = bool(p.points.requires_grad)
            if hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                st["reqw0"][pid] = bool(p.stroke_width.requires_grad)

            any_inside = self._any_point_in_bbox(p.points.detach(), bbox_f)
            p.points.requires_grad_(any_inside)
            if hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                p.stroke_width.requires_grad_(any_inside)

        # apply forward transform to ALL paths
        with torch.no_grad():
            for p in paths:
                p.points.copy_((p.points - c_bbox) * s + c_canvas)
                if scale_width and hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                    p.stroke_width.mul_(s)

        self._st = st

    def zoomout(self, shapes):
        st = self._st
        if st is None:
            return

        c_canvas = st["c_canvas"]
        c_bbox   = st["c_bbox"]
        s        = float(st["s"])
        inv_s    = 1.0 / max(s, 1e-8)
        scale_width = bool(st["scale_width"])

        paths = [p for p in shapes if hasattr(p, "points") and isinstance(p.points, torch.Tensor)]
        if not paths:
            self._st = None
            return

        # inverse transform ALL paths (including newly added ones)
        with torch.no_grad():
            for p in paths:
                p.points.copy_((p.points - c_canvas) * inv_s + c_bbox)
                if scale_width and hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                    p.stroke_width.mul_(inv_s)

        # restore requires_grad for paths that existed at zoomin time
        for p in paths:
            pid = id(p)
            if pid in st["reqp0"]:
                p.points.requires_grad_(st["reqp0"][pid])
            if pid in st["reqw0"] and hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                p.stroke_width.requires_grad_(st["reqw0"][pid])

        self._st = None


# ---------------------------------------------------------------------
# 3) Crop helpers
# ---------------------------------------------------------------------
def crop_resize_pil(img, bbox, *, out_size: int, ref_size: float = None, is_mask: bool = False):
    """
    Crop + resize in PIL, returning (crop_rs, (x0,y0,x1,y1)).

    - bbox: (x0,y0,x1,y1)
      * If ref_size is not None: bbox is defined in ref_size coord system (e.g. 512) and will be scaled to pixels.
      * If ref_size is None: bbox is already in image pixel coordinates.
    """
    pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
    W, H = pil.size
    x0, y0, x1, y1 = map(float, bbox)

    if ref_size is not None:
        sx = W / float(ref_size)
        sy = H / float(ref_size)
        x0 *= sx; x1 *= sx
        y0 *= sy; y1 *= sy

    x0 = int(max(0, min(W - 1, round(x0))))
    y0 = int(max(0, min(H - 1, round(y0))))
    x1 = int(max(1, min(W,     round(x1))))
    y1 = int(max(1, min(H,     round(y1))))

    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)

    crop = pil.crop((x0, y0, x1, y1))
    resample = Image.NEAREST if is_mask else Image.BICUBIC
    crop_rs = crop.resize((out_size, out_size), resample=resample)
    return crop_rs, (x0, y0, x1, y1)


@torch.inference_mode()
def sdxl_refine_fp16(pipe_sdxl, image, prompt, negative_prompt, steps, guidance_scale, strength):
    out = pipe_sdxl(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
    ).images[0]
    return out


def build_local_support_mask(refined_rgb: np.ndarray, base_rgb: np.ndarray, save_dir=None):
    """
    Build local fitting support mask from SDXL refined vs original crop.
    Both inputs are RGB uint8 arrays in the SAME resolution (e.g. canvas_w x canvas_w).
    """
    ref_lab  = cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2LAB)
    base_lab = cv2.cvtColor(base_rgb,    cv2.COLOR_RGB2LAB)

    diff = cv2.absdiff(ref_lab, base_lab)

    diff_ab = diff[:, :, 1:].astype(np.float32)
    diff_mag = np.linalg.norm(diff_ab, axis=2)

    diff_norm = diff_mag / (diff_mag.max() + 1e-6)
    diff_vis = (diff_norm * 255).astype(np.uint8)

    mask = (diff_norm > 0.1).astype(np.uint8) * 255
    kernel_big = np.ones((31, 31), np.uint8)
    mask = cv2.dilate(mask, kernel_big, iterations=1)

    if save_dir is not None:
        Image.fromarray(diff_vis).save(os.path.join(save_dir, "diff_map.png"))
        Image.fromarray(mask).save(os.path.join(save_dir, "support_mask.png"))

    return mask


# ---------------------------------------------------------------------
# 4) Post-local refinement loop (operates on loaded shapes)
# ---------------------------------------------------------------------
def post_local_refine(
    args,
    device,
    pipe_sdxl,
    final_svg_path: str,
    base_img_path: str,
    out_svg_path: str,
):
    """
    Key revision:
      - SAM auto-bboxes are in *base_img pixel coordinates*.
      - We DO NOT rescale bboxes with canvas_w anymore.
      - We convert pixel-bboxes -> SVG-canvas coords before zoomin.
      - add_visual_paths / svg_optimize_img_visual operate in zoom canvas of size canvas_w,
        so refined/base crops and support masks are resized to (canvas_w, canvas_w).
    """

    # ----------------------------
    # small helpers: argparse.Namespace / dict-like
    # ----------------------------
    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_path(obj, path: str, default=None):
        cur = obj
        for k in path.split("."):
            cur = _get(cur, k, None)
            if cur is None:
                return default
        return cur

    # load svg scene
    canvas_w, canvas_h, shapes, shape_groups = pydiffvg.svg_to_scene(final_svg_path)
    canvas_w, canvas_h = int(canvas_w), int(canvas_h)

    base_img = Image.open(base_img_path).convert("RGB")
    base_w, base_h = base_img.size

    pseudo_masks = []  # used by optional final merge
    is_opt_list = []
    struct_path_num = len(shapes)

    # IMPORTANT: zoom canvas size must match what your original add_visual_paths expects
    # (typically canvas_w used by LayerVec). If not provided, fall back to canvas_w (assuming square).
    # canvas_w = int(getattr(args, "img_size", canvas_w))
    zoom = LocalZoomHelper(im_size=canvas_w)

    # scale factors: base image pixel -> SVG canvas coords
    sx_canvas = canvas_w / float(base_w)
    sy_canvas = canvas_h / float(base_h)

    # ----------------------------
    # SAM-only AutoBBox (reads args.sam.*)
    # ----------------------------
    if bool(_get(args, "auto_bbox", False)):
        sam_ckpt = _get_path(args, "sam.sam_checkpoint", None) or _get(args, "sam_ckpt", None)
        if sam_ckpt is None:
            raise ValueError(
                "auto_bbox=True but SAM checkpoint not found. "
                "Provide args.sam.sam_checkpoint (preferred) or args.sam_ckpt."
            )

        gen = init_sam_mask_generator(
            device=device,
            ckpt_path=sam_ckpt,
            model_type=str(_get_path(args, "sam.model_type", "vit_h")),
            points_per_side=int(_get_path(args, "sam.points_per_side", 32)),
            pred_iou_thresh=float(_get_path(args, "sam.pred_iou_thresh", 0.88)),
            stability_score_thresh=float(_get_path(args, "sam.stability_score_thresh", 0.92)),
            crop_n_layers=int(_get_path(args, "sam.crop_n_layers", 1)),
            crop_n_points_downscale_factor=int(_get_path(args, "sam.crop_n_points_downscale_factor", 2)),
            min_mask_region_area=int(_get_path(args, "sam.min_mask_region_area", 200)),
        )

        # NOTE: ref_size is only for *internal filtering/scoring* in your detect helper.
        # Using the real image min side is the least confusing choice.
        cfg = SamOnlyConfig(
            ref_size=int(min(base_w, base_h)),
            topk=int(_get(args, "bbox_topk", 8)),
            square=True,
            enlarge=float(_get(args, "bbox_enlarge", 1.05)),
            min_area_frac=float(_get(args, "bbox_min_area_frac", 0.002)),
            max_area_frac=float(_get(args, "bbox_max_area_frac", 0.20)),
            nms_iou=float(_get_path(args, "sam.box_nms_thresh", _get(args, "bbox_nms_iou", 0.65))),
        )

        debug_dir = f"./workdir/{args.file_save_name}/sam_bbox_debug"
        args.bboxes = detect_bboxes_sam_only(
            mask_generator=gen,
            base_img_pil=base_img,
            cfg=cfg,
            debug_dir=debug_dir,
        )
        print("[SAM-only AutoBBox] bboxes (base_img pixel coords):", args.bboxes)

    # sanity: require bboxes
    bboxes = _get(args, "bboxes", None)
    if not bboxes:
        raise ValueError("No bboxes found. Provide args.bboxes or set auto_bbox=True.")

    # ----------------------------
    # Local refine loop
    # ----------------------------
    for bi, bb in enumerate(bboxes):
        # bb is in base image pixel coords (x0,y0,x1,y1)
        crop_sdxl, (x0, y0, x1, y1) = crop_resize_pil(
            base_img, bb,
            out_size=args.local_size,
            ref_size=canvas_w,
            is_mask=False,
        )

        # SAVE DEBUG IMAGES
        check_root = f"./workdir/{args.file_save_name}/sdxl_check"
        bbox_dir = f"{check_root}/bbox_{bi}"
        os.makedirs(bbox_dir, exist_ok=True)

        refined_sdxl = sdxl_refine_fp16(
            pipe_sdxl, crop_sdxl,
            prompt=args.sdxl_prompt,
            negative_prompt=args.sdxl_negative,
            steps=args.sdxl_steps,
            guidance_scale=args.sdxl_gs,
            strength=args.sdxl_strength,
        )

        base_img.save(os.path.join(bbox_dir, "base_img.png"))
        crop_sdxl.save(os.path.join(bbox_dir, "crop_img.png"))
        refined_sdxl.save(os.path.join(bbox_dir, "refined_img.png"))
        print(f"[SDXL] Saved bbox {bi} results → {bbox_dir}")

        # Prepare images for vector stage at canvas_w (the zoom canvas)
        refined_vec = refined_sdxl.resize((canvas_w, canvas_w), Image.BICUBIC)
        base_crop_pix = base_img.crop((x0, y0, x1, y1)).resize((canvas_w, canvas_w), Image.BICUBIC)

        refined_np = np.clip(np.array(refined_vec), 0, 255).astype(np.uint8)
        base_crop_np = np.clip(np.array(base_crop_pix), 0, 255).astype(np.uint8)

        support_mask = build_local_support_mask(refined_np, base_crop_np, save_dir=bbox_dir)
        pseudo_masks_local = [support_mask]

        # zoom svg into bbox -> local canvas
        zoom.zoomin(bb, shapes, scale_width=True)

        # add paths + optimize in local canvas
        for it in range(int(args.local_iters)):
            remaining = 128
            shapes, shape_groups, pseudo_masks_local, is_opt_list, struct_path_num = add_visual_paths(
                shapes, shape_groups, device,
                struct_path_num,
                refined_np,
                pseudo_masks_local,
                is_opt_list,
                epsilon=args.approxpolydp_epsilon,
                N=remaining,
            )

            print("[Add] shapes:", len(shapes), "is_opt_list:", len(is_opt_list), "struct_path_num:", struct_path_num)

            if bool(_get(args, "sdxl_only", True)):
                continue

            if struct_path_num == -1:
                print(f"[Local bbox {bi}] no new paths")
                break

            save_dir = f"./workdir/{args.file_save_name}/post_local_bbox_{bi}"
            os.makedirs(save_dir, exist_ok=True)

            shapes, shape_groups, _ = svg_optimize_img_visual(
                device, shapes, shape_groups,
                refined_np,  # optimize to refined crop in zoom canvas
                file_save_path=save_dir,
                is_opt_list=is_opt_list,
                train_conf=args.train,
                base_lr_conf=args.base_lr,
                count=0,
                struct_path_num=struct_path_num,
                is_path_merging_phase=False,
            )

        # zoom back out
        zoom.zoomout(shapes)

    # one global merge pass at end (optional)
    if bool(_get(args, "final_merge", False)):
        shapes, shape_groups, pseudo_masks, is_opt_list, struct_path_num = merge_path(
            shapes, shape_groups, device,
            canvas_w, canvas_h,
            struct_path_num,
            pseudo_masks,
            is_opt_list,
            color_threshold=args.paths_merge_color_threshold,
            overlapping_area_threshold=args.paths_merge_distance_threshold,
        )

    pydiffvg.save_svg(out_svg_path, canvas_w, canvas_h, shapes, shape_groups)
    print(f"[Post] saved: {out_svg_path}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def load_config(file_path, args):
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)

    # keep your behavior: set top-level keys onto args
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config/base_config.yaml")
    parser.add_argument("--file_save_name", default="man")
    parser.add_argument("--target_image", default="./target_imgs/Snipaste_2024-11-19_16-31-12.png")
    args = parser.parse_args()
    args = load_config(args.config, args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_diffvg(device=device)

    # 1) run ORIGINAL layervec once (optional; keep commented if you already ran it)
    # pipe_sd15 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device) 
    # args.max_path_num_limit = args.max_path_num_limit / 2
    # layered_vectorization(args, device, pipe=pipe_sd15)
    # try: 
    #     pipe_sd15.to("cpu") 
    # except Exception: 
    #     pass 
    # del pipe_sd15 
    # if device.type == "cuda": 
    #     torch.cuda.empty_cache() 
    #     torch.cuda.ipc_collect()

    final_svg = f"./workdir/{args.file_save_name}/final.svg"
    out_svg = f"./workdir/{args.file_save_name}/final_local.svg"

    # 2) SDXL fp16 init for post stage
    pipe_sdxl = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.local_sdxl["model_id"],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    # required local settings (provide defaults if not in yaml)
    args.local_size = int(getattr(args, "local_size", 1024))
    args.local_iters = int(getattr(args, "local_iters", 1))
    args.final_merge = bool(getattr(args, "final_merge", True))

    args.sdxl_prompt = getattr(args, "sdxl_prompt", "photo-realistic, detailed, high quality")
    args.sdxl_negative = getattr(args, "sdxl_negative", "cartoon, illustration, lowres, blurry, artifacts")
    args.sdxl_steps = int(getattr(args, "sdxl_steps", 20))
    args.sdxl_gs = float(getattr(args, "sdxl_gs", 6.0))
    args.sdxl_strength = float(getattr(args, "sdxl_strength", 0.55))
    # args.max_path_num_limit = args.max_path_num_limit * 2

    # 3) post local refinement (pure post-process)
    post_local_refine(
        args=args,
        device=device,
        pipe_sdxl=pipe_sdxl,
        final_svg_path=final_svg,
        base_img_path=args.target_image,
        out_svg_path=out_svg,
    )
