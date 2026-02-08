# post_local.py
import os
import argparse
import yaml
import numpy as np
from PIL import Image

import torch
import pydiffvg
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline

# ---------------------------------------------------------------------
# 1) IMPORT YOUR ORIGINAL LAYERVEC (UNCHANGED)
#    Replace `main` with your actual module name that defines:
#      - layered_vectorization(args, device, pipe=...)
#      - init_diffvg(device)
#      - add_visual_paths, svg_optimize_img_visual, merge_path, remove_lowquality_paths (optional)
# ---------------------------------------------------------------------
from main import layered_vectorization, init_diffvg
from main import add_visual_paths, svg_optimize_img_visual, merge_path, remove_lowquality_paths  # use your originals


# ---------------------------------------------------------------------
# 2) Minimal zoom helper (post stage only)
# ---------------------------------------------------------------------
class LocalZoomHelper:
    def __init__(self, im_size: int):
        self.im_size = int(im_size)
        self._st = None

    def _compute_scale(self, bbox):
        x0, y0, x1, y1 = bbox
        bw = max(x1 - x0, 1.0)
        bh = max(y1 - y0, 1.0)
        return float(self.im_size) / float(max(bw, bh))

    def zoomin(self, bbox, shapes, scale_width=True):
        if self._st is not None:
            return
        x0, y0, x1, y1 = map(float, bbox)
        paths = [p for p in shapes if hasattr(p, "points") and isinstance(p.points, torch.Tensor)]
        if not paths:
            self._st = {}
            return

        device = paths[0].points.device
        c_canvas = torch.tensor([self.im_size * 0.5, self.im_size * 0.5], device=device, dtype=torch.float32)
        c_bbox = torch.tensor([(x0 + x1) * 0.5, (y0 + y1) * 0.5], device=device, dtype=torch.float32)
        s = self._compute_scale((x0, y0, x1, y1))

        st = {"paths": [], "points0": [], "width0": [], "reqp0": [], "reqw0": []}
        for p in paths:
            st["paths"].append(p)
            st["points0"].append(p.points.detach().clone())
            st["reqp0"].append(bool(p.points.requires_grad))
            if hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                st["width0"].append(p.stroke_width.detach().clone())
                st["reqw0"].append(bool(p.stroke_width.requires_grad))
            else:
                st["width0"].append(None)
                st["reqw0"].append(None)

            # IMPORTANT: don't freeze here; let your optimizer + is_opt_list decide
            # (freezing was making it too brittle for small bboxes)

        with torch.no_grad():
            for p in paths:
                p.points.copy_((p.points - c_bbox) * s + c_canvas)
                if scale_width and hasattr(p, "stroke_width") and isinstance(p.stroke_width, torch.Tensor):
                    p.stroke_width.mul_(s)

        self._st = st

    def zoomout(self, shapes, restore_width=True):
        st = self._st
        if st is None:
            return
        with torch.no_grad():
            for p, pts0, w0 in zip(st["paths"], st["points0"], st["width0"]):
                p.points.copy_(pts0)
                if restore_width and w0 is not None:
                    p.stroke_width.copy_(w0)
        for p, reqp0, reqw0 in zip(st["paths"], st["reqp0"], st["reqw0"]):
            p.points.requires_grad_(reqp0)
            if reqw0 is not None and hasattr(p, "stroke_width"):
                p.stroke_width.requires_grad_(reqw0)
        self._st = None


# ---------------------------------------------------------------------
# 3) SDXL refine crop (fp16)
# ---------------------------------------------------------------------
def crop_resize_pil(img, bbox, *, out_size: int, ref_size: float = None, is_mask: bool = False) -> Image.Image:
    """
    Crop + resize in PIL, returning a PIL.Image.

    - img: np.ndarray (H,W,C) or PIL.Image
    - bbox: (x0,y0,x1,y1) defined in `ref_size` coord system (e.g. 512). If ref_size is None, treated as pixel coords.
    - out_size: output square size (e.g. 1024)
    - is_mask: if True, use NEAREST; else BICUBIC
    """
    # ensure PIL
    if isinstance(img, Image.Image):
        pil = img
    else:
        pil = Image.fromarray(img)

    W, H = pil.size
    x0, y0, x1, y1 = map(float, bbox)

    # rescale bbox from reference coord system -> current image pixels
    if ref_size is not None:
        sx = W / float(ref_size)
        sy = H / float(ref_size)
        x0 *= sx; x1 *= sx
        y0 *= sy; y1 *= sy

    # clamp + int
    x0 = int(max(0, min(W - 1, round(x0))))
    y0 = int(max(0, min(H - 1, round(y0))))
    x1 = int(max(1, min(W,     round(x1))))
    y1 = int(max(1, min(H,     round(y1))))

    # avoid empty crop
    if x1 <= x0: x1 = min(W, x0 + 1)
    if y1 <= y0: y1 = min(H, y0 + 1)

    crop = pil.crop((x0, y0, x1, y1))

    resample = Image.NEAREST if is_mask else Image.BICUBIC
    crop_rs = crop.resize((out_size, out_size), resample=resample)
    return crop_rs, (x0, x1, y0, y1)

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
    # load svg scene
    canvas_w, canvas_h, shapes, shape_groups = pydiffvg.svg_to_scene(final_svg_path)
    canvas_w, canvas_h = int(canvas_w), int(canvas_h)

    base_img = Image.open(base_img_path).convert("RGB")

    # shared lists like original code
    pseudo_masks = []   # we will use a fallback full mask for local stage
    is_opt_list = []
    struct_path_num = len(shapes)

    zoom = LocalZoomHelper(im_size=args.local_size)

    for bi, bb in enumerate(args.bboxes):
        # SDXL crop -> resize -> refine
        crop_rs, (x0, x1, y0, y1) = crop_resize_pil(base_img, bb, out_size=args.local_size, ref_size=args.img_size, is_mask=False)

        # ----------------------------
        # SAVE DEBUG IMAGES
        # ----------------------------
        check_root = f"./workdir/{args.file_save_name}/sdxl_check"
        bbox_dir = f"{check_root}/bbox_{bi}"
        os.makedirs(bbox_dir, exist_ok=True)

        refined = sdxl_refine_fp16(
            pipe_sdxl, crop_rs,
            prompt=args.sdxl_prompt,
            negative_prompt=args.sdxl_negative,
            steps=args.sdxl_steps,
            guidance_scale=args.sdxl_gs,
            strength=args.sdxl_strength,
        )
        base_img.save(os.path.join(bbox_dir, "base_img.png"))
        crop_rs.save(os.path.join(bbox_dir, "crop_img.png"))        
        refined.save(os.path.join(bbox_dir, "refined_img.png"))
        print(f"[SDXL] Saved bbox {bi} results → {bbox_dir}")
        refined = np.clip(np.array(refined), 0, 255).astype(np.uint8)

        # local masks fallback: whole crop
        pseudo_masks_local = [np.ones((args.local_size, args.local_size), dtype=np.uint8) * 255]

        # zoom svg into bbox -> local canvas
        zoom.zoomin((float(x0), float(y0), float(x1), float(y1)), shapes, scale_width=True)

        # add paths + optimize in local canvas
        # NOTE: we reuse YOUR add_visual_paths + svg_optimize_img_visual
        for it in range(int(args.local_iters)):
            # remaining = max(1, int(args.max_path_num_limit - len(shapes)))
            # print(remaining)
            remaining = 64
            shapes, shape_groups, pseudo_masks_local, is_opt_list, struct_path_num = add_visual_paths(
                shapes, shape_groups, device,
                struct_path_num,
                refined,                  # local target image
                pseudo_masks_local,
                is_opt_list,
                epsilon=args.approxpolydp_epsilon,
                N=remaining
            )            

            # stop here if debugging only
            if getattr(args, "sdxl_only", True):
                continue

            if struct_path_num == -1:
                print(f"[Local bbox {bi}] no new paths")
                break

            save_dir = f"./workdir/{args.file_save_name}/post_local_bbox_{bi}"
            os.makedirs(save_dir, exist_ok=True)

            shapes, shape_groups, _ = svg_optimize_img_visual(
                device, shapes, shape_groups,
                refined,                  # optimize to refined crop
                file_save_path=save_dir,
                is_opt_list=is_opt_list,
                train_conf=args.train,
                base_lr_conf=args.base_lr,
                count=0,
                struct_path_num=struct_path_num,
                is_path_merging_phase=False
            )

        # zoom back out
        # zoom.zoomout(shapes)

    # one global merge pass at end (optional)
    # if args.final_merge:
    #     shapes, shape_groups, pseudo_masks, is_opt_list, struct_path_num = merge_path(
    #         shapes, shape_groups, device,
    #         canvas_w, canvas_h,
    #         struct_path_num,
    #         pseudo_masks,
    #         is_opt_list,
    #         color_threshold=args.paths_merge_color_threshold,
    #         overlapping_area_threshold=args.paths_merge_distance_threshold
    #     )

    pydiffvg.save_svg(out_svg_path, canvas_w, canvas_h, shapes, shape_groups)
    print(f"[Post] saved: {out_svg_path}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def load_config(file_path, args):
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)
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

    # 1) run ORIGINAL layervec once (identical behavior)
    # assumes your original code loads SD1.5 pipe internally or you already pass it in;
    # keep it exactly as your main does.
    
    # pipe_sd15 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    # layered_vectorization(args, device, pipe=pipe_sd15)
    # # IMPORTANT: free SD1.5 ASAP
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
    # pipe_sdxl.enable_attention_slicing()
    # pipe_sdxl.enable_vae_slicing()
    # pipe_sdxl.enable_vae_tiling()

    # required local settings (provide defaults if not in yaml)
    args.local_size = int(getattr(args, "local_size", 1024))
    args.local_iters = int(getattr(args, "local_iters", 1))
    args.final_merge = bool(getattr(args, "final_merge", True))

    args.sdxl_prompt = getattr(args, "sdxl_prompt", "photo-realistic, detailed, high quality")
    args.sdxl_negative = getattr(args, "sdxl_negative", "cartoon, illustration, lowres, blurry, artifacts")
    args.sdxl_steps = int(getattr(args, "sdxl_steps", 20))
    args.sdxl_gs = float(getattr(args, "sdxl_gs", 6.0))
    args.sdxl_strength = float(getattr(args, "sdxl_strength", 0.55))

    # 3) post local refinement (pure post-process)
    post_local_refine(
        args=args,
        device=device,
        pipe_sdxl=pipe_sdxl,
        final_svg_path=final_svg,
        base_img_path=args.target_image,
        out_svg_path=out_svg,
    )
