import argparse
import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cairosvg
import io
import torch
from diffusers import AutoPipelineForInpainting
from depth_main import _load_depth_estimator, _run_depth_estimation

SVG_NS = "{http://www.w3.org/2000/svg}"
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stimulus-driven SVG deformation via Gaussian displacement field. Anchors are taken only from the stimulus shapes.")
    parser.add_argument(
        "--svgcomp",
        type=Path,
        default=Path("/home/lwchen/CompSVG/collected_images/picd_018/06_three_points_triangle/svgcomp.svg"),
    )
    parser.add_argument(
        "--stimulus",
        type=Path,
        default=Path("/home/lwchen/.openclaw/workspace/octa_elementgenai/stimulus/stimulus.svg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lwchen/layered_vectorization/LayeredVectorization/outputs/stimulus_deformation"),
    )
    parser.add_argument("--scale", type=float, default=0.85, help="Uniform scaling applied to stimulus around its center.")
    parser.add_argument("--translate-x", type=float, default=0.0, help="Translation in x applied after scaling.")
    parser.add_argument("--translate-y", type=float, default=0.0, help="Translation in y applied after scaling.")
    parser.add_argument("--rotate-deg", type=float, default=0.0, help="Rotation in degrees applied around stimulus center.")
    parser.add_argument("--sigma-xy", type=float, default=12.0, help="Gaussian sigma for x/y in stimulus canvas coordinates.")
    parser.add_argument("--sigma-z", type=float, default=5.0, help="Gaussian sigma for z/depth channel after depth scaling.")
    parser.add_argument("--depth-weight", type=float, default=120.0, help="Scale factor for depth channel when building the Gaussian field in (x, y, z) space.")
    parser.add_argument("--grid", type=int, default=300, help="Preview raster size for 2x2 plot.")
    parser.add_argument("--run-inpaint", action="store_true", help="Run SDXL inpainting on uncovered regions after deformation.")
    parser.add_argument("--inpaint-model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="SDXL inpainting model id.")
    parser.add_argument("--inpaint-prompt", type=str, default="clean natural continuation background", help="Prompt for filling uncovered background regions.")
    parser.add_argument("--inpaint-steps", type=int, default=30)
    parser.add_argument("--inpaint-guidance", type=float, default=7.5)
    parser.add_argument("--inpaint-strength", type=float, default=0.99)
    return parser.parse_args()


def get_svg_size(root: ET.Element) -> Tuple[float, float]:
    width = float(root.attrib["width"])
    height = float(root.attrib["height"])
    return width, height


def collect_stimulus_controls(root: ET.Element) -> List[Tuple[float, float, ET.Element]]:
    controls = []
    for el in root.iter():
        if el.tag == f"{SVG_NS}ellipse":
            cx = float(el.attrib["cx"])
            cy = float(el.attrib["cy"])
            controls.append((cx, cy, el))
        elif el.tag == f"{SVG_NS}circle":
            cx = float(el.attrib["cx"])
            cy = float(el.attrib["cy"])
            controls.append((cx, cy, el))
        elif el.tag == f"{SVG_NS}rect" and el.attrib.get("fill") != "none":
            x = float(el.attrib["x"])
            y = float(el.attrib["y"])
            w = float(el.attrib["width"])
            h = float(el.attrib["height"])
            controls.append((x + w / 2.0, y + h / 2.0, el))
        elif el.tag == f"{SVG_NS}polygon":
            pts = []
            for pair in el.attrib["points"].strip().split():
                x_str, y_str = pair.split(",")
                pts.append([float(x_str), float(y_str)])
            pts_arr = np.array(pts, dtype=np.float64)
            center = pts_arr.mean(axis=0)
            controls.append((float(center[0]), float(center[1]), el))
    if not controls:
        raise RuntimeError("No supported control shapes found in stimulus.svg")
    return controls


def affine_transform_points(points: np.ndarray, center: np.ndarray, scale: float, translate: np.ndarray, rotate_deg: float) -> np.ndarray:
    centered = scale * (points - center)
    theta = math.radians(rotate_deg)
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ], dtype=np.float64)
    rotated = centered @ rot.T
    return center + rotated + translate


def update_stimulus_svg(root: ET.Element, scale: float, translate: np.ndarray, rotate_deg: float) -> ET.Element:
    width, height = get_svg_size(root)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    for el in root.iter():
        tag = el.tag
        if tag == f"{SVG_NS}ellipse":
            p = np.array([float(el.attrib["cx"]), float(el.attrib["cy"])] , dtype=np.float64)
            p2 = affine_transform_points(p[None, :], center, scale, translate, rotate_deg)[0]
            el.attrib["cx"] = f"{p2[0]:.6f}"
            el.attrib["cy"] = f"{p2[1]:.6f}"
            el.attrib["rx"] = f"{float(el.attrib['rx']) * scale:.6f}"
            el.attrib["ry"] = f"{float(el.attrib['ry']) * scale:.6f}"
        elif tag == f"{SVG_NS}circle":
            p = np.array([float(el.attrib["cx"]), float(el.attrib["cy"])] , dtype=np.float64)
            p2 = affine_transform_points(p[None, :], center, scale, translate, rotate_deg)[0]
            el.attrib["cx"] = f"{p2[0]:.6f}"
            el.attrib["cy"] = f"{p2[1]:.6f}"
            el.attrib["r"] = f"{float(el.attrib['r']) * scale:.6f}"
        elif tag == f"{SVG_NS}rect":
            if el.attrib.get("fill") == "none":
                continue
            x = float(el.attrib["x"])
            y = float(el.attrib["y"])
            w = float(el.attrib["width"])
            h = float(el.attrib["height"])
            center_rect = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
            center_new = affine_transform_points(center_rect[None, :], center, scale, translate, rotate_deg)[0]
            new_w = w * scale
            new_h = h * scale
            el.attrib["x"] = f"{center_new[0] - new_w / 2.0:.6f}"
            el.attrib["y"] = f"{center_new[1] - new_h / 2.0:.6f}"
            el.attrib["width"] = f"{new_w:.6f}"
            el.attrib["height"] = f"{new_h:.6f}"
        elif tag == f"{SVG_NS}polygon":
            pts = []
            for pair in el.attrib["points"].strip().split():
                x_str, y_str = pair.split(",")
                pts.append([float(x_str), float(y_str)])
            pts_arr = np.array(pts, dtype=np.float64)
            pts_new = affine_transform_points(pts_arr, center, scale, translate, rotate_deg)
            el.attrib["points"] = " ".join(f"{x:.6f},{y:.6f}" for x, y in pts_new)
    return root


def sample_depth_at_points(depth_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return depth_map[ys, xs]


def gaussian_field_displacement(query_points: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, sigma_xy: float, sigma_z: float, query_depth: np.ndarray | None = None, src_depth: np.ndarray | None = None, depth_weight: float = 1.0) -> np.ndarray:
    disp = dst_points - src_points
    field = np.zeros_like(query_points, dtype=np.float64)
    weight_sum = np.zeros((query_points.shape[0], 1), dtype=np.float64)
    for i in range(src_points.shape[0]):
        diff_xy = query_points - src_points[i]
        dist2_xy = np.sum(diff_xy * diff_xy, axis=1, keepdims=True) / (2.0 * sigma_xy * sigma_xy)
        dist2 = dist2_xy
        if query_depth is not None and src_depth is not None:
            dz = (query_depth - src_depth[i]).reshape(-1, 1) * depth_weight
            dist2_z = (dz * dz) / (2.0 * sigma_z * sigma_z)
            dist2 = dist2 + dist2_z
        w = np.exp(-dist2)
        field += w * disp[i]
        weight_sum += w
    weight_sum = np.clip(weight_sum, 1e-8, None)
    return field / weight_sum


def warp_svgcomp_paths(svg_root: ET.Element, src_pts: np.ndarray, dst_pts: np.ndarray, sigma_xy: float, sigma_z: float, depth_map: np.ndarray | None = None, depth_weight: float = 1.0) -> ET.Element:
    # Note: anchors come only from the stimulus. svgcomp contributes only query points to be warped.
    warped = copy.deepcopy(svg_root)
    src_depth = sample_depth_at_points(depth_map, src_pts) if depth_map is not None else None
    for path_el in warped.iter():
        if path_el.tag != f"{SVG_NS}path":
            continue
        d = path_el.attrib.get("d")
        if not d:
            continue
        numbers = [float(m.group(0)) for m in NUMBER_RE.finditer(d)]
        if len(numbers) % 2 != 0:
            continue
        pts = np.array(numbers, dtype=np.float64).reshape(-1, 2)
        query_depth = sample_depth_at_points(depth_map, pts) if depth_map is not None else None
        disp = gaussian_field_displacement(pts, src_pts, dst_pts, sigma_xy, sigma_z, query_depth=query_depth, src_depth=src_depth, depth_weight=depth_weight)
        pts_warped = pts + disp
        replacement_vals = pts_warped.reshape(-1)
        idx = 0

        def repl(_match):
            nonlocal idx
            val = replacement_vals[idx]
            idx += 1
            return f"{val:.6f}"

        path_el.attrib["d"] = NUMBER_RE.sub(repl, d)
    return warped


def save_svg(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def rasterize_svg(svg_path: Path, size: int, keep_alpha: bool = False) -> Image.Image:
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
    img = Image.open(io.BytesIO(png_bytes))
    return img.convert("RGBA" if keep_alpha else "RGB")


def make_plot(old_stim: Path, new_stim: Path, old_svg: Path, new_svg: Path, out_path: Path, size: int) -> None:
    imgs = [
        rasterize_svg(old_stim, size),
        rasterize_svg(new_stim, size),
        rasterize_svg(old_svg, size),
        rasterize_svg(new_svg, size),
    ]
    titles = ["old stimulus", "new stimulus", "old svgcomp", "warped svgcomp"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, img, title in zip(axes.flat, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_uncovered_mask(original_svg: Path, warped_svg: Path, out_mask: Path, size: int) -> Image.Image:
    orig = np.array(rasterize_svg(original_svg, size, keep_alpha=True))
    warped = np.array(rasterize_svg(warped_svg, size, keep_alpha=True))
    orig_alpha = orig[..., 3] > 0
    warped_alpha = warped[..., 3] > 0
    uncovered = np.logical_and(orig_alpha, np.logical_not(warped_alpha)).astype(np.uint8) * 255
    mask_img = Image.fromarray(uncovered, mode='L')
    mask_img.save(out_mask)
    return mask_img


def run_sdxl_inpaint(image_path: Path, mask_path: Path, output_path: Path, model_id: str, prompt: str, steps: int, guidance: float, strength: float) -> None:
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'torch_dtype': dtype, 'use_safetensors': True}
    if dtype == torch.float16:
        kwargs['variant'] = 'fp16'
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs).to(device)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance,
        num_inference_steps=steps,
        strength=strength,
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def save_anchor_debug(src_pts: np.ndarray, dst_pts: np.ndarray, canvas_size: tuple[float, float], out_path: Path) -> None:
    width, height = canvas_size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(src_pts[:, 0], src_pts[:, 1], c='tab:blue', label='src anchors')
    ax.scatter(dst_pts[:, 0], dst_pts[:, 1], c='tab:red', label='dst anchors')
    for s, d in zip(src_pts, dst_pts):
        ax.arrow(s[0], s[1], d[0] - s[0], d[1] - s[1], head_width=4, head_length=6, fc='gray', ec='gray', alpha=0.7)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('stimulus anchor displacement')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stim_root = ET.parse(args.stimulus).getroot()
    svgcomp_root = ET.parse(args.svgcomp).getroot()

    old_stim_root = copy.deepcopy(stim_root)
    controls = collect_stimulus_controls(stim_root)
    src_pts = np.array([[cx, cy] for cx, cy, _ in controls], dtype=np.float64)
    width, height = get_svg_size(stim_root)
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    translate = np.array([args.translate_x, args.translate_y], dtype=np.float64)
    dst_pts = affine_transform_points(src_pts, center, args.scale, translate, args.rotate_deg)

    candidate_images = [
        args.svgcomp.with_name('svgcomp.jpg'),
        args.svgcomp.with_name('svgcomp.png'),
        args.svgcomp.with_name('svgcomp_raw.jpg'),
        args.svgcomp.with_name('svgcomp_raw.png'),
    ]
    image_path = next((p for p in candidate_images if p.exists()), None)
    if image_path is None:
        raise FileNotFoundError(f'Could not find source image for depth estimation next to {args.svgcomp}')
    image_pil = Image.open(image_path).convert('RGB')
    depth_bundle, depth_model_name = _load_depth_estimator(device='cuda:0')
    depth_map = _run_depth_estimation(depth_bundle, image_pil, device='cuda:0')

    new_stim_root = update_stimulus_svg(copy.deepcopy(stim_root), args.scale, translate, args.rotate_deg)
    warped_svg_root = warp_svgcomp_paths(svgcomp_root, src_pts, dst_pts, args.sigma_xy, args.sigma_z, depth_map=depth_map, depth_weight=args.depth_weight)

    disp = dst_pts - src_pts
    mag = np.linalg.norm(disp, axis=1)
    print(f"[INFO] depth_model={depth_model_name}")
    print(f"[INFO] depth_image={image_path}")
    print(f"[INFO] scale={args.scale}, translate=({args.translate_x}, {args.translate_y}), rotate_deg={args.rotate_deg}, sigma_xy={args.sigma_xy}, sigma_z={args.sigma_z}, depth_weight={args.depth_weight}")
    print(f"[INFO] anchors={len(src_pts)}, disp_mag_min={mag.min():.4f}, disp_mag_max={mag.max():.4f}, disp_mag_mean={mag.mean():.4f}")

    old_stim_path = args.output_dir / "stimulus_original.svg"
    new_stim_path = args.output_dir / "stimulus_affined.svg"
    old_svg_path = args.output_dir / "svgcomp_original.svg"
    new_svg_path = args.output_dir / "svgcomp_warped.svg"
    plot_path = args.output_dir / "stimulus_deformation_grid.png"
    anchor_debug_path = args.output_dir / "stimulus_anchor_debug.png"
    depth_gray_path = args.output_dir / "depth_map_gray.png"
    depth_inferno_path = args.output_dir / "depth_map_inferno.png"
    uncovered_mask_path = args.output_dir / "uncovered_mask.png"
    inpaint_output_path = args.output_dir / "svgcomp_warped_inpainted.png"

    save_svg(old_stim_root, old_stim_path)
    save_svg(new_stim_root, new_stim_path)
    save_svg(svgcomp_root, old_svg_path)
    save_svg(warped_svg_root, new_svg_path)
    Image.fromarray(np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)).save(depth_gray_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_map, cmap='inferno')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(depth_inferno_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    make_plot(old_stim_path, new_stim_path, old_svg_path, new_svg_path, plot_path, args.grid)
    save_anchor_debug(src_pts, dst_pts, (width, height), anchor_debug_path)
    build_uncovered_mask(old_svg_path, new_svg_path, uncovered_mask_path, args.grid)
    if args.run_inpaint:
        warped_png_path = args.output_dir / 'svgcomp_warped_for_inpaint.png'
        rasterize_svg(new_svg_path, args.grid).save(warped_png_path)
        run_sdxl_inpaint(
            warped_png_path,
            uncovered_mask_path,
            inpaint_output_path,
            args.inpaint_model,
            args.inpaint_prompt,
            args.inpaint_steps,
            args.inpaint_guidance,
            args.inpaint_strength,
        )

    print(f"[OK] wrote: {old_stim_path}")
    print(f"[OK] wrote: {new_stim_path}")
    print(f"[OK] wrote: {old_svg_path}")
    print(f"[OK] wrote: {new_svg_path}")
    print(f"[OK] wrote: {plot_path}")
    print(f"[OK] wrote: {anchor_debug_path}")
    print(f"[OK] wrote: {depth_gray_path}")
    print(f"[OK] wrote: {depth_inferno_path}")
    print(f"[OK] wrote: {uncovered_mask_path}")
    if args.run_inpaint:
        print(f"[OK] wrote: {inpaint_output_path}")


if __name__ == "__main__":
    main()
