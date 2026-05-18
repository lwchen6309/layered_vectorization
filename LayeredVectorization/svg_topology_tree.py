import argparse
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
import copy
from pathlib import Path
from typing import List

import cairosvg
import numpy as np
from PIL import Image, ImageDraw
from depth_main import _load_depth_estimator, _run_depth_estimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SVG_NS = "{http://www.w3.org/2000/svg}"
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Node:
    index: int
    depth_mean: float
    bbox: list[float]
    area: float
    center: list[float]
    mask_area: int
    parent: int | None
    children: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topology tree of SVG paths using geometry + depth map.")
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None, help="Optional raster image for depth estimation; defaults to svgcomp.jpg next to svg.")
    parser.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-size", type=int, default=512)
    return parser.parse_args()


def rasterize_svg(svg_path: Path, size: int) -> Image.Image:
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def extract_path_points(d: str) -> np.ndarray:
    nums = [float(m.group(0)) for m in NUMBER_RE.finditer(d)]
    if len(nums) < 4 or len(nums) % 2 != 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(nums, dtype=np.float64).reshape(-1, 2)


def sample_depth(depth_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    h, w = depth_map.shape
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return depth_map[ys, xs]


def bbox_from_points(points: np.ndarray) -> list[float]:
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


def area_from_bbox(b: list[float]) -> float:
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def center_from_bbox(b: list[float]) -> list[float]:
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def rasterize_single_path_mask(svg_root: ET.Element, target_idx: int, size: int) -> np.ndarray:
    new_root = copy.deepcopy(svg_root)
    path_counter = 0
    for el in new_root.iter():
        if el.tag == f"{SVG_NS}path":
            if path_counter == target_idx:
                el.attrib['fill'] = '#ffffff'
                el.attrib['stroke'] = 'none'
            else:
                el.attrib['fill'] = 'none'
                el.attrib['stroke'] = 'none'
            path_counter += 1
    svg_bytes = ET.tostring(new_root, encoding='utf-8', xml_declaration=True)
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
    rgba = np.array(Image.open(io.BytesIO(png_bytes)).convert('RGBA'))
    return rgba[..., 3] > 0


def choose_parent(nodes: list[Node], masks: list[np.ndarray], idx: int, overflow_tolerance: float = 0.02) -> int | None:
    me = nodes[idx]
    me_mask = masks[idx]
    candidates = []
    me_area = max(float(me_mask.sum()), 1.0)
    for j, other in enumerate(nodes):
        if j == idx:
            continue
        if other.depth_mean <= me.depth_mean:
            continue
        outside = np.logical_and(me_mask, np.logical_not(masks[j]))
        outside_ratio = float(outside.sum()) / me_area
        if outside_ratio <= overflow_tolerance:
            candidates.append((other.mask_area, j))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def draw_tree_preview(svg_path: Path, nodes: list[Node], out_path: Path, size: int) -> None:
    img = rasterize_svg(svg_path, size).convert("RGB")
    draw = ImageDraw.Draw(img)
    for node in nodes:
        cx, cy = node.center
        for child_idx in node.children:
            ccx, ccy = nodes[child_idx].center
            draw.line((cx, cy, ccx, ccy), fill=(255, 0, 0), width=2)
        r = 4
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 255, 0))
        draw.text((cx + 4, cy + 4), str(node.index), fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_decomposition_preview(svg_root: ET.Element, nodes: list[Node], out_path: Path, size: int) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for node in nodes:
        new_root = copy.deepcopy(svg_root)
        path_counter = 0
        target_fill = '#66c2ff'
        for el in new_root.iter():
            if el.tag == f"{SVG_NS}path":
                if path_counter == node.index:
                    el.attrib['fill'] = target_fill
                    el.attrib['stroke'] = '#003366'
                    el.attrib['stroke-width'] = '1'
                else:
                    el.attrib['fill'] = 'none'
                    el.attrib['stroke'] = 'none'
                path_counter += 1
        svg_bytes = ET.tostring(new_root, encoding='utf-8', xml_declaration=True)
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
        rgba = np.array(Image.open(io.BytesIO(png_bytes)).convert('RGBA'))
        alpha = rgba[..., 3] > 0
        ys, xs = np.where(alpha)
        if len(xs) == 0:
            continue
        # Subsample for speed/clarity
        step = max(1, len(xs) // 4000)
        xs = xs[::step]
        ys = ys[::step]
        zs = np.full_like(xs, fill_value=node.depth_mean, dtype=np.float64)
        colors = rgba[ys, xs, :3] / 255.0
        ax.scatter(xs, ys, zs, c=colors, s=1)
        cx, cy = node.center
        ax.text(cx, cy, node.depth_mean, f"{node.index}", color='red')

    for node in nodes:
        if node.parent is not None:
            parent = nodes[node.parent]
            ax.plot(
                [parent.center[0], node.center[0]],
                [parent.center[1], node.center[1]],
                [parent.depth_mean, node.depth_mean],
                color='black', linewidth=1.5
            )

    ax.set_title('3D topology decomposition (x, y, depth)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('depth')
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    svg_root = ET.parse(args.svg).getroot()
    image_path = args.image
    if image_path is None:
        for cand in [args.svg.with_name('svgcomp.jpg'), args.svg.with_name('svgcomp.png')]:
            if cand.exists():
                image_path = cand
                break
    if image_path is None:
        raise FileNotFoundError(f"No image specified and no svgcomp.jpg/png next to {args.svg}")

    image = Image.open(image_path).convert('RGB')
    depth_bundle, depth_model_name = _load_depth_estimator(model_name=args.depth_model, device='cuda:0')
    depth_map = _run_depth_estimation(depth_bundle, image, device='cuda:0')

    path_points = []
    for el in svg_root.iter():
        if el.tag != f"{SVG_NS}path":
            continue
        d = el.attrib.get('d', '')
        pts = extract_path_points(d)
        if len(pts) == 0:
            continue
        path_points.append(pts)

    masks = [rasterize_single_path_mask(svg_root, i, args.preview_size) for i in range(len(path_points))]

    nodes: list[Node] = []
    for i, pts in enumerate(path_points):
        bbox = bbox_from_points(pts)
        area = area_from_bbox(bbox)
        center = center_from_bbox(bbox)
        depths = sample_depth(depth_map, pts)
        depth_mean = float(depths.mean()) if len(depths) else 0.0
        mask_area = int(masks[i].sum())
        nodes.append(Node(index=i, depth_mean=depth_mean, bbox=bbox, area=area, center=center, mask_area=mask_area, parent=None, children=[]))

    for i in range(len(nodes)):
        nodes[i].parent = choose_parent(nodes, masks, i)
    for i, node in enumerate(nodes):
        if node.parent is not None:
            nodes[node.parent].children.append(i)

    with open(args.output_dir / 'topology_tree.json', 'w') as f:
        json.dump([asdict(n) for n in nodes], f, indent=2)

    np.save(args.output_dir / 'depth_map.npy', depth_map)
    depth_gray = Image.fromarray(np.clip(depth_map * 255.0, 0, 255).astype(np.uint8))
    depth_gray.save(args.output_dir / 'depth_map_gray.png')
    draw_tree_preview(args.svg, nodes, args.output_dir / 'topology_tree_preview.png', args.preview_size)
    draw_decomposition_preview(svg_root, nodes, args.output_dir / 'topology_decomposition_3d.png', args.preview_size)

    print(f"[OK] depth_model={depth_model_name}")
    print(f"[OK] wrote: {args.output_dir / 'topology_tree.json'}")
    print(f"[OK] wrote: {args.output_dir / 'depth_map_gray.png'}")
    print(f"[OK] wrote: {args.output_dir / 'topology_tree_preview.png'}")
    print(f"[OK] wrote: {args.output_dir / 'topology_decomposition_3d.png'}")


if __name__ == '__main__':
    main()
