#!/usr/bin/env python3
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

from PIL import Image

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
SVG_NS = "{http://www.w3.org/2000/svg}"


def infer_square_size(root: ET.Element) -> float:
    vals = []
    for path_el in root.iter():
        if path_el.tag.endswith('path'):
            d = path_el.get('d', '')
            vals.extend(float(x) for x in FLOAT_RE.findall(d))
    if not vals:
        raise RuntimeError('No numeric path coordinates found in SVG.')
    max_abs = max(abs(v) for v in vals)
    # LayerVec square outputs typically live in roughly 0..512 or 0..800 space.
    # Default to current svg width/height if present; else fallback to max coord.
    w = root.get('width')
    h = root.get('height')
    candidates = []
    for v in (w, h):
        if v is None:
            continue
        try:
            candidates.append(float(str(v).replace('px', '').strip()))
        except Exception:
            pass
    if candidates:
        return max(candidates)
    return max_abs


def scale_path_d(d: str, sx: float, sy: float) -> str:
    idx = 0
    out = []
    coord_index = 0
    for m in FLOAT_RE.finditer(d):
        out.append(d[idx:m.start()])
        val = float(m.group(0))
        scaled = val * (sx if coord_index % 2 == 0 else sy)
        out.append(f"{scaled:.6f}".rstrip('0').rstrip('.'))
        idx = m.end()
        coord_index += 1
    out.append(d[idx:])
    return ''.join(out)


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size


def main():
    parser = argparse.ArgumentParser(description='Scale square SVG geometry back to original image aspect ratio.')
    parser.add_argument('--input-svg', required=True, help='Input square SVG path file')
    parser.add_argument('--reference-image', required=True, help='Original image whose aspect ratio should be restored')
    parser.add_argument('--output-svg', required=True, help='Output SVG path file')
    parser.add_argument('--square-size', type=float, default=None, help='Optional explicit square canvas size (default: infer from SVG)')
    args = parser.parse_args()

    input_svg = Path(args.input_svg)
    ref_img = Path(args.reference_image)
    output_svg = Path(args.output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_svg)
    root = tree.getroot()

    orig_w, orig_h = get_image_size(ref_img)
    square_size = args.square_size if args.square_size is not None else infer_square_size(root)
    sx = orig_w / float(square_size)
    sy = orig_h / float(square_size)

    root.set('width', str(orig_w))
    root.set('height', str(orig_h))
    root.set('viewBox', f'0 0 {orig_w} {orig_h}')

    for el in root.iter():
        if el.tag.endswith('path') and el.get('d'):
            el.set('d', scale_path_d(el.get('d'), sx, sy))

    tree.write(output_svg, encoding='utf-8', xml_declaration=True)
    print(f'Wrote {output_svg}')
    print(f'square_size={square_size} sx={sx:.6f} sy={sy:.6f}')


if __name__ == '__main__':
    main()
