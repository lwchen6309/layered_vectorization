#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
import cairosvg

from svg_aspect_ratio_fix import fix_svg_aspect_ratio


def normalize_svg_namespace_text(text: str) -> str:
    text = text.replace('<svg:svg', '<svg').replace('</svg:svg>', '</svg>')
    text = text.replace('<svg:g', '<g').replace('</svg:g>', '</g>')
    text = text.replace('<svg:path', '<path').replace('</svg:path>', '</path>')
    text = text.replace('<svg:defs', '<defs').replace('</svg:defs>', '</defs>')
    text = text.replace('xmlns:svg=', 'xmlns=')
    return text


def main():
    parser = argparse.ArgumentParser(
        description='External wrapper: aspect-fix SVG first, then rasterize to PNG. Keeps main.py untouched.'
    )
    parser.add_argument('--input-svg', type=Path, required=True)
    parser.add_argument('--target-image', type=Path, required=True)
    parser.add_argument('--fixed-svg', type=Path, required=True)
    parser.add_argument('--normalized-svg', type=Path, default=None)
    parser.add_argument('--output-png', type=Path, required=True)
    args = parser.parse_args()

    args.fixed_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: aspect-fix externally
    fix_svg_aspect_ratio(
        str(args.input_svg),
        str(args.fixed_svg),
        reference_image_path=str(args.target_image),
    )

    # Step 2: normalize namespaced tags for downstream SVG tooling compatibility
    normalized_svg = args.normalized_svg or args.fixed_svg.with_name(args.fixed_svg.stem + '_normalized.svg')
    text = args.fixed_svg.read_text(encoding='utf-8')
    normalized_svg.write_text(normalize_svg_namespace_text(text), encoding='utf-8')

    # Step 3: rasterize using target image size
    target = Image.open(args.target_image).convert('RGB')
    w, h = target.size
    png_bytes = cairosvg.svg2png(url=str(normalized_svg), output_width=w, output_height=h)
    img = Image.open(BytesIO(png_bytes)).convert('RGB')
    img.save(args.output_png)

    print(f'input_svg={args.input_svg}')
    print(f'fixed_svg={args.fixed_svg}')
    print(f'normalized_svg={normalized_svg}')
    print(f'output_png={args.output_png}')
    print(f'target_size={w}x{h}')


if __name__ == '__main__':
    main()
