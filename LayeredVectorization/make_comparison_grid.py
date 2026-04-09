#!/usr/bin/env python3
import argparse
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from PIL import Image


def find_first(patterns):
    for p in patterns:
        matches = sorted(glob.glob(str(p), recursive=True))
        if matches:
            return Path(matches[0])
    return None


def open_img(p):
    if p and p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/leuven/362/vsc36208/CompSVG/logs/picd")
    parser.add_argument("--output", default="/data/leuven/362/vsc36208/CompSVG/grid.png")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []

    for stim in sorted(root.iterdir()):
        if not stim.is_dir():
            continue

        row = {
            "name": stim.name,
            "Stimulus": find_first([stim / "sd21b_cn1_w1e1_pww1_segline" / "**" / "init_image.png"]),
            "SDXL": find_first([stim / "sd21b_cn1_w1e1_pww1_segline" / "**" / "init_image_sdxl_refined.png"]),
            "SVGD": find_first([stim / "sd21b" / "**" / "all_particles.png"]),
            "SVGComp": find_first([stim / "sd21b_cn1_w1e1_pww1_segline" / "**" / "all_particles.png"]),
            "SVGComp-ReVec": find_first([stim / "sd21b_cn1_w1e1_pww1_segline" / "**" / "layervec" / "final.svg"]),
        }
        rows.append(row)

    if not rows:
        print("No matching images found.")
        raise SystemExit(0)

    cols = ["Stimulus", "SDXL", "SVGD", "SVGComp", "SVGComp-ReVec"]
    n_rows = len(rows)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i][j]
            img = open_img(row[col])
            ax.set_title(f"{row['name']}\n{col}")
            ax.axis("off")
            if img:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()