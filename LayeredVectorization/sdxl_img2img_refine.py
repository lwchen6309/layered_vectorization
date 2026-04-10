#!/usr/bin/env python3
import argparse
import re
from io import BytesIO
from pathlib import Path

import cairosvg
import torch
import yaml
from PIL import Image
from diffusers import AutoPipelineForImage2Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0")
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--extract_prompt_only", action="store_true")
    parser.add_argument("--overrides_yaml", type=str, default=None)
    return parser.parse_args()


def extract_prompt_from_overrides_yaml(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, str):
                continue
            if item.startswith("prompt="):
                value = item[len("prompt="):].strip()
                if len(value) >= 2 and (
                    (value[0] == value[-1] == "'") or
                    (value[0] == value[-1] == '"')
                ):
                    value = value[1:-1]
                return value

    raw = path.read_text(encoding="utf-8")
    m = re.search(r"(?m)^-\s+prompt=(.+)$", raw)
    if m:
        value = m.group(1).strip()
        if len(value) >= 2 and (
            (value[0] == value[-1] == "'") or
            (value[0] == value[-1] == '"')
        ):
            value = value[1:-1]
        return value

    return ""


def do_extract_prompt(args):
    if not args.overrides_yaml:
        raise ValueError("--overrides_yaml is required with --extract_prompt_only")
    prompt = extract_prompt_from_overrides_yaml(Path(args.overrides_yaml))
    print(prompt, end="")


def open_image_any_format(image_path: str) -> Image.Image:
    """
    Open an image from SVG or raster formats while preserving transparency.
    Returns a PIL image.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".svg":
        png_bytes = cairosvg.svg2png(url=str(path))
        image = Image.open(BytesIO(png_bytes))
    else:
        image = Image.open(path)

    return image


def load_image_with_white_bg(image_path: str) -> Image.Image:
    """
    Load an image, preserve transparency if present, composite it onto a
    white background, and return an RGB image.
    """
    image = open_image_any_format(image_path)

    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")

        # Composite only when real transparency exists
        if alpha.getextrema()[0] < 255:
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(white_bg, rgba)
            return composited.convert("RGB")
        return rgba.convert("RGB")

    if image.mode == "P":
        # Palette images may contain transparency information
        if "transparency" in image.info:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            if alpha.getextrema()[0] < 255:
                white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                composited = Image.alpha_composite(white_bg, rgba)
                return composited.convert("RGB")
        return image.convert("RGB")

    return image.convert("RGB")


def do_refine(args):
    if not args.input or not args.output or not args.prompt:
        raise ValueError("--input, --output, and --prompt are required for refinement")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input image, support SVG, and convert to white-background RGB
    image = load_image_with_white_bg(str(input_path))

    # Save a debug JPEG to verify the white background before diffusion
    debug_input_path = output_path.parent / f"{output_path.stem}_input_white.jpg"
    image.save(debug_input_path, quality=95)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        kwargs["variant"] = "fp16"

    pipe = AutoPipelineForImage2Image.from_pretrained(args.model, **kwargs).to(device)

    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    result = pipe(
        prompt=args.prompt,
        image=image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    ).images[0]

    result.save(str(output_path))
    print(f"Saved: {output_path}")
    print(f"Saved debug input: {debug_input_path}")


def main():
    args = parse_args()
    if args.extract_prompt_only:
        do_extract_prompt(args)
    else:
        do_refine(args)


if __name__ == "__main__":
    main()