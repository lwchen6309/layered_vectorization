#!/usr/bin/env bash
set -euo pipefail

TARGET_GLOB="/data/leuven/362/vsc36208/CompSVG/logs/picd/*/sd21b_cn1_w1e1_pww1_segline/SVGDreamer*/sd42-sive-iconography-P1024/all_particles.png"

CONFIG="config/256_config.yaml"
MAIN_PY="main.py"
REFINE_PY="sdxl_img2img_refine.py"

SDXL_MODEL="stabilityai/stable-diffusion-xl-refiner-1.0"
GUIDANCE="7.5"
STEPS="30"

shopt -s nullglob

for f in $TARGET_GLOB; do
    base_dir="$(dirname "$f")"

    echo "============================================================"
    echo "Processing: $f"
    echo "============================================================"

    hydra_dir="$(dirname "$base_dir")/.hydra"
    overrides_yaml="${hydra_dir}/overrides.yaml"

    if [ ! -f "$overrides_yaml" ]; then
        echo "[Skip] overrides.yaml not found: $overrides_yaml"
        continue
    fi

    original_prompt="$(python "$REFINE_PY" --extract_prompt_only --overrides_yaml "$overrides_yaml")"

    if [ -z "$original_prompt" ]; then
        echo "[Skip] prompt not found in $overrides_yaml"
        continue
    fi

    # ------------------------------------------------------------
    # refine prompt (for all_particles)
    # ------------------------------------------------------------
    # refine_prompt="Ultra-realistic aerial photograph of ${original_prompt}, natural color grading, high dynamic range, professional wildlife photography, shot on a high-resolution DSLR."

    refined_particles="${base_dir}/all_particles_sdxl_refined.png"

    if [ ! -f "$refined_particles" ]; then
        python "$REFINE_PY" \
            --input "$f" \
            --output "$refined_particles" \
            --prompt "$original_prompt" \
            --model "$SDXL_MODEL" \
            --strength 0.75 \
            --guidance_scale "$GUIDANCE" \
            --steps "$STEPS"
    else
        echo "[Skip] $refined_particles exists"
    fi

    # ------------------------------------------------------------
    # refine init_image (用原 prompt)
    # ------------------------------------------------------------
    # init_img="${base_dir}/init_image.png"
    init_img="${base_dir}/VPSD_svg_logs/svg_iter0_p0.svg"
    refined_init="${base_dir}/init_image_sdxl_refined.png"

    if [ -f "$init_img" ]; then
        if [ ! -f "$refined_init" ]; then
            python "$REFINE_PY" \
                --input "$init_img" \
                --output "$refined_init" \
                --prompt "$original_prompt" \
                --model "$SDXL_MODEL" \
                --strength 0.95 \
                --guidance_scale "$GUIDANCE" \
                --steps "$STEPS"
        else
            echo "[Skip] $refined_init exists"
        fi
    else
        echo "[Skip] init_image not found: $init_img"
    fi

    # ------------------------------------------------------------
    # layervec
    # ------------------------------------------------------------
    mkdir -p "${base_dir}/layervec"

    save_name="$(basename "$base_dir")_sdxl_refined"

    python "$MAIN_PY" \
        --config "$CONFIG" \
        --target_image "$refined_particles" \
        --file_save_name "$(basename "$base_dir")_sdxl_refined" \
        --output_root "${base_dir}/layervec"
done