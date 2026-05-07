#!/usr/bin/env bash
set -euo pipefail

TARGET_NAME="${1:-picd}"
TARGET_GLOB="/data/leuven/362/vsc36208/CompSVG/logs/${TARGET_NAME}/*/sd21b_cn1_w1e1_pww1_segline/SVGDreamer*/sd42-sive-iconography-P1024/all_particles.png"

echo "[Info] TARGET_NAME=${TARGET_NAME}"
echo "[Info] TARGET_GLOB=${TARGET_GLOB}"

CONFIG="config/256_config.yaml"
MAIN_PY="main.py"
REFINE_PY="sdxl_img2img_refine.py"

SDXL_MODEL="stabilityai/stable-diffusion-xl-refiner-1.0"
GUIDANCE="7.5"
STEPS="30"

shopt -s nullglob

matched_files=( $TARGET_GLOB )
if [ ${#matched_files[@]} -eq 0 ]; then
    echo "[Warn] No files matched for TARGET_NAME=${TARGET_NAME}"
    echo "[Warn] TARGET_GLOB=${TARGET_GLOB}"
    exit 0
fi

echo "[Info] Matched ${#matched_files[@]} file(s)"

for f in "${matched_files[@]}"; do
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

    layervec_dir="${base_dir}/layervec"
    save_name="$(basename "$base_dir")_sdxl_refined"
    run_output_dir="${layervec_dir}/${save_name}"
    final_svg="${run_output_dir}/final.svg"

    if [ -f "$final_svg" ]; then
        echo "[Skip] layervec final.svg exists: $final_svg"
        find "$run_output_dir" -mindepth 1 ! -name 'final.svg' -exec rm -rf {} +
        echo "[Info] Cleaned layervec output, kept only final.svg: $run_output_dir"
    else
        if [ -d "$run_output_dir" ]; then
            echo "[Info] Incomplete layervec output detected, removing: $run_output_dir"
            rm -rf "$run_output_dir"
        fi

        mkdir -p "$layervec_dir"

        python "$MAIN_PY" \
            --config "$CONFIG" \
            --target_image "$refined_particles" \
            --file_save_name "$save_name" \
            --output_root "$layervec_dir"

        final_svg="${run_output_dir}/final.svg"
        if [ -f "$final_svg" ]; then
            find "$run_output_dir" -mindepth 1 ! -name 'final.svg' -exec rm -rf {} +
            echo "[Info] Cleaned layervec output after successful run, kept only final.svg: $run_output_dir"
        else
            echo "[Warn] layervec finished without final.svg: $run_output_dir"
        fi
    fi

done
