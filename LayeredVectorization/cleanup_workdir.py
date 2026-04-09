import os
import glob
import shutil
import cairosvg

def cleanup_exp_logs(root_dir, results_dir="cadb_results", jpg_dir="cadb_results_jpg"):
    """
    Clean up Layerwise Vectorization experiment logs.
    - Keeps only final.svg, cluster_img.png, and color-adjusted.svg if final.svg exists.
    - Copies final.svg into results_dir as <exp_index>.svg (only if not already copied).
    - Converts final.svg to JPG and saves in jpg_dir as <exp_index>.jpg.

    Args:
        root_dir (str): Path to the directory containing experiment log folders.
        results_dir (str): Path to store copied final.svg files.
        jpg_dir (str): Path to store converted JPG files.
    """
    keep_files = {"final.svg", "cluster_img.png", "color-adjusted.svg"}

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    exp_dirs = glob.glob(os.path.join(root_dir, "*/"))
    finished_runs = 0
    running_runs = 0

    for exp_dir in exp_dirs:
        final_path = os.path.join(exp_dir, "final.svg")
        exp_index = os.path.basename(os.path.normpath(exp_dir))  # folder name as index
        target_svg = os.path.join(results_dir, f"{exp_index}.svg")
        target_jpg = os.path.join(jpg_dir, f"{exp_index}.jpg")

        if os.path.exists(final_path):
            finished_runs += 1

            # Copy only if target SVG does not exist
            if not os.path.exists(target_svg):
                try:
                    shutil.copy(final_path, target_svg)
                except Exception as e:
                    print(f"⚠️ Could not copy {final_path} -> {target_svg}: {e}")

            # Convert to JPG only if not already exists
            if not os.path.exists(target_jpg):
                try:
                    cairosvg.svg2png(url=final_path, write_to="temp.png")
                    from PIL import Image
                    Image.open("temp.png").convert("RGB").save(target_jpg, "JPEG", quality=95)
                    os.remove("temp.png")
                except Exception as e:
                    print(f"⚠️ Could not convert {final_path} -> {target_jpg}: {e}")

            # Cleanup everything except keep_files
            for f in glob.glob(os.path.join(exp_dir, "*")):
                fname = os.path.basename(f)
                if fname not in keep_files:
                    if os.path.isfile(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                    elif os.path.isdir(f):
                        try:
                            shutil.rmtree(f)
                        except Exception:
                            pass
        else:
            running_runs += 1

    print("=== Cleanup Summary ===")
    print(f"✅ Finished runs cleaned: {finished_runs}")
    print(f"⏳ Still running (no final.svg): {running_runs}")


if __name__ == "__main__":
    # change this to your experiment log root directory
    exp_root = "./workdir/batch"
    # cleanup_exp_logs(exp_root, results_dir="cadb_results", jpg_dir="cadb_results_jpg")
    cleanup_exp_logs(exp_root, results_dir="picd_results", jpg_dir="picd_results_jpg")
