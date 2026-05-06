#!/usr/bin/env python3
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
SLURM_DIR = ROOT / "slurm_picd"
LOG_DIR = ROOT / "logs"
SUBMIT_ALL = ROOT / "submit_all.sh"
SUBMIT_7_TO_50 = ROOT / "submit_7_to_50.sh"
RUN_SCRIPT = "run_refine_and_layervec.sh"

TARGETS = ["picd"] + [f"picd_{i:03d}" for i in range(1, 51)]
RESUME_TARGETS = [f"picd_{i:03d}" for i in range(7, 51)]

SLURM_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=LayerVec_{target}
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4g
#SBATCH --gpus-per-node=1
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --account=lp_grappa
#SBATCH --mail-user=li-wei.chen@kuleuven.be
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/slurm-{target}.out

source activate lv_svg
cd /vsc-hard-mounts/leuven-data/362/vsc36208/layered_vectorization/LayeredVectorization

echo "Running LayeredVectorization refine+layervec for {target}"
bash {run_script} {target}
"""


def write_text(path: Path, text: str, executable: bool = False):
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def main():
    SLURM_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    for target in TARGETS:
        slurm_path = SLURM_DIR / f"run_layervec_{target}.slurm"
        write_text(slurm_path, SLURM_TEMPLATE.format(target=target, run_script=RUN_SCRIPT))

    submit_all_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for target in TARGETS:
        submit_all_lines.append(f"sbatch {SLURM_DIR.name}/run_layervec_{target}.slurm")
    write_text(SUBMIT_ALL, "\n".join(submit_all_lines) + "\n", executable=True)

    submit_resume_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for target in RESUME_TARGETS:
        submit_resume_lines.append(f"sbatch {SLURM_DIR.name}/run_layervec_{target}.slurm")
    write_text(SUBMIT_7_TO_50, "\n".join(submit_resume_lines) + "\n", executable=True)

    print(f"Generated {len(TARGETS)} slurm files in {SLURM_DIR}")
    print(f"Generated: {SUBMIT_ALL}")
    print(f"Generated: {SUBMIT_7_TO_50}")


if __name__ == "__main__":
    main()
