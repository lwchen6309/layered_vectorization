import os

n_split = 32

# Ensure directories exist
os.makedirs("slurm", exist_ok=True)
os.makedirs("logs", exist_ok=True)

template = """#!/bin/bash -l
#SBATCH --job-name=LayerVec_{i}
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
#SBATCH --output=logs/slurm-{i}.out

source activate lv_svg
cd /vsc-hard-mounts/leuven-data/362/vsc36208/layered_vectorization/LayeredVectorization

echo "Hello from split {i}"

# /data/leuven/362/vsc36208/datasets/PICD_dataset/small_train/
python main.py \\
    --config config/256_config.yaml \\
    --target_image /data/leuven/362/vsc36208/datasets/PICD_dataset/small_train/ \\
    --input_type dir \\
    --split_index {i} \\
    --n_split {n_split}
"""

# Generate individual slurm job files
for i in range(n_split):
    with open(f"slurm/run_layervec_{i}.slurm", "w") as f:
        f.write(template.format(i=i, n_split=n_split))

# Generate submit_all.sh
with open("submit_all.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    for i in range(n_split):
        f.write(f"sbatch slurm/run_layervec_{i}.slurm\n")

# Make submit_all.sh executable
os.chmod("submit_all.sh", 0o755)
