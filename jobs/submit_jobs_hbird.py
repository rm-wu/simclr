from pathlib import Path

mem_sizes = [128, 64, 8, -1]
model_names = [
    "dino_vits16",
    "dino_vitb16",
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]
emb_sizes = [384, 768, 384, 768, 1024, 1536]
img_sizes = [512, 512, 504, 504, 504, 504]
patch_sizes = [16, 16, 14, 14, 14, 14]
batch_sizes = [128, 64, 128, 64, 32, 32]

jobs_path = Path("jobs")
jobs_path.mkdir(parents=True, exist_ok=True)

out_path = Path("output")
out_path.mkdir(parents=True, exist_ok=True)

err_path = Path("error")
err_path.mkdir(parents=True, exist_ok=True)

script_template = """#!/bin/bash
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu-h100-80g
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=64

#SBATCH --mail-type=ALL        
#SBATCH --mail-user=riccardo.mereu@aalto.fi

#SBATCH --output=output/job_output_%j.log   
#SBATCH --error=error/job_error_%j.log     

module load mamba
conda init
conda activate hbird_faiss

cd $SCRATCH/ssl_nat_aug/dense_prediction/open-hummingbird-eval/
python eval.py --seed 42 --batch-size {batch_size} --input-size {img_size} --patch-size {patch_size} --memory-size {mem_size} --embeddings-size {emb_size} --data-dir data --model {model_name}
"""

job_files = []

for mem_size in mem_sizes:
    for model_idx, model_name in enumerate(model_names):
        emb_size = emb_sizes[model_idx]
        img_size = img_sizes[model_idx]
        patch_size = patch_sizes[model_idx]
        batch_size = batch_sizes[model_idx]

        job_file = (
            jobs_path
            / f"dino_h100_faiss_{mem_size if mem_size != -1 else '1'}_{model_name}_{emb_size}.sh"
        )
        job_file.write_text(
            script_template.format(
                mem_size=mem_size,
                model_name=model_name,
                emb_size=emb_size,
                img_size=img_size,
                patch_size=patch_size,
                batch_size=batch_size,
            )
        )
        job_files.append(job_file)

with open("submit_jobs.sh", "w") as f:
    for job_file in job_files:
        f.write(f"sbatch {job_file}\n")
