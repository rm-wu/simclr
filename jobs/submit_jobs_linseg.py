from pathlib import Path

model_names = [
    "dino_vits16",
    "dino_vitb16",
    "dinov2_vits14",
    "dinov2_vitb14",
    # "dinov2_vitl14",
    # "dinov2_vitg14",
    "ibot_vits16",
    "ibot_vitb16",
    # TODO: MAE and CriBO
]
emb_sizes = [384, 768, 384, 768,384, 768]
img_sizes = [512, 512, 504, 504, 512, 512]
patch_sizes = [16, 16, 14, 14, 16, 16]
seed = 42
# batch_sizes = [128, 64, 128, 64]
project_path = Path("/scratch/project_462000585/mereuric/ssl_nat_aug/")
# project_path = Path("/home/mereur1/projects/ocl/ssl_nat_aug")
jobs_path = project_path / "jobs" / "linear_seg"
jobs_path.mkdir(parents=True, exist_ok=True)

logs_path = project_path / "logs" / "linear_seg"
logs_path.mkdir(parents=True, exist_ok=True)



dataset_name = "voc"
data_path = Path("/flash/project_462000585/mereuric/data")

out_path = project_path / "outputs" / "linear_seg"
out_path.mkdir(parents=True, exist_ok=True)

# srun --time=10:00:00 --mem=200G --pty --account=project_462000585 --cpus-per-task=8 --gres=gpu:1 --partition=small-g bash
# python3 linear_eval.py --model=dinov2_vitb14 --input-size=504 --patch-size=14 --embeddings-size=768 --dataset-name=voc --data-dir=/flash/project_462000585/mereuric/data/ --out-dir=outputs/ --num-workers=64 --seed=42
script_template = """#!/bin/bash
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --partition=small-g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=project_462000585

#SBATCH --mail-type=ALL        
#SBATCH --mail-user=riccardo.mereu@aalto.fi

#SBATCH --output={logs_path}/job_output_%j.log   
#SBATCH --error={logs_path}/job_error_%j.log     

module use /appl/local/training/modules/AI-20250204/
module load cotainr
module load singularity-userfilesystems

cd {project_path}
singularity exec lumi_env.sif python3 linear_eval.py --seed {seed} --model={model_name} --input-size {img_size} --patch-size {patch_size} --embeddings-size {emb_size} --dataset-name={dataset_name} --data-dir {data_path} --out-dir={out_path} --num-workers=7
"""

job_files = []

for model_idx, model_name in enumerate(model_names):
    emb_size = emb_sizes[model_idx]
    img_size = img_sizes[model_idx]
    patch_size = patch_sizes[model_idx]

    job_file = (
        jobs_path
        / f"{model_name}_{dataset_name}_{img_size}_{patch_size}_{emb_size}_{seed}.sh"
    )
    job_file.write_text(
        script_template.format(
            model_name=model_name,
            dataset_name=dataset_name,
            emb_size=emb_size,
            img_size=img_size,
            patch_size=patch_size,
            project_path=project_path,
            data_path=data_path,
            logs_path=logs_path,
            out_path=out_path,
            seed=seed,
        )
    )
    job_files.append(job_file)

with open("submit_jobs.sh", "w") as f:
    for job_file in job_files:
        f.write(f"sbatch {job_file}\n")
