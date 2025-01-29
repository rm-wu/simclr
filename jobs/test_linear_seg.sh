#!/bin/bash
# #SBATCH --mem=100G
# #SBATCH --time=00:45:00
# #SBATCH --partition=gpu-h100-80g
# #SBATCH --gres=gpu:h100:1
# #SBATCH --cpus-per-gpu=64
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1


# #SBATCH --mail-type=ALL        
# #SBATCH --mail-user=riccardo.mereu@aalto.fi

#SBATCH --output=logs/lin_seg_job_output_%A_%a.log   
#SBATCH --error=logs/lin_seg_job_error_%A_%a.log     

module purge
module load mamba
echo $PATH
conda deactivate
conda activate hbird_faiss
echo $PATH
conda deactivate
echo $PATH

export PATH=/scratch/work/mereur1/.conda_envs/hbird_faiss/bin:$PATH

which python3
echo $PATH
python3 linear_eval.py --model=dinov2_vitb14 --input-size=504 --patch-size=14 --embeddings-size=768 --dataset-name=voc --data-dir=/flash/project_462000585/mereuric/data/ --out-dir=outputs/ --num-workers=64 --seed=42
