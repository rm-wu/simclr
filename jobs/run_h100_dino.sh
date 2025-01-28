#!/bin/bash
#SBATCH --mem=256G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu-h100-80g
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=64
#SBATCH --array=0-35

#SBATCH --mail-type=ALL        
#SBATCH --mail-user=riccardo.mereu@aalto.fi

#SBATCH --output=output/job_output_%A_%a.log   
#SBATCH --error=error/job_error_%A_%a.log     

module load mamba
conda init
conda activate hbird_faiss

MEM_SIZE=(128 64 8 -1)
MODEL_NAME=(dino_vits16 dino_vitb16 dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14)
EMB_SIZE=(384 768 384 768 1024 1536)
IMG_SIZE=(512 512 504 504 504 504)
PATCH_SIZE=(16 16 14 14 14 14)
BATCH_SIZE=(128 64 128 64 32 32)

N_MEM=${#MEM_SIZE[@]}
N_MODEL=${#MODEL_NAME[@]}

mem_i=${MEM_SIZE[($SLURM_ARRAY_TASK_ID / $N_MODEL) % $N_MEM]}
model_i=${MODEL_NAME[$SLURM_ARRAY_TASK_ID % $N_MODEL]}
emb_i=${EMB_SIZE[$SLURM_ARRAY_TASK_ID % $N_MODEL]}
img_i=${IMG_SIZE[$SLURM_ARRAY_TASK_ID % $N_MODEL]}
patch_i=${PATCH_SIZE[$SLURM_ARRAY_TASK_ID % $N_MODEL]}
batch_i=${BATCH_SIZE[$SLURM_ARRAY_TASK_ID % $N_MODEL]}

echo "Model: $model_i"
echo "Memory: $mem_i"
echo "Embeddings: $emb_i"
echo "Image Size: $img_i"
echo "Patch Size: $patch_i"
echo "Batch Size: $batch_i"

cd $SCRATCH/ssl_nat_aug/dense_prediction/open-hummingbird-eval/
python eval.py --seed 42 --batch-size $batch_i --input-size $img_i --patch-size $patch_i --memory-size $mem_i --embeddings-size $emb_i --data-dir data --model $model_i

