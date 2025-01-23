#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --output=heval_ibot_run_%A_%a.out
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --cpus-per-task=8

module load mamba
source activate hummingbird-eval

MEM_SIZE=(128 64 8 -1)
MODEL_NAME=(ibot_vits16 ibot_vitb16 ibot_vitl16)
EMB_SIZE=(384 768 1024)
IMG_SIZE=(512 512 512)
PATCH_SIZE=(16 16 16)
BATCH_SIZE=(64 32 16)

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

python eval.py --seed 42 --batch-size $batch_i --input-size $img_i --patch-size $patch_i --memory-size $mem_i --embeddings-size $emb_i --data-dir data --model $model_i
