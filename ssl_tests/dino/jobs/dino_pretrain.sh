#!/bin/bash
#SBATCH --job-name=dino_pretrain      # Job name
#SBATCH --output=/scratch/project_462000585/mereuric/ssl_nat_aug/ssl_tests/dino/logs/dino_pretrain.o%j # Name of stdout output file
#SBATCH --error=/scratch/project_462000585/mereuric/ssl_nat_aug/ssl_tests/dino/logs/dino_pretrain.e%j  # Name of stderr error file
#SBATCH --partition=standard-g   # partition name
#SBATCH --time=0-00:20:00        # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000585  # Project for billing
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.mereu@aalto.fi
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=7
#SBATCH --gpus-per-node=8

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Select the host and set up a random port
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
MASTER_ADDR=${nodes_array[0]}
export LOGLEVEL=DEBUG
export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"

# Set LUMI additional LUMI parameters, others are included in the pytorch module.
export NCCL_DEBUG=INFO
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export NCCL_NET_GDR_LEVEL=3

echo NODE IP: $MASTER_ADDR
srun python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    /scratch/project_462000585/mereuric/ssl_nat_aug/ssl_tests/dino/main_dino.py \
    --arch=vit_small \
    --patch_size=16 \
    --out_dim=65536 \
    --norm_last_layer=false \
    --warmup_teacher_temp=0.04 \
    --teacher_temp=0.07 \
    --warmup_teacher_temp_epochs=30 \
    --use_fp16=false \
    --weight_decay=0.04 \
    --weight_decay_end=0.4 \
    --clip_grad=0 \
    --batch_size_per_gpu=64 \
    --epochs=800 \
    --freeze_last_layer=1 \
    --lr=0.0005 \
    --warmup_epochs=10 \
    --min_lr=1e-05 \
    --global_crops_scale 0.25 1.0 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number=10 \
    --seed=0 \
    --num_workers=10 \
    --optimizer=adamw \
    --momentum_teacher=0.996 \
    --use_bn_in_head=false \
    --drop_path_rate=0.1 \
    --data_path=/scratch/project_462000585/mereuric/ssl_nat_aug/ssl_tests/dino/data/imagenet/ \
    --output_dir=/scratch/project_462000585/mereuric/ssl_nat_aug/ssl_tests/dino/outputs/