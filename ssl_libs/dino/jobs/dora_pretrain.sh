#!/bin/bash -l

#SBATCH --job-name=dora_pretrain   # Job name
#SBATCH --output=/scratch/project_462000585/mereuric/DoRA/logs/dora.o%j # Name of stdout output file
#SBATCH --error=/scratch/project_462000585/mereuric/DoRA/logs/dora.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # partition name
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus=8       # Allocate one gpu per MPI rank
#SBATCH --time=2-00:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000585  # Project for billing
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.mereu@aalto.fi
#SBATCH --hint=nomultithread

module purge
module load LUMI/23.09
module load lumi-container-wrapper/0.3.1-cray-python-3.10.10
module use /appl/local/csc/modulefiles/
module load pytorch
# export PATH="/scratch/project_462000585/mereuric/DoRA/venv_old/bin:$PATH"


set -x

torchrun --nproc_per_node=8 main.py --arch vit_small --data_path /data/venice.mp4   --output_dir /scratch/project_462000585/mereuric/DoRA/logs/weights_WTours/DoRA/venice_8frames/ --optimizer adamw   --use_bn_in_head False --out_dim 65536 --batch_size_per_gpu 64 --local_crops_number 6 --epochs 100 --num_workers 8 --lr 0.0005 --min_lr 0.00001  --norm_last_layer False --warmup_teacher_temp_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4 --frame_per_clip 8 --step_between_clips 60


set -x

python -m torch.distributed.launch --nproc_per_node=4 main.py --arch vit_small --data_path /tmp/data/venice.mp4   --output_dir /scratch/project_462000585/mereuric/DoRA/logs/weights_WTours/DoRA/venice_8frames/ --optimizer adamw   --use_bn_in_head False --out_dim 65536 --batch_size_per_gpu 64 --local_crops_number 6 --epochs 100 --num_workers 8 --lr 0.0005 --min_lr 0.00001  --norm_last_layer False --warmup_teacher_temp_epochs 30 --weight_decay 0.04 --weight_decay_end 0.4 --frame_per_clip 8 --step_between_clips 60

