#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-10:00            # Runtime in D-HH:MM
#SBATCH --mem=20000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/cyildiz40/slurm_logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/bethge/cyildiz40/slurm_logs/%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU
# #SBATCH --partition=a100-galvani
#SBATCH --partition=2080-galvani

# ssh -t cyildiz40@134.2.168.72 "cd /mnt/qb/work/bethge/cyildiz40/contrastive-continual-dynamics; squeue --user cyildiz40; conda activate default; bash -l"

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

source ~/.bashrc
conda activate riccardo
python eval.py --num_workers 8 --batch_size_per_device=64 --max_epochs=10 --ckpt_path=/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface/epoch=0-step=3000.ckpt --data_dir=/mnt/qb/work/bethge/cyildiz40/simclr/PetFace --seed=42 

# python runner.py --use_wandb --batch_size 64 --num_enc_filt 64 --num_dec_filt 64 --lamb0 100 --lamb1 1000 --q 64 --num_groups 4 --num_epoch 10 --num_tasks 2 --split random --num_shapes 8 --q_shp 64 --act_fn GELU --dataset idsprites --hard --use_gt_interv --num_frames 2 --object_flow --resnet
# python runner.py --batch_size 64 --num_enc_filt 64 --num_dec_filt 64 --plot_interval 500  --lamb0 100 --lamb1 10000 --q 8 --num_groups 4 --num_epoch 200 --split random --num_shapes 10 --q_shp 16 --use_gt_shapes --act_fn GELU --dataset idsprites --hard --node --solver 'rk4' --positive_t 
