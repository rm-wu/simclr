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

# python linear_prob.py --num_workers 8 --batch_size_per_device=64 --max_epochs=10 --ckpt_path=/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface/epoch=9-step=48000.ckpt --data_dir=/mnt/qb/work/bethge/cyildiz40/simclr/PetFace --seed=42 

python retrieval_task.py --ckpt_path=/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface-nat/epoch=8-step=43000.ckpt --data_dir=/mnt/qb/work/bethge/cyildiz40/simclr/PetFace 
