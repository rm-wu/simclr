#!/bin/bash
#SBATCH --job-name=test_ddp      # Job name
#SBATCH --output=/scratch/project_462000585/mereuric/ssl_nat_aug/dino/logs/test_ddp.o%j # Name of stdout output file
#SBATCH --error=/scratch/project_462000585/mereuric/ssl_nat_aug/dino/logs/test_ddp.e%j  # Name of stderr error file
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
    /scratch/project_462000585/mereuric/ssl_nat_aug/dino/lumi_test.py 50 10
