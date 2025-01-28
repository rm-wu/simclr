module purge
export EBU_USER_PREFIX=/scratch/project_462000585/mereuric/EasyBuild
module load LUMI partition/container EasyBuild-user
mv modules/container/PyTorch/2.2.0-rocm-5.6.1-pythoin-3.10-siingularity-20240315.lua modules/LUMI/24.03/partition/container/
module load CrayEnv
module load 2.2.0-rocm-5.6.1-python-3.10-singularity-20240315
echo $SIF
singularity shell $SIF
