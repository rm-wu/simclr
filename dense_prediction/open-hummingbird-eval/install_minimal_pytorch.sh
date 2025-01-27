#!/bin/bash
module purge
module use /appl/local/training/modules/AI-20241126
module load cotainr
module load singularity-userfilesystems
# cotainr build python312.sif --system=lumi-g --conda-env=python312.yml
cotainr build lumi_env.sif --system=lumi-g --conda-env=minimal_env.yml
singularity shell lumi_env.sif


srun --overlap --pty --jobid=<jobid> $SHELL

