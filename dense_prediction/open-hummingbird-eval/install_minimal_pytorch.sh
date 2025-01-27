#!/bin/bash
module purge
module use /appl/local/training/modules/AI-20241126
module load cotainr
module load singularity-userfilesystems
# cotainr build python312.sif --system=lumi-g --conda-env=python312.yml
# cotainr build lumi_env.sif --system=lumi-g --conda-env=minimal_env.yml
cotainr build lumi_env.sif --system=lumi-g --conda-env=environment_lumi.yml --accept-licenses
singularity shell lumi_env.sif


# Inside the container
python linear_eval.py --model=dino_vits16 --input-size=512 --patch-size=16 --embeddings-size=384 --dataset-name=voc --data-dir=/flash/project_462000585/mereuric/data --out-dir=outputs/ --num-workers=8 --seed=36

# how to attach to a running job
srun --overlap --pty --jobid=<jobid> $SHELL

