#!/bin/bash
module purge
module use /appl/local/training/modules/AI-20241126
module load cotainr
# cotainr build python312.sif --system=lumi-g --conda-env=python312.yml
