# How to install the environment on Lumi cluster

## Using modules
```bash
mkdir venv/
module purge
module load LUMI/23.09
module load lumi-container-wrapper/0.3.1-cray-python-3.10.10
module use /appl/local/csc/modulefiles/
module load pytorch
```
