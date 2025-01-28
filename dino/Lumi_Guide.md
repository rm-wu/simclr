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


```bash
srun --partition=standard-g --time=0-01:00:00 --account=project_462000585 --mem=200G --exclusive --nodes=2 --ntasks-per-node=1 --cpus-per-gpu=7 --gpus-per-node=8 bash
```
