# Install

```bash
module load mamba
mamba env create -f environment.yml
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
```

## Interactive session with a H100 on Tritoni
```bash
srun -p gpu-h100-80g --gres=gpu:h100:1 --time=12:00:00 --mem=256G -c 16 --mail-type=ALL --mail-user=riccardo.mereu@aalto.fi --pty bash
```
