### Create python env

```bash
conda create -n coco_dense python=3.12
```
### Install latest torch

```bash
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### Install pycocotools (required for torch.datasets.CocoDataset)

```bash
conda install conda-forge::pycocotools
```
### Install fit3d requirements:
```
pycocotools
jupyterlab
tqdm
plyfile==0.8.1
albumentations==1.3.1
omegaconf
scikit-learn
timm==0.9.10
torch_kmeans
gradio
spaces
```
### Download MS-COCO validation set and annotations json.
TODO: add link

### Run
- Run get_coco_patch_embeddings.py to save the patch embeddings for the whole validation set (5k images).
- Run play_w_embeddings.ipynb

