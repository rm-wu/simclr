### Create python env

```bash
conda create -n coco_dense python=3.12
conda activate coco_dense
```
### Install latest torch

```bash
conda install pytorch::pytorch torchvision -c pytorch 
```

### Install pycocotools (required for torch.datasets.CocoDataset)

```bash
conda install conda-forge::pycocotools
```
### Install fit3d requirements:
```bash
pip install -r requirements.txt
```
### Download MS-COCO validation set and annotations json.
Download MS-COCO from this [link](https://cocodataset.org/#download)

### Run
- Run get_coco_patch_embeddings.py to save the patch embeddings for the whole validation set (5k images).
- Run play_w_embeddings.ipynb

