import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import timm
import torch
import pickle
import torch.nn.functional as F
from torchvision import models, datasets, tv_tensors
from torchvision import transforms as T
from torchvision.utils import make_grid

from torch.utils.data import Dataset, DataLoader

import types
import albumentations as A
import seaborn as sns

from PIL import Image
from sklearn.decomposition import PCA
from torch_kmeans import KMeans, CosineSimilarity

from matplotlib.colors import ListedColormap
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
cmaps = [
    ListedColormap([(1, 0, 0, i / 255) for i in range(255)]),
    ListedColormap([(0, 1, 0, i / 255) for i in range(255)]),
    ListedColormap([(0, 0, 1, i / 255) for i in range(255)]),
    ListedColormap([(1, 1, 0, i / 255) for i in range(255)])
]
plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = T.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# Data loader for the saved objects
class ObjectDataset(Dataset):
    def __init__(self, metadata_file):
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        img = Image.open(entry["file_path"]).convert("RGB")
        label = entry["label"]
        # Crop the non-zero regions of the image
        img_tensor = T.ToTensor()(img)
        non_zero_coords = torch.nonzero(img_tensor.sum(dim=0))
        if non_zero_coords.size(0) > 0:
            y_min, x_min = non_zero_coords.min(dim=0).values
            y_max, x_max = non_zero_coords.max(dim=0).values
            img_tensor = img_tensor[:, y_min:y_max+1, x_min:x_max+1]
        return img_tensor, label

def return_dataset(output_dir="ms_coco_objects", metadata_file="metadata.pkl"):
    metadata_file = os.path.join(output_dir, metadata_file)
    dataset = ObjectDataset(metadata_file)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader

# compute embeddings
from embed_utils import compute_embeddings, PadToMultipleOf14
from util_utils import get_device
from ssl_libs.load_model import load_model
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

for MODEL_NAME in ['CLIP', 'DINOv2-reg', 'MAE']:
    model = load_model(MODEL_NAME).to(device)
    data_loader = return_dataset(output_dir=f'mscoco_objects')

