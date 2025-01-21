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

from embed_utils import compute_embeddings, PadToMultipleOf
from util_utils import get_device
from ssl_libs.load_model import load_model
from knn import compute_knn
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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
    def __init__(self, output_dir, metadata_file, transform=None):
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        self.transform  = transform
        self.output_dir = output_dir
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        file_path = os.path.join(self.output_dir, entry["file_path"].split("/")[-1])
        file_path = os.path.join(self.output_dir, entry["file_path"])
        img = Image.open(file_path).convert("RGB")
        label = entry["label"]
        # Crop the non-zero regions of the image
        img_tensor = T.ToTensor()(img)
        non_zero_coords = torch.nonzero(img_tensor.sum(dim=0))
        if non_zero_coords.size(0) > 0:
            y_min, x_min = non_zero_coords.min(dim=0).values
            y_max, x_max = non_zero_coords.max(dim=0).values
            img_tensor = img_tensor[:, y_min:y_max+1, x_min:x_max+1]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label, file_path

def return_dataset(output_dir="ms_coco/ms_coco_objects", metadata_file="metadata.pkl", pad=-1):
    metadata_file = os.path.join(output_dir, metadata_file)
    if pad>0:
        transform = T.Compose([PadToMultipleOf(pad)])
    else:
        transform = None
    dataset = ObjectDataset(output_dir, metadata_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader


for MODEL_NAME in ['CLIP', 'DINOv2-reg', 'MAE']:
    # if embeddings are already computed, load them
    fname = os.path.join('ms_coco', f'{MODEL_NAME}_ms_coco_embeddings.pt')
    if os.path.exists(fname):
        embeddings, labels, file_paths = torch.load(fname)
    else:
        model = load_model(MODEL_NAME).to(device)
        data_loader = return_dataset(pad=16 if MODEL_NAME=='CLIP' else -1)
        embeddings, labels, file_paths = [],[],[]
        for img, label, paths in data_loader:
            img = img.to(device)
            embeddings.append(compute_embeddings(img, model, patchwise=False, normalize=False))
            labels.append(label)
            file_paths.extend(paths) 
        embeddings, labels = torch.cat(embeddings), torch.cat(labels)
        torch.save([embeddings, labels, file_paths], fname)
    print(embeddings.shape, labels.shape)
    retrieval_rate, misclassified_idx, nns = compute_knn(embeddings, labels, normalize=False)
    print(f'{MODEL_NAME} retrieval rate: {retrieval_rate} accuracy: {1-len(misclassified_idx)/embeddings.shape[0]}')
    retrieval_rate, misclassified_idx, nns = compute_knn(embeddings, labels, normalize=False)
    print(f'{MODEL_NAME} (normalized) retrieval rate: {retrieval_rate} accuracy: {1-len(misclassified_idx)/embeddings.shape[0]}')

    # for the first n images, plot the image and nearest neighbors
import matplotlib.pyplot as plt
invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])
N = 5
NNcount = nns.shape[1]
for n in range(N):
    fig, ax = plt.subplots(1, NNcount+1, figsize=(NNcount*2, 2))
    original_img = T.ToTensor()(Image.open(file_paths[n]).convert("RGB"))
    ax[0].imshow(invTrans(original_img).permute(1,2,0))
    ax[0].set_title(f'Original {labels[n]}')
    ax[0].axis('off')
    for i in range(NNcount):
        img = T.ToTensor()(Image.open(file_paths[nns[n,i]]).convert("RGB"))
        ax[i+1].imshow(invTrans(img).permute(1,2,0))
        ax[i+1].set_title(f'NN {i+1} ({labels[nns[n,i]]})')
        ax[i+1].axis('off')
    plt.savefig(f'ms_coco/{MODEL_NAME}_NN_{n}.png')
    plt.close()
