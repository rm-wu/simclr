
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import timm
import json
import torch
import pickle
import torch.nn.functional as F
from torchvision import models, datasets, tv_tensors
from torchvision import transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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


ROOT = '/weka/datasets/coco'
# ROOT = '/Users/hizlic1/repository-object-centric/ms-coco'
ANNOTATIONS_PATH = f'{ROOT}/annotations/instances_val2017.json'
with open(ANNOTATIONS_PATH, 'r') as f:
    root = json.load(f)
categ_map = {x['id']: '_'.join(x['name'].split( )) for x in root['categories']}
for k in categ_map.keys():
    print(k,'->',categ_map[k], end="\n")


# Data loader for the saved objects
class ObjectDataset(Dataset):
    def __init__(self, output_dir, metadata_file, transform=None, crop_nonzero=False, pad=-1):
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        self.transform  = transform
        self.output_dir = output_dir
        self.crop_nonzero = crop_nonzero
        self.pad = PadToMultipleOf(pad) if pad>0 else None
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        file_path = os.path.join(self.output_dir, entry["file_path"].split("/")[-1])
        # file_path = os.path.join(self.output_dir, entry["file_path"])
        img = Image.open(file_path).convert("RGB")
        label = entry["label"]
        # Crop the non-zero regions of the image
        if self.transform:
            img_tensor = self.transform(img)
        if self.crop_nonzero:
            non_zero_coords = torch.nonzero(img_tensor.sum(dim=0))
            if non_zero_coords.size(0) > 0:
                y_min, x_min = non_zero_coords.min(dim=0).values
                y_max, x_max = non_zero_coords.max(dim=0).values
                img_tensor = img_tensor[:, y_min:y_max+1, x_min:x_max+1]
        if self.pad is not None:
            img_tensor = self.pad(img_tensor)
        return img_tensor, label, file_path

def return_data_loader(output_dir="ms_coco/ms_coco_objects", metadata_file="metadata.pkl", pad=-1):
    # assumes that saved images are 322 x 322
    metadata_file = os.path.join(output_dir, metadata_file)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    dataset = ObjectDataset(output_dir, metadata_file, transform=transform, pad=pad)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])

MODELS = ['DINOv2-reg']
for MODEL_NAME in MODELS:
    # if embeddings are already computed, load them
    fname = os.path.join('ms_coco', f'{MODEL_NAME}_ms_coco_embeddings.pt')
    if os.path.exists(fname):
        embeddings, labels, file_paths = torch.load(fname)
    else:
        model = load_model(MODEL_NAME).to(device)
        data_loader = return_data_loader(pad=16 if MODEL_NAME=='CLIP' else -1)
        embeddings, labels, file_paths = [],[],[]
        for img, label, paths in data_loader:
            img = img.to(device)
            embeddings.append(compute_embeddings(img, model, patchwise=False, normalize=False))
            labels.append(label)
            file_paths.extend(paths) 
            # print(len(file_paths)//32)
        embeddings, labels = torch.cat(embeddings), torch.cat(labels)
        torch.save([embeddings, labels, file_paths], fname)
    print(embeddings.shape, labels.shape)
    retrieval_rate, misclassified_idx, nns = compute_knn(embeddings, labels, normalize=False, data_portion=1/32)
    print(f'{MODEL_NAME} retrieval rate: {retrieval_rate} accuracy: {1-len(misclassified_idx)/embeddings.shape[0]}')
    retrieval_rate, misclassified_idx, nns = compute_knn(embeddings, labels, normalize=True, data_portion=1/32)
    print(f'{MODEL_NAME} (normalized) retrieval rate: {retrieval_rate} accuracy: {1-len(misclassified_idx)/embeddings.shape[0]}')

# for the first n images, plot the image and nearest neighbors
misclass_idx, small_idx = 0,0
NNcount = nns.shape[1]
for n in range(embeddings.shape[0]):
    original_img = Image.open(file_paths[n]).convert("RGB");
    # if the label is the same as the nearest neighbor, then it is a correct retrieval
    if labels[n] == labels[nns[n,0]]:
        continue
    if (T.ToTensor()(original_img)!=0).to(torch.float16).mean() < 0.04:
        # print('too small object')
        small_idx += 1
        continue
    try:
        fig, ax = plt.subplots(1, NNcount+1, figsize=(NNcount*2, 2))
        ax[0].imshow(original_img);
        ax[0].set_title(f'Original {categ_map[labels[n]]}');
        ax[0].axis('off');
        for i in range(NNcount):
            img = Image.open(file_paths[nns[n,i]]).convert("RGB");
            ax[i+1].imshow(img);
            ax[i+1].set_title(f'NN {i+1} ({categ_map[labels[nns[n,i]]]})');
            ax[i+1].axis('off');
        plt.savefig(f'ms_coco/{MODEL_NAME}_NN_{misclass_idx}.png');
        misclass_idx += 1
        plt.close();
        print(n+1, small_idx/(n+1), misclass_idx/(n+1))
    except Exception as e:
        print(e)
        continue