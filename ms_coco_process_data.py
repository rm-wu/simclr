# %%
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

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

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


# %%
torch.manual_seed(0)
#

ROOT = '/weka/datasets/coco'
# ROOT = '/Users/hizlic1/repository-object-centric/ms-coco'
IMAGES_PATH = f'{ROOT}/images/val2017'
ANNOTATIONS_PATH = f'{ROOT}/annotations/instances_val2017.json'

dataset_untransformed = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)
dataset_untransformed = datasets.wrap_dataset_for_transforms_v2(dataset_untransformed, target_keys=("boxes", "labels", "masks", "image_id", "segmentation"))

# %%
import json

with open(ANNOTATIONS_PATH, 'r') as f:
    root = json.load(f)

root.keys()

n_images = len(root['images'])
n_boxes = len(root['annotations'])
n_categories = len(root['categories'])

heights = [x['height'] for x in root['images']]
widths = [x['width'] for x in root['images']]

# print('Dataset Name: ',src_desc)
print('Number of images: ',n_images)
print('Number of bounding boxes: ', n_boxes)
print('Number of classes: ', n_categories)
print('Max min avg height: ', max(heights), min(heights), int(sum(heights)/len(heights)))
print('Max min avg width: ', max(widths), min(widths), int(sum(widths)/len(widths)))

categ_map = {x['id']: '_'.join(x['name'].split( )) for x in root['categories']}
for k in categ_map.keys():
    print(k,'->',categ_map[k], end="\n")


# %%
res = 322
transform = T.Compose([T.Resize(res, Image.NEAREST), T.CenterCrop(res), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
dataset_val = datasets.CocoDetection(root=IMAGES_PATH, annFile=ANNOTATIONS_PATH, transform=transform)
dataset_val = datasets.wrap_dataset_for_transforms_v2(dataset_val, target_keys=["boxes", "labels", "masks", "image_id", "segmentation"])
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=100,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),])

# %%
for i, batch in enumerate(dataloader_val):
    break

images, targets = batch
images = torch.stack(images)
print(images.shape)

# %%
idx = 1
img, target = dataset_untransformed[idx]
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()


# %%

# Create directory for saving images
output_dir = "mscoco_objects"
os.makedirs(output_dir, exist_ok=True)

# Initialize transform for resizing and cropping
resize_transform = T.Compose([
    T.Resize(322, interpolation=T.InterpolationMode.NEAREST),
    T.CenterCrop(322)
])

# Loop through the dataset
save_interval = 100  # Save progress every 100 images
saved_metadata = []

def save_object(image_tensor, label, obj_idx, img_idx):
    file_name = f"img_{img_idx}_obj_{obj_idx}.png"
    file_path = os.path.join(output_dir, file_name)
    # Convert tensor to PIL image and save
    img_pil = T.ToPILImage()(image_tensor)
    img_pil.save(file_path)
    return {
        "file_path": file_path,
        "label": label
    }

MASK_OUT = False
L = len(dataset_val)
# L = 201
for idx in range(L):
    img, target = dataset_val[idx]
    if len(list(target.keys()))!=5:
        continue
    for obj_idx in range(len(target['labels'])):
        obj_mask = target['masks'][obj_idx]
        obj_mask = resize_transform(obj_mask.unsqueeze(0).float())
        if obj_mask.any():  # Ensure there is an object
            if MASK_OUT:
                # Remove black (masked-out) areas
                obj_mask = obj_mask.squeeze(0).byte()
                img_masked = img.clone()
                img_masked[:, obj_mask == 0] = 0  # Black out masked regions
                # Save the object image and label metadata
                metadata = save_object(img_masked, target['labels'][obj_idx].item(), obj_idx, idx)
            else:
                metadata = save_object(img, target['labels'][obj_idx].item(), obj_idx, idx)
            saved_metadata.append(metadata)
    # Periodically save metadata to a file
    if idx % save_interval == 0 and idx > 0:
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(saved_metadata, f)
        print(f"Metadata saved. Processed {idx} images.")

# Final save of metadata
with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(saved_metadata, f)

print("Final metadata saved.")

# %% [markdown]
# ## data loader to load all objects

# %%
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
        return T.ToTensor()(img), label

# Example usage of the data loader
metadata_file = os.path.join(output_dir, "metadata.pkl")
dataset = ObjectDataset(metadata_file)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for images, labels in data_loader:
    print(images.shape, labels.shape)


# %%

def get_device(model):
    return list(model.parameters())[0].device

def compute_embeddings(frames, network, patchwise=False, normalize=True):
    device = get_device(network)
    if isinstance(frames,list):
        assert len(frames[0].shape)==3, 'frames should have 4 dims'
        if frames[0].shape[-3] != 3:
            frames = [frame.permute(2,0,1) for frame in frames]
        frames = [frame.to(device) for frame in frames]
        with torch.no_grad():
            if patchwise:
                embeddings = [network.get_intermediate_layers(frame.unsqueeze(0), n=[11], reshape=True, return_prefix_tokens=False, norm=True)[0] for frame in frames]
            else:
                embeddings = [network(frame.unsqueeze(0)) for frame in frames]
        embeddings = torch.cat(embeddings)
    else:
        assert frames.ndim==4, 'frames should have 4 dimensions'
        if frames.shape[-3] != 3:
            frames = frames.permute(0,3,1,2).to(device)
        with torch.no_grad():
            if patchwise:
                embeddings = network.get_intermediate_layers(frames, n=[11], reshape=True, return_prefix_tokens=False, norm=True)[0]
            else:
                embeddings = network(frames)
    if normalize:
        embeddings = embeddings / embeddings.norm(2,-1,keepdim=True)
    return embeddings


