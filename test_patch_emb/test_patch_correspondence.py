# %%
from argparse import Namespace
import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image


# %%
# Arguments

# Local
# IMAGES_PATH = '/Users/hizlic1/repository-object-centric/ms-coco/val2017'
# ANNOTATIONS_PATH = '/Users/hizlic1/repository-object-centric/ms-coco/annotations/instances_val2017.json'

# ml4h-gpu
IMAGES_PATH = "/ssd/hizlic1/repository-object-centric/ms-coco-ml4h/val2017/1"
ANNOTATIONS_PATH = "/ssd/hizlic1/repository-object-centric/ms-coco-ml4h/annotations/instances_val2017.json"

model_option = 'DINOv2'
args = Namespace(
    dataset_path=IMAGES_PATH,
    annotations_path=ANNOTATIONS_PATH,
    num_query_patches=5,  # > 1 for majority voting
    task='obj_level_1nn',  # obj_level_1nn or img_level_1nn
)

# %%
import json

with open(args.annotations_path, 'r') as f:
    root = json.load(f)

categ_map = {x['id']: '_'.join(x['name'].split( )) for x in root['categories']}


# %%
# validation dataset
res = 322
transform = T.Compose([T.Resize(res, Image.NEAREST), T.CenterCrop(res), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
dataset_val = datasets.CocoDetection(root=args.dataset_path, annFile=args.annotations_path, transform=transform)
dataset_val = datasets.wrap_dataset_for_transforms_v2(dataset_val, target_keys=["boxes", "labels", "masks", "image_id", "segmentation"])
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=100,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)

invTrans = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),])

# %%
# Local
# emb_path = 'ssl_feats/DINOv2/embeddings_val.pt'
# ml4h-gpu
emb_path = '/ssd/hizlic1/repository-object-centric/ssl_nat_aug/test_patch_emb/ssl_feats/DINOv2/embeddings_val.pt'
z = torch.load(emb_path)

# %%

torch.manual_seed(1)
num_img = 5000
# Sample image
dataset_accuracy = 0.0
num_objects = 0
num_query_patches = args.num_query_patches
for img_idx in range(num_img):
    img, target = dataset_val[img_idx]
    if 'labels' in target:
        # Find objects 
        resize_transform = T.Compose([T.Resize(322, Image.NEAREST), T.CenterCrop(322)])
        obj_ids, obj_labels, obj_strs = [], [], []
        obj_patch_coords = []
        for ll in range(len(target['labels'])):
            obj_mask = target['masks'][ll]
            obj_mask = resize_transform(obj_mask.unsqueeze(0).float()).repeat(3, 1, 1)
            if obj_mask.any():
                obj_ids.append(ll)
                obj_labels.append(target['labels'][ll].item())
                obj_str = categ_map[target['labels'][ll].item()]
                obj_strs.append(obj_str)
                # img_masked = invTrans(img) * obj_mask

                obj_mask_patches = obj_mask.unsqueeze(0).unfold(2, 14, 14).unfold(3, 14, 14).squeeze(0)
                obj_mask_patches = obj_mask_patches.squeeze(0).permute(1, 2, 0, 3, 4)
                obj_patch_ids = obj_mask_patches.nonzero()[:, :2].unique(dim=(0))

                query_indices = torch.randperm(len(obj_patch_ids))[:num_query_patches]
                if len(query_indices) < num_query_patches:
                    query_indices = torch.cat([query_indices, torch.randint(0, len(obj_patch_ids), (num_query_patches - len(query_indices),))])

                single_obj_coords = []
                for idx in query_indices:
                    obj_patch_idx = obj_patch_ids[idx]
                    obj_row, obj_col = obj_patch_idx[0].item(), obj_patch_idx[1].item()
                    single_obj_coords.append((obj_row, obj_col))
                obj_patch_coords.append(single_obj_coords)
                
        # print('All objects after resize (Source)', obj_strs)
        num_obj = len(obj_ids)
        if num_obj == 0:
            continue

        zs = []
        # for loop over objects
        for single_obj_coords in obj_patch_coords:
            # num_query_patches patches for each object
            z_objects = torch.stack([z[img_idx, :, obj_row, obj_col] for obj_row, obj_col in single_obj_coords])
            zs.append(z_objects)
        z_objects = torch.stack(zs)

        # print(z_objects.shape)
        attn_inter = torch.einsum("oqc,ncij->oqnij", F.normalize(z_objects, dim=1), F.normalize(z, dim=1))
        target_img_indices = list(range(num_img))
        target_img_indices.remove(img_idx)
        attn_inter = attn_inter[:, :, target_img_indices].contiguous()
        attn_inter -= attn_inter.mean([3, 4], keepdims=True)
        attn_inter = attn_inter.clamp(0)

        obj_patch_similarities = attn_inter
        img_acc = 0.0

        for ii, (obj_label, obj_str) in enumerate(zip(obj_labels, obj_strs)):
            obj_acc = 0.0
            for qq in range(num_query_patches):
                obj_row, obj_col = obj_patch_coords[ii][qq]
                topk_tensor = torch.topk(obj_patch_similarities[ii, qq].flatten(), 1, sorted=True)
                # print(topk_tensor.values)
                indices = torch.unravel_index(topk_tensor.indices, obj_patch_similarities.shape[2:])
                target_idx = indices[0].item()
                target_obj_row, target_obj_col = indices[1].item(), indices[2].item()
                # print(target_idx)
                if target_idx >= img_idx:
                    target_idx += 1
                img_target, target_target = dataset_val[target_idx]

                # print('All objects before resize (Target)', [categ_map[l.item()] for l in target_target['labels']])
                if 'labels' in target_target:
                    obj_labels_target, obj_strs_target = [], []
                    for jj in range(len(target_target['labels'])):
                        obj_mask = target_target['masks'][jj]
                        obj_mask = resize_transform(obj_mask.unsqueeze(0).float()).repeat(3, 1, 1)
                        obj_mask_patches = obj_mask.unsqueeze(0).unfold(2, 14, 14).unfold(3, 14, 14).squeeze(0)
                        obj_mask_patches = obj_mask_patches.squeeze(0).permute(1, 2, 0, 3, 4)

                        # image-level or object-level task
                        if args.task.startswith('img'):
                            obj_exists_at_target = obj_mask_patches.any()
                        else:
                            obj_exists_at_target = obj_mask_patches[target_obj_row, target_obj_col].any()
                            
                        if obj_exists_at_target:
                            obj_labels_target.append(target_target['labels'][jj].item())
                            obj_strs_target.append(categ_map[target_target['labels'][jj].item()])
                    # print('All objects after resize (Target)', obj_strs_target)

                    if len(obj_labels_target) > 0:
                        obj_acc += np.any(np.array(obj_labels_target) == obj_label).astype(int)
                # print(img_idx, ii, qq, obj_label, obj_str, obj_acc)

            # Majority voting
            if obj_acc > num_query_patches // 2:
                obj_acc = 1.0
            else:
                obj_acc = 0.0
            # print(f'Object Accuracy for {obj_label}:', obj_acc)
            img_acc += obj_acc

            # fig = plt.figure(figsize=(5*2, 5))
            # plt.subplot(121)
            # plt.title(f'Source Image, Obj. Label: {obj_label}, Name: {obj_str}')
            # plt.imshow(invTrans(img).permute(1,2,0))
            # plt.axis('off')

            # plt.subplot(122)
            # plt.title(f'Target Image, Object Accuracy: {obj_acc}')

            # plt.imshow(invTrans(img_target).permute(1,2,0))
            # plt.axis('off')

            # plt.tight_layout()
            # plt.show()
        print(f'Img:{img_idx} w/ {len(obj_labels)} objects, Img Acc: {img_acc / len(obj_labels):.3f}')
        dataset_accuracy += img_acc
        num_objects += len(obj_labels)
        print(f'Until this point,  #Objects={num_objects},  Dataset Acc={dataset_accuracy / num_objects:.3f}')
dataset_accuracy /= num_objects
print(f'Dataset Accuracy: {dataset_accuracy:.3f}')

# %%
