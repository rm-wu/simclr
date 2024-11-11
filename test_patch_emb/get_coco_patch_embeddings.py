# %%
import os
import torch
import timm
import types
from utils import (
    get_intermediate_layers,
    plot_feats,
    kmeans_clustering,
    process_image,
    transforms,
    get_attention_map,
    get_last_selfattention,
    forward_attn,
)
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from PIL import Image
from pathlib import Path
import einops as ein
from torchvision import datasets
from torchvision import transforms as T


# %%
from timm.models.vision_transformer import Block, Attention

setattr(Block, "get_attention_map", get_attention_map)
setattr(Attention, "forward_attn", forward_attn)

options = ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
options = ['DINOv2']
# options = ["DeiT-III"]

timm_model_card = {
    "DINOv2": "vit_small_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "CLIP": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "MAE": "vit_base_patch16_224.mae",
    "DeiT-III": "deit3_base_patch16_224.fb_in1k",
}

our_model_card = {
    "DINOv2": "dinov2_small_fine",
    "DINOv2-reg": "dinov2_reg_small_fine",
    "CLIP": "clip_base_fine",
    "MAE": "mae_base_fine",
    "DeiT-III": "deit3_base_fine",
}

os.environ["TORCH_HOME"] = "/tmp/.cache"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Pre-load all models
original_models = {}
fine_models = {}
for option in options:
    original_models[option] = timm.create_model(
        timm_model_card[option],
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )
    original_models[option].get_intermediate_layers = types.MethodType(
        get_intermediate_layers, original_models[option]
    )
    original_models[option].get_last_selfattention = types.MethodType(
        get_last_selfattention, original_models[option]
    )

    # Uncomment the following lines to use the fine-tuned models from Fit3D
    # fine_models[option] = torch.hub.load("ywyue/FiT3D", our_model_card[option]).to(device)
    # fine_models[option].get_intermediate_layers = types.MethodType(
    #     get_intermediate_layers,
    #     fine_models[option]
    # )


# %%

# %%
# Get embeddings
def get_embeddings(model, dataloader):
    p = model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    embeddings = []
    for i, batch in enumerate(dataloader):
        img, _ = batch
        img = torch.stack(img)
        h, w = img.shape[2:]
        height_int = (h // stride)*stride
        width_int = (w // stride)*stride
        img_resized = torch.nn.functional.interpolate(img, size=(height_int, width_int), mode='bilinear')
        with torch.no_grad():
            ori_feats = model.get_intermediate_layers(
                img_resized,
                n=[11],
                reshape=True,
                return_prefix_tokens=False,
                return_class_token=False,
                norm=True,
            )
        ori_feats = ori_feats[-1]
        embeddings.append(ori_feats.cpu())
    embeddings = torch.cat(embeddings)
    return embeddings

# %%
# for model_option in options:

model_option = 'DINOv2'
args = Namespace(
    dataset_path="/ssd/hizlic1/repository-object-centric/ms-coco-ml4h/val2017/1",
    annotations_path="/ssd/hizlic1/repository-object-centric/ms-coco-ml4h/annotations/instances_val2017.json",
    output_dir="/ssd/hizlic1/repository-object-centric/ssl_nat_aug/ssl_tests/ssl_feats/",
    kmeans=-1,
    model_option=model_option,
    use_cbar=False,
)

# Load COCO dataset
res = 330
transform = T.Compose([T.Resize(res, Image.NEAREST), T.CenterCrop(res), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
dataset_val = datasets.CocoDetection(root=args.dataset_path, annFile=args.annotations_path, transform=transform)
dataset_val = datasets.wrap_dataset_for_transforms_v2(dataset_val, target_keys=["boxes", "labels"])
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=100,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.output_dir = (Path(args.output_dir) / args.model_option).resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)

model = original_models[args.model_option].to(device)

embeddings = get_embeddings(model, dataloader_val)
torch.save(embeddings, os.path.join(args.output_dir, 'embeddings_val.pt'))

# if args.kmeans != -1:
#     ori_labels = kmeans_clustering(ori_feats, args.kmeans)
# else:
#     ori_labels = None
# plot_feats(image, model_option, ori_feats, None, ori_labels, None, args.output_dir, 0)
