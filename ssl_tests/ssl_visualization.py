# %%
import os
import torch
import timm
import types
from utils import get_intermediate_layers, plot_feats, kmeans_clustering, process_image, transforms
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from PIL import Image
from icecream import ic
from pathlib import Path



#%%
options = ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']

timm_model_card = {
    "DINOv2": "vit_small_patch14_dinov2.lvd142m",
    "DINOv2-reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "CLIP": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "MAE": "vit_base_patch16_224.mae",
    "DeiT-III": "deit3_base_patch16_224.fb_in1k"
}

our_model_card = {
    "DINOv2": "dinov2_small_fine",
    "DINOv2-reg": "dinov2_reg_small_fine",
    "CLIP": "clip_base_fine",
    "MAE": "mae_base_fine",
    "DeiT-III": "deit3_base_fine"
}

os.environ['TORCH_HOME'] = '/tmp/.cache'

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
        get_intermediate_layers,
        original_models[option]
    )

    # Uncomment the following lines to use the fine-tuned models from Fit3D
    # fine_models[option] = torch.hub.load("ywyue/FiT3D", our_model_card[option]).to(device)
    # fine_models[option].get_intermediate_layers = types.MethodType(
    #     get_intermediate_layers,
    #     fine_models[option]
    # )

# %%


# %%
for model_option in options:
    args = Namespace(image_path="/home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/images/1.png", 
                    output_dir="ssl_feats_vis/", 
                    kmeans=20,
                    model_option=model_option)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.output_dir = (Path(args.output_dir) / args.model_option).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    # model = prepare_model(args.pretrained_weights, args.arch, )

    model = original_models[args.model_option].to(device)
    p = model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(args.image_path)
    image = image.convert("RGB")
    image_resized = process_image(image, stride, transforms)
    image_resized = image_resized.to(device)
    ic(image_resized.shape)
    ic(image_resized.dtype)
    with torch.no_grad():
        ori_feats = model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                    return_class_token=False, norm=True)


    ori_feats = ori_feats[-1]

    if args.kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, args.kmeans)
    else:
        ori_labels = None
        
    plot_feats(image, "MAE", ori_feats, None, ori_labels, None, args.output_dir, 0)
# %%
