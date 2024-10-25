import os
import torch
import timm
import types
from .utils import (
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
from icecream import ic
from pathlib import Path


options = ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
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


def load_model(model_name):
    from timm.models.vision_transformer import Block, Attention
    setattr(Block, "get_attention_map", get_attention_map)
    setattr(Attention, "forward_attn", forward_attn)
    model = timm.create_model(timm_model_card[model_name], pretrained=True, num_classes=0, dynamic_img_size=True, dynamic_img_pad=False).to(device)
    model.get_intermediate_layers = types.MethodType(get_intermediate_layers, model)
    model.get_last_selfattention  = types.MethodType(get_last_selfattention,  model)
    return model

def compute_features(model, imgs):
    with torch.no_grad():
        ori_feats = model.get_intermediate_layers(imgs, n=[11], reshape=True, return_prefix_tokens=False, return_class_token=True, norm=True) # list of [B,768,24,24]
        # attn = model.get_last_selfattention(image_resized)
        return ori_feats[0][1]


