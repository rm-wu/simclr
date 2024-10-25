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
from icecream import ic
from pathlib import Path
import einops as ein


# %%
from timm.models.vision_transformer import Block, Attention

setattr(Block, "get_attention_map", get_attention_map)
setattr(Attention, "forward_attn", forward_attn)

options = ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
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
for model_option in options:
    ic(model_option)
    args = Namespace(
        image_path="/home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/images/1.png",
        output_dir="/home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/ssl_feats_vis/",
        kmeans=20,
        model_option=model_option,
        use_cbar=False,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.output_dir = (Path(args.output_dir) / args.model_option).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = original_models[args.model_option].to(device)

    p = model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(args.image_path)
    image = image.convert("RGB")
    image_resized = process_image(image, stride, transforms)
    image_resized = image_resized.to(device)

    with torch.no_grad():
        ori_feats = model.get_intermediate_layers(
            image_resized,
            n=[8, 9, 10, 11],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        attn = model.get_last_selfattention(image_resized)
    
    ic(attn.shape)
    ic(attn.max()) 
    ic(attn.min())
    ic(attn.mean())
    
    ori_feats = ori_feats[-1]
    if args.kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, args.kmeans)
    else:
        ori_labels = None
    plot_feats(image, model_option, ori_feats, None, ori_labels, None, args.output_dir, 0)

    if model_option in ["MAE", "DeiT-III"]:
        mean_attn = attn.mean(dim=1, keepdim=True)
        # attn = torch.where(attn > mean_attn, mean_attn, attn)
        attn = attn - attn.mean(dim=1, keepdim=True)
        # attn = attn / attn.std(dim=1, keepdim=True)
        if model_option == "MAE":
            attn = attn / attn.std(dim=1, keepdim=True)

    for idx, at in enumerate(attn[0]):
        at = at[0]
        if model.cls_token is not None:
            at = at[1:]
        if model.reg_token is not None:
            at = at[(model.reg_token.shape[1]) :]
        # ic(at.shape)
        if model_option in ["MAE", "CLIP", "DeiT-III"]:
            h, w = 24, 24
        elif model_option in ["DINOv2-reg", "DINOv2"]:
            h, w = 27, 27
        else:
            raise ValueError(f"Model {model_option} not supported")
        
        # TODO: make h and w dynamic
        at = ein.rearrange(at, "(h w) -> h w", h=h, w=w)
        if args.use_cbar:
            fig, (ax_img, ax_cbar) = plt.subplots(1, 2, figsize=(11, 10), 
                                            gridspec_kw={'width_ratios': [20, 1]})

            # Plot the attention map
            im = ax_img.imshow(at.cpu().numpy(), aspect='equal')
            ax_img.axis("off")
            
            # Add a small, vertical colorbar
            cbar = fig.colorbar(im, cax=ax_cbar)
            ax_cbar.yaxis.tick_right()
            ax_cbar.yaxis.set_label_position("right")
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"attention_vis_cbar_{idx}.png"))
            plt.show()
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            ax.imshow(at.cpu().numpy())
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"attention_vis{idx}.png"))
            plt.show()
            plt.close(fig)