# %%
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import skimage.io
from skimage.measure import find_contours
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import timm
import albumentations as A
from torch_kmeans import KMeans, CosineSimilarity
import types

import models_mae


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def viz_feat(feat):

    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

    return res_pred


def plot_feats(image, model_option, ori_feats, fine_feats, ori_labels=None, fine_labels=None, output_dir=None, n_head=None):

    ori_feats_map = viz_feat(ori_feats)
    # fine_feats_map = viz_feat(fine_feats)

    if ori_labels is not None:
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax[0][0].imshow(image)
        ax[0][0].set_title("Input image", fontsize=15)
        ax[0][1].imshow(ori_feats_map)
        ax[0][1].set_title("Original " + model_option, fontsize=15)
        ax[1][1].imshow(ori_labels)
        for xx in ax:
          for x in xx:
            x.xaxis.set_major_formatter(plt.NullFormatter())
            x.yaxis.set_major_formatter(plt.NullFormatter())
            x.set_xticks([])
            x.set_yticks([])
            x.axis('off')

    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=15)
        ax[1].imshow(ori_feats_map)
        ax[1].set_title("Original " + model_option, fontsize=15)

        for x in ax:
          x.xaxis.set_major_formatter(plt.NullFormatter())
          x.yaxis.set_major_formatter(plt.NullFormatter())
          x.set_xticks([])
          x.set_yticks([])
          x.axis('off')

    plt.tight_layout()
    if output_dir is not None and n_head is not None:
        plt.savefig(os.path.join(output_dir, f"attention_vis{n_head}.png"))
    plt.close(fig)
    return fig


def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)


def process_image(image, stride, transforms):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed['image'])
    image_tensor = image_tensor.permute(2,0,1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    h, w = image_tensor.shape[2:]

    height_int = (h // stride)*stride
    width_int = (w // stride)*stride

    image_resized = torch.nn.functional.interpolate(image_tensor, size=(height_int, width_int), mode='bilinear')

    return image_resized


def kmeans_clustering(feats_map, n_clusters=20):

    B, D, h, w = feats_map.shape
    feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats_map_flattened)
    labels = kmeans_engine.predict(
        feats_map_flattened
        )
    labels = labels.reshape(
        B, h, w
        ).float()
    labels = labels[0].cpu().numpy()

    label_map = cmap(labels / n_clusters)[..., :3]
    label_map = np.uint8(label_map * 255)
    label_map = Image.fromarray(label_map)

    return label_map



def run_demo(model_option, image_path, kmeans=20):
    """
    Run the demo for a given model option and image
    model_option: ['DINOv2', 'DINOv2-reg', 'CLIP', 'MAE', 'DeiT-III']
    image_path: path to the image
    kmeans: number of clusters for kmeans. Default is 20. -1 means no kmeans.
    """
    original_model = original_models[model_option]
    fine_model = fine_models[model_option]
    p = original_model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(image_path)
    image_resized = process_image(image, stride, transforms)
    with torch.no_grad():
        ori_feats = original_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                    return_class_token=False, norm=True)
        fine_feats = fine_model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                    return_class_token=False, norm=True)

    ori_feats = ori_feats[-1]
    fine_feats = fine_feats[-1]

    if kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, kmeans)
        fine_labels = kmeans_clustering(fine_feats, kmeans)
    else:
        ori_labels = None
        fine_labels = None


    return plot_feats(image, model_option, ori_feats, fine_feats, ori_labels, fine_labels, output_dir, 0)

# def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
#     # build model
#     model = getattr(models_mae, arch)()
#     # load model    
#     checkpoint = torch.load(chkpt_dir, map_location='cpu')
#     msg = model.load_state_dict(checkpoint['model'], strict=False)
#     print(msg)
#     return model

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
        ).to(device)
    original_models[option].get_intermediate_layers = types.MethodType(
        get_intermediate_layers,
        original_models[option]
    )

    fine_models[option] = torch.hub.load("ywyue/FiT3D", our_model_card[option]).to(device)
    fine_models[option].get_intermediate_layers = types.MethodType(
        get_intermediate_layers,
        fine_models[option]
    )


cmap = plt.get_cmap("tab20")
MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

transforms = A.Compose([
            A.Normalize(mean=list(MEAN), std=list(STD)),
    ])




def get_args_parser():
    parser = argparse.ArgumentParser('MAE Attention Visualization', add_help=False)
    parser.add_argument('--arch', default='mae_vit_large_patch16', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained_weights', default='mae/mae_visualize_vit_large.pth', type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--pretrained_weights", default=None, type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='./mae_attention_vis', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for attention map visualization.")
    parser.add_argument("--kmeans", type=int, default=-1, help="Number of clusters for kmeans. -1 means no kmeans.")
    return parser


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # model = prepare_model(args.pretrained_weights, args.arch, )
    
    model = original_models["MAE"]
    p = model.patch_embed.patch_size
    stride = p if isinstance(p, int) else p[0]
    image = Image.open(args.image_path)
    image = image.convert("RGB")
    image_resized = process_image(image, stride, transforms)
    with torch.no_grad():
        ori_feats = model.get_intermediate_layers(image_resized, n=[8,9,10,11], reshape=True, return_prefix_tokens=False,
                                    return_class_token=False, norm=True)
    

    ori_feats = ori_feats[-1]

    if args.kmeans != -1:
        ori_labels = kmeans_clustering(ori_feats, args.kmeans)
    else:
        ori_labels = None
        
    plot_feats(image, "MAE", ori_feats, None, ori_labels, None, args.output_dir, 0)
    # return
    
    # # build model
    # # model = models_mae.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    # for p in model.parameters():
    #     p.requires_grad = False
    # model.eval()
    # model.to(device)
    
    # # if args.pretrained_weights is not None:     
    # #     if os.path.isfile(args.pretrained_weights):
    # #         state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    # #     if 'model' in state_dict:
    # #             state_dict = state_dict['model']
    # #         msg = model.load_state_dict(state_dict, strict=False)
    # #         print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    # #     else:
    # #         print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    # #         return

    # # open image
    # # if args.image_path is None:
    # #     # user has not specified any image - we use our own image
    # #     print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
    # #     return
    # # elif os.path.isfile(args.image_path):
    # #     with open(args.image_path, 'rb') as f:
    # #         img = Image.open(f)
    # #         img = img.convert('RGB')
    # # else:
    # #     print(f"Provided image path {args.image_path} is non valid.")
    # #     return
    
    # img_path = Path(args.image_path).resolve()
    # img = Image.open(str(img_path))
    # img = img.convert("RGB")
    # img = img.resize((224, 224))
    # img = np.array(img) / 255.

    # assert img.shape == (224, 224, 3)

    # # normalize by ImageNet mean and std
    # imagenet_mean = np.array([0.485, 0.456, 0.406])
    # imagenet_std = np.array([0.229, 0.224, 0.225])
    # img = img - imagenet_mean
    # img = img / imagenet_std
    # # img = torch.tensor(img)
    # img = pth_transforms.ToTensor()(img).unsqueeze(0).float()

    # # # make it a batch-like
    # # x = x.unsqueeze(dim=0)
    # # x = torch.einsum('nhwc->nchw', x)
    # # transform = pth_transforms.Compose([
    # #     pth_transforms.Resize(args.image_size),
    # #     pth_transforms.ToTensor(),
    # #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # # ])
    # # img = transform(img)
    
    # # make the image divisible by the patch size


    # # w_featmap = img.shape[-2] // args.patch_size
    # # h_featmap = img.shape[-1] // args.patch_size
    # # attentions = model.get_last_selfattention(img.to(device))

    # nh = attentions.shape[1] # number of head

    # # we keep only the output patch attention
    # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # if args.threshold is not None:
    #     # we keep only a certain percentage of the mass
    #     val, idx = torch.sort(attentions)
    #     val /= torch.sum(val, dim=1, keepdim=True)
    #     cumval = torch.cumsum(val, dim=1)
    #     th_attn = cumval > (1 - args.threshold)
    #     idx2 = torch.argsort(idx)
    #     for head in range(nh):
    #         th_attn[head] = th_attn[head][idx2[head]]
    #     th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    #     # interpolate
    #     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # attentions = attentions.reshape(nh, w_featmap, h_featmap)
    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]
    # attentions = torch.clamp(attentions, max=attentions.mean()).cpu().numpy()
    
    # # save attentions heatmaps
    # os.makedirs(args.output_dir, exist_ok=True)
    # torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    # for j in range(nh):
    #     fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
    #     plt.imsave(fname=fname, arr=attentions[j], format='png')
    #     print(f"{fname} saved.")

    # if args.threshold is not None:
    #     image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
    #     for j in range(nh):
    #         display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)

#%%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser('MAE Attention Visualization', parents=[get_args_parser()])
    # args = parser.parse_args()
    from argparse import Namespace
    args = Namespace(image_path="/home/mereur1/projects/ocl/ssl_nat_aug/dino/images/1.png", output_dir="mae_attention_vis", kmeans=20)
    main(args)


# %%
