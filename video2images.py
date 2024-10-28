
import os, cv2, numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

from ssl_libs.load_model import load_model, compute_features
from video          import Video, compute_masks_per_single_object, get_unique_colors
from io_utils       import build_video
from util_utils     import print_gpu_memory, get_root_folder, apply_transform
from embed_utils    import compute_cos_sims_per_objects
from plot_utils     import visualize_traj

ROOT        = get_root_folder()
VIDEO_NAMES = os.listdir(os.path.join(ROOT, 'videos'))
device      = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# transform = T.Compose([T.Resize(224),T.CenterCrop(224),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
transform   = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))


def video_2_image(video, ROOT, mean_thr=0.05):
    seg_maps = video.seg_maps
    unique_colors = get_unique_colors(seg_maps)
    masks_per_object = []
    for color in unique_colors:
        masks = torch.stack([(m == color).all(dim=-1) for m in seg_maps]) # num_frames, width, height
        means_across_frames = masks.float().mean(-1).mean(-1)
        if means_across_frames.mean()>mean_thr: # if object is large enough
            masks_per_object.append(masks)
    if len(masks_per_object)==0:
        raise ValueError('No good masks found')
    video.masks_per_object = torch.stack(masks_per_object) # num_good_masks,N,W,H
    transformed_seg_cropped_imgs = video.transformed_seg_cropped_imgs
    video.masks_per_object = video.masks_per_object.to(torch.float16)
    # save the video
    from pathlib import Path
    from torchvision.utils import save_image
    Path(f"{ROOT}/imgs/{video.name}").mkdir(parents=True, exist_ok=True)
    print(video.name, 'num_sequences:', len(transformed_seg_cropped_imgs), f'{ROOT}/imgs/{video.name}')
    for i,(imgs,masks) in enumerate(zip(transformed_seg_cropped_imgs,video.masks_per_object)):
        for j,(img,mask) in enumerate(zip(imgs,masks)):
            if mask.mean([-1,-2]) > 1e-3:
                save_image(img.permute(2,0,1).cpu(), f"{ROOT}/imgs/{video.name}/{i}_{j}.JPEG")


for i,video_name in enumerate(VIDEO_NAMES):
    if i%10==0:
        print(f'{i}/{len(VIDEO_NAMES)}')
    if os.path.exists(f'{ROOT}/imgs/{video_name[:-4]}'):
        continue
    frames_, seg_maps_ = build_video(ROOT, video_name[:-4], device)
    video = Video(video_name[:-4], frames_, seg_maps_, transform=None)
    compute_masks_per_single_object(video)
    video_2_image(video, ROOT, mean_thr=0.01)
    del video
