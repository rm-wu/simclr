import os, cv2, numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

from util_utils import apply_transform

def visualize_traj(video, Nmax=5, imagenet_reverse_transform=None):
    ''' video - [T,W,H,C] '''
    frames,transformed_seg_imgs,S,S_idx = video.frames, video.transformed_seg_cropped_imgs, video.S, video.S_idx
    if S is None:
        print(f'No similarity matrix computed, skipping video {video.name}')
    if imagenet_reverse_transform is None:
        imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
    N_ = len(transformed_seg_imgs)
    video_len = len(transformed_seg_imgs[0])
    T_ = 20 if S is None else S.shape[1] 
    S_idx = range(0,(video_len//20)*20,video_len//20) if S_idx is None else S_idx
    N_ = min(N_,Nmax)
    fig,ax = plt.subplots(N_+1, T_+1, figsize=(3*(T_+1),3*(N_+1)))
    # plot the video first
    for j in range(T_):
        ax[0,j].imshow(frames[S_idx[j]].cpu())
        ax[0,j].axis('off')
        ax[0,j].set_title('Frame {:d}'.format(S_idx[j]))
    for i in range(1,N_+1):
        for j in range(T_):
            ax[i,j].imshow(apply_transform(transformed_seg_imgs[i-1][S_idx[j]],imagenet_reverse_transform).cpu())
            # ax[i,j].imshow(imagenet_reverse_transform(transformed_seg_imgs[i-1,j*every].permute(2,0,1)).permute(1,2,0))
            ax[i,j].axis('off')
        if S is not None:
            img_ = ax[i,-1].imshow(video.S[i-1].cpu())
            fig.colorbar(img_, ax=ax[i,-1])
            ax[i,-1].axis('off')
            ax[i,-1].axis('off')
    plt.tight_layout()
    plt.savefig(f'figs/{video.name}.png',dpi=200)
