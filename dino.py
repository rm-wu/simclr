# list all the folders in a directory
import os, cv2, numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

from video          import Video, compute_masks_per_single_object
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

# load dino stuff
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)

for i,video_name in enumerate(VIDEO_NAMES):
    if i%10==0:
        print(f'{i}/{len(VIDEO_NAMES)}')
    print(video_name)
    if video_name.endswith('.mp4')  and video_name[:-4] not in ' '.join(os.listdir('videos')):
        frames_, seg_maps_ = build_video(ROOT, video_name[:-4], device)
        if frames_ is not None:
            video = Video(video_name[:-4], frames_, seg_maps_, transform)
            try:
                compute_masks_per_single_object(video) # num_good_masks,N,224,224,3
                compute_cos_sims_per_objects(video, vits8)
                visualize_traj(video)
                print(video.S.mean())
                torch.save([video.frames, video.seg_maps, video.masks_per_object, video.embeddings], f'videos/{video.name}.pt')
                del video
            except Exception as e:
                print(f'Skipping {video.name} due to Exception: ', e)
                continue
        else:
            print(f'Skipping {video_name} as no good segmentation maps found')
    # print_gpu_memory()
