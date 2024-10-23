import os, cv2, numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

def video2frames(mp4_file):
    vidcap = cv2.VideoCapture(mp4_file)
    success,image = vidcap.read()
    frames = []
    while success:
        frames.append(image)   
        success,image = vidcap.read()
    frames = np.array(frames)
    return frames # num_frames, width, height, channels

def read_segmentation_maps(ROOT, mp4_file, width, height):
    seg_map_files = os.listdir(os.path.join(ROOT, 'masks', mp4_file))
    seg_map_files = sorted(seg_map_files)
    seg_maps = []
    for seg in seg_map_files:
        if seg.endswith('.png'):
            seg_map = cv2.imread(os.path.join(ROOT, 'masks', mp4_file, seg))
            assert seg_map.shape[0]==width and seg_map.shape[1]==height
            seg_maps.append(seg_map)
    seg_maps = np.array(seg_maps) # num_frames, width, height, channels)
    return seg_maps

def build_video(ROOT, video_name, device):
    frames = video2frames(os.path.join(ROOT, 'videos', video_name+'.mp4'))
    seg_maps = read_segmentation_maps(ROOT, video_name, frames.shape[1], frames.shape[2])
    if len(seg_maps)!=len(frames):
        return None,None
    else:
        frames = torch.from_numpy(frames).to(device)
        seg_maps = torch.from_numpy(seg_maps).to(device)
        return frames, seg_maps
