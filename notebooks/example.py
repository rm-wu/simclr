# %%
# list all the folders in a directory
import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

ROOT = '/home/mereur1/projects/ocl/data/PVSG_dataset/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor'
video_names = os.listdir(os.path.join(ROOT, 'videos'))

def video2frames(mp4_file):
    vidcap = cv2.VideoCapture(mp4_file)
    success,image = vidcap.read()
    frames = []
    while success:
        frames.append(Image.fromarray(image))   
        success,image = vidcap.read()
    return frames # num_frames, [width, height, channels]

def read_segmentation_maps(mp4_file, width, height):
    seg_map_files = os.listdir(os.path.join(ROOT, 'masks', mp4_file))
    seg_map_files = sorted(seg_map_files)
    seg_maps = []
    for seg in seg_map_files:
        if seg.endswith('.png'):
            seg_map = cv2.imread(os.path.join(ROOT, 'masks', mp4_file, seg))
            assert seg_map.shape[0]==width and seg_map.shape[1]==height
            seg_maps.append(seg_map)
    seg_maps = np.array(seg_maps) # num_frames, width, height, channels
    return seg_maps

def build_video(video_name):
    frames = video2frames(os.path.join(ROOT, 'videos', video_name+'.mp4'))
    seg_maps = read_segmentation_maps(video_name, *frames[0].size[::-1])
    if len(seg_maps)!=len(frames):
        return None,None
    else:
        return frames, seg_maps


class VideoDataset(Dataset):
    def __init__(self, video_names, num_videos=10, positive_range=10, transform=None):
        self.frames = []
        self.seg_maps = []
        self.transform = transform
        count = 0
        self.range = positive_range
        for video_name in video_names:
            if count>=num_videos:
                break
            if video_name.endswith('.mp4'):
                frames, seg_maps = build_video(video_name[:-4])
                if frames is not None:
                    print(video_name, len(frames))
                    self.frames.append(frames)
                    self.seg_maps.append(seg_maps)
                    count += 1
                else:
                    print(video_name, 'number of frames and segmentation maps do not match:')
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames, seg_maps = self.frames[idx],self.seg_maps[idx]
        len_video = len(frames)
        while True:
            first_idx  = np.random.randint(0, len_video-2*self.range) + self.range
            second_idx = np.random.randint(first_idx-self.range, first_idx+self.range)
            if 0<=second_idx<len_video:
                break
        if self.transform:
            first_frame  = self.transform(frames[first_idx]), self.transform(seg_maps[first_idx])
            second_frame = self.transform(frames[second_idx]),self.transform(seg_maps[second_idx])
        else:
            first_frame  = frames[first_idx], seg_maps[first_idx]
            second_frame = frames[second_idx], seg_maps[second_idx]
        return first_frame, second_frame

transform = T.Compose(
    [
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((480,640)),
    ]
)        

video_dataset = VideoDataset(video_names, 5, transform=transform)
data_loader = torch.utils.data.DataLoader(video_dataset, batch_size=1, shuffle=True)


# %%
figsize = (12, 6)
sample_1, sample_2 = next(iter(data_loader))

fig, ax = plt.subplots(1, 2, figsize=figsize)

ax[0].imshow(sample_1[0][0].permute(1,2,0))
ax[1].imshow(sample_2[0][0].permute(1,2,0))
ax[0].axis('off')
ax[1].axis('off')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=figsize)
ax[0].imshow(sample_1[1][0].permute(1,2,0))
ax[1].imshow(sample_2[1][0].permute(1,2,0))
ax[0].axis('off')
ax[1].axis('off')
plt.show()


# %%
