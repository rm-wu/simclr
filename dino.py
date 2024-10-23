# list all the folders in a directory
import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

class Video:
    def __init__(self, name, frames, seg_maps, transform):
        self.name = name
        self.frames = frames # num_frames, width, height, channels, int between [0,255]
        self.seg_maps = seg_maps # num_frames, width, height, channels
        self.masks_per_object = None # will be of shape [num_good_masks,W,H]
        self.embeddings = None # embeddings of shape [num_good_masks,N_,q] where N_ is a user input
        self.transform = transform
    @property
    def Nf(self):
        return len(self.frames)
    @property
    def S(self):
        ''' cos sim of the same obj across diff frames.  of shape [num_good_masks,N_,N_] where N_ is a user input'''
        return (self.embeddings.unsqueeze(1) * self.embeddings.unsqueeze(2)).sum(-1)
    @property
    def transformed_seg_imgs(self):
        if self.masks_per_object is None:
            raise ValueError('No masks found')
        transformed_seg_imgs = []
        for masks in self.masks_per_object:
            seg_imgs = self.frames.to(torch.float32) / 255 * masks[:,:,:,None]
            transformed_seg_imgs_ = apply_transform(seg_imgs, self.transform) # [N,W,H,3] or [W,H,3]
            transformed_seg_imgs.append(transformed_seg_imgs_)
        transformed_seg_imgs = torch.stack(transformed_seg_imgs) # num_good_masks,N,W,H,3
        return transformed_seg_imgs
    @property
    def transformed_seg_cropped_imgs(self):
        transformed_seg_imgs = self.transformed_seg_imgs.clone()
        transformed_seg_cropped_imgs = []
        for objects in transformed_seg_imgs:
            transformed_seg_cropped_imgs_ = []
            for frame in objects:
                nonzero_rows = torch.where(frame.sum(1)!=0)[0]
                nonzero_cols = torch.where(frame.sum(0)!=0)[0]
                up,down = nonzero_rows.min(),nonzero_rows.max()
                left,right = nonzero_cols.min(),nonzero_cols.max()
                transformed_seg_cropped_imgs_.append(frame[up:down,left:right])
            transformed_seg_cropped_imgs.append(transformed_seg_cropped_imgs_)
        return transformed_seg_cropped_imgs

def print_gpu_memory():
    try:
        # print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        # print(f"Memory Reserved (Cached): {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
        print(f"Free GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2 - torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
        # print("--------------------")
    except:
        pass

def get_root_folder():
    if 'cagatay' in os.getcwd():
        return '/Users/cagatay/Downloads/VidOR'
    else:
        if os.path.exists('/weka'):
            return '/home/bethge/cyildiz40/data/VidOR'
        else:
            return '/mnt/lustre/work/bethge/cyildiz40/projects/videossl/VidOR'

def video2frames(mp4_file):
    vidcap = cv2.VideoCapture(mp4_file)
    success,image = vidcap.read()
    frames = []
    while success:
        frames.append(image)   
        success,image = vidcap.read()
    frames = np.array(frames)
    return frames # num_frames, width, height, channels

def read_segmentation_maps(mp4_file, width, height):
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

def build_video(video_name):
    frames = video2frames(os.path.join(ROOT, 'videos', video_name+'.mp4'))
    seg_maps = read_segmentation_maps(video_name, frames.shape[1], frames.shape[2])
    if len(seg_maps)!=len(frames):
        return None,None
    else:
        frames = torch.from_numpy(frames).to(device)
        seg_maps = torch.from_numpy(seg_maps).to(device)
        return frames, seg_maps

def read_data(num_videos=5, transform=None):
    videos = []
    for video_name in VIDEO_NAMES:
        if video_name.endswith('.mp4'):
            frames_, seg_maps_ = build_video(video_name[:-4])
            if frames_ is not None:
                video = Video(video_name[:-4], frames_, seg_maps_, transform)
                # print(video_name, len(frames_))
                videos.append(video)
            else:
                print(video_name, 'number of frames and segmentation maps do not match:')
        print_gpu_memory()
        if len(videos)>=num_videos:
            break
    return videos

def apply_transform(frames, transform):
    ''' frames could be channel first or last'''
    # assert frames.shape[-3]==3, 'channel dimension wrong'
    if frames.shape[-3]!=3:
        if frames.ndim==3:
            frames = frames.permute(2,0,1) # 3, width, height
            transformed_frames = transform(frames).permute(1,2,0) # 224, 224, 3
        elif frames.ndim==4:
            frames = frames.permute(0,3,1,2) # N, 3, width, height
            transformed_frames = transform(frames).permute(0,2,3,1) # N, 224, 224, 3
        elif frames.ndim==5:
            frames = frames.permute(0,1,4,2,3) # N, T, 3, width, height
            transformed_frames = transform(frames).permute(0,1,3,4,2) # N, T, 224, 224, 3
    else:
        transformed_frames = transform(frames)
    return transformed_frames # [N,T,224,224,3] or [N,224,224,3]  or [224,224,3]

def get_unique_colors(seg_maps):
    # get unique segmentation colors
    len_frames = len(seg_maps)
    unique_colors = torch.zeros(0,3).to(seg_maps.device).to(seg_maps.dtype)
    rand_idx = torch.randperm(len_frames)[:10]
    for idx in rand_idx:
        unique_colors = torch.cat([unique_colors, seg_maps[idx].reshape(-1,3)], dim=0)
    unique_colors = unique_colors.unique(dim=0)
    return unique_colors

def compute_masks_per_single_object(video, mean_thr=0.05, min_thr=0.02):
    seg_maps = video.seg_maps
    unique_colors = get_unique_colors(seg_maps)
    masks_per_object = []
    for color in unique_colors:
        masks = torch.stack([(m == color).all(dim=-1) for m in seg_maps]) # num_frames, width, height
        means_across_frames = masks.float().mean(-1).mean(-1)
        fraction_of_frames_wo_object = (means_across_frames<min_thr).float().mean()
        if means_across_frames.mean()>mean_thr: # if object is large enough
            if fraction_of_frames_wo_object<0.25: # if object is present in at least 75% of the frames
                masks_per_object.append(masks)
    if len(masks_per_object)>0:
        video.masks_per_object = torch.stack(masks_per_object) # num_good_masks,N,W,H
    else:
        raise ValueError('No good masks found')

def compute_embeddings(frames, network, normalize=True):
    assert frames.ndim==4, 'frames should have 4 dimensions'
    if frames.shape[-3] != 3:
        frames = frames.permute(0,3,1,2)
    with torch.no_grad():
        embeddings = network(frames)
        if normalize:
            embeddings = embeddings / embeddings.norm(2,-1,keepdim=True)
    return embeddings

def compute_cos_sims_per_objects(video, network, N_=20):
    Nf = video.Nf
    video.S_idx = range(0,(Nf//N_)*N_, (Nf//N_))
    embeddings = []
    for seg in video.transformed_seg_cropped_imgs:
        embeddings_ = compute_embeddings(seg[video.S_idx], network)
        embeddings.append(embeddings_) 
    video.embeddings = torch.stack(embeddings) # num_good_masks,N_,N_

def visualize_traj(video, Nmax=5, imagenet_reverse_transform=None):
    ''' video - [T,W,H,C] '''
    frames,transformed_seg_imgs,S,S_idx = video.frames, video.transformed_seg_cropped_imgs, video.S, video.S_idx
    if S is None:
        print(f'No similarity matrix computed, skipping video {video.name}')
    if imagenet_reverse_transform is None:
        imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
    N_,video_len = transformed_seg_imgs.shape[:2]
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
            ax[i,j].imshow(apply_transform(transformed_seg_imgs[i-1,S_idx[j]],imagenet_reverse_transform).cpu())
            # ax[i,j].imshow(imagenet_reverse_transform(transformed_seg_imgs[i-1,j*every].permute(2,0,1)).permute(1,2,0))
            ax[i,j].axis('off')
        if S is not None:
            img_ = ax[i,-1].imshow(video.S[i-1].cpu())
            fig.colorbar(img_, ax=ax[i,-1])
            ax[i,-1].axis('off')
            ax[i,-1].axis('off')
    plt.tight_layout()
    plt.savefig(f'figs/{video.name}.png',dpi=200)

ROOT = get_root_folder()
VIDEO_NAMES = os.listdir(os.path.join(ROOT, 'videos'))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# transform = T.Compose([T.Resize(224),T.CenterCrop(224),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
imagenet_reverse_transoform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))

# load dino stuff
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device)

for i,video_name in enumerate(VIDEO_NAMES):
    if i%10==0:
        print(f'{i}/{len(VIDEO_NAMES)}')
    print(video_name)
    if video_name.endswith('.mp4') and video_name[:-4] not in ' '.join(os.listdir('videos')):
        frames_, seg_maps_ = build_video(video_name[:-4])
        if frames_ is not None:
            video = Video(video_name[:-4], frames_, seg_maps_, transform)
            try:
                compute_masks_per_single_object(video) # num_good_masks,N,224,224,3
                compute_cos_sims_per_objects(video, vits8)
                visualize_traj(video)
                print(video.S.mean())
                torch.save([video.frames, video.seg_maps, video.masks_per_object, video.embeddings], f'videos/{video.name}.pt')
                del video
            except:
                print(f'Skipping {video.name} as no good segmentation maps found')
                continue
        else:
            print(f'Skipping {video_name} as no good segmentation maps found')
    # print_gpu_memory()
