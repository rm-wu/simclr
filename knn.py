# list all the folders in a directory
import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

from video          import Video, compute_masks_per_single_object
from io_utils       import build_video
from util_utils     import print_gpu_memory, get_root_folder, apply_transform
from embed_utils    import compute_cos_sims_per_objects
from plot_utils     import visualize_traj

ROOT = get_root_folder()
VIDEO_NAMES = os.listdir(os.path.join(ROOT, 'videos'))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# transform = T.Compose([T.Resize(224),T.CenterCrop(224),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))

# read the embeddings
# saved_videos = os.listdir('videos/')
# all_embeddings = []
# all_labels     = []
# num_objects    = []
# loaded_videos  = []
# last_label     = -1
# for i,video in enumerate(saved_videos):
#     print(i,len(saved_videos))
#     if i==185:
#         continue
#     try:
#         frames, seg_maps, masks_per_object, embeddings = torch.load(f'videos/{video}',map_location=device) # embeddings: [num_good_masks,N_,q]
#         class_labels = (last_label+1+torch.arange(embeddings.shape[0])).unsqueeze(0).repeat(embeddings.shape[1],1).T.flatten()
#         reshaped_embeddings = embeddings.reshape(-1,embeddings.shape[-1])
#         all_embeddings.append(reshaped_embeddings) # num_good_masks*N_,q
#         all_labels.append(class_labels) # num_good_masks*N_
#         last_label += embeddings.shape[0]
#         num_objects.append(embeddings.shape[0])
#         loaded_videos.append(video)
#     except Exception as e:
#         print(video, e)
#         continue

# all_labels = torch.cat(all_labels)
# all_embeddings = torch.cat(all_embeddings)
# num_objects = torch.tensor(num_objects)
# print(all_labels.shape, all_embeddings.shape, num_objects.shape, num_objects.sum())
# torch.save([all_labels, all_embeddings, num_objects, loaded_videos], 'videos_embeddings.pt')

all_labels, all_embeddings, num_objects, loaded_videos = torch.load('videos_embeddings.pt')

# compute nearest neighbors
similarity = torch.zeros(all_embeddings.shape[0],all_embeddings.shape[0])
for i in range(0,all_embeddings.shape[0],1000):
    for j in range(0,all_embeddings.shape[0],1000):
        similarity[i:i+1000,j:j+1000] = (all_embeddings[i:i+1000].unsqueeze(1) * all_embeddings[j:j+1000]).sum(-1)

for i in range(all_embeddings.shape[0]):
    similarity[i,i] = -1e0

_,nearest_neighbors = similarity.topk(5,dim=1)
retrieval_rate = torch.stack([all_labels[i]==all_labels[nearest_neighbors[i,0]] for i in range(similarity.shape[0])]).to(torch.float32).mean() # check if the labels are the same
print(retrieval_rate.item())

misclassified_idx = [i for i in range(similarity.shape[0]) if all_labels[i]!=all_labels[nearest_neighbors[i,0]]]

transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))
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

def get_frame(anchor_idx):
    video_id_anchor   = torch.where(num_objects.cumsum(0)>(anchor_idx//20))[0][0].item()
    seg_map_id_anchor = anchor_idx//20 - num_objects.cumsum(0)[video_id_anchor-1].item() if anchor_idx//20>=num_objects[0] else anchor_idx//20
    frame_id_anchor   = anchor_idx%20
    video_name = loaded_videos[video_id_anchor]
    print(video_id_anchor, seg_map_id_anchor, frame_id_anchor)
    frames, seg_maps, masks_per_object, embeddings = torch.load(f'videos/{video_name}',map_location=device)
    video = Video(None, frames, seg_maps, transform)
    video.masks_per_object = masks_per_object
    frame = video.transformed_seg_cropped_imgs[seg_map_id_anchor][frame_id_anchor]
    del video
    return apply_transform(frame,imagenet_reverse_transform).cpu(), video_id_anchor, seg_map_id_anchor, frame_id_anchor

for j in range(50):
    if f'{j}.png' in os.listdir('errors'):
        continue
    fig,ax = plt.subplots(6,10,figsize=(30,18),squeeze=False)
    for i in range(j*10,(j+1)*10):
        anchor_idx = misclassified_idx[i]
        anchor_frame, video_id, seg_map_id, frame_id = get_frame(anchor_idx)
        if video_id==194 and seg_map_id==6 and frame_id==2:
            continue
        ax[0,i%10].imshow(anchor_frame);
        ax[0,i%10].set_title(f'Anchor (video {video_id}, seg_map {seg_map_id}, frame {frame_id*20})');
        ax[0,i%10].axis('off');
        for j in range(5):
            nn_frame, video_id, seg_map_id, frame_id = get_frame(nearest_neighbors[anchor_idx,j].item())
            if video_id==194 and seg_map_id==6 and frame_id==2:
                continue
            ax[j+1,i%10].imshow(nn_frame);
            ax[j+1,i%10].set_title(f'NN (video {video_id}, seg_map {seg_map_id}, frame {frame_id*20})');
            ax[j+1,i%10].axis('off');
    plt.tight_layout()
    plt.show()
    plt.savefig(f'errors/{j}.png',dpi=200)
    plt.close()