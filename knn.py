# list all the folders in a direc§:tory
import os, cv2, numpy as np

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

from video          import Video, compute_masks_per_single_object
from io_utils       import build_video
from util_utils     import print_gpu_memory, get_root_folder, apply_transform
from embed_utils    import compute_cos_sims_per_objects
from plot_utils     import visualize_traj

# def compute_knn(all_embeddings, all_labels, normalize=True):
#     if normalize:
#         all_embeddings = F.normalize(all_embeddings,dim=-1)
#     # compute nearest neighbors
#     similarity = torch.zeros(all_embeddings.shape[0],all_embeddings.shape[0])
#     for i in range(0,all_embeddings.shape[0],1000):
#         for j in range(0,all_embeddings.shape[0],1000):
#             similarity[i:i+1000,j:j+1000] = (all_embeddings[i:i+1000].unsqueeze(1) * all_embeddings[j:j+1000]).sum(-1)

#     for i in range(all_embeddings.shape[0]):
#         similarity[i,i] = -1e0

#     _,nearest_neighbors = similarity.topk(5,dim=1)
#     retrieval_rate = torch.stack([all_labels[i]==all_labels[nearest_neighbors[i,0]] for i in range(similarity.shape[0])]).to(torch.float32).mean() # check if the labels are the same
#     misclassified_idx = [i for i in range(similarity.shape[0]) if all_labels[i]!=all_labels[nearest_neighbors[i,0]]]
    # return retrieval_rate, misclassified_idx, nearest_neighbors

def compute_knn(all_embeddings, all_labels, batch_size=256, normalize=True, data_portion=1.0):
    if normalize:
        all_embeddings = F.normalize(all_embeddings, dim=-1)
    
    num_embeddings = all_embeddings.shape[0]
    retrieval_set_size = int(num_embeddings * data_portion)
    
    nearest_neighbors_list = []
    retrieval_rates = []
    misclassified_indices = []

    for start in range(0, num_embeddings, batch_size):
        end = min(start + batch_size, num_embeddings)
        current_embeddings = all_embeddings[start:end] # batch_size,q
        similarity = torch.zeros(current_embeddings.shape[0], num_embeddings, device=all_embeddings.device)
        if data_portion==1.0:
            retrieval_set_idx = torch.arange(num_embeddings).to(all_embeddings.device)
        else:
            retrieval_set_idx = torch.randperm(num_embeddings)[:retrieval_set_size].to(all_embeddings.device)
        retrieval_set = all_embeddings[retrieval_set_idx]

        # Compute similarity for current batchprint(start)
        for i in range(0, retrieval_set_size, batch_size):
            last_idx = min(i + batch_size, retrieval_set_size)
            similarity[:, i:last_idx] = (current_embeddings.unsqueeze(1) * retrieval_set[i:last_idx].unsqueeze(0)).sum(-1)

        # Set diagonal elements to a large negative value for the current batch
        if data_portion >= 1.0:
            for i in range(end - start):
                similarity[i, start + i] = -1e0

        # Get top-5 nearest neighbors for the current batch
        _, nearest_neighbors = similarity.topk(5, dim=1)
        nearest_neighbors = retrieval_set_idx[nearest_neighbors]
        nearest_neighbors_list.append(nearest_neighbors)

        # Compute retrieval rate for the current batch
        retrieval_rate_batch = torch.stack([
            all_labels[start + i] == all_labels[nearest_neighbors[i, 0]]
            for i in range(end - start)
        ]).to(torch.float32).mean()
        retrieval_rates.append(retrieval_rate_batch.item())

        # Track misclassified indices for the current batch
        misclassified_idx_batch = [
            start + i for i in range(end - start)
            if all_labels[start + i] != all_labels[nearest_neighbors[i, 0]]
        ]
        misclassified_indices.extend(misclassified_idx_batch)

    # Combine all metrics
    overall_retrieval_rate = sum(retrieval_rates) / len(retrieval_rates)
    nearest_neighbors_combined = torch.cat(nearest_neighbors_list, dim=0)

    return overall_retrieval_rate, misclassified_indices, nearest_neighbors_combined

def main():
    VIDEO_FOLDER = 'videos_mae'
    VIDEO_NAMES  = os.listdir(VIDEO_FOLDER)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # transform = T.Compose([T.Resize(224),T.CenterCrop(224),T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    imagenet_reverse_transform = T.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1/0.229, 1/0.224, 1/0.225))

    # :read the embeddings
    saved_videos = os.listdir(VIDEO_FOLDER)
    all_embeddings = []
    all_labels     = []
    num_objects    = []
    loaded_videos  = []
    last_label     = -1
    for i,video in enumerate(saved_videos):
        print(i,len(saved_videos))
        if i==185:
            continue
        try:
            frames, seg_maps, masks_per_object, embeddings = torch.load(f'{VIDEO_FOLDER}/{video}',map_location=device) # embeddings: [num_good_masks,N_,q]
            class_labels = (last_label+1+torch.arange(embeddings.shape[0])).unsqueeze(0).repeat(embeddings.shape[1],1).T.flatten()
            reshaped_embeddings = embeddings.reshape(-1,embeddings.shape[-1])
            all_embeddings.append(reshaped_embeddings) # num_good_masks*N_,q
            all_labels.append(class_labels) # num_good_masks*N_
            last_label += embeddings.shape[0]
            num_objects.append(embeddings.shape[0])
            loaded_videos.append(video)
        except Exception as e:
            print(video, e)
            continue

    all_labels = torch.cat(all_labels)
    all_embeddings = torch.cat(all_embeddings)
    num_objects = torch.tensor(num_objects)
    print(all_labels.shape, all_embeddings.shape, num_objects.shape, num_objects.sum())
    torch.save([all_labels, all_embeddings, num_objects, loaded_videos], f'{VIDEO_FOLDER}/videos_embeddings.pt')

    # all_labels, all_embeddings, num_objects, loaded_videos = torch.load(f'{VIDEO_FOLDER}/videos_embeddings.pt')

    retrieval_rate, misclassified_idx, nearest_neighbors = compute_knn(all_embeddings, all_labels, normalize=True)

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
        frames, seg_maps, masks_per_object, embeddings = torch.load(f'{VIDEO_FOLDER}/{video_name}',map_location=device)
        video = Video(None, frames, seg_maps, transform)
        video.masks_per_object = masks_per_object
        frame = video.transformed_seg_cropped_imgs[seg_map_id_anchor][frame_id_anchor]
        del video
        return apply_transform(frame,imagenet_reverse_transform).cpu(), video_id_anchor, seg_map_id_anchor, frame_id_anchor

    for j in range(50):
        if f'{j}.png' in os.listdir(f'{VIDEO_FOLDER}/errors'):
            continue
        fig,ax = plt.subplots(6,10,figsize=(30,18),squeeze=False)
        for i in range(j*10,(j+1)*10):
            anchor_idx = misclassified_idx[i]
            anchor_frame, video_id, seg_map_id, frame_id = get_frame(anchor_idx)
            if video_id==194 and seg_map_id==6 and frame_id==2:
                continue
            ax[0,i%10].imshow(anchor_frame);
            ax[0,i%10].set_title(f'Anchor (video {video_id}, map {seg_map_id}, fr. {frame_id*20})');
            ax[0,i%10].axis('off');
            for k in range(5):
                nn_frame, video_id, seg_map_id, frame_id = get_frame(nearest_neighbors[anchor_idx,k].item())
                if video_id==194 and seg_map_id==6 and frame_id==2:
                    continue
                ax[k+1,i%10].imshow(nn_frame);
                ax[k+1,i%10].set_title(f'NN (video {video_id}, map {seg_map_id}, fr. {frame_id*20})');
                ax[k+1,i%10].axis('off');
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{VIDEO_FOLDER}/errors/{j}.png',dpi=200)
        plt.close()

if __name__ == '__main__':
    main()
