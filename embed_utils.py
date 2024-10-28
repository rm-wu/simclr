import torch
from util_utils import get_device

def compute_embeddings(frames, network, normalize=True):
    device = get_device(network)
    if isinstance(frames,list):
        assert len(frames[0].shape)==3, 'frames should have 4 dims'
        if frames[0].shape[-3] != 3:
            frames = [frame.permute(2,0,1) for frame in frames]
        frames = [frame.to(device) for frame in frames]
        with torch.no_grad():
            # print([f.size() for f in frames])
            embeddings = [network(frame.unsqueeze(0)) for frame in frames]
        embeddings = torch.cat(embeddings)
    else:
        assert frames.ndim==4, 'frames should have 4 dimensions'
        if frames.shape[-3] != 3:
            frames = frames.permute(0,3,1,2).to(device)
        with torch.no_grad():
            embeddings = network(frames)
    if normalize:
        embeddings = embeddings / embeddings.norm(2,-1,keepdim=True)
    return embeddings

def compute_cos_sims_per_objects(video, network, N_=20):
    Nf = video.Nf
    video.S_idx = range(0,(Nf//N_)*N_, (Nf//N_))
    # video.embeddings = torch.randn(video.masks_per_object.shape[0], N_, 384).to(next(network.parameters()))
    # video.embeddings = video.embeddings / video.embeddings.norm(2,-1,keepdim=True)
    embeddings = []
    for seg in video.transformed_seg_cropped_imgs:
        embeddings_ = compute_embeddings([seg[i] for i in video.S_idx], network)
        embeddings.append(embeddings_) 
    video.embeddings = torch.stack(embeddings) # num_good_masks,N_,N_
