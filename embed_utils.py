import torch
from util_utils import get_device
import torch.nn.functional as F
import math

class PadToMultipleOf:
    def __init__(self, multiple):
        self.multiple = multiple
    def __call__(self, image):
        # Ensure the input is a torch tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch tensor")
        # Check if image has a channel dimension (e.g., CxHxW or HxW)
        if image.dim() == 3:
            c, h, w = image.shape
        elif image.dim() == 2:
            h, w = image.shape
            c = 1  # Single channel grayscale assumed
            image = image.unsqueeze(0)  # Add channel dimension
        else:
            raise ValueError("Unsupported image dimensions")
        # Calculate the nearest multiple of 14 that is greater than the current size
        new_size = math.ceil(max(h, w) / self.multiple) * self.multiple
        # Calculate padding needed on each side
        pad_left = (new_size - w) // 2
        pad_top = (new_size - h) // 2
        pad_right = new_size - w - pad_left
        pad_bottom = new_size - h - pad_top
        # Apply padding with black (0) pixels
        padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        return padded_image

def compute_embeddings(frames, network, patchwise=False, normalize=True):
    device = get_device(network)
    if isinstance(frames,list):
        assert len(frames[0].shape)==3, 'frames should have 4 dims'
        if frames[0].shape[-3] != 3:
            frames = [frame.permute(2,0,1) for frame in frames]
        frames = [frame.to(device) for frame in frames]
        with torch.no_grad():
            # print([f.size() for f in frames])
            if patchwise:
                embeddings = [network.get_intermediate_layers(frame.unsqueeze(0), n=[11], reshape=True, return_prefix_tokens=False, norm=True)[0] for frame in frames]
            else:
                embeddings = [network(frame.unsqueeze(0)) for frame in frames]
        embeddings = torch.cat(embeddings)
    else:
        assert frames.ndim==4, 'frames should have 4 dimensions'
        if frames.shape[-3] != 3:
            frames = frames.permute(0,3,1,2).to(device)
        with torch.no_grad():
            if patchwise:
                embeddings = network.get_intermediate_layers(frames, n=[11], reshape=True, return_prefix_tokens=False, norm=True)[0]
            else:
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
