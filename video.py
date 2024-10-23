import os, cv2, numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

from util_utils import apply_transform

def make_square_tensor(image_tensor, fill_values=(-0.485/0.229, -0.456/0.224, -0.406/0.225)):
    """
    Takes a non-square tensor image of shape [w, h, c] and returns a square tensor with the image centered.

    Parameters:
        image_tensor (torch.Tensor): Input tensor image of shape [w, h, c].
        fill_value (int): The value to fill the padding with (default is 0, i.e., black).

    Returns:
        torch.Tensor: A square tensor image with the original image centered.
    """
    # Get original dimensions
    original_height, original_width, channels = image_tensor.shape
    # Calculate the size of the square
    max_dim = max(14,max(original_width, original_height))
    # Calculate padding for each side
    pad_left = (max_dim - original_width) // 2
    pad_right = max_dim - original_width - pad_left
    pad_top = (max_dim - original_height) // 2
    pad_bottom = max_dim - original_height - pad_top
    # Pad the image using F.pad
    padded_channels = []
    for channel in range(channels):
        channel_tensor = image_tensor[:, :, channel]
        padded_channel = F.pad(channel_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=fill_values[channel])
        padded_channels.append(padded_channel)
    padded_image = torch.stack(padded_channels, dim=0)
    # padding = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)
    # padded_image = F.pad(image_tensor.permute(2, 0, 1), padding, mode='constant', value=fill_value)
    # Permute back to [w, h, c] format
    padded_image = padded_image.permute(1, 2, 0)
    return padded_image


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
        if self.embeddings is None:
            return None
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
        for masks,objects in zip(self.masks_per_object,transformed_seg_imgs):
            transformed_seg_cropped_imgs_ = []
            for mask,frame in zip(masks,objects):
                nonzero_rows = torch.where(mask.sum(1).abs()>1e-5)[0]
                nonzero_cols = torch.where(mask.sum(0).abs()>1e-5)[0]
                try:
                    up,down = nonzero_rows.min(),nonzero_rows.max()
                    assert down-up>14
                except:
                    up,down = 0,14
                try:
                    left,right = nonzero_cols.min(),nonzero_cols.max()
                    assert right-left>14
                except:
                    left,right = 0,14
                cropped_image = frame[up:down,left:right]
                cropped_image = make_square_tensor(cropped_image)
                transformed_seg_cropped_imgs_.append(cropped_image)
            transformed_seg_cropped_imgs.append(transformed_seg_cropped_imgs_)
        return transformed_seg_cropped_imgs


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
