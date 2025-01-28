import random
from PIL import Image

# Custom function to apply resizing and cropping with probability
import torchvision.transforms.functional as F
from torchvision import transforms
import torchvision.transforms as T
import torch
import numpy as np
from PIL import ImageFilter


def random_resize_crop(
    image, target, size=(256, 256), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=scale, ratio=ratio
    )
    image = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
    target = F.resized_crop(target, i, j, h, w, size, interpolation=Image.NEAREST)
    return image, target


def resize(image, target, size=(256, 256)):
    ## convert target to tensor
    # TODO: I commented it 
    # if not isinstance(target, torch.Tensor):
    #     target = transforms.ToTensor()(target)
    image = F.resize(image, size, interpolation=Image.BILINEAR)
    target = F.resize(target, size, interpolation=Image.NEAREST)
    return image, target


def apply_horizontal_flip(image, target):
    # Generate a random seed for the transformation
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    seed = torch.randint(0, 2**32, size=(1,)).item()
    torch.manual_seed(seed)

    # Apply horizontal flip to the image
    image = F.hflip(image)

    # Use the same seed for the target to ensure consistent flip
    torch.manual_seed(seed)
    target = F.hflip(target)

    return image, target


# class RandomResizedCrop(object):
#     def __init__(
#         self, size, scale=(0.5, 2), ratio=(3.0 / 4.0, 4.0 / 3.0), probability=1.0
#     ):
#         self.size = size
#         self.scale = scale
#         self.ratio = ratio
#         self.probability = probability

#     def __call__(self, img, target):
#         if random.random() < self.probability:
#             return random_resize_crop(img, target, self.size, self.scale, self.ratio)
#         return img, target
class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.5, 2), ratio=(3.0 / 4.0, 4.0 / 3.0), p=1.0):
        self.rrc_transform = T.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
        self.p = p

    def __call__(self, img, target=None):
        if random.random() < self.p:
            y1, x1, h, w = self.rrc_transform.get_params(
                img, self.rrc_transform.scale, self.rrc_transform.ratio
            )
            img = F.resized_crop(
                img, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.BILINEAR
            )
            target = F.resized_crop(
                target, y1, x1, h, w, self.rrc_transform.size, F.InterpolationMode.NEAREST
            )
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return apply_horizontal_flip(img, target)
        return img, target



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class CombTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None, img_tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform
        self.img_tgt_transform = img_tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        if self.img_tgt_transform:
            return self.img_tgt_transform(img, tgt)
        else:
            return img, tgt

class SepTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        return img, tgt


class ToTensor(object):
    def __call__(self, img, target):
        if not isinstance(target, torch.Tensor):
            target = transforms.ToTensor()(target)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        return img, target
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target