from typing import Dict, List, Optional, Sequence, Tuple, Union

from torchvision import transforms as T
from torch import Tensor
from PIL.Image import Image

from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


class DINONaturalTransform(DINOTransform):
    """Extends the global and local view augmentations of DINO [0] to work with 
    natural augmentaitons.

    Input to this transform:
        A pair of PIL Images or Tensors.

    Output of this transform:
        List of Tensor of length 2 * global + n_local_views. (8 by default)

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - Random solarization
        - ImageNet normalization

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
        apply_local_transform_to_both (bool):
            apply the same transform to both images in the pair (anchor, nat_positive)
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
        apply_local_transform_to_both: bool = False
    ):
        super().__init__(
            global_crop_size=global_crop_size,
            global_crop_scale=global_crop_scale,
            local_crop_size=local_crop_size,
            local_crop_scale=local_crop_scale,
            n_local_views=n_local_views,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=solarization_prob,
            normalize=normalize,
        )
        self.apply_local_transforms_to_both = apply_local_transform_to_both
    
    
    def __call__(self, images: Union[List[Tensor], List[Image]]) -> Union[List[Tensor], List[Image]]:
        """Transforms the pair (anchor, nat_positive) into multiple views.
        Every transform in self.transforms creates a new view.

        Args: 
            images (list):
                (anchor, nat_posive) images to be transformed
        
        Returns:
            List of views.
        """
        if len(images) != 2:
            raise ValueError(f"The transform works only with two images")
        
        views = [
            self.transforms[0](images[0]),  # global aug. anchor img
            self.transforms[1](images[1]),  # global aug. nat_pos img
        ]

        if self.apply_local_transforms_to_both:
            for transform in self.transforms[2: 2 + (self.n_local_views // 2)]:
                views.append(transform(images[0]))
            for transform in self.transforms[2 + (self.n_local_views // 2):]:
                views.append(transform(images[1]))
        else:
            for transform in self.transforms[2:]:
                views.append(transform(images[0]))
        return views

            
