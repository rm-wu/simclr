import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
from torch_kmeans import KMeans, CosineSimilarity
import albumentations as A


cmap = plt.get_cmap("tab20")
MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

transforms = A.Compose(
    [
        A.Normalize(mean=list(MEAN), std=list(STD)),
    ]
)


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0]) // self.patch_embed.proj.stride[0] + 1,
            (W - self.patch_embed.patch_size[1]) // self.patch_embed.proj.stride[1] + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def viz_feat(feat):

    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1, 2, 0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))

    return res_pred


def plot_feats(
    image,
    model_option,
    ori_feats,
    fine_feats,
    ori_labels=None,
    fine_labels=None,
    output_dir=None,
    n_head=None,
):

    ori_feats_map = viz_feat(ori_feats)
    # fine_feats_map = viz_feat(fine_feats)

    if ori_labels is not None:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
        ax[0].imshow(ori_feats_map)
        ax[0].set_title("Original " + model_option, fontsize=15)
        ax[1].imshow(ori_labels)
        for x in ax:
            x.xaxis.set_major_formatter(plt.NullFormatter())
            x.yaxis.set_major_formatter(plt.NullFormatter())
            x.set_xticks([])
            x.set_yticks([])
            x.axis("off")

    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=15)
        ax[1].imshow(ori_feats_map)
        ax[1].set_title("Original " + model_option, fontsize=15)

        for x in ax:
            x.xaxis.set_major_formatter(plt.NullFormatter())
            x.yaxis.set_major_formatter(plt.NullFormatter())
            x.set_xticks([])
            x.set_yticks([])
            x.axis("off")

    plt.tight_layout()
    if output_dir is not None and n_head is not None:
        plt.savefig(os.path.join(output_dir, f"attention_vis{n_head}.png"))
    plt.close(fig)
    return fig


def process_image(image, stride, transforms, device=torch.device("cuda")):
    transformed = transforms(image=np.array(image))
    image_tensor = torch.tensor(transformed["image"])
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    h, w = image_tensor.shape[2:]

    height_int = (h // stride) * stride
    width_int = (w // stride) * stride

    image_resized = torch.nn.functional.interpolate(
        image_tensor, size=(height_int, width_int), mode="bilinear"
    )

    return image_resized


def kmeans_clustering(feats_map, n_clusters=20):

    B, D, h, w = feats_map.shape
    feats_map_flattened = feats_map.permute((0, 2, 3, 1)).reshape(B, -1, D)

    kmeans_engine = KMeans(n_clusters=n_clusters, distance=CosineSimilarity)
    kmeans_engine.fit(feats_map_flattened)
    labels = kmeans_engine.predict(feats_map_flattened)
    labels = labels.reshape(B, h, w).float()
    labels = labels[0].cpu().numpy()

    label_map = cmap(labels / n_clusters)[..., :3]
    label_map = np.uint8(label_map * 255)
    label_map = Image.fromarray(label_map)

    return label_map
