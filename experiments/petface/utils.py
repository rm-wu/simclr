#%%
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))



def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig



#%%
from methods import SimCLR, DINO, VICReg
from argparse import Namespace
from petface import PetFaceDataset

from torch.utils.data import DataLoader
import torchvision.transforms as T

from lightly.transforms.utils import IMAGENET_NORMALIZE

TRAIN_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster__', 'hedgehog__', 'javasparrow__', 'parakeet', 'pig', 'rabbit']
VAL_SPLIT = ['cat', 'chimp', 'chinchilla', 'degus', 'dog', 'ferret', 'guineapig', 'hamster', 'hedgehog', 'javasparrow', 'parakeet', 'pig', 'rabbit']
TEST_SPLIT = ['hamster', 'hedgehog', 'javasparrow']


args = Namespace(
    method="vicreg",
    ckpt_path="/home/mereur1/projects/ocl/ssl_nat_aug/logs/mahti/vicreg_petface.ckpt",
    batch_size_per_device=512, 
    num_workers=8,
    data_dir="/home/mereur1/projects/ocl/data/PetFace"
)

if args.method == "simclr":
    model = SimCLR.load_from_checkpoint(args.ckpt_path)
elif args.method == "dino":
    model = DINO.load_from_checkpoint(args.ckpt_path)
elif args.method == "vicreg":
    model = VICReg.load_from_checkpoint(args.ckpt_path)


val_transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ]
)
val_dataset = PetFaceDataset(
    root=args.data_dir, split="val", transform=val_transform, class_names=VAL_SPLIT
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=False,
)



#%%
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision, MulticlassF1Score

acc_metric = MulticlassAccuracy(num_classes=13, average=None).to(model.device)

for i, batch in enumerate(train_dataloader):
    images = batch[0].to(model.device)
    labels = batch[1].to(model.device)

    output = model.backbone(images).flatten(start_dim=1)
    output = model.online_classifier(output)
    _, preds_tensor = torch.max(output, 1)
    acc = acc_metric(preds_tensor, labels)
    # acc = acc_metric(output, labels)
    
    print(f"Acc. Metric in batch {i}: {acc}")

acc = acc_metric.compute()

print(f"Acc. on all validation data: {acc}")


# #%%

# batch = next(iter(val_dataloader))
# for batch in val_dataloader:
#     images = batch[0].to(model.device)
#     labels = batch[1].to(model.device)

#     output = model.backbone(images).flatten(start_dim=1)
#     output = model.online_classifier(output).cpu()
#     _, preds_tensor = torch.max(output, 1)
#     preds = np.squeeze(preds_tensor.cpu().numpy())
#     probs = [F.softmax(el, dim=0)[i].cpu().item() for i, el in zip(preds, output)]
#     num_imgs = 10
#     classes = VAL_SPLIT 
#     fig = plt.figure(figsize=(12, 12*num_imgs))
#     for idx in np.arange(num_imgs):
#         ax = fig.add_subplot(1, num_imgs, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images[idx].cpu(), one_channel=False)
#         ax.set_title(
#             "{0}, {1:.1f}%\n(label: {2})".format(
#                 classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
#             ),
#             color=("green" if preds[idx] == labels[idx].item() else "red"),
#         )
#     plt.show()

# #%%
# output = net(images)
# # convert output probabilities to predicted class
# _, preds_tensor = torch.max(output, 1)
# preds = np.squeeze(preds_tensor.numpy())
# # return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


# preds, probs = images_to_probs(net, images)
# # plot the images in the batch, along with predicted and true labels
# fig = plt.figure(figsize=(12, 48))
# for idx in np.arange(4):
#     ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
#     matplotlib_imshow(images[idx], one_channel=True)
#     ax.set_title(
#         "{0}, {1:.1f}%\n(label: {2})".format(
#             classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
#         ),
#         color=("green" if preds[idx] == labels[idx].item() else "red"),
#     )
# %%
