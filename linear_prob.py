import pathlib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightly.transforms.simclr_transform import SimCLRViewTransform, SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from parser import parse_arguments
from simclr import SimCLR
from petface import PetFaceDataset

TEST_SPLIT = ['hamster', 'hedgehog', 'javasparrow']

args = parse_arguments()
print(f"Command line arguments {args}")
# args.ckpt_path = '/mnt/qb/work/bethge/cyildiz40/simclr/logs/lightning/petface/epoch=0-step=2000.ckpt'

#### Set seed for reproducibility
if args.seed != -1:
    pl.seed_everything(args.seed, workers=True)

transform = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ]
)

args.data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
train_dataset = PetFaceDataset(
    root=args.data_dir,
    split="train",
    transform=transform,
    class_names=TEST_SPLIT
)
args.num_classes = len(train_dataset.classes)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)

val_dataset = PetFaceDataset(
    root=args.data_dir,
    split="val",
    transform=transform,
    class_names=TEST_SPLIT
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)

model = SimCLR.load_from_checkpoint(args.ckpt_path)
proj_head = nn.Linear(512, args.num_classes).to(model.device)
loss = nn.CrossEntropyLoss()
opt  = torch.optim.Adam(proj_head.parameters(),2e-3)

for ep in range(5):
    for i,(X,y) in enumerate(train_dataloader):
        opt.zero_grad()
        X,y = X.to(model.device), y.to(model.device)
        with torch.no_grad():
            z = model.backbone(X)[:,:,0,0]
        yhat = proj_head(z)
        loss_i = loss(yhat,y)
        loss_i.backward()
        opt.step()
        acc = (yhat.argmax(1)==y).float().mean().item()
        if i//10==0:
            print(f"\tEpoch={ep}, step={i}, tr_acc={acc}")
    with torch.no_grad():
        accs = []
        for i,(X,y) in enumerate(val_dataloader):
            X,y = X.to(model.device), y.to(model.device)
            z = model.backbone(X)[:,:,0,0]
            yhat = proj_head(z)
            acc = (yhat.argmax(1)==y).float().mean().item()
            accs.append(acc)
        val_acc = np.mean(acc)
        print(f"Epoch={ep}, val_acc={val_acc}")


