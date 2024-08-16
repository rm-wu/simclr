import pathlib
import torch
import torch.nn as nn
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

model = SimCLR.load_from_checkpoint(args.ckpt_path)
proj_head = nn.Linear(512, args.num_classes)
loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(proj_head.parameters())

for i,(X,y) in enumerate(train_dataloader):
	opt.zero_grad()
	with torch.no_grad():
		z = model.backbone(X)[:,:,0,0]
	yhat = proj_head(z)
	loss_i = loss(yhat,y)
	loss_i.backward()
	opt.step()
	acc = (yhat.argmax(1)==y).float().mean().item()
	print(f"Step={i}, Acc={acc}")
	


