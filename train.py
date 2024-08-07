import pathlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightly.transforms.simclr_transform import SimCLRViewTransform, SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from parser import parse_arguments
from simclr import SimCLR
from petface import PetFaceDataset


#### Parse command line arguments
args = parse_arguments()
print(f"Command line arguments {args}")

#### Set seed for reproducibility
if args.seed != -1:
    pl.seed_everything(args.seed, workers=True)

#### Set up Weights & Biases logger
if args.use_wandb:
    wandb_logger = WandbLogger(
        project="simclr",
        name="petface-nat" if args.natural_augmentation else "petface",
        save_dir=args.log_dir,
    )
else:
    wandb_logger = None

#### Select the typology of augmentation to use in the experiment
if args.natural_augmentation:
    # Applies the SimCLR view on each one of the "natural" augmentations
    # Note: This alters the default SimCLR augmentation because uses different
    # images.
    transform = SimCLRViewTransform(
        input_size=args.input_size,
        cj_prob=args.cj_prob,
        cj_strength=args.cj_strength,
        cj_bright=args.cj_bright,
        cj_contrast=args.cj_contrast,
        cj_sat=args.cj_sat,
        cj_hue=args.cj_hue,
        min_scale=args.min_scale,
        random_gray_scale=args.random_gray_scale,
        gaussian_blur=args.gaussian_blur,
    )
else:
    # Applies the default SimCLR transforms and generates two views
    transform = SimCLRTransform(
        input_size=args.input_size,
        cj_prob=args.cj_prob,
        cj_strength=args.cj_strength,
        cj_bright=args.cj_bright,
        cj_contrast=args.cj_contrast,
        cj_sat=args.cj_sat,
        cj_hue=args.cj_hue,
        min_scale=args.min_scale,
        random_gray_scale=args.random_gray_scale,
        gaussian_blur=args.gaussian_blur,
    )

#### Create Dataset and DataLoaders
# DATA_PATH = pathlib.Path("~/projects/ocl/data/PetFace/").expanduser().resolve()
args.data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
train_dataset = PetFaceDataset(
    root=args.data_dir,
    split="train",
    transform=transform,
    natural_augmentation=args.natural_augmentation,
)
args.num_classes = len(train_dataset.classes)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)

if args.skip_validation:
    val_dataloader = None
else:   
    # Setup validation data.
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
        ]
    )
    val_dataset = PetFaceDataset(
        root=args.data_dir, split="val", transform=val_transform
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_device,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False,
    )


# # Old implementation
# from simplified_simclr import SimCLR as SimplifiedSimCLR
# model = SimplifiedSimCLR(
#     # backbone=args.backbone,
#     max_epochs=args.max_epochs,
#     lr=args.learning_rate,
#     momentum=args.momentum,
#     weight_decay=args.weight_decay,
#     input_size=args.input_size,
#     train_batchsize=args.batch_size_per_device,
# )

model = SimCLR(
    backbone=args.backbone,
    batch_size_per_device=args.batch_size_per_device,
    num_classes=args.num_classes,
)

# Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    # limit_train_batches=0.01,
    fast_dev_run=args.fast_dev_run,
    # profiler="simple",
    default_root_dir=args.log_dir,
    devices=args.devices,
    accelerator=args.accelerator,
    strategy="ddp",
    sync_batchnorm=True,
    use_distributed_sampler=True,
    logger=wandb_logger,
    deterministic=False if args.seed == -1 else True,
)
trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
