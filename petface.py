import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from typing import Optional
import pandas as pd
import pathlib as pl
from PIL import Image


class PetFaceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        natural_augmentation: bool = False,
        transform: Optional[nn.Module] = None,
    ):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"split value {split} is not valid")
        self.root = pl.Path(root)
        self.split = split
        self.transform = transform
        self.natural_augmentation = natural_augmentation
        self.files_df = pd.DataFrame()
        self._prepare_metadata()

    def _prepare_metadata(self):
        split_dir = self.root / "split"
        for dir in split_dir.glob("*"):
            if self.split == "train":
                df = pd.read_csv(dir / "train.csv")
                df["class"] = dir.name
            else:
                df = pd.read_csv(dir / f"{self.split}.txt")
                df.columns = ["filename"]
                df["class"] = dir.name
                df["label"] = ...
            self.files_df = pd.concat([self.files_df, df], ignore_index=True)

    def __len__(self):
        return len(self.files_df)

    def __getitem__(self, index):
        img_path = self.files_df.iloc[index].filename
        img_path = self.root / "images" / img_path
        image = Image.open(img_path).convert("RGB")

        if self.natural_augmentation:
            pos = self.get_positive(index)
            if self.transform:
                image = self.transform(image)
                pos = self.transform(pos)
            return [image, pos], 0
        else:
            if self.transform:
                image = self.transform(image)
            return image, 0

    def get_positive(self, index):
        image = self.files_df.iloc[index]
        image_path = pl.Path(self.root / "images" / image.filename)
        positives = list(image_path.parent.glob("*.png"))

        # Uncomment to remove the current image from the list of positives
        # positives = [str(p) for p in positives if p != image_path]
        
        # Select one of the positives
        positive_idx = np.random.randint(len(positives))
        pos_path = positives[positive_idx]
        pos = Image.open(pos_path).convert("RGB")
        return pos


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    DATA_PATH = pl.Path("~/projects/ocl/data/PetFace/").expanduser().resolve()
    train_dataset = PetFaceDataset(
        root=DATA_PATH, split="train", transform=transforms.ToTensor()
    )
    print(len(train_dataset))
    img = train_dataset[1_000]
    pos = train_dataset.get_positive(1_000)

    print(img.shape)
    print(pos.shape)

    val_dataset = PetFaceDataset(
        root=DATA_PATH, split="val", transform=transforms.ToTensor()
    )
    print(len(val_dataset))
    val_img = val_dataset[30_000]
    print(val_img.shape)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=16,
    )
    print(len(train_loader))

    for i, batch in enumerate(train_loader):
        print(batch.shape)
        break
