# %%
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(
        self,
        max_epochs: int = 10,
        optimizer: str = "SGD",
        lr: float = 6e-2,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        input_size: int = 224,
        train_batchsize: int = 512,
        channels: int = 3,
    ):
        super().__init__()
        self.example_input_array = torch.Tensor(
            train_batchsize, channels, input_size, input_size
        )
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=True)

        self.max_epochs = max_epochs
        self.train_batchsize = train_batchsize
        self.input_size = input_size
        self.channels = channels

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, self.max_epochs
            )
            return [optim], [scheduler]
        elif self.optimizer == "LARS":
            raise NotImplementedError("LARS is not implemented yet")
        else:
            raise ValueError(f"Optimizer {self.optimizer} is not supported")
