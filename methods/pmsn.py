from typing import Tuple, List
import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch import Tensor
from torch.optim import AdamW

from lightly.loss import PMSNLoss
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead
from lightly.transforms import MSNTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler


class PMSN(pl.LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int):
        super().__init__()

        # ViT small configuration (ViT-S/16)
        self.feature_dim = 384
        self.batch_size_per_device = batch_size_per_device
        self.num_classes = num_classes

        self.mask_ratio = 0.15
        vit = torchvision.models.VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=self.feature_dim,
            mlp_dim=self.feature_dim * 4,
        )
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        # or use a torchvision ViT backbone:
        # vit = torchvision.models.vit_b_32(pretrained=False)
        # self.backbone = MAEBackbone.from_vit(vit)
        self.projection_head = MSNProjectionHead(self.feature_dim)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

        # set gather_distributed to True for distributed training
        self.criterion = PMSNLoss(gather_distributed=True)

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes
        )
        

    def training_step(self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int) -> Tensor:
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views = batch[0]
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_features = self.backbone(images=targets)
        targets_out = self.projection_head(targets_features)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        
        self.log(
            "train/loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )
        labels = batch[1]
        cls_loss, cls_log = self.online_classifier.training_step(
            (targets_features.detach(), labels), batch_idx=batch_idx
        )
        cls_log = {k.replace("train_online_", "train_online/"): v
                   for (k, v) in cls_log.items()}
        self.log_dict(cls_log, sync_dist=True, batch_size=len(labels))
        
        return loss + cls_loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(images=anchors, idx_keep=idx_keep)
        return self.anchor_projection_head(out)
    
    def validation_step(self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int ) -> Tensor:
        images, labels = batch[0], batch[1]
        features = self.backbone(images)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), labels), batch_idx
        )
        cls_log = {k.replace("val_online_", "val_online/"): 
                   v for k, v in cls_log.items()}
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(labels))
        return cls_loss


    def configure_optimizers(self):
        # Don't use weight decay for batch norm, bias parameters, and classification
        # head to improve performance.
        params, params_no_weight_decay = utils.get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        optimizer = AdamW(
            [
                {"name": "pmsn", "params": params},
                {
                    "name": "pmsn_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=1.5e-4 * self.batch_size_per_device * self.trainer.world_size / 256,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 40
                ),
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]
    
    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=3.0,
            gradient_clip_algorithm="norm",
        )
        # self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

    # def configure_optimizers(self):
    #     params = [
    #         *list(self.anchor_backbone.parameters()),
    #         *list(self.anchor_projection_head.parameters()),
    #         self.prototypes,
    #     ]
    #     optim = torch.optim.AdamW(params, lr=1.5e-4)
    #     return optim


# model = PMSN()

# transform = MSNTransform()
# # we ignore object detection annotations by setting target_transform to return 0
# dataset = torchvision.datasets.VOCDetection(
#     "datasets/pascal_voc",
#     download=True,
#     transform=transform,
#     target_transform=lambda t: 0,
# )
# # or create a dataset from a folder containing images or videos:
# # dataset = LightlyDataset("path/to/folder")

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     drop_last=True,
#     num_workers=8,
# )

# gpus = torch.cuda.device_count()

# # Train with DDP on multiple gpus. Distributed sampling is also enabled with
# # replace_sampler_ddp=True.
# trainer = pl.Trainer(
#     max_epochs=10,
#     devices="auto",
#     accelerator="gpu",
#     strategy="ddp",
#     use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
# )
# trainer.fit(model=model, train_dataloaders=dataloader)