import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

from typing import Callable
import einops as ein
from tqdm import tqdm, trange
import numpy as np

from src.dataset.voc_data import VOCDataModule
from src.dataset.ade20kdata import Ade20kDataModule
from src.ls_utils import PredsmIoU

from src.transforms.image_transformations import (
    SepTransforms,
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    CombTransforms,
)


def ls_finetune(
    backbone: nn.Module,
    feat_extr_fn: Callable,
    patch_size: int,
    head_type: str,
    max_epochs: int,
    lr: float,
    decay_rate: float,
    drop_at: int,
    batch_size: int,
    dataset_name: str,
    data_dir: Path,
    num_workers: int = 32,
    input_size: int = 448,
    train_mask_size: int = 100,
    val_mask_size: int = 100,
    device: torch.device = torch.device("cuda"),
):
    # TODO: These hyperparameters should be checked with the original code
    # input_size = args.input_size # 448
    # train_mask_size = 100
    # val_mask_size = 100

    img_train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    shared_train_transform = Compose(
        [RandomResizedCrop(size=input_size, scale=(0.8, 1.0))]
    )

    img_val_transforms = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    shared_val_transform = T.Compose(
        [
            Resize((input_size, input_size)),
            T.ToTensor(),
        ]
    )
    val_target_transforms = T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            Resize((input_size, input_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transforms = Compose(
        [
            RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if dataset_name == "voc":
        num_classes = 21
        ignore_index = 255
        data_module = VOCDataModule(
            batch_size=batch_size,
            return_masks=True,
            num_workers=num_workers,
            train_split="trainaug",
            val_split="val",
            data_dir=data_dir,
            train_image_transform=train_transforms,
            drop_last=True,
            # val_image_transform=val_transforms,
            # val_target_transform=None,
            val_transforms=val_transforms,
        )
    # TODO: This part is missing since the open-hummingbird-eval does not support
    # COCO. But it should be possible to integrate that from NeCo repository maybe.
    # elif "coco" in dataset_name:
    #     assert len(dataset_name.split("-")) == 2
    #     mask_type = dataset_name.split("-")[-1]
    #     assert mask_type in ["thing", "stuff"]
    #     if mask_type == "thing":
    #         num_classes = 12
    #     else:
    #         num_classes = 15
    #     ignore_index = 255
    #     file_list = os.listdir(os.path.join(data_dir, "images", "train2017"))
    #     file_list_val = os.listdir(os.path.join(data_dir, "images", "val2017"))
    #     random.shuffle(file_list_val)
    #     # sample 10% of train images
    #     random.shuffle(file_list)
    #     file_list = file_list[:int(len(file_list)*0.1)]
    #     print(f"sampled {len(file_list)} COCO images for training")

    #     data_module = CocoDataModule(batch_size=train_config["batch_size"],
    #                                  num_workers=_config["num_workers"],
    #                                  file_list=file_list,
    #                                  data_dir=data_dir,
    #                                  file_list_val=file_list_val,
    #                                  mask_type=mask_type,
    #                                  train_transforms=train_transforms,
    #                                  val_transforms=val_image_transforms,
    #                                  val_target_transforms=val_target_transforms)
    elif dataset_name == "ade20k":
        # TODO: Evaluate its correctness
        num_classes = 151
        ignore_index = 0
        # val_transforms = SepTransforms(val_image_transforms, val_target_transforms)
        data_module = Ade20kDataModule(
            data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"{dataset_name} not supported")

    # Init Method
    assert (input_size / patch_size).is_integer()
    spatial_res = int(input_size // patch_size)

    linear_head = nn.Conv2d(backbone.embed_dim, num_classes, 1).to(device)

    # freeze all layers of backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    optimizer = torch.optim.SGD(
        linear_head.parameters(), weight_decay=1e-4, momentum=0.9, lr=lr
    )
    scheduler = StepLR(optimizer, gamma=decay_rate, step_size=drop_at)

    data_module.setup()
    dataset_size = data_module.get_train_dataset_size()
    num_classes = data_module.get_num_classes()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    train_losses = []
    pbar = trange(max_epochs, ncols=80)

    for epoch in pbar:
        pbar.set_description(f"Epoch [{epoch + 1}]")
        pbar_iter = tqdm(train_loader, ncols=80)
        for batch in pbar_iter:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            _, _, H, W = images.shape
            assert H == W == input_size

            with torch.no_grad():
                tokens, _ = feat_extr_fn(backbone, images)
                tokens = ein.rearrange(
                    tokens, "b (h w) d -> b d h w", h=spatial_res, w=spatial_res
                )
                tokens = nn.functional.interpolate(
                    tokens, size=(train_mask_size, train_mask_size), mode="bilinear"
                )

            # optimizer.zero_grad()
            # outputs = linear_head(tokens)
            # masks *= 255
            # if train_mask_size != input_size:
            #     with torch.no_grad():
            #         masks = nn.functional.interpolate(
            #             masks, size=(train_mask_size, train_mask_size), mode="nearest"
            #         )

            # loss = nn.CrossEntropyLoss()(outputs, masks.long().squeeze())
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            outputs = linear_head(tokens)
            masks *= 255
            if train_mask_size != input_size:
                with torch.no_grad():
                    masks = nn.functional.interpolate(
                        masks, size=(train_mask_size, train_mask_size), mode="nearest"
                    )
            loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(
                outputs, masks.long().squeeze()
            )
            loss.backward()
            optimizer.step()
            pbar_iter.set_postfix(loss=loss.item())
            train_losses.append(loss.item())

        scheduler.step()
        print()
        print(f"Epoch [{epoch+1}/{max_epochs}]: mean loss : {np.mean(train_losses)}")

        # Validation Step
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            miou_metric = PredsmIoU(num_classes, num_classes)
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    images, masks = batch
                    images = images.to(device)
                    masks = masks.to(device)

                    tokens, _ = feat_extr_fn(backbone, images)
                    tokens = ein.rearrange(
                        tokens, "b (h w) d -> b d h w", h=spatial_res, w=spatial_res
                    )
                    tokens = nn.functional.interpolate(
                        tokens, size=(val_mask_size, val_mask_size), mode="bilinear"
                    )
                    outputs = linear_head(tokens)

                    mask_preds = torch.argmax(outputs, dim=1).unsqueeze(1)

                    gt = masks * 255
                    gt = nn.functional.interpolate(
                        gt, size=(val_mask_size, val_mask_size), mode="nearest"
                    )
                    val_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(
                        outputs, gt.long().squeeze()
                    )
                    val_losses.append(val_loss.item())
                    valid = gt != ignore_index  # mask to remove object boundary class
                    # update metric
                    miou_metric.update(gt[valid], mask_preds[valid])

                print(f"mean val loss : {np.mean(val_losses)}")
                miou = miou_metric.compute(True, many_to_one=False, linear_probe=True)[
                    0
                ]
                miou_metric.reset()
                print(f"miou : {miou}")
                print()

    # model = LinearFinetune(
    #     patch_size=patch_size,
    #     head_type=head_type,
    #     backbone=backbone,
    #     # arch_version=train_config.get("arch_version"),
    #     num_classes=num_classes,
    #     lr=lr,
    #     input_size=input_size,
    #     spatial_res=int(spatial_res),
    #     val_iters=val_iters,
    #     decay_rate=decay_rate if decay_rate is not None else 0.1,
    #     drop_at=drop_at,
    #     ignore_index=ignore_index,
    # )

    # # Optionally load weights
    # if not restart:
    #     weights = get_backbone_weights(
    #         arch, method, patch_size=patch_size, ckpt_path=train_config.get("ckpt_path")
    #     )
    #     msg = model.load_state_dict(weights, strict=False)
    #     print(msg)

    # # Init checkpoint callback storing top 3 heads
    # checkpoint_dir = os.path.join(
    #     train_config["ckpt_dir"], _run.experiment_info["name"]
    # )
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     monitor="miou_val",
    #     filename="ckp-{epoch:02d}-{miou_val:.4f}",
    #     save_top_k=3,
    #     mode="max",
    #     verbose=True,
    # )

    # # Init trainer and start training head
    # trainer = Trainer(
    #     num_sanity_val_steps=0,
    #     logger=neptune_logger,
    #     max_epochs=train_config["max_epochs"],
    #     gpus=_config["gpus"],
    #     accelerator="ddp" if _config["gpus"] > 1 else None,
    #     fast_dev_run=train_config["fast_dev_run"],
    #     log_every_n_steps=50,
    #     benchmark=True,
    #     deterministic=False,
    #    resume_from_checkpoint=train_config["ckpt_path"] if restart else None,
    #     amp_backend="native",
    #     terminate_on_nan=True,
    #     callbacks=[checkpoint_callback],
    # )
    # trainer.fit(model, datamodule=data_module)
    # pass
