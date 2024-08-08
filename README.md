# SimCLR using natural augmentation

This repo is meant to test the use of natural augmentations, meaning using images containing images of the same object with different lighting conditions, points of view etc., that arise in the real world. 

The experiments use [PetFace dataset](https://dahlian00.github.io/PetFacePage/).

The implementation builds on top of  [Lightly SSL](https://github.com/lightly-ai/lightly).

### Install Environment 
```bash
conda create env -f environment.yml
```

### SimCLR Pretrain + Natural Augmentations
Run the pretrain expoiting the additional "natural" augmentations pictures:
```bash
python train.py --use_wandb \
--natural_augmentaiton \
--devices=2 \
--backbone=resnet18 \
--batch_size_per_device=512 \
--max_epochs=<NUM_EPOCHS> \
--data_dir=<PETFACE_PATH> \
--seed=<SEED>
```

### SimCLR Pretrain
Run the pretrain with basic SimCLR:
```bash 
python train.py --use_wandb \
--devices=2 \
--backbone=resnet18 \
--batch_size_per_device=512 \
--max_epochs=<NUM_EPOCHS> \
--data_dir=<PETFACE_PATH> \
--seed=<SEED>
```