# DINO training
How to run DINO with 2 GPUs on 1 node for ImageNet

```shell
torchrun --standalone --nnodes=1 --nproc-per-node=2 /home/mereur1/projects/ocl/ssl_nat_aug/dino/main_dino.py \      (sam2) 
                                                --data_path /home/mereur1/projects/ocl/ssl_nat_aug/dino/data/imagenet \
                                                --output_dir /home/mereur1/projects/ocl/ssl_nat_aug/dino/outputs/dino_vanilla_deitsmall16 \
                                                --arch vit_small \
                                                --epochs 100 --num_workers=16 --batch_size_per_gpu=128
```