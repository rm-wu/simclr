# DINO pretraining
How to run DINO with 2 GPUs on 1 node for ImageNet
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 \
    --nnode=1 \
    /home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/dino/main_dino.py \
    --data_path /home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/dino/data/imagenet \
    --output_dir /home/mereur1/projects/ocl/ssl_nat_aug/ssl_tests/dino/outputs/ \
    --arch vit_small \
    --epochs 100 --num_workers=16 --batch_size_per_gpu=128
```


## Old script using `torchrun`
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    /home/mereur1/projects/ocl/ssl_nat_aug/dino/main_dino.py \
    --data_path /home/mereur1/projects/ocl/ssl_nat_aug/dino/data/imagenet \
    --output_dir /home/mereur1/projects/ocl/ssl_nat_aug/dino/outputs/dino_vanilla_deitsmall16 \
    --arch vit_small \
    --epochs 100 --num_workers=16 --batch_size_per_gpu=128
```

