python linear_eval.py \
    --model=dinov2_vits14 \
    --input-size=504 \
    --patch-size=14 \
    --embeddings-size=384 \
    --dataset-name=coco-stuff \
    --num-workers=16 \
    --data-dir=/home/mereur1/projects/ocl/ssl_nat_aug/data/coco_stuff164k \
    --out-dir=/home/mereur1/projects/ocl/ssl_nat_aug/outputs/debug/coco_stuff

python linear_eval.py \
    --model=dinov2_vits14 \
    --input-size=504 \
    --patch-size=14 \
    --embeddings-size=384 \
    --dataset-name=coco-thing \
    --num-workers=16 \
    --data-dir=/home/mereur1/projects/ocl/ssl_nat_aug/data/coco_stuff164k \
    --out-dir=/home/mereur1/projects/ocl/ssl_nat_aug/outputs/debug/coco_thing

    