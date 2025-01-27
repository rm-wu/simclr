python linear_eval.py --model=dino_vits16 --input-size=512 --patch-size=16 --embeddings-size=384 --dataset-name=voc --data-dir=/flash/project_462000585/mereuric/data --out-dir=outputs/ --num-workers=8 --seed=36

mean loss : 0.5431032480081407
mean val loss : 0.42640994624658063
miou : 0.482565130558726



python  linear_eval.py --model=dinov2_vits14 --input-size=504 --patch-size=14 --embeddings-size=384 --data-dir=/home/mereur1/projects/ocl/ssl_nat_aug/dense_prediction/open-hummingbird-eval/data/ADEChallengeData2016 --out-dir=/home/mereur1/projects/ocl/ssl_nat_aug/dense_prediction/open-hummingbird-eval/outputs/dinov2/ --num-workers=32 --seed=36 --dataset-name=ade20k
mean loss : 0.9831461662638791
mean val loss : 0.942629013210535
miou : 0.37897348668400765



python  linear_eval.py --model=dinov2_vitb14 --input-size=504 --patch-size=14 --embeddings-size=384 --dataset-name=voc --data-dir=data --out-dir=outputs/dinov2b/ --num-workers=32 --seed=42
mean loss : 0.23175469174435953
mean val loss : 0.19737774404612454
miou : 0.8007869666783003


python linear_eval.py --model=ibot_vits16 --input-size=512 --patch-size=16 --embeddings-size=384 --dataset-name=voc --data-dir=/flash/project_462000585/mereuric/data --out-dir=outputs/ --num-workers=8 --seed=36
mean loss : 0.34986543043902735
mean val loss : 0.3058376935395328
miou : 0.658344337409814
