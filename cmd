pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install mxnet-cu101==1.7.0

CUDA_VISIBLE_DEVICES=0 python train.py --epochs 500 --batch-size=8 --datasetdir '/storages/data/dungpm/All_data/data_body_segmentation/LV-MHP-v1' --dataset mhpv1 --model icnet --backbone resnet50 --lr 0.001 --checkname myckpt_glab_icnet --resume './runs/mhpv1/icnet/resnet50/last.params'

CUDA_VISIBLE_DEVICES=0 python train.py --epochs 500 --batch-size=8 --dataset mhpv1 --model icnet --backbone resnet50 --resume './runs/mhpv1/icnet/resnet50/last.params'

