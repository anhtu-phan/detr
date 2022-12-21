#!/bin/bash

NAME="detr"
PROJECT_NAME="visdrone"
NUM_CLASSES=3

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path ./datasets/$PROJECT_NAME --output_dir ./logs/$PROJECT_NAME/$NAME --lr_drop 20 --wandb_name $NAME --wandb_project_name $PROJECT_NAME --resume ./logs/detr-r50-e632da11.pth --epochs 300 --num_classes $NUM_CLASSES
