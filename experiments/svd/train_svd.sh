#!/bin/bash
# finetune svd
python main/train_svd.py --config main/svd/train_svd.yaml

# multi-gpu
# accelerate launch --config_file accelerate_configs/deepspeed_zero2_config.yaml main/train_svd.py --config main/svd/train_svd.yaml