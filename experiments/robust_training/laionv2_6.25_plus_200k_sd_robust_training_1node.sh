#!/bin/bash

MODEL_NAME="checkpoints/stable-diffusion-v1-5"
TRIAN_DATA_DIR=./log/robust_training/laion_aes/preprocessed_212k
OUTPUT_DIR=./log/robust_training/laionv2_6.25_plus_200k_sd_robust_training_1node

BATCH_SIZE=32
NUM_GPUS=8

StartTime=$(date +%s)

accelerate launch --mixed_precision="fp16" --multi_gpu --num-processes $NUM_GPUS main/train_sd_robust.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir $TRIAN_DATA_DIR \
  --dataloader_num_workers 10 \
  --use_ema \
  --death_mode="uniform" \
  --death_rate=0.5 \
  --replace_strategy="conv" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --max_train_steps=50000 \
  --validation_epochs 1 \
  --learning_rate=5e-05 \
  --max_grad_norm=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="all" \
  --lambda_sd 1.0 \
  --output_dir=$OUTPUT_DIR \
  --allow_tf32
  # --gradient_accumulation_steps=1 \
  # --gradient_checkpointing \

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."