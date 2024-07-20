#!/bin/bash

MODEL_NAME="checkpoints/stable-diffusion-v1-5"
TRIAN_DATA_DIR=./log/robust_training/laion_aes/preprocessed_11k
OUTPUT_DIR=./log/robust_training/laionv2_6.5_plus_sd_robust_training_distill_1node_toy

BATCH_SIZE=2
NUM_GPUS=1

StartTime=$(date +%s)

accelerate launch --mixed_precision="bf16" --num-processes $NUM_GPUS main/train_sd_robust.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir $TRIAN_DATA_DIR \
  --dataloader_num_workers 10 \
  --use_ema \
  --death_mode="uniform" \
  --death_rate=0.5 \
  --replace_strategy="conv" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --max_train_samples=100 \
  --max_train_steps=1000 \
  --validation_steps 1 \
  --checkpointing_steps 1 \
  --learning_rate=5e-05 \
  --max_grad_norm=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="all" \
  --enable_teacher_distillation \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --output_dir=$OUTPUT_DIR \
  --allow_tf32 \
  --is_debug

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."