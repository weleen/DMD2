#!/bin/bash
export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

python main/test_folder_sd.py   --folder $CHECKPOINT_PATH/laion6.25_sd_baseline_1node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_gradient_accum_3/cache/time_1719753487_seed10 \
    --wandb_name test_laion6.25_sd_baseline_1node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_grad_accum_3 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --image_resolution 512 \
    --latent_resolution 64 \
    --num_train_timesteps 1000 \
    --test_visual_batch_size 64 \
    --per_image_object 16 \
    --seed 10 \
    --anno_path $CHECKPOINT_PATH/captions_coco14_test.pkl \
    --eval_res 256 \
    --ref_dir $CHECKPOINT_PATH/val2014 \
    --total_eval_samples 30000 \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --pred_eps 

# bash experiments/sdv1.5/test_laion6.25_sd_baseline_1node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch.sh log/sdv15 yimingwu0 DMD2