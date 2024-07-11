export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3

accelerate launch --config_file accelerate_configs/default_config.yaml main/train_sd.py \
    --generator_lr 5e-7  \
    --guidance_lr 5e-7 \
    --train_iters 10000 \
    --output_path $CHECKPOINT_PATH/laion6.25_sd_baseline_1node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata_accum_3/output \
    --log_path $CHECKPOINT_PATH/laion6.25_sd_baseline_1node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata_accum_3/log \
    --cache_dir $CHECKPOINT_PATH/laion6.25_sd_baseline_1node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata_accum_3/cache \
    --batch_size 32 \
    --grid_size 2 \
    --initialie_generator --log_iters 1000 \
    --resolution 512 \
    --latent_resolution 64 \
    --seed 10 \
    --real_guidance_scale 1.75 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "runwayml/stable-diffusion-v1-5" \
    --train_prompt_path $CHECKPOINT_PATH/captions_laion_score6.25.pkl \
    --real_image_path $CHECKPOINT_PATH/sensei-fs/users/tyin/cvpr_data/sd_vae_latents_laion_500k_lmdb \
    --wandb_iters 50 \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --wandb_name "laion6.25_sd_baseline_1node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata_accum_3"  \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --gradient_checkpointing \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 1e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --ckpt_only_path $CHECKPOINT_PATH/laion6.25_sd_baseline_1node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_gradient_accum_3/cache/time_1719753487_seed10/checkpoint_model_102000 \
    --gradient_accumulation_steps 8
