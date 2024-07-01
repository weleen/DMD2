export CHECKPOINT_PATH=$1
export WANDB_ENTITY=$2
export WANDB_PROJECT=$3
export MASTER_IP=$4
export NODE_RANK=$5

# accelerate launch --config_file accelerate_configs/default_config.yaml \
accelerate launch --config_file accelerate_configs/default_config_debug.yaml \
    --num_machines 2 \
    --machine_rank $NODE_RANK \
    main/train_sd.py \
    --generator_lr 1e-5  \
    --guidance_lr 1e-5 \
    --train_iters 100000000 \
    --output_path  $CHECKPOINT_PATH/laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/output \
    --log_path $CHECKPOINT_PATH/laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/log \
    --cache_dir $CHECKPOINT_PATH/laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/cache \
    --batch_size 96 \
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
    --wandb_name "laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch"  \
    --wandb_project $WANDB_PROJECT \
    --use_fp16 \
    --log_loss \
    --dfake_gen_update_ratio 10 \
    --gradient_checkpointing 

# on dongxu-g1
# export NCCL_SOCKET_IFNAME=enp45s0f0
# bash experiments/sdv1.5/laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch.sh log/sdv15 yimingwu0 DMD2 10.21.32.12:29500 0
# on dongxu-g3
# export NCCL_SOCKET_IFNAME=enp45s0f0
# bash experiments/sdv1.5/laion6.25_sd_baseline_2node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch.sh log/sdv15 yimingwu0 DMD2 10.21.32.12:29500 1
