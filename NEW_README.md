### Setup DMD2 SDv1.5

#### Download SDv1.5 training and evaluation datasets
Follow instructions in [sdv1.5](experiments/sdv1.5/README.md)
```bash
# Make sure current directory is at DMD2's root directory
export CHECKPOINT_PATH="./log/sdv15" # change this to your own checkpoint folder 
export WANDB_ENTITY="" # change this to your own wandb entity
export WANDB_PROJECT="" # change this to your own wandb project
export MASTER_IP=""  # change this to your own master ip

# Not sure why but we found the following line necessary to work with the accelerate package in our system. 
# Change YOUR_MASTER_IP/YOUR_MASTER_NODE_NAME to the correct value 
echo "YOUR_MASTER_IP 	YOUR_MASTER_NODE_NAME" | sudo tee -a /etc/hosts

mkdir -p $CHECKPOINT_PATH

bash scripts/download_sdv15.sh $CHECKPOINT_PATH
```

#### Setup Environment
```bash
conda env create -f environment.yml

python setup.py develop
```

#### Training and Testing
```bash
# start a training with 64 gpu. we need to run this script on all 8 nodes. 
bash experiments/laion/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch.sh $CHECKPOINT_PATH  $WANDB_ENTITY $WANDB_PROJECT $MASTER_IP

# on some other machine, start a testing process that continually reads from the checkpoint folder and evaluate the FID 
# Change TIMESTAMP_TBD to the real one
python main/test_folder_sd.py   --folder $CHECKPOINT_PATH/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch/TIMESTAMP_TBD \
    --wandb_name test_laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch \
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
```