#!/bin/bash
# GR00T N1.7 fine-tune on lerobot_flipsteak_500_224 dataset.
# Bimanual Franka, 16-dim state/action. 5 epochs, max BS for 8× L20Z 80GB.
source ~/miniconda3/bin/activate
conda activate gr00t

export WANDB_API_KEY=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

# Keep all HF cache on NAS; /root may be wiped.
export HF_HOME=/public/hz_nas/tong/model/hf_cache
# Cluster has no HF egress.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

BASE_MODEL=/public/hz_nas/tong/model/GR00T/GR00T-N1.7-3B
DATASET_PATH=/public/hz_nas/tong/data/lerobot_flipsteak_500_224
OUTPUT_DIR=/public/hz_nas/tong/model/gr00t_franka_flip_steak
MODALITY_CONFIG=/public/hz_nas/tong/code/Isaac-GR00T/examples/franka_steak/franka_steak_config.py

NUM_GPUS=8
# Max tested per-GPU BS=320 → global=2560, 89% VRAM (72.6/81 GB per card).
# BS=360 (98%) works but too risky for long runs; BS=400 OOMs.
GLOBAL_BATCH_SIZE=2560
# 5 epochs: 5 × 1,074,876 frames / 2560 = 2099 steps
MAX_STEPS=2100
# LR sqrt-scaled from 1e-4 base (original BS=128): 1e-4 × sqrt(2560/128) ≈ 4.5e-4
LEARNING_RATE=5e-4
# Save every ~1 epoch
SAVE_STEPS=420

cd /public/hz_nas/tong/code/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path ${BASE_MODEL} \
    --dataset-path ${DATASET_PATH} \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path ${MODALITY_CONFIG} \
    --num-gpus ${NUM_GPUS} \
    --output-dir ${OUTPUT_DIR} \
    --experiment-name gr00t_franka_flip_steak \
    --wandb-project finetune-gr00t-franka-steak \
    --use-wandb \
    --max-steps ${MAX_STEPS} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --save-steps ${SAVE_STEPS} \
    --save-total-limit 6 \
    --learning-rate ${LEARNING_RATE} \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    --dataloader-num-workers 4
