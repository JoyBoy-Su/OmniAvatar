#!/bin/bash
#
# 多GPU训练启动脚本
# 支持单节点多卡和多节点分布式训练
#

set -e

# ========================
# Configuration
# ========================

# Model Configuration
MODEL_CONFIG="configs/model_config.yaml"
TRAINING_STAGE="stage2"  # stage1, stage2, or stage3

# Data Configuration
TRAIN_DATA_DIR="/data/audio_video_processed/shards"
VAL_DATA_DIR="/data/audio_video_val/shards"

# Output Configuration
OUTPUT_DIR="outputs/${TRAINING_STAGE}_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${OUTPUT_DIR}/logs"

# Training Configuration
NUM_GPUS=8  # Number of GPUs per node
NUM_NODES=1  # Number of nodes
BATCH_SIZE=16  # Per GPU
GRADIENT_ACCUMULATION=4
MIXED_PRECISION="bf16"  # bf16, fp16, or no

# Distributed Configuration (for multi-node)
MASTER_ADDR="localhost"
MASTER_PORT=29500
NODE_RANK=0

# DeepSpeed Configuration
USE_DEEPSPEED=true
DEEPSPEED_CONFIG="configs/deepspeed_config.json"

# WandB Configuration
USE_WANDB=true
WANDB_PROJECT="audio-video-generation"
WANDB_RUN_NAME="${TRAINING_STAGE}_$(date +%Y%m%d_%H%M%S)"

# Resume from checkpoint
RESUME_FROM_CHECKPOINT=""  # Leave empty for new training

# ========================
# Setup
# ========================

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"

# Log configuration
echo "======================================"
echo "Training Configuration"
echo "======================================"
echo "Model Config:        ${MODEL_CONFIG}"
echo "Training Stage:      ${TRAINING_STAGE}"
echo "Output Directory:    ${OUTPUT_DIR}"
echo "Number of GPUs:      ${NUM_GPUS}"
echo "Number of Nodes:     ${NUM_NODES}"
echo "Batch Size:          ${BATCH_SIZE}"
echo "Gradient Accum:      ${GRADIENT_ACCUMULATION}"
echo "Mixed Precision:     ${MIXED_PRECISION}"
echo "Use DeepSpeed:       ${USE_DEEPSPEED}"
echo "======================================"

# ========================
# Environment Setup
# ========================

# Activate virtual environment (if needed)
# source /path/to/venv/bin/activate

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set NCCL parameters for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=NVL

# Set PyTorch CUDA optimization
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # Adjust for your GPUs

# ========================
# Training Command
# ========================

TRAINING_ARGS=(
    --config "${MODEL_CONFIG}"
    --stage "${TRAINING_STAGE}"
    --train_data_dir "${TRAIN_DATA_DIR}"
    --val_data_dir "${VAL_DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}"
    --mixed_precision "${MIXED_PRECISION}"
    --logging_dir "${LOG_DIR}"
    --logging_steps 100
    --save_steps 5000
    --eval_steps 2000
    --save_total_limit 5
    --dataloader_num_workers 8
    --dataloader_pin_memory
    --gradient_checkpointing
)

# Add resume checkpoint if specified
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TRAINING_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# Add WandB configuration
if [ "${USE_WANDB}" = true ]; then
    TRAINING_ARGS+=(
        --report_to wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_run_name "${WANDB_RUN_NAME}"
    )
fi

# ========================
# Launch Training
# ========================

if [ "${USE_DEEPSPEED}" = true ]; then
    # DeepSpeed launcher
    echo "Launching with DeepSpeed..."
    
    deepspeed --num_gpus="${NUM_GPUS}" \
              --num_nodes="${NUM_NODES}" \
              --master_addr="${MASTER_ADDR}" \
              --master_port="${MASTER_PORT}" \
              --node_rank="${NODE_RANK}" \
              training/train.py \
              "${TRAINING_ARGS[@]}" \
              --deepspeed "${DEEPSPEED_CONFIG}" \
              2>&1 | tee "${LOG_DIR}/train.log"
              
else
    # Accelerate launcher (DDP or FSDP)
    echo "Launching with Accelerate..."
    
    accelerate launch \
        --multi_gpu \
        --num_processes="${NUM_GPUS}" \
        --num_machines="${NUM_NODES}" \
        --machine_rank="${NODE_RANK}" \
        --main_process_ip="${MASTER_ADDR}" \
        --main_process_port="${MASTER_PORT}" \
        --mixed_precision="${MIXED_PRECISION}" \
        training/train.py \
        "${TRAINING_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/train.log"
fi

# ========================
# Post-training
# ========================

echo "Training completed!"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "Logs saved to: ${LOG_DIR}"

# Optional: Run evaluation
# python evaluation/evaluate.py --checkpoint "${CHECKPOINT_DIR}/best_model"

# Optional: Convert to ONNX or other formats
# python scripts/export_model.py --checkpoint "${CHECKPOINT_DIR}/best_model"

