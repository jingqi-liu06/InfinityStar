#!/usr/bin/env bash
# ============================================================================
# EEG-to-Video LoRA 训练启动脚本 - 4卡A100配置
# ============================================================================

# 设置4卡GPU
export ARNOLD_WORKER_GPU=4
export ARNOLD_WORKER_NUM=1

# 可选：设置可见的GPU（如果需要指定特定GPU）
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# 进入项目目录
cd /baai-cwm-vepfs/cwm/jingqi.liu/brain_video/codebase/InfinityStar

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate infinitystar

# 运行训练
bash scripts/train_eeg_lora.sh

echo "训练完成！"

