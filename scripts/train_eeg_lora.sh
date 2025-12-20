#!/usr/bin/env bash
set -x

# ============================================================================
# EEG-to-Video LoRA Fine-tuning Script
# ============================================================================
# This script trains an EEG Projector and LoRA adapters on InfinityStar
# for EEG-conditioned video generation.
# ============================================================================

# NCCL settings (adjust based on your cluster)
unset NCCL_NET_PLUGIN
unset NCCL_FASTRAK_ENABLE
unset NCCL_FASTRAK_USE_SNAP
unset NCCL_FASTRAK_NUM_FLOWS
unset NCCL_FASTRAK_*
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_NET_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_PLUGIN=NONE
export NCCL_NVLS_ENABLE=0

# Arnold settings (for distributed training)
unset ARNOLD_WORKER_0_HOST
TORCHRUN_RDZV_READ_TIMEOUT=${TORCHRUN_RDZV_READ_TIMEOUT:-600}
ARNOLD_ID=${ARNOLD_ID:-0}
ARNOLD_WORKER_0_HOST=${ARNOLD_WORKER_0_HOST:-'localhost'}
ARNOLD_WORKER_0_PORT=${ARNOLD_WORKER_0_PORT:-'9591'}
PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-1}
ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-1}

# Torch settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=infinity/models/:$PYTHONPATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=8

# ============================================================================
# Experiment Configuration
# ============================================================================

# Experiment name
exp_name="eeg_to_video_lora"

# Paths
checkpoints_dir="./checkpoints/"
bed_path="./eeg_lora_checkpoints/"
LOCAL_OUT="eeg_lora_checkpoints"
mkdir -p $LOCAL_OUT

# Model paths
t5_path="${checkpoints_dir}text_encoder/flan-t5-xl-official"
vae_path="${checkpoints_dir}infinitystar_videovae.pth"
resume_path="${checkpoints_dir}infinitystar_8b_720p_weights"

# EEG data paths
eeg_tokenizer_path="./eeg_outputs/brain_tokenizer_quant_per_window.pt"
video_gt_root="./eeg_data/Video/Video_sections_original_resolution"
caption_root="./eeg_data/Video/BLIP-caption"

# Token cache directory
token_cache_dir="./checkpoints/local_cache/cached_visual_tokens_480p"

# Video configuration
vae_type=64
videovae=10
video_fps=16
video_frames=81

# ============================================================================
# EEG Projector Configuration
# ============================================================================
eeg_dim=14880
eeg_seq_len=64
eeg_hidden_dim=4096
eeg_num_layers=2
eeg_projector_type="mlp"
eeg_projector_dropout=0.1
eeg_projector_lr=1e-4

# ============================================================================
# LoRA Configuration
# ============================================================================
lora_rank=32
lora_alpha=64.0
lora_dropout=0.05
lora_target_modules="q_proj,k_proj,v_proj,o_proj"
lora_lr=1e-4

# ============================================================================
# Training Configuration
# ============================================================================
batch_size=1
num_epochs=100
log_freq=10
save_model_iters_freq=500
grad_clip=1.0

# Noise apply strength
noise_apply_strength=()
noise_apply_strength+=($(printf "0.3\n%.0s" {1..200}))
noise_apply_strength_str=$(IFS=,; echo "${noise_apply_strength[*]}")

# Wandb
wandb offline

# ============================================================================
# Launch Training
# ============================================================================

# Debug environment variables for CUDA/C++ crash debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NAN_INF_STACK=1

torchrun --nproc_per_node=$ARNOLD_WORKER_GPU \
    --nnodes=$ARNOLD_WORKER_NUM \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --node_rank=$ARNOLD_ID \
    --master_port=$PORT \
    --rdzv_conf=read_timeout=$TORCHRUN_RDZV_READ_TIMEOUT \
    train_eeg.py \
    --bed=${bed_path} \
    --t5_path=${t5_path} \
    --vae_type=${vae_type} \
    --videovae=${videovae} \
    --vae_path=${vae_path} \
    --token_cache_dir=${token_cache_dir} \
    --pn 0.40M \
    --model=infinity_qwen8b \
    --project_name=eeg_to_video \
    --exp_name=${exp_name} \
    --checkpoint_type='torch' \
    --enable_checkpointing=full-block \
    --video_fps=${video_fps} \
    --video_frames=${video_frames} \
    --dynamic_scale_schedule=infinity_elegant_clip20frames_v2 \
    --mask_type=infinity_elegant_clip20frames_v2 \
    --use_flex_attn=True \
    --sdpa_mem=False \
    --flash=False \
    --use_vae_token_cache=0 \
    --train_with_var_seq_len=1 \
    --image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]' \
    --video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]' \
    --wp_it=0 \
    --use_two_stage_lfq=1 \
    --semantic_scale_dim=16 \
    --detail_scale_min_tokens=350 \
    --semantic_scales=11 \
    --use_feat_proj=2 \
    --train_max_token_len=16384 \
    --noise_apply_strength="$noise_apply_strength_str" \
    --batch_size=${batch_size} \
    --epoch=${num_epochs} \
    --log_freq=${log_freq} \
    --save_model_iters_freq=${save_model_iters_freq} \
    --grad_clip=${grad_clip} \
    --tlr=${lora_lr} \
    --workers=4 \
    --torchshard_resume=${resume_path} \
    --eeg_tokenizer_path=${eeg_tokenizer_path} \
    --video_gt_root=${video_gt_root} \
    --caption_root=${caption_root} \
    --eeg_dim=${eeg_dim} \
    --eeg_seq_len=${eeg_seq_len} \
    --eeg_hidden_dim=${eeg_hidden_dim} \
    --eeg_num_layers=${eeg_num_layers} \
    --eeg_projector_type=${eeg_projector_type} \
    --eeg_projector_dropout=${eeg_projector_dropout} \
    --eeg_projector_lr=${eeg_projector_lr} \
    --lora_rank=${lora_rank} \
    --lora_alpha=${lora_alpha} \
    --lora_dropout=${lora_dropout} \
    --lora_target_modules=${lora_target_modules} \
    --lora_lr=${lora_lr} \
    --use_concat_eeg=1

echo "Training completed!"

