#!/usr/bin/env bash
set -x

# ============================================================================
# EEG-Text Alignment Training Script (Stage 1)
# ============================================================================
# This script trains the EEG Projector to align EEG embeddings with Text embeddings.
# It does NOT involve video generation or loading the Infinity model.
# ============================================================================

# Settings
export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Set GPU ID

# Data Configuration
sub_id="sub9"
data_root="./eeg_data"
output_root="./checkpoints/alignment"

# Paths
eeg_tokenizer_path="./eeg_outputs/${sub_id}_quant.pt"
video_gt_root="${data_root}/Video/Video_sections_original_resolution"
caption_root="${data_root}/Video/BLIP-caption"
text_encoder_ckpt="./checkpoints/text_encoder/flan-t5-xl-official/"

# Projector Configuration
eeg_dim=14880
eeg_seq_len=64
eeg_hidden_dim=4096
eeg_num_layers=2
projector_type="mlp"

# Training Configuration
batch_size=32
epochs=200
lr=1e-3
log_freq=10
save_freq=50

# Output Directory
exp_name="${sub_id}_mlp_align"
output_dir="${output_root}/${exp_name}"

echo "Starting Alignment Training for ${sub_id}..."
echo "Output Directory: ${output_dir}"

python train_alignment.py \
    --eeg_tokenizer_path "${eeg_tokenizer_path}" \
    --video_gt_root "${video_gt_root}" \
    --caption_root "${caption_root}" \
    --text_encoder_ckpt "${text_encoder_ckpt}" \
    --eeg_dim ${eeg_dim} \
    --eeg_seq_len ${eeg_seq_len} \
    --eeg_hidden_dim ${eeg_hidden_dim} \
    --eeg_num_layers ${eeg_num_layers} \
    --eeg_projector_type "${projector_type}" \
    --output_dir "${output_dir}" \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --lr ${lr} \
    --log_freq ${log_freq} \
    --save_freq ${save_freq} \
    --lambda_mse 1.0 \
    --lambda_nce 1.0 \
    --temperature 0.07 \
    --project_name "EEG-Alignment" \
    --exp_name "${exp_name}"

echo "Training completed."

