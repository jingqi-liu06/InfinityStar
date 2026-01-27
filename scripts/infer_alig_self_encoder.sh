export CUDA_VISIBLE_DEVICES=3
python infer_alignment.py \
    --raw_eeg_path "./eeg_data/Preprocessing/Segmented_1000ms_sw/sub6.npy" \
    --encoder_model "glmnet" \
    --checkpoint_path "./checkpoints_alignment/checkpoint_best.pth" \
    --output_path "./eeg_glmnet_results/sub6_latents.pt" \
    --raw_stats_path "./checkpoints_alignment/sub6_stats.npz"