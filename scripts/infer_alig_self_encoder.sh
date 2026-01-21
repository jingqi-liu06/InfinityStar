export CUDA_VISIBLE_DEVICES=2
python infer_alignment.py \
    --raw_eeg_path "./eeg_data/Preprocessing/Segmented_1000ms_sw/sub9.npy" \
    --encoder_model "glmnet" \
    --checkpoint_path "./checkpoints_alignment/sub9_glmnet_raw/checkpoint_best.pth" \
    --output_path "./eeg_glmnet_results/sub9_latents.pt"