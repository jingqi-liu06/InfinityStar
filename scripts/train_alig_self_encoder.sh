# 设置显卡 ID
# export CUDA_VISIBLE_DEVICES=6
# # 设置 WANDB 目录为用户有权限的路径
# export WANDB_DIR="./wandb_logs"

/baai-cwm-vepfs/cwm/jingqi.liu/anaconda3/envs/infinitystar/bin/python train_alignment.py \
    --raw_eeg_path "./eeg_data/Preprocessing/Segmented_1000ms_sw/sub8.npy" \
    --encoder_model "glmnet" \
    --video_gt_root "./eeg_data/Video/Video_sections_original_resolution" \
    --caption_root "./eeg_data/Video/BLIP-caption" \
    --text_encoder_ckpt "./checkpoints/text_encoder/flan-t5-xl-official/" \
    --output_dir "./checkpoints_alignment/sub8_glmnet_raw" \
    --batch_size 192 \
    --epochs 200 \
    --lr 1e-4 \
    --project_name "EEG-Alignment-GLMNet" \
    --exp_name "sub8"
