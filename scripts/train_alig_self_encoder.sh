# 设置显卡 ID
# export CUDA_VISIBLE_DEVICES=2
# # 设置 WANDB 目录为用户有权限的路径
# export WANDB_DIR="./wandb_logs"

for sub in $(seq 1 20); do
    /baai-cwm-vepfs/cwm/jingqi.liu/anaconda3/envs/infinitystar/bin/python train_alignment.py \
        --raw_eeg_path "./eeg_data/Preprocessing/Segmented_1000ms_sw/sub${sub}.npy" \
        --encoder_model "glmnet" \
        --video_gt_root "./eeg_data/Video/Video_sections_original_resolution" \
        --caption_root "./eeg_data/Video/BLIP-caption" \
        --text_encoder_ckpt "./checkpoints/text_encoder/flan-t5-xl-official/" \
        --output_dir "/baai-cwm-backup/cwm/jingqi.liu/weight/eeg_glm_alignment/checkpoints_alignment/sub${sub}_glmnet_raw" \
        --batch_size 768 \
        --epochs 200 \
        --lr 1e-4 \
        --project_name "EEG-Alignment-GLMNet-reusing_stats" \
        --exp_name "sub${sub}" \
        --raw_stats_path "/baai-cwm-backup/cwm/jingqi.liu/weight/eeg_glm_alignment/checkpoints_alignment/sub${sub}_glmnet_raw/sub${sub}_stats.npz"
done
