# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Inference script for EEG-to-Video generation.

This script loads a trained EEG Projector and LoRA-adapted Infinity model
to generate videos from EEG signals.
"""

import sys
import json
import os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import argparse
from PIL import Image

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.models.eeg_projector import build_eeg_projector
from infinity.models.lora import load_lora_state_dict, LoRAConfig, apply_lora_to_model
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args


class EEGInferencePipe:
    """Inference pipeline for EEG-to-Video generation."""
    
    def __init__(self, args, eeg_checkpoint_path: str):
        """
        Initialize the EEG inference pipeline.
        
        Args:
            args: Inference arguments
            eeg_checkpoint_path: Path to the EEG-LoRA checkpoint
        """
        self.args = args
        
        # Load checkpoint
        print(f"Loading EEG checkpoint from {eeg_checkpoint_path}...")
        checkpoint = torch.load(eeg_checkpoint_path, map_location='cpu')
        
        # Get EEG projector config from checkpoint
        ckpt_args = checkpoint.get('args', {})
        self.eeg_dim = ckpt_args.get('eeg_dim', 14880)
        self.eeg_seq_len = ckpt_args.get('eeg_seq_len', 64)
        self.eeg_hidden_dim = ckpt_args.get('eeg_hidden_dim', 4096)
        self.eeg_num_layers = ckpt_args.get('eeg_num_layers', 2)
        self.eeg_projector_type = ckpt_args.get('eeg_projector_type', 'mlp')
        
        # Load text encoder (still needed for text_proj layers)
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        
        # Load VAE
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to('cuda')
        self.vae.eval()
        
        # Load Infinity transformer
        self.infinity = load_transformer(self.vae, args)
        
        # Apply LoRA
        lora_rank = ckpt_args.get('lora_rank', 32)
        lora_alpha = ckpt_args.get('lora_alpha', 64.0)
        lora_dropout = ckpt_args.get('lora_dropout', 0.05)
        lora_target_modules = ckpt_args.get('lora_target_modules', 'q_proj,k_proj,v_proj,o_proj').split(',')
        
        lora_config = LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        
        print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
        self.infinity, _ = apply_lora_to_model(self.infinity, lora_config, verbose=True)
        
        # Load LoRA weights
        if 'lora' in checkpoint:
            print("Loading LoRA weights...")
            load_lora_state_dict(self.infinity, checkpoint['lora'])
        
        self.infinity.eval()
        
        # Build EEG projector
        print(f"Building EEG projector: {self.eeg_projector_type}, dim={self.eeg_dim}, seq_len={self.eeg_seq_len}")
        self.eeg_projector = build_eeg_projector(
            projector_type=self.eeg_projector_type,
            eeg_dim=self.eeg_dim,
            t5_dim=2048,
            seq_len=self.eeg_seq_len,
            hidden_dim=self.eeg_hidden_dim,
            num_layers=self.eeg_num_layers,
        )
        
        # Load EEG projector weights
        if 'eeg_projector' in checkpoint:
            print("Loading EEG projector weights...")
            self.eeg_projector.load_state_dict(checkpoint['eeg_projector'])
        
        self.eeg_projector = self.eeg_projector.to('cuda')
        self.eeg_projector.eval()
        
        # Self correction module
        self.self_correction = SelfCorrection(self.vae, args)
        
        # Get encode/decode functions
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = \
            get_encode_decode_func(args.dynamic_scale_schedule)
        
        print("EEG Inference Pipeline initialized!")
    
    def prepare_eeg_condition(
        self,
        eeg_features: torch.Tensor,
        batch_size: int = 1,
    ):
        """
        Prepare EEG features as conditioning for video generation.
        
        Args:
            eeg_features: EEG tokenizer output, shape (batch, eeg_dim) or (batch, 2, 7440)
            batch_size: Batch size
            
        Returns:
            Text condition tuple compatible with Infinity forward
        """
        device = next(self.eeg_projector.parameters()).device
        
        # Ensure correct shape
        if eeg_features.dim() == 3:
            # (batch, 2, 7440) -> (batch, 14880)
            eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)
        
        eeg_features = eeg_features.to(device)
        
        # Project EEG to T5 embedding space
        with torch.no_grad():
            eeg_embeddings = self.eeg_projector(eeg_features)  # (batch, seq_len, 2048)
        
        # Get sequence info
        seq_len = self.eeg_seq_len
        lens = [seq_len] * batch_size
        cu_seqlens_k = F.pad(
            torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device).cumsum_(0),
            (1, 0)
        )
        
        # Apply text_norm and text_proj (matching Infinity's text processing)
        kv_compact = self.infinity.text_norm(eeg_embeddings.reshape(-1, 2048))
        kv_compact = self.infinity.text_proj(kv_compact).contiguous()
        
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, seq_len)
        
        return text_cond_tuple
    
    @torch.no_grad()
    def generate_video(
        self,
        eeg_features: torch.Tensor,
        duration: float = 5.0,
        seed: int = 42,
        cfg: float = 34.0,
        tau_image: float = 1.0,
        tau_video: float = 0.4,
        negative_prompt: str = "",
    ):
        """
        Generate video from EEG features.
        
        Args:
            eeg_features: EEG tokenizer output
            duration: Video duration in seconds
            seed: Random seed
            cfg: Classifier-free guidance scale
            tau_image: Temperature for image scales
            tau_video: Temperature for video scales
            negative_prompt: Negative prompt for guidance
            
        Returns:
            Generated video tensor
        """
        args = self.args
        num_frames = int(duration * 16) + 1
        
        # Get scale schedule
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
            args.dynamic_scale_schedule, args.video_frames
        )
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
        
        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        args.first_full_spatial_size_scale_index = first_full_spatial_size_scale_index
        args.tower_split_index = first_full_spatial_size_scale_index + 1
        
        context_info = self.get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args)
        
        # Temperature schedule
        tau = [tau_image] * args.tower_split_index + [tau_video] * (len(scale_schedule) - args.tower_split_index)
        
        # CFG list
        if isinstance(cfg, (int, float)):
            cfg_list = [cfg] * len(scale_schedule)
        else:
            cfg_list = cfg
        
        # Prepare EEG condition
        text_cond_tuple = self.prepare_eeg_condition(eeg_features, batch_size=1)
        
        # Prepare negative condition (using cfg_uncond)
        negative_label_B_or_BLT = None
        if negative_prompt:
            # Use text encoder for negative prompt
            tokens = self.text_tokenizer(
                text=[negative_prompt],
                max_length=self.text_tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = tokens.input_ids.cuda()
            mask = tokens.attention_mask.cuda()
            neg_text_features = self.text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
            neg_lens = mask.sum(dim=-1).tolist()
            neg_cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
            neg_Ltext = max(neg_lens)
            neg_kv_compact = []
            for i, (len_i, feat_i) in enumerate(zip(neg_lens, neg_text_features.unbind(0))):
                neg_kv_compact.append(feat_i[:len_i])
            neg_kv_compact = torch.cat(neg_kv_compact, dim=0)
            negative_label_B_or_BLT = (neg_kv_compact, neg_lens, neg_cu_seqlens_k, neg_Ltext)
        
        # Set random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        print(f"Generating {duration}s video with {len(scale_schedule)} scales...")
        start_time = time.time()
        
        # Generate video
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            idx_Bl_list, generated_video = self.infinity.autoregressive_infer(
                vae=self.vae,
                scale_schedule=scale_schedule,
                label_B_or_BLT=text_cond_tuple,
                B=1,
                negative_label_B_or_BLT=negative_label_B_or_BLT,
                g_seed=seed,
                cfg_list=cfg_list,
                tau_list=tau,
                args=args,
                get_visual_rope_embeds=self.get_visual_rope_embeds,
                context_info=context_info,
            )
        
        elapsed_time = time.time() - start_time
        print(f"Generation completed in {elapsed_time:.2f}s")
        
        return generated_video


def load_eeg_tokenizer_outputs(eeg_path: str, use_concat: bool = True):
    """
    Load EEG tokenizer outputs from file.
    
    Args:
        eeg_path: Path to .pt file with EEG tokenizer outputs
        use_concat: Whether to use concatenated features
        
    Returns:
        EEG features tensor
    """
    print(f"Loading EEG features from {eeg_path}...")
    data = torch.load(eeg_path, weights_only=True)
    
    if use_concat and 'quant_per_window_concat' in data:
        features = data['quant_per_window_concat']
    else:
        features = data['quant_per_window']
    
    print(f"Loaded {len(features)} EEG samples, shape: {features.shape}")
    return features


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description='EEG-to-Video Inference')
    
    # Model paths
    parser.add_argument('--checkpoint_dir', type=str, default='./',
                        help='Directory containing model checkpoints')
    parser.add_argument('--eeg_checkpoint', type=str, required=True,
                        help='Path to EEG-LoRA checkpoint')
    
    # EEG input
    parser.add_argument('--eeg_path', type=str, 
                        default='./eeg_outputs/brain_tokenizer_quant_per_window.pt',
                        help='Path to EEG tokenizer outputs')
    parser.add_argument('--eeg_idx', type=int, default=0,
                        help='Index of EEG sample to use')
    parser.add_argument('--use_concat_eeg', type=int, default=1,
                        help='Use concatenated EEG features')
    
    # Generation config
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Video duration in seconds (5 or 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cfg', type=float, default=34.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--tau_image', type=float, default=1.0,
                        help='Temperature for image scales')
    parser.add_argument('--tau_video', type=float, default=0.4,
                        help='Temperature for video scales')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output/eeg_videos',
                        help='Output directory for generated videos')
    
    cli_args = parser.parse_args()
    
    # Build inference args
    args = Args()
    args.pn = '0.40M'
    args.fps = 16
    args.video_frames = int(cli_args.duration * 16) + 1
    args.model_path = os.path.join(cli_args.checkpoint_dir, 'infinitystar_8b_480p_weights')
    args.checkpoint_type = 'torch_shard'
    args.vae_path = os.path.join(cli_args.checkpoint_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt = os.path.join(cli_args.checkpoint_dir, 'text_encoder/flan-t5-xl-official/')
    args.videovae = 10
    args.model_type = 'infinity_qwen8b'
    args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = cli_args.cfg
    args.tau_image = cli_args.tau_image
    args.tau_video = cli_args.tau_video
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    args.max_repeat_times = 10000
    
    # Initialize pipeline
    pipe = EEGInferencePipe(args, cli_args.eeg_checkpoint)
    
    # Load EEG features
    eeg_features = load_eeg_tokenizer_outputs(cli_args.eeg_path, bool(cli_args.use_concat_eeg))
    
    # Get specific sample
    eeg_sample = eeg_features[cli_args.eeg_idx].unsqueeze(0)  # Add batch dimension
    print(f"Using EEG sample {cli_args.eeg_idx}, shape: {eeg_sample.shape}")
    
    # Generate video
    generated_video = pipe.generate_video(
        eeg_features=eeg_sample,
        duration=cli_args.duration,
        seed=cli_args.seed,
        cfg=cli_args.cfg,
        tau_image=cli_args.tau_image,
        tau_video=cli_args.tau_video,
    )
    
    # Save video
    os.makedirs(cli_args.output_dir, exist_ok=True)
    output_path = os.path.join(
        cli_args.output_dir,
        f'eeg_{cli_args.eeg_idx}_seed_{cli_args.seed}_{cli_args.duration}s.mp4'
    )
    
    save_video(generated_video, output_path, fps=16)
    print(f"Saved video to {output_path}")


def batch_inference(
    pipe: EEGInferencePipe,
    eeg_features: torch.Tensor,
    output_dir: str,
    start_idx: int = 0,
    end_idx: int = -1,
    duration: float = 5.0,
    seed: int = 42,
    cfg: float = 34.0,
):
    """
    Batch inference for multiple EEG samples.
    
    Args:
        pipe: EEG inference pipeline
        eeg_features: All EEG features
        output_dir: Output directory
        start_idx: Starting index
        end_idx: Ending index (-1 for all)
        duration: Video duration
        seed: Random seed
        cfg: CFG scale
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if end_idx == -1:
        end_idx = len(eeg_features)
    
    print(f"Generating videos for samples {start_idx} to {end_idx}...")
    
    for idx in tqdm(range(start_idx, end_idx)):
        eeg_sample = eeg_features[idx].unsqueeze(0)
        
        try:
            generated_video = pipe.generate_video(
                eeg_features=eeg_sample,
                duration=duration,
                seed=seed,
                cfg=cfg,
            )
            
            output_path = os.path.join(output_dir, f'eeg_{idx:04d}.mp4')
            save_video(generated_video, output_path, fps=16)
            
        except Exception as e:
            print(f"Error generating video for sample {idx}: {e}")
            continue
    
    print(f"Batch inference completed! Videos saved to {output_dir}")


if __name__ == '__main__':
    main()

