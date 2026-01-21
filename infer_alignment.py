#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Inference script for EEG-Text Alignment.

This script demonstrates how to:
1. Load a trained EEG Projector.
2. Inference on EEG data to produce T5-compatible embeddings (Latents).
3. These latents can be directly used as input to InfinityStar.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from infinity.models.eeg_projector import build_eeg_projector
from infinity.models.custom_encoders import deepnet, eegnet, shallownet, glmnet
from tools.run_infinity import load_tokenizer
from infinity.utils.eeg_utils import compute_raw_stats, load_raw_stats, normalize_raw


class EEGEncoderSystem(nn.Module):
    """
    Combines a raw EEG encoder (DeepNet/EEGNet/etc.) with a Projector (MLP).
    Same class as in train_alignment.py to ensure state_dict compatibility.
    """
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        
    def forward(self, x):
        # x: (B, 1, C, T)
        # Encoder returns (B, Embed_Dim)
        feats = self.encoder(x, return_features=True)
        # Projector returns (B, Seq, T5_Dim)
        out = self.projector(feats)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Infer EEG Alignment")
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained projector checkpoint (e.g., checkpoint_best.pth)')
    
    # Input Data (Mutually exclusive preferably, or prioritize raw)
    parser.add_argument('--eeg_tokenizer_path', type=str, default=None,
                        help='Path to EEG tokenizer output file (.pt) - Legacy Mode')
    parser.add_argument('--raw_eeg_path', type=str, default=None,
                        help='Path to raw EEG file (.npy) - Raw Mode')
    
    parser.add_argument('--text_encoder_ckpt', type=str, default='./checkpoints/text_encoder/flan-t5-xl-official/',
                        help='Path to T5 checkpoint (optional, only for verification)')
    parser.add_argument('--output_path', type=str, default='eeg_latents.pt',
                        help='Where to save the inferred latents')
    
    # Model config (Must match training!)
    parser.add_argument('--encoder_model', type=str, default='none', 
                        choices=['deepnet', 'eegnet', 'shallownet', 'glmnet', 'none'],
                        help='Type of EEG encoder for raw data. Use "none" for tokenizer input.')
    parser.add_argument('--eeg_dim', type=int, default=14880)
    parser.add_argument('--eeg_seq_len', type=int, default=64)
    parser.add_argument('--eeg_hidden_dim', type=int, default=4096)
    parser.add_argument('--eeg_num_layers', type=int, default=2)
    parser.add_argument('--projector_type', type=str, default='mlp')
    
    parser.add_argument('--verify_prompt', type=str, default=None,
                        help='If provided, computes similarity with this text prompt to verify alignment')
    
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip normalization for raw EEG (not recommended unless data is already normalized)')

    parser.add_argument('--raw_stats_path', type=str, default=None,
                        help='Path to saved raw EEG stats (.npz) from training (mean/std)')
    
    return parser.parse_args()


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate args
    if args.raw_eeg_path is None and args.eeg_tokenizer_path is None:
        raise ValueError("Must provide either --raw_eeg_path or --eeg_tokenizer_path")

    # 1. Build Model (Encoder + Projector OR Projector only)
    print("Building Model...")
    
    if args.raw_eeg_path and args.encoder_model != 'none':
        # --- Raw EEG Mode ---
        print(f"Using Raw EEG Mode with {args.encoder_model}...")
        
        # Load a sample to determine dimensions C and T
        print(f"Peeking at {args.raw_eeg_path} for shapes...")
        # Assuming .npy is (N, C, T) or (N, 1, C, T)
        raw_data_sample = np.load(args.raw_eeg_path, mmap_mode='r')
        
        # Determine shapes
        if raw_data_sample.ndim == 3:
            # (N, C, T)
            N, C_dim, T_dim = raw_data_sample.shape
        elif raw_data_sample.ndim == 4:
            # (N, 1, C, T)
            N, _, C_dim, T_dim = raw_data_sample.shape
        elif raw_data_sample.ndim > 4:
             # Handle multi-dim case like (7, 40, 5, 3, 62, 200) -> flatten to (N, C, T)
             # Assume last two dimensions are always (Channels, Time)
             C_dim, T_dim = raw_data_sample.shape[-2:]
             N = int(np.prod(raw_data_sample.shape[:-2]))
             print(f"Flattening multi-dim input from {raw_data_sample.shape} to ({N}, {C_dim}, {T_dim})")
        else:
            raise ValueError(f"Unexpected raw data shape: {raw_data_sample.shape}")
            
        print(f"Detected Raw EEG Shape: Samples={N}, Channels={C_dim}, Time={T_dim}")
        
        # Build Encoder
        if args.encoder_model == 'deepnet':
            encoder = deepnet(out_dim=1, C=C_dim, T=T_dim)
        elif args.encoder_model == 'eegnet':
            encoder = eegnet(out_dim=1, C=C_dim, T=T_dim)
        elif args.encoder_model == 'shallownet':
            encoder = shallownet(out_dim=1, C=C_dim, T=T_dim)
        elif args.encoder_model == 'glmnet':
            # Logic from train_alignment.py
            if C_dim >= 62:
                occ_idx = list(range(50, 62))
            else:
                occ_idx = list(range(C_dim))
            encoder = glmnet(occipital_idx=occ_idx, C=C_dim, T=T_dim, out_dim=1)
        else:
            raise ValueError(f"Unknown encoder: {args.encoder_model}")
            
        enc_out_dim = encoder.out_features
        print(f"Encoder output dimension: {enc_out_dim}")
        
        # Build Projector
        projector_head = build_eeg_projector(
            projector_type=args.projector_type,
            eeg_dim=enc_out_dim,
            t5_dim=2048,
            seq_len=args.eeg_seq_len,
            hidden_dim=args.eeg_hidden_dim,
            num_layers=args.eeg_num_layers
        )
        
        # Combine
        model = EEGEncoderSystem(encoder, projector_head)
        
    else:
        # --- Tokenizer/Legacy Mode ---
        print("Using Tokenizer/Legacy Mode...")
        model = build_eeg_projector(
            projector_type=args.projector_type,
            eeg_dim=args.eeg_dim,
            t5_dim=2048,
            seq_len=args.eeg_seq_len,
            hidden_dim=args.eeg_hidden_dim,
            num_layers=args.eeg_num_layers
        )

    # 2. Load Weights
    print(f"Loading weights from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    # Check if checkpoint has 'projector' key (standard from train_alignment.py)
    if 'projector' in checkpoint:
        state_dict = checkpoint['projector']
    else:
        state_dict = checkpoint # Fallback if direct state dict
        
    # Handle key prefix mismatch if any (though EEGEncoderSystem structure matches train)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
        
    model.to(device)
    model.eval()
    
    # 3. Load EEG Data
    if args.raw_eeg_path and args.encoder_model != 'none':
        # Load Raw
        print(f"Loading raw EEG data from {args.raw_eeg_path}...")
        eeg_data_np = np.load(args.raw_eeg_path)
        
        # Flatten if multi-dim
        if eeg_data_np.ndim > 3:
             # Keep last 2 dims (C, T) and flatten the rest
             C, T = eeg_data_np.shape[-2:]
             eeg_data_np = eeg_data_np.reshape(-1, C, T)
        
        # Normalization
        if not args.no_normalize:
            if args.raw_stats_path:
                print(f"Loading raw stats from {args.raw_stats_path}...")
                mean, std = load_raw_stats(args.raw_stats_path)
                print(f"Loaded stats - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")
            else:
                print("Computing statistics for normalization (matching training logic)...")
                mean, std = compute_raw_stats(eeg_data_np)
                print(f"Stats - Mean: {mean.mean():.4f}, Std: {std.mean():.4f}")
            eeg_data_np = normalize_raw(eeg_data_np, mean, std)
        else:
            print("Skipping normalization as requested.")
            
        eeg_input = torch.from_numpy(eeg_data_np).float()
        
        # Ensure shape (N, 1, C, T)
        if eeg_input.ndim == 3:
            eeg_input = eeg_input.unsqueeze(1) # (N, 1, C, T)
            
    else:
        # Load Tokenized
        print(f"Loading EEG data from {args.eeg_tokenizer_path}...")
        eeg_data = torch.load(args.eeg_tokenizer_path, map_location='cpu')
        
        # Handle both formats
        if 'quant_per_window_concat' in eeg_data:
            eeg_input = eeg_data['quant_per_window_concat'] # (N, 14880)
        else:
            eeg_input = eeg_data['quant_per_window'] # (N, 2, 7440)
            eeg_input = eeg_input.reshape(eeg_input.shape[0], -1)
            
    eeg_input = eeg_input.to(device)
    print(f"EEG Input shape: {eeg_input.shape}")
    
    # 4. Inference
    print("Running inference...")
    batch_size = 32
    latents_list = []
    
    with torch.no_grad():
        for i in range(0, len(eeg_input), batch_size):
            batch = eeg_input[i:i+batch_size]
            batch_latents = model(batch) # (B, 64, 2048)
            latents_list.append(batch_latents.cpu())
            
    latents = torch.cat(latents_list, dim=0)
    print(f"Generated Latents shape: {latents.shape}")
    
    # 5. Save Results
    torch.save(latents, args.output_path)
    print(f"Saved latents to {args.output_path}")
    print("These latents can now be passed to InfinityStar as 'kv_compact' (reshaped) or conditioning embeddings.")
    
    # 6. Verification (Optional)
    if args.verify_prompt:
        print(f"\nVerifying alignment with prompt: '{args.verify_prompt}'")
        tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        text_encoder.to(device)
        text_encoder.eval()
        
        with torch.no_grad():
            text_tokens = tokenizer(
                [args.verify_prompt],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            text_outputs = text_encoder(
                input_ids=text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask
            )
            text_emb = text_outputs['last_hidden_state'] # (1, L, 2048)
            text_mask = text_tokens.attention_mask.unsqueeze(-1)
            
            # Compute global vectors
            text_gap = (text_emb * text_mask).sum(dim=1) / text_mask.sum(dim=1)
            eeg_gap = latents.mean(dim=1).to(device)
            
            # Cosine similarity
            sim = F.cosine_similarity(eeg_gap, text_gap, dim=-1)
            
            print(f"Average Cosine Similarity with prompt: {sim.mean().item():.4f}")
            print("(Note: This is just a sanity check. High similarity on unseen data requires valid generalization.)")


if __name__ == "__main__":
    args = parse_args()
    infer(args)
