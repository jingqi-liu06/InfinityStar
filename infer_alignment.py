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
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.getcwd())

from infinity.models.eeg_projector import build_eeg_projector
from tools.run_infinity import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Infer EEG Alignment")
    
    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained projector checkpoint (e.g., checkpoint_best.pth)')
    parser.add_argument('--eeg_tokenizer_path', type=str, required=True,
                        help='Path to EEG input file (.pt)')
    parser.add_argument('--text_encoder_ckpt', type=str, default='./checkpoints/text_encoder/flan-t5-xl-official/',
                        help='Path to T5 checkpoint (optional, only for verification)')
    parser.add_argument('--output_path', type=str, default='eeg_latents.pt',
                        help='Where to save the inferred latents')
    
    # Model config (Must match training!)
    parser.add_argument('--eeg_dim', type=int, default=14880)
    parser.add_argument('--eeg_seq_len', type=int, default=64)
    parser.add_argument('--eeg_hidden_dim', type=int, default=4096)
    parser.add_argument('--eeg_num_layers', type=int, default=2)
    parser.add_argument('--projector_type', type=str, default='mlp')
    
    parser.add_argument('--verify_prompt', type=str, default=None,
                        help='If provided, computes similarity with this text prompt to verify alignment')
    
    return parser.parse_args()


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Build Projector
    print("Building Projector...")
    projector = build_eeg_projector(
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
    projector.load_state_dict(checkpoint['projector'])
    projector.to(device)
    projector.eval()
    
    # 3. Load EEG Data
    print(f"Loading EEG data from {args.eeg_tokenizer_path}...")
    eeg_data = torch.load(args.eeg_tokenizer_path, map_location='cpu')
    
    # Handle both formats
    if 'quant_per_window_concat' in eeg_data:
        eeg_input = eeg_data['quant_per_window_concat'] # (N, 14880)
    else:
        eeg_input = eeg_data['quant_per_window'] # (N, 2, 7440)
        eeg_input = eeg_input.reshape(eeg_input.shape[0], -1)
        
    eeg_input = eeg_input.float().to(device)
    print(f"EEG Input shape: {eeg_input.shape}")
    
    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        # Batch inference if data is large, but here we do all at once for simplicity
        # Output shape: (N, 64, 2048)
        latents = projector(eeg_input)
        
    print(f"Generated Latents shape: {latents.shape}")
    
    # 5. Save Results
    torch.save(latents.cpu(), args.output_path)
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
            eeg_gap = latents.mean(dim=1)
            
            # Cosine similarity
            sim = F.cosine_similarity(eeg_gap, text_gap, dim=-1)
            
            print(f"Average Cosine Similarity with prompt: {sim.mean().item():.4f}")
            print("(Note: This is just a sanity check. High similarity on unseen data requires valid generalization.)")


if __name__ == "__main__":
    args = parse_args()
    infer(args)

