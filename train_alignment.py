#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Training script for EEG-Text Alignment (Stage 1) - Optimized.

This script trains the EEG Projector to align EEG features with T5 text embeddings.
It uses a combination of MSE Loss (reconstruction) and Contrastive Loss (InfoNCE).

Optimizations:
1. Pre-computes T5 embeddings for all captions to avoid re-running T5 during training.
2. Pre-loads EEG data to GPU memory.
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from infinity.models.eeg_projector import build_eeg_projector
from infinity.dataset.dataset_alignment import EEGAlignmentDataset, collate_alignment_batch
from tools.run_infinity import load_tokenizer


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """Calculate ground-truth and cache if enabled"""
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """Compute logits between image and text features"""
        # For single-GPU training (world_size=1), simplified version
        if self.world_size == 1:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        else:
            # Distributed training would need gather_features implementation
            raise NotImplementedError("Distributed training not implemented yet")

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        """
        Args:
            image_features: (B, D) normalized features (EEG in our case)
            text_features: (B, D) normalized features
            logit_scale: scalar or learnable parameter for scaling
            logit_bias: optional bias term
            output_dict: whether to return dict or scalar
        """
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG-Text Alignment")
    
    # Data paths
    parser.add_argument('--eeg_tokenizer_path', type=str, required=True,
                        help='Path to EEG tokenizer output .pt file')
    parser.add_argument('--video_gt_root', type=str, required=True,
                        help='Root directory for videos (for alignment verification)')
    parser.add_argument('--caption_root', type=str, required=True,
                        help='Root directory for captions')
    parser.add_argument('--text_encoder_ckpt', type=str, required=True,
                        help='Path to T5 text encoder checkpoint')
    
    # Projector config
    parser.add_argument('--eeg_dim', type=int, default=14880, help='Input EEG dimension')
    parser.add_argument('--eeg_seq_len', type=int, default=64, help='Output sequence length')
    parser.add_argument('--eeg_hidden_dim', type=int, default=4096, help='Hidden dimension')
    parser.add_argument('--eeg_num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--eeg_projector_type', type=str, default='mlp', choices=['mlp', 'cross_attention'])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training config
    parser.add_argument('--output_dir', type=str, default='./checkpoints/alignment', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--preload_data', action='store_true', default=True, help='Preload EEG data to GPU')
    
    # Loss config
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--lambda_nce', type=float, default=1.0, help='Weight for NCE loss')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='Weight for MSE loss')
    
    # Wandb config
    parser.add_argument('--project_name', type=str, default='EEG-Alignment', help='Wandb project name')
    parser.add_argument('--exp_name', type=str, default='mlp_align', help='Wandb experiment name')
    
    return parser.parse_args()


def precompute_text_embeddings(dataset, tokenizer, text_encoder, device, batch_size=32):
    """
    Pre-compute T5 embeddings for all unique captions in the dataset.
    """
    print("Pre-computing text embeddings for caching...")
    unique_captions = sorted(list(set(dataset.captions)))
    # Filter out empty strings
    unique_captions = [c for c in unique_captions if c.strip()]
    
    cache = {}
    
    # Process in batches
    for i in tqdm(range(0, len(unique_captions), batch_size), desc="Caching Text Embeddings"):
        batch_captions = unique_captions[i:i+batch_size]
        
        with torch.no_grad():
            text_tokens = tokenizer(
                batch_captions,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            text_outputs = text_encoder(
                input_ids=text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask
            )
            # We need to store both embeddings and mask
            # Move to CPU to save GPU memory if dataset is huge, 
            # but for 1400 samples, keeping on GPU is fine and faster.
            # Let's keep on GPU for max speed since we have few captions.
            text_emb = text_outputs['last_hidden_state'] # (B, L, 2048)
            text_mask = text_tokens.attention_mask.unsqueeze(-1) # (B, L, 1)
            
            for j, caption in enumerate(batch_captions):
                cache[caption] = (text_emb[j], text_mask[j])
                
    print(f"Cached embeddings for {len(cache)} unique captions.")
    return cache


def train(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Wandb
    wandb.init(
        project=args.project_name,
        name=args.exp_name,
        config=vars(args)
    )
    
    # Load Text Encoder (Frozen) - Only for pre-computation
    print("Loading Text Encoder...")
    tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    text_encoder.to(device)
    text_encoder.eval()
    
    # Initial dataset load (to get captions)
    print("Loading Dataset Metadata...")
    # Temporarily create dataset without preload to scan captions
    temp_dataset = EEGAlignmentDataset(
        eeg_tokenizer_path=args.eeg_tokenizer_path,
        caption_root=args.caption_root,
        video_root=args.video_gt_root,
        split="train",
        seed=args.seed,
        preload_to_gpu=False
    )
    
    # Pre-compute text embeddings
    cached_embeddings = precompute_text_embeddings(
        temp_dataset, tokenizer, text_encoder, device, batch_size=args.batch_size
    )
    
    # Unload text encoder to free up GPU memory
    del text_encoder
    del tokenizer
    torch.cuda.empty_cache()
    print("Text Encoder unloaded to save memory.")
    
    # Re-initialize dataset with cache and preloading
    print("Initializing Optimized Dataset...")
    dataset = EEGAlignmentDataset(
        eeg_tokenizer_path=args.eeg_tokenizer_path,
        caption_root=args.caption_root,
        video_root=args.video_gt_root,
        split="train",
        seed=args.seed,
        preload_to_gpu=args.preload_data,
        device=device,
        cached_text_embeddings=cached_embeddings
    )
    
    # DataLoader
    # Note: num_workers=0 is usually faster if data is already on GPU
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.preload_data else args.num_workers,
        collate_fn=collate_alignment_batch,
        drop_last=True
    )
    
    # Build EEG Projector
    print("Building EEG Projector...")
    projector = build_eeg_projector(
        projector_type=args.eeg_projector_type,
        eeg_dim=args.eeg_dim,
        t5_dim=2048,  # Flan-T5-XL dimension
        seq_len=args.eeg_seq_len,
        hidden_dim=args.eeg_hidden_dim,
        num_layers=args.eeg_num_layers,
        dropout=args.dropout
    )
    projector.to(device)
    
    # Initialize CLIP-style Contrastive Loss
    print("Initializing CLIP Loss...")
    clip_loss_fn = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False
    )
    
    # Learnable temperature parameter (logit_scale)
    # Initialize with 1/temperature, as in CLIP
    logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / args.temperature)))
    logit_scale.to(device)
    
    # Optimizer (include logit_scale)
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + [logit_scale],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f"Start training for {args.epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        projector.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_nce = 0.0
        steps = 0
        
        start_time = time.time()
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Data is likely already on GPU if preloaded
            eeg_features = batch['eeg_features'].to(device).float() # (B, 14880)
            
            # 1. Forward Pass Projector
            eeg_emb = projector(eeg_features) # (B, Seq, 2048)
            
            # 2. Get Cached Text Embeddings
            # They should be in batch dict if everything went right
            if 'text_embeddings' not in batch:
                # Skip batch if no embeddings (e.g. all empty captions)
                continue
                
            text_emb = batch['text_embeddings'].to(device) # (B, L, 2048)
            text_mask = batch['text_masks'].to(device)     # (B, L, 1)
            
            # 3. Compute Global Average Pooling for Alignment
            # EEG GAP: Mean over sequence
            eeg_gap = eeg_emb.mean(dim=1) # (B, 2048)
            
            # Text GAP: Mean over valid tokens
            text_sum = (text_emb * text_mask).sum(dim=1)
            text_lens = text_mask.sum(dim=1).clamp(min=1.0)
            text_gap = text_sum / text_lens # (B, 2048)
            
            # 4. Compute Losses
            # MSE Loss (Reconstruction)
            mse_loss = F.mse_loss(eeg_gap, text_gap)
            
            # CLIP-style Contrastive Loss
            # Normalize for cosine similarity
            eeg_norm = F.normalize(eeg_gap, p=2, dim=-1)
            text_norm = F.normalize(text_gap, p=2, dim=-1)
            # Use learnable logit_scale (exp to ensure positive)
            nce_loss = clip_loss_fn(
                image_features=eeg_norm,
                text_features=text_norm,
                logit_scale=logit_scale.exp(),
                output_dict=False
            )
            
            # Total Loss
            loss = (args.lambda_mse * mse_loss) + (args.lambda_nce * nce_loss)
            
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_nce += nce_loss.item()
            steps += 1
            
            if steps % args.log_freq == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Step {steps}: "
                      f"Loss={loss.item():.4f} (MSE={mse_loss.item():.4f}, NCE={nce_loss.item():.4f})")
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/nce_loss": nce_loss.item(),
                    "train/epoch": epoch + 1,
                    "train/step": epoch * len(dataloader) + steps
                })
        
        # End of epoch stats
        if steps > 0:
            avg_loss = epoch_loss / steps
            avg_mse = epoch_mse / steps
            avg_nce = epoch_nce / steps
            duration = time.time() - start_time
            
            print(f"Epoch {epoch+1} Done. Time: {duration:.2f}s")
            print(f"Avg Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, NCE: {avg_nce:.4f})")
            
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_mse": avg_mse,
                "epoch/avg_nce": avg_nce,
            })
            
            # Save checkpoints
            save_path = os.path.join(args.output_dir, "checkpoint_latest.pth")
            torch.save({
                'epoch': epoch,
                'projector': projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'logit_scale': logit_scale.data,
                'loss': avg_loss
            }, save_path)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
                torch.save({
                    'epoch': epoch,
                    'projector': projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'logit_scale': logit_scale.data,
                    'loss': avg_loss
                }, best_path)
                print(f"New best model saved to {best_path}")
                
            if (epoch + 1) % args.save_freq == 0:
                periodic_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'projector': projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'logit_scale': logit_scale.data,
                    'loss': avg_loss
                }, periodic_path)
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
