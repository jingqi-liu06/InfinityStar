# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Training script for EEG-to-Video generation with LoRA fine-tuning.

This script trains an EEG Projector and LoRA adapters on InfinityStar
for EEG-conditioned video generation.
"""

import gc
import json
import math
import os
import os.path as osp
import random
import sys
import time
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple, Dict

# Fix for DDPOptimizer backend issue in torch.compile
import torch._dynamo
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function
from torch.utils.data import DataLoader
import torch.distributed as tdist

from tools.run_infinity import load_tokenizer  # Import for text encoder loading

import infinity.utils.dist as dist
from infinity.utils.save_and_load import CKPTSaver, omnistoreCheckpoint, auto_resume
from infinity.models.ema import get_ema_model
from infinity.utils import arg_util, misc, wandb_utils
from infinity.trainer import get_trainer
from infinity.schedules import get_encode_decode_func

# Import EEG-specific modules
from infinity.models.eeg_projector import EEGProjector, build_eeg_projector
from infinity.models.self_correction import SelfCorrection
from infinity.models.lora import (
    LoRAConfig, 
    apply_lora_to_model, 
    get_lora_state_dict, 
    load_lora_state_dict,
    set_lora_only_trainable,
    count_lora_parameters,
)
from infinity.dataset.dataset_eeg import (
    EEGVideoDataset,
    collate_eeg_video_batch,
    build_eeg_video_dataset,
)


def add_eeg_args(parser):
    """Add EEG-specific arguments to parser."""
    # EEG data paths
    parser.add_argument('--eeg_tokenizer_path', type=str, 
                        default='./eeg_outputs/brain_tokenizer_quant_per_window.pt',
                        help='Path to EEG tokenizer output')
    parser.add_argument('--video_gt_root', type=str,
                        default='./eeg_data/Video/Video_sections_original_resolution',
                        help='Root directory for GT videos')
    parser.add_argument('--caption_root', type=str,
                        default='./eeg_data/Video/BLIP-caption',
                        help='Root directory for captions')
    
    # EEG Projector config
    parser.add_argument('--eeg_dim', type=int, default=14880,
                        help='EEG tokenizer output dimension')
    parser.add_argument('--eeg_seq_len', type=int, default=64,
                        help='Output sequence length of EEG projector')
    parser.add_argument('--eeg_hidden_dim', type=int, default=4096,
                        help='Hidden dimension of EEG projector')
    parser.add_argument('--eeg_num_layers', type=int, default=2,
                        help='Number of MLP layers in EEG projector')
    parser.add_argument('--eeg_projector_type', type=str, default='mlp',
                        choices=['mlp', 'cross_attention'],
                        help='Type of EEG projector')
    parser.add_argument('--eeg_projector_dropout', type=float, default=0.1,
                        help='Dropout in EEG projector')
    
    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=64.0,
                        help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, 
                        default='q_proj,k_proj,v_proj,o_proj',
                        help='Comma-separated list of modules to apply LoRA')
    
    # Training config
    parser.add_argument('--eeg_projector_lr', type=float, default=1e-4,
                        help='Learning rate for EEG projector')
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                        help='Learning rate for LoRA parameters')
    parser.add_argument('--use_concat_eeg', type=int, default=1,
                        help='Use concatenated EEG features (14880) vs per-window (2, 7440)')
    parser.add_argument('--text_encoder_ckpt', type=str, default='./checkpoints/text_encoder/flan-t5-xl-official/',
                        help='Path to text encoder checkpoint for alignment loss')
    
    return parser


def build_eeg_projector_and_lora(args, gpt_wo_ddp):
    """
    Build EEG projector and apply LoRA to the GPT model.
    
    Args:
        args: Training arguments
        gpt_wo_ddp: GPT model without DDP wrapper
        
    Returns:
        Tuple of (eeg_projector, lora_params_dict)
    """
    # Build EEG Projector
    eeg_projector = build_eeg_projector(
        projector_type=args.eeg_projector_type,
        eeg_dim=args.eeg_dim,
        t5_dim=2048,  # flan-t5-xl dimension
        seq_len=args.eeg_seq_len,
        hidden_dim=args.eeg_hidden_dim,
        num_layers=args.eeg_num_layers,
        dropout=args.eeg_projector_dropout,
    )
    eeg_projector = eeg_projector.to(args.device)
    
    print(f"[EEG Projector] {eeg_projector}")
    proj_params = sum(p.numel() for p in eeg_projector.parameters())
    print(f"[EEG Projector] Parameters: {proj_params / 1e6:.2f}M")
    
    # Apply LoRA to GPT model
    target_modules = args.lora_target_modules.split(',')
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    
    gpt_wo_ddp, lora_params = apply_lora_to_model(gpt_wo_ddp, lora_config, verbose=True)
    
    # Set only LoRA as trainable in GPT
    set_lora_only_trainable(gpt_wo_ddp)
    
    trainable, total = count_lora_parameters(gpt_wo_ddp)
    print(f"[LoRA] Trainable: {trainable / 1e6:.2f}M, Total: {total / 1e9:.2f}B")
    
    return eeg_projector, lora_params


def build_optimizer_for_eeg_lora(args, eeg_projector, gpt_with_lora):
    """
    Build optimizer with separate learning rates for EEG projector and LoRA.
    
    Args:
        args: Training arguments
        eeg_projector: EEG projector module
        gpt_with_lora: GPT model with LoRA layers
        
    Returns:
        Optimizer
    """
    # EEG projector parameters
    eeg_params = list(eeg_projector.parameters())
    
    # LoRA parameters (only trainable params from GPT)
    lora_params = [p for p in gpt_with_lora.parameters() if p.requires_grad]
    
    param_groups = [
        {'params': eeg_params, 'lr': args.eeg_projector_lr, 'name': 'eeg_projector'},
        {'params': lora_params, 'lr': args.lora_lr, 'name': 'lora'},
    ]
    
    # IMPORTANT: Use eps=1e-8 to avoid division by zero (SIGFPE)
    # The default adam_eps=0.0 in arg_util.py causes SIGFPE with fused_adam
    eps = getattr(args, 'adam_eps', 0.0)
    if eps == 0.0:
        eps = 1e-8  # Safe default to prevent division by zero
        print(f"[Optimizer] WARNING: adam_eps was 0.0, using 1e-8 to prevent SIGFPE")
    
    # Use standard AdamW instead of fused version for stability
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=eps,
    )
    
    print(f"[Optimizer] EEG Projector params: {sum(p.numel() for p in eeg_params) / 1e6:.2f}M, lr={args.eeg_projector_lr}")
    print(f"[Optimizer] LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M, lr={args.lora_lr}")
    print(f"[Optimizer] Using eps={eps}, fused_adam=False (using standard AdamW)")
    
    return optimizer


def build_everything_from_args(args):
    """Build all components for EEG-to-Video training."""
    args.set_initial_seed(benchmark=True)
    
    # Build VAE and GPT
    from infinity.utils.load import build_vae_gpt
    from infinity.models.infinity import Infinity
    from infinity.models.init_param import init_weights
    
    # Disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    
    # Initialize GPT weights
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init()
    
    # Load pretrained weights if specified
    if args.rush_resume:
        print(f"Loading pretrained weights from {args.rush_resume}")
        cpu_d = torch.load(args.rush_resume, 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
        else:
            state_dict = cpu_d
        missing, unexpected = gpt_wo_ddp.load_state_dict(state_dict, strict=False)
        print("missing keys sample:", [k for k in missing if "text_" in k][:50])
        print("unexpected keys sample:", [k for k in unexpected if "text_" in k][:50])
    elif args.torchshard_resume:
        from transformers.modeling_utils import load_sharded_checkpoint
        print(f"Loading sharded checkpoint from {args.torchshard_resume}")
        # load_sharded_checkpoint returns None or missing_keys depending on version
        # We don't unpack it to avoid errors.
        load_sharded_checkpoint(gpt_wo_ddp, args.torchshard_resume, strict=False)
        
    # === Weight Check & Fix ===
    if hasattr(gpt_wo_ddp, 'text_proj'):
        tp = gpt_wo_ddp.text_proj
        if isinstance(tp, torch.nn.Linear):
            w = tp.weight.detach()
            print(f"[Check] text_proj weight: shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.6f}, max={w.max():.6f}")
            # Heuristic check: if std is close to 0.02 (initialization) and we expected pretrained weights
            if abs(w.std() - 0.02) < 0.005 and abs(w.mean()) < 0.005:
                print(f"[WARNING] text_proj seems to be using initialization weights (std~0.02)!")
            
            # OPTIONAL: Force re-init to debug SIGFPE
            # print("DEBUG: Force re-initializing text_proj to ensure clean state")
            # torch.nn.init.xavier_normal_(tp.weight)
            # if tp.bias is not None: torch.nn.init.zeros_(tp.bias)
        elif isinstance(tp, torch.nn.Sequential):
             print(f"[Check] text_proj is Sequential")
    # ==========================
    
    gpt_wo_ddp = gpt_wo_ddp.to(args.device)
    
    # Freeze VAE
    vae_local.eval()
    for param in vae_local.parameters():
        param.requires_grad = False
    
    # Build EEG Projector and apply LoRA
    eeg_projector, lora_params = build_eeg_projector_and_lora(args, gpt_wo_ddp)
    
    # Move model to GPU after LoRA application (LoRA params are created on CPU)
    gpt_wo_ddp = gpt_wo_ddp.to(args.device)
    
    # Wrap GPT with DDP if needed
    if dist.initialized() and dist.get_world_size() > 1:
        gpt_ddp = DDP(
            gpt_wo_ddp, 
            device_ids=[dist.get_local_rank()], 
            find_unused_parameters=True,
            broadcast_buffers=False
        )
    else:
        gpt_ddp = gpt_wo_ddp
    
    # Also wrap EEG projector with DDP
    if dist.initialized() and dist.get_world_size() > 1:
        eeg_projector = DDP(
            eeg_projector,
            device_ids=[dist.get_local_rank()],
            find_unused_parameters=False,
            broadcast_buffers=False
        )
    
    # Build optimizer (use unwrapped modules for parameter access)
    gpt_for_opt = gpt_ddp.module if hasattr(gpt_ddp, 'module') else gpt_ddp
    eeg_proj_for_opt = eeg_projector.module if hasattr(eeg_projector, 'module') else eeg_projector
    optimizer = build_optimizer_for_eeg_lora(args, eeg_proj_for_opt, gpt_for_opt)
    
    # Load Text Encoder for Alignment Loss
    tokenizer = None
    text_encoder = None
    if hasattr(args, 'text_encoder_ckpt') and args.text_encoder_ckpt:
        print(f"Loading Text Encoder from {args.text_encoder_ckpt} for alignment supervision...")
        try:
            tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
            text_encoder.to(args.device)
            text_encoder.eval()
            for p in text_encoder.parameters():
                p.requires_grad = False
            print("[Text Encoder] Loaded and frozen.")
        except Exception as e:
            print(f"[Text Encoder] Failed to load: {e}")
            print("[Text Encoder] Alignment loss will be disabled.")
    
    return vae_local, gpt_wo_ddp, gpt_ddp, eeg_projector, optimizer, tokenizer, text_encoder


def build_eeg_dataset(args):
    """Build EEG-Video training dataset."""
    dataset = EEGVideoDataset(
        eeg_tokenizer_path=args.eeg_tokenizer_path,
        video_root=args.video_gt_root,
        caption_root=args.caption_root,
        video_fps=args.video_fps,
        video_frames=args.video_frames,
        pn=args.pn,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        temporal_compress_rate=args.temporal_compress_rate,
        use_concat_features=bool(args.use_concat_eeg),
        split="train",
        train_ratio=0.9,
        other_args=args,
    )
    return dataset


def train_step_eeg(
    args,
    vae_local,
    gpt: nn.Module,
    eeg_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict,
    video_encode_func,
    device: torch.device,
    self_correction: Optional[object] = None,
    grad_clip: float = 1.0,
    tokenizer=None,
    text_encoder=None,
) -> Tuple[float, float, float]:
    """
    Single training step for EEG-to-Video.
    
    Args:
        args: Training arguments
        vae_local: VAE model
        gpt: GPT model (with LoRA)
        eeg_projector: EEG projector module
        optimizer: Optimizer
        batch: Batch dict with eeg_features and video_frames
        video_encode_func: Function to encode video to tokens
        device: Device
        self_correction: SelfCorrection object
        grad_clip: Gradient clipping value
        tokenizer: Text tokenizer for alignment loss
        text_encoder: Text encoder for alignment loss
        
    Returns:
        Tuple of (loss, accuracy, total_norm)
    """
    optimizer.zero_grad()
    
    # Get rank for debugging (all ranks print)
    rk = dist.get_rank() if dist.initialized() else 0
    
    # Get EEG features and video frames
    eeg_features = batch['eeg_features'].to(device)  # (B, 14880) or (B, 2, 7440)
    video_frames = batch['video_frames'].to(device)  # (B, T, 3, H, W)
    
    B = eeg_features.shape[0]
    
    print(f"[R{rk}] eeg_features: {eeg_features.shape}, video_frames: {video_frames.shape}", flush=True)
    
    # === CRITICAL: Check video_frames for NaN/Inf and value range ===
    video_finite = torch.isfinite(video_frames).all().item()
    video_min = video_frames.min().item()
    video_max = video_frames.max().item()
    print(f"[R{rk}] video_frames: finite={video_finite}, min={video_min:.4f}, max={video_max:.4f}", flush=True)
    
    if not video_finite:
        print(f"[R{rk}] ERROR: NaN/Inf in video_frames! paths={batch.get('video_paths', 'unknown')}", flush=True)
        return float('nan'), 0.0, 0.0
    
    # Check EEG features too
    eeg_finite = torch.isfinite(eeg_features).all().item()
    eeg_min = eeg_features.min().item()
    eeg_max = eeg_features.max().item()
    print(f"[R{rk}] eeg_features: finite={eeg_finite}, min={eeg_min:.4f}, max={eeg_max:.4f}", flush=True)
    
    if not eeg_finite:
        print(f"[R{rk}] ERROR: NaN/Inf in eeg_features!", flush=True)
        return float('nan'), 0.0, 0.0
    
    # Replace any remaining NaN/Inf just in case (defensive)
    video_frames = torch.nan_to_num(video_frames, nan=0.0, posinf=1.0, neginf=-1.0)
    eeg_features = torch.nan_to_num(eeg_features, nan=0.0)
    
    # Project EEG to T5 space
    # Force FP32 to avoid numerical instability in the large output projection
    with torch.cuda.amp.autocast(enabled=False):
        eeg_features_f32 = eeg_features.float()
        eeg_embeddings = eeg_projector(eeg_features_f32)  # (B, seq_len, 2048)
    
    # Calculate Alignment Loss
    alignment_loss = 0.0
    if tokenizer is not None and text_encoder is not None and 'captions' in batch:
        captions = batch['captions']
        # Filter out empty captions if any
        valid_indices = [i for i, c in enumerate(captions) if c and isinstance(c, str)]
        
        if valid_indices:
            valid_captions = [captions[i] for i in valid_indices]
            
            # Tokenize captions
            # Note: We need to handle potential length mismatches carefully.
            # Here we rely on global average pooling for alignment, so length doesn't need to match exactly.
            with torch.no_grad():
                text_tokens = tokenizer(
                    valid_captions, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=512, # Standard T5 max length
                    return_tensors="pt"
                ).to(device)
                
                # Get T5 embeddings
                text_outputs = text_encoder(
                    input_ids=text_tokens.input_ids, 
                    attention_mask=text_tokens.attention_mask
                )
                gt_text_embeddings = text_outputs['last_hidden_state'] # (B_valid, L_text, 2048)
                text_mask = text_tokens.attention_mask.unsqueeze(-1) # (B_valid, L_text, 1)
            
            # Get corresponding EEG embeddings
            valid_eeg_embeddings = eeg_embeddings[valid_indices] # (B_valid, L_eeg, 2048)
            
            # Compute Global Average Pooling (GAP) for Alignment
            # GAP for Text: Mean over valid tokens
            text_sum = (gt_text_embeddings * text_mask).sum(dim=1)
            text_lens = text_mask.sum(dim=1).clamp(min=1.0)
            text_gap = text_sum / text_lens # (B_valid, 2048)
            
            # GAP for EEG: Mean over all tokens
            eeg_gap = valid_eeg_embeddings.mean(dim=1) # (B_valid, 2048)
            
            # L2 Normalize both vectors for stable cosine similarity
            eeg_gap_norm = F.normalize(eeg_gap, p=2, dim=-1)
            text_gap_norm = F.normalize(text_gap, p=2, dim=-1)
            
            # Alignment Loss: 1 - Cosine Similarity (range [0, 2])
            # This is more stable than MSE and encourages directional alignment
            cosine_sim = F.cosine_similarity(eeg_gap_norm, text_gap_norm, dim=-1)
            alignment_loss = (1.0 - cosine_sim).mean()
            
            print(f"[R{rk}] Alignment Loss: {alignment_loss.item():.4f}, Cosine Sim: {cosine_sim.mean().item():.4f}", flush=True)
        else:
             print(f"[R{rk}] Warning: No valid captions for alignment loss.", flush=True)
    
    # Ensure output is cast back to what the rest of the model expects if needed, 
    # but for now keeping it float for the text_norm/proj checks is fine.
    # If the next steps expect BF16, they will handle it or we cast it.
    
    eeg_emb_finite = torch.isfinite(eeg_embeddings).all().item()
    print(f"[R{rk}] eeg_embeddings: {eeg_embeddings.shape}, dtype={eeg_embeddings.dtype}, finite={eeg_emb_finite}", flush=True)
    
    if not eeg_emb_finite:
        print(f"[R{rk}] ERROR: NaN/Inf in eeg_embeddings after projection!", flush=True)
        # Debug: check weights if NaN
        eeg_proj_module = eeg_projector.module if hasattr(eeg_projector, 'module') else eeg_projector
        for name, param in eeg_proj_module.named_parameters():
             if not torch.isfinite(param).all():
                 print(f"[R{rk}] FATAL: EEG Projector param {name} contains NaN/Inf!", flush=True)
        return float('nan'), 0.0, 0.0
    
    # Get EEG sequence info for attention
    eeg_proj_module = eeg_projector.module if hasattr(eeg_projector, 'module') else eeg_projector
    seq_len = eeg_proj_module.seq_len
    lens = [seq_len] * B
    cu_seqlens_k = F.pad(
        torch.tensor([seq_len] * B, dtype=torch.int32, device=device).cumsum_(0),
        (1, 0)
    )
    
    # Project EEG embeddings through GPT's text_proj
    # This matches what happens with T5 embeddings
    gpt_module = gpt.module if hasattr(gpt, 'module') else gpt
    
    # Apply text_norm and text_proj (same as in infinity.py forward)
    # Force FP32 to avoid SIGFPE in fused kernels
    # DEBUG: We run this just to check if it crashes, but we don't use the result for GPT forward
    # because GPT forward expects raw embeddings to apply drop-cond.
    with torch.cuda.amp.autocast(enabled=False):
        x = eeg_embeddings.reshape(-1, 2048).float().contiguous()
        
        # Debug print
        print(f"[R{rk}] before text_norm: {x.shape} {x.dtype} finite={torch.isfinite(x).all().item()}", flush=True)
        torch.cuda.synchronize()
        
        # Check if we need to monkey-patch the model's text_proj to force CPU execution
        # if the previous run proved it crashes on GPU.
        # Since we saw earlier that text_proj (Linear) execution on GPU caused SIGFPE, 
        # and CPU execution worked, we should apply this fix to the model instance itself!
        
        gpt_module = gpt.module if hasattr(gpt, 'module') else gpt
        if hasattr(gpt_module, 'text_proj') and isinstance(gpt_module.text_proj, torch.nn.Linear):
             # Define a hooked forward that runs on CPU
             original_proj = gpt_module.text_proj
             
             # Only patch if not already patched
             if not hasattr(original_proj, 'is_cpu_patched'):
                 print(f"[R{rk}] PATCHING gpt_module.text_proj to run on CPU to avoid SIGFPE...", flush=True)
                 
                 class CpuLinear(torch.nn.Module):
                     def __init__(self, original):
                         super().__init__()
                         self.weight = original.weight
                         self.bias = original.bias
                         self.is_cpu_patched = True
                     
                     def forward(self, input):
                         device = input.device
                         input_cpu = input.cpu().float()
                         weight_cpu = self.weight.cpu().float()
                         bias_cpu = self.bias.cpu().float() if self.bias is not None else None
                         out = torch.nn.functional.linear(input_cpu, weight_cpu, bias_cpu)
                         return out.to(device)
                 
                 gpt_module.text_proj = CpuLinear(original_proj)
        
        # Verify text_norm works (usually fine on GPU)
        kv_compact = gpt_module.text_norm(x)
        
        # Debug print
        print(f"[R{rk}] after  text_norm: {kv_compact.shape} {kv_compact.dtype} finite={torch.isfinite(kv_compact).all().item()}", flush=True)
        torch.cuda.synchronize()
        
        # === Check text_proj weights ===
        if hasattr(gpt_module, 'text_proj'):
            pass # Skipping detailed weight check to reduce log spam
        # ================================

        # kv_compact = gpt_module.text_proj(kv_compact).contiguous()
        
        # Split text_proj execution to identify SIGFPE source
        tp = gpt_module.text_proj
        if isinstance(tp, torch.nn.Linear):
             print(f"[R{rk}] Executing text_proj (Linear)...", flush=True)
             
             # SIGFPE WORKAROUND: Force CPU execution for this specific linear layer
             # The CUDA kernel for (64, 2048) x (2048, 4096) seems broken on this setup
             try:
                 device_orig = kv_compact.device
                 # Move to CPU, compute, move back. 
                 # This preserves autograd history!
                 kv_compact_cpu = kv_compact.cpu().float()
                 weight_cpu = tp.weight.cpu().float()
                 bias_cpu = tp.bias.cpu().float() if tp.bias is not None else None
                 
                 kv_compact = F.linear(kv_compact_cpu, weight_cpu, bias_cpu).to(device_orig)
                 print(f"[R{rk}]   Executed on CPU successfully.", flush=True)
                 
             except Exception as e:
                 print(f"[R{rk}]   CPU execution failed: {e}", flush=True)
                 raise

        elif isinstance(tp, torch.nn.Sequential):
             print(f"[R{rk}] Executing text_proj (Sequential)...", flush=True)
             # Manually execute sequential to find which layer crashes
             for idx, layer in enumerate(tp):
                 print(f"[R{rk}]   Running layer {idx}: {type(layer).__name__}", flush=True)
                 if isinstance(layer, torch.nn.Linear):
                      kv_compact = F.linear(kv_compact, layer.weight.float(), layer.bias.float() if layer.bias is not None else None)
                 else:
                      kv_compact = layer(kv_compact)
                 torch.cuda.synchronize()
        else:
             print(f"[R{rk}] Executing text_proj (Generic)...", flush=True)
             kv_compact = tp(kv_compact)
        
        kv_compact = kv_compact.contiguous()
        
        # Debug print
        print(f"[R{rk}] after  text_proj: {kv_compact.shape} {kv_compact.dtype} finite={torch.isfinite(kv_compact).all().item()}", flush=True)
        torch.cuda.synchronize()
    
    # Prepare text condition tuple (matches format from train.py)
    # NOTE: kv_compact is now (B*seq_len, 4096) or (B*seq_len, 2048) depending on text_proj
    # Infinity.forward expects (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
    # But it does its OWN text_norm and text_proj internally!
    # If we do it here, we are doing it twice, OR we need to pass pre-projected features carefully.
    
    # In Infinity.forward:
    # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    # ...
    # kv_compact[total:total+le] = self.cfg_uncond[:le]  <-- CRASH HERE: cfg_uncond is (1, max_len, 2048), kv_compact is (64, 4096)
    # ...
    # kv_compact = self.text_norm(kv_compact)
    # kv_compact = self.text_proj(kv_compact)
    
    # PROBLEM: We projected eeg_embeddings to 4096 in train_eeg.py to avoid SIGFPE.
    # But Infinity.forward expects raw 2048-dim embeddings because it wants to apply drop-cond (replace with cfg_uncond)
    # AND then apply text_norm/text_proj itself.
    
    # SOLUTION: We must NOT project to 4096 here if Infinity.forward is going to do it again.
    # But we moved the projection here to avoid SIGFPE.
    
    # WORKAROUND:
    # 1. We pass the RAW 2048-dim embeddings to Infinity.
    # 2. But we need to ensure Infinity.forward doesn't crash with SIGFPE.
    #    Infinity.forward has `with torch.amp.autocast('cuda', enabled=False):` block around text processing.
    #    This matches what we did.
    
    # Wait, if we pass 2048-dim embeddings, will Infinity.forward crash?
    # Earlier we saw it crashed in text_proj.
    # In Infinity.forward (line 402-403):
    # kv_compact = self.text_norm(kv_compact)
    # kv_compact = self.text_proj(kv_compact).contiguous()
    
    # If we pass pre-projected (4096) embeddings, the drop-cond logic fails because cfg_uncond is 2048.
    # And then line 402 text_norm will fail or do weird things.
    
    # CORRECT FIX:
    # We should pass the 2048-dim embeddings (eeg_embeddings) to gpt(), NOT the projected kv_compact.
    # AND we need to rely on Infinity.forward's internal SIGFPE protection (which it seems to have: `with torch.amp.autocast('cuda', enabled=False):`).
    # If Infinity.forward crashes, we need to patch Infinity.forward, not do it outside.
    
    # However, our manual projection proved that running it on CPU works. 
    # If we pass 2048 dim, Infinity.forward runs it on GPU and might crash.
    
    # Strategy:
    # 1. Pass the original 2048-dim `x` (which is `eeg_embeddings.reshape(...)`) to `gpt()`.
    # 2. HOPE that Infinity.forward's `autocast(False)` is enough. 
    #    (Previously it crashed even with autocast(False) in our manual test? No, manual test showed CPU fallback was needed).
    
    # If we need CPU fallback, we MUST modify Infinity.forward or monkey-patch it.
    # Let's try passing the 2048-dim tensor first.
    
    # Re-retrieve the 2048-dim tensor (before text_norm/proj)
    # eeg_embeddings is (B, seq_len, 2048).
    
    kv_compact_raw = eeg_embeddings.reshape(-1, 2048).contiguous()
    text_cond_tuple = (kv_compact_raw, lens, cu_seqlens_k, seq_len)
    
    # Check kv_compact
    # kv_finite = torch.isfinite(kv_compact).all().item()
    # print(f"[R{rk}] kv_compact: {kv_compact.shape}, finite={kv_finite}", flush=True)
    
    # Encode video with VAE
    print(f"[R{rk}] Starting VAE encoding...", flush=True)
    
    with torch.no_grad():
        # Reshape video: (B, T, 3, H, W) -> process each sample
        raw_features_list = []
        for b_idx in range(B):
            video_b = video_frames[b_idx]  # (T, 3, H, W)
            # Convert to (1, 3, T, H, W) format expected by VAE
            video_b = video_b.permute(1, 0, 2, 3).unsqueeze(0)  # (3, T, H, W) -> (1, 3, T, H, W)
            
            print(f"[R{rk}] VAE input[{b_idx}]: {video_b.shape}, min={video_b.min().item():.4f}, max={video_b.max().item():.4f}", flush=True)
            
            try:
                raw_features, _, _ = vae_local.encode_for_raw_features(
                    video_b, scale_schedule=None, slice=getattr(args, 'use_slice', False)
                )
                raw_features_list.append(raw_features)
                print(f"[R{rk}] VAE output[{b_idx}]: {raw_features.shape}, finite={torch.isfinite(raw_features).all().item()}", flush=True)
            except Exception as e:
                print(f"[R{rk}] VAE encode FAILED for sample {b_idx}: {e}", flush=True)
                raise
    
    print(f"[R{rk}] VAE encoding done. raw_features_list len: {len(raw_features_list)}", flush=True)
    
    # Encode video tokens for training
    full_pts_this_batch = [item.shape[-3] for item in raw_features_list]
    
    print(f"[R{rk}] Starting video_encode_func...", flush=True)
    
    try:
        x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, \
        sequece_packing_scales, super_scale_lengths, super_querysid_super_refsid, \
        other_info_by_scale = video_encode_func(
            vae=vae_local,
            inp_B3HW=None,
            vae_features=raw_features_list,
            self_correction=self_correction, # Pass proper SelfCorrection object
            args=args,
            device=device,
            rope2d_freqs_grid=gpt_module.rope2d_freqs_grid,
            dynamic_resolution_h_w=gpt_module.dynamic_resolution_h_w,
            text_lens=lens,
            tokens_remain=args.train_max_token_len,
        )

        # Removed incorrect unpacking logic that was here

    except Exception as e:
        print(f"[R{rk}] video_encode_func FAILED: {e}", flush=True)
        # Check if it is the specific known error and print suggestion
        if "NoneType" in str(e) and "noise_apply_strength" in str(e):
             print(f"[R{rk}] HINT: This is likely due to 'self_correction' being None. We patched it to pass 'args'.", flush=True)
        raise
    
    if hasattr(gt_BLC, 'shape'):
        gt_shape = gt_BLC.shape
    else:
        gt_shape = f"List len={len(gt_BLC)}"

    print(f"[R{rk}] video_encode_func done. x_BLC: {x_BLC.shape}, gt_BLC: {gt_shape}, finite={torch.isfinite(x_BLC).all().item()}", flush=True)
    
    # Check if x_BLC is empty, which can happen if validation/encoding failed
    if x_BLC.shape[1] == 0:
        print(f"[R{rk}] WARNING: x_BLC is empty! Skipping this batch.", flush=True)
        return float('nan'), 0.0, 0.0
    
    # Forward through GPT
    print(f"[R{rk}] Starting GPT forward...", flush=True)
    
    try:
        loss_list, acc_list, valid_sequence_ratio = gpt(
            text_cond_tuple,
            x_BLC,
            gt_BL=gt_BLC,
            is_image_batch=0,
            visual_rope_cache=visual_rope_cache,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            super_querysid_super_refsid=super_querysid_super_refsid,
            other_info_by_scale=other_info_by_scale,
        )
    except Exception as e:
        print(f"[R{rk}] GPT forward FAILED: {e}", flush=True)
        raise
    
    print(f"[R{rk}] GPT forward done. loss_list: {loss_list.shape}", flush=True)
    
    # Compute loss (mean over sequence)
    video_loss = loss_list.mean()
    acc = acc_list.mean()
    
    # Total loss
    # alpha controls the strength of alignment supervision
    # Lower alpha (0.1) to balance with video loss (~0.67)
    alpha = 0.1 
    loss = video_loss + alpha * alignment_loss
    
    align_val = alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else alignment_loss
    print(f"[R{rk}] loss={loss.item():.4f} (video={video_loss.item():.4f}, align={align_val:.4f}, weighted_align={alpha * align_val:.4f}), acc={acc.item():.4f}", flush=True)
    
    # Check for NaN/Inf before backward (helps debug numerical issues)
    if not torch.isfinite(loss):
        print(f"[R{rk}] WARNING: loss is not finite: {loss.item()}", flush=True)
        print(f"[R{rk}]   loss_list stats: min={loss_list.min().item():.4f}, max={loss_list.max().item():.4f}", flush=True)
        # Skip this step to avoid corrupting optimizer state
        return float('nan'), 0.0, 0.0
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    total_norm = 0.0
    for p in eeg_projector.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    for p in gpt.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            list(eeg_projector.parameters()) + [p for p in gpt.parameters() if p.requires_grad],
            grad_clip
        )
    
    # Optimizer step
    optimizer.step()
    
    return loss.item(), acc.item() * 100, total_norm


def main_train(args):
    """Main training loop."""
    # Build everything
    ret = build_everything_from_args(args)
    vae_local, gpt_wo_ddp, gpt_ddp, eeg_projector, optimizer, tokenizer, text_encoder = ret
    
    # Get video encode function
    video_encode, _, _, _ = get_encode_decode_func(args.dynamic_scale_schedule)
    
    # Build dataset
    train_dataset = build_eeg_dataset(args)
    
    # Use DistributedSampler for proper DDP data distribution
    from torch.utils.data.distributed import DistributedSampler
    
    train_sampler = None
    if dist.initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        print(f"[Rank {dist.get_rank()}] Using DistributedSampler with {len(train_dataset)} samples")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.video_batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        sampler=train_sampler,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=False,  # Disable pinned memory for debugging
        collate_fn=collate_eeg_video_batch,
        drop_last=True,
    )
    
    print(f"[Rank {dist.get_rank()}] Dataset: {len(train_dataset)} samples, {len(train_dataloader)} batches per rank")
    
    # Training loop
    gc.collect()
    torch.cuda.empty_cache()
    
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config=vars(args))
    
    if dist.initialized():
        tdist.barrier(device_ids=[dist.get_local_rank()])
    else:
        # No-op for non-distributed
        pass
        
    print(f"[Rank {dist.get_rank()}] Ready to start training loop")
    
    # Initialize SelfCorrection object
    self_correction = SelfCorrection(vae_local, args)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epoch):
        # Set epoch for DistributedSampler to ensure different shuffling each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        gpt_ddp.train()
        eeg_projector.train()
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        rk = dist.get_rank() if dist.initialized() else 0
        print(f"[R{rk}] Starting epoch {epoch+1}/{args.epoch}...", flush=True)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # All ranks print first batch info for debugging
            if batch_idx == 0:
                print(f"[R{rk}] Got first batch! Keys: {batch.keys()}", flush=True)
                for k, v in batch.items():
                    if hasattr(v, 'shape'):
                        print(f"[R{rk}]   {k}: {v.shape}, dtype={v.dtype}", flush=True)
                    elif isinstance(v, list):
                        print(f"[R{rk}]   {k}: list of {len(v)} items", flush=True)
            try:
                loss, acc, grad_norm = train_step_eeg(
                    args=args,
                    vae_local=vae_local,
                    gpt=gpt_ddp,
                    eeg_projector=eeg_projector,
                    optimizer=optimizer,
                    batch=batch,
                    video_encode_func=video_encode,
                    device=args.device,
                    self_correction=self_correction,
                    grad_clip=args.grad_clip,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                )
            except Exception as e:
                print(f"[ERROR] batch_idx={batch_idx}, error: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % args.log_freq == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_acc / num_batches
                
                print(f"[Epoch {epoch+1}][Step {global_step}] Loss: {loss:.4f}, Acc: {acc:.2f}%, "
                      f"Grad Norm: {grad_norm:.4f}, Avg Loss: {avg_loss:.4f}")
                
                if dist.is_master():
                    wandb_utils.log({
                        'train/loss': loss,
                        'train/acc': acc,
                        'train/grad_norm': grad_norm,
                        'train/avg_loss': avg_loss,
                        'train/avg_acc': avg_acc,
                        'train/epoch': epoch + 1,
                    }, step=global_step)
            
            # Save checkpoint
            if global_step % args.save_model_iters_freq == 0 and dist.is_master():
                save_checkpoint(
                    args, global_step, epoch,
                    gpt_wo_ddp, eeg_projector, optimizer,
                    loss, best_loss
                )
                if loss < best_loss:
                    best_loss = loss
        
        # End of epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        print(f"[Epoch {epoch+1}] Completed. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%")
    
    # Final save
    if dist.is_master():
        save_checkpoint(
            args, global_step, args.epoch,
            gpt_wo_ddp, eeg_projector, optimizer,
            avg_loss, best_loss, final=True
        )
    
    print("Training completed!")


def save_checkpoint(
    args, 
    global_step: int, 
    epoch: int,
    gpt_wo_ddp: nn.Module,
    eeg_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    best_loss: float,
    final: bool = False,
):
    """Save training checkpoint."""
    save_dir = osp.join(args.bed, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get EEG projector state dict
    eeg_proj_module = eeg_projector.module if hasattr(eeg_projector, 'module') else eeg_projector
    
    checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'eeg_projector': eeg_proj_module.state_dict(),
        'lora': get_lora_state_dict(gpt_wo_ddp),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'best_loss': best_loss,
        # 'args': vars(args),  <-- Removed to avoid pickling error
    }
    
    if final:
        save_path = osp.join(save_dir, 'checkpoint_final.pth')
    else:
        save_path = osp.join(save_dir, f'checkpoint_step_{global_step}.pth')
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    # Also save best model
    if loss < best_loss:
        best_path = osp.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    gpt_wo_ddp: nn.Module,
    eeg_projector: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load EEG projector
    eeg_proj_module = eeg_projector.module if hasattr(eeg_projector, 'module') else eeg_projector
    eeg_proj_module.load_state_dict(checkpoint['eeg_projector'])
    
    # Load LoRA weights
    load_lora_state_dict(gpt_wo_ddp, checkpoint['lora'])
    
    # Load optimizer if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Global step: {checkpoint['global_step']}, Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def parse_eeg_args_from_argv():
    """Parse EEG-specific arguments from sys.argv."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    
    # EEG data paths
    parser.add_argument('--eeg_tokenizer_path', type=str, 
                        default='./eeg_outputs/brain_tokenizer_quant_per_window.pt')
    parser.add_argument('--video_gt_root', type=str,
                        default='./eeg_data/Video/Video_sections_original_resolution')
    parser.add_argument('--caption_root', type=str,
                        default='./eeg_data/Video/BLIP-caption')
    
    # EEG Projector config
    parser.add_argument('--eeg_dim', type=int, default=14880)
    parser.add_argument('--eeg_seq_len', type=int, default=64)
    parser.add_argument('--eeg_hidden_dim', type=int, default=4096)
    parser.add_argument('--eeg_num_layers', type=int, default=2)
    parser.add_argument('--eeg_projector_type', type=str, default='mlp')
    parser.add_argument('--eeg_projector_dropout', type=float, default=0.1)
    
    # LoRA config
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--lora_alpha', type=float, default=64.0)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj')
    
    parser.add_argument('--text_encoder_ckpt', type=str, default='./checkpoints/text_encoder/flan-t5-xl-official/')

    # Training config
    parser.add_argument('--eeg_projector_lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=1e-4)
    parser.add_argument('--use_concat_eeg', type=int, default=1)
    
    eeg_args, _ = parser.parse_known_args()
    return eeg_args


def main():
    """Main entry point."""
    # Parse EEG-specific args first (they're not in Args class)
    eeg_args = parse_eeg_args_from_argv()
    
    # Parse standard arguments
    args = arg_util.init_dist_and_get_args()
    
    # Merge EEG args into main args
    args.eeg_tokenizer_path = eeg_args.eeg_tokenizer_path
    args.video_gt_root = eeg_args.video_gt_root
    args.caption_root = eeg_args.caption_root
    args.eeg_dim = eeg_args.eeg_dim
    args.eeg_seq_len = eeg_args.eeg_seq_len
    args.eeg_hidden_dim = eeg_args.eeg_hidden_dim
    args.eeg_num_layers = eeg_args.eeg_num_layers
    args.eeg_projector_type = eeg_args.eeg_projector_type
    args.eeg_projector_dropout = eeg_args.eeg_projector_dropout
    args.lora_rank = eeg_args.lora_rank
    args.lora_alpha = eeg_args.lora_alpha
    args.lora_dropout = eeg_args.lora_dropout
    args.lora_target_modules = eeg_args.lora_target_modules
    args.text_encoder_ckpt = eeg_args.text_encoder_ckpt
    args.eeg_projector_lr = eeg_args.eeg_projector_lr
    args.lora_lr = eeg_args.lora_lr
    args.use_concat_eeg = eeg_args.use_concat_eeg
    
    # Set temporal_compress_rate if not already set
    if not hasattr(args, 'temporal_compress_rate') or args.temporal_compress_rate == 0:
        args.temporal_compress_rate = 4
    
    main_train(args)
    
    print(f'Final args:\n\n{str(args)}')
    args.dump_log()
    
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close()
        sys.stderr.close()
    
    dist.barrier()


if __name__ == '__main__':
    main()

