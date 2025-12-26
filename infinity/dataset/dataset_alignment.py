# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
EEG-Text Alignment Dataset.

This dataset loads:
1. EEG tokenizer outputs (from .pt file)
2. Text captions (from BLIP-caption)
It does NOT load video frames, making it lightweight for alignment training.
"""

import glob
import os
import os.path as osp
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGAlignmentDataset(Dataset):
    """
    Dataset for EEG-Text alignment training (Stage 1).
    
    Loads paired EEG tokenizer outputs and captions.
    Optimized to pre-load data into memory and support cached text embeddings.
    """
    
    def __init__(
        self,
        eeg_tokenizer_path: str,
        caption_root: str,
        video_root: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        preload_to_gpu: bool = False,
        device: str = "cuda",
        cached_text_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            eeg_tokenizer_path: Path to EEG tokenizer output .pt file
            caption_root: Root directory for captions (BLIP-caption)
            video_root: Root directory for video files (needed to ensure alignment)
            split: "train" or "val"
            train_ratio: Ratio of data to use for training
            seed: Random seed for train/val split
            preload_to_gpu: Whether to move all EEG data to GPU at initialization
            device: Target device if preloading
            cached_text_embeddings: Optional dict mapping caption string -> embedding tensor
        """
        super().__init__()
        
        self.eeg_tokenizer_path = eeg_tokenizer_path
        self.caption_root = caption_root
        self.video_root = video_root
        self.split = split
        self.cached_text_embeddings = cached_text_embeddings
        self.use_cache = cached_text_embeddings is not None
        
        # Load EEG tokenizer outputs
        print(f"Loading EEG tokenizer outputs from {eeg_tokenizer_path}...")
        eeg_data = torch.load(eeg_tokenizer_path, weights_only=True)
        
        # We prefer concatenated features for the projector
        if 'quant_per_window_concat' in eeg_data:
            self.eeg_features = eeg_data['quant_per_window_concat']  # (N, 14880)
        else:
            self.eeg_features = eeg_data['quant_per_window']  # (N, 2, 7440)
        
        # Ensure EEG features are floats
        self.eeg_features = self.eeg_features.float()
        
        if preload_to_gpu:
            print(f"Pre-loading {len(self.eeg_features)} EEG samples to {device}...")
            self.eeg_features = self.eeg_features.to(device)
            
        self.num_samples = len(self.eeg_features)
        print(f"Loaded {self.num_samples} EEG samples, shape: {self.eeg_features.shape}")
        
        # Build video and caption mappings
        # We need video_paths just to ensure we map captions correctly (assuming block structure)
        self.video_paths, self.captions = self._build_data_mapping()
        
        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.video_paths))
        split_idx = int(len(indices) * train_ratio)
        
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        print(f"[{split}] {len(self.indices)} samples")
    
    def _build_data_mapping(self) -> Tuple[List[str], List[str]]:
        """
        Build mapping between EEG samples and captions.
        
        Assumes:
        - Videos are in Block0/1.mp4, Block0/2.mp4, ..., Block6/200.mp4
        - Captions are in 1st_10min.txt, 2nd_10min.txt, ..., 7th_10min.txt
        """
        video_paths = []
        captions = []
        
        # Load all captions from caption files
        all_captions = []
        caption_files = sorted(glob.glob(osp.join(self.caption_root, "*.txt")))
        
        for caption_file in caption_files:
            with open(caption_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_captions.extend(lines)
        
        print(f"Loaded {len(all_captions)} captions from {len(caption_files)} files")
        
        # Build video paths - iterate through blocks
        # This logic must match EEGVideoDataset to ensure EEG indices line up
        block_dirs = sorted(glob.glob(osp.join(self.video_root, "Block*")))
        
        video_idx = 0
        for block_dir in block_dirs:
            # Get sorted video files in this block
            video_files = sorted(
                glob.glob(osp.join(block_dir, "*.mp4")),
                key=lambda x: int(osp.splitext(osp.basename(x))[0])
            )
            
            for video_file in video_files:
                video_paths.append(video_file)
                
                # Get corresponding caption
                if video_idx < len(all_captions):
                    captions.append(all_captions[video_idx])
                else:
                    captions.append("")  # Empty caption if not available
                
                video_idx += 1
        
        print(f"Found {len(video_paths)} videos (used for alignment verification)")
        
        # Verify alignment with EEG samples
        if len(video_paths) != self.num_samples:
            print(f"Warning: Number of videos ({len(video_paths)}) != EEG samples ({self.num_samples})")
            # Use minimum
            min_samples = min(len(video_paths), self.num_samples)
            video_paths = video_paths[:min_samples]
            captions = captions[:min_samples]
            self.eeg_features = self.eeg_features[:min_samples]
            self.num_samples = min_samples
        
        return video_paths, captions
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.
        
        Returns:
            Dict with keys:
                - eeg_features: EEG tokenizer output (14880,)
                - caption: Text caption string
                - text_embedding: Cached text embedding (if available)
                - text_mask: Cached text mask (if available)
                - idx: Sample index
        """
        real_idx = self.indices[idx]
        
        # Get EEG features and ensure consistent shape (14880,)
        eeg_features = self.eeg_features[real_idx]
        if eeg_features.dim() == 2:  # Shape (2, 7440), flatten to (14880,)
            eeg_features = eeg_features.reshape(-1)
        
        caption = self.captions[real_idx]
        
        item = {
            'eeg_features': eeg_features,
            'caption': caption,
            'idx': real_idx,
        }
        
        # If cache is available, retrieve embedding directly
        if self.use_cache:
            if caption in self.cached_text_embeddings:
                cached_data = self.cached_text_embeddings[caption]
                # If cached data is a tuple (emb, mask)
                if isinstance(cached_data, tuple):
                    item['text_embedding'] = cached_data[0]
                    item['text_mask'] = cached_data[1]
                else:
                    item['text_embedding'] = cached_data
            else:
                # Fallback for missing captions (should not happen if prep is correct)
                # Just return None keys, let collate handle or training fail
                pass
                
        return item

def collate_alignment_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for Alignment batches.
    """
    # Stack EEG features
    eeg_list = []
    for s in batch:
        eeg = s['eeg_features']
        if eeg.dim() == 2:
            eeg = eeg.reshape(-1)
        eeg_list.append(eeg)
    eeg_features = torch.stack(eeg_list, dim=0)
    
    # Collect captions
    captions = [s['caption'] for s in batch]
    indices = [s['idx'] for s in batch]
    
    batch_dict = {
        'eeg_features': eeg_features,
        'captions': captions,
        'indices': indices,
    }
    
    # If cached embeddings are present
    if 'text_embedding' in batch[0]:
        # They should already be tensors on GPU/CPU
        text_emb_list = [s['text_embedding'] for s in batch]
        # Pad them if lengths differ (they shouldn't if we used max_length in cache)
        # But to be safe if they are different lengths
        # For T5 max_length=512, usually they are padded already.
        # Assuming they are already padded tensors
        batch_dict['text_embeddings'] = torch.stack(text_emb_list, dim=0)
        
        if 'text_mask' in batch[0]:
            text_mask_list = [s['text_mask'] for s in batch]
            batch_dict['text_masks'] = torch.stack(text_mask_list, dim=0)
            
    return batch_dict
