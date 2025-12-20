# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
EEG-Video paired dataset for EEG-conditioned video generation.

This dataset loads:
1. EEG tokenizer outputs (from .pt file)
2. Video files (from Video_sections_original_resolution)
3. Text captions (from BLIP-caption, for reference)
"""

import glob
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.functional import to_tensor

from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta
from infinity.utils.video_decoder import EncodedVideoOpencv


def transform_video_frame(pil_img, tgt_h, tgt_w):
    """Transform a video frame to target resolution."""
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
    # Crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)  # Normalize to [-1, 1]


class EEGVideoDataset(Dataset):
    """
    Dataset for EEG-to-Video generation training.
    
    Loads paired EEG tokenizer outputs, video files, and captions.
    """
    
    def __init__(
        self,
        eeg_tokenizer_path: str,
        video_root: str,
        caption_root: str,
        video_fps: int = 16,
        video_frames: int = 81,
        pn: str = "0.40M",
        dynamic_scale_schedule: str = "infinity_elegant_clip20frames_v2",
        temporal_compress_rate: int = 4,
        use_concat_features: bool = True,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        other_args=None,
    ):
        """
        Args:
            eeg_tokenizer_path: Path to EEG tokenizer output .pt file
            video_root: Root directory for video files (Video_sections_original_resolution)
            caption_root: Root directory for captions (BLIP-caption)
            video_fps: Video sampling FPS
            video_frames: Number of frames to sample
            pn: Patch number configuration
            dynamic_scale_schedule: Scale schedule for dynamic resolution
            temporal_compress_rate: Temporal compression rate
            use_concat_features: Whether to use concatenated EEG features (14880) or per-window (2, 7440)
            split: "train" or "val"
            train_ratio: Ratio of data to use for training
            seed: Random seed for train/val split
            other_args: Additional arguments
        """
        super().__init__()
        
        self.eeg_tokenizer_path = eeg_tokenizer_path
        self.video_root = video_root
        self.caption_root = caption_root
        self.video_fps = video_fps
        self.video_frames = video_frames
        self.pn = pn
        self.temporal_compress_rate = temporal_compress_rate
        self.use_concat_features = use_concat_features
        self.split = split
        self.other_args = other_args
        
        # Load dynamic resolution config
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            dynamic_scale_schedule, video_frames
        )
        
        # Load EEG tokenizer outputs
        print(f"Loading EEG tokenizer outputs from {eeg_tokenizer_path}...")
        eeg_data = torch.load(eeg_tokenizer_path, weights_only=True)
        
        if use_concat_features and 'quant_per_window_concat' in eeg_data:
            self.eeg_features = eeg_data['quant_per_window_concat']  # (N, 14880)
        else:
            self.eeg_features = eeg_data['quant_per_window']  # (N, 2, 7440)
        
        self.num_samples = len(self.eeg_features)
        print(f"Loaded {self.num_samples} EEG samples, shape: {self.eeg_features.shape}")
        
        # Build video and caption mappings
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
        Build mapping between EEG samples and video/caption files.
        
        Assumes:
        - Videos are in Block0/1.mp4, Block0/2.mp4, ..., Block6/200.mp4
        - Captions are in 1st_10min.txt, 2nd_10min.txt, ..., 7th_10min.txt
        - Each block has 200 videos, each line in caption file corresponds to one video
        
        Returns:
            Tuple of (video_paths, captions)
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
        
        print(f"Found {len(video_paths)} videos")
        
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
                - video_frames: Video frames tensor (T, 3, H, W)
                - caption: Text caption string
                - video_path: Path to video file
        """
        real_idx = self.indices[idx]
        
        # Get EEG features and ensure consistent shape (14880,)
        eeg_features = self.eeg_features[real_idx]
        if eeg_features.dim() == 2:  # Shape (2, 7440), flatten to (14880,)
            eeg_features = eeg_features.reshape(-1)
        
        # Get video path and caption
        video_path = self.video_paths[real_idx]
        caption = self.captions[real_idx]
        
        # Load video frames
        try:
            video_frames = self._load_video(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return -1s if video loading fails (VAE expects [-1, 1] range)
            h_div_w_template = self.h_div_w_templates[0]
            tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
            video_frames = torch.full((self.video_frames, 3, tgt_h, tgt_w), -1.0)
        
        return {
            'eeg_features': eeg_features,
            'video_frames': video_frames,
            'caption': caption,
            'video_path': video_path,
            'idx': real_idx,
        }
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video frames tensor of shape (T, 3, H, W)
        """
        video = EncodedVideoOpencv(video_path, osp.basename(video_path), num_threads=0)
        
        # Sample frames at specified FPS
        duration = video.duration
        sample_frames = min(self.video_frames, int(duration * self.video_fps) + 1)
        
        # Get frames
        raw_video, _ = video.get_clip(0, duration, sample_frames)
        
        if len(raw_video) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Determine target resolution
        h, w, _ = raw_video[0].shape
        h_div_w = h / w
        h_div_w_template = self.h_div_w_templates[np.argmin(np.abs(h_div_w - self.h_div_w_templates))]
        tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
        
        # Transform frames
        frames = []
        for frame in raw_video:
            # OpenCV returns BGR, convert to RGB
            frame_rgb = frame[:, :, ::-1]
            pil_img = Image.fromarray(frame_rgb)
            transformed = transform_video_frame(pil_img, tgt_h, tgt_w)
            frames.append(transformed)
        
        video_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)
        
        # Pad if necessary (use -1 for padding since VAE expects [-1, 1] range)
        if len(frames) < self.video_frames:
            pad_frames = self.video_frames - len(frames)
            padding = torch.full((pad_frames, 3, tgt_h, tgt_w), -1.0)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
        
        del video
        
        return video_tensor


class EEGVideoIterableDataset(IterableDataset):
    """
    Iterable dataset for EEG-to-Video generation with VAE feature caching.
    
    Similar to JointViIterableDataset but uses EEG features instead of text.
    """
    
    def __init__(
        self,
        eeg_tokenizer_path: str,
        video_root: str,
        caption_root: str,
        token_cache_dir: str,
        video_fps: int = 16,
        video_frames: int = 81,
        pn: str = "0.40M",
        dynamic_scale_schedule: str = "infinity_elegant_clip20frames_v2",
        temporal_compress_rate: int = 4,
        use_concat_features: bool = True,
        use_vae_token_cache: bool = True,
        rank: int = 0,
        num_replicas: int = 1,
        dataloader_workers: int = 2,
        seed: int = 42,
        other_args=None,
    ):
        """
        Args:
            eeg_tokenizer_path: Path to EEG tokenizer output .pt file
            video_root: Root directory for video files
            caption_root: Root directory for captions
            token_cache_dir: Directory for VAE token cache
            video_fps: Video sampling FPS
            video_frames: Number of frames to sample
            pn: Patch number configuration
            dynamic_scale_schedule: Scale schedule for dynamic resolution
            temporal_compress_rate: Temporal compression rate
            use_concat_features: Whether to use concatenated EEG features
            use_vae_token_cache: Whether to use VAE token cache
            rank: Distributed training rank
            num_replicas: Number of distributed replicas
            dataloader_workers: Number of dataloader workers
            seed: Random seed
            other_args: Additional arguments
        """
        super().__init__()
        
        self.video_fps = video_fps
        self.video_frames = video_frames
        self.pn = pn
        self.temporal_compress_rate = temporal_compress_rate
        self.use_vae_token_cache = use_vae_token_cache
        self.token_cache_dir = token_cache_dir
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataloader_workers = dataloader_workers
        self.seed = seed
        self.other_args = other_args
        
        # Load dynamic resolution config
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            dynamic_scale_schedule, video_frames
        )
        
        # Load EEG features
        print(f"Loading EEG tokenizer outputs from {eeg_tokenizer_path}...")
        eeg_data = torch.load(eeg_tokenizer_path, weights_only=True)
        
        if use_concat_features and 'quant_per_window_concat' in eeg_data:
            self.eeg_features = eeg_data['quant_per_window_concat']
        else:
            self.eeg_features = eeg_data['quant_per_window']
        
        print(f"Loaded {len(self.eeg_features)} EEG samples")
        
        # Build base dataset for video/caption info
        self.base_dataset = EEGVideoDataset(
            eeg_tokenizer_path=eeg_tokenizer_path,
            video_root=video_root,
            caption_root=caption_root,
            video_fps=video_fps,
            video_frames=video_frames,
            pn=pn,
            dynamic_scale_schedule=dynamic_scale_schedule,
            temporal_compress_rate=temporal_compress_rate,
            use_concat_features=use_concat_features,
            split="train",
            train_ratio=1.0,  # Use all data
            seed=seed,
            other_args=other_args,
        )
        
        self.num_samples = len(self.base_dataset)
        self.worker_id = 0
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
    
    def __iter__(self):
        """Iterate over dataset samples."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            self.worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            self.worker_id = 0
            num_workers = 1
        
        # Compute global worker ID
        global_worker_id = self.rank * num_workers + self.worker_id
        total_workers = self.num_replicas * num_workers
        
        # Shuffle indices
        generator = np.random.default_rng(self.seed + self.epoch)
        indices = generator.permutation(self.num_samples)
        
        # Shard across workers
        indices = indices[global_worker_id::total_workers]
        
        for idx in indices:
            try:
                sample = self.base_dataset[idx]
                
                # Process video for VAE
                video_frames = sample['video_frames']
                
                # Prepare output
                yield {
                    'eeg_features': sample['eeg_features'],
                    'video_frames': video_frames,  # (T, 3, H, W)
                    'caption': sample['caption'],
                    'video_path': sample['video_path'],
                    'idx': sample['idx'],
                }
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    def __len__(self) -> int:
        return self.num_samples // (self.num_replicas * self.dataloader_workers)


def collate_eeg_video_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for EEG-Video batches.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Collated batch dict
    """
    # Ensure EEG features are flattened to (14880,) for consistent stacking
    eeg_list = []
    for s in batch:
        eeg = s['eeg_features']
        if eeg.dim() == 2:  # Shape (2, 7440), flatten to (14880,)
            eeg = eeg.reshape(-1)
        eeg_list.append(eeg)
    eeg_features = torch.stack(eeg_list, dim=0)
    video_frames = torch.stack([s['video_frames'] for s in batch], dim=0)
    captions = [s['caption'] for s in batch]
    video_paths = [s['video_path'] for s in batch]
    indices = [s['idx'] for s in batch]
    
    return {
        'eeg_features': eeg_features,
        'video_frames': video_frames,
        'captions': captions,
        'video_paths': video_paths,
        'indices': indices,
    }


def build_eeg_video_dataset(
    eeg_tokenizer_path: str,
    video_root: str,
    caption_root: str,
    use_iterable: bool = False,
    **kwargs,
) -> Union[EEGVideoDataset, EEGVideoIterableDataset]:
    """
    Factory function to build EEG-Video dataset.
    
    Args:
        eeg_tokenizer_path: Path to EEG tokenizer output
        video_root: Root directory for videos
        caption_root: Root directory for captions
        use_iterable: Whether to use iterable dataset
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if use_iterable:
        return EEGVideoIterableDataset(
            eeg_tokenizer_path=eeg_tokenizer_path,
            video_root=video_root,
            caption_root=caption_root,
            **kwargs,
        )
    else:
        return EEGVideoDataset(
            eeg_tokenizer_path=eeg_tokenizer_path,
            video_root=video_root,
            caption_root=caption_root,
            **kwargs,
        )

