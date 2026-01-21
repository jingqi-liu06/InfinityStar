import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from infinity.utils.eeg_utils import compute_raw_stats, normalize_raw

class EEGAlignmentRawDataset(Dataset):
    """
    Dataset for EEG-Text alignment using RAW EEG signals.
    
    Loads .npy files with shape (n_blocks, n_concepts, n_rep, n_win, C, T).
    Maps these to captions assuming:
    - n_blocks * n_concepts = Total Videos
    - n_rep * n_win = EEG samples per video
    """
    
    def __init__(
        self,
        raw_eeg_path: str,
        caption_root: str,
        video_root: str,  # Used for checking/mapping logic if needed
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        cached_text_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        device: str = "cpu",
        raw_stats_path: Optional[str] = None
    ):
        super().__init__()
        self.split = split
        self.cached_text_embeddings = cached_text_embeddings
        self.use_cache = cached_text_embeddings is not None
        
        # 1. Load Raw EEG
        print(f"Loading raw EEG from {raw_eeg_path}...")
        raw = np.load(raw_eeg_path)
        # Expected shape: (n_blocks, n_concepts, n_rep, n_win, C, T)
        # If shape is different, we might need adjustments.
        if raw.ndim == 6:
            n_blocks, n_concepts, n_rep, n_win, C, T = raw.shape
        elif raw.ndim == 5:
             # Handle case where n_win is merged or missing?
             # Assume (n_blocks, n_concepts, n_rep, C, T) -> n_win=1
             n_blocks, n_concepts, n_rep, C, T = raw.shape
             n_win = 1
             raw = raw.reshape(n_blocks, n_concepts, n_rep, n_win, C, T)
        else:
            raise ValueError(f"Unexpected raw data shape: {raw.shape}")
            
        print(f"Raw shape: {raw.shape}")
        
        # 2. Flatten to List of Samples
        # We want to keep track of which video each sample belongs to.
        # Total samples = n_blocks * n_concepts * n_rep * n_win
        # Video index = block_idx * n_concepts + concept_idx
        
        # Reshape to (Total_Videos, Samples_Per_Video, C, T)
        # Total_Videos = n_blocks * n_concepts
        # Samples_Per_Video = n_rep * n_win
        
        self.n_videos = n_blocks * n_concepts
        self.samples_per_video = n_rep * n_win # 5*3
        self.C = C
        self.T = T
        
        # Reshape for normalization and storage
        # (Total_Videos * Samples_Per_Video, C, T)
        self.raw_data = raw.reshape(-1, C, T)
        total_samples = len(self.raw_data)
        
        # Create video indices for each sample
        # sample i belongs to video i // samples_per_video
        #NOTE: 
        self.sample_to_video_idx = np.arange(total_samples) // self.samples_per_video
        
        # 3. Load Captions
        self.captions = self._load_captions(caption_root, video_root)
        
        if len(self.captions) != self.n_videos:
            print(f"Warning: Number of captions ({len(self.captions)}) != Number of videos inferred from EEG ({self.n_videos})")
            # Truncate to minimum
            min_len = min(len(self.captions), self.n_videos)
            self.captions = self.captions[:min_len]
            # Filter EEG samples
            valid_mask = self.sample_to_video_idx < min_len
            self.raw_data = self.raw_data[valid_mask]
            self.sample_to_video_idx = self.sample_to_video_idx[valid_mask]
            self.n_videos = min_len
            
        # 4. Train/Val Split logic (Determined BEFORE normalization to avoid leakage)
        # We need to know which videos are in TRAIN set to compute stats only on them.
        np.random.seed(seed)
        video_indices = np.random.permutation(self.n_videos)
        split_idx = int(len(video_indices) * train_ratio)
        
        train_videos_set = set(video_indices[:split_idx])
        
        # Create masks
        # map sample_idx -> video_idx -> check if in train_videos_set
        video_mask_all = np.zeros(self.n_videos, dtype=bool)
        video_mask_all[list(train_videos_set)] = True
        
        # Identify which SAMPLES belong to the training set
        train_sample_mask = video_mask_all[self.sample_to_video_idx]
        
        # 5. Normalize Data (Strictly using TRAIN statistics)
        print("Computing statistics on TRAIN set only...")
        # Only use training samples to compute mean/std
        train_data_subset = self.raw_data[train_sample_mask]
        mean, std = compute_raw_stats(train_data_subset)
        self.raw_mean = mean
        self.raw_std = std
        print(f"Train stats - Mean shape: {mean.shape}, Std shape: {std.shape}")

        if raw_stats_path and split == "train":
            stats_dir = os.path.dirname(raw_stats_path)
            if stats_dir:
                os.makedirs(stats_dir, exist_ok=True)
            np.savez(raw_stats_path, mean=mean, std=std)
            print(f"Saved raw stats to {raw_stats_path}")
        
        # Apply normalization to ALL data (Train + Val) using TRAIN stats
        self.raw_data = normalize_raw(self.raw_data, mean, std)
        
        # 6. Finalize Indices based on requested split
        if split == "train":
            self.indices = np.where(train_sample_mask)[0]
        else:
            # Val/Test split
            val_sample_mask = ~train_sample_mask
            self.indices = np.where(val_sample_mask)[0]
            
        print(f"[{split}] {len(self.indices)} samples from {self.n_videos} total videos")

    def _load_captions(self, caption_root, video_root):
        """
        Load captions matching the block structure.
        """
        all_captions = []
        # Assumption: Captions are in text files like 1st_10min.txt, etc.
        # And they correspond sequentially to videos in Block0, Block1...
        
        caption_files = sorted(glob.glob(os.path.join(caption_root, "*.txt")))
        for caption_file in caption_files:
            with open(caption_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_captions.extend(lines)
                
        # We need to ensure we have one caption per video.
        # If there are missing captions, we might need to pad or align carefully.
        # But usually in this dataset, they align.
        return all_captions

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Get EEG Data (C, T) -> add channel dim for model (1, C, T)
        # Models in custom_encoders expect (B, 1, C, T)
        eeg_data = self.raw_data[real_idx] # (C, T)
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0) # (1, C, T)
        
        # Get Caption
        video_idx = self.sample_to_video_idx[real_idx]
        caption = self.captions[video_idx]
        
        item = {
            'eeg_features': eeg_tensor, # Raw data, named features for compatibility
            'caption': caption,
            'idx': real_idx
        }
        
        # Cache handling
        if self.use_cache:
            if caption in self.cached_text_embeddings:
                cached_data = self.cached_text_embeddings[caption]
                if isinstance(cached_data, tuple):
                    item['text_embedding'] = cached_data[0]
                    item['text_mask'] = cached_data[1]
                else:
                    item['text_embedding'] = cached_data
                    
        return item
