# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
EEG Projector Module for mapping EEG tokenizers to T5 text embedding space.

This module takes EEG tokenizer outputs and projects them to match the T5 
encoder output format, enabling EEG-conditioned video generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGProjector(nn.Module):
    """
    Projects EEG tokenizer outputs to T5 text embedding space.
    
    Input: (batch, eeg_dim) - typically (batch, 14880) from quant_per_window_concat
    Output: (batch, seq_len, t5_dim) - matches T5EncoderModel output format
    
    The projector uses a multi-layer MLP with residual connections and layer 
    normalization for stable training.
    """
    
    def __init__(
        self,
        eeg_dim: int = 14880,
        t5_dim: int = 2048,
        seq_len: int = 64,
        hidden_dim: int = 4096,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            eeg_dim: Input dimension of EEG tokenizer output (default: 14880)
            t5_dim: Output dimension matching T5 encoder (default: 2048 for flan-t5-xl)
            seq_len: Sequence length of output tokens (default: 64)
            hidden_dim: Hidden dimension for MLP layers (default: 4096)
            num_layers: Number of MLP layers (default: 2)
            dropout: Dropout probability (default: 0.1)
            use_layer_norm: Whether to use layer normalization (default: True)
        """
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.t5_dim = t5_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection: eeg_dim -> hidden_dim
        self.input_proj = nn.Linear(eeg_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            if use_layer_norm:
                self.hidden_norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.hidden_norms.append(nn.Identity())
        
        # Output projection: hidden_dim -> seq_len * t5_dim
        self.output_proj = nn.Linear(hidden_dim, seq_len * t5_dim)
        self.output_norm = nn.LayerNorm(t5_dim) if use_layer_norm else nn.Identity()
        
        # Learnable position encoding for output sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, t5_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Scale down output projection for stable initialization
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02 / math.sqrt(self.num_layers + 1))
    
    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EEG projector.
        
        Args:
            eeg_features: EEG tokenizer output, shape (batch, eeg_dim) or (batch, 2, 7440)
            
        Returns:
            T5-compatible embeddings, shape (batch, seq_len, t5_dim)
        """
        # Handle different input shapes
        if eeg_features.dim() == 3:
            # Shape: (batch, 2, 7440) -> flatten to (batch, 14880)
            batch_size = eeg_features.shape[0]
            eeg_features = eeg_features.reshape(batch_size, -1)
        
        batch_size = eeg_features.shape[0]
        
        # Input projection
        x = self.input_proj(eeg_features)  # (batch, hidden_dim)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Hidden layers with residual connections
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)
        
        # Output projection
        x = self.output_proj(x)  # (batch, seq_len * t5_dim)
        x = x.reshape(batch_size, self.seq_len, self.t5_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Final normalization
        x = self.output_norm(x)
        
        return x
    
    def get_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate attention mask for the projected EEG sequence.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Attention mask of shape (batch, seq_len), all ones
        """
        return torch.ones(batch_size, self.seq_len, dtype=torch.long, device=device)
    
    def get_cu_seqlens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate cumulative sequence lengths for flash attention.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Cumulative sequence lengths, shape (batch + 1,)
        """
        return torch.arange(
            0, (batch_size + 1) * self.seq_len, self.seq_len,
            dtype=torch.int32, device=device
        )
    
    def extra_repr(self) -> str:
        return (
            f"eeg_dim={self.eeg_dim}, t5_dim={self.t5_dim}, "
            f"seq_len={self.seq_len}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}"
        )


class EEGProjectorWithCrossAttention(nn.Module):
    """
    Alternative EEG projector using cross-attention mechanism.
    
    Uses learnable query tokens that attend to EEG features,
    similar to Q-Former in BLIP-2.
    """
    
    def __init__(
        self,
        eeg_dim: int = 14880,
        t5_dim: int = 2048,
        seq_len: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            eeg_dim: Input dimension of EEG tokenizer output
            t5_dim: Output dimension matching T5 encoder
            seq_len: Number of query tokens / output sequence length
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.t5_dim = t5_dim
        self.seq_len = seq_len
        
        # Project EEG features to key/value dimension
        # Reshape 14880 -> (num_eeg_tokens, t5_dim)
        self.num_eeg_tokens = 64  # Divide EEG dim into tokens
        self.eeg_token_dim = eeg_dim // self.num_eeg_tokens  # 14880 / 64 = 232.5, need adjustment
        
        # Use linear projection to create EEG tokens
        self.eeg_tokenizer = nn.Linear(eeg_dim, self.num_eeg_tokens * t5_dim)
        self.eeg_norm = nn.LayerNorm(t5_dim)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, seq_len, t5_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(
                    d_model=t5_dim,
                    nhead=num_heads,
                    dim_feedforward=t5_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
            )
        
        self.output_norm = nn.LayerNorm(t5_dim)
    
    def forward(self, eeg_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using cross-attention.
        
        Args:
            eeg_features: EEG tokenizer output, shape (batch, eeg_dim)
            
        Returns:
            T5-compatible embeddings, shape (batch, seq_len, t5_dim)
        """
        if eeg_features.dim() == 3:
            batch_size = eeg_features.shape[0]
            eeg_features = eeg_features.reshape(batch_size, -1)
        
        batch_size = eeg_features.shape[0]
        
        # Create EEG tokens as memory for cross-attention
        eeg_tokens = self.eeg_tokenizer(eeg_features)  # (batch, num_eeg_tokens * t5_dim)
        eeg_tokens = eeg_tokens.reshape(batch_size, self.num_eeg_tokens, self.t5_dim)
        eeg_tokens = self.eeg_norm(eeg_tokens)
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Apply cross-attention layers
        x = queries
        for layer in self.layers:
            x = layer(x, eeg_tokens)
        
        x = self.output_norm(x)
        
        return x
    
    def get_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones(batch_size, self.seq_len, dtype=torch.long, device=device)
    
    def get_cu_seqlens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0, (batch_size + 1) * self.seq_len, self.seq_len,
            dtype=torch.int32, device=device
        )


def build_eeg_projector(
    projector_type: str = "mlp",
    eeg_dim: int = 14880,
    t5_dim: int = 2048,
    seq_len: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to build EEG projector.
    
    Args:
        projector_type: Type of projector ("mlp" or "cross_attention")
        eeg_dim: Input dimension of EEG features
        t5_dim: Output dimension matching T5 encoder
        seq_len: Output sequence length
        **kwargs: Additional arguments passed to projector
        
    Returns:
        EEG projector module
    """
    if projector_type == "mlp":
        return EEGProjector(
            eeg_dim=eeg_dim,
            t5_dim=t5_dim,
            seq_len=seq_len,
            **kwargs
        )
    elif projector_type == "cross_attention":
        return EEGProjectorWithCrossAttention(
            eeg_dim=eeg_dim,
            t5_dim=t5_dim,
            seq_len=seq_len,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")

