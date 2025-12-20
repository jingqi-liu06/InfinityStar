# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
LoRA (Low-Rank Adaptation) implementation for Infinity transformer model.

This module provides LoRA wrappers for linear layers, enabling efficient 
fine-tuning by only training low-rank adaptation matrices.

Reference: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA wrapper for linear layers.
    
    Implements: output = W @ x + (alpha/r) * B @ A @ x
    where W is the frozen original weight, A and B are trainable low-rank matrices.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Args:
            original_layer: The original nn.Linear layer to wrap
            rank: LoRA rank (r in the paper)
            alpha: LoRA scaling factor
            dropout: Dropout probability for LoRA path
            merge_weights: Whether to merge weights for inference (not recommended during training)
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        # A: (in_features, rank) - initialized with Kaiming uniform
        # B: (rank, out_features) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros so that initial output equals original layer
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Original layer output
        result = self.original_layer(x)
        
        if not self.merged:
            # LoRA path: x @ A @ B * scaling
            lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            result = result + lora_out * self.scaling
        
        return result
    
    def merge(self):
        """Merge LoRA weights into original layer for efficient inference."""
        if not self.merged:
            # W' = W + (alpha/r) * B @ A^T
            self.original_layer.weight.data += (
                self.scaling * (self.lora_B.t() @ self.lora_A.t())
            )
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from original layer."""
        if self.merged:
            self.original_layer.weight.data -= (
                self.scaling * (self.lora_B.t() @ self.lora_A.t())
            )
            self.merged = False
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    def __init__(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        merge_weights: bool = False,
    ):
        """
        Args:
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
            target_modules: List of module names to apply LoRA to.
                           Default targets attention projections: q_proj, k_proj, v_proj, o_proj
            merge_weights: Whether to merge weights during inference
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.merge_weights = merge_weights


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, nn.Parameter]]:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
        verbose: Whether to print applied modules
        
    Returns:
        Tuple of (modified model, dict of LoRA parameters)
    """
    lora_params = {}
    applied_modules = []
    
    def _apply_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module should have LoRA applied
            if isinstance(child, nn.Linear):
                # Check if module name matches any target
                should_apply = any(
                    target in name for target in config.target_modules
                )
                
                if should_apply:
                    # Create LoRA wrapper
                    lora_layer = LoRALinear(
                        original_layer=child,
                        rank=config.rank,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        merge_weights=config.merge_weights,
                    )
                    
                    # Replace the module
                    setattr(module, name, lora_layer)
                    
                    # Track LoRA parameters
                    lora_params[f"{full_name}.lora_A"] = lora_layer.lora_A
                    lora_params[f"{full_name}.lora_B"] = lora_layer.lora_B
                    
                    applied_modules.append(full_name)
            else:
                # Recurse into child modules
                _apply_lora_recursive(child, full_name)
    
    _apply_lora_recursive(model)
    
    if verbose:
        print(f"[LoRA] Applied to {len(applied_modules)} modules:")
        for mod_name in applied_modules[:10]:  # Show first 10
            print(f"  - {mod_name}")
        if len(applied_modules) > 10:
            print(f"  ... and {len(applied_modules) - 10} more")
        
        total_lora_params = sum(p.numel() for p in lora_params.values())
        print(f"[LoRA] Total LoRA parameters: {total_lora_params / 1e6:.2f}M")
    
    return model, lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from model state dict.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA parameters into model.
    
    Args:
        model: Model with LoRA layers
        state_dict: LoRA state dict to load
        strict: Whether to require exact match
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"
            
            if lora_a_key in state_dict:
                module.lora_A.data.copy_(state_dict[lora_a_key])
                unexpected_keys.remove(lora_a_key)
            elif strict:
                missing_keys.append(lora_a_key)
            
            if lora_b_key in state_dict:
                module.lora_B.data.copy_(state_dict[lora_b_key])
                unexpected_keys.remove(lora_b_key)
            elif strict:
                missing_keys.append(lora_b_key)
    
    return missing_keys, unexpected_keys


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into original layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from original layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in model with LoRA.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, total_params


def set_lora_only_trainable(model: nn.Module):
    """
    Set only LoRA parameters as trainable, freeze everything else.
    
    Args:
        model: Model with LoRA layers
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then enable LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


class LoRAModel(nn.Module):
    """
    Wrapper class for models with LoRA adaptation.
    
    Provides convenient methods for LoRA management.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig,
    ):
        """
        Args:
            base_model: The base model to wrap
            config: LoRA configuration
        """
        super().__init__()
        
        self.config = config
        self.base_model, self.lora_params = apply_lora_to_model(
            base_model, config, verbose=True
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model."""
        return self.base_model(*args, **kwargs)
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get list of LoRA parameters."""
        return list(self.lora_params.values())
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get LoRA state dict."""
        return get_lora_state_dict(self.base_model)
    
    def load_lora_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ):
        """Load LoRA state dict."""
        return load_lora_state_dict(self.base_model, state_dict, strict)
    
    def merge_weights(self):
        """Merge LoRA weights for inference."""
        merge_lora_weights(self.base_model)
    
    def unmerge_weights(self):
        """Unmerge LoRA weights for continued training."""
        unmerge_lora_weights(self.base_model)
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count trainable and total parameters."""
        return count_lora_parameters(self.base_model)
    
    def set_trainable(self):
        """Set only LoRA parameters as trainable."""
        set_lora_only_trainable(self.base_model)

