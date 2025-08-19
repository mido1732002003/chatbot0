"""Multi-head self-attention implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking.
    
    Implements scaled dot-product attention with multiple heads,
    supporting both full sequence processing and incremental generation
    with key-value caching.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            use_cache: Whether to return key/value for caching
            past_key_value: Cached key/value from previous forward pass
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Optional cached key/value tuple if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        # Shape: (batch_size, seq_len, d_model)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, n_heads, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Handle key-value caching for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Concatenate past and current key/value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            
        # Cache current key/value if requested
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
            
        # Compute attention scores
        # Shape: (batch_size, n_heads, seq_len, key_len)
        key_len = key.shape[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: (batch_size, 1, seq_len, key_len)
            # Expand for all heads
            scores = scores + attention_mask
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # Shape: (batch_size, n_heads, seq_len, d_k)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value