"""Transformer building blocks: layers and blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.
    
    Expands dimension by a factor (typically 4x), applies GELU activation,
    then projects back to model dimension.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply first linear layer with GELU activation
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block.
    
    Consists of:
    1. Multi-head self-attention with causal masking
    2. Position-wise feed-forward network
    3. Layer normalization and residual connections
    
    Uses pre-norm architecture (LayerNorm before sub-layers).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through transformer block.
        
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
        # Self-attention with residual connection
        # Pre-norm: normalize before attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_output
        
        # Feed-forward with residual connection
        # Pre-norm: normalize before feed-forward
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states, present_key_value


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Alternative to LayerNorm that normalizes by RMS of activations
    rather than standardizing (no mean subtraction).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize RMS normalization.
        
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x = x / rms * self.weight
        
        return x