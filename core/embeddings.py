"""Embedding layers for transformer model."""

import torch
import torch.nn as nn
import math
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional scaling."""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        """Initialize token embedding.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings
            padding_idx: Optional padding token index
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Token embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) for stability
        return self.embedding(input_ids) * self.scale
    
    @property
    def weight(self):
        """Get embedding weight matrix for weight tying."""
        return self.embedding.weight


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, max_seq_len: int, d_model: int):
        """Initialize positional embedding.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of embeddings
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings.
        
        Args:
            positions: Position indices of shape (seq_len,) or (batch_size, seq_len)
            
        Returns:
            Positional embeddings of shape (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        return self.embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (not learnable)."""
    
    def __init__(self, max_seq_len: int, d_model: int):
        """Initialize sinusoidal positional embedding.
        
        Args:
            max_seq_len: Maximum sequence length
            d_model: Dimension of embeddings
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Create sinusoidal embeddings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Create div_term for even and odd dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
            
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings.
        
        Args:
            positions: Position indices of shape (seq_len,)
            
        Returns:
            Positional embeddings of shape (seq_len, d_model)
        """
        return self.pe[positions]