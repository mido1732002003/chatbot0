"""Main Transformer Language Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import json
from pathlib import Path

from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding, PositionalEmbedding
from .layers import TransformerBlock, FeedForward


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.
    
    Implements a GPT-style autoregressive transformer with:
    - Token and positional embeddings
    - Multi-head self-attention with causal masking
    - Feed-forward networks with GELU activation
    - Layer normalization and residual connections
    - Weight tying between input embeddings and output projection
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 640,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        tie_weights: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize the Transformer Language Model.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of model embeddings and hidden states
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network (default: 4 * d_model)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            tie_weights: Whether to tie input/output embeddings
            device: Device to place model on
        """
        super().__init__()
        
        # Store hyperparameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or (4 * d_model)
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_weights = tie_weights
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=self.d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output projection
        if tie_weights:
            # Tie weights with token embedding
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Print parameter count
        param_count = self.count_parameters()
        print(f"Model initialized with {param_count:,} parameters")
        
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            use_cache: Whether to return key/value states for generation
            past_key_values: Cached key/value states from previous forward pass
            
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - past_key_values: Cached key/value states if use_cache=True
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Get position embeddings
        if past_key_values is not None:
            # For incremental generation
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            positions = torch.arange(past_len, past_len + seq_len, device=device)
        else:
            positions = torch.arange(seq_len, device=device)
        pos_embeds = self.pos_embedding(positions)
        
        # Combine embeddings
        hidden_states = self.embedding_dropout(token_embeds + pos_embeds.unsqueeze(0))
        
        # Create causal mask
        if attention_mask is None:
            # Create default causal mask
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            
        # Expand attention mask for attention computation
        # Shape: (batch_size, 1, seq_len, seq_len)
        if past_key_values is not None:
            # For incremental generation, adjust mask
            past_len = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            full_seq_len = past_len + seq_len
            
            # Create full attention mask
            full_attention_mask = torch.ones(batch_size, 1, seq_len, full_seq_len, device=device)
            
            # Apply causal masking
            causal_mask = torch.triu(
                torch.ones(seq_len, full_seq_len, device=device),
                diagonal=past_len + 1
            )
            full_attention_mask = full_attention_mask * (1.0 - causal_mask).unsqueeze(0).unsqueeze(0)
            attention_mask = full_attention_mask
        else:
            # Standard causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            )
            expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_attention_mask = expanded_attention_mask.expand (batch_size, 1, seq_len, seq_len)
            attention_mask = expanded_attention_mask * (1.0 - causal_mask).unsqueeze(0).unsqueeze(0)
            
        # Convert to negative infinity for masked positions
        attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e9)
        
        # Pass through transformer blocks
        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, kv = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_kv
            )
            if use_cache:
                present_key_values.append(kv)
                
        # Final layer norm
        hidden_states = self.ln_final(hidden_states)
        
        # Get logits
        if self.tie_weights:
            # Use tied weights from embedding
            logits = F.linear(hidden_states, self.token_embedding.weight)
        else:
            logits = self.lm_head(hidden_states)
            
        output = {'logits': logits}
        if use_cache:
            output['past_key_values'] = tuple(present_key_values)
            
        return output
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, step: int = 0, config: Dict = None):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            step: Current training step
            config: Optional config dict to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'step': step,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout,
                'tie_weights': self.tie_weights
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if config is not None:
            checkpoint['training_config'] = config
            
        torch.save(checkpoint, path)
        
    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        model = cls(**model_config, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model