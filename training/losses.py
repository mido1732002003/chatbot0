"""Loss functions for training."""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Label value to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', 'none')
        label_smoothing: Label smoothing factor
        
    Returns:
        Loss tensor
    """
    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    
    # Compute cross-entropy loss
    if label_smoothing > 0:
        # With label smoothing
        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    else:
        # Standard cross-entropy
        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction=reduction
        )
        
    return loss


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Compute perplexity from logits and labels.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Label value to ignore
        
    Returns:
        Perplexity value
    """
    # Get loss
    loss = compute_loss(logits, labels, ignore_index=ignore_index, reduction='mean')
    
    # Compute perplexity
    perplexity = torch.exp(loss).item()
    
    # Cap perplexity to avoid overflow
    perplexity = min(perplexity, 1e4)
    
    return perplexity


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Compute token-level accuracy.
    
    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Label value to ignore
        
    Returns:
        Accuracy value
    """
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Mask ignored tokens
    mask = (labels != ignore_index)
    
    # Compute accuracy
    correct = ((predictions == labels) & mask).float().sum()
    total = mask.float().sum()
    
    accuracy = (correct / total).item() if total > 0 else 0.0
    
    return accuracy