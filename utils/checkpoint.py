"""Checkpoint utilities for saving and loading models."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model_state: Model state dict
        optimizer_state: Optional optimizer state dict
        scheduler_state: Optional scheduler state dict
        epoch: Current epoch
        step: Current step
        best_metric: Best metric value
        config: Configuration dictionary
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model_state,
        'epoch': epoch,
        'step': step
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
        
    if scheduler_state is not None:
        checkpoint['scheduler_state_dict'] = scheduler_state
        
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
        
    if config is not None:
        checkpoint['config'] = config
        
    # Add any additional items
    checkpoint.update(kwargs)
    
    # Create directory if needed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    

def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint