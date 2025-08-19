"""Seed utilities for reproducibility."""

import torch
import numpy as np
import random
from typing import Optional


def set_seed(seed: int, deterministic: bool = True):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic operations
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    # Set environment variable for hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_random_seed() -> int:
    """Generate a random seed.
    
    Returns:
        Random seed value
    """
    return random.randint(0, 2**32 - 1)