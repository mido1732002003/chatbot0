"""Learning rate scheduling utilities."""

import math
from typing import Optional


class CosineScheduleWithWarmup:
    """Cosine learning rate schedule with linear warmup."""
    
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ):
        """Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total number of training steps
            num_cycles: Number of cosine cycles
            last_epoch: Last epoch number
        """
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Initialize
        self.step(last_epoch + 1)
        
    def get_lr(self, step: int) -> list:
        """Get learning rate for a given step.
        
        Args:
            step: Current step
            
        Returns:
            List of learning rates for each param group
        """
        if step < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * step / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            return [
                base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)))
                for base_lr in self.base_lrs
            ]
            
    def step(self, epoch: Optional[int] = None):
        """Update learning rate.
        
        Args:
            epoch: Current epoch (or step)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        lrs = self.get_lr(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
            
    def get_last_lr(self) -> list:
        """Get the last computed learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self) -> dict:
        """Get scheduler state dict."""
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            'num_warmup_steps': self.num_warmup_steps,
            'num_training_steps': self.num_training_steps,
            'num_cycles': self.num_cycles
        }
        
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state dict."""
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.num_warmup_steps = state_dict['num_warmup_steps']
        self.num_training_steps = state_dict['num_training_steps']
        self.num_cycles = state_dict['num_cycles']


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> CosineScheduleWithWarmup:
    """Create cosine schedule with warmup.
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch number
        
    Returns:
        Scheduler instance
    """
    return CosineScheduleWithWarmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        last_epoch=last_epoch
    )