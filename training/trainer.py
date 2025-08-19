"""Main trainer class for supervised fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable, Tuple
import time
import json
from pathlib import Path

from .losses import compute_loss
from utils.scheduling import get_cosine_schedule_with_warmup
from utils.logging_utils import setup_logger
from core.generation import generate


class Trainer:
    """Trainer for supervised fine-tuning of language models.
    
    Handles training loop, optimization, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        eval_interval: int = 100,
        save_interval: int = 500,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = False,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Language model to train
            tokenizer: Tokenizer instance
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps (overrides max_epochs)
            max_epochs: Maximum training epochs
            gradient_clip: Gradient clipping value
            gradient_accumulation_steps: Steps to accumulate gradients
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoints
            checkpoint_dir: Directory to save checkpoints
            use_amp: Whether to use automatic mixed precision
            device: Device to train on
            logger: Logger instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Determine total steps
        if max_steps is not None:
            self.max_steps = max_steps
        elif max_epochs is not None:
            steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
            self.max_steps = max_epochs * steps_per_epoch
        else:
            self.max_steps = 1000  # Default
            
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.max_steps
        )
        
        # Setup AMP if requested
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        if logger is None:
            self.logger = setup_logger('trainer', log_file=self.checkpoint_dir / 'train.log')
        else:
            self.logger = logger
            
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Log configuration
        self.logger.info(f"Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        self.logger.info(f"Training samples: {len(train_dataloader.dataset)}")
        self.logger.info(f"Batch size: {train_dataloader.batch_size}")
        self.logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        self.logger.info(f"Effective batch size: {train_dataloader.batch_size * gradient_accumulation_steps}")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Warmup steps: {warmup_steps}")
        self.logger.info(f"AMP enabled: {self.use_amp}")
        
    def train(self, resume_from: Optional[str] = None):
        """Run training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume from
        """
        # Resume if requested
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            self.logger.info(f"Resumed from checkpoint: {resume_from}")
            
        self.logger.info("Starting training...")
        self.model.train()
        
        # Training metrics
        train_loss = 0.0
        grad_norm = 0.0
        samples_per_sec = 0.0
        start_time = time.time()
        last_log_step = self.global_step
        
        # Training loop
        while self.global_step < self.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with optional AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        loss = compute_loss(outputs['logits'], labels)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = compute_loss(outputs['logits'], labels)
                    
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                train_loss += loss.item()
                
                # Backward pass with optional AMP
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                    
                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % 10 == 0:
                        # Calculate metrics
                        elapsed = time.time() - start_time
                        steps_done = self.global_step - last_log_step
                        samples_per_sec = (steps_done * input_ids.shape[0]) / elapsed
                        
                        avg_loss = train_loss / steps_done
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        self.logger.info(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Grad Norm: {grad_norm:.3f} | "
                            f"Samples/sec: {samples_per_sec:.1f}"
                        )
                        
                        # Reset metrics
                        train_loss = 0.0
                        start_time = time.time()
                        last_log_step = self.global_step
                        
                    # Evaluation
                    if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                        val_loss, val_perplexity = self.evaluate()
                        self.logger.info(
                            f"Validation - Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}"
                        )
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
                            self.logger.info("Saved best model")
                            
                        # Generate sample
                        self.generate_sample()
                        
                        self.model.train()
                        
                    # Checkpointing
                    if self.global_step % self.save_interval == 0:
                        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
                        self.save_checkpoint(checkpoint_path)
                        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
                        
                    # Check if done
                    if self.global_step >= self.max_steps:
                        break
                        
            self.epoch += 1
            
        # Final checkpoint
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
        self.logger.info("Training completed!")
        
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set.
        
        Returns:
            Tuple of (validation loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
            
                # Reshape for loss computation
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                labels_flat = labels.reshape(-1)
            
            # Compute loss per token
                loss = F.cross_entropy(
                    logits_flat,
                    labels_flat,
                    ignore_index=-100,
                    reduction='none'
                )
            
            # Reshape loss back to (batch_size, seq_len)
                loss = loss.reshape(batch_size, seq_len)
            
            # Create mask for non-padded and non-ignored tokens
                mask = (labels != -100).float()
            
            # Sum loss only for valid tokens
                total_loss += (loss * mask).sum().item()
                total_tokens += mask.sum().item()
                
                # Count only non-padded tokens
                mask = (labels != -100).float()
                total_loss += (loss * mask).sum().item()
                total_tokens += mask.sum().item()
                
        # Calculate average loss and perplexity
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1000.0)  # Cap perplexity
        
        return avg_loss, perplexity
    
    def generate_sample(self, prompt: str = "Hello, how are you?"):
        """Generate a sample for inspection during training.
        
        Args:
            prompt: Prompt to generate from
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = generate(
                self.model,
                input_ids,
                max_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode
        generated_ids = output['generated_ids'][0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        self.logger.info(f"Sample generation:\n{generated_text}")
        
    def save_checkpoint(self, path: str):
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'n_heads': self.model.n_heads,
                'd_ff': self.model.d_ff,
                'max_seq_len': self.model.max_seq_len,
                'dropout': self.model.dropout,
                'tie_weights': self.model.tie_weights
            }
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])