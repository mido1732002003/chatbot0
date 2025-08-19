#!/usr/bin/env python3
"""Run supervised fine-tuning."""

import json
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import TransformerLM
from training.trainer import Trainer
from training.dataset import create_dataloader
from utils.tokenizer import BytePairTokenizer
from utils.seed import set_seed
from utils.logging_utils import setup_logger


def main():
    """Run SFT training."""
    logger = setup_logger('sft_training')
    
    # Load configuration
    config_path = Path('configs/train.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model_config_path = Path('configs/model.json')
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
        
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_path = Path(config['data']['tokenizer_path'])
    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at {tokenizer_path}. Run build_tokenizer.py first!")
        return
        
    tokenizer = BytePairTokenizer.load(tokenizer_path)
    
    # Create model
    logger.info("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        tie_weights=model_config['tie_weights'],
        device=device
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dataloader(
        data_path=config['data']['train_path'],
        tokenizer=tokenizer,
        batch_size=config['data']['batch_size'],
        max_seq_len=config['data']['max_seq_len'],
        shuffle=config['data']['shuffle'],
        num_workers=config['data']['num_workers'],
        include_loss_mask=config['data']['include_loss_mask']
    )
    
    val_loader = create_dataloader(
        data_path=config['data']['val_path'],
        tokenizer=tokenizer,
        batch_size=config['data']['batch_size'],
        max_seq_len=config['data']['max_seq_len'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        include_loss_mask=config['data']['include_loss_mask']
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=config['optimization']['learning_rate'],
        weight_decay=config['optimization']['weight_decay'],
        warmup_steps=config['optimization']['warmup_steps'],
        max_steps=config['optimization']['max_steps'],
        gradient_clip=config['optimization']['gradient_clip'],
        gradient_accumulation_steps=config['optimization']['gradient_accumulation_steps'],
        eval_interval=config['scheduling']['eval_interval'],
        save_interval=config['scheduling']['save_interval'],
        checkpoint_dir=config['checkpoint']['checkpoint_dir'],
        use_amp=config['optimization']['use_amp'],
        device=device,
        logger=logger
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from=config['checkpoint']['resume_from'])
    
    logger.info("Training complete!")
    

if __name__ == '__main__':
    main()