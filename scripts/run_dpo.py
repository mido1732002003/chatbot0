#!/usr/bin/env python3
"""Run DPO alignment training."""

import json
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import TransformerLM
from alignment.dpo import DPOTrainer
from utils.tokenizer import BytePairTokenizer
from utils.seed import set_seed
from utils.logging_utils import setup_logger


def main():
    """Run DPO training."""
    logger = setup_logger('dpo_training')
    
    # Load configuration
    config_path = Path('configs/alignment.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    dpo_config = config['dpo']
    
    # Set seed
    set_seed(42)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BytePairTokenizer.load('configs/tokenizer.json')
    
    # Load models
    logger.info("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Policy model (will be trained)
    policy_model = TransformerLM.from_checkpoint(
        dpo_config['reference_model_path'],
        device=device
    )
    
    # Reference model (frozen)
    reference_model = TransformerLM.from_checkpoint(
        dpo_config['reference_model_path'],
        device=device
    )
    
    # Create DPO trainer
    logger.info("Creating DPO trainer...")
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        beta=dpo_config['beta'],
        learning_rate=dpo_config['learning_rate'],
        weight_decay=dpo_config['weight_decay'],
        max_steps=dpo_config['max_steps'],
        batch_size=dpo_config['batch_size'],
        gradient_clip=dpo_config['gradient_clip'],
        device=device,
        logger=logger
    )
    
    # Train
    logger.info("Starting DPO training...")
    trainer.train(
        data_path=dpo_config['preference_data_path'],
        output_dir=dpo_config['output_dir']
    )
    
    logger.info("DPO training complete!")
    

if __name__ == '__main__':
    main()