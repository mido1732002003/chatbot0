#!/usr/bin/env python3
"""Evaluate model performance."""

import json
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import TransformerLM
from evaluation.evaluator import Evaluator
from training.dataset import create_dataloader
from utils.tokenizer import BytePairTokenizer
from utils.logging_utils import setup_logger


def main():
    """Run model evaluation."""
    logger = setup_logger('evaluation')
    
    # Load configuration
    config_path = Path('configs/eval.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BytePairTokenizer.load(config['tokenizer_path'])
    
    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerLM.from_checkpoint(
        config['model_checkpoint'],
        device=device
    )
    
    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        logger=logger
    )
    
    # Create validation data loader
    val_loader = None
    if 'val_path' in config['data']:
        logger.info("Loading validation data...")
        val_loader = create_dataloader(
            data_path=config['data']['val_path'],
            tokenizer=tokenizer,
            batch_size=config['data']['batch_size'],
            max_seq_len=config['data']['max_seq_len'],
            shuffle=False,
            num_workers=0,
            include_loss_mask=True
        )
        
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_all(
        val_dataloader=val_loader,
        test_prompts=config['generation']['test_prompts'],
        generation_config=config['generation']
    )
    
    # Save results
    output_path = Path(config['output_path'])
    evaluator.save_results(results, output_path)
    
    logger.info("Evaluation complete!")
    

if __name__ == '__main__':
    main()