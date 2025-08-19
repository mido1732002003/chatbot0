#!/usr/bin/env python3
"""Start CLI chat interface."""

import json
import torch
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from serving.cli_chat import ChatInterface
from utils.tokenizer import BytePairTokenizer
from utils.logging_utils import setup_logger


def main():
    """Start chat interface."""
    parser = argparse.ArgumentParser(description='Start CLI chat interface')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/sft/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/serving.json',
        help='Path to serving config'
    )
    args = parser.parse_args()
    
    logger = setup_logger('chat_cli')
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BytePairTokenizer.load(config['tokenizer_path'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create chat interface
    logger.info("Starting chat interface...")
    chat = ChatInterface(
        model_path=args.model,
        tokenizer=tokenizer,
        max_history=config['cli']['max_history'],
        generation_config=config['generation'],
        device=device,
        logger=logger
    )
    
    # Start chat loop
    chat.chat_loop()
    

if __name__ == '__main__':
    main()