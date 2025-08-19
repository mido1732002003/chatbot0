#!/usr/bin/env python3
"""Start HTTP API server."""

import json
import torch
from pathlib import Path
import sys
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from serving.api_server import APIServer
from utils.tokenizer import BytePairTokenizer
from utils.logging_utils import setup_logger


def main():
    """Start API server."""
    parser = argparse.ArgumentParser(description='Start HTTP API server')
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
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Host to bind to (overrides config)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Port to listen on (overrides config)'
    )
    args = parser.parse_args()
    
    logger = setup_logger('api_server')
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Override host/port if provided
    host = args.host or config['api']['host']
    port = args.port or config['api']['port']
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = BytePairTokenizer.load(config['tokenizer_path'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create and start server
    logger.info("Starting API server...")
    server = APIServer(
        model_path=args.model,
        tokenizer=tokenizer,
        host=host,
        port=port,
        enable_safety=config['api']['enable_safety'],
        device=device,
        logger=logger
    )
    
    server.start()
    

if __name__ == '__main__':
    main()