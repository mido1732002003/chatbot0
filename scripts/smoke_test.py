#!/usr/bin/env python3
"""Smoke test to verify everything works."""

import sys
import torch
import json
from pathlib import Path
import subprocess
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import setup_logger
from utils.seed import set_seed


def run_command(cmd, logger):
    """Run a command and check for errors."""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(f"Error: {result.stderr}")
        return False
    return True


def main():
    """Run smoke test."""
    logger = setup_logger('smoke_test')
    logger.info("Starting smoke test...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create tiny config for smoke test
    smoke_config = {
        "data": {
            "train_path": "data/train.jsonl",
            "val_path": "data/val.jsonl",
            "tokenizer_path": "configs/tokenizer.json",
            "max_seq_len": 128,
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0,
            "include_loss_mask": True
        },
        "optimization": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_steps": 2,
            "max_steps": 5,
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 1,
            "use_amp": False
        },
        "scheduling": {
            "eval_interval": 2,
            "save_interval": 3,
            "log_interval": 1
        },
        "checkpoint": {
            "checkpoint_dir": "checkpoints/smoke_test",
            "resume_from": None
        },
        "seed": 42
    }
    
    smoke_model_config = {
        "vocab_size": 8192,
        "d_model": 128,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 256,
        "max_seq_len": 128,
        "dropout": 0.1,
        "tie_weights": True
    }
    
    # Save smoke test configs
    with open('configs/smoke_train.json', 'w') as f:
        json.dump(smoke_config, f, indent=2)
        
    with open('configs/smoke_model.json', 'w') as f:
        json.dump(smoke_model_config, f, indent=2)
        
    success = True
    
    # Step 1: Build tokenizer
    logger.info("\n=== Step 1: Building tokenizer ===")
    if not run_command("python scripts/build_tokenizer.py", logger):
        success = False
        
    # Step 2: Train model (tiny config)
    if success:
        logger.info("\n=== Step 2: Training model (smoke test config) ===")
        # Modify run_sft.py to use smoke configs
        from core.model import TransformerLM
        from training.trainer import Trainer
        from training.dataset import create_dataloader
        from utils.tokenizer import BytePairTokenizer
        
        tokenizer = BytePairTokenizer.load('configs/tokenizer.json')
        
        model = TransformerLM(
            vocab_size=len(tokenizer.vocab),
            d_model=smoke_model_config['d_model'],
            n_layers=smoke_model_config['n_layers'],
            n_heads=smoke_model_config['n_heads'],
            d_ff=smoke_model_config['d_ff'],
            max_seq_len=smoke_model_config['max_seq_len'],
            dropout=smoke_model_config['dropout'],
            tie_weights=smoke_model_config['tie_weights'],
            device=device
        )
        
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        train_loader = create_dataloader(
            data_path=smoke_config['data']['train_path'],
            tokenizer=tokenizer,
            batch_size=smoke_config['data']['batch_size'],
            max_seq_len=smoke_config['data']['max_seq_len'],
            shuffle=smoke_config['data']['shuffle']
        )
        
        val_loader = create_dataloader(
            data_path=smoke_config['data']['val_path'],
            tokenizer=tokenizer,
            batch_size=smoke_config['data']['batch_size'],
            max_seq_len=smoke_config['data']['max_seq_len'],
            shuffle=False
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            learning_rate=smoke_config['optimization']['learning_rate'],
            max_steps=smoke_config['optimization']['max_steps'],
            checkpoint_dir=smoke_config['checkpoint']['checkpoint_dir'],
            device=device,
            logger=logger
        )
        
        trainer.train()
        
    # Step 3: Test generation
    if success:
        logger.info("\n=== Step 3: Testing generation ===")
        from core.generation import generate
        
        model_path = Path(smoke_config['checkpoint']['checkpoint_dir']) / 'final_model.pt'
        if model_path.exists():
            model = TransformerLM.from_checkpoint(str(model_path), device=device)
            model.eval()
            
            test_prompt = "Hello, how are you?"
            input_ids = torch.tensor(
                [tokenizer.bos_token_id] + tokenizer.encode(test_prompt),
                dtype=torch.long
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = generate(
                    model,
                    input_ids,
                    max_tokens=20,
                    temperature=0.8
                )
                
            generated_text = tokenizer.decode(output['generated_ids'][0].tolist())
            logger.info(f"Generated: {generated_text}")
        else:
            logger.error("Model checkpoint not found!")
            success = False
            
    # Step 4: Test evaluation
    if success:
        logger.info("\n=== Step 4: Testing evaluation ===")
        from evaluation.metrics import compute_distinct_n
        
        test_texts = ["Hello world", "How are you", "Nice day today"]
        distinct_1 = compute_distinct_n(test_texts, n=1)
        logger.info(f"Distinct-1: {distinct_1:.4f}")
        
    # Step 5: Test safety filter
    if success:
        logger.info("\n=== Step 5: Testing safety filter ===")
        from alignment.safety_filter import SafetyFilter
        
        safety_filter = SafetyFilter()
        safe_text = "Hello, how are you?"
        unsafe_text = "How to make a bomb"  # Test pattern
        
        safe_check = safety_filter.check_safety(safe_text)
        unsafe_check = safety_filter.check_safety(unsafe_text)
        
        logger.info(f"Safe text check: {safe_check['is_safe']}")
        logger.info(f"Unsafe text check: {unsafe_check['is_safe']}")
        
    # Cleanup
    logger.info("\n=== Cleaning up ===")
    import shutil
    if Path('checkpoints/smoke_test').exists():
        shutil.rmtree('checkpoints/smoke_test')
    if Path('configs/smoke_train.json').exists():
        Path('configs/smoke_train.json').unlink()
    if Path('configs/smoke_model.json').exists():
        Path('configs/smoke_model.json').unlink()
        
    if success:
        logger.info("\n✅ SMOKE TEST PASSED!")
    else:
        logger.error("\n❌ SMOKE TEST FAILED!")
        sys.exit(1)
        

if __name__ == '__main__':
    main()