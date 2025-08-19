"""Build BPE tokenizer from training data."""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.tokenizer import BytePairTokenizer
from utils.logging_utils import setup_logger


def main():
    """Build tokenizer from training data."""
    logger = setup_logger('build_tokenizer')
    
    # Paths
    train_path = Path('data/train.jsonl')
    val_path = Path('data/val.jsonl')
    output_path = Path('configs/tokenizer.json')
    
    logger.info("Loading training data...")
    
    # Load texts
    texts = []
    
    for path in [train_path, val_path]:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            texts.append(data['prompt'])
                            texts.append(data['response'])
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSON at line {i} in {path}")
                        except KeyError:
                            logger.warning(f"Skipping line {i} missing 'prompt' or 'response' in {path}")
                    
    logger.info(f"Loaded {len(texts)} texts")
    
    # Create and train tokenizer
    logger.info("Training tokenizer...")
    tokenizer = BytePairTokenizer(vocab_size=8192, min_frequency=2)
    tokenizer.train(texts)
    
    # Test tokenizer
    test_text = "Hello, how are you? <|user|> Test <|assistant|> Response"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    logger.info(f"Test encoding: {test_text}")
    logger.info(f"Encoded: {encoded[:20]}...")
    logger.info(f"Decoded: {decoded}")
    
    # Save tokenizer
    tokenizer.save(output_path)
    logger.info(f"Tokenizer saved to {output_path}")
    

if __name__ == '__main__':
    main()
