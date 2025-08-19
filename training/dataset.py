"""Dataset and data loading utilities for training."""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random


class ChatDataset(Dataset):
    """Dataset for chat-style supervised fine-tuning.
    
    Expects JSONL files with 'prompt' and 'response' fields.
    Handles tokenization, padding, and preparation for language modeling.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 256,
        include_loss_mask: bool = True
    ):
        """Initialize chat dataset.
        
        Args:
            data_path: Path to JSONL data file
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            include_loss_mask: Whether to mask prompt tokens in loss
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.include_loss_mask = include_loss_mask
        
        # Load data
        self.examples = self._load_data(data_path)
        
        # Get special token IDs
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load data from JSONL file.
        
        Args:
            data_path: Path to JSONL file
            
        Returns:
            List of examples with 'prompt' and 'response' fields
        """
        examples = []
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                # Validate required fields
                        if 'prompt' in example and 'response' in example:
                            examples.append(example)
                        else:
                            print(f"Warning: Skipping invalid example at line {i}: {example}")
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON at line {i}")

#        with open(path, 'r', encoding='utf-8') as f:
 #           for line in f:
  #              line = line.strip()
   #             if line:
    #                example = json.loads(line)
     #               # Validate required fields
      #              if 'prompt' in example and 'response' in example:
       #                 examples.append(example)
        #            else:
         #               print(f"Warning: Skipping invalid example: {example}")
                        
        return examples
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.examples)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]
        prompt = example['prompt']
        response = example['response']
    
    # Format as conversation
        formatted_text = f"<|user|> {prompt} <|assistant|> {response}"
    
    # Tokenize
        tokens = self.tokenizer.encode(formatted_text)
    
    # Add BOS and EOS tokens
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
    
    # Truncate if necessary
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            tokens[-1] = self.eos_token_id
    
    # Pad to max_seq_len
        num_pad = self.max_seq_len - len(tokens)
        if num_pad > 0:
            tokens = tokens + [self.pad_token_id] * num_pad
    
    # For language modeling: input is tokens[:-1], labels are tokens[1:]
        input_ids = tokens[:-1]  # Length: max_seq_len - 1
        labels = tokens[1:]      # Length: max_seq_len - 1
    
    # Create attention mask for input_ids (same length as input_ids!)
        attention_mask = [1] * (len(tokens) - num_pad - 1) + [0] * num_pad
    # Ensure attention_mask has same length as input_ids
        if len(attention_mask) < len(input_ids):
            attention_mask = attention_mask + [0] * (len(input_ids) - len(attention_mask))
        elif len(attention_mask) > len(input_ids):
            attention_mask = attention_mask[:len(input_ids)]
    
    # Apply loss masking if needed
        if self.include_loss_mask:
            assistant_marker = "<|assistant|>"
            prompt_with_marker = f"<|user|> {prompt} {assistant_marker}"
            prompt_tokens = self.tokenizer.encode(prompt_with_marker)
            prompt_len = len(prompt_tokens) + 1  # +1 for BOS
        
        # Mask prompt tokens in labels
        labels_masked = labels.copy()
        for i in range(min(prompt_len, len(labels_masked))):
            if i < len(labels_masked) and labels_masked[i] != self.pad_token_id:
                labels_masked[i] = -100
        labels = labels_masked
    
    # Final verification
        assert len(input_ids) == len(attention_mask), f"Length mismatch: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}"
        assert len(input_ids) == len(labels), f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}"
    
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }



def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Batched tensors
    """
    # Stack all tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_seq_len: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    include_loss_mask: bool = True
) -> DataLoader:
    """Create DataLoader for training.
    
    Args:
        data_path: Path to data file
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        include_loss_mask: Whether to mask prompt tokens in loss
        
    Returns:
        DataLoader instance
    """
    dataset = ChatDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        include_loss_mask=include_loss_mask
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader