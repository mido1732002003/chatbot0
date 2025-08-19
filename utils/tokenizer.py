"""Byte-level BPE tokenizer implementation from scratch."""

import json
import re
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


class BytePairTokenizer:
    """Byte-level BPE tokenizer implemented from scratch.
    
    Implements byte-level Byte Pair Encoding for robust text tokenization.
    Handles arbitrary text including emojis and special characters.
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None
    ):
        """Initialize tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for merges
            special_tokens: Dictionary of special tokens
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Initialize special tokens
        self.special_tokens = special_tokens or {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
            '<|user|>': 4,
            '<|assistant|>': 5
        }
        
        # Reserve IDs for special tokens
        self.special_token_ids = {token: idx for token, idx in self.special_tokens.items()}
        self.id_to_special = {idx: token for token, idx in self.special_tokens.items()}
        
        # Initialize byte-level base vocabulary (256 bytes)
        self.byte_to_token = {}
        self.token_to_byte = {}
        
        # Start token IDs after special tokens
        token_id = len(self.special_tokens)
        
        # Create base byte vocabulary
        for byte_val in range(256):
            byte_char = chr(byte_val)
            self.byte_to_token[byte_char] = token_id
            self.token_to_byte[token_id] = byte_char
            token_id += 1
            
        # Merge rules learned during training
        self.merges = []
        self.merge_dict = {}
        
        # Final vocabulary
        self.vocab = {**self.special_token_ids, **self.byte_to_token}
        self.id_to_token = {**self.id_to_special, **self.token_to_byte}
        
        # Cache for faster encoding
        self.cache = {}
        
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.special_token_ids['<pad>']
        
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.special_token_ids['<bos>']
        
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.special_token_ids['<eos>']
        
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.special_token_ids['<unk>']
        
    @property
    def eos_token(self) -> str:
        """Get end of sequence token string."""
        return '<eos>'
        
    def _get_byte_pairs(self, word_bytes: List[str]) -> Counter:
        """Get all adjacent byte pairs in a word.
        
        Args:
            word_bytes: List of byte tokens
            
        Returns:
            Counter of byte pairs
        """
        pairs = Counter()
        for i in range(len(word_bytes) - 1):
            pairs[(word_bytes[i], word_bytes[i + 1])] += 1
        return pairs
        
    def _merge_pair(self, pair: Tuple[str, str], word_bytes: List[str]) -> List[str]:
        """Merge a byte pair in a word.
        
        Args:
            pair: Byte pair to merge
            word_bytes: List of byte tokens
            
        Returns:
            Word with merged pair
        """
        if len(word_bytes) < 2:
            return word_bytes
            
        merged = []
        i = 0
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i + 1]) == pair:
                merged.append(word_bytes[i] + word_bytes[i + 1])
                i += 2
            else:
                merged.append(word_bytes[i])
                i += 1
                
        return merged
        
    def train(self, texts: List[str]):
        """Train BPE tokenizer on texts.
        
        Args:
            texts: List of training texts
        """
        print(f"Training tokenizer on {len(texts)} texts...")
        
        # Convert texts to bytes and count frequencies
        word_freqs = Counter()
        
        for text in texts:
            # Handle special tokens
            for special_token in self.special_tokens:
                text = text.replace(special_token, f' {special_token} ')
                
            # Split into words (simple whitespace + punctuation)
            words = re.findall(r'\S+|\s+', text)
            
            for word in words:
                if word in self.special_tokens:
                    continue
                    
                # Convert to bytes
                word_bytes = [chr(b) for b in word.encode('utf-8')]
                word_bytes_tuple = tuple(word_bytes)
                word_freqs[word_bytes_tuple] += 1
                
        # Learn merges
        vocab = dict(word_freqs)
        num_merges = self.vocab_size - len(self.byte_to_token) - len(self.special_tokens)
        
        print(f"Learning {num_merges} merges...")
        
        for merge_idx in range(num_merges):
            # Count all pairs
            pair_freqs = Counter()
            
            for word_bytes, freq in vocab.items():
                if len(word_bytes) < 2:
                    continue
                    
                for i in range(len(word_bytes) - 1):
                    pair = (word_bytes[i], word_bytes[i + 1])
                    pair_freqs[pair] += freq
                    
            # Find most frequent pair
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            if pair_freqs[best_pair] < self.min_frequency:
                break
                
            # Add merge rule
            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            
            # Update vocabulary with merged words
            new_vocab = {}
            for word_bytes, freq in vocab.items():
                word_list = list(word_bytes)
                word_list = self._merge_pair(best_pair, word_list)
                new_vocab[tuple(word_list)] = freq
                
            vocab = new_vocab
            
            # Add to token vocabulary
            token_id = len(self.vocab)
            self.vocab[merged_token] = token_id
            self.id_to_token[token_id] = merged_token
            self.merge_dict[best_pair] = merged_token
            
            if (merge_idx + 1) % 100 == 0:
                print(f"  Learned {merge_idx + 1} merges...")
                
        print(f"Tokenizer training complete. Vocabulary size: {len(self.vocab)}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        # Check cache
        if text in self.cache:
            return self.cache[text]
            
        tokens = []
        
        # Handle special tokens
        parts = [text]
        for special_token in self.special_tokens:
            new_parts = []
            for part in parts:
                if special_token in part:
                    split_parts = part.split(special_token)
                    for i, split_part in enumerate(split_parts):
                        if split_part:
                            new_parts.append(split_part)
                        if i < len(split_parts) - 1:
                            new_parts.append(special_token)
                else:
                    new_parts.append(part)
            parts = new_parts
            
        # Encode each part
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.special_token_ids[part])
            else:
                # Convert to bytes
                part_bytes = [chr(b) for b in part.encode('utf-8')]
                
                # Apply merges
                for merge in self.merges:
                    part_bytes = self._merge_pair(merge, part_bytes)
                    
                # Convert to token IDs
                for token in part_bytes:
                    if token in self.vocab:
                        tokens.append(self.vocab[token])
                    else:
                        # This shouldn't happen with byte-level encoding
                        tokens.append(self.unk_token_id)
                        
        # Cache result
        if len(self.cache) < 10000:  # Limit cache size
            self.cache[text] = tokens
            
        return tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            elif token_id in self.id_to_special:
                tokens.append(self.id_to_special[token_id])
                
        # Join tokens
        text = ''.join(tokens)
        
        # Convert bytes back to text
        byte_array = []
        for token in tokens:
            if token in self.special_tokens:
                # Add special token as-is
                if byte_array:
                    # Decode accumulated bytes first
                    try:
                        text_part = bytes([ord(b) for b in byte_array]).decode('utf-8', errors='replace')
                        byte_array = []
                    except:
                        text_part = ''.join(byte_array)
                        byte_array = []
                else:
                    text_part = ''
                text = text_part + token
            else:
                # Accumulate bytes
                for char in token:
                    byte_array.append(char)
                    
        # Decode remaining bytes
        if byte_array:
            try:
                text = bytes([ord(b) for b in byte_array]).decode('utf-8', errors='replace')
            except:
                text = ''.join(byte_array)
                
        return text
        
    def save(self, path: str):
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        save_data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'merges': [list(merge) for merge in self.merges],
            'vocab': self.vocab,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'BytePairTokenizer':
        """Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
            
        Returns:
            Loaded tokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
            
        tokenizer = cls(
            vocab_size=save_data['vocab_size'],
            min_frequency=save_data['min_frequency'],
            special_tokens=save_data['special_tokens']
        )
        
        tokenizer.merges = [tuple(merge) for merge in save_data['merges']]
        tokenizer.vocab = save_data['vocab']
        tokenizer.id_to_token = {int(k): v for k, v in save_data['id_to_token'].items()}
        
        # Rebuild merge dict
        for merge in tokenizer.merges:
            merged_token = merge[0] + merge[1]
            tokenizer.merge_dict[merge] = merged_token
            
        return tokenizer