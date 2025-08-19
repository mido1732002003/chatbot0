"""Evaluation metrics for language models."""

import torch
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np


def compute_perplexity_dataset(
    model,
    dataloader,
    device: torch.device
) -> Tuple[float, float]:
    """Compute perplexity over entire dataset.
    
    Args:
        model: Language model
        dataloader: Data loader with validation data
        device: Device to run on
        
    Returns:
        Tuple of (average loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Compute loss
            shift_logits = logits.reshape(-1, logits.size(-1))
            shift_labels = labels.reshape(-1)
            
            # Calculate cross-entropy
            loss = torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction='none'
            )
            
            # Only count non-ignored tokens
            mask = (shift_labels != -100)
            total_loss += loss[mask].sum().item()
            total_tokens += mask.sum().item()
            
    # Calculate perplexity
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    
    # Cap perplexity to avoid overflow
    perplexity = min(perplexity, 1000.0)
    
    return avg_loss, perplexity


def compute_distinct_n(texts: List[str], n: int = 1) -> float:
    """Compute distinct-n metric for diversity.
    
    Measures the ratio of unique n-grams to total n-grams.
    Higher values indicate more diverse text.
    
    Args:
        texts: List of generated texts
        n: N-gram size (1, 2, or 3 typically)
        
    Returns:
        Distinct-n score between 0 and 1
    """
    if not texts:
        return 0.0
        
    all_ngrams = []
    
    for text in texts:
        # Tokenize by whitespace (simple approach)
        tokens = text.lower().split()
        
        # Extract n-grams
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            all_ngrams.append(ngram)
            
    if not all_ngrams:
        return 0.0
        
    # Count unique n-grams
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def compute_repetition_rate(text: str, n: int = 3) -> float:
    """Compute repetition rate in generated text.
    
    Measures how often n-grams are repeated.
    Lower values indicate less repetition.
    
    Args:
        text: Generated text
        n: N-gram size
        
    Returns:
        Repetition rate
    """
    tokens = text.lower().split()
    
    if len(tokens) < n:
        return 0.0
        
    ngram_counts = Counter()
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_counts[ngram] += 1
        
    # Calculate repetition rate
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    total = len(ngram_counts)
    
    return repeated / total if total > 0 else 0.0


def compute_average_length(texts: List[str]) -> float:
    """Compute average length of generated texts.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Average length in tokens
    """
    if not texts:
        return 0.0
        
    lengths = [len(text.split()) for text in texts]
    return sum(lengths) / len(lengths)


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    n: int = 4
) -> float:
    """Compute BLEU score for generated texts.
    
    Simplified BLEU implementation for single references.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        n: Maximum n-gram order
        
    Returns:
        BLEU score between 0 and 1
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
        
    if not predictions:
        return 0.0
        
    # Compute precision for each n-gram order
    precisions = []
    
    for order in range(1, n + 1):
        matches = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # Get n-grams
            pred_ngrams = Counter()
            ref_ngrams = Counter()
            
            for i in range(len(pred_tokens) - order + 1):
                ngram = tuple(pred_tokens[i:i + order])
                pred_ngrams[ngram] += 1
                
            for i in range(len(ref_tokens) - order + 1):
                ngram = tuple(ref_tokens[i:i + order])
                ref_ngrams[ngram] += 1
                
            # Count matches (clip by reference counts)
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
                
            total += sum(pred_ngrams.values())
            
        # Calculate precision
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
        
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precision = sum(np.log(p) for p in precisions) / len(precisions)
        bleu = np.exp(log_precision)
    else:
        bleu = 0.0
        
    # Brevity penalty
    pred_len = sum(len(p.split()) for p in predictions)
    ref_len = sum(len(r.split()) for r in references)
    
    if pred_len < ref_len:
        brevity_penalty = np.exp(1 - ref_len / pred_len)
    else:
        brevity_penalty = 1.0
        
    return bleu * brevity_penalty


def compute_coherence_score(
    texts: List[str],
    model,
    tokenizer,
    device: torch.device
) -> float:
    """Compute coherence score using perplexity of generated texts.
    
    Lower perplexity indicates more coherent text.
    
    Args:
        texts: List of generated texts
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        
    Returns:
        Average coherence score
    """
    if not texts:
        return 0.0
        
    model.eval()
    total_perplexity = 0.0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
            input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
            labels = torch.tensor(tokens[1:]).unsqueeze(0).to(device)
            
            # Get model predictions
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Compute perplexity
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction='mean'
            )
            
            perplexity = torch.exp(loss).item()
            total_perplexity += min(perplexity, 1000.0)  # Cap to avoid overflow
            
    return total_perplexity / len(texts)