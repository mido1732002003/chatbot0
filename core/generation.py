"""Text generation utilities with various sampling strategies."""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import math


def generate(
    model,
    input_ids: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate text using various sampling strategies.
    
    Args:
        model: The language model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens with highest probability
        top_p: Keep tokens with cumulative probability <= p (nucleus sampling)
        repetition_penalty: Penalty for repeating tokens
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        use_cache: Whether to use key-value caching for efficiency
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - generated_ids: Generated token IDs
            - generated_tokens: Number of tokens generated
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model.eval()
    device = next(model.parameters()).device
    batch_size = input_ids.shape[0]
    
    # Move input to model device
    input_ids = input_ids.to(device)
    
    # Initialize generation
    generated_ids = input_ids.clone()
    past_key_values = None
    
    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Get model predictions
            if use_cache and past_key_values is not None:
                # Use only the last token as input with cached keys/values
                model_inputs = generated_ids[:, -1:] if step > 0 else generated_ids
            else:
                model_inputs = generated_ids
                
            outputs = model(
                model_inputs,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            
            logits = outputs['logits']
            if use_cache:
                past_key_values = outputs.get('past_key_values')
                
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Gather all generated tokens
                for i in range(batch_size):
                    for token_id in generated_ids[i].tolist():
                        if token_id != pad_token_id:
                            # Apply penalty
                            if next_token_logits[i, token_id] > 0:
                                next_token_logits[i, token_id] /= repetition_penalty
                            else:
                                next_token_logits[i, token_id] *= repetition_penalty
                                
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                # Keep only top k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                # Set filtered tokens to -inf
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
                    
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Update finished sequences
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)
                
            # Replace tokens for finished sequences with padding
            if pad_token_id is not None:
                next_tokens = torch.where(finished, pad_token_id, next_tokens)
                
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
                
    return {
        'generated_ids': generated_ids,
        'generated_tokens': generated_ids.shape[1] - input_ids.shape[1]
    }


def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 4,
    max_tokens: int = 100,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    length_penalty: float = 1.0,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Beam search decoding for better quality generation.
    
    Args:
        model: The language model
        input_ids: Input token IDs of shape (1, seq_len) - single sequence only
        beam_size: Number of beams
        max_tokens: Maximum number of tokens to generate
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        length_penalty: Length penalty factor
        use_cache: Whether to use key-value caching
        
    Returns:
        Dictionary containing:
            - generated_ids: Best generated sequence
            - scores: Beam scores
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Beam search only supports batch size 1
    assert input_ids.shape[0] == 1, "Beam search only supports batch size 1"
    
    input_ids = input_ids.to(device)
    batch_size = 1
    vocab_size = model.vocab_size
    
    # Expand input for beam size
    input_ids = input_ids.repeat(beam_size, 1)
    
    # Initialize beams
    beam_scores = torch.zeros(beam_size, device=device)
    beam_scores[1:] = float('-inf')  # Only first beam is active initially
    
    # Track finished beams
    finished_sequences = []
    finished_scores = []
    
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Get model predictions
            if use_cache and past_key_values is not None:
                model_inputs = input_ids[:, -1:] if step > 0 else input_ids
            else:
                model_inputs = input_ids
                
            outputs = model(
                model_inputs,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            
            logits = outputs['logits']
            if use_cache:
                past_key_values = outputs.get('past_key_values')
                
            # Get log probabilities for next token
            next_token_logits = logits[:, -1, :]  # Shape: (beam_size, vocab_size)
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores to get cumulative scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            
            # Flatten to consider all beam * vocab_size candidates
            next_token_scores = next_token_scores.view(-1)
            
            # Get top beam_size candidates
            next_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, largest=True, sorted=True
            )
            
            # Get beam indices
            next_beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Select beams for next step
            active_beams = []
            active_scores = []
            
            for i in range(len(next_scores)):
                beam_idx = next_beam_indices[i]
                token_id = next_tokens[i]
                score = next_scores[i]
                
                # Check if this beam ends with EOS
                if eos_token_id is not None and token_id == eos_token_id:
                    # Add to finished sequences
                    seq = torch.cat([input_ids[beam_idx], token_id.unsqueeze(0)])
                    seq_len = seq.shape[0] - 1  # Exclude initial input
                    # Apply length penalty
                    normalized_score = score / (seq_len ** length_penalty)
                    finished_sequences.append(seq)
                    finished_scores.append(normalized_score)
                else:
                    # Continue with this beam
                    if len(active_beams) < beam_size:
                        active_beams.append((beam_idx.item(), token_id.item()))
                        active_scores.append(score.item())
                        
            # Update beams
            if len(active_beams) == 0:
                break  # All beams finished
                
            # Prepare next step inputs
            new_input_ids = []
            new_beam_scores = []
            
            for (beam_idx, token_id), score in zip(active_beams, active_scores):
                seq = torch.cat([input_ids[beam_idx], torch.tensor([token_id], device=device)])
                new_input_ids.append(seq)
                new_beam_scores.append(score)
                
            input_ids = torch.stack(new_input_ids)
            beam_scores = torch.tensor(new_beam_scores, device=device)
            
            # Reorder cached states if using cache
            if use_cache and past_key_values is not None:
                reordered_past = []
                for layer_past in past_key_values:
                    reordered_layer_past = []
                    for past_state in layer_past:
                        beam_indices_tensor = torch.tensor(
                            [b[0] for b in active_beams], device=device
                        )
                        reordered_layer_past.append(past_state[beam_indices_tensor])
                    reordered_past.append(tuple(reordered_layer_past))
                past_key_values = tuple(reordered_past)
                
    # Return best sequence
    if finished_sequences:
        best_idx = torch.tensor(finished_scores).argmax()
        best_sequence = finished_sequences[best_idx]
    else:
        # No sequence finished, return best active beam
        best_sequence = input_ids[0]
        
    return {
        'generated_ids': best_sequence.unsqueeze(0),
        'generated_tokens': best_sequence.shape[0] - 1
    }