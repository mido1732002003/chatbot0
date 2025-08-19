"""Main evaluator class for comprehensive model evaluation."""

import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from .metrics import (
    compute_perplexity_dataset,
    compute_distinct_n,
    compute_average_length,
    compute_repetition_rate,
    compute_coherence_score
)
from core.generation import generate
from utils.logging_utils import setup_logger


class Evaluator:
    """Comprehensive model evaluator.
    
    Evaluates language models on various metrics including perplexity,
    diversity, repetition, and generation quality.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer instance
            device: Device to run evaluation on
            logger: Logger instance
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logger
        if logger is None:
            self.logger = setup_logger('evaluator')
        else:
            self.logger = logger
            
    def evaluate_perplexity(self, dataloader) -> Dict[str, float]:
        """Evaluate perplexity on dataset.
        
        Args:
            dataloader: Data loader with evaluation data
            
        Returns:
            Dictionary with loss and perplexity
        """
        self.logger.info("Evaluating perplexity...")
        
        loss, perplexity = compute_perplexity_dataset(
            self.model,
            dataloader,
            self.device
        )
        
        return {
            'loss': loss,
            'perplexity': perplexity
        }
    
    def evaluate_generation(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> Dict[str, Any]:
        """Evaluate generation quality.
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_samples: Number of samples per prompt
            
        Returns:
            Dictionary with generation metrics
        """
        self.logger.info(f"Evaluating generation on {len(prompts)} prompts...")
        
        all_generations = []
        generation_times = []
        
        for prompt in prompts:
            prompt_generations = []
            
            for _ in range(num_samples):
                # Tokenize prompt
                input_ids = torch.tensor(
                    [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt),
                    dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                # Generate
                start_time = time.time()
                
                output = generate(
                    self.model,
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                # Decode
                generated_ids = output['generated_ids'][0]
                # Remove prompt tokens
                generated_ids = generated_ids[input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids.tolist())
                
                prompt_generations.append(generated_text)
                all_generations.append(generated_text)
                
        # Compute metrics
        distinct_1 = compute_distinct_n(all_generations, n=1)
        distinct_2 = compute_distinct_n(all_generations, n=2)
        distinct_3 = compute_distinct_n(all_generations, n=3)
        
        avg_length = compute_average_length(all_generations)
        
        # Compute repetition rate for each generation
        repetition_rates = [
            compute_repetition_rate(text, n=3)
            for text in all_generations
        ]
        avg_repetition = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
        
        # Compute coherence
        coherence_score = compute_coherence_score(
            all_generations[:10],  # Use subset for efficiency
            self.model,
            self.tokenizer,
            self.device
        )
        
        # Average generation time
        avg_generation_time = sum(generation_times) / len(generation_times)
        tokens_per_second = max_tokens / avg_generation_time
        
        results = {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'average_length': avg_length,
            'repetition_rate': avg_repetition,
            'coherence_score': coherence_score,
            'avg_generation_time': avg_generation_time,
            'tokens_per_second': tokens_per_second,
            'num_samples': len(all_generations),
            'sample_generations': all_generations[:5]  # Include some samples
        }
        
        return results
    
    def evaluate_all(
        self,
        val_dataloader,
        test_prompts: List[str],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run all evaluations.
        
        Args:
            val_dataloader: Validation data loader
            test_prompts: Prompts for generation evaluation
            generation_config: Configuration for generation
            
        Returns:
            Complete evaluation results
        """
        self.logger.info("Running complete evaluation...")
        
        # Default generation config
        if generation_config is None:
            generation_config = {
                'max_tokens': 100,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'num_samples': 1
            }
            
        results = {}
        
        # Perplexity evaluation
        if val_dataloader is not None:
            perplexity_results = self.evaluate_perplexity(val_dataloader)
            results['perplexity'] = perplexity_results
            self.logger.info(f"Perplexity: {perplexity_results['perplexity']:.2f}")
            
        # Generation evaluation
        generation_results = self.evaluate_generation(
            test_prompts,
            **generation_config
        )
        results['generation'] = generation_results
        
        # Log generation metrics
        self.logger.info("Generation Metrics:")
        self.logger.info(f"  Distinct-1: {generation_results['distinct_1']:.4f}")
        self.logger.info(f"  Distinct-2: {generation_results['distinct_2']:.4f}")
        self.logger.info(f"  Distinct-3: {generation_results['distinct_3']:.4f}")
        self.logger.info(f"  Average Length: {generation_results['average_length']:.1f} tokens")
        self.logger.info(f"  Repetition Rate: {generation_results['repetition_rate']:.4f}")
        self.logger.info(f"  Coherence Score: {generation_results['coherence_score']:.2f}")
        self.logger.info(f"  Tokens/Second: {generation_results['tokens_per_second']:.1f}")
        
        # Log sample generations
        self.logger.info("\nSample Generations:")
        for i, text in enumerate(generation_results['sample_generations'], 1):
            self.logger.info(f"  {i}. {text[:100]}...")  # First 100 chars
            
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_path}")