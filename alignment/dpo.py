"""Direct Preference Optimization (DPO) for RLHF."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, Optional, Tuple, List
import json
from pathlib import Path
import copy

from utils.logging_utils import setup_logger


class DPOTrainer:
    """Trainer for Direct Preference Optimization.
    
    Implements DPO algorithm for aligning language models with human preferences
    using pairwise preference data.
    """
    
    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        max_steps: int = 100,
        batch_size: int = 1,
        gradient_clip: float = 1.0,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """Initialize DPO trainer.
        
        Args:
            policy_model: Model to train (policy)
            reference_model: Fixed reference model (e.g., SFT checkpoint)
            tokenizer: Tokenizer instance
            beta: DPO beta parameter (controls KL regularization)
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            max_steps: Maximum training steps
            batch_size: Batch size
            gradient_clip: Gradient clipping value
            device: Device to train on
            logger: Logger instance
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move models to device
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # Setup optimizer (only for policy model)
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup logger
        if logger is None:
            self.logger = setup_logger('dpo_trainer')
        else:
            self.logger = logger
            
        # Log configuration
        self.logger.info("DPO Trainer initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Beta: {beta}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Max steps: {max_steps}")
        
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for given sequences.
        
        Args:
            model: Model to use
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs
            
        Returns:
            Log probabilities for each sequence
        """
        with torch.no_grad() if model is self.reference_model else torch.enable_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        gathered_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = gathered_log_probs * mask
        
        # Sum log probs for each sequence
        sequence_log_probs = masked_log_probs.sum(dim=1)
        
        return sequence_log_probs
        
    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss.
        
        Args:
            chosen_input_ids: Input IDs for chosen responses
            chosen_attention_mask: Attention mask for chosen
            chosen_labels: Labels for chosen
            rejected_input_ids: Input IDs for rejected responses
            rejected_attention_mask: Attention mask for rejected
            rejected_labels: Labels for rejected
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        # Compute log probabilities for policy model
        policy_chosen_logps = self.compute_log_probs(
            self.policy_model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels
        )
        
        policy_rejected_logps = self.compute_log_probs(
            self.policy_model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels
        )
        
        # Compute log probabilities for reference model
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.reference_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels
            )
            
            ref_rejected_logps = self.compute_log_probs(
                self.reference_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels
            )
            
        # Compute DPO loss
        # L_DPO = -log(sigmoid(beta * (log(pi(y_w|x)) - log(pi_ref(y_w|x)) 
        #                            - log(pi(y_l|x)) + log(pi_ref(y_l|x)))))
        
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        
        # Compute preference loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = ((chosen_rewards > rejected_rewards).float().mean().item())
            chosen_reward_mean = chosen_rewards.mean().item()
            rejected_reward_mean = rejected_rewards.mean().item()
            margin = (chosen_rewards - rejected_rewards).mean().item()
            
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'chosen_reward': chosen_reward_mean,
            'rejected_reward': rejected_reward_mean,
            'reward_margin': margin
        }
        
        return loss, metrics
        
    def load_preference_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load preference data from JSONL file.
        
        Args:
            data_path: Path to preference data file
            
        Returns:
            List of preference examples
        """
        examples = []
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Preference data not found: {data_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    # Validate required fields
                    if all(k in example for k in ['prompt', 'chosen', 'rejected']):
                        examples.append(example)
                    else:
                        self.logger.warning(f"Skipping invalid example: {example}")
                        
        self.logger.info(f"Loaded {len(examples)} preference examples")
        return examples
        
    def prepare_batch(
        self,
        examples: List[Dict[str, str]]
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare batch of preference data.
        
        Args:
            examples: List of preference examples
            
        Returns:
            Tuple of tensors for training
        """
        chosen_input_ids = []
        chosen_attention_masks = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_attention_masks = []
        rejected_labels = []
        
        max_len = self.policy_model.max_seq_len
        
        for example in examples:
            prompt = example['prompt']
            chosen = example['chosen']
            rejected = example['rejected']
            
            # Format as conversations
            chosen_text = f"<|user|> {prompt} <|assistant|> {chosen}"
            rejected_text = f"<|user|> {prompt} <|assistant|> {rejected}"
            
            # Tokenize
            chosen_tokens = [self.tokenizer.bos_token_id] + \
                          self.tokenizer.encode(chosen_text) + \
                          [self.tokenizer.eos_token_id]
            rejected_tokens = [self.tokenizer.bos_token_id] + \
                            self.tokenizer.encode(rejected_text) + \
                            [self.tokenizer.eos_token_id]
                            
            # Truncate if needed
            chosen_tokens = chosen_tokens[:max_len]
            rejected_tokens = rejected_tokens[:max_len]
            
            # Pad
            chosen_padding = max_len - len(chosen_tokens)
            rejected_padding = max_len - len(rejected_tokens)
            
            chosen_ids = chosen_tokens + [self.tokenizer.pad_token_id] * chosen_padding
            rejected_ids = rejected_tokens + [self.tokenizer.pad_token_id] * rejected_padding
            
            chosen_mask = [1] * len(chosen_tokens) + [0] * chosen_padding
            rejected_mask = [1] * len(rejected_tokens) + [0] * rejected_padding
            
            # Labels are same as input_ids for DPO
            chosen_input_ids.append(chosen_ids)
            chosen_attention_masks.append(chosen_mask)
            chosen_labels.append(chosen_ids)
            
            rejected_input_ids.append(rejected_ids)
            rejected_attention_masks.append(rejected_mask)
            rejected_labels.append(rejected_ids)
            
        # Convert to tensors
        return (
            torch.tensor(chosen_input_ids, dtype=torch.long, device=self.device),
            torch.tensor(chosen_attention_masks, dtype=torch.long, device=self.device),
            torch.tensor(chosen_labels, dtype=torch.long, device=self.device),
            torch.tensor(rejected_input_ids, dtype=torch.long, device=self.device),
            torch.tensor(rejected_attention_masks, dtype=torch.long, device=self.device),
            torch.tensor(rejected_labels, dtype=torch.long, device=self.device)
        )
        
    def train(
        self,
        data_path: str,
        output_dir: str = "checkpoints/dpo"
    ):
        """Run DPO training.
        
        Args:
            data_path: Path to preference data
            output_dir: Directory to save checkpoints
        """
        # Load preference data
        preference_data = self.load_preference_data(data_path)
        
        if not preference_data:
            self.logger.error("No preference data found!")
            return
            
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting DPO training...")
        self.policy_model.train()
        
        # Training loop
        step = 0
        total_loss = 0.0
        total_accuracy = 0.0
        
        while step < self.max_steps:
            # Sample batch
            batch_size = min(self.batch_size, len(preference_data))
            batch_examples = torch.randperm(len(preference_data))[:batch_size]
            batch_data = [preference_data[i] for i in batch_examples]
            
            # Prepare batch
            batch_tensors = self.prepare_batch(batch_data)
            
            # Compute loss
            loss, metrics = self.compute_dpo_loss(*batch_tensors)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.gradient_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            step += 1
            
            # Logging
            if step % 10 == 0:
                avg_loss = total_loss / 10
                avg_accuracy = total_accuracy / 10
                
                self.logger.info(
                    f"Step {step}/{self.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {avg_accuracy:.3f} | "
                    f"Reward Margin: {metrics['reward_margin']:.3f}"
                )
                
                total_loss = 0.0
                total_accuracy = 0.0
                
            # Checkpointing
            if step % 50 == 0:
                checkpoint_path = output_dir / f"dpo_step_{step}.pt"
                self.save_checkpoint(checkpoint_path)
                
        # Final checkpoint
        final_path = output_dir / "dpo_final.pt"
        self.save_checkpoint(final_path)
        self.logger.info(f"DPO training completed! Final model saved to {final_path}")
        
    def save_checkpoint(self, path: str):
        """Save DPO checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        self.policy_model.save_checkpoint(
            path,
            optimizer=self.optimizer,
            config={'beta': self.beta}
        )
        self.logger.info(f"Saved checkpoint: {path}")