"""Command-line chat interface."""

import torch
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

from core.model import TransformerLM
from core.generation import generate
from alignment.safety_filter import SafetyFilter
from utils.text_utils import format_chat_prompt
from utils.logging_utils import setup_logger


class ChatInterface:
    """Interactive command-line chat interface.
    
    Provides multi-turn conversation with safety filtering
    and customizable generation parameters.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer,
        safety_filter: Optional[SafetyFilter] = None,
        max_history: int = 10,
        generation_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """Initialize chat interface.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            safety_filter: Safety filter instance
            max_history: Maximum conversation history to maintain
            generation_config: Default generation parameters
            device: Device to run on
            logger: Logger instance
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load model
        self.model = TransformerLM.from_checkpoint(model_path, device=self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        
        # Setup safety filter
        if safety_filter is None:
            self.safety_filter = SafetyFilter(enable_filter=True)
        else:
            self.safety_filter = safety_filter
            
        # Conversation history
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        
        # Generation config
        self.generation_config = generation_config or {
            'max_tokens': 150,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        # Setup logger
        if logger is None:
            self.logger = setup_logger('chat_interface')
        else:
            self.logger = logger
            
        self.logger.info(f"Chat interface initialized on {self.device}")
        
    def generate_response(self, prompt: str) -> str:
        """Generate response for user prompt.
        
        Args:
            prompt: User input prompt
            
        Returns:
            Generated response
        """
        # Check prompt safety
        prompt_check = self.safety_filter.check_prompt_safety(prompt)
        if not prompt_check['is_safe']:
            return prompt_check['suggested_response']
            
        # Format conversation with history
        formatted_prompt = format_chat_prompt(prompt, self.history)
        
        # Tokenize
        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id] + self.tokenizer.encode(formatted_prompt),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Ensure we don't exceed max sequence length
        if input_ids.shape[1] > self.model.max_seq_len - self.generation_config['max_tokens']:
            # Truncate history if needed
            self.history = self.history[-(self.max_history // 2):]
            formatted_prompt = format_chat_prompt(prompt, self.history)
            input_ids = torch.tensor(
                [self.tokenizer.bos_token_id] + self.tokenizer.encode(formatted_prompt),
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
        # Generate response
        with torch.no_grad():
            output = generate(
                self.model,
                input_ids,
                max_tokens=self.generation_config['max_tokens'],
                temperature=self.generation_config['temperature'],
                top_k=self.generation_config['top_k'],
                top_p=self.generation_config['top_p'],
                repetition_penalty=self.generation_config['repetition_penalty'],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode response
        generated_ids = output['generated_ids'][0]
        # Remove prompt tokens
        response_ids = generated_ids[input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids.tolist())
        
        # Clean up response (remove special tokens)
        response = response.replace('<|assistant|>', '').strip()
        response = response.replace('<|user|>', '').strip()
        if self.tokenizer.eos_token:
            response = response.replace(self.tokenizer.eos_token, '').strip()
            
        # Check response safety
        response_check = self.safety_filter.check_response_safety(response)
        if not response_check['is_safe']:
            response = response_check.get('filtered_response', "[Response filtered for safety]")
            
        return response
        
    def chat_loop(self):
        """Run interactive chat loop."""
        print("\n" + "="*50)
        print("Chat Interface - Type 'quit' to exit")
        print("Commands: /clear (clear history), /config (show config)")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Assistant: Goodbye!")
                    break
                elif user_input == '/clear':
                    self.history.clear()
                    print("Assistant: Conversation history cleared.")
                    continue
                elif user_input == '/config':
                    print(f"Assistant: Current generation config:")
                    for key, value in self.generation_config.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.startswith('/set '):
                    # Update generation config
                    parts = user_input[5:].split('=')
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        if key in self.generation_config:
                            try:
                                if key in ['max_tokens', 'top_k']:
                                    self.generation_config[key] = int(value)
                                else:
                                    self.generation_config[key] = float(value)
                                print(f"Assistant: Updated {key} to {value}")
                            except ValueError:
                                print(f"Assistant: Invalid value for {key}")
                        else:
                            print(f"Assistant: Unknown parameter {key}")
                    continue
                elif not user_input:
                    continue
                    
                # Generate response
                response = self.generate_response(user_input)
                
                # Print response
                print(f"Assistant: {response}")
                
                # Update history
                self.history.append({'role': 'user', 'content': user_input})
                self.history.append({'role': 'assistant', 'content': response})
                
                # Trim history if needed
                if len(self.history) > self.max_history * 2:
                    self.history = self.history[-self.max_history * 2:]
                    
            except KeyboardInterrupt:
                print("\nAssistant: Chat interrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"Assistant: An error occurred: {str(e)}")
                self.logger.error(f"Chat error: {str(e)}", exc_info=True)
                
    def reset(self):
        """Reset conversation history."""
        self.history.clear()
        self.logger.info("Conversation history reset")