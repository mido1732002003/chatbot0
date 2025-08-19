"""Text processing utilities."""

from typing import List, Dict, Any, Optional


def format_chat_prompt(
    current_prompt: str,
    history: List[Dict[str, str]] = None,
    system_prompt: Optional[str] = None
) -> str:
    """Format chat prompt with conversation history.
    
    Args:
        current_prompt: Current user prompt
        history: Conversation history
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    formatted_parts = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_parts.append(f"System: {system_prompt}")
        
    # Add conversation history
    if history:
        for turn in history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_parts.append(f"<|user|> {content}")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|> {content}")
                
    # Add current prompt
    formatted_parts.append(f"<|user|> {current_prompt}")
    formatted_parts.append("<|assistant|>")
    
    return " ".join(formatted_parts)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
        
    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove control characters (except newline and tab)
    cleaned = []
    for char in text:
        if ord(char) >= 32 or char in '\n\t':
            cleaned.append(char)
            
    return ''.join(cleaned)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text.
    
    Args:
        text: Text to count tokens in
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


def split_into_chunks(
    text: str,
    tokenizer,
    max_chunk_size: int,
    overlap: int = 0
) -> List[str]:
    """Split text into chunks of maximum token size.
    
    Args:
        text: Text to split
        tokenizer: Tokenizer instance
        max_chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    # Tokenize full text
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position (with overlap)
        start = end - overlap if overlap > 0 else end
        
    return chunks