from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of floats)
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            stream: If True, returns a generator yielding text chunks
            max_tokens: Optional cap on the number of generated tokens (None = provider default)
            
        Returns:
            If stream=False: Generated response text (str)
            If stream=True: Generator yielding text chunks
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.
        
        Returns:
            Embedding dimension size
        """
        pass
