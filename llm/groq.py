from groq import Groq
from typing import List, Dict, Optional
from .base import BaseLLMProvider
from .google import EMBEDDING_DIMENSION, GoogleLLMProvider


class GroqLLMProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        google_api_key: str,
        chat_model: str = "llama-3.3-70b-versatile",
        embedding_model: str = "gemini-embedding-2-preview"
    ):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key
            google_api_key: Google API key for embeddings (required since Groq doesn't provide embeddings)
            chat_model: Model name for chat completions (e.g., "llama-3.3-70b-versatile", "mixtral-8x7b-32768")
            embedding_model: Model name for embeddings (uses Google AI for embeddings)
        """
        self.client = Groq(api_key=api_key)
        self.google_api_key = google_api_key
        self.chat_model = chat_model
        self.embedding_model_name = embedding_model
        
        # Groq serves no embedding models: embeddings come from Google AI, through the Google
        # provider so that the model default, the 768 width and the retries are the same.
        self._google_embedding_provider = None
        
    def _init_embedding_provider(self):
        """Lazy initialize Google embedding provider."""
        if self._google_embedding_provider is None:
            if not self.google_api_key:
                raise ValueError(
                    "google_api_key is required for embeddings when using Groq. "
                    "Groq doesn't provide embedding models."
                )
            self._google_embedding_provider = GoogleLLMProvider(
                self.google_api_key, embedding_model=self.embedding_model_name
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Google's embedding model.
        
        Note: Groq doesn't provide embedding models, so we use Google AI directly.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        self._init_embedding_provider()
        return self._google_embedding_provider.generate_embeddings(texts)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a chat completion using Groq.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            stream: If True, returns a generator for streaming responses
            max_tokens: Optional cap on the number of generated tokens (None = provider default)
            
        Returns:
            If stream=False: Generated response text (str)
            If stream=True: Generator yielding text chunks
        """
        if stream:
            # Streaming response
            chat_completion = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            def chunk_generator():
                for chunk in chat_completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            
            return chunk_generator()
        else:
            # Non-streaming response
            chat_completion = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            return chat_completion.choices[0].message.content
    
    def get_embedding_dimension(self) -> int:
        """Embedding size of the Google embedding provider."""
        return EMBEDDING_DIMENSION
