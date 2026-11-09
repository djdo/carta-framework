import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider
from .google import EMBEDDING_DIMENSION, GoogleLLMProvider


class OpenRouterLLMProvider(BaseLLMProvider):
    """OpenRouter LLM provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        google_api_key: str,
        chat_model: str = "google/gemini-2.5-flash",
        embedding_model: str = "gemini-embedding-2-preview",
        site_url: str = "",
        site_name: str = "Carta App",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            google_api_key: Google API key for embeddings (required since OpenRouter doesn't provide embeddings)
            chat_model: Model name for chat completions (e.g., "google/gemini-2.5-flash")
            embedding_model: Model name for embeddings (uses Google AI for embeddings)
            site_url: Optional site URL for OpenRouter
            site_name: Optional site name for OpenRouter
            base_url: API root (override for a proxy or a compatible gateway)
        """
        self.api_key = api_key
        self.google_api_key = google_api_key
        self.chat_model = chat_model
        self.embedding_model_name = embedding_model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.site_name = site_name
        
        # OpenRouter serves no embedding models: embeddings come from Google AI, through the Google
        # provider so that the model default, the 768 width and the retries are the same.
        self._google_embedding_provider = None
        
    def _init_embedding_provider(self):
        """Lazy initialize Google embedding provider."""
        if self._google_embedding_provider is None:
            if not self.google_api_key:
                raise ValueError(
                    "google_api_key is required for embeddings when using OpenRouter. "
                    "OpenRouter doesn't provide embedding models."
                )
            self._google_embedding_provider = GoogleLLMProvider(
                self.google_api_key, embedding_model=self.embedding_model_name
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Google's embedding model.
        
        Note: OpenRouter doesn't provide embedding models, so we use Google AI directly.
        
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
        Generate a chat completion using OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            stream: If True, returns a generator for streaming responses
            max_tokens: Optional cap on the number of generated tokens (None = provider default)
            
        Returns:
            If stream=False: Generated response text (str)
            If stream=True: Generator yielding text chunks
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional headers
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if stream:
            # Streaming response
            def chunk_generator():
                with httpx.Client(timeout=60.0) as client:
                    with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        response.raise_for_status()
                        
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                data = line[6:]  # Remove "data: " prefix
                                
                                if data == "[DONE]":
                                    break
                                
                                try:
                                    import json
                                    chunk = json.loads(data)
                                    content = chunk["choices"][0]["delta"].get("content", "")
                                    if content:
                                        yield content
                                except (json.JSONDecodeError, KeyError):
                                    continue
            
            return chunk_generator()
        else:
            # Non-streaming response (original behavior)
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    def get_embedding_dimension(self) -> int:
        """Embedding size of the Google embedding provider."""
        return EMBEDDING_DIMENSION
