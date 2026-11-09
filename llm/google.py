import time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import errors, types
from typing import List, Dict, Optional
from .base import BaseLLMProvider

# Embedding size requested from the Gemini embedding models (they support reduced output sizes).
EMBEDDING_DIMENSION = 768
# Attempts per embedding request on rate limits and server errors, with exponential backoff.
EMBEDDING_ATTEMPTS = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GoogleLLMProvider(BaseLLMProvider):
    """Google AI (Gemini) LLM provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        chat_model: str = "gemini-2.5-flash",
        embedding_model: str = "gemini-embedding-2-preview"
    ):
        """
        Initialize Google AI provider.
        
        Args:
            api_key: Google AI Studio API key
            chat_model: Model name for chat completions
            embedding_model: Model name for embeddings
        """
        self.client = genai.Client(api_key=api_key)
        self.chat_model_name = chat_model
        self.embedding_model_name = embedding_model
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Google's embedding model.

        Rate limits and server errors are retried with backoff; if a request still fails, the
        error is raised, so no placeholder vector ever reaches the index.

        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [self._embed_one(texts[0])]
        with ThreadPoolExecutor(max_workers=10) as executor:
            return list(executor.map(self._embed_one, texts))

    def _embed_one(self, text: str) -> List[float]:
        dim = self.get_embedding_dimension()
        if not text or not text.strip():
            return [0.0] * dim
        for attempt in range(EMBEDDING_ATTEMPTS):
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model_name,
                    contents=text.strip(),
                    config=types.EmbedContentConfig(output_dimensionality=dim)
                )
                return result.embeddings[0].values
            except errors.APIError as e:
                if e.code not in RETRYABLE_STATUS or attempt == EMBEDDING_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a chat completion using Gemini.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            stream: If True, returns a generator for streaming responses
            max_tokens: Optional cap on the number of generated tokens (None = provider default)
            
        Returns:
            If stream=False: Generated response text (str)
            If stream=True: Generator yielding text chunks
        """
        # The system message becomes the system instruction; the turns keep their roles
        # (Gemini calls the assistant role "model").
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        
        # Generate response using new API
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            # Thinking tokens count against max_output_tokens; with a reply cap they would
            # consume it and leave an almost empty answer, so thinking is disabled when capped.
            thinking_config=types.ThinkingConfig(thinking_budget=0) if max_tokens is not None else None
        )
        
        if stream:
            # Streaming response
            response_stream = self.client.models.generate_content_stream(
                model=self.chat_model_name,
                contents=contents,
                config=config
            )
            
            def chunk_generator():
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            
            return chunk_generator()
        else:
            # Non-streaming response (original behavior)
            response = self.client.models.generate_content(
                model=self.chat_model_name,
                contents=contents,
                config=config
            )
            
            return response.text
    
    def get_embedding_dimension(self) -> int:
        """
        Embedding size: every request asks for EMBEDDING_DIMENSION (768) values, so indexes built
        with different Gemini embedding models still have the same width.
        """
        return EMBEDDING_DIMENSION
