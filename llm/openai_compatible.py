import json
import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """
    Provider for any server that implements the OpenAI chat-completions and embeddings API:
    self-hosted runtimes (Ollama, vLLM) and hosted endpoints alike.

    Chat and embeddings may live on different servers (e.g. chat on vLLM, embeddings on
    Ollama), so embeddings take their own base URL and key, defaulting to the chat ones.
    With both URLs pointing at university machines, no query leaves the institution (NFR4).
    """

    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        api_key: str = "",
        embedding_base_url: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        timeout: float = 300.0,
        embedding_batch_size: int = 64,
        extra_body: Optional[Dict] = None,
    ):
        """
        Initialize the provider.

        Args:
            base_url: API root including the version, e.g. "http://localhost:11434/v1"
            chat_model: Model name for chat completions (e.g. "llama3.1:8b")
            embedding_model: Model name for embeddings (e.g. "nomic-embed-text")
            api_key: Bearer token; local runtimes usually need none
            embedding_base_url: API root for embeddings (default: base_url)
            embedding_api_key: Bearer token for embeddings (default: api_key)
            embedding_dimension: Embedding size; probed from the server on first use if not given
            timeout: Request timeout in seconds (local models can take long to load)
            embedding_batch_size: Texts per embeddings request
            extra_body: Extra fields merged into every chat request (e.g. OpenRouter's
                {"reasoning": {"enabled": False}} so a reasoning model's thinking does not use up max_tokens)
        """
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.api_key = api_key or ""
        self.embedding_base_url = (embedding_base_url or base_url).rstrip("/")
        self.embedding_api_key = self.api_key if embedding_api_key is None else embedding_api_key
        self._embedding_dimension = embedding_dimension
        self.timeout = timeout
        self.embedding_batch_size = embedding_batch_size
        self.extra_body = dict(extra_body or {})

    @staticmethod
    def _headers(api_key: str) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.embedding_base_url}/embeddings",
                headers=self._headers(self.embedding_api_key),
                json={"model": self.embedding_model, "input": texts},
            )
            response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings via POST {embedding_base_url}/embeddings.
        Empty texts get a zero vector, as with the other providers.
        """
        if not texts:
            return []
        non_empty = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        vectors: Dict[int, List[float]] = {}
        for start in range(0, len(non_empty), self.embedding_batch_size):
            batch = non_empty[start:start + self.embedding_batch_size]
            for (i, _), vector in zip(batch, self._embed_batch([t for _, t in batch])):
                vectors[i] = vector
        if vectors and self._embedding_dimension is None:
            self._embedding_dimension = len(next(iter(vectors.values())))
        dim = self.get_embedding_dimension() if len(vectors) < len(texts) else None
        return [vectors[i] if i in vectors else [0.0] * dim for i in range(len(texts))]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        """
        Generate a chat completion via POST {base_url}/chat/completions.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            stream: If True, returns a generator for streaming responses
            max_tokens: Optional cap on the number of generated tokens (None = provider default)

        Returns:
            If stream=False: Generated response text (str)
            If stream=True: Generator yielding text chunks
        """
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(self.extra_body)
        url = f"{self.base_url}/chat/completions"
        headers = self._headers(self.api_key)

        if stream:
            def chunk_generator():
                with httpx.Client(timeout=self.timeout) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                content = json.loads(data)["choices"][0]["delta"].get("content")
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                            if content:
                                yield content
            return chunk_generator()

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def get_embedding_dimension(self) -> int:
        """Embedding size: as configured, or probed once from the embeddings endpoint."""
        if self._embedding_dimension is None:
            self._embedding_dimension = len(self._embed_batch(["dimension probe"])[0])
        return self._embedding_dimension
