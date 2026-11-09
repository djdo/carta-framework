from .base import BaseLLMProvider
from .google import GoogleLLMProvider
from .openrouter import OpenRouterLLMProvider
from .groq import GroqLLMProvider
from .openai_compatible import OpenAICompatibleLLMProvider
from typing import Optional

# Default API roots of the self-hosted runtimes (OpenAI-compatible endpoints)
LOCAL_RUNTIME_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}
OPENAI_COMPATIBLE_PROVIDERS = ("openai_compatible", *LOCAL_RUNTIME_URLS)


def create_llm_provider(
    provider_name: str,
    api_key: str,
    chat_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    google_api_key: Optional[str] = None,
    site_url: str = "",
    site_name: str = "Carta App",
    base_url: Optional[str] = None,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_dimension: Optional[int] = None
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider_name: Name of the provider ("google", "openrouter", "groq",
                       "ollama", "vllm", "openai_compatible")
        api_key: API key for the provider
        chat_model: Optional chat model name
        embedding_model: Optional embedding model name
        google_api_key: Google API key (required for OpenRouter and Groq embeddings)
        site_url: Optional site URL (for OpenRouter)
        site_name: Optional site name (for OpenRouter)
        base_url: API root for OpenAI-compatible providers (default per runtime for ollama/vllm)
        embedding_base_url: Separate API root for embeddings (OpenAI-compatible providers)
        embedding_api_key: Separate key for embeddings (OpenAI-compatible providers)
        embedding_dimension: Known embedding size; probed from the server if omitted
        
    Returns:
        Initialized LLM provider instance
        
    Raises:
        ValueError: If provider_name is not supported or required params are missing
    """
    if provider_name.lower() == "google":
        kwargs = {}
        if chat_model:
            kwargs["chat_model"] = chat_model
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        return GoogleLLMProvider(api_key, **kwargs)
    
    elif provider_name.lower() == "openrouter":
        if not google_api_key:
            raise ValueError(
                "google_api_key is required when using OpenRouter provider "
                "(for generating embeddings)"
            )
        kwargs = {
            "google_api_key": google_api_key,
            "site_url": site_url,
            "site_name": site_name
        }
        if chat_model:
            kwargs["chat_model"] = chat_model
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        return OpenRouterLLMProvider(api_key, **kwargs)
    
    elif provider_name.lower() == "groq":
        if not google_api_key:
            raise ValueError(
                "google_api_key is required when using Groq provider "
                "(for generating embeddings)"
            )
        kwargs = {
            "google_api_key": google_api_key,
        }
        if chat_model:
            kwargs["chat_model"] = chat_model
        if embedding_model:
            kwargs["embedding_model"] = embedding_model
        return GroqLLMProvider(api_key, **kwargs)
    
    elif provider_name.lower() in OPENAI_COMPATIBLE_PROVIDERS:
        name = provider_name.lower()
        url = base_url or LOCAL_RUNTIME_URLS.get(name)
        if not url:
            raise ValueError("base_url is required for the openai_compatible provider")
        if not chat_model or not embedding_model:
            raise ValueError(f"{name} needs both chat_model and embedding_model")
        return OpenAICompatibleLLMProvider(
            base_url=url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            api_key=api_key or "",
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_dimension=embedding_dimension,
        )
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: google, openrouter, groq, {', '.join(OPENAI_COMPATIBLE_PROVIDERS)}"
        )
