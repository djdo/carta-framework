from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class BaseConfig(BaseSettings):
    """
    Base configuration for Carta Framework.
    
    Consumer applications should extend this class to add their own settings.
    Only framework-level settings are included here.
    """
    
    # LLM Provider Configuration (the names create_llm_provider accepts)
    llm_provider: Literal["google", "openrouter", "groq", "ollama", "vllm", "openai_compatible"] = "google"
    google_api_key: str = ""
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    
    # Google AI Models (also the embeddings of OpenRouter and Groq)
    google_chat_model: str = "gemini-2.5-flash"
    google_embedding_model: str = "gemini-embedding-2-preview"
    
    # OpenRouter Models
    openrouter_chat_model: str = "google/gemini-2.5-flash"
    openrouter_site_url: str = ""
    openrouter_site_name: str = "Carta App"
    
    # Groq Models
    groq_chat_model: str = "llama-3.3-70b-versatile"
    
    # OpenAI-compatible servers (ollama, vllm, openai_compatible). An empty base URL means the
    # runtime's local default for ollama and vllm; openai_compatible needs one.
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_chat_model: str = ""
    openai_compatible_embedding_model: str = ""
    openai_compatible_embedding_base_url: str = ""
    
    # Vector Database Configuration
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "documents"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    # Generation Configuration
    default_temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
