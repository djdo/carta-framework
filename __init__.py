"""
Carta Framework - A flexible RAG framework for building intelligent chatbots.

This framework provides reusable components for building RAG-based chatbots:
- ChatEngine: Generic RAG orchestrator
- VectorStore: Document storage and retrieval (ChromaVectorStore)
- StateManagers: Conversational and pedagogical state managers
- LLM Providers: Abstraction over different AI providers
- Configuration: Base settings for your application

Example usage:
    ```python
    from carta_framework import ChatEngine, ChromaVectorStore, create_llm_provider, BaseConfig
    
    # Initialize configuration
    config = BaseConfig()
    
    # Create LLM provider
    llm_provider = create_llm_provider(
        provider_name=config.llm_provider,
        api_key=config.google_api_key,
        chat_model=config.google_chat_model
    )
    
    # Create vector store
    vector_store = ChromaVectorStore(
        llm_provider=llm_provider,
        persist_directory=config.chroma_persist_dir,
        collection_name=config.collection_name
    )
    
    # Define your custom system prompt
    system_prompt = "You are a helpful assistant..."
    
    # Create chat engine
    chat_engine = ChatEngine(
        llm_provider=llm_provider,
        vector_store=vector_store,
        system_prompt=system_prompt,
        top_k=config.top_k_results,
        temperature=config.default_temperature
    )
    
    # Answer questions
    result = chat_engine.answer_question("What is...")
    print(result["answer"])
    ```
"""

__version__ = "0.1.0"

from .chat_engine import ChatEngine
from .vector_store import BaseVectorStore, ChromaVectorStore
VectorStore = ChromaVectorStore
from .state import BaseStateManager, MemoryStateManager
from .config import BaseConfig
from .models import ChatRequest, ChatResponse, SourceDocument, ChatMessage
from .llm import (
    BaseLLMProvider,
    GoogleLLMProvider,
    OpenRouterLLMProvider,
    GroqLLMProvider,
    OpenAICompatibleLLMProvider,
    create_llm_provider,
)
from .extensions import PDFPageChunker, RecursiveCharacterTextSplitter

__all__ = [
    # Core components
    "ChatEngine",
    "BaseVectorStore",
    "ChromaVectorStore",
    "VectorStore",
    "BaseStateManager",
    "MemoryStateManager",
    "BaseConfig",
    
    # Models
    "ChatRequest",
    "ChatResponse",
    "SourceDocument",
    "ChatMessage",
    
    # LLM providers
    "BaseLLMProvider",
    "GoogleLLMProvider",
    "OpenRouterLLMProvider",
    "GroqLLMProvider",
    "OpenAICompatibleLLMProvider",
    "create_llm_provider",
    
    # Extensions
    "PDFPageChunker",
    "RecursiveCharacterTextSplitter",
    
    # Version
    "__version__",
]

