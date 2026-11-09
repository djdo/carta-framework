# Carta Framework

A flexible, production-ready RAG (Retrieval-Augmented Generation) framework for building intelligent chatbots.

## Overview

Carta Framework provides reusable components for building RAG-based chatbots with custom personalities and behaviors. It handles the complex parts (vector storage, LLM integration, retrieval) so you can focus on your application's unique requirements.

## Features

- 🎯 **Generic RAG Engine**: Plug in any system prompt and personality
- 🔌 **Multiple LLM Providers**: any OpenAI-compatible server (Ollama, vLLM, hosted APIs), Google AI, OpenRouter, Groq; easily extensible
- 🗓️ **Scoped Retrieval**: per-tenant (course) isolation and release-date filtering inside the vector search
- 🌊 **Streaming**: Server-Sent Events with citations sent before the first token
- 📚 **Vector Storage**: ChromaDB integration with persistent storage
- ⚙️ **Configurable**: Temperature, top-k, prompt builders, and more
- 🏗️ **Clean Architecture**: Separation of concerns, dependency injection
- 📦 **Pip Installable**: Easy to integrate into your projects

## Installation

```bash
pip install git+https://github.com/djdo/carta-framework.git
```

Or, from a clone, `pip install -e .`. Requires Python 3.10 or later.

## Quick Start

```python
from carta_framework import (
    ChatEngine,
    VectorStore,
    create_llm_provider,
    BaseConfig
)

# 1. Configure
config = BaseConfig()

# 2. Create LLM provider
llm_provider = create_llm_provider(
    provider_name="google",
    api_key=config.google_api_key
)

# 3. Create vector store
vector_store = VectorStore(
    llm_provider=llm_provider,
    persist_directory="./chroma_db",
    collection_name="my-docs"
)

# 4. Add documents
vector_store.add_documents(
    documents=["Your document text here..."],
    metadatas=[{"source": "doc1.pdf"}]
)

# 5. Create chat engine with your custom personality
system_prompt = """You are a helpful assistant that answers
questions based only on the provided context."""

chat_engine = ChatEngine(
    llm_provider=llm_provider,
    vector_store=vector_store,
    system_prompt=system_prompt,
    temperature=0.1,
    top_k=5
)

# 6. Ask questions
result = chat_engine.answer_question("What is in the documents?")
print(result["answer"])
print(result["sources"])
```

## Core Components

### ChatEngine

Generic RAG orchestrator that:
- Retrieves relevant documents
- Builds context from retrieved docs
- Generates answers using LLM
- Returns answers with source citations

**Key Parameters:**
- `system_prompt`: Defines AI personality and behavior
- `temperature`: Controls response creativity (0.0-1.0)
- `top_k`: Number of relevant docs to retrieve
- `prompt_builder`: Optional custom function to build user prompts
- `relevance_threshold`: Minimum similarity score s(d) = 1 − d/2, where d is Chroma's cosine distance (default 0.3, see below)
- `no_answer_response`: Reply returned without calling the model when no chunk passes the threshold; `None` sends the question with an empty context instead, for assistants that may answer from general knowledge
- `state_manager`: Optional session store; calls that pass a `session_id` read their history from it and record each turn
- `max_reply_tokens`, `context_window`: Reply cap, and the window the oldest history messages are trimmed to fit

**Relevance threshold.** On the s = 1 − d/2 scale the default 0.3 corresponds to a cosine similarity of −0.4, so it only drops degenerate matches; unrelated text usually scores around 0.5. To filter off-topic chunks, calibrate a cutoff on your own corpus.

### VectorStore

Manages document storage and retrieval:
- ChromaDB backend with persistence (cosine distance)
- Automatic embedding generation
- Tenant and release-date filters applied inside the search (`tenant_id`, `available_from_ts`, `TemporalScope`)
- Records the embedding model with the index and refuses to mix models (`embedding_signature`)
- Multiple collection support

### LLM Providers

Abstraction over different AI providers:
- **OpenAICompatibleLLMProvider**: any server with the OpenAI chat-completions and embeddings API, such as Ollama, vLLM or a hosted endpoint. Chat and embeddings can run on different servers.
- **GoogleLLMProvider**: Google AI (Gemini)
- **OpenRouterLLMProvider**: OpenRouter for alternative models
- **GroqLLMProvider**: Groq for fast inference (Llama, Mixtral, etc.)
- **BaseLLMProvider**: Interface for custom providers

Use the factory: `create_llm_provider(provider_name, api_key, ...)` with `"google"`, `"openrouter"`, `"groq"`, `"ollama"`, `"vllm"` or `"openai_compatible"`.

OpenRouter and Groq serve no embedding models, so they take a `google_api_key` and embed through Google AI, like the Google provider: the same default model, 768 dimensions. An index built with one of these three providers can therefore be queried through the others. Indexes built with an OpenAI-compatible embedding model cannot be mixed with them.

### BaseConfig

Pydantic settings with environment variable support:
- LLM provider configuration
- Vector database settings
- RAG parameters (chunk size, overlap, top-k)
- Extend for your app-specific settings

## Configuration

### Environment Variables

```env
# LLM Provider
LLM_PROVIDER=google  # or openrouter, groq, ollama, vllm, openai_compatible
GOOGLE_API_KEY=your-key
GROQ_API_KEY=your-groq-key  # if using groq

# OpenAI-compatible servers (ollama, vllm, openai_compatible)
OPENAI_COMPATIBLE_BASE_URL=http://localhost:11434/v1  # empty: runtime default for ollama/vllm
OPENAI_COMPATIBLE_CHAT_MODEL=gemma3:12b
OPENAI_COMPATIBLE_EMBEDDING_MODEL=bge-m3

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=documents

# RAG Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
DEFAULT_TEMPERATURE=0.1
```

### Extending Configuration

```python
from carta_framework import BaseConfig

class MyAppConfig(BaseConfig):
    # Add your app-specific settings
    app_name: str = "My Chatbot"
    max_users: int = 100
    feature_flag: bool = True
```

## Custom Prompt Building

```python
def my_prompt_builder(question: str, context: str) -> str:
    return f"""Context:\n{context}\n\nQ: {question}\nA:"""

chat_engine = ChatEngine(
    llm_provider=llm_provider,
    vector_store=vector_store,
    system_prompt="You are helpful...",
    prompt_builder=my_prompt_builder
)
```

## Scoped Retrieval and Sessions

```python
from datetime import datetime, timezone
from carta_framework import MemoryStateManager
from carta_framework.models import TemporalScope

# Each chunk carries its tenant (e.g. a course) and release timestamp
released = datetime(2026, 10, 19, tzinfo=timezone.utc).timestamp()
vector_store.add_documents(
    ["GPIOA_CFGLR selects the mode of pins PA0 to PA7 ..."],
    metadatas=[{"source": "A06.pdf", "page_number": 12}],
    tenant_id="microcontrollers", available_from_ts=released
)

state = MemoryStateManager()
chat_engine = ChatEngine(llm_provider, vector_store, system_prompt, state_manager=state)

# Only chunks of this tenant released by the horizon are searched
horizon = TemporalScope(horizon_timestamp=datetime.now(timezone.utc).timestamp())
result = chat_engine.answer_question(
    "Why does PA0 stay low?", session_id="session-1",
    tenant_id="microcontrollers", temporal_scope=horizon
)
```

## Advanced Usage

### Multiple Collections

```python
# Different collections for different domains
ben_store = VectorStore(llm_provider, "./db", "ben-docs")
study_store = VectorStore(llm_provider, "./db", "study-materials")
```

### Custom LLM Providers

```python
from carta_framework.llm import BaseLLMProvider

class MyLLMProvider(BaseLLMProvider):
    def generate_embeddings(self, texts):
        # Your implementation
        pass
    
    def chat(self, messages, temperature=0.7, stream=False, max_tokens=None):
        # Return the reply, or a generator of text chunks if stream
        pass
    
    def get_embedding_dimension(self):
        return 768
```

### Streaming Responses

`answer_question_stream()` takes the same arguments as `answer_question()` and yields Server-Sent Events: one `sources` event with the citations, one `message` event per token (JSON-encoded), and `end` with `[DONE]`.

```python
for event in chat_engine.answer_question_stream("What is in the documents?"):
    print(event, end="")
```

## API Reference

### ChatEngine

```python
class ChatEngine:
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        vector_store: BaseVectorStore,
        system_prompt: str,
        state_manager: Optional[BaseStateManager] = None,
        top_k: int = 5,
        temperature: float = 0.1,
        prompt_builder: Optional[Callable[[str, str], str]] = None,
        no_answer_response: Optional[str] = "I don't have that information available.",
        relevance_threshold: float = 0.3,
        max_reply_tokens: Optional[int] = None,
        context_window: Optional[int] = None
    )
    
    def answer_question(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        filter_metadata: Optional[Dict] = None,
        tenant_id: Optional[str] = None,
        temporal_scope: Optional[TemporalScope] = None,
        system_prompt_override: Optional[str] = None,
        workflow: Optional[AgenticWorkflow] = None,
        usage: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """Returns: {"answer": str, "sources": List[SourceDocument]}"""
    
    def answer_question_stream(...) -> Generator[str, None, None]:
        """Same arguments; yields SSE events (sources, message..., end)"""
```

### VectorStore

```python
class VectorStore:
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        persist_directory: str,
        collection_name: str,
        embedding_signature: Optional[str] = None
    )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        available_from_ts: Optional[float] = None
    )
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        tenant_id: Optional[str] = None,
        temporal_scope: Optional[TemporalScope] = None
    ) -> Dict
    
    def delete(self, where: Dict)
    
    def update_metadata(self, where: Dict, new_metadata: Dict)
    
    def count(self) -> int
    
    def get_unique_sources(self) -> List[str]
    
    def clear(self)
```

### Factory Function

```python
def create_llm_provider(
    provider_name: str,  # "google", "openrouter", "groq", "ollama", "vllm", "openai_compatible"
    api_key: str,
    chat_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    google_api_key: Optional[str] = None,  # For OpenRouter/Groq embeddings
    site_url: str = "",
    site_name: str = "Carta App",
    base_url: Optional[str] = None,  # OpenAI-compatible providers
    embedding_base_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_dimension: Optional[int] = None
) -> BaseLLMProvider
```

## Development

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code (the repository root is the package)
black .
ruff check .
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
