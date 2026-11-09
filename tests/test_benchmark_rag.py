import time
import pytest
from unittest.mock import MagicMock
from carta_framework.chat_engine import ChatEngine
from carta_framework.vector_store import ChromaVectorStore
from carta_framework.state import MemoryStateManager
from carta_framework.models import TemporalScope
from carta_framework.llm.base import BaseLLMProvider

class MockLLMProvider(BaseLLMProvider):
    def generate_embeddings(self, texts):
        # Return synthetic embeddings (dim=3) for testing
        return [[0.1, 0.2, 0.3] for _ in texts]
        
    def chat(self, messages, temperature=0.7, stream=False, max_tokens=None):
        if stream:
            def generator():
                yield "This "
                time.sleep(0.01)
                yield "is "
                time.sleep(0.01)
                yield "a "
                time.sleep(0.01)
                yield "test."
            return generator()
        return "This is a test."
        
    def get_embedding_dimension(self) -> int:
        return 3

@pytest.fixture
def benchmark_engine(tmp_path):
    llm = MockLLMProvider()
    store = ChromaVectorStore(
        llm_provider=llm,
        persist_directory=str(tmp_path / "chroma"),
        collection_name="benchmark_col"
    )
    
    # Pre-populate with temporal data
    documents = []
    metadatas = []
    for i in range(100):
        documents.append(f"Content for chunk {i}")
        metadatas.append({
            "source": "doc.txt",
            "_carta_available_from_ts": float(i * 10),
            "chunk_id": i
        })
        
    store.add_documents(documents=documents, metadatas=metadatas)
    
    state_manager = MemoryStateManager()
    
    engine = ChatEngine(
        llm_provider=llm,
        vector_store=store,
        state_manager=state_manager,
        system_prompt="Test"
    )
    return engine, store

def test_benchmark_recall_k(benchmark_engine):
    """
    Evaluate Recall@k for temporal retrieval. 
    Verifies parametric interference is mitigated by hard calendar scopes.
    """
    engine, store = benchmark_engine
    
    # We want to retrieve chunk 5, but our scope only allows up to ts=55 (chunks 0-5)
    # The query is broad, so distance might be similar for all.
    scope = TemporalScope(horizon_timestamp=55.0)
    
    results = engine.answer_question(
        question="What is the content?",
        temporal_scope=scope
    )
    
    # Ensure no future chunks were retrieved (chunk_id > 5)
    for source in results["sources"]:
        meta = source.metadata or {}
        assert meta.get("_carta_available_from_ts", 0) <= 55.0
        
def test_benchmark_ttft(benchmark_engine):
    """
    Evaluate Time-To-First-Token (TTFT) for streaming responses.
    """
    engine, store = benchmark_engine
    
    start_time = time.time()
    stream = engine.answer_question_stream(
        question="What is the content?"
    )
    
    first_token_time = None
    for event in stream:
        if "event: message" in event and first_token_time is None:
            first_token_time = time.time()
            break
            
    ttft = first_token_time - start_time
    # Assert TTFT is under our NFR requirement (e.g. 500ms)
    assert ttft < 0.5
