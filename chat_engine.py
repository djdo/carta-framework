import json
import re
from typing import List, Dict, Optional, Callable, Generator, Tuple
from .llm.base import BaseLLMProvider
from .vector_store.base import BaseVectorStore
from .state.base import BaseStateManager
from .models import SourceDocument, TemporalScope, AgenticWorkflow


# Sliding history window: three student queries and three tutor replies (M = 6).
HISTORY_WINDOW = 6

# Closing delimiter of a context block.
CONTEXT_BLOCK_END = "-" * 62


class ChatEngine:
    """
    Generic RAG-based chat engine for answering questions based on documentation.

    This is a framework component - consumer applications should provide their own
    system prompts and configuration.
    """

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
    ):
        """
        Initialize chat engine.

        Args:
            llm_provider: LLM provider for generating responses
            vector_store: Vector store for retrieving relevant documents
            system_prompt: System prompt that defines the AI's personality and behavior
            top_k: Number of relevant documents to retrieve
            temperature: Temperature for text generation (0.0 to 1.0)
            prompt_builder: Optional custom function to build user prompt from question and context.
                          Signature: (question: str, context: str) -> str
                          If None, uses default prompt builder.
            state_manager: Optional session store. When given, calls that pass a session_id read
                          their history from it and record the question and the answer in it.
            no_answer_response: Reply returned without calling the model when no chunk passes the
                          relevance threshold. None sends the query with an empty context instead,
                          for assistants whose prompt lets the model fall back on general knowledge.
            relevance_threshold: Minimum similarity score s(d) = 1 - d/2 (0.0-1.0) for including
                          documents. The default 0.3 (cosine >= -0.4) only guards against degenerate
                          matches; a cutoff that separates relevant from irrelevant chunks has to be
                          calibrated on the corpus.
            max_reply_tokens: Optional cap on generated tokens per reply (None = provider default)
            context_window: Optional model context window in tokens. When set, the oldest
                          history messages are dropped until the request plus the reply cap fits.
        """
        self.llm_provider = llm_provider
        self.vector_store = vector_store
        self.state_manager = state_manager
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.temperature = temperature
        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.no_answer_response = no_answer_response
        self.relevance_threshold = relevance_threshold
        self.max_reply_tokens = max_reply_tokens
        self.context_window = context_window

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
        """
        Answer a question using RAG with relevance filtering.

        Args:
            question: User's question
            history: Optional conversation history as list of {"role": str, "content": str} dicts
            filter_metadata: Optional metadata filter dict applied during retrieval
            tenant_id: Optional framework-level isolated tenancy identifier
            temporal_scope: Optional framework-level time filter for calendar scopes
            system_prompt_override: Optional framework-level personality substitution
            workflow: Optional framework-level prompt extension pipeline
            usage: Optional dict that receives estimated 'prompt_tokens' and 'completion_tokens'
            session_id: Optional session in the state manager; supplies the history when none is
                        passed, and receives the question and the answer

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        # 1. Scoped retrieval and relevance thresholding
        search_results = self.vector_store.query(
            question,
            n_results=self.top_k,
            where=filter_metadata,
            tenant_id=tenant_id,
            temporal_scope=temporal_scope
        )
        documents, metadatas, distances = self._filter_by_threshold(search_results)

        if not documents and self.no_answer_response is not None:
            answer = self.no_answer_response
            self._record_usage(usage, [], answer)
        else:
            # 2. Assemble prompt: system + workflow, history, then context + query
            messages = self._build_messages(
                question, documents, metadatas, self._session_history(history, session_id),
                system_prompt_override, workflow
            )

            # 3. Generate answer
            answer = self.llm_provider.chat(messages, **self._generation_kwargs())
            self._record_usage(usage, messages, answer)
        self._save_turn(session_id, question, answer)

        return {
            "answer": answer,
            "sources": self._build_source_documents(documents, metadatas, distances)
        }

    def answer_question_stream(
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
    ) -> Generator[str, None, None]:
        """
        Answer a question using RAG with streaming response via SSE-formatted events.

        Yields SSE events:
          - event: sources\ndata: {json sources}\n\n
          - event: message\ndata: {json token}\n\n
          - event: end\ndata: [DONE]\n\n

        Args:
            question: User's question
            history: Optional conversation history
            filter_metadata: Optional metadata filter for retrieval
            tenant_id: Optional framework-level isolated tenancy identifier
            temporal_scope: Optional framework-level time filter for calendar scopes
            system_prompt_override: Optional framework-level personality substitution
            workflow: Optional framework-level prompt extension pipeline
            usage: Optional dict that receives estimated 'prompt_tokens' and 'completion_tokens'
                   once the stream has finished
            session_id: Optional session in the state manager; supplies the history when none is
                        passed, and receives the question and the answer once the stream has finished
        """
        # 1. Scoped retrieval and relevance thresholding
        search_results = self.vector_store.query(
            question,
            n_results=self.top_k,
            where=filter_metadata,
            tenant_id=tenant_id,
            temporal_scope=temporal_scope
        )
        documents, metadatas, distances = self._filter_by_threshold(search_results)

        # 2. Build and emit SourceDocument citations first
        sources = self._build_source_documents(documents, metadatas, distances)
        sources_data = [s.model_dump() for s in sources]
        yield f"event: sources\ndata: {json.dumps(sources_data)}\n\n"

        if not documents and self.no_answer_response is not None:
            answer = self.no_answer_response
            yield f"event: message\ndata: {json.dumps(answer)}\n\n"
            self._record_usage(usage, [], answer)
        else:
            # 3. Assemble prompt: system + workflow, history, then context + query
            messages = self._build_messages(
                question, documents, metadatas, self._session_history(history, session_id),
                system_prompt_override, workflow
            )

            # 4. Stream tokens; JSON-encode to keep newlines and spaces intact in SSE
            answer_parts = []
            for token in self.llm_provider.chat(messages, stream=True, **self._generation_kwargs()):
                answer_parts.append(token)
                yield f"event: message\ndata: {json.dumps(token)}\n\n"
            answer = "".join(answer_parts)
            self._record_usage(usage, messages, answer)
        self._save_turn(session_id, question, answer)

        # 5. Explicit stream termination marker
        yield "event: end\ndata: [DONE]\n\n"

    def _filter_by_threshold(self, search_results: Dict) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Drop retrieved chunks whose similarity score s(d) = max(0, 1 - d/2) is below
        the relevance threshold. Cosine distance d ranges from 0 (identical) to 2 (opposite).
        """
        documents = search_results.get("documents") or []
        metadatas = search_results.get("metadatas") or [None] * len(documents)
        distances = search_results.get("distances") or []

        kept_docs, kept_metadatas, kept_distances = [], [], []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            if self._similarity(distance) >= self.relevance_threshold:
                kept_docs.append(doc)
                kept_metadatas.append(metadata)
                kept_distances.append(distance)
        return kept_docs, kept_metadatas, kept_distances

    def _build_source_documents(
        self, documents: List[str], metadatas: List[Optional[Dict]], distances: List[float]
    ) -> List[SourceDocument]:
        """Create one SourceDocument citation per retained chunk."""
        sources = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            meta = metadata or {}
            sources.append(
                SourceDocument(
                    content=doc[:200] + "..." if len(doc) > 200 else doc,
                    metadata=metadata,
                    filename=meta.get("source"),
                    similarity_score=round(self._similarity(distance), 3),
                    page_number=meta.get("page_number"),
                    total_pages=meta.get("total_pages"),
                    document_type=meta.get("document_type") or meta.get("doc_type"),
                    chapter=meta.get("chapter") or meta.get("topic")
                )
            )
        return sources

    def _build_messages(
        self,
        question: str,
        documents: List[str],
        metadatas: List[Optional[Dict]],
        history: Optional[List[Dict]],
        system_prompt_override: Optional[str],
        workflow: Optional[AgenticWorkflow]
    ) -> List[Dict[str, str]]:
        """Build the message array: system prompt (+ workflow), recent history, context + query (+ workflow reminder)."""
        context = self._build_context(documents, metadatas)
        user_prompt = self.prompt_builder(question, context)
        if workflow is not None:
            user_prompt = f"{user_prompt.rstrip()}\n\n[Reminder] {workflow.reminder_text()}"

        active_system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        if workflow is not None:
            active_system_prompt = f"{active_system_prompt}\n\n=== EXPLICIT INSTRUCTION OVERRIDE ===\n{workflow.system_prompt_extension}"

        system_message = {"role": "system", "content": active_system_prompt}
        user_message = {"role": "user", "content": user_prompt}
        recent_history = self._fit_history(system_message, list(history or [])[-HISTORY_WINDOW:], user_message)

        return [system_message, *recent_history, user_message]

    def _fit_history(self, system_message: Dict, history: List[Dict], user_message: Dict) -> List[Dict]:
        """
        Drop the oldest history messages until system prompt, history, current turn and
        the reply cap fit in the context window. No-op when no context window is configured.
        """
        if self.context_window is None:
            return history
        budget = self.context_window - (self.max_reply_tokens or 0)
        fixed = self._estimate_tokens(system_message["content"]) + self._estimate_tokens(user_message["content"])
        while history and fixed + sum(self._estimate_tokens(m["content"]) for m in history) > budget:
            history = history[1:]
        return history

    def _session_history(self, history: Optional[List[Dict]], session_id: Optional[str]) -> Optional[List[Dict]]:
        """An explicit history wins; otherwise the session's recent messages, if a store is configured."""
        if history is not None or self.state_manager is None or session_id is None:
            return history
        return self.state_manager.get_history(session_id, limit=HISTORY_WINDOW)

    def _save_turn(self, session_id: Optional[str], question: str, answer: str) -> None:
        if self.state_manager is None or session_id is None:
            return
        self.state_manager.add_message(session_id, "user", question)
        self.state_manager.add_message(session_id, "assistant", answer)

    def _record_usage(self, usage: Optional[Dict], messages: List[Dict[str, str]], answer: str) -> None:
        if usage is None:
            return
        usage["prompt_tokens"] = sum(self._estimate_tokens(m["content"]) for m in messages)
        usage["completion_tokens"] = self._estimate_tokens(answer)

    def _generation_kwargs(self) -> Dict:
        kwargs = {"temperature": self.temperature}
        if self.max_reply_tokens is not None:
            kwargs["max_tokens"] = self.max_reply_tokens
        return kwargs

    @staticmethod
    def _similarity(distance: float) -> float:
        return max(0.0, 1.0 - (distance / 2.0))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough heuristic: about four characters per token.
        return len(text or "") // 4 + 1

    def _build_context(self, documents: List[str], metadatas: Optional[List[Optional[Dict]]] = None) -> str:
        """
        Build the context block. Each chunk is wrapped in delimiters with a provenance header,
        so the model can tell course text apart from instructions:

            --- Context Document [1] (Source: Lecture10_Timers.pdf, Page 12) ---
            <chunk text>
            --------------------------------------------------------------

        Args:
            documents: List of document texts
            metadatas: Optional metadata dicts aligned with documents (source, page_number)

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        metadatas = metadatas or [None] * len(documents)
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(documents[:self.top_k], metadatas), 1):
            meta = metadata or {}
            # The ingestion pipeline prefixes chunks with a "Source..." line; the header replaces it.
            doc_clean = re.sub(r"^\s*Source( File)?( Path)?:[^\n]*\n*", "", doc, flags=re.IGNORECASE).strip()

            provenance = [f"Source: {meta.get('source') or 'unknown'}"]
            if meta.get("page_number") is not None:
                provenance.append(f"Page {meta['page_number']}")
            header = f"--- Context Document [{i}] ({', '.join(provenance)}) ---"
            context_parts.append(f"{header}\n{doc_clean}\n{CONTEXT_BLOCK_END}")

        return "\n\n".join(context_parts)

    def _default_prompt_builder(self, question: str, context: str) -> str:
        """
        Default prompt builder that combines question and context.

        Args:
            question: User's question
            context: Context from retrieved documents

        Returns:
            Formatted prompt
        """
        return f"""Context:
{context}

Question: {question}

Use the context above when responding."""
