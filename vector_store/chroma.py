import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any
from ..llm.base import BaseLLMProvider
from ..models import TemporalScope
from .base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """Wrapper for ChromaDB vector store operations."""
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        persist_directory: str,
        collection_name: str,
        embedding_signature: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            llm_provider: LLM provider for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_signature: Identifies the embedding model (e.g. "openai:nomic-embed-text").
                Stored with the collection; if an existing index was built with another model,
                queries and inserts fail with a re-ingest message instead of returning
                meaningless neighbours (vectors of different models are not comparable).
        """
        self.llm_provider = llm_provider
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_signature = embedding_signature
        self.embedding_mismatch: Optional[str] = None
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=self._collection_metadata()
        )
        self._check_embedding_signature()
    
    def _collection_metadata(self) -> Dict:
        metadata = {"hnsw:space": "cosine"}
        if self.embedding_signature:
            metadata[self.SIGNATURE_KEY] = self.embedding_signature
        return metadata
    
    SIGNATURE_KEY = "carta_embedding"
    
    def _check_embedding_signature(self) -> None:
        """Compare the index's recorded embedding model with the configured one."""
        if not self.embedding_signature:
            return
        stored = (self.collection.metadata or {}).get(self.SIGNATURE_KEY)
        if stored == self.embedding_signature:
            self.embedding_mismatch = None
        elif stored is None or self.collection.count() == 0:
            # Empty index, or one built before signatures existed: adopt the current model.
            # (modify() must not repeat hnsw:space; the distance function is kept.)
            self.collection.modify(metadata={self.SIGNATURE_KEY: self.embedding_signature})
            self.embedding_mismatch = None
        else:
            self.embedding_mismatch = (
                f"The vector index was built with embeddings '{stored}', but the configured "
                f"embedding model is '{self.embedding_signature}'. Re-ingest all documents."
            )
    
    def _raise_on_mismatch(self) -> None:
        if self.embedding_mismatch:
            raise RuntimeError(self.embedding_mismatch)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        available_from_ts: Optional[float] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document
            tenant_id: Optional identifier for multitenancy
            available_from_ts: Optional timestamp enforcing calendar scope restriction
        """
        if not documents:
            return
        self._check_embedding_signature()
        self._raise_on_mismatch()
        
        # Generate embeddings
        embeddings = self.llm_provider.generate_embeddings(documents)
        
        # Generate IDs if not provided. Count-based IDs would repeat after a delete,
        # so random UUIDs are used instead.
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Inject mandatory framework-level scopes if provided, into copies of the caller's dicts
        if tenant_id is not None or available_from_ts is not None:
            metadatas = [dict(meta or {}) for meta in (metadatas or [None] * len(documents))]
            for meta in metadatas:
                if tenant_id is not None:
                    meta["_carta_tenant_id"] = tenant_id
                if available_from_ts is not None:
                    meta["_carta_available_from_ts"] = available_from_ts

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        tenant_id: Optional[str] = None,
        temporal_scope: Optional[TemporalScope] = None
    ) -> Dict:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            where: Optional dictionary of extra metadata filters
            tenant_id: Optional identifier enforcing multi-tenancy isolation
            temporal_scope: Optional TemporalScope to constrain future documents
            
        Returns:
            Dict containing documents, metadatas, and distances
        """
        self._raise_on_mismatch()
        
        # Generate query embedding
        query_embedding = self.llm_provider.generate_embeddings([query_text])[0]
        
        # Query collection
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        # Build composite where clause honoring tenancy and chronology
        conditions = []
        if where:
            for k, v in where.items():
                if isinstance(v, dict):
                    conditions.append({k: v})
                else:
                    conditions.append({k: {"$eq": v}})
        if tenant_id:
            conditions.append({"_carta_tenant_id": {"$eq": tenant_id}})
        if temporal_scope:
            conditions.append({"_carta_available_from_ts": {"$lte": temporal_scope.horizon_timestamp}})
            
        if len(conditions) == 1:
            query_kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            query_kwargs["where"] = {"$and": conditions}
            
        results = self.collection.query(**query_kwargs)
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def get_unique_sources(self, page_size: int = 5000) -> List[str]:
        """
        Get list of unique document sources in the collection.
        
        Args:
            page_size: Chunks read per request while paging through the collection
        
        Returns:
            List of unique source file names
        """
        sources = set()
        offset = 0
        while True:
            results = self.collection.get(include=["metadatas"], limit=page_size, offset=offset)
            metadatas = results.get("metadatas") or []
            for metadata in metadatas:
                if metadata and "source" in metadata:
                    # Keep just the filename from a path
                    sources.add(metadata["source"].split("/")[-1].split("\\")[-1])
            if len(metadatas) < page_size:
                return sorted(sources)
            offset += page_size
    
    def delete(self, where: Dict) -> None:
        """
        Delete documents from the collection matching the metadata filter.
        
        Args:
            where: Metadata filter dictionary
        """
        if where:
            normalized_where = self._format_where(where)
            self.collection.delete(where=normalized_where)
            
    def update_metadata(self, where: Dict, new_metadata: Dict) -> None:
        """
        Update metadata for documents matching the filter.
        
        Args:
            where: Metadata filter to find documents
            new_metadata: Metadata key-values to update
        """
        if not where:
            return
            
        normalized_where = self._format_where(where)
        results = self.collection.get(where=normalized_where)
        if results and results.get("ids"):
            updated_metadatas = []
            for meta in results["metadatas"]:
                updated = dict(meta) if meta else {}
                updated.update(new_metadata)
                updated_metadatas.append(updated)
                
            self.collection.update(
                ids=results["ids"],
                metadatas=updated_metadatas
            )
            
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata=self._collection_metadata()
        )
        self.embedding_mismatch = None

    def _format_where(self, where: Dict) -> Dict:
        """
        Normalize a flat metadata filter dictionary into a ChromaDB-compliant
        filter (using $and for multiple conditions).
        """
        if not where:
            return {}
        if len(where) == 1:
            return where
        
        # Wrap multiple filters in $and for ChromaDB compatibility
        return {"$and": [{k: v} for k, v in where.items()]}
