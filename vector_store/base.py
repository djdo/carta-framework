from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from ..llm.base import BaseLLMProvider
from ..models import TemporalScope

class BaseVectorStore(ABC):
    """Abstract base class for vector store operations."""
    
    @abstractmethod
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
        pass
        
    @abstractmethod
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
        pass

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the collection."""
        pass
        
    @abstractmethod
    def get_unique_sources(self) -> List[str]:
        """Get list of unique document sources in the collection."""
        pass
        
    @abstractmethod
    def delete(self, where: Dict) -> None:
        """Delete documents from the collection matching the metadata filter."""
        pass
        
    @abstractmethod
    def update_metadata(self, where: Dict, new_metadata: Dict) -> None:
        """Update metadata for documents matching the filter."""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the collection."""
        pass
