import re
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class BaseMetadata(BaseModel):
    """Standard metadata properties that are universally supported."""
    carta_tenant_id: Optional[str] = Field(default=None, alias="_carta_tenant_id")
    carta_available_from_ts: Optional[float] = Field(default=None, alias="_carta_available_from_ts")
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    document_type: Optional[str] = None
    doc_type: Optional[str] = None
    chapter: Optional[str] = None
    topic: Optional[str] = None
    source: Optional[str] = None


class TemporalScope(BaseModel):
    """
    Temporal constraint for RAG filtering.
    Used to enforce 'calendar scope' filtering, where documents are only visible
    if their availability timestamp is less than or equal to the query horizon.
    """
    horizon_timestamp: float = Field(..., description="The maximum timestamp allowed for retrieved documents.")


class ChatMessage(BaseModel):
    """Single message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint (thesis Listing 3.3)."""
    message: str = Field(..., description="The student's message", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Conversation session; history is kept server-side")
    workflow: Optional[str] = Field(default=None, description="AgenticWorkflow id to apply")
    

class SourceDocument(BaseModel):
    """Model for source document information."""
    content: str = Field(..., description="Document content snippet")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata properties")
    filename: Optional[str] = Field(default=None, description="Source file of the chunk")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score (0-1)")
    page_number: Optional[int] = Field(default=None, description="Page number of the source document")
    total_pages: Optional[int] = Field(default=None, description="Total pages of the source document")
    document_type: Optional[str] = Field(default=None, description="Document classification (e.g. Slides, Lab, Datasheet)")
    chapter: Optional[str] = Field(default=None, description="Designated topic/chapter for this content")



class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")


class AgenticWorkflow(BaseModel):
    """Generic configuration for a Prompt Pipeline overriding the default generation instructions."""
    id: str = Field(..., description="Unique identifier for the generic workflow")
    name: str = Field(..., description="Human-readable name for the workflow (e.g. 'Flashcards')")
    description: Optional[str] = Field(None, description="Short definition of the workflow behavior")
    system_prompt_extension: str = Field(..., description="Direct prompt instructions appended to the LLM context to override generation style")
    reminder: Optional[str] = Field(None, description="Short restatement placed after the student's question; defaults to the first two sentences of the extension")

    def reminder_text(self) -> str:
        """The line restated after the question: models follow the instruction nearest to it (follow-up #25)."""
        if self.reminder:
            return self.reminder
        sentences = re.split(r"(?<=[.!?])\s+", self.system_prompt_extension.strip())
        return " ".join(sentences[:2])
