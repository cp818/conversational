"""Pydantic DTOs for API requests and responses."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    
    source: str = Field(..., description="Source of the document, usually a filename")
    mime_type: str = Field(..., description="MIME type of the document")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    author: Optional[str] = Field(None, description="Document author")
    title: Optional[str] = Field(None, description="Document title")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentChunk(BaseModel):
    """A chunk of a document with its metadata."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    metadata: DocumentMetadata = Field(..., description="Metadata for the document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")


class QueryRequest(BaseModel):
    """Request for retrieving documents based on a query."""
    
    query: str = Field(..., description="Query text")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    top_k: int = Field(4, description="Number of results to return")
    namespace: Optional[str] = Field(None, description="Namespace to search in")


class QueryResponse(BaseModel):
    """Response with retrieved document chunks."""
    
    chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks with scores")
    query_id: str = Field(..., description="Unique ID for the query")
    total_latency_ms: float = Field(..., description="Total latency in milliseconds")


class ChatRequest(BaseModel):
    """Request for chat completion with RAG."""
    
    prompt: str = Field(..., description="User prompt")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    top_k: int = Field(4, description="Number of results to return")
    stream: bool = Field(True, description="Whether to stream the response")
    namespace: Optional[str] = Field(None, description="Namespace to search in")


class StreamingMessage(BaseModel):
    """Message for streaming responses."""
    
    type: str = Field(..., description="Message type: 'token', 'context', 'error', or 'done'")
    content: Union[str, Dict[str, Any]] = Field(..., description="Message content")


class IngestRequest(BaseModel):
    """Request to ingest a document."""
    
    document_id: str = Field(..., description="Unique ID for the document")
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class IngestResponse(BaseModel):
    """Response after ingesting a document."""
    
    document_id: str = Field(..., description="Document ID")
    chunks_count: int = Field(..., description="Number of chunks created")
    success: bool = Field(..., description="Success status")
    message: Optional[str] = Field(None, description="Additional message")


class HealthStatus(BaseModel):
    """Health status response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    components: Dict[str, bool] = Field(..., description="Component statuses")
