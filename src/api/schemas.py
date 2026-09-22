"""Pydantic V2 Schemas for Enterprise Cognitive Hybrid RAG Platform."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata schema attached to documents and chunks."""

    source: str = Field(..., description="Document source name or URI")
    doc_id: str = Field(..., description="Unique document ID")
    chunk_id: str = Field(..., description="Unique chunk ID (e.g. doc_id#chunk_index)")
    chunk_index: int = Field(..., ge=0, description="0-indexed position in document")
    title: Optional[str] = Field(default=None, description="Document title")
    page: Optional[int] = Field(default=None, description="Original page number if PDF")
    sha256: Optional[str] = Field(default=None, description="SHA-256 hash of chunk content")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional enterprise metadata")


class DocumentChunk(BaseModel):
    """Single textual chunk of an enterprise document with metadata."""

    content: str = Field(..., min_length=1, description="Text chunk content")
    metadata: DocumentMetadata = Field(..., description="Chunk metadata")


class IngestDocument(BaseModel):
    """Raw document payload for ingestion."""

    doc_id: Optional[str] = Field(default=None, description="Optional custom document ID")
    source: str = Field(..., description="Filename, URL, or identifier")
    title: Optional[str] = Field(default=None, description="Document title")
    content: str = Field(..., min_length=1, description="Raw document text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")


class IngestRequest(BaseModel):
    """Batch ingestion request."""

    documents: List[IngestDocument] = Field(..., min_length=1, description="List of documents to ingest")
    chunk_size: Optional[int] = Field(default=512, ge=64, le=4096, description="Target chunk size in tokens")
    chunk_overlap: Optional[int] = Field(default=100, ge=0, le=512, description="Chunk overlap in tokens")


class IngestResponse(BaseModel):
    """Ingestion result summary."""

    status: str = Field(default="success", description="Status code")
    documents_processed: int = Field(..., description="Number of source documents processed")
    total_chunks_created: int = Field(..., description="Total chunks parsed")
    unique_chunks_indexed: int = Field(..., description="Chunks indexed after deduplication")
    duplicates_skipped: int = Field(..., description="Duplicate chunks skipped via SHA-256")
    duration_ms: float = Field(..., description="Total processing time in milliseconds")


class ScoredChunk(BaseModel):
    """Document chunk with retrieval and reranking scores."""

    content: str
    metadata: DocumentMetadata
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class StageLatencyBreakdown(BaseModel):
    """Millisecond latency breakdown across all pipeline stages."""

    cache_lookup_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    synthesis_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    """User query request payload."""

    query: str = Field(..., min_length=1, max_length=4096, description="Search query or question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Final context chunks to feed LLM")
    similarity_threshold: Optional[float] = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic cache hit",
    )
    bypass_cache: bool = Field(default=False, description="Force bypass of semantic cache")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filtering criteria")


class QueryResponse(BaseModel):
    """Synchronous query response."""

    query: str
    answer: str = Field(..., description="Synthesized grounded answer")
    sources: List[ScoredChunk] = Field(default_factory=list, description="Top context chunks used")
    cache_hit: bool = Field(default=False, description="Whether served from Redis Semantic Cache")
    similarity_score: Optional[float] = Field(default=None, description="Semantic cache similarity if hit")
    latency: StageLatencyBreakdown = Field(..., description="Pipeline execution latency breakdown")


class CacheInvalidateRequest(BaseModel):
    """Request to clear or invalidate semantic cache."""

    query: Optional[str] = Field(default=None, description="Specific query to invalidate (all if omitted)")


class HealthResponse(BaseModel):
    """Healthcheck response status."""

    status: str
    environment: str
    version: str
    services: Dict[str, str]
