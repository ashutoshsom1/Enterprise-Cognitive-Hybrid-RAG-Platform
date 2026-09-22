"""Pytest configuration and shared fixtures."""

import pytest
import pytest_asyncio
from typing import List

from config.settings import Settings
from src.api.schemas import DocumentChunk, DocumentMetadata, ScoredChunk
from src.api.routes import RAGService
from src.cache.memory_cache import MemorySemanticCache
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever
from src.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def test_settings() -> Settings:
    """Provides isolated test configuration."""
    return Settings(
        ENVIRONMENT="test",
        LOG_LEVEL="DEBUG",
        REDIS_CACHE_ENABLED=False,
        SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.92,
        SEMANTIC_CACHE_TTL_SECONDS=300,
        EMBEDDING_DIMENSION=128,
        RRF_K=60,
        FUSED_TOP_N=10,
        RERANKER_TOP_K=3,
        DENSE_TOP_K=10,
        SPARSE_TOP_K=10,
    )


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    """Provides structured sample enterprise chunks."""
    c1 = DocumentChunk(
        content="Liquidated damages under Clause 14.2 are capped at $2,500,000 per annual cycle.",
        metadata=DocumentMetadata(
            source="sla_v4.pdf",
            doc_id="doc_sla",
            chunk_id="doc_sla#c0001",
            chunk_index=1,
            title="SLA Contract",
            sha256="hash_sla_c0001",
        ),
    )
    c2 = DocumentChunk(
        content="When junction temp exceeds 90C, accelerator triggers error code ERR_THERMAL_THROTTLE_90C.",
        metadata=DocumentMetadata(
            source="gpu_spec.pdf",
            doc_id="doc_gpu",
            chunk_id="doc_gpu#c0002",
            chunk_index=2,
            title="GPU Tech Spec",
            sha256="hash_gpu_c0002",
        ),
    )
    c3 = DocumentChunk(
        content="Kubernetes worker node eviction terminates with exit code OOMKilled_137. Run kubectl drain.",
        metadata=DocumentMetadata(
            source="k8s_runbook.md",
            doc_id="doc_k8s",
            chunk_id="doc_k8s#c0003",
            chunk_index=3,
            title="K8s Runbook",
            sha256="hash_k8s_c0003",
        ),
    )
    return [c1, c2, c3]


@pytest.fixture
def memory_cache(test_settings) -> MemorySemanticCache:
    return MemorySemanticCache(default_ttl_seconds=test_settings.SEMANTIC_CACHE_TTL_SECONDS)


@pytest.fixture
def dense_retriever(test_settings) -> DenseRetriever:
    return DenseRetriever(
        qdrant_url="http://mock-qdrant:6333",
        collection_name="test_collection",
        dimension=test_settings.EMBEDDING_DIMENSION,
    )


@pytest.fixture
def sparse_retriever() -> SparseBM25Retriever:
    return SparseBM25Retriever(k1=1.5, b=0.75)


@pytest.fixture
def cross_encoder() -> CrossEncoderReranker:
    return CrossEncoderReranker(top_k=3)


@pytest.fixture
def test_rag_service(test_settings) -> RAGService:
    return RAGService(test_settings)
