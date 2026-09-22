"""Tests for document chunking, deduplication, and ingestion pipeline."""

import pytest
from src.api.schemas import IngestDocument, IngestRequest
from src.ingestion.chunker import RecursiveTokenChunker
from src.ingestion.deduplicator import ChunkDeduplicator
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever


def test_recursive_chunking_with_overlap():
    chunker = RecursiveTokenChunker(chunk_size=5, chunk_overlap=2)
    text = "Paragraph one with several words here.\n\nParagraph two contains additional important statements.\n\nParagraph three closes."

    chunks = chunker.chunk_document(
        doc_id="doc1",
        source="test.txt",
        content=text,
        title="Test Doc",
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata.sha256 is not None
        assert len(c.metadata.sha256) == 64  # SHA-256 hex length
        assert c.metadata.doc_id == "doc1"


def test_chunk_deduplication():
    dedup = ChunkDeduplicator()
    chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=10)

    text = "Identical text content for version 1 and version 2."
    chunks_v1 = chunker.chunk_document("doc_v1", "v1.txt", text)
    chunks_v2 = chunker.chunk_document("doc_v2", "v2.txt", text)

    # First pass: all unique
    unique_v1, dupes_v1 = dedup.filter_unique(chunks_v1)
    assert len(unique_v1) == len(chunks_v1)
    assert dupes_v1 == 0

    # Second pass: duplicate content with identical SHA-256
    unique_v2, dupes_v2 = dedup.filter_unique(chunks_v2)
    assert len(unique_v2) == 0
    assert dupes_v2 == len(chunks_v2)


@pytest.mark.asyncio
async def test_end_to_end_ingestion_pipeline():
    dense = DenseRetriever(dimension=64)
    sparse = SparseBM25Retriever()
    pipeline = IngestionPipeline(dense_retriever=dense, sparse_retriever=sparse)

    docs = [
        IngestDocument(
            source="manual.pdf",
            title="User Manual",
            content="Section 1: Initial System Setup.\n\nSection 2: Maintenance routines.",
        )
    ]
    req = IngestRequest(documents=docs, chunk_size=128, chunk_overlap=20)
    resp = await pipeline.ingest_batch(req)

    assert resp.status == "success"
    assert resp.documents_processed == 1
    assert resp.unique_chunks_indexed > 0
