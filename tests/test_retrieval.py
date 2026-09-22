"""Tests for Dense and Sparse retrieval engines."""

import pytest
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever


@pytest.mark.asyncio
async def test_sparse_bm25_exact_code_matching(sample_chunks):
    sparse = SparseBM25Retriever()
    sparse.index_chunks(sample_chunks)

    # Search for specific error code that dense search could easily miss
    results = await sparse.search("ERR_THERMAL_THROTTLE_90C", top_k=2)

    assert len(results) > 0
    assert "ERR_THERMAL_THROTTLE_90C" in results[0].content
    assert results[0].metadata.chunk_id == "doc_gpu#c0002"
    assert results[0].sparse_score is not None and results[0].sparse_score > 0.0


@pytest.mark.asyncio
async def test_dense_retrieval_indexing_and_search(sample_chunks):
    dense = DenseRetriever(dimension=128)
    indexed_count = await dense.index_chunks(sample_chunks)
    assert indexed_count == len(sample_chunks)

    results = await dense.search("What happens when temperature is high?", top_k=2)
    assert len(results) <= 2
    assert results[0].dense_score is not None
