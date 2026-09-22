"""Tests for Deep Cross-Encoder Reranker."""

import pytest
from src.api.schemas import DocumentMetadata, ScoredChunk
from src.retrieval.reranker import CrossEncoderReranker


def make_chunk(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        content=text,
        metadata=DocumentMetadata(
            source="manual.pdf",
            doc_id="doc1",
            chunk_id=cid,
            chunk_index=0,
        ),
    )


@pytest.mark.asyncio
async def test_cross_encoder_promotes_relevant_chunk():
    reranker = CrossEncoderReranker(top_k=2)

    c_irrelevant = make_chunk("c1", "The company cafeteria is open between 11:30 AM and 2:00 PM.")
    c_marginal = make_chunk("c2", "General server rack guidelines and power standards for the datacenter.")
    c_target = make_chunk(
        "c3", "For H100 accelerators, temperature exceeding 90C triggers fault ERR_THERMAL_THROTTLE_90C."
    )

    query = "What error code is raised when H100 temperature exceeds 90C?"
    candidates = [c_irrelevant, c_marginal, c_target]

    reranked = await reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    # The target chunk containing ERR_THERMAL_THROTTLE_90C must be ranked #1
    assert reranked[0].metadata.chunk_id == "c3"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


@pytest.mark.asyncio
async def test_cross_encoder_truncation():
    reranker = CrossEncoderReranker(top_k=3)
    candidates = [make_chunk(f"c{i}", f"Document text segment {i}") for i in range(10)]

    reranked = await reranker.rerank("Query text", candidates, top_k=3)
    assert len(reranked) == 3
