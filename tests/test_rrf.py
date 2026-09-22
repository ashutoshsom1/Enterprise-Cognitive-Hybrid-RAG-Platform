"""Tests for Reciprocal Rank Fusion (RRF) mathematical properties."""

import pytest
from src.api.schemas import DocumentMetadata, ScoredChunk
from src.retrieval.rrf import reciprocal_rank_fusion


def make_chunk(cid: str, text: str = "sample text") -> ScoredChunk:
    return ScoredChunk(
        content=text,
        metadata=DocumentMetadata(
            source="test.pdf",
            doc_id=f"doc_{cid}",
            chunk_id=cid,
            chunk_index=0,
        ),
    )


def test_rrf_formula_exact_computation():
    """Validates exact numerical score: 1 / (k + rank)."""
    k = 60
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")

    # Dense: c1 is rank 1, c2 is rank 2
    dense = [c1, c2]
    # Sparse: empty
    sparse = []

    fused = reciprocal_rank_fusion(dense, sparse, k=k, top_n=10)

    assert len(fused) == 2
    # Rank 1: 1 / (60 + 1) = 1/61
    assert pytest.approx(fused[0].rrf_score, 1e-6) == (1.0 / 61.0)
    # Rank 2: 1 / (60 + 2) = 1/62
    assert pytest.approx(fused[1].rrf_score, 1e-6) == (1.0 / 62.0)


def test_rrf_dual_source_agreement_elevation():
    """
    Validates that a document appearing in BOTH dense and sparse results
    receives higher score than documents appearing in only one list.
    """
    k = 60
    cA = make_chunk("A")  # Ranked #1 in dense, #1 in sparse
    cB = make_chunk("B")  # Ranked #2 in dense only
    cC = make_chunk("C")  # Ranked #2 in sparse only

    dense = [cA, cB]
    sparse = [cA, cC]

    fused = reciprocal_rank_fusion(dense, sparse, k=k, top_n=5)

    # cA score should be 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    assert fused[0].metadata.chunk_id == "A"
    assert pytest.approx(fused[0].rrf_score, 1e-6) == (2.0 / 61.0)

    # cB and cC should tie at 1/(60+2) = 1/62 ≈ 0.01613
    assert pytest.approx(fused[1].rrf_score, 1e-6) == (1.0 / 62.0)
    assert pytest.approx(fused[2].rrf_score, 1e-6) == (1.0 / 62.0)


def test_rrf_top_n_truncation():
    """Ensures output is properly bounded by top_n."""
    dense = [make_chunk(f"d_{i}") for i in range(20)]
    sparse = [make_chunk(f"s_{i}") for i in range(20)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=15)
    assert len(fused) == 15
