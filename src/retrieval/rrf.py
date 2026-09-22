"""Reciprocal Rank Fusion (RRF) implementation for Hybrid Search."""

from typing import Dict, List
from src.api.schemas import ScoredChunk


def reciprocal_rank_fusion(
    dense_results: List[ScoredChunk],
    sparse_results: List[ScoredChunk],
    k: int = 60,
    top_n: int = 50,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[ScoredChunk]:
    """
    Combines dense and sparse ranked lists using Reciprocal Rank Fusion (RRF):

        RRF_Score(d) = sum_{m in M} ( w_m / (k + r_m(d)) )

    Parameters:
        dense_results: Ranked candidates from dense vector search.
        sparse_results: Ranked candidates from BM25 sparse search.
        k: Smoothing factor to stabilize ranking variances (default: 60).
        top_n: Maximum number of fused candidate chunks to return (default: 50).
        dense_weight: Relative weighting multiplier for dense ranking (default: 1.0).
        sparse_weight: Relative weighting multiplier for sparse ranking (default: 1.0).

    Returns:
        List of ScoredChunk sorted descending by fused rrf_score.
    """
    # Mapping: chunk_id -> dict with accumulated rrf_score and chunk data
    fused_scores: Dict[str, float] = {}
    chunk_store: Dict[str, ScoredChunk] = {}

    # Process Dense Results (1-based rank)
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk.metadata.chunk_id
        score = dense_weight / (k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + score

        if cid not in chunk_store:
            chunk_store[cid] = chunk
        else:
            chunk_store[cid].dense_score = chunk.dense_score

    # Process Sparse Results (1-based rank)
    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk.metadata.chunk_id
        score = sparse_weight / (k + rank)
        fused_scores[cid] = fused_scores.get(cid, 0.0) + score

        if cid not in chunk_store:
            chunk_store[cid] = chunk
        else:
            chunk_store[cid].sparse_score = chunk.sparse_score

    # Sort candidates by combined RRF score descending
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Compile fused top_n candidates
    fused_results: List[ScoredChunk] = []
    for cid, rrf_score in sorted_items[:top_n]:
        candidate = chunk_store[cid]
        # Attach the calculated RRF score
        fused_chunk = ScoredChunk(
            content=candidate.content,
            metadata=candidate.metadata,
            dense_score=candidate.dense_score,
            sparse_score=candidate.sparse_score,
            rrf_score=rrf_score,
        )
        fused_results.append(fused_chunk)

    return fused_results
