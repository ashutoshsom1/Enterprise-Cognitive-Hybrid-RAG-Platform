"""Sparse Lexical Retriever using BM25 with tokenization for exact matching."""

import re
import time
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from config.logging_config import get_logger
from config.settings import get_settings
from src.api.schemas import DocumentChunk, ScoredChunk

logger = get_logger(__name__)

# Regex that preserves alphanumeric codes, hyphenated identifiers, and technical terms
TOKEN_PATTERN = re.compile(r"(?u)\b[\w\-\._]+\b")


def default_tokenizer(text: str) -> List[str]:
    """
    Tokenizes text while preserving technical IDs, part numbers, and error codes.
    Lowercases tokens for uniform lexical matching.
    """
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


class SparseBM25Retriever:
    """
    BM25 Sparse Lexical Search Engine.
    Engineered for exact token matching, error codes, legal references,
    and product part identifiers with sub-15ms retrieval latency.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        settings = get_settings()
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes document chunks into BM25 inverted index."""
        if not chunks:
            return 0

        t0 = time.perf_counter()
        self.chunks.extend(chunks)

        new_tokens = [default_tokenizer(c.content) for c in chunks]
        self.corpus_tokens.extend(new_tokens)

        # Rebuild BM25 index with all tokens
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Indexed {len(chunks)} chunks into BM25 (total {len(self.chunks)} chunks in {elapsed_ms:.2f}ms)"
        )
        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredChunk]:
        """
        Retrieves top-k chunks matching the query using BM25 scoring.
        """
        if not self.bm25 or not self.chunks:
            return []

        t0 = time.perf_counter()
        query_tokens = default_tokenizer(query)
        if not query_tokens:
            return []

        # Calculate scores across corpus
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with corresponding chunks
        scored_pairs = []
        for idx, score in enumerate(scores):
            if score > 0.0:  # Only consider non-zero lexical matches
                scored_pairs.append((score, self.chunks[idx]))

        # Sort descending
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        top_pairs = scored_pairs[:top_k]

        results = [
            ScoredChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                sparse_score=float(score),
            )
            for score, chunk in top_pairs
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"BM25 sparse search returned {len(results)} chunks in {elapsed_ms:.2f}ms")
        return results
