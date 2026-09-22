"""High-performance in-memory semantic cache with NumPy vector similarity."""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.cache.base import SemanticCacheBase
from config.logging_config import get_logger

logger = get_logger(__name__)


class MemorySemanticCache(SemanticCacheBase):
    """In-memory semantic vector cache using cosine similarity."""

    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        # Mapping key -> dict(query, vector, answer, sources, created_at, expires_at)
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def get(
        self,
        query: str,
        query_vector: List[float],
        similarity_threshold: float = 0.92,
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """Search in-memory entries for closest semantic match above threshold."""
        now = time.time()
        q_vec = np.array(query_vector, dtype=np.float32)

        best_score = -1.0
        best_entry: Optional[Dict[str, Any]] = None

        # Clean expired keys and find highest cosine similarity
        expired_keys = []
        for key, entry in self._entries.items():
            if entry["expires_at"] <= now:
                expired_keys.append(key)
                continue

            entry_vec = entry["vector"]
            score = self._cosine_similarity(q_vec, entry_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        for exp_key in expired_keys:
            del self._entries[exp_key]

        if best_entry and best_score >= similarity_threshold:
            logger.info(
                f"Memory Semantic Cache HIT (similarity: {best_score:.4f} >= {similarity_threshold})",
                extra={"cache_hit": True, "similarity": best_score},
            )
            return (best_entry["answer"], best_entry["sources"], best_score)

        return None

    async def set(
        self,
        query: str,
        query_vector: List[float],
        answer: str,
        sources: List[Dict[str, Any]],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Insert query embedding and answer into in-memory store."""
        ttl = ttl_seconds or self.default_ttl
        now = time.time()
        entry_key = f"cache:{hash(query)}:{int(now * 1000)}"

        self._entries[entry_key] = {
            "query": query,
            "vector": np.array(query_vector, dtype=np.float32),
            "answer": answer,
            "sources": sources,
            "created_at": now,
            "expires_at": now + ttl,
        }
        return True

    async def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate single query or clear entire cache."""
        if query is None:
            count = len(self._entries)
            self._entries.clear()
            return count

        to_remove = [k for k, v in self._entries.items() if v["query"].strip().lower() == query.strip().lower()]
        for k in to_remove:
            del self._entries[k]
        return len(to_remove)

    async def health(self) -> bool:
        """In-memory cache is always operational."""
        return True
