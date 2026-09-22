"""Abstract Base Class for Semantic Cache implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class SemanticCacheBase(ABC):
    """Interface contract for semantic vector caches."""

    @abstractmethod
    async def get(
        self,
        query: str,
        query_vector: List[float],
        similarity_threshold: float = 0.92,
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """
        Lookup semantic cache for a query vector.

        Returns:
            Tuple of (synthesized_answer, sources_list, similarity_score) if cache hit (>= threshold),
            otherwise None.
        """
        pass

    @abstractmethod
    async def set(
        self,
        query: str,
        query_vector: List[float],
        answer: str,
        sources: List[Dict[str, Any]],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Store query, embedding, answer, and sources into the cache."""
        pass

    @abstractmethod
    async def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate single entry or entire cache. Returns number of keys removed."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Check cache backend connectivity."""
        pass
