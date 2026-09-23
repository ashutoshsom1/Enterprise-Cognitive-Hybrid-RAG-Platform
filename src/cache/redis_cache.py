"""Production Redis Semantic Vector Cache with sub-25ms response time."""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.cache.base import SemanticCacheBase
from src.cache.memory_cache import MemorySemanticCache
from config.logging_config import get_logger

logger = get_logger(__name__)

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None


class RedisSemanticCache(SemanticCacheBase):
    """
    Redis-backed Semantic Vector Cache.
    Stores query embeddings, checks cosine similarity > 0.92,
    and returns cached synthesis with sub-25ms latency.
    Seamlessly falls back to high-performance in-memory cache if Redis is offline.
    """

    INDEX_NAME = "idx:semantic_cache"
    PREFIX = "semcache:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        similarity_threshold: float = 0.92,
        default_ttl_seconds: int = 86400,
        dimension: int = 3072,
    ):
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl_seconds
        self.dimension = dimension
        self.client: Optional[Any] = None
        self._index_initialized = False
        self._memory_fallback: Optional[MemorySemanticCache] = None
        self._redis_failed: bool = False

    def _get_fallback(self) -> MemorySemanticCache:
        if self._memory_fallback is None:
            self._memory_fallback = MemorySemanticCache(default_ttl_seconds=self.default_ttl)
        return self._memory_fallback

    async def _get_client(self):
        if self.client is None and aioredis is not None:
            self.client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
        return self.client

    def _pack_vector(self, vector: List[float]) -> bytes:
        """Convert float list into 32-bit floating point byte buffer."""
        return np.array(vector, dtype=np.float32).tobytes()

    async def _ensure_index(self):
        """Creates RediSearch vector HNSW index if supported by the Redis instance."""
        if self._index_initialized:
            return
        client = await self._get_client()
        if client is None:
            return

        try:
            # Check if index exists
            await client.execute_command("FT.INFO", self.INDEX_NAME)
            self._index_initialized = True
        except Exception:
            try:
                # Create HNSW vector index
                await client.execute_command(
                    "FT.CREATE",
                    self.INDEX_NAME,
                    "ON",
                    "HASH",
                    "PREFIX",
                    "1",
                    self.PREFIX,
                    "SCHEMA",
                    "query",
                    "TEXT",
                    "vector",
                    "VECTOR",
                    "HNSW",
                    "6",
                    "TYPE",
                    "FLOAT32",
                    "DIM",
                    str(self.dimension),
                    "DISTANCE_METRIC",
                    "COSINE",
                )
                self._index_initialized = True
                logger.info(f"Initialized RediSearch HNSW index '{self.INDEX_NAME}' (dim={self.dimension})")
            except Exception as e:
                logger.warning(f"RediSearch HNSW index creation skipped (standard Redis fallback mode): {e}")
                self._index_initialized = False

    async def get(
        self,
        query: str,
        query_vector: List[float],
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, List[Dict[str, Any]], float]]:
        """
        Query Redis vector index or scan cached keys for cosine similarity match >= threshold.
        """
        if self._redis_failed:
            return await self._get_fallback().get(query, query_vector, similarity_threshold)

        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        client = await self._get_client()
        if client is None:
            return await self._get_fallback().get(query, query_vector, similarity_threshold)

        t_start = time.perf_counter()
        q_vec = np.array(query_vector, dtype=np.float32)
        q_bytes = self._pack_vector(query_vector)

        try:
            # Attempt 1: Native RediSearch Vector KNN search
            await self._ensure_index()
            if self._index_initialized:
                query_str = "*=>[KNN 1 @vector $query_vec AS vector_score]"
                res = await client.execute_command(
                    "FT.SEARCH",
                    self.INDEX_NAME,
                    query_str,
                    "PARAMS",
                    "2",
                    "query_vec",
                    q_bytes,
                    "RETURN",
                    "3",
                    "answer",
                    "sources",
                    "vector_score",
                    "DIALECT",
                    "2",
                )
                if res and len(res) > 1 and res[0] > 0:
                    props = res[2]
                    score_idx = props.index(b"vector_score") if b"vector_score" in props else -1
                    ans_idx = props.index(b"answer") if b"answer" in props else -1
                    src_idx = props.index(b"sources") if b"sources" in props else -1

                    if score_idx != -1 and ans_idx != -1:
                        dist = float(props[score_idx + 1])
                        sim_score = 1.0 - dist
                        if sim_score >= threshold:
                            answer = props[ans_idx + 1].decode("utf-8")
                            sources_raw = props[src_idx + 1].decode("utf-8") if src_idx != -1 else "[]"
                            sources = json.loads(sources_raw)
                            elapsed_ms = (time.perf_counter() - t_start) * 1000
                            logger.info(f"Redis Semantic Cache HIT ({elapsed_ms:.2f}ms, score: {sim_score:.4f})")
                            return (answer, sources, sim_score)

            # Attempt 2: Fallback in-memory scan across keys
            keys = await client.keys(f"{self.PREFIX}*")
            if not keys:
                return None

            best_sim = -1.0
            best_data = None
            norm_q = np.linalg.norm(q_vec)

            for key in keys[:50]:
                entry = await client.hgetall(key)
                if not entry or b"vector" not in entry:
                    continue
                stored_vec = np.frombuffer(entry[b"vector"], dtype=np.float32)
                norm_s = np.linalg.norm(stored_vec)
                if norm_q > 0 and norm_s > 0:
                    sim = float(np.dot(q_vec, stored_vec) / (norm_q * norm_s))
                    if sim > best_sim:
                        best_sim = sim
                        best_data = entry

            if best_data and best_sim >= threshold:
                answer = best_data[b"answer"].decode("utf-8")
                sources = json.loads(best_data[b"sources"].decode("utf-8"))
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                logger.info(f"Redis Semantic Cache HIT via fallback ({elapsed_ms:.2f}ms, score: {best_sim:.4f})")
                return (answer, sources, best_sim)

        except Exception as e:
            logger.warning(f"Error querying Redis semantic cache: {e}. Switching to in-memory fallback.")
            self._redis_failed = True
            return await self._get_fallback().get(query, query_vector, similarity_threshold)

        return None

    async def set(
        self,
        query: str,
        query_vector: List[float],
        answer: str,
        sources: List[Dict[str, Any]],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Store query embedding, answer, and sources with TTL."""
        if self._redis_failed:
            return await self._get_fallback().set(query, query_vector, answer, sources, ttl_seconds)

        client = await self._get_client()
        if client is None:
            return await self._get_fallback().set(query, query_vector, answer, sources, ttl_seconds)

        ttl = ttl_seconds or self.default_ttl
        key = f"{self.PREFIX}{abs(hash(query))}_{int(time.time()*1000)}"

        try:
            q_bytes = self._pack_vector(query_vector)
            mapping = {
                "query": query,
                "answer": answer,
                "sources": json.dumps(sources),
                "vector": q_bytes,
                "created_at": str(time.time()),
            }
            await client.hset(key, mapping=mapping)
            await client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning(f"Failed writing to Redis semantic cache: {e}. Switching to in-memory fallback.")
            self._redis_failed = True
            return await self._get_fallback().set(query, query_vector, answer, sources, ttl_seconds)

    async def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate single entry or entire prefix."""
        if self._redis_failed:
            return await self._get_fallback().invalidate(query)

        client = await self._get_client()
        if client is None:
            return await self._get_fallback().invalidate(query)

        try:
            if query is None:
                keys = await client.keys(f"{self.PREFIX}*")
                if keys:
                    return await client.delete(*keys)
                return 0
            else:
                # Find matching query
                keys = await client.keys(f"{self.PREFIX}*")
                removed = 0
                for k in keys:
                    stored_q = await client.hget(k, "query")
                    if stored_q and stored_q.decode("utf-8").strip().lower() == query.strip().lower():
                        await client.delete(k)
                        removed += 1
                return removed
        except Exception as e:
            logger.warning(f"Error invalidating Redis cache: {e}. Switching to in-memory fallback.")
            self._redis_failed = True
            return await self._get_fallback().invalidate(query)

    async def health(self) -> bool:
        """Verify Redis ping or fallback health."""
        if self._redis_failed:
            return await self._get_fallback().health()
        try:
            client = await self._get_client()
            if client is None:
                return await self._get_fallback().health()
            return await client.ping()
        except Exception:
            return await self._get_fallback().health()
