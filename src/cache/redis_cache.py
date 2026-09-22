"""Production Redis Semantic Vector Cache with sub-25ms response time."""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.cache.base import SemanticCacheBase
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
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        client = await self._get_client()
        if client is None:
            return None

        t_start = time.perf_counter()
        q_vec = np.array(query_vector, dtype=np.float32)
        q_bytes = self._pack_vector(query_vector)

        try:
            # Attempt 1: Native RediSearch Vector KNN search
            await self._ensure_index()
            if self._index_initialized:
                query_str = f"*=>[KNN 1 @vector $query_vec AS vector_score]"
                res = await client.execute_command(
                    "FT.SEARCH",
                    self.INDEX_NAME,
                    query_str,
                    "PARAMS",
                    "2",
                    "query_vec",
                    q_bytes,
                    "SORTBY",
                    "vector_score",
                    "ASC",
                    "DIALECT",
                    "2",
                )
                if res and res[0] > 0:
                    # Parse RediSearch results
                    doc_fields = res[2]
                    field_dict = {}
                    for i in range(0, len(doc_fields), 2):
                        k = doc_fields[i].decode("utf-8") if isinstance(doc_fields[i], bytes) else doc_fields[i]
                        v = doc_fields[i + 1]
                        field_dict[k] = v

                    # Cosine distance in Redis = 1 - cosine_similarity
                    distance = float(field_dict.get("vector_score", 1.0))
                    similarity = 1.0 - distance

                    if similarity >= threshold:
                        answer = field_dict.get("answer", b"").decode("utf-8")
                        sources_raw = field_dict.get("sources", b"[]").decode("utf-8")
                        sources = json.loads(sources_raw)
                        elapsed_ms = (time.perf_counter() - t_start) * 1000
                        logger.info(
                            f"Redis Semantic Cache HIT via RediSearch ({elapsed_ms:.2f}ms, score: {similarity:.4f})"
                        )
                        return (answer, sources, similarity)

            # Attempt 2: Pipelined scan fallback (standard Redis)
            keys = await client.keys(f"{self.PREFIX}*")
            if not keys:
                return None

            best_sim = -1.0
            best_data: Optional[Dict[str, Any]] = None

            pipe = client.pipeline()
            for k in keys[:100]:  # Evaluate recent active keys
                pipe.hgetall(k)
            results = await pipe.execute()

            for entry in results:
                if not entry or b"vector" not in entry:
                    continue
                v_bytes = entry[b"vector"]
                stored_vec = np.frombuffer(v_bytes, dtype=np.float32)
                norm_q = np.linalg.norm(q_vec)
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
            logger.warning(f"Error querying Redis semantic cache: {e}")

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
        client = await self._get_client()
        if client is None:
            return False

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
            logger.error(f"Failed writing to Redis semantic cache: {e}")
            return False

    async def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate single entry or entire prefix."""
        client = await self._get_client()
        if client is None:
            return 0

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
            logger.error(f"Error invalidating Redis cache: {e}")
            return 0

    async def health(self) -> bool:
        """Verify Redis ping."""
        try:
            client = await self._get_client()
            if client is None:
                return False
            return await client.ping()
        except Exception:
            return False
