"""FastAPI API Endpoints for Enterprise Cognitive Hybrid RAG Platform."""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from config.logging_config import get_logger
from config.settings import Settings, get_settings
from src.api.schemas import (
    CacheInvalidateRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
    StageLatencyBreakdown,
)
from src.cache.base import SemanticCacheBase
from src.cache.memory_cache import MemorySemanticCache
from src.cache.redis_cache import RedisSemanticCache
from src.ingestion.pipeline import IngestionPipeline
from src.observability.metrics import (
    CONTENT_TYPE_LATEST,
    generate_latest,
    record_query_metrics,
)
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.sparse import SparseBM25Retriever
from src.synthesis.generator import LLMGenerator

logger = get_logger(__name__)
router = APIRouter()


class RAGService:
    """Singleton service bundle managing pipelines and state."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # 1. Semantic Cache (Redis or In-Memory fallback)
        if settings.REDIS_CACHE_ENABLED:
            self.cache: SemanticCacheBase = RedisSemanticCache(
                redis_url=settings.REDIS_URL,
                similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
                default_ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS,
                dimension=settings.EMBEDDING_DIMENSION,
            )
        else:
            self.cache = MemorySemanticCache(
                default_ttl_seconds=settings.SEMANTIC_CACHE_TTL_SECONDS
            )

        # 2. Retrieval Engines
        self.dense = DenseRetriever(
            qdrant_url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            embedding_model=settings.EMBEDDING_MODEL,
            settings=settings,
        )
        self.sparse = SparseBM25Retriever(
            k1=settings.BM25_K1,
            b=settings.BM25_B,
        )

        # 3. Reranker
        self.reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            top_k=settings.RERANKER_TOP_K,
            device=settings.RERANKER_DEVICE,
            batch_size=settings.RERANKER_BATCH_SIZE,
        )

        # 4. Ingestion Pipeline
        self.ingestion = IngestionPipeline(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
        )

        # 5. LLM Synthesis
        self.generator = LLMGenerator(settings=settings)


_rag_service: Optional[RAGService] = None


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
    return _rag_service


@router.post("/query", response_model=QueryResponse, summary="Query Enterprise Knowledge Engine")
async def query_knowledge_base(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Executes high-throughput Hybrid RAG with sub-500ms P99 SLA:
    1. Query embedding generation
    2. Redis Semantic Cache vector lookup (cosine > 0.92, <25ms response)
    3. Parallel Dense Vector (HNSW: 20ms) & Sparse BM25 (Exact: 15ms)
    4. Reciprocal Rank Fusion (RRF k=60, Top 50 candidates)
    5. Deep Cross-Encoder Reranking (bge-reranker-large: 35ms, Top 5 chunks)
    6. Grounded LLM Contextual Synthesis with citations
    """
    t_start = time.perf_counter()
    breakdown = StageLatencyBreakdown()

    # Step 1: Compute query embedding
    query_vector = await service.dense.get_embedding(request.query)

    # Step 2: Semantic Cache Lookup
    t_c0 = time.perf_counter()
    cached = None
    if not request.bypass_cache:
        cached = await service.cache.get(
            query=request.query,
            query_vector=query_vector,
            similarity_threshold=request.similarity_threshold or 0.92,
        )
    breakdown.cache_lookup_ms = (time.perf_counter() - t_c0) * 1000

    if cached:
        answer, sources_data, sim_score = cached
        total_ms = (time.perf_counter() - t_start) * 1000
        breakdown.total_ms = total_ms

        sources = [ScoredChunk(**s) for s in sources_data]
        record_query_metrics(total_ms, cache_hit=True, breakdown=breakdown.model_dump())

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            cache_hit=True,
            similarity_score=sim_score,
            latency=breakdown,
        )

    # Step 3: Parallel Hybrid Retrieval (Dense HNSW + Sparse BM25)
    t_dense0 = time.perf_counter()
    t_sparse0 = time.perf_counter()

    dense_task = asyncio.create_task(
        service.dense.search(
            query=request.query,
            query_vector=query_vector,
            top_k=service.settings.DENSE_TOP_K,
            filters=request.filters,
        )
    )
    sparse_task = asyncio.create_task(
        service.sparse.search(
            query=request.query,
            top_k=service.settings.SPARSE_TOP_K,
            filters=request.filters,
        )
    )

    dense_chunks, sparse_chunks = await asyncio.gather(dense_task, sparse_task)
    breakdown.dense_retrieval_ms = (time.perf_counter() - t_dense0) * 1000
    breakdown.sparse_retrieval_ms = (time.perf_counter() - t_sparse0) * 1000

    # Step 4: Reciprocal Rank Fusion (k=60)
    t_rrf0 = time.perf_counter()
    fused_candidates = reciprocal_rank_fusion(
        dense_results=dense_chunks,
        sparse_results=sparse_chunks,
        k=service.settings.RRF_K,
        top_n=service.settings.FUSED_TOP_N,
    )
    breakdown.rrf_fusion_ms = (time.perf_counter() - t_rrf0) * 1000

    # Step 5: Deep Cross-Encoder Reranker
    t_rerank0 = time.perf_counter()
    top_chunks = await service.reranker.rerank(
        query=request.query,
        candidates=fused_candidates,
        top_k=request.top_k or service.settings.RERANKER_TOP_K,
    )
    breakdown.rerank_ms = (time.perf_counter() - t_rerank0) * 1000

    # Step 6: Grounded LLM Context Synthesis
    t_synth0 = time.perf_counter()
    answer = await service.generator.generate(request.query, top_chunks)
    breakdown.synthesis_ms = (time.perf_counter() - t_synth0) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    breakdown.total_ms = total_ms

    # Step 7: Asynchronously update Semantic Cache
    if not request.bypass_cache and answer:
        sources_payload = [c.model_dump() for c in top_chunks]
        asyncio.create_task(
            service.cache.set(
                query=request.query,
                query_vector=query_vector,
                answer=answer,
                sources=sources_payload,
            )
        )

    # Record Prometheus metrics
    record_query_metrics(total_ms, cache_hit=False, breakdown=breakdown.model_dump())

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=top_chunks,
        cache_hit=False,
        similarity_score=None,
        latency=breakdown,
    )


@router.post("/query/stream", summary="Stream Synthesized Answer via Server-Sent Events (SSE)")
async def stream_query_knowledge_base(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    """
    Streams tokens in real time via Server-Sent Events (SSE) while preserving
    the full two-stage hybrid retrieval & reranking pipeline.
    """
    query_vector = await service.dense.get_embedding(request.query)

    # Check cache first
    if not request.bypass_cache:
        cached = await service.cache.get(
            query=request.query,
            query_vector=query_vector,
            similarity_threshold=request.similarity_threshold or 0.92,
        )
        if cached:
            cached_answer, _, _ = cached
            async def cached_stream():
                payload = json.dumps({"event": "cache_hit", "content": cached_answer})
                yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # Run Parallel Hybrid Retrieval
    dense_task = asyncio.create_task(
        service.dense.search(request.query, query_vector, top_k=service.settings.DENSE_TOP_K)
    )
    sparse_task = asyncio.create_task(
        service.sparse.search(request.query, top_k=service.settings.SPARSE_TOP_K)
    )
    dense_chunks, sparse_chunks = await asyncio.gather(dense_task, sparse_task)

    # RRF Fusion & Cross-Encoder Rerank
    fused = reciprocal_rank_fusion(dense_chunks, sparse_chunks, k=service.settings.RRF_K, top_n=service.settings.FUSED_TOP_N)
    top_chunks = await service.reranker.rerank(request.query, fused, top_k=request.top_k or 5)

    async def token_event_generator() -> AsyncGenerator[str, None]:
        full_tokens = []
        # Emit retrieved source metadata first
        sources_meta = [{"source": c.metadata.source, "chunk_id": c.metadata.chunk_id, "score": c.rerank_score} for c in top_chunks]
        yield f"data: {json.dumps({'event': 'sources', 'sources': sources_meta})}\n\n"

        async for token in service.generator.stream_generate(request.query, top_chunks):
            full_tokens.append(token)
            yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

        # Cache full synthesized answer in background
        complete_answer = "".join(full_tokens)
        if not request.bypass_cache and complete_answer:
            asyncio.create_task(
                service.cache.set(
                    query=request.query,
                    query_vector=query_vector,
                    answer=complete_answer,
                    sources=[c.model_dump() for c in top_chunks],
                )
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_event_generator(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse, summary="Ingest Corporate Documents")
async def ingest_documents(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """
    Ingests, chunks, deduplicates (SHA-256), and indexes corporate documents into
    both dense vector and sparse lexical engines.
    """
    return await service.ingestion.ingest_batch(request)


@router.post("/cache/invalidate", summary="Invalidate Semantic Vector Cache")
async def invalidate_cache(
    request: CacheInvalidateRequest,
    service: RAGService = Depends(get_rag_service),
):
    """Purges matching or all entries from the Redis Semantic Cache."""
    count = await service.cache.invalidate(request.query)
    return {"status": "success", "keys_invalidated": count}


@router.get("/health", response_model=HealthResponse, summary="Platform Health Check")
async def health_check(
    service: RAGService = Depends(get_rag_service),
) -> HealthResponse:
    """Verifies operational status of vector stores, cache, and search components."""
    cache_healthy = await service.cache.health()

    return HealthResponse(
        status="healthy" if cache_healthy else "degraded",
        environment=service.settings.ENVIRONMENT,
        version="1.0.0",
        services={
            "semantic_cache": "connected" if cache_healthy else "offline",
            "dense_retriever": "connected" if service.dense._qdrant_available else "in-memory-fallback",
            "sparse_bm25": "active",
            "cross_encoder": "active" if service.reranker.model is not None else "heuristic-fallback",
        },
    )


@router.get("/metrics", summary="Prometheus Metrics Endpoint")
async def metrics_endpoint():
    """Exposes Prometheus time-series metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
