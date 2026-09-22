"""End-to-End Document Ingestion Pipeline."""

import time
import uuid
from typing import List, Optional
from config.logging_config import get_logger
from src.api.schemas import IngestDocument, IngestRequest, IngestResponse
from src.ingestion.chunker import RecursiveTokenChunker
from src.ingestion.deduplicator import ChunkDeduplicator
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseBM25Retriever

logger = get_logger(__name__)


class IngestionPipeline:
    """
    High-throughput ingestion pipeline handling chunking, deduplication,
    and parallel dual-indexing into Dense Vector (Qdrant) and Sparse (BM25) search.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseBM25Retriever,
        deduplicator: Optional[ChunkDeduplicator] = None,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.deduplicator = deduplicator or ChunkDeduplicator()

    async def ingest_batch(self, request: IngestRequest) -> IngestResponse:
        """Processes and indexes a batch of corporate documents."""
        t0 = time.perf_counter()
        chunker = RecursiveTokenChunker(
            chunk_size=request.chunk_size or 512,
            chunk_overlap=request.chunk_overlap or 100,
        )

        all_chunks = []
        for doc in request.documents:
            doc_id = doc.doc_id or f"doc_{uuid.uuid4().hex[:12]}"
            chunks = chunker.chunk_document(
                doc_id=doc_id,
                source=doc.source,
                content=doc.content,
                title=doc.title,
                custom_metadata=doc.metadata,
            )
            all_chunks.extend(chunks)

        total_chunks = len(all_chunks)

        # Deduplicate identical chunks
        unique_chunks, skipped = self.deduplicator.filter_unique(all_chunks)

        if unique_chunks:
            # Dual index into both Dense and Sparse engines
            await self.dense.index_chunks(unique_chunks)
            self.sparse.index_chunks(unique_chunks)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Ingested {len(request.documents)} docs ({total_chunks} chunks, {len(unique_chunks)} unique, {skipped} dupes) in {elapsed_ms:.2f}ms"
        )

        return IngestResponse(
            status="success",
            documents_processed=len(request.documents),
            total_chunks_created=total_chunks,
            unique_chunks_indexed=len(unique_chunks),
            duplicates_skipped=skipped,
            duration_ms=elapsed_ms,
        )
