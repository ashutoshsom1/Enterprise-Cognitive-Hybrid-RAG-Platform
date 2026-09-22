"""Cryptographic SHA-256 Chunk Deduplicator."""

from typing import List, Set, Tuple
from config.logging_config import get_logger
from src.api.schemas import DocumentChunk

logger = get_logger(__name__)


class ChunkDeduplicator:
    """
    Deduplicates incoming document chunks using SHA-256 hashes.
    Prevents duplicate embeddings and redundant index storage across document versions.
    """

    def __init__(self):
        self._seen_hashes: Set[str] = set()

    def filter_unique(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], int]:
        """
        Filters a list of chunks, returning only new unique chunks and the duplicate count.
        """
        unique_chunks: List[DocumentChunk] = []
        duplicates_count = 0

        for chunk in chunks:
            h = chunk.metadata.sha256
            if not h:
                # If no hash attached, keep chunk
                unique_chunks.append(chunk)
                continue

            if h in self._seen_hashes:
                duplicates_count += 1
            else:
                self._seen_hashes.add(h)
                unique_chunks.append(chunk)

        if duplicates_count > 0:
            logger.info(f"Deduplicator skipped {duplicates_count} duplicate chunks.")

        return unique_chunks, duplicates_count

    def clear(self):
        """Reset deduplication state."""
        self._seen_hashes.clear()
