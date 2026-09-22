"""Recursive Token-Aware Document Chunker."""

import hashlib
from typing import Any, Dict, List, Optional
from src.api.schemas import DocumentChunk, DocumentMetadata

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_ENCODER = None


def estimate_tokens(text: str) -> int:
    """Accurately count or estimate tokens in text."""
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass
    # Fast estimation: ~4 chars per token in English
    return max(1, len(text.split()))


class RecursiveTokenChunker:
    """
    Recursively splits corporate documents on structural boundaries (paragraphs, sentences, words)
    while respecting token budget and preserving contextual overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separator: str) -> List[str]:
        if not separator:
            return list(text)
        return text.split(separator)

    def chunk_document(
        self,
        doc_id: str,
        source: str,
        content: str,
        title: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Splits full document content into structured DocumentChunk objects."""
        if not content or not content.strip():
            return []

        raw_chunks = self._recursive_split(content, self.chunk_size, self.chunk_overlap)
        document_chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_cleaned = chunk_text.strip()
            if not chunk_cleaned:
                continue

            # Compute cryptographic hash for chunk deduplication
            chunk_sha256 = hashlib.sha256(chunk_cleaned.encode("utf-8")).hexdigest()
            chunk_id = f"{doc_id}#c{idx:04d}"

            meta = DocumentMetadata(
                source=source,
                doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=idx,
                title=title,
                sha256=chunk_sha256,
                custom_metadata=custom_metadata or {},
            )
            document_chunks.append(DocumentChunk(content=chunk_cleaned, metadata=meta))

        return document_chunks

    def _recursive_split(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """Recursively decomposes text using hierarchical separators."""
        if estimate_tokens(text) <= max_tokens:
            return [text]

        # Choose the first separator present in text
        separator = self.separators[-1]
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        splits = self._split_text(text, separator)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for split in splits:
            split_token_count = estimate_tokens(split)
            if split_token_count > max_tokens:
                # Sub-split long segments recursively
                sub_chunks = self._recursive_split(split, max_tokens, overlap_tokens)
                for sc in sub_chunks:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    chunks.append(sc)
                continue

            if current_tokens + split_token_count > max_tokens:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                # Implement overlap window from end of previous chunk
                overlap_accum = []
                overlap_count = 0
                for item in reversed(current_chunk):
                    item_tok = estimate_tokens(item)
                    if overlap_count + item_tok <= overlap_tokens:
                        overlap_accum.insert(0, item)
                        overlap_count += item_tok
                    else:
                        break

                current_chunk = overlap_accum + [split]
                current_tokens = overlap_count + split_token_count
            else:
                current_chunk.append(split)
                current_tokens += split_token_count

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks
