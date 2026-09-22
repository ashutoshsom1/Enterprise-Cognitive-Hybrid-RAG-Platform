"""Enterprise RAG prompts enforcing strict factual grounding and citations."""

from typing import List
from src.api.schemas import ScoredChunk

ENTERPRISE_RAG_SYSTEM_PROMPT = """You are an Enterprise Knowledge Intelligence Assistant for corporate decision-makers, engineers, and analysts.
Your objective is to provide precise, factually grounded, and comprehensive answers based EXCLUSIVELY on the provided retrieved context documents.

CRITICAL OPERATIONAL RULES:
1. Grounding & Zero Hallucinations: Every factual statement, claim, figure, part number, error code, and policy you mention MUST be directly supported by the context snippets provided below. Never speculate or fabricate facts.
2. Mandatory In-Text Citations: Whenever you state a fact derived from a context document, cite the source immediately using the format: [Doc X: <source>#<chunk_id>].
3. Exact Identifiers: Preserve exact technical codes, version numbers, error codes, and legal terms as they appear in the documents.
4. Insufficient Information: If the provided documents do not contain the answer, explicitly state: "The provided enterprise context does not contain sufficient information to answer this query." Do not attempt to guess.
5. Tone: Maintain a professional, objective, and executive-ready tone.
"""


def build_user_prompt(query: str, context_chunks: List[ScoredChunk]) -> str:
    """Formats context documents and user query into structured prompt."""
    formatted_context = []
    for idx, chunk in enumerate(context_chunks, start=1):
        src = chunk.metadata.source
        cid = chunk.metadata.chunk_id
        title = chunk.metadata.title or "Untitled"
        formatted_context.append(
            f"--- Context Document {idx} [Doc {idx}: {src}#{cid} | Title: {title}] ---\n{chunk.content}\n"
        )

    context_str = "\n".join(formatted_context) if formatted_context else "No context documents retrieved."

    return f"""RETRIEVED CONTEXT DOCUMENTS:
{context_str}

USER QUERY:
{query}

Synthesize a complete, factually grounded response with explicit citations [Doc X: <source>#<chunk_id>] for every claim:"""
