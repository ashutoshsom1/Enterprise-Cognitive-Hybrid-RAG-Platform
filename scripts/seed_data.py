"""Seed Enterprise Knowledge Base with Sample Documents."""

import asyncio
import sys
import httpx

from src.evaluation.benchmark_dataset import SAMPLE_ENTERPRISE_DOCUMENTS


async def seed(base_url: str = "http://localhost:8000"):
    """Sends sample enterprise documents to the /api/v1/ingest endpoint."""
    print(f"Connecting to Enterprise RAG Platform at {base_url}...")

    payload = {
        "documents": SAMPLE_ENTERPRISE_DOCUMENTS,
        "chunk_size": 256,
        "chunk_overlap": 50,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{base_url}/api/v1/ingest", json=payload)
            resp.raise_for_status()
            data = resp.json()
            print("Successfully seeded Enterprise Knowledge Base:")
            print(f"  - Documents Processed: {data['documents_processed']}")
            print(f"  - Total Chunks:        {data['total_chunks_created']}")
            print(f"  - Unique Chunks:       {data['unique_chunks_indexed']}")
            print(f"  - Duplicate Skipped:   {data['duplicates_skipped']}")
            print(f"  - Duration:            {data['duration_ms']:.2f}ms")
        except Exception as e:
            print(f"Error seeding documents: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    asyncio.run(seed(target))
