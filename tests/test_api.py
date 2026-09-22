"""FastAPI Integration and Endpoints Tests."""

import pytest
import httpx
from httpx import ASGITransport
from src.api.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "X-Correlation-ID" in resp.headers
        assert "X-Response-Time-ms" in resp.headers


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Health check
        h_resp = await client.get("/api/v1/health")
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert "status" in h_data
        assert "services" in h_data

        # Metrics
        m_resp = await client.get("/api/v1/metrics")
        assert m_resp.status_code == 200


@pytest.mark.asyncio
async def test_full_rag_ingest_and_query_flow():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest document
        ingest_payload = {
            "documents": [
                {
                    "source": "contract_2024.pdf",
                    "title": "Cloud SLA",
                    "content": "Uptime guarantee is 99.99%. Liquidated damages are governed by Clause 14.2.",
                }
            ],
            "chunk_size": 256,
            "chunk_overlap": 50,
        }
        ing_resp = await client.post("/api/v1/ingest", json=ingest_payload)
        assert ing_resp.status_code == 200
        assert ing_resp.json()["status"] == "success"

        # 2. Query knowledge base (first time: Cache Miss)
        q_payload = {
            "query": "What clause governs liquidated damages?",
            "top_k": 3,
        }
        q_resp1 = await client.post("/api/v1/query", json=q_payload)
        assert q_resp1.status_code == 200
        data1 = q_resp1.json()
        assert data1["cache_hit"] is False
        assert len(data1["sources"]) > 0
        assert data1["latency"]["total_ms"] > 0

        # 3. Query knowledge base again (second time: Cache Hit via Semantic Cache)
        # Give async cache set a tiny moment
        import asyncio
        await asyncio.sleep(0.05)

        q_resp2 = await client.post("/api/v1/query", json=q_payload)
        assert q_resp2.status_code == 200
        data2 = q_resp2.json()
        # Should be a cache hit with sub-25ms cache lookup
        assert data2["cache_hit"] is True
        assert data2["latency"]["cache_lookup_ms"] < 25.0

        # 4. Invalidate cache
        inv_resp = await client.post("/api/v1/cache/invalidate", json={})
        assert inv_resp.status_code == 200
        assert inv_resp.json()["status"] == "success"


@pytest.mark.asyncio
async def test_streaming_query_endpoint():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        q_payload = {
            "query": "Explain the uptime guarantee under the contract.",
            "top_k": 3,
            "bypass_cache": True,
        }
        resp = await client.post("/api/v1/query/stream", json=q_payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "data: " in text
        assert "[DONE]" in text
