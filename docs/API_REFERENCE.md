# REST & Streaming API Reference

Base URL: `http://<host>:8000/api/v1`

---

## 1. Document Ingestion

### `POST /ingest`
Batched document ingestion, recursive token chunking, SHA-256 deduplication, and dual indexing into Qdrant & BM25.

#### Request Headers
| Header | Value |
| :--- | :--- |
| `Content-Type` | `application/json` |

#### Request Body
```json
{
  "documents": [
    {
      "doc_id": "doc_contract_001",
      "source": "contracts/master_sla.pdf",
      "title": "Master SLA Contract",
      "content": "Full text of document...",
      "metadata": {
        "department": "legal",
        "security_level": "confidential"
      }
    }
  ],
  "chunk_size": 512,
  "chunk_overlap": 100
}
```

#### Response (`200 OK`)
```json
{
  "status": "success",
  "documents_processed": 1,
  "total_chunks_created": 12,
  "unique_chunks_indexed": 12,
  "duplicates_skipped": 0,
  "duration_ms": 142.8
}
```

---

## 2. Knowledge Base Query

### `POST /query`
Performs end-to-end hybrid retrieval, RRF fusion, Cross-Encoder reranking, and citation-grounded synthesis.

#### Request Body
```json
{
  "query": "What is the penalty credit if uptime drops below 99.90%?",
  "top_k": 5,
  "similarity_threshold": 0.92,
  "bypass_cache": false
}
```

#### Response (`200 OK`)
```json
{
  "query": "What is the penalty credit if uptime drops below 99.90%?",
  "answer": "If uptime drops below 99.90%, the customer is entitled to a 30% service credit under Clause 8.1. [Doc 1: contracts/master_sla.pdf#doc_contract_001#c0002]",
  "sources": [
    {
      "content": "Section 8.1... If uptime drops below 99.90%, the Customer is entitled to a 30% service credit.",
      "metadata": {
        "source": "contracts/master_sla.pdf",
        "doc_id": "doc_contract_001",
        "chunk_id": "doc_contract_001#c0002",
        "chunk_index": 2,
        "title": "Master SLA Contract",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      "dense_score": 0.892,
      "sparse_score": 14.21,
      "rrf_score": 0.0328,
      "rerank_score": 0.9841
    }
  ],
  "cache_hit": false,
  "similarity_score": null,
  "latency": {
    "cache_lookup_ms": 2.1,
    "dense_retrieval_ms": 19.4,
    "sparse_retrieval_ms": 13.8,
    "rrf_fusion_ms": 0.7,
    "rerank_ms": 32.1,
    "synthesis_ms": 241.2,
    "total_ms": 309.3
  }
}
```

---

## 3. Real-Time Token Streaming

### `POST /query/stream`
Server-Sent Events (SSE) endpoint providing streaming tokens with context citations.

#### Response Headers
| Header | Value |
| :--- | :--- |
| `Content-Type` | `text/event-stream` |
| `Cache-Control`| `no-cache` |

#### Event Stream Format
```
data: {"event": "sources", "sources": [{"source": "contracts/master_sla.pdf", "chunk_id": "doc_contract_001#c0002", "score": 0.9841}]}

data: {"event": "token", "token": "Based "}

data: {"event": "token", "token": "on "}

data: {"event": "token", "token": "Clause 8.1... "}

data: [DONE]
```

---

## 4. Cache Invalidation

### `POST /cache/invalidate`
Purges cached queries from the Redis vector semantic cache.

#### Request Body
```json
{
  "query": "What is the penalty credit if uptime drops below 99.90%?"
}
```
*(Leave `query` empty or null to clear all semantic cache entries)*

---

## 5. Health & Monitoring

### `GET /health`
Returns connection status of Redis, Qdrant, BM25, and LLM backends.

### `GET /metrics`
Returns standard Prometheus metrics for Prometheus/Grafana dashboards.
