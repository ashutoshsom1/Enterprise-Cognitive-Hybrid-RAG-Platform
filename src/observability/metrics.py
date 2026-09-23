"""Prometheus Telemetry Metrics for Enterprise Cognitive Hybrid RAG."""

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Dummy fallbacks if prometheus_client is not installed
    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
    def Counter(*args, **kwargs):
        return DummyMetric()

    def Histogram(*args, **kwargs):
        return DummyMetric()

    def generate_latest():
        return b""

    CONTENT_TYPE_LATEST = "text/plain"

__all__ = [
    "Counter",
    "Histogram",
    "generate_latest",
    "CONTENT_TYPE_LATEST",
    "QUERY_LATENCY_HISTOGRAM",
    "STAGE_LATENCY_HISTOGRAM",
    "CACHE_HITS_TOTAL",
    "CACHE_MISSES_TOTAL",
    "INGESTED_CHUNKS_TOTAL",
    "record_query_metrics",
]

# Query latency histogram with sub-500ms P99 resolution buckets
QUERY_LATENCY_HISTOGRAM = Histogram(
    "rag_query_latency_seconds",
    "End-to-end user query latency in seconds",
    buckets=[0.010, 0.025, 0.050, 0.100, 0.200, 0.350, 0.500, 0.750, 1.0, 2.0, 5.0],
)

# Granular per-stage latency breakdown
STAGE_LATENCY_HISTOGRAM = Histogram(
    "rag_stage_latency_seconds",
    "Latency breakdown across internal RAG stages",
    labelnames=["stage"],
    buckets=[0.005, 0.010, 0.020, 0.035, 0.050, 0.100, 0.200, 0.500, 1.0],
)

# Semantic Cache metrics
CACHE_HITS_TOTAL = Counter(
    "rag_cache_hits_total",
    "Count of queries served directly from Redis semantic vector cache",
)

CACHE_MISSES_TOTAL = Counter(
    "rag_cache_misses_total",
    "Count of queries bypassing or missing semantic vector cache",
)

# Ingestion metrics
INGESTED_CHUNKS_TOTAL = Counter(
    "rag_ingested_chunks_total",
    "Total document chunks indexed into the system",
    labelnames=["status"],
)


def record_query_metrics(
    total_ms: float,
    cache_hit: bool,
    breakdown: dict,
):
    """Utility to record all query metrics in one call."""
    QUERY_LATENCY_HISTOGRAM.observe(total_ms / 1000.0)

    if cache_hit:
        CACHE_HITS_TOTAL.inc()
    else:
        CACHE_MISSES_TOTAL.inc()

    for stage, ms in breakdown.items():
        if ms > 0:
            STAGE_LATENCY_HISTOGRAM.labels(stage=stage).observe(ms / 1000.0)
