"""High-throughput Concurrency and Latency Benchmark Runner."""

import asyncio
import sys
import time
from typing import List
import httpx
import numpy as np

BENCHMARK_QUERIES = [
    "What is the penalty credit if Monthly Uptime drops to 99.85% under the Master Cloud SLA?",
    "What hardware error code is raised when H100 accelerator junction temperature exceeds 90°C?",
    "What is the immediate Step 1 action when pods are terminated with exit code OOMKilled_137?",
    "Who must approve capital expenditures exceeding $250,000 and how long are audit logs kept?",
    "What is the Thermal Design Power rating for the H100 accelerator?",
    "What are the liquidated damages under Clause 14.2?",
    "What is the required notice period for termination of the Master Cloud SLA?",
    "What script rebalances nodepools during a Kubernetes P1 incident?",
]


async def benchmark_run(
    base_url: str = "http://localhost:8000",
    total_queries: int = 100,
    concurrency: int = 10,
):
    print(f"Starting Enterprise RAG Benchmark:")
    print(f"  Target:      {base_url}")
    print(f"  Queries:     {total_queries}")
    print(f"  Concurrency: {concurrency}")
    print("-" * 65)

    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    cache_hits = 0
    cache_misses = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def send_query(idx: int, q_text: str):
            nonlocal cache_hits, cache_misses
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{base_url}/api/v1/query",
                        json={"query": q_text, "top_k": 5},
                    )
                    latency = (time.perf_counter() - t0) * 1000
                    latencies.append(latency)

                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("cache_hit"):
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    else:
                        print(f"Query {idx} failed: HTTP {resp.status_code}")
                except Exception as e:
                    print(f"Query {idx} error: {e}")

        tasks = []
        for i in range(total_queries):
            query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
            tasks.append(send_query(i, query))

        await asyncio.gather(*tasks)

    if not latencies:
        print("No successful queries recorded.")
        return

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)
    hit_rate = (cache_hits / len(latencies)) * 100

    print("\n================ BENCHMARK RESULTS ================")
    print(f"  Total Requests:      {len(latencies)}")
    print(f"  Cache Hits:          {cache_hits} ({hit_rate:.1f}%)")
    print(f"  Cache Misses:        {cache_misses}")
    print(f"  Mean Latency:        {mean_lat:.2f} ms")
    print(f"  P50 Latency:         {p50:.2f} ms")
    print(f"  P90 Latency:         {p90:.2f} ms")
    print(f"  P95 Latency:         {p95:.2f} ms")
    print(f"  P99 Latency:         {p99:.2f} ms")
    print("---------------------------------------------------")
    if p99 < 500.0:
        print("  SLA Target (<500ms P99): PASSED [COMPLIANT]")
    else:
        print("  SLA Target (<500ms P99): EXCEEDED")
    print("===================================================\n")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    asyncio.run(benchmark_run(url))
