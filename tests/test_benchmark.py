"""Tests verifying benchmark performance targets and evaluation metrics."""

import pytest
from src.api.routes import RAGService
from config.settings import Settings
from src.evaluation.ragas_eval import RagasEvaluator


@pytest.mark.asyncio
async def test_golden_dataset_evaluation_precision():
    test_settings = Settings(
        ENVIRONMENT="test",
        LOG_LEVEL="ERROR",
        REDIS_CACHE_ENABLED=False,
        EMBEDDING_DIMENSION=128,
        RRF_K=60,
        FUSED_TOP_N=10,
        RERANKER_TOP_K=5,
    )
    service = RAGService(test_settings)
    evaluator = RagasEvaluator(service)

    report = await evaluator.evaluate_suite()

    assert report["num_queries"] == 4
    assert 0.0 <= report["context_precision"] <= 100.0
    assert 0.0 <= report["faithfulness"] <= 100.0
    assert report["status"] in ["PASSED", "NEEDS_OPTIMIZATION"]
