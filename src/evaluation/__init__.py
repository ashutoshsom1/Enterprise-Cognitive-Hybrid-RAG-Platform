"""Evaluation framework and benchmark package."""

from src.evaluation.benchmark_dataset import GOLDEN_BENCHMARK_DATASET, SAMPLE_ENTERPRISE_DOCUMENTS
from src.evaluation.ragas_eval import RagasEvaluator

__all__ = ["GOLDEN_BENCHMARK_DATASET", "SAMPLE_ENTERPRISE_DOCUMENTS", "RagasEvaluator"]
