"""Enterprise Cognitive Hybrid RAG Platform Configuration Settings."""

from functools import lru_cache
from typing import List, Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings with environment variable bindings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General App Settings
    ENVIRONMENT: str = Field(default="production", description="Environment name (development, staging, production)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity")
    APP_HOST: str = Field(default="0.0.0.0", description="FastAPI host binding")
    APP_PORT: int = Field(default=8000, description="FastAPI port binding")
    API_KEY: Optional[str] = Field(default=None, description="Optional API key for gateway authorization")
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # Semantic Cache (Redis)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_CACHE_ENABLED: bool = Field(default=True, description="Enable Redis semantic vector cache")
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hit (default: 0.92)",
    )
    SEMANTIC_CACHE_TTL_SECONDS: int = Field(
        default=86400,
        description="Semantic cache TTL in seconds (default: 24 hours)",
    )

    # Dense Vector Storage (Qdrant)
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant service URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key if secured")
    QDRANT_COLLECTION_NAME: str = Field(
        default="enterprise_knowledge_base",
        description="Qdrant collection name",
    )
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", description="Embedding model identifier")
    EMBEDDING_DIMENSION: int = Field(default=3072, description="Vector dimension for embeddings")
    DENSE_TOP_K: int = Field(default=50, description="Dense candidates to retrieve")

    # Sparse Lexical Search (BM25)
    BM25_K1: float = Field(default=1.5, description="BM25 term frequency saturation parameter")
    BM25_B: float = Field(default=0.75, description="BM25 document length normalization parameter")
    SPARSE_TOP_K: int = Field(default=50, description="Sparse candidates to retrieve")

    # Reciprocal Rank Fusion (RRF)
    RRF_K: int = Field(
        default=60,
        description="Reciprocal Rank Fusion smoothing parameter (k=60 for stabilization)",
    )
    FUSED_TOP_N: int = Field(default=50, description="Candidate pool size after RRF fusion")

    # Deep Cross-Encoder Reranker
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-large",
        description="Cross-encoder model for all-to-all token scoring",
    )
    RERANKER_TOP_K: int = Field(default=5, description="Final context chunks after reranking")
    RERANKER_DEVICE: str = Field(default="cpu", description="Compute device for reranker (cpu, cuda, mps)")
    RERANKER_BATCH_SIZE: int = Field(default=32, description="Batch size for cross-encoder inference")

    # Synthesis LLM Providers
    DEFAULT_LLM_PROVIDER: Literal["openai", "azure_openai", "anthropic", "vllm"] = Field(
        default="openai",
        description="Default LLM synthesis provider",
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI chat completion model")

    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None, description="Azure OpenAI API Key")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None, description="Azure OpenAI Endpoint URL")
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(default=None, description="Azure deployment name")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")

    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API Key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20240620", description="Anthropic model name")

    VLLM_BASE_URL: str = Field(default="http://localhost:8000/v1", description="vLLM OpenAI-compatible endpoint")
    VLLM_MODEL: str = Field(default="meta-llama/Meta-Llama-3-70B-Instruct", description="vLLM model identifier")

    TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    MAX_TOKENS: int = Field(default=2048, description="Max generated tokens")
    STREAM_TIMEOUT_SECONDS: int = Field(default=30, description="Streaming connection timeout")

    # Ingestion & Deduplication
    DEFAULT_CHUNK_SIZE_TOKENS: int = Field(default=512, description="Target chunk size in tokens")
    DEFAULT_CHUNK_OVERLAP_TOKENS: int = Field(default=100, description="Overlap tokens between chunks")
    DEDUPLICATION_ENABLED: bool = Field(default=True, description="Enable SHA-256 chunk deduplication")

    # Observability
    PROMETHEUS_METRICS_ENABLED: bool = Field(default=True, description="Expose /metrics endpoint")
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(default=None, description="OpenTelemetry OTLP endpoint")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
