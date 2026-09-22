"""Enterprise Cognitive Hybrid RAG Platform FastAPI Application Factory."""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.api.routes import router as rag_router
from src.observability.tracer import setup_tracer

logger = get_logger("enterprise_rag.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown handler."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger.info("Initializing Enterprise Cognitive Hybrid RAG Platform...")

    # Initialize tracing
    setup_tracer()

    logger.info(
        f"Platform ready on {settings.APP_HOST}:{settings.APP_PORT} (Env: {settings.ENVIRONMENT})"
    )
    yield
    logger.info("Shutting down Enterprise Cognitive Hybrid RAG Platform...")


def create_app() -> FastAPI:
    """Instantiates and configures the production FastAPI gateway."""
    settings = get_settings()

    app = FastAPI(
        title="Enterprise Cognitive Hybrid RAG Platform",
        version="1.0.0",
        description=(
            "High-throughput enterprise knowledge engine combining Dense Vector & BM25 Sparse Search "
            "with Reciprocal Rank Fusion (k=60), Cross-Encoder reranking (bge-reranker-large), "
            "Redis Semantic Caching (<25ms), and sub-500ms P99 latency."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Correlation ID and Timing Middleware
    @app.middleware("http")
    async def add_observability_headers(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        t0 = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.2f}"
        return response

    # Mount RAG API routes
    app.include_router(rag_router, prefix="/api/v1", tags=["RAG Pipeline"])

    # Root redirect / status
    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": "Enterprise Cognitive Hybrid RAG Platform",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs",
            "metrics": "/api/v1/metrics",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("src.api.main:app", host=cfg.APP_HOST, port=cfg.APP_PORT, reload=False)
