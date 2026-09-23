"""
Single-command launcher for Enterprise Cognitive Hybrid RAG Platform.
Runs fully locally using Ollama models and Python 3.11 (.venv).
"""

import sys
import os
import asyncio
import httpx
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_settings
from config.logging_config import get_logger

logger = get_logger("run_local")


async def verify_ollama():
    """Checks Ollama server connectivity and model readiness."""
    settings = get_settings()
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    print(f"[*] Checking Ollama server at {settings.OLLAMA_BASE_URL}...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                print(f"[+] Ollama is ONLINE! Available models: {', '.join(models)}")

                # Check embedding model
                emb_model = settings.OLLAMA_EMBEDDING_MODEL
                has_emb = any(emb_model in m for m in models)
                if has_emb:
                    print(f"    - Embedding model '{emb_model}': READY")
                else:
                    print(f"    [!] Warning: Embedding model '{emb_model}' not found in Ollama.")
                    print(f"        Run: ollama pull {emb_model}")

                # Check LLM model
                llm_model = settings.OLLAMA_MODEL
                has_llm = any(llm_model in m for m in models)
                if has_llm:
                    print(f"    - LLM chat model '{llm_model}': READY")
                else:
                    print(f"    [!] Warning: LLM model '{llm_model}' not found in Ollama.")
                    print(f"        Run: ollama pull {llm_model}")
            else:
                print(f"[!] Ollama returned HTTP {resp.status_code}")
    except Exception as e:
        print(f"[!] Could not connect to Ollama ({e}).")
        print("    Please ensure Ollama is running: 'ollama serve'")


def main():
    settings = get_settings()

    print("=" * 70)
    print("  Enterprise Cognitive Hybrid RAG Platform (Local Ollama Edition)")
    print("=" * 70)
    print(f"  Environment:         {settings.ENVIRONMENT}")
    print(f"  LLM Provider:        {settings.DEFAULT_LLM_PROVIDER} ({settings.OLLAMA_MODEL})")
    print(f"  Embedding Provider:  {settings.EMBEDDING_PROVIDER} ({settings.OLLAMA_EMBEDDING_MODEL})")
    print(f"  Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"  Host:                http://{settings.APP_HOST}:{settings.APP_PORT}")
    print(f"  Swagger Docs:        http://{settings.APP_HOST}:{settings.APP_PORT}/docs")
    print("=" * 70)

    # Check Ollama
    asyncio.run(verify_ollama())

    print("\n[*] Starting Uvicorn Gateway Server...")
    uvicorn.run(
        "src.api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
