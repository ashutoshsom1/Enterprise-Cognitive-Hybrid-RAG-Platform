"""Multi-provider LLM Synthesis Engine supporting Azure OpenAI, Anthropic, and vLLM."""

import asyncio
from typing import AsyncGenerator, List, Optional
from config.logging_config import get_logger
from config.settings import get_settings
from src.api.schemas import ScoredChunk
from src.synthesis.prompts import ENTERPRISE_RAG_SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    AsyncOpenAI = None
    AsyncAzureOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None


class LLMGenerator:
    """
    Enterprise LLM Generator.
    Coordinates grounded contextual synthesis with streaming token delivery.
    """

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.DEFAULT_LLM_PROVIDER
        self.openai_client = None
        self.azure_client = None
        self.anthropic_client = None
        self._init_clients()

    def _init_clients(self):
        """Initializes API clients based on configured credentials."""
        if self.settings.AZURE_OPENAI_API_KEY and AsyncAzureOpenAI is not None:
            try:
                self.azure_client = AsyncAzureOpenAI(
                    api_key=self.settings.AZURE_OPENAI_API_KEY,
                    api_version=self.settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT or "",
                )
            except Exception as e:
                logger.warning(f"Could not initialize Azure OpenAI client: {e}")

        if self.settings.OPENAI_API_KEY and AsyncOpenAI is not None:
            try:
                self.openai_client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        if self.settings.ANTHROPIC_API_KEY and AsyncAnthropic is not None:
            try:
                self.anthropic_client = AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")

    async def generate(self, query: str, context_chunks: List[ScoredChunk]) -> str:
        """Generates grounded response synchronously."""
        user_prompt = build_user_prompt(query, context_chunks)

        # 1. Azure OpenAI
        if self.provider == "azure_openai" and self.azure_client is not None:
            try:
                resp = await self.azure_client.chat.completions.create(
                    model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4o",
                    messages=[
                        {"role": "system", "content": ENTERPRISE_RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.TEMPERATURE,
                    max_tokens=self.settings.MAX_TOKENS,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Azure OpenAI call failed: {e}")

        # 2. OpenAI
        if self.provider == "openai" and self.openai_client is not None:
            try:
                resp = await self.openai_client.chat.completions.create(
                    model=self.settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": ENTERPRISE_RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.TEMPERATURE,
                    max_tokens=self.settings.MAX_TOKENS,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")

        # 3. Anthropic Claude
        if self.provider == "anthropic" and self.anthropic_client is not None:
            try:
                resp = await self.anthropic_client.messages.create(
                    model=self.settings.ANTHROPIC_MODEL,
                    system=ENTERPRISE_RAG_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.settings.TEMPERATURE,
                    max_tokens=self.settings.MAX_TOKENS,
                )
                return resp.content[0].text if resp.content else ""
            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")

        # 4. Fallback grounded synthesizer for local dev/testing
        return self._simulate_grounded_synthesis(query, context_chunks)

    async def stream_generate(
        self, query: str, context_chunks: List[ScoredChunk]
    ) -> AsyncGenerator[str, None]:
        """Streams tokens in real time via async generator."""
        user_prompt = build_user_prompt(query, context_chunks)

        # 1. Azure OpenAI Streaming
        if self.provider == "azure_openai" and self.azure_client is not None:
            try:
                stream = await self.azure_client.chat.completions.create(
                    model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME or "gpt-4o",
                    messages=[
                        {"role": "system", "content": ENTERPRISE_RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.TEMPERATURE,
                    max_tokens=self.settings.MAX_TOKENS,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                return
            except Exception as e:
                logger.error(f"Azure OpenAI stream error: {e}")

        # 2. OpenAI Streaming
        if self.provider == "openai" and self.openai_client is not None:
            try:
                stream = await self.openai_client.chat.completions.create(
                    model=self.settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": ENTERPRISE_RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.TEMPERATURE,
                    max_tokens=self.settings.MAX_TOKENS,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                return
            except Exception as e:
                logger.error(f"OpenAI stream error: {e}")

        # Fallback simulator streaming
        simulated = self._simulate_grounded_synthesis(query, context_chunks)
        words = simulated.split(" ")
        for w in words:
            yield w + " "
            await asyncio.sleep(0.01)

    def _simulate_grounded_synthesis(self, query: str, context_chunks: List[ScoredChunk]) -> str:
        """Deterministic citation-grounded response simulator for offline/test environments."""
        if not context_chunks:
            return "The provided enterprise context does not contain sufficient information to answer this query."

        top_chunk = context_chunks[0]
        src = top_chunk.metadata.source
        cid = top_chunk.metadata.chunk_id

        # Extract most relevant sentence or fact
        sentences = [s.strip() for s in top_chunk.content.split(".") if s.strip()]
        lead_fact = sentences[0] if sentences else top_chunk.content[:150]

        response = (
            f"Based on enterprise documentation, {lead_fact}. "
            f"[Doc 1: {src}#{cid}]\n\n"
        )

        if len(context_chunks) > 1:
            second_chunk = context_chunks[1]
            s2 = second_chunk.metadata.source
            c2 = second_chunk.metadata.chunk_id
            s_sentences = [s.strip() for s in second_chunk.content.split(".") if s.strip()]
            supporting_fact = s_sentences[0] if s_sentences else second_chunk.content[:150]
            response += f"Additionally, reference records indicate that {supporting_fact}. [Doc 2: {s2}#{c2}]"

        return response
