"""
Prism Multi-Provider LLM Client Factory.

Supports: LiteLLM Gateway
All calls are async. Synchronous SDKs are wrapped with asyncio.to_thread().

Usage:
    from utils.llm_factory import LLMClient
    client = LLMClient()
    response = await client.generate("Analyze this code...")
    embedding = await client.embed("some text")
"""

import asyncio

from observability.logging import get_logger
from utils.config import settings

logger = get_logger(__name__)

# Global cache for the embedding model to avoid reloading on every request
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Lazy load to avoid memory overhead on startup
        from sentence_transformers import SentenceTransformer

        logger.info("loading_sentence_transformers_model", model=settings.embedding_model)
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


class LLMClient:
    """
    Unified async LLM client using LiteLLM as a library.
    """

    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.llm_provider
        # Map generic provider to specific model string for litellm
        if self.provider == "groq":
            self.model = "groq/llama-3.1-70b-versatile"
        elif self.provider == "gemini":
            self.model = "gemini/gemini-2.0-flash"
        elif self.provider == "openai":
            self.model = "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.model = "claude-3-haiku-20240307"
        elif self.provider == "bedrock":
            # Claude 3.5 Haiku: 200K context, $0.25/M input, $1.25/M output
            # Best price/performance for code review on Bedrock
            self.model = "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0"
        else:
            self.model = "groq/llama-3.1-70b-versatile"  # fallback default

    # ── Text Generation ───────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        """
        Generate text using litellm.
        Returns the response text content.
        """
        from observability.metrics import track_llm_call
        import litellm

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with track_llm_call(agent="LLMClient", model=self.model) as tracker:
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if hasattr(response, "usage") and response.usage:
                    tracker.record_tokens(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )

                return response.choices[0].message.content
            except Exception as e:
                logger.error(
                    "llm_generation_failed",
                    model=self.model,
                    error=str(e),
                )
                raise

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text using sentence-transformers locally.
        """
        if not text or not text.strip():
            return [0.0] * settings.embedding_dim

        try:
            global _embedding_model
            if _embedding_model is None:
                # First load involves disk I/O and heavy parsing — do not block event loop
                model = await asyncio.to_thread(_get_embedding_model)
            else:
                model = _embedding_model

            # encode is synchronous, run it in a thread pool
            embedding = await asyncio.to_thread(model.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            # Return zero vector as fallback — prevents crashes during indexing
            return [0.0] * settings.embedding_dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using native batch encoding in a single thread.

        SentenceTransformer.encode() supports list input for vectorized
        batch computation — significantly faster and more memory-efficient
        than spawning N separate threads (critical on t2.micro).
        """
        if not texts:
            return []

        # Filter blanks but remember positions so we can reassemble
        non_empty: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not non_empty:
            return [[0.0] * settings.embedding_dim for _ in texts]

        try:
            global _embedding_model
            if _embedding_model is None:
                await asyncio.to_thread(_get_embedding_model)
            model = _embedding_model

            batch_texts = [t for _, t in non_empty]
            # Single thread, single batched call — uses vectorized ops internally
            embeddings = await asyncio.to_thread(model.encode, batch_texts)

            # Reassemble: fill zero vectors for blank inputs
            result: list[list[float]] = [[0.0] * settings.embedding_dim for _ in range(len(texts))]
            for idx, (orig_pos, _) in enumerate(non_empty):
                result[orig_pos] = embeddings[idx].tolist()
            return result
        except Exception as e:
            logger.error("batch_embedding_failed", error=str(e), count=len(texts))
            return [[0.0] * settings.embedding_dim for _ in texts]

    # ── Info ──────────────────────────────────────────────────────────────

    def get_provider_info(self) -> dict[str, str]:
        """Return current provider and model info."""
        return {"provider": self.provider, "model": self.model}
