"""
Prism Multi-Provider LLM Client Factory.

Supports: Gemini, OpenAI, Anthropic, Groq, Bedrock
All calls are async. Synchronous SDKs are wrapped with asyncio.to_thread().

Usage:
    from utils.llm_factory import LLMClient
    client = LLMClient()                      # Uses default provider from config
    client = LLMClient(provider="openai")      # Override provider
    response = await client.generate("Analyze this code...")
    embedding = await client.embed("some text")
"""

import asyncio
from typing import Any

from observability.logging import get_logger
from utils.config import settings

logger = get_logger(__name__)


class LLMClient:
    """
    Unified async LLM client with multi-provider support.

    Each provider's SDK is lazily imported only when first used,
    so missing SDKs don't crash the app at import time.
    """

    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.llm_provider
        self.model = settings.get_model_for_provider(self.provider)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the provider-specific client."""
        if self._client is not None:
            return self._client

        api_key = settings.get_api_key_for_provider(self.provider)

        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=api_key)
        elif self.provider == "bedrock":
            import boto3
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=settings.aws_region,
            )
        elif self.provider == "ollama":
            from openai import OpenAI
            self._client = OpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",  # Ollama doesn't need a real key
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        logger.info(
            "llm_client_initialized",
            provider=self.provider,
            model=self.model,
        )
        return self._client

    # ── Text Generation ───────────────────────────────────────────────────

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate text from the configured LLM provider.
        Returns the response text content.
        """
        from observability.metrics import track_llm_call

        client = self._get_client()

        with track_llm_call(agent="LLMClient", model=self.model):
            try:
                if self.provider == "gemini":
                    return await self._generate_gemini(client, prompt, system_prompt)
                elif self.provider == "openai":
                    return await self._generate_openai(client, prompt, system_prompt)
                elif self.provider == "anthropic":
                    return await self._generate_anthropic(client, prompt, system_prompt)
                elif self.provider == "groq":
                    return await self._generate_groq(client, prompt, system_prompt)
                elif self.provider == "bedrock":
                    return await self._generate_bedrock(client, prompt, system_prompt)
                elif self.provider == "ollama":
                    return await self._generate_ollama(client, prompt, system_prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                logger.error(
                    "llm_generation_failed",
                    provider=self.provider,
                    model=self.model,
                    error=str(e),
                )
                raise

    async def _generate_gemini(self, client: Any, prompt: str, system_prompt: str) -> str:
        model = client.GenerativeModel(
            self.model,
            system_instruction=system_prompt if system_prompt else None,
        )
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text

    async def _generate_openai(self, client: Any, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    async def _generate_anthropic(self, client: Any, prompt: str, system_prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await asyncio.to_thread(client.messages.create, **kwargs)
        return response.content[0].text

    async def _generate_groq(self, client: Any, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    async def _generate_bedrock(self, client: Any, prompt: str, system_prompt: str) -> str:
        import json

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": full_prompt}],
        })

        response = await asyncio.to_thread(
            client.invoke_model,
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    async def _generate_ollama(self, client: Any, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Explicitly configure Ollama context window size to 8192 via extra_body
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=self.model,
            messages=messages,
            extra_body={
                "options": {
                    "num_ctx": 8192
                }
            }
        )
        return response.choices[0].message.content

    # ── Embeddings ────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.
        Uses Ollama's all-minilm model via its OpenAI-compatible API,
        keeping embeddings consistent across the vector DB.
        """
        if not text or not text.strip():
            return [0.0] * settings.embedding_dim

        try:
            from openai import OpenAI
            embed_client = OpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",
            )

            result = await asyncio.to_thread(
                embed_client.embeddings.create,
                model=settings.embedding_model,
                input=text,
            )
            return result.data[0].embedding
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            # Return zero vector as fallback — prevents crashes during indexing
            return [0.0] * settings.embedding_dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts concurrently."""
        tasks = [self.embed(t) for t in texts]
        return await asyncio.gather(*tasks)

    # ── Info ──────────────────────────────────────────────────────────────

    def get_provider_info(self) -> dict[str, str]:
        """Return current provider and model info."""
        return {"provider": self.provider, "model": self.model}
