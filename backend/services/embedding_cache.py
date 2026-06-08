"""
Prism Embedding Cache — Content-Addressed Vector Cache in Redis.

Inspired by Cursor's indexing approach: embeddings are cached by the
SHA-256 hash of the chunk content, not by file path. If a chunk's content
didn't change between two indexing runs, its embedding is identical —
skip the embedding API call entirely.

Impact:
  - First full index (2,000 chunks): 2,000 API calls (~25s)
  - Incremental update (15 chunks changed): 15 API calls (~0.2s)
  - Fork with 95% overlap: 100 API calls (~1.2s)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from observability.logging import get_logger
from utils.config import settings
from utils.connections import get_redis

logger = get_logger(__name__)

# Default TTL: 7 days
DEFAULT_TTL = 604800


class EmbeddingCache:
    """Redis-backed embedding cache keyed by content SHA-256.

    Cache key format: ``prism:embed:{sha256_hex}``
    Cache value: JSON-encoded list[float]
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL):
        self._ttl = ttl_seconds
        self._redis: Any | None = None
        # Stats for observability
        self._hits = 0
        self._misses = 0

    async def _ensure_redis(self) -> Any:
        """Get the shared async Redis connection pool."""
        if self._redis is None:
            self._redis = get_redis()
            if self._redis is None:
                logger.warning("embedding_cache_redis_pool_not_initialized")
        return self._redis

    async def get_or_embed(
        self,
        content_hashes: list[str],
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        texts: list[str],
    ) -> list[list[float]]:
        """For each content hash, return its embedding — from cache or freshly generated.

        Args:
            content_hashes: SHA-256 hex hashes for each text (same length as texts).
            embed_fn: Async function that takes a list of texts and returns embeddings.
            texts: The actual text to embed (only used for cache misses).

        Returns:
            list of embedding vectors, same length and order as inputs.
        """
        n = len(content_hashes)
        results: list[list[float] | None] = [None] * n
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        redis = await self._ensure_redis()

        # Phase 1: Check cache for each hash
        if redis:
            try:
                pipe = redis.pipeline(transaction=False)
                for h in content_hashes:
                    pipe.get(self._cache_key(h))
                cached_values = await pipe.execute()

                from observability.metrics import index_chunks_embedded

                for i, cached in enumerate(cached_values):
                    if cached is not None:
                        results[i] = json.loads(cached)
                        self._hits += 1
                        try:
                            index_chunks_embedded.labels(cache_status="hit").inc()
                        except Exception:
                            pass
                    else:
                        miss_indices.append(i)
                        miss_texts.append(texts[i])
                        self._misses += 1
                        try:
                            index_chunks_embedded.labels(cache_status="miss").inc()
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("embedding_cache_read_failed", error=str(e))
                # Fall through to embed everything
                miss_indices = list(range(n))
                miss_texts = list(texts)
                self._misses += n
        else:
            miss_indices = list(range(n))
            miss_texts = list(texts)
            self._misses += n

        # Phase 2: Embed cache misses in batch
        if miss_texts:
            try:
                new_embeddings = await embed_fn(miss_texts)
            except Exception as e:
                logger.error("embedding_batch_failed", error=str(e), count=len(miss_texts))
                # Return zero vectors for failures
                dim = settings.embedding_dim
                new_embeddings = [[0.0] * dim] * len(miss_texts)

            # Store in results and cache
            for idx, embedding in zip(miss_indices, new_embeddings):
                results[idx] = embedding

            # Phase 3: Cache the new embeddings
            if redis:
                try:
                    pipe = redis.pipeline(transaction=False)
                    for idx, embedding in zip(miss_indices, new_embeddings):
                        key = self._cache_key(content_hashes[idx])
                        pipe.setex(key, self._ttl, json.dumps(embedding))
                    await pipe.execute()
                except Exception as e:
                    logger.warning("embedding_cache_write_failed", error=str(e))

        if self._hits + self._misses > 0 and (self._hits + self._misses) % 100 == 0:
            logger.info(
                "embedding_cache_stats",
                hits=self._hits,
                misses=self._misses,
                hit_rate=f"{self._hits / (self._hits + self._misses):.1%}",
            )

        # All results should be filled now
        return [r if r is not None else [0.0] * settings.embedding_dim for r in results]

    async def get_cached(self, content_hash: str) -> list[float] | None:
        """Retrieve a single cached embedding vector."""
        redis = await self._ensure_redis()
        if not redis:
            return None
        try:
            cached = await redis.get(self._cache_key(content_hash))
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        return None

    async def store(self, content_hash: str, vector: list[float]) -> None:
        """Cache a single embedding vector."""
        redis = await self._ensure_redis()
        if not redis:
            return
        try:
            await redis.setex(
                self._cache_key(content_hash),
                self._ttl,
                json.dumps(vector),
            )
        except Exception as e:
            logger.warning("embedding_cache_store_failed", error=str(e))

    def stats(self) -> dict[str, Any]:
        """Return cache performance stats."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "N/A",
            "total": total,
        }

    @staticmethod
    def _cache_key(content_hash: str) -> str:
        return f"prism:embed:{content_hash}"

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
