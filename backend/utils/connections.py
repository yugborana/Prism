"""
Prism Connection Pool Manager.

Centralized async connection pools for Redis and httpx, initialized once at
FastAPI startup and shared across all requests/workers.

Why: Creating a new TCP connection per request adds ~1-3ms (Redis) or
~50-100ms (httpx with TLS). Pooling amortizes this to ~0ms per request.

Usage:
    from utils.connections import get_redis, get_httpx_client

    # In any async handler:
    redis = get_redis()
    await redis.set("key", "value")

    client = get_httpx_client()
    resp = await client.get("https://api.github.com/...")
"""

from __future__ import annotations

import redis.asyncio as aioredis
import httpx

from observability.logging import get_logger

logger = get_logger(__name__)

# ── Module-level singletons ──────────────────────────────────────────────
_redis_pool: aioredis.Redis | None = None
_httpx_client: httpx.AsyncClient | None = None


# ── Redis Pool ───────────────────────────────────────────────────────────

async def init_redis_pool() -> None:
    """Initialize the shared async Redis connection pool.

    Called once during FastAPI lifespan startup. Uses ConnectionPool
    with min/max connections from settings.
    """
    global _redis_pool
    from utils.config import settings

    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=settings.redis_pool_max,
        socket_connect_timeout=3,
        socket_keepalive=True,
    )
    _redis_pool = aioredis.Redis(connection_pool=pool)

    # Verify connectivity
    try:
        await _redis_pool.ping()
        logger.info(
            "redis_pool_initialized",
            url=settings.redis_url.split("@")[-1],
            max_connections=settings.redis_pool_max,
        )
    except Exception as e:
        logger.warning("redis_pool_ping_failed", error=str(e))


async def close_redis_pool() -> None:
    """Gracefully close the Redis connection pool on shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("redis_pool_closed")


def get_redis() -> aioredis.Redis | None:
    """Get the shared Redis client.

    If the pool hasn't been initialized (e.g., Celery worker context where
    FastAPI lifespan doesn't run), lazily creates the pool on first call.
    Returns None only if initialization fails.
    """
    global _redis_pool
    if _redis_pool is not None:
        return _redis_pool

    # Lazy init for non-FastAPI contexts (Celery workers)
    try:
        from utils.config import settings
        pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=settings.redis_pool_max,
            socket_connect_timeout=3,
            socket_keepalive=True,
        )
        _redis_pool = aioredis.Redis(connection_pool=pool)
        logger.info(
            "redis_pool_lazy_initialized",
            url=settings.redis_url.split("@")[-1],
            max_connections=settings.redis_pool_max,
        )
        return _redis_pool
    except Exception as e:
        logger.warning("redis_pool_lazy_init_failed", error=str(e))
        return None


# ── httpx Client ─────────────────────────────────────────────────────────

async def init_httpx_client() -> None:
    """Initialize the shared httpx.AsyncClient with connection pooling.

    Keeps TCP+TLS connections alive across requests to the same hosts
    (e.g., api.github.com), eliminating per-request handshake overhead.
    """
    global _httpx_client
    from utils.config import settings

    _httpx_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.httpx_max_connections,
            max_keepalive_connections=settings.httpx_max_keepalive,
        ),
        timeout=httpx.Timeout(60.0, connect=10.0),
        http2=True,
        follow_redirects=True,
    )
    logger.info(
        "httpx_client_initialized",
        max_connections=settings.httpx_max_connections,
        max_keepalive=settings.httpx_max_keepalive,
    )


async def close_httpx_client() -> None:
    """Gracefully close the httpx client on shutdown."""
    global _httpx_client
    if _httpx_client is not None:
        await _httpx_client.aclose()
        _httpx_client = None
        logger.info("httpx_client_closed")


# Cached fallback for non-FastAPI contexts (Celery workers)
_httpx_fallback: httpx.AsyncClient | None = None


def get_httpx_client() -> httpx.AsyncClient:
    """Get the shared httpx client.

    Falls back to a cached persistent client if pool isn't initialized
    (e.g., during Celery worker startup where FastAPI lifespan doesn't run).
    The fallback is created once and reused, preventing TCP connection leaks.
    """
    if _httpx_client is not None:
        return _httpx_client

    global _httpx_fallback
    if _httpx_fallback is not None:
        return _httpx_fallback

    # Create a persistent fallback for Celery workers (created once, reused)
    from utils.config import settings
    _httpx_fallback = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.httpx_max_connections,
            max_keepalive_connections=settings.httpx_max_keepalive,
        ),
        timeout=httpx.Timeout(60.0, connect=10.0),
        http2=True,
        follow_redirects=True,
    )
    logger.info("httpx_client_fallback_cached")
    return _httpx_fallback
