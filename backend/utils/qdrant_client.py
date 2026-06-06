"""
Prism Qdrant Client with Graceful Degradation.

Pattern: Multi-Agent observability/metrics.py (_NoOpMetric fallback)

When Qdrant is unavailable (e.g., local dev without Docker),
the system returns a NoOp client that silently succeeds.
This prevents the app from crashing at startup.
"""

from typing import Any

from observability.logging import get_logger
from utils.config import settings

logger = get_logger(__name__)


class _NoOpQdrantClient:
    """
    Fallback Qdrant client that does nothing.
    Used when Qdrant is unavailable (local dev, CI tests).
    All methods return empty/success values.
    """

    def __getattr__(self, name: str) -> Any:
        """Return a no-op callable for any method."""
        def noop(*args: Any, **kwargs: Any) -> Any:
            logger.debug("qdrant_noop_called", method=name)
            # Return sensible defaults for common methods
            if name == "get_collections":
                return _FakeCollections()
            if name == "search":
                return []
            if name == "scroll":
                return ([], None)
            if name == "upsert":
                return None
            if name == "create_collection":
                return None
            if name == "create_payload_index":
                return None
            return None
        return noop


class _FakeCollections:
    """Fake response for get_collections() in NoOp mode."""
    def __init__(self) -> None:
        self.collections: list[Any] = []


def _create_qdrant_client() -> Any:
    """
    Create a Qdrant client, falling back to NoOp if unavailable.
    This is the graceful degradation pattern from Multi-Agent observability.
    """
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
            timeout=10,
        )
        # Test connection
        client.get_collections()
        logger.info(
            "qdrant_connected",
            url=settings.qdrant_url,
        )
        return client

    except ImportError:
        logger.warning("qdrant_sdk_missing", msg="qdrant-client not installed, using NoOp")
        return _NoOpQdrantClient()

    except Exception as e:
        logger.warning(
            "qdrant_connection_failed",
            url=settings.qdrant_url,
            error=str(e),
            msg="Falling back to NoOp client — vector features disabled",
        )
        return _NoOpQdrantClient()


# Singleton — imported by db/index.py, db/vector_indexer.py, etc.
qdrant_client = _create_qdrant_client()


def get_qdrant_client() -> Any:
    """Return the singleton Qdrant client (real or NoOp)."""
    return qdrant_client
