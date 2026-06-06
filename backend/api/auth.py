"""
Prism API Authentication — Shared dependency for internal endpoints.

When PRISM_API_KEY is set in config, all protected endpoints require the
caller to pass the key via the X-API-Key header. When the key is empty
(local dev), authentication is skipped.
"""

from fastapi import Header, HTTPException

from utils.config import settings
from observability.logging import get_logger

logger = get_logger(__name__)


async def require_api_key(x_api_key: str = Header(default=None)):
    """
    FastAPI dependency — validates X-API-Key header.

    - If PRISM_API_KEY is not set (empty): auth is skipped (dev mode).
    - If PRISM_API_KEY is set: the header must match exactly.
    """
    if not settings.prism_api_key:
        # No key configured — allow all requests (development)
        return

    if not x_api_key or x_api_key != settings.prism_api_key:
        logger.warning("unauthorized_api_request", has_key=bool(x_api_key))
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
