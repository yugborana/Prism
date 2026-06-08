"""
Prism Review Memory — Shared Session State for a Review Pipeline.

Source: Proximus orchestrator/memory/project_memory.py (adapted)

Provides a multi-layer memory system for a single PR review session.
All agents (Security, Quality, Performance) read/write to this shared store,
enabling cross-agent awareness (e.g., Performance agent can see Security findings).

Layers:
  - L1 (Hot):  In-process dict — sub-millisecond access during a review
  - L2 (Warm): Redis — survives worker restarts, shared across Celery workers
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)


class ReviewMemory:
    """
    Session-scoped memory for a single PR review.
    
    Unlike ProjectMemory (which tracks an entire software project lifecycle),
    ReviewMemory is scoped to one PR review session and expires after completion.
    """

    def __init__(self, review_id: str, redis_client=None):
        self.review_id = review_id
        self._redis = redis_client
        self._hot: dict[str, Any] = {}
        self._created_at = datetime.now(UTC)

        logger.info("review_memory_initialized", review_id=review_id)

    # ── Core CRUD ─────────────────────────────────────────────────────

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Write to both L1 (hot cache) and L2 (Redis)."""
        self._hot[key] = value

        if self._redis:
            try:
                redis_key = f"prism:review:{self.review_id}:{key}"
                serialized = json.dumps(value, default=str)
                await self._redis.setex(redis_key, ttl, serialized)
            except Exception as e:
                logger.debug("redis_set_failed", key=key, error=str(e))

    async def get(self, key: str, default: Any = None) -> Any:
        """Read from fastest available layer."""
        # L1 hot cache
        if key in self._hot:
            return self._hot[key]

        # L2 Redis
        if self._redis:
            try:
                redis_key = f"prism:review:{self.review_id}:{key}"
                raw = await self._redis.get(redis_key)
                if raw:
                    val = json.loads(raw)
                    self._hot[key] = val  # Promote to L1
                    return val
            except Exception:
                pass

        return default

    async def delete(self, key: str) -> None:
        self._hot.pop(key, None)
        if self._redis:
            try:
                await self._redis.delete(f"prism:review:{self.review_id}:{key}")
            except Exception:
                pass

    # ── Agent State Tracking ──────────────────────────────────────────

    async def set_agent_state(self, agent_role: str, state: dict[str, Any]) -> None:
        """Track what an individual agent has produced so far."""
        state["updated_at"] = datetime.now(UTC).isoformat()
        await self.set(f"agent:{agent_role}", state)

    async def get_agent_state(self, agent_role: str) -> dict[str, Any]:
        return await self.get(f"agent:{agent_role}", default={})

    # ── Cross-Agent Context ───────────────────────────────────────────

    async def share_finding(self, agent_role: str, finding: dict[str, Any]) -> None:
        """
        Allow agents to share findings with each other.
        
        Example: Security agent finds an auth bypass; the Quality agent
        can check if the same code path has other issues.
        """
        findings_key = "shared_findings"
        
        # Thread/Async-safe in-memory append
        current = self._hot.setdefault(findings_key, [])
        current.append({
            "agent": agent_role,
            "finding": finding,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        
        # Best-effort sync to L2 (Redis) if available
        if self._redis:
            try:
                redis_key = f"prism:review:{self.review_id}:{findings_key}"
                serialized = json.dumps(current, default=str)
                await self._redis.setex(redis_key, 3600, serialized)
            except Exception as e:
                logger.debug("redis_set_failed", key=findings_key, error=str(e))

    async def get_shared_findings(self) -> list[dict[str, Any]]:
        """Get all findings shared by other agents."""
        return await self.get("shared_findings", default=[])

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return the full in-memory state for debugging/logging."""
        return {
            "review_id": self.review_id,
            "created_at": self._created_at.isoformat(),
            "keys": list(self._hot.keys()),
            "hot_cache_size": len(self._hot),
        }

    # ── Cleanup ───────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """
        Expire all Redis keys for this review.
        Called after the review is posted to GitHub.
        """
        if self._redis:
            try:
                # Scan and delete all keys for this review
                pattern = f"prism:review:{self.review_id}:*"
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug("redis_cleanup_failed", error=str(e))

        self._hot.clear()
        logger.info("review_memory_cleaned", review_id=self.review_id)
