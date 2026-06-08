"""
Prism Dead Letter Queue API.

Provides endpoints for inspecting and replaying failed tasks
that have been routed to the DLQ after exhausting all retries.

Endpoints:
  GET  /api/v1/dlq          — List all DLQ messages (with pagination)
  POST /api/v1/dlq/replay   — Replay a specific message or all messages
  DELETE /api/v1/dlq/{index} — Remove a specific DLQ message
  GET  /api/v1/dlq/stats    — DLQ depth and failure breakdown

All Redis calls use the shared async connection pool from utils.connections
to avoid blocking the event loop (the previous sync redis calls would
starve all concurrent webhook requests).
"""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from utils.connections import get_redis
from workers.tasks import process_pr_review
from observability.logging import get_logger
from observability.metrics import dlq_depth

logger = get_logger(__name__)
router = APIRouter(prefix="/dlq", tags=["dlq"])

_DLQ_KEY = "prism:dlq:messages"


async def _get_async_redis():
    """Get the shared async Redis client, raising 503 if unavailable."""
    redis = get_redis()
    if redis is None:
        raise HTTPException(
            status_code=503,
            detail="Redis pool not initialized — DLQ unavailable",
        )
    return redis


# ── Request Models ────────────────────────────────────────────────────────


class ReplayRequest(BaseModel):
    """Request to replay DLQ messages."""

    index: int | None = Field(
        default=None,
        description="Index of a specific DLQ message to replay (0-based). If None, replays ALL messages.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("")
async def list_dlq_messages(
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Max messages to return"),
    _=Depends(require_api_key),
) -> dict[str, Any]:
    """List messages currently in the Dead Letter Queue.

    Returns the most recent failures first (LIFO order from Redis LPUSH).
    """
    r = await _get_async_redis()
    total = await r.llen(_DLQ_KEY)
    raw_messages = await r.lrange(_DLQ_KEY, offset, offset + limit - 1)

    messages = []
    for i, raw in enumerate(raw_messages):
        try:
            msg = json.loads(raw)
            # Strip large fields for the list view
            pr_data = msg.get("pr_data", {})
            msg["pr_summary"] = {
                "repo": pr_data.get("repo_name", "unknown"),
                "pr_number": pr_data.get("number", 0),
                "title": pr_data.get("title", "")[:100],
            }
            # Don't send the full diff/pr_data in list view
            msg.pop("pr_data", None)
            msg["index"] = offset + i
            messages.append(msg)
        except json.JSONDecodeError:
            messages.append({"index": offset + i, "raw": raw, "parse_error": True})

    # Update Prometheus gauge
    dlq_depth.set(total)

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "messages": messages,
    }


@router.post("/replay")
async def replay_dlq_messages(
    request: ReplayRequest,
    _=Depends(require_api_key),
) -> dict[str, Any]:
    """Replay DLQ messages by re-enqueuing them as fresh Celery tasks.

    - If `index` is provided, replays that specific message.
    - If `index` is None, replays ALL messages and flushes the DLQ.
    """
    r = await _get_async_redis()
    total = await r.llen(_DLQ_KEY)

    if total == 0:
        return {"status": "empty", "replayed": 0, "message": "DLQ is empty"}

    replayed = 0
    errors = []

    if request.index is not None:
        # ── Replay a single message ──────────────────────────────────
        if request.index < 0 or request.index >= total:
            raise HTTPException(
                status_code=400,
                detail=f"Index {request.index} out of range (0-{total - 1})",
            )

        raw = await r.lindex(_DLQ_KEY, request.index)
        if not raw:
            raise HTTPException(status_code=404, detail="Message not found")

        try:
            msg = json.loads(raw)
            pr_data = msg.get("pr_data", {})
            task = process_pr_review.delay(pr_data)
            # Remove the message from DLQ
            # Use a tombstone approach since Redis doesn't support index-based removal
            await r.lset(_DLQ_KEY, request.index, "__REPLAYED__")
            await r.lrem(_DLQ_KEY, 1, "__REPLAYED__")
            replayed = 1
            logger.info(
                "dlq_message_replayed",
                index=request.index,
                task_id=task.id,
                repo=pr_data.get("repo_name"),
                pr=pr_data.get("number"),
            )
        except Exception as e:
            errors.append({"index": request.index, "error": str(e)})
    else:
        # ── Replay ALL messages ──────────────────────────────────────
        all_messages = await r.lrange(_DLQ_KEY, 0, -1)
        for i, raw in enumerate(all_messages):
            try:
                msg = json.loads(raw)
                pr_data = msg.get("pr_data", {})
                process_pr_review.delay(pr_data)
                replayed += 1
            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        # Flush the DLQ
        await r.delete(_DLQ_KEY)
        logger.info("dlq_flushed_and_replayed", total=replayed)

    # Update depth gauge
    new_depth = await r.llen(_DLQ_KEY)
    dlq_depth.set(new_depth)

    return {
        "status": "replayed",
        "replayed": replayed,
        "errors": errors,
        "remaining_depth": new_depth,
    }


@router.delete("/{index}")
async def delete_dlq_message(
    index: int,
    _=Depends(require_api_key),
) -> dict[str, Any]:
    """Delete a specific DLQ message by index without replaying it."""
    r = await _get_async_redis()
    total = await r.llen(_DLQ_KEY)

    if index < 0 or index >= total:
        raise HTTPException(
            status_code=400,
            detail=f"Index {index} out of range (0-{total - 1})",
        )

    await r.lset(_DLQ_KEY, index, "__DELETED__")
    await r.lrem(_DLQ_KEY, 1, "__DELETED__")

    new_depth = await r.llen(_DLQ_KEY)
    dlq_depth.set(new_depth)

    logger.info("dlq_message_deleted", index=index)
    return {"status": "deleted", "remaining_depth": new_depth}


@router.get("/stats")
async def dlq_stats(_=Depends(require_api_key)) -> dict[str, Any]:
    """Get DLQ statistics — depth, failure type breakdown."""
    r = await _get_async_redis()
    total = await r.llen(_DLQ_KEY)
    all_messages = await r.lrange(_DLQ_KEY, 0, -1)

    # Update Prometheus gauge
    dlq_depth.set(total)

    # Build failure breakdown
    breakdown: dict[str, int] = {}
    repos: dict[str, int] = {}
    for raw in all_messages:
        try:
            msg = json.loads(raw)
            error = msg.get("error", "unknown")
            # Extract the error category (PERMANENT vs RETRIES_EXHAUSTED)
            category = error.split(":")[0] if ":" in error else "unknown"
            breakdown[category] = breakdown.get(category, 0) + 1

            repo = msg.get("pr_data", {}).get("repo_name", "unknown")
            repos[repo] = repos.get(repo, 0) + 1
        except json.JSONDecodeError:
            breakdown["parse_error"] = breakdown.get("parse_error", 0) + 1

    return {
        "depth": total,
        "failure_breakdown": breakdown,
        "repos": repos,
    }
