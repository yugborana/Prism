"""
Prism Background Tasks.

This file defines the Celery tasks that run in the worker processes.
The primary task 'process_pr_review' orchestrates the entire agentic pipeline.

Retry & DLQ Strategy:
  - Transient errors (network, LLM timeout, DB down) → retry up to 3 times
    with exponential backoff: 30s → 60s → 120s
  - Permanent errors (bad JSON, missing key, schema violation) → skip retry,
    route directly to DLQ
  - After 3 retries exhausted → route to prism.dlq for manual replay

References:
  - https://docs.celeryq.dev/en/stable/userguide/configuration.html#task-reject-on-worker-lost
  - https://inventivehq.com/blog/webhook-best-practices-guide
"""

import asyncio
import time

import httpx
import uuid
import json
from datetime import datetime, timezone
from typing import Any

import redis as sync_redis
from opentelemetry import context as otel_context
from pydantic import ValidationError

from workers.celery_app import celery_app
from orchestrator.engine import ReviewOrchestrator
from services.github_service import GitHubService
from utils.config import settings
from observability.logging import get_logger, bind_correlation_id, clear_correlation_context
from observability.metrics import dlq_tasks_total, dlq_depth, review_e2e_seconds, celery_task_retries_total
from observability.tracing import get_tracer, extract_trace_context

logger = get_logger(__name__)
tracer = get_tracer(__name__)

# Exponential backoff schedule: 30s, 60s, 120s
_RETRY_COUNTDOWNS = [30, 60, 120]
_MAX_RETRIES = 3

# Permanent exceptions that should never be retried — bad input, not transient
_PERMANENT_EXCEPTIONS = (
    ValidationError,
    json.JSONDecodeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)


def _send_to_dlq(task_name: str, pr_data: dict, error: str, retries: int):
    """Route a failed task payload to the Dead Letter Queue.

    The DLQ message preserves the original pr_data plus failure metadata
    so it can be inspected and replayed later via /api/v1/dlq/replay.
    """
    dlq_payload = {
        "original_task": task_name,
        "pr_data": pr_data,
        "error": error,
        "retries_exhausted": retries,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }

    celery_app.send_task(
        "prism.dlq.store",
        args=[dlq_payload],
        queue="prism.dlq",
        routing_key="prism.dlq",
    )

    dlq_tasks_total.inc()
    logger.error(
        "task_routed_to_dlq",
        task=task_name,
        retries=retries,
        error=error[:200],
        repo=pr_data.get("repo_name", "unknown"),
        pr=pr_data.get("number", 0),
    )


@celery_app.task(
    name="process_pr_review",
    bind=True,
    max_retries=_MAX_RETRIES,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_pr_review(self, pr_data: dict[str, Any]):
    """
    Background task to run the full agentic review pipeline.

    Args:
        pr_data: Dictionary containing PR details (diff, title, etc)
    """
    review_id = str(uuid.uuid4())
    e2e_start = time.monotonic()

    # ── Bind correlation_id to all logs in this worker process ─────
    # This is the same GitHub delivery ID that was bound in the webhook.
    # Because Celery runs in a separate process, we re-bind it here
    # from the serialized pr_data (contextvars don't cross processes).
    correlation_id = pr_data.get("correlation_id", review_id)
    bind_correlation_id(correlation_id)

    # ── Restore OTel trace context ────────────────────────────────
    # This makes the worker span a child of the webhook span in the
    # distributed trace, even though they run in different processes.
    trace_ctx = pr_data.get("trace_context", {})
    restored_ctx = extract_trace_context(trace_ctx) if trace_ctx else None
    token = otel_context.attach(restored_ctx) if restored_ctx else None

    logger.info(
        "celery_task_started",
        task="process_pr_review",
        review_id=review_id,
        retry=self.request.retries,
    )

    # Celery tasks are synchronous by default, so we run the async orchestrator
    # using a dedicated event loop. Always create a new one — Celery worker
    # processes don't have a running asyncio loop.
    # IMPORTANT: Use a SINGLE loop for the entire task lifetime. The shared
    # httpx client binds to the first loop it sees; creating/closing loops
    # per coroutine causes "Event loop is closed" on subsequent calls.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _run_async(coro):
        """Run an async coroutine on the task's shared event loop."""
        return loop.run_until_complete(coro)

    with tracer.start_as_current_span(
        "prism.review.process",
        attributes={
            "correlation_id": correlation_id,
            "review.id": review_id,
            "github.repo": pr_data.get("repo_name", ""),
            "github.pr": pr_data.get("number", 0),
            "celery.retry": self.request.retries,
        },
    ):
        try:
            # ── Fetch diff + files if not provided ────────────────────────
            # The webhook endpoint enqueues immediately without fetching
            # (to stay within GitHub's 10s timeout). The /test-review
            # endpoint may pre-fill these fields for convenience.
            if "diff" not in pr_data or "changed_files" not in pr_data or not pr_data.get("head_sha"):
                gh = GitHubService(installation_id=pr_data.get("installation_id"))
                repo_name = pr_data["repo_name"]
                pr_number = pr_data["number"]

                # Fetch head_sha from PR details if not provided
                # (comment trigger path doesn't have it in the webhook payload)
                if not pr_data.get("head_sha"):
                    try:
                        pr_details = _run_async(gh.fetch_pr_details(repo_name, pr_number))
                        pr_data["head_sha"] = pr_details.get("head", {}).get("sha", "")
                    except Exception as sha_err:
                        logger.warning("head_sha_fetch_failed", error=str(sha_err))
                        pr_data["head_sha"] = ""

                if "diff" not in pr_data:
                    pr_data["diff"] = _run_async(gh.fetch_pr_diff(repo_name, pr_number))
                if "changed_files" not in pr_data:
                    pr_data["changed_files"] = _run_async(gh.fetch_pr_files(repo_name, pr_number))
                logger.info(
                    "worker_fetched_pr_data",
                    review_id=review_id,
                    diff_len=len(pr_data["diff"]),
                    files=len(pr_data["changed_files"]),
                    has_head_sha=bool(pr_data.get("head_sha")),
                )

            # ── Non-blocking repo index check (Cursor pattern) ──────────
            # First PR for a repo: agents review with diff-only context
            # while a background task builds the full index. Second PR
            # gets cross-file context from the repo_chunks collection.
            try:
                from services.simhash import SimHashIndex

                simhash_idx = SimHashIndex()
                index_status = _run_async(simhash_idx.check_index_status(pr_data.get("repo_name", "")))
                _run_async(simhash_idx.close())

                if index_status == "fresh":
                    pr_data["has_repo_index"] = True
                elif index_status == "stale":
                    pr_data["has_repo_index"] = True
                    celery_app.send_task(
                        "refresh_repo_index",
                        args=[
                            pr_data.get("repo_name", ""),
                            pr_data.get("base_branch", "main"),
                            pr_data.get("installation_id", 0),
                        ],
                        queue="prism.index",
                    )
                else:
                    pr_data["has_repo_index"] = False
                    celery_app.send_task(
                        "build_repo_index",
                        args=[
                            pr_data.get("repo_name", ""),
                            pr_data.get("base_branch", "main"),
                            pr_data.get("installation_id", 0),
                        ],
                        queue="prism.index",
                    )
                logger.info(
                    "repo_index_status",
                    repo=pr_data.get("repo_name", ""),
                    status=index_status,
                )
            except Exception as idx_err:
                logger.warning("repo_index_check_failed", error=str(idx_err))
                pr_data["has_repo_index"] = False

            orchestrator = ReviewOrchestrator(review_id=review_id)
            # Run the full review
            result_state = _run_async(orchestrator.run_review(pr_data))

            # Post the final aggregated review back to GitHub
            if result_state.final_review:
                try:
                    github_service = GitHubService(installation_id=pr_data.get("installation_id"))
                    logger.info(
                        "posting_review_to_github",
                        review_id=review_id,
                        repo=result_state.repo_full_name,
                        pr=result_state.pr_number,
                    )
                    _run_async(
                        github_service.post_review(
                            repo_full_name=result_state.repo_full_name,
                            pr_number=result_state.pr_number,
                            review_data=result_state.final_review.model_dump(),
                        )
                    )
                    logger.info("review_posted_to_github_successfully", review_id=review_id)
                except Exception as post_err:
                    logger.error("github_post_review_failed", review_id=review_id, error=str(post_err))

            else:
                logger.warning("no_final_review_available_to_post", review_id=review_id)

            logger.info(
                "celery_task_finished",
                review_id=review_id,
                issues=result_state.final_review.total_issues if result_state.final_review else 0,
            )

            return {
                "review_id": review_id,
                "status": "success",
                "total_issues": result_state.final_review.total_issues if result_state.final_review else 0,
            }

        except Exception as exc:
            # ── Permanent errors: skip retry, route to DLQ immediately ────
            # Also treat HTTP 401/403 as permanent (bad credentials, not transient)
            is_permanent = isinstance(exc, _PERMANENT_EXCEPTIONS)
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403, 404, 422):
                is_permanent = True

            if is_permanent:
                logger.error(
                    "celery_task_failed_permanent",
                    review_id=review_id,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
                _send_to_dlq(
                    task_name="process_pr_review",
                    pr_data=pr_data,
                    error=f"PERMANENT: {exc.__class__.__name__}: {str(exc)}",
                    retries=self.request.retries,
                )
                return {
                    "review_id": review_id,
                    "status": "failed_permanent",
                    "error": f"{exc.__class__.__name__}: {str(exc)}",
                }

            # ── Transient errors: retry with exponential backoff ──────────
            retries = self.request.retries
            logger.error(
                "celery_task_failed_transient",
                review_id=review_id,
                error_type=exc.__class__.__name__,
                error=str(exc),
                retries=retries,
                max_retries=_MAX_RETRIES,
            )

            if retries < _MAX_RETRIES:
                countdown = _RETRY_COUNTDOWNS[retries]
                logger.info(
                    "celery_task_retrying",
                    review_id=review_id,
                    retry_number=retries + 1,
                    countdown_seconds=countdown,
                )
                celery_task_retries_total.labels(
                    task="process_pr_review",
                    retry_number=str(retries + 1),
                ).inc()
                raise self.retry(exc=exc, countdown=countdown)
            else:
                # All retries exhausted — route to DLQ
                _send_to_dlq(
                    task_name="process_pr_review",
                    pr_data=pr_data,
                    error=f"RETRIES_EXHAUSTED: {exc.__class__.__name__}: {str(exc)}",
                    retries=retries,
                )
                return {
                    "review_id": review_id,
                    "status": "failed_dlq",
                    "error": f"Moved to DLQ after {retries} retries: {str(exc)}",
                }

        finally:
            # ── Record end-to-end latency ─────────────────────────────────
            e2e_duration = time.monotonic() - e2e_start
            review_e2e_seconds.labels(repo=pr_data.get("repo_name", "unknown")).observe(e2e_duration)

            # ── Cleanup ───────────────────────────────────────────────────
            if token is not None:
                otel_context.detach(token)
            clear_correlation_context()
            loop.close()


# ── DLQ Storage Task ──────────────────────────────────────────────────────
# This task runs on the prism.dlq queue. It simply stores the failed payload
# in Redis as a list so it can be inspected/replayed via the API.


@celery_app.task(
    name="prism.dlq.store",
    bind=True,
    queue="prism.dlq",
    max_retries=0,
)
def dlq_store(self, dlq_payload: dict[str, Any]):
    """Store a failed task payload in the DLQ Redis list for later replay."""
    # Bind correlation_id from the original failed task so DLQ logs are traceable
    original_pr = dlq_payload.get("pr_data", {})
    cid = original_pr.get("correlation_id", "")
    if cid:
        bind_correlation_id(cid)

    r = sync_redis.from_url(settings.redis_url, decode_responses=True)
    try:
        r.lpush("prism:dlq:messages", json.dumps(dlq_payload))

        # Update the DLQ depth gauge proactively so alerts
        # fire immediately when tasks enter the DLQ (not just when the
        # DLQ API is polled by a human).
        new_depth = r.llen("prism:dlq:messages")
        dlq_depth.set(new_depth)

        logger.info(
            "dlq_message_stored",
            task=dlq_payload.get("original_task"),
            error=dlq_payload.get("error", "")[:100],
            dlq_depth=new_depth,
        )
    finally:
        r.close()

    if cid:
        clear_correlation_context()


@celery_app.task(name="update_dlq_depth", max_retries=0)
def update_dlq_depth():
    """Periodic task to sync the DLQ depth gauge with Redis.

    Ensures the Prometheus gauge stays accurate even when no new tasks
    are entering the DLQ (e.g., after messages are replayed via the API).
    """
    r = sync_redis.from_url(settings.redis_url, decode_responses=True)
    try:
        depth = r.llen("prism:dlq:messages")
        dlq_depth.set(depth)
    finally:
        r.close()
