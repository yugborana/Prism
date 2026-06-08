"""
Prism GitHub Webhook Router.

Receives events from GitHub:
  - pull_request (opened/synchronized) → automatic review
  - issue_comment (created) → on-demand review via "/prism-review" comment

Security:
  1. HMAC-SHA256 signature verification (X-Hub-Signature-256)
  2. Idempotency via Redis — X-GitHub-Delivery IDs are stored with 24h TTL
     to prevent duplicate reviews from redeliveries.
  3. Returns 200 OK immediately after Celery enqueue — GitHub times out
     after 10 seconds, so we never block on diff fetching.

References:
  - https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
  - https://hookdeck.com/blog/webhooks-at-scale
"""

import hmac
import hashlib
import uuid

from fastapi import APIRouter, Request, Header, HTTPException, Depends
from workers.tasks import process_pr_review
from api.auth import require_api_key
from utils.config import settings
from observability.logging import get_logger, bind_correlation_id
from observability.metrics import webhook_requests_total
from observability.tracing import get_tracer, inject_trace_context

logger = get_logger(__name__)
tracer = get_tracer(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Delivery dedup TTL: 24 hours (in seconds)
_DELIVERY_TTL = 86400


# ── Helpers ──────────────────────────────────────────────────────────────


def _verify_signature(payload: bytes, signature_header: str | None) -> bool:
    """Verify that the webhook payload was signed by GitHub.

    Follows GitHub's recommended HMAC-SHA256 verification:
    1. Reject if no signature header is present.
    2. Reject if no webhook secret is configured (fail-closed in prod).
    3. Compute HMAC-SHA256 of the raw body and compare in constant time.
    """
    if not settings.github_webhook_secret:
        # Fail-closed: no secret configured → reject all webhooks.
        # In development, set GITHUB_WEBHOOK_SECRET to a test value.
        logger.error("webhook_secret_not_configured")
        return False

    if not signature_header:
        return False

    # Header format: "sha256=<hex>"
    try:
        sha_name, signature_hex = signature_header.split("=", 1)
    except ValueError:
        return False

    if sha_name != "sha256":
        return False

    expected = hmac.new(
        settings.github_webhook_secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature_hex)


async def _is_duplicate_delivery(delivery_id: str | None) -> bool:
    """Check if this X-GitHub-Delivery ID has already been processed.

    Uses Redis SETNX with a 24h TTL via the shared connection pool.
    If the key already exists, it means we've seen this delivery → duplicate.
    """
    if not delivery_id:
        # No delivery ID → can't dedup, allow through
        return False

    from utils.connections import get_redis

    redis = get_redis()
    if redis is None:
        # Pool not initialized → skip dedup, allow through (best-effort)
        return False

    try:
        key = f"prism:webhook:delivery:{delivery_id}"
        # SET NX (only if not exists) with 24h expiry
        was_set = await redis.set(key, "1", nx=True, ex=_DELIVERY_TTL)
        if not was_set:
            # Key already existed → duplicate delivery
            logger.warning("duplicate_delivery_rejected", delivery_id=delivery_id)
            return True
        return False
    except Exception as e:
        logger.warning("delivery_dedup_failed", error=str(e))
        return False


# ── Webhook Endpoint ─────────────────────────────────────────────────────


@router.post("/github")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None),
    x_github_delivery: str = Header(None),
):
    """Handle GitHub Webhook events.

    Flow:
    1. Verify HMAC-SHA256 signature (reject spoofed requests)
    2. Check idempotency (reject redeliveries)
    3. Extract minimal PR metadata from the payload
    4. Enqueue Celery task immediately (diff fetching happens in the worker)
    5. Return 200 OK within milliseconds (GitHub 10s timeout)
    """
    body = await request.body()

    # ── Step 0: Bind correlation_id to all logs in this request ───────
    # The GitHub delivery ID becomes the correlation_id that flows through
    # every log line: webhook → Celery task → agents → GitHub comment.
    correlation_id = x_github_delivery or str(uuid.uuid4())
    bind_correlation_id(correlation_id)

    with tracer.start_as_current_span(
        "prism.webhook.receive",
        attributes={
            "correlation_id": correlation_id,
            "github.event": x_github_event or "unknown",
        },
    ):
        # ── Step 1: Signature Verification ────────────────────────────────
        if not _verify_signature(body, x_hub_signature_256):
            logger.warning(
                "invalid_webhook_signature",
                delivery_id=x_github_delivery,
                github_event=x_github_event,
            )
            webhook_requests_total.labels(event_type=x_github_event or "unknown", status="rejected").inc()
            raise HTTPException(status_code=401, detail="Invalid signature")

        payload = await request.json()
        action = payload.get("action")

        # ── Route: PR Comment trigger ("/prism-review") ───────────────────
        # When a user comments "/prism-review" on any PR, GitHub sends an
        # issue_comment event. We extract the PR number from the issue URL
        # and enqueue a review — same pipeline as the automatic trigger.
        if x_github_event == "issue_comment" and action == "created":
            return await _handle_comment_trigger(payload, correlation_id, x_github_delivery)

        # ── Route: Pull Request events (opened / synchronized) ────────────
        if x_github_event != "pull_request" or action not in ("opened", "synchronize"):
            webhook_requests_total.labels(event_type=x_github_event or "unknown", status="ignored").inc()
            return {"status": "ignored", "event": x_github_event, "action": action}

        # ── Step 3: Idempotency Check ─────────────────────────────────────
        if await _is_duplicate_delivery(x_github_delivery):
            webhook_requests_total.labels(event_type="pull_request", status="duplicate").inc()
            return {"status": "duplicate", "delivery_id": x_github_delivery}

        # ── Step 4: Extract PR metadata (no API calls here!) ──────────────
        pr = payload["pull_request"]
        repo = payload["repository"]
        pr_number = pr["number"]
        repo_name = repo["full_name"]

        logger.info(
            "pr_event_accepted",
            action=action,
            repo=repo_name,
            pr=pr_number,
            delivery_id=x_github_delivery,
        )
        webhook_requests_total.labels(event_type="pull_request", status="accepted").inc()

        # ── Step 5: Enqueue to Celery immediately ─────────────────────────
        # Pass only the metadata — the Celery worker will fetch the diff
        # and file list itself. This keeps the webhook response time < 1s.
        pr_data = {
            "number": pr_number,
            "repo_name": repo_name,
            "title": pr.get("title", ""),
            "body": pr.get("body", ""),
            "installation_id": payload.get("installation", {}).get("id"),
            "base_branch": pr.get("base", {}).get("ref", "main"),
            # Correlation ID for log + trace context across processes
            "correlation_id": correlation_id,
            # OTel trace context — allows the Celery worker span to become
            # a child of this webhook span in the distributed trace.
            "trace_context": inject_trace_context(),
        }

        task = process_pr_review.delay(pr_data)
        logger.info("review_enqueued", task_id=task.id, pr=pr_number)

        # ── Step 6: Return 200 immediately ────────────────────────────────
        return {"status": "queued", "task_id": task.id, "pr": pr_number}


# ── Comment Trigger Handler ──────────────────────────────────────────────
# Trigger keyword (case-insensitive). Users type this as a PR comment.
_TRIGGER_KEYWORD = "/prism-review"


async def _handle_comment_trigger(
    payload: dict,
    correlation_id: str,
    delivery_id: str | None,
) -> dict:
    """Handle a /prism-review comment on a PR.

    GitHub sends `issue_comment` for both Issue and PR comments.
    We detect PRs by checking for `pull_request` in the issue object.
    """
    comment_body = payload.get("comment", {}).get("body", "").strip().lower()
    issue = payload.get("issue", {})
    repo = payload.get("repository", {})
    repo_name = repo.get("full_name", "")

    # ── Only respond to the trigger keyword ───────────────────────────
    if _TRIGGER_KEYWORD not in comment_body:
        webhook_requests_total.labels(event_type="issue_comment", status="ignored").inc()
        return {"status": "ignored", "reason": "no trigger keyword"}

    # ── Only handle PR comments (not Issue comments) ──────────────────
    # GitHub includes a `pull_request` key in the issue object only for PRs.
    if "pull_request" not in issue:
        webhook_requests_total.labels(event_type="issue_comment", status="ignored").inc()
        return {"status": "ignored", "reason": "comment is on an issue, not a PR"}

    pr_number = issue["number"]

    # ── Idempotency ───────────────────────────────────────────────────
    if await _is_duplicate_delivery(delivery_id):
        webhook_requests_total.labels(event_type="issue_comment", status="duplicate").inc()
        return {"status": "duplicate", "delivery_id": delivery_id}

    logger.info(
        "comment_trigger_accepted",
        repo=repo_name,
        pr=pr_number,
        commenter=payload.get("comment", {}).get("user", {}).get("login", ""),
        delivery_id=delivery_id,
    )
    webhook_requests_total.labels(event_type="issue_comment", status="accepted").inc()

    # ── Enqueue review ────────────────────────────────────────────────
    pr_data = {
        "number": pr_number,
        "repo_name": repo_name,
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "installation_id": payload.get("installation", {}).get("id"),
        "base_branch": "main",  # Worker will fetch actual base from PR details
        "correlation_id": correlation_id,
        "trace_context": inject_trace_context(),
    }

    task = process_pr_review.delay(pr_data)
    logger.info("review_enqueued_via_comment", task_id=task.id, pr=pr_number)

    return {"status": "queued", "task_id": task.id, "pr": pr_number, "trigger": "comment"}


# ── Manual Test Endpoint ─────────────────────────────────────────────────


@router.post("/test-review")
async def test_review(request: Request, _=Depends(require_api_key)):
    """Manual trigger endpoint for testing without a real GitHub webhook.

    Accepts: {"repo": "owner/repo", "pr_number": 123}

    Unlike the webhook endpoint, this fetches diff/files inline because
    there's no GitHub timeout constraint and the caller expects metadata
    in the response.
    """
    data = await request.json()
    repo_name = data.get("repo")
    pr_number = data.get("pr_number")
    installation_id = data.get("installation_id")  # Optional for local testing

    if not repo_name or not pr_number:
        raise HTTPException(status_code=400, detail="Missing 'repo' or 'pr_number'")

    # Generate a synthetic correlation_id for test reviews (no GitHub delivery ID)
    correlation_id = str(uuid.uuid4())
    bind_correlation_id(correlation_id)

    logger.info("test_review_triggered", repo=repo_name, pr=pr_number)

    with tracer.start_as_current_span(
        "prism.test_review",
        attributes={"correlation_id": correlation_id, "github.repo": repo_name},
    ):
        # For test-review, we fetch inline because the caller isn't GitHub
        from services.github_service import GitHubService

        github_service = GitHubService(installation_id=installation_id)

        diff = await github_service.fetch_pr_diff(repo_name, pr_number)
        changed_files = await github_service.fetch_pr_files(repo_name, pr_number)
        pr_details = await github_service.fetch_pr_details(repo_name, pr_number)

        pr_data = {
            "number": pr_number,
            "repo_name": repo_name,
            "title": pr_details.get("title", ""),
            "body": pr_details.get("body", ""),
            "changed_files": changed_files,
            "diff": diff,
            "installation_id": installation_id,
            "base_branch": pr_details.get("base", {}).get("ref", "main"),
            "correlation_id": correlation_id,
            "trace_context": inject_trace_context(),
        }

        logger.info(
            "test_review_data_ready",
            files=len(changed_files),
            diff_len=len(diff),
            title=pr_data["title"][:80],
        )

        task = process_pr_review.delay(pr_data)

        return {
            "status": "queued",
            "task_id": task.id,
            "pr": pr_number,
            "title": pr_data["title"],
            "files_changed": len(changed_files),
            "diff_size_bytes": len(diff),
        }
