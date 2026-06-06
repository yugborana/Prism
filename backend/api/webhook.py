"""
Prism GitHub Webhook Router.

Receives events from GitHub (pull_request opened/synchronized).
Validates signatures and triggers the background review task via Celery.
"""

from fastapi import APIRouter, Request, Header, HTTPException, Depends
from services.github_service import GitHubService
from workers.tasks import process_pr_review
from api.auth import require_api_key
from observability.logging import get_logger
from observability.metrics import webhook_requests_total

logger = get_logger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])
github_service = GitHubService()


@router.post("/github")
async def github_webhook(
    request: Request,
    x_github_event: str = Header(None),
    x_hub_signature_256: str = Header(None)
):
    """Handle GitHub Webhook events."""
    body = await request.body()

    # 1. Verify Signature
    if not github_service.verify_signature(body, x_hub_signature_256):
        logger.warning("invalid_webhook_signature")
        webhook_requests_total.labels(event_type=x_github_event or "unknown", status="rejected").inc()
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    action = payload.get("action")

    # 2. Filter for Pull Request events (opened or updated)
    if x_github_event == "pull_request" and action in ("opened", "synchronize"):
        pr = payload["pull_request"]
        repo = payload["repository"]

        pr_number = pr["number"]
        repo_name = repo["full_name"]

        logger.info("pr_event_received", action=action, repo=repo_name, pr=pr_number)
        webhook_requests_total.labels(event_type="pull_request", status="accepted").inc()

        # 3. Fetch Diff + File List from GitHub API
        diff = await github_service.fetch_pr_diff(repo_name, pr_number)
        changed_files = await github_service.fetch_pr_files(repo_name, pr_number)

        # 4. Prepare data for the agent fleet
        pr_data = {
            "number": pr_number,
            "repo_name": repo_name,
            "title": pr.get("title", ""),
            "body": pr.get("body", ""),
            "changed_files": changed_files,
            "diff": diff,
            "installation_id": payload.get("installation", {}).get("id")
        }

        # 5. Trigger Background Review via Celery
        task = process_pr_review.delay(pr_data)
        logger.info("review_queued", task_id=task.id, pr=pr_number)

        return {"status": "queued", "task_id": task.id, "pr": pr_number}

    webhook_requests_total.labels(event_type=x_github_event or "unknown", status="ignored").inc()
    return {"status": "ignored", "event": x_github_event, "action": action}


@router.post("/test-review")
async def test_review(request: Request, _=Depends(require_api_key)):
    """
    Manual trigger endpoint for testing without a real GitHub webhook.
    Accepts: {"repo": "owner/repo", "pr_number": 123}
    """
    data = await request.json()
    repo_name = data.get("repo")
    pr_number = data.get("pr_number")

    if not repo_name or not pr_number:
        raise HTTPException(status_code=400, detail="Missing 'repo' or 'pr_number'")

    logger.info("test_review_triggered", repo=repo_name, pr=pr_number)

    # Fetch everything from GitHub API
    diff = await github_service.fetch_pr_diff(repo_name, pr_number)
    changed_files = await github_service.fetch_pr_files(repo_name, pr_number)
    pr_details = await github_service.fetch_pr_details(repo_name, pr_number)

    pr_data = {
        "number": pr_number,
        "repo_name": repo_name,
        "title": pr_details.get("title", ""),
        "body": pr_details.get("body", ""),
        "changed_files": changed_files,
        "diff": diff,  # Per-file size cap handled by github_service.fetch_pr_files_with_patches()
    }

    logger.info(
        "test_review_data_ready",
        files=len(changed_files),
        diff_len=len(diff),
        title=pr_data["title"][:80],
    )

    # Trigger Celery task
    task = process_pr_review.delay(pr_data)

    return {
        "status": "queued",
        "task_id": task.id,
        "pr": pr_number,
        "title": pr_data["title"],
        "files_changed": len(changed_files),
        "diff_size_bytes": len(diff),
    }
