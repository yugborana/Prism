"""
Prism Comment Service — Posts Review Results to GitHub PRs.

Provides methods for:
1. Posting inline code suggestions at exact file + line positions
2. Posting summary comments to the PR conversation

Uses GitHub's comfort-fade preview API for multi-line suggestions.
"""

from observability.logging import get_logger
from utils.config import settings
from utils.connections import get_httpx_client

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"


class CommentService:
    """Posts structured review results to GitHub PRs."""

    def __init__(self, token: str | None = None, installation_id: int | None = None):
        """Initialize with an explicit token or an installation_id.

        In production, the Celery worker passes the installation_id from
        pr_data. The service lazily fetches a proper installation access
        token via GitHubService, avoiding the empty-PAT problem.
        """
        self._token = token
        self._installation_id = installation_id
        self._headers_cache: dict | None = None

    async def _get_headers(self) -> dict:
        """Lazily build auth headers, fetching an installation token if needed."""
        if self._headers_cache is not None:
            return self._headers_cache

        token = self._token
        if not token:
            if self._installation_id:
                try:
                    from services.github_service import GitHubService

                    gh = GitHubService(installation_id=self._installation_id)
                    token = await gh._get_installation_token()
                except Exception as e:
                    logger.warning("comment_service_token_fetch_failed", error=str(e))

        if not token:
            token = settings.github_token

        if not token:
            logger.error("comment_service_no_auth_token")

        self._headers_cache = {
            "Accept": "application/vnd.github.comfort-fade-preview+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Prism-Reviewer-App",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        return self._headers_cache

    # ── Inline Suggestions ────────────────────────────────────────────────

    async def post_inline_suggestion(
        self,
        repo_full_name: str,
        pr_number: int,
        file_path: str,
        line: int,
        body: str,
        suggestion: str | None = None,
        commit_sha: str | None = None,
    ) -> bool:
        """
        Post an inline comment at a specific file and line on a PR.

        If `suggestion` is provided, it is wrapped in GitHub's suggestion
        block so the reviewer can apply it with one click:
            ```suggestion
            <suggested_code>
            ```
        """
        comment_body = body
        if suggestion:
            comment_body += f"\n\n```suggestion\n{suggestion}\n```"

        payload: dict = {
            "path": file_path,
            "line": line,
            "body": comment_body,
        }
        if commit_sha:
            payload["commit_id"] = commit_sha

        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/comments"

        client = get_httpx_client()
        try:
            headers = await self._get_headers()
            response = await client.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code < 200 or response.status_code > 299:
                logger.warning(
                    "inline_comment_failed",
                    status=response.status_code,
                    file=file_path,
                    line=line,
                    response=response.text[:300],
                )
                return False
            logger.debug("inline_comment_posted", file=file_path, line=line)
            return True
        except Exception as e:
            logger.error("inline_comment_error", file=file_path, line=line, error=str(e))
            return False

    # ── Summary Comments ──────────────────────────────────────────────────

    async def post_summary_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        summary: str,
    ) -> bool:
        """Post a top-level summary comment on the PR conversation."""
        if not summary:
            logger.debug("no_summary_to_post")
            return True

        url = f"{GITHUB_API}/repos/{repo_full_name}/issues/{pr_number}/comments"
        payload = {"body": summary}

        client = get_httpx_client()
        try:
            headers = await self._get_headers()
            response = await client.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code < 200 or response.status_code > 299:
                logger.warning(
                    "summary_comment_failed",
                    status=response.status_code,
                    response=response.text[:300],
                )
                return False
            logger.info("summary_comment_posted", pr=pr_number)
            return True
        except Exception as e:
            logger.error("summary_comment_error", pr=pr_number, error=str(e))
            return False
