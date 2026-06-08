"""
Prism GitHub Service — GitHub App Authentication.

Uses short-lived installation tokens (1hr expiry, auto-refreshed) instead
of long-lived Personal Access Tokens. Supports multi-org installations.

Auth flow:
  1. Load private key (from env var, file, or AWS Secrets Manager)
  2. Generate a 10-minute JWT signed with RS256
  3. Exchange JWT → installation access token via GitHub API
  4. Cache token in Redis for 55 minutes (tokens expire at 60m)
  5. All API calls use the short-lived token

References:
  - https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app
"""

import time
import asyncio
from typing import Any

import jwt
from utils.config import settings
from utils.connections import get_httpx_client, get_redis
from observability.logging import get_logger
from observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

# GitHub API base
GITHUB_API = "https://api.github.com"

# Token cache TTL — refresh 5 minutes before actual expiry (60m)
_TOKEN_CACHE_TTL = 55 * 60  # 55 minutes in seconds


class GitHubService:
    """Interface for GitHub API operations using GitHub App auth."""

    def __init__(self, installation_id: int | None = None):
        """
        Args:
            installation_id: The GitHub App installation ID for this org/repo.
                             Passed from the webhook payload or from pr_data.
                             If None, uses a fallback PAT for local testing.
        """
        self.installation_id = installation_id
        self._private_key: str | None = None
        self._token_cache: dict[int, tuple[str, float]] = {}  # in-memory fallback

        self._base_headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Prism-Reviewer-App",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # ── Private Key Loading ───────────────────────────────────────────────

    async def _get_private_key(self) -> str:
        """Load the GitHub App private key.

        Resolution order:
        1. Already cached in memory
        2. GITHUB_APP_PRIVATE_KEY env var (for AWS — injected from Secrets Manager)
        3. GITHUB_APP_PRIVATE_KEY_PATH file on disk (for local dev)

        In production on EKS, the key is stored in AWS Secrets Manager and
        injected as an environment variable by the External Secrets Operator
        or init container.
        """
        if self._private_key:
            return self._private_key

        # Option 1: Direct env var (production — injected from AWS Secrets Manager)
        key_content = settings.github_app_private_key
        if key_content and key_content.startswith("-----BEGIN"):
            self._private_key = key_content
            logger.info("github_private_key_loaded", source="env_var")
            return self._private_key

        # Option 2: File path (local development)
        key_path = settings.github_app_private_key_path
        if key_path:
            try:

                def _read_file():
                    with open(key_path, "r") as f:
                        return f.read()

                self._private_key = await asyncio.to_thread(_read_file)
                logger.info("github_private_key_loaded", source="file", path=key_path)
                return self._private_key
            except FileNotFoundError:
                logger.warning("github_private_key_file_not_found", path=key_path)

        raise ValueError(
            "GitHub App private key not found. Set GITHUB_APP_PRIVATE_KEY "
            "(env var with PEM content) or GITHUB_APP_PRIVATE_KEY_PATH (file path)."
        )

    # ── JWT Generation ────────────────────────────────────────────────────

    async def _generate_jwt(self) -> str:
        """Generate a short-lived JWT (10 min) signed with the App's private key.

        This JWT is used to authenticate as the GitHub App itself,
        before requesting installation-specific access tokens.
        """
        private_key = await self._get_private_key()

        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60s ago (clock skew tolerance)
            "exp": now + (10 * 60),  # Expires in 10 minutes
            "iss": settings.github_app_id,  # GitHub App ID
        }

        token = jwt.encode(payload, private_key, algorithm="RS256")
        logger.debug("github_jwt_generated", app_id=settings.github_app_id)
        return token

    # ── Installation Token ────────────────────────────────────────────────

    async def _get_installation_token(self) -> str:
        """Get a short-lived installation access token (1hr).

        Flow:
        1. Check Redis cache → return if valid
        2. Generate a JWT
        3. POST /app/installations/{id}/access_tokens
        4. Cache the token in Redis for 55 minutes
        5. Return the token

        Falls back to in-memory cache if Redis is unavailable.
        """
        if not self.installation_id:
            raise ValueError(
                "No installation_id provided. Cannot generate installation token. "
                "For local testing, set GITHUB_TOKEN in .env."
            )

        # ── Step 1: Check Redis cache ─────────────────────────────────
        cache_key = f"prism:github:token:{self.installation_id}"
        redis = get_redis()
        if redis is not None:
            try:
                cached = await redis.get(cache_key)
                if cached:
                    logger.debug("github_token_cache_hit", installation_id=self.installation_id)
                    return cached
            except Exception as e:
                logger.warning("github_token_redis_unavailable", error=str(e))
                # Check in-memory fallback
                if self.installation_id in self._token_cache:
                    token, expires_at = self._token_cache[self.installation_id]
                    if time.time() < expires_at:
                        return token
        else:
            # No Redis pool — check in-memory fallback
            if self.installation_id in self._token_cache:
                token, expires_at = self._token_cache[self.installation_id]
                if time.time() < expires_at:
                    return token

        # ── Step 2: Generate JWT and request new token ────────────────
        app_jwt = await self._generate_jwt()

        client = get_httpx_client()
        response = await client.post(
            f"{GITHUB_API}/app/installations/{self.installation_id}/access_tokens",
            headers={
                **self._base_headers,
                "Authorization": f"Bearer {app_jwt}",
            },
            timeout=15,
        )
        response.raise_for_status()
        token_data = response.json()

        token = token_data["token"]

        logger.info(
            "github_installation_token_generated",
            installation_id=self.installation_id,
            expires_at=token_data.get("expires_at"),
        )

        # ── Step 3: Cache the token ───────────────────────────────────
        if redis is not None:
            try:
                await redis.set(cache_key, token, ex=_TOKEN_CACHE_TTL)
            except Exception as e:
                logger.warning("github_token_cache_store_failed", error=str(e))

        # In-memory fallback
        self._token_cache[self.installation_id] = (
            token,
            time.time() + _TOKEN_CACHE_TTL,
        )

        return token

    # ── Authenticated Headers ─────────────────────────────────────────────

    async def _get_headers(self, accept: str | None = None) -> dict[str, str]:
        """Build authenticated headers for GitHub API calls.

        Uses installation token if installation_id is set,
        otherwise falls back to GITHUB_TOKEN (local dev only).
        """
        headers = {**self._base_headers}
        if accept:
            headers["Accept"] = accept

        if self.installation_id:
            token = await self._get_installation_token()
            headers["Authorization"] = f"Bearer {token}"
        elif settings.github_token:
            # Fallback for local development / test-review endpoint
            headers["Authorization"] = f"Bearer {settings.github_token}"
            logger.debug("github_using_pat_fallback")
        else:
            logger.warning("github_no_auth_configured")

        return headers

    # ── API Methods ───────────────────────────────────────────────────────

    async def fetch_pr_details(self, repo_full_name: str, pr_number: int) -> dict[str, Any]:
        """Fetch full PR metadata from GitHub API."""
        with tracer.start_as_current_span(
            "prism.github.fetch_pr_details",
            attributes={"github.repo": repo_full_name, "github.pr": pr_number},
        ) as span:
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}"
            headers = await self._get_headers()
            client = get_httpx_client()
            response = await client.get(url, headers=headers, timeout=30)
            span.set_attribute("http.status_code", response.status_code)
            response.raise_for_status()
            return response.json()

    async def fetch_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Fetch the raw unified diff of a Pull Request."""
        with tracer.start_as_current_span(
            "prism.github.fetch_pr_diff",
            attributes={"github.repo": repo_full_name, "github.pr": pr_number},
        ) as span:
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}"
            headers = await self._get_headers(accept="application/vnd.github.v3.diff")

            client = get_httpx_client()
            try:
                response = await client.get(url, headers=headers, timeout=60)
                span.set_attribute("http.status_code", response.status_code)
                response.raise_for_status()
                span.set_attribute("github.diff_size", len(response.text))
                return response.text
            except Exception as e:
                span.record_exception(e)
                logger.error("github_diff_fetch_failed", pr=pr_number, error=str(e))
                return ""

    async def fetch_pr_files(self, repo_full_name: str, pr_number: int) -> list[str]:
        """Fetch the list of files changed in a PR."""
        with tracer.start_as_current_span(
            "prism.github.fetch_pr_files",
            attributes={"github.repo": repo_full_name, "github.pr": pr_number},
        ) as span:
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/files"
            headers = await self._get_headers()
            client = get_httpx_client()
            try:
                response = await client.get(url, headers=headers, timeout=30)
                span.set_attribute("http.status_code", response.status_code)
                response.raise_for_status()
                files = response.json()
                span.set_attribute("github.files_count", len(files))
                return [f["filename"] for f in files]
            except Exception as e:
                span.record_exception(e)
                logger.error("github_files_fetch_failed", pr=pr_number, error=str(e))
                return []

    async def fetch_pr_files_with_patches(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """
        Fetch per-file metadata including patches, with cumulative diff size cap.

        Files are skipped individually if they would push total diff size
        past settings.max_diff_size, instead of truncating the entire diff.
        """
        with tracer.start_as_current_span(
            "prism.github.fetch_pr_files_with_patches",
            attributes={"github.repo": repo_full_name, "github.pr": pr_number},
        ) as span:
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/files"
            headers = await self._get_headers()
            client = get_httpx_client()
            try:
                response = await client.get(url, headers=headers, timeout=30)
                span.set_attribute("http.status_code", response.status_code)
                response.raise_for_status()
                raw_files = response.json()
            except Exception as e:
                span.record_exception(e)
                logger.error("github_files_fetch_failed", pr=pr_number, error=str(e))
                return []

            file_details = []
            total_diff_size = 0

            for f in raw_files:
                patch = f.get("patch", "")
                patch_size = len(patch)

                # Skip individual files that would exceed the cap
                if total_diff_size + patch_size > settings.max_diff_size:
                    logger.info(
                        "file_skipped_diff_cap",
                        file=f.get("filename"),
                        patch_size=patch_size,
                        total_so_far=total_diff_size,
                        max_diff_size=settings.max_diff_size,
                    )
                    continue

                total_diff_size += patch_size
                file_details.append(
                    {
                        "filename": f.get("filename", ""),
                        "status": f.get("status", ""),
                        "additions": f.get("additions", 0),
                        "deletions": f.get("deletions", 0),
                        "patch": patch,
                    }
                )

            span.set_attribute("github.total_files", len(raw_files))
            span.set_attribute("github.included_files", len(file_details))
            span.set_attribute("github.total_diff_size", total_diff_size)

            logger.info(
                "files_fetched_with_patches",
                pr=pr_number,
                total_files=len(raw_files),
                included_files=len(file_details),
                total_diff_size=total_diff_size,
            )
            return file_details

    async def post_review(self, repo_full_name: str, pr_number: int, review_data: dict[str, Any]):
        """Post the final review to GitHub as a PR review.

        Fetches the HEAD SHA first
        so inline comments attach to the correct diff position.
        """
        with tracer.start_as_current_span(
            "prism.github.post_review",
            attributes={
                "github.repo": repo_full_name,
                "github.pr": pr_number,
                "github.comments_count": len(review_data.get("inline_comments", [])),
            },
        ) as span:
            headers = await self._get_headers()
            client = get_httpx_client()

            # 1. Fetch HEAD SHA — required for inline comments to land correctly
            try:
                pr_resp = await client.get(
                    f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}",
                    headers=headers,
                    timeout=30,
                )
                pr_resp.raise_for_status()
                head_sha = pr_resp.json().get("head", {}).get("sha", "")
            except Exception as e:
                logger.warning("head_sha_fetch_failed", pr=pr_number, error=str(e))
                head_sha = ""

            # 2. Build the review payload
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/reviews"

            comments = []
            for c in review_data.get("inline_comments", []):
                comment = {
                    "path": c["path"],
                    "line": c["line"],
                    "body": c["body"],
                }
                if head_sha:
                    comment["commit_id"] = head_sha
                comments.append(comment)

            payload: dict[str, Any] = {
                "body": review_data.get("summary_comment", ""),
                "event": review_data.get("review_event", "COMMENT"),
                "comments": comments,
            }
            if head_sha:
                payload["commit_id"] = head_sha

            # 3. Post the review
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=30)
                span.set_attribute("http.status_code", response.status_code)
                response.raise_for_status()
                logger.info(
                    "github_review_posted",
                    repo=repo_full_name,
                    pr=pr_number,
                    commit=head_sha[:8] if head_sha else "none",
                )
                return response.json()
            except Exception as e:
                span.record_exception(e)
                logger.error("github_review_post_failed", pr=pr_number, error=str(e))
                raise
