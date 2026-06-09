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
            self._private_key = key_content.replace("\r", "")
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
            except FileNotFoundError:
                logger.warning("github_private_key_file_not_found", path=key_path)

        if not self._private_key:
            raise ValueError(
                "GitHub App private key not found. Set GITHUB_APP_PRIVATE_KEY "
                "(env var with PEM content) or GITHUB_APP_PRIVATE_KEY_PATH (file path)."
            )
        return self._private_key

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

    async def fetch_file_contents(
        self,
        repo_full_name: str,
        file_paths: list[str],
        commit_sha: str,
    ) -> dict[str, str]:
        """Fetch raw source code of files at a specific commit.

        Uses the GitHub Contents API to retrieve full file content.
        Returns a dict mapping file_path -> source code string.
        """
        import base64

        headers = await self._get_headers()
        client = get_httpx_client()
        results: dict[str, str] = {}

        for file_path in file_paths:
            try:
                url = f"{GITHUB_API}/repos/{repo_full_name}/contents/{file_path}"
                response = await client.get(
                    url,
                    headers=headers,
                    params={"ref": commit_sha},
                    timeout=15,
                )
                if response.status_code != 200:
                    continue

                data = response.json()
                content_b64 = data.get("content", "")
                if content_b64:
                    results[file_path] = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug("file_content_fetch_failed", file=file_path, error=str(e))

        logger.info(
            "file_contents_fetched",
            requested=len(file_paths),
            fetched=len(results),
        )
        return results

    async def post_review(self, repo_full_name: str, pr_number: int, review_data: dict[str, Any]):
        """Post the final review to GitHub as a PR review.

        Validates each inline comment against the actual diff to prevent
        422 errors from GitHub. Comments targeting lines outside the diff
        are moved to the summary comment instead of being lost.
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

            # 2. Fetch the diff and validate inline comments against it
            raw_comments = review_data.get("inline_comments", [])
            comments: list[dict] = []
            rejected_comments: list[dict] = []

            if raw_comments:
                try:
                    diff_text = await self.fetch_pr_diff(repo_full_name, pr_number)
                    from services.diff_parser import parse_diff_valid_lines, filter_valid_comments

                    diff_info = parse_diff_valid_lines(diff_text)

                    # Build comment dicts
                    all_comments = []
                    for c in raw_comments:
                        all_comments.append(
                            {
                                "path": c["path"],
                                "line": c["line"],
                                "body": c["body"],
                            }
                        )

                    comments, rejected_comments = filter_valid_comments(all_comments, diff_info)

                    logger.info(
                        "inline_comments_validated",
                        total=len(all_comments),
                        valid=len(comments),
                        rejected=len(rejected_comments),
                    )
                except Exception as val_err:
                    logger.warning("comment_validation_failed", error=str(val_err))
                    # Fall back to sending all comments unvalidated
                    comments = [{"path": c["path"], "line": c["line"], "body": c["body"]} for c in raw_comments]

            # 3. Append rejected comments to the summary so they aren't lost
            summary = review_data.get("summary_comment", "")
            if rejected_comments:
                summary += (
                    "\n\n---\n\n<details>\n<summary>📌 Additional findings (could not be placed inline)</summary>\n\n"
                )
                for rc in rejected_comments:
                    summary += f"**`{rc['path']}`** (line {rc['line']})\n{rc['body']}\n\n"
                summary += "</details>"

            # 4. Build the review payload
            url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/reviews"

            payload: dict[str, Any] = {
                "body": summary,
                "event": review_data.get("review_event", "COMMENT"),
                "comments": comments,
            }
            if head_sha:
                payload["commit_id"] = head_sha

            # 5. Post the review
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=30)
                span.set_attribute("http.status_code", response.status_code)

                if response.status_code == 422:
                    # Even after validation, GitHub may still reject — retry without inline
                    logger.warning(
                        "github_review_422_after_validation",
                        repo=repo_full_name,
                        pr=pr_number,
                        response_body=response.text[:500],
                        comment_count=len(comments),
                    )
                    # Move all comments to summary
                    for c in comments:
                        summary += f"\n\n**`{c['path']}`** (line {c['line']})\n{c['body']}"

                    fallback_payload: dict[str, Any] = {
                        "body": summary,
                        "event": review_data.get("review_event", "COMMENT"),
                        "comments": [],
                    }
                    if head_sha:
                        fallback_payload["commit_id"] = head_sha

                    fallback_resp = await client.post(url, json=fallback_payload, headers=headers, timeout=30)
                    if fallback_resp.status_code < 300:
                        logger.info(
                            "github_review_posted_without_inline",
                            repo=repo_full_name,
                            pr=pr_number,
                        )
                        return fallback_resp.json()

                    # If even the fallback fails, post as a plain issue comment
                    logger.warning(
                        "github_review_fallback_failed",
                        status=fallback_resp.status_code,
                        body=fallback_resp.text[:300],
                    )
                    comment_url = f"{GITHUB_API}/repos/{repo_full_name}/issues/{pr_number}/comments"
                    comment_resp = await client.post(
                        comment_url,
                        json={"body": summary},
                        headers=headers,
                        timeout=30,
                    )
                    if comment_resp.status_code < 300:
                        logger.info("github_review_posted_as_comment", pr=pr_number)
                        return comment_resp.json()
                    logger.error(
                        "github_review_all_fallbacks_failed",
                        pr=pr_number,
                        status=comment_resp.status_code,
                    )
                    return None

                response.raise_for_status()
                logger.info(
                    "github_review_posted",
                    repo=repo_full_name,
                    pr=pr_number,
                    commit=head_sha[:8] if head_sha else "none",
                    inline_count=len(comments),
                    rejected_count=len(rejected_comments),
                )
                return response.json()
            except Exception as e:
                span.record_exception(e)
                logger.error("github_review_post_failed", pr=pr_number, error=str(e))
                raise
