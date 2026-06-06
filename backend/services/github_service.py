"""
Prism GitHub Service.

Handles authentication and provides methods for:
1. Verifying webhook signatures.
2. Fetching PR details, diffs, and file lists.
3. Posting review comments back to GitHub.
"""

import hmac
import hashlib
from typing import Any

import httpx
from utils.config import settings
from observability.logging import get_logger

logger = get_logger(__name__)

# GitHub API base
GITHUB_API = "https://api.github.com"

class GitHubService:
    """Interface for GitHub API operations."""

    def __init__(self):
        self.webhook_secret = settings.github_webhook_secret
        self.token = settings.github_token
        self._headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Prism-Reviewer-App",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"

    def verify_signature(self, payload: bytes, signature: str | None) -> bool:
        """Verify that the webhook came from GitHub using HMAC-SHA256."""
        if not signature or not self.webhook_secret:
            return False

        sha_name, signature_val = signature.split("=")
        if sha_name != "sha256":
            return False

        mac = hmac.new(self.webhook_secret.encode(), msg=payload, digestmod=hashlib.sha256)
        return hmac.compare_digest(mac.hexdigest(), signature_val)

    async def fetch_pr_details(self, repo_full_name: str, pr_number: int) -> dict[str, Any]:
        """Fetch full PR metadata from GitHub API."""
        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=self._headers)
            response.raise_for_status()
            return response.json()

    async def fetch_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Fetch the raw unified diff of a Pull Request."""
        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}"
        headers = {**self._headers, "Accept": "application/vnd.github.v3.diff"}

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.error("github_diff_fetch_failed", pr=pr_number, error=str(e))
                return ""

    async def fetch_pr_files(self, repo_full_name: str, pr_number: int) -> list[str]:
        """Fetch the list of files changed in a PR."""
        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/files"
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(url, headers=self._headers)
                response.raise_for_status()
                files = response.json()
                return [f["filename"] for f in files]
            except Exception as e:
                logger.error("github_files_fetch_failed", pr=pr_number, error=str(e))
                return []

    async def fetch_pr_files_with_patches(
        self, repo_full_name: str, pr_number: int
    ) -> list[dict]:
        """
        Fetch per-file metadata including patches, with cumulative diff size cap.

        Files are skipped individually if they would push total diff size
        past settings.max_diff_size, instead of truncating the entire diff.
        """
        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/files"
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(url, headers=self._headers)
                response.raise_for_status()
                raw_files = response.json()
            except Exception as e:
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
            file_details.append({
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": patch,
            })

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
        async with httpx.AsyncClient(timeout=30) as client:
            # 1. Fetch HEAD SHA — required for inline comments to land correctly
            try:
                pr_resp = await client.get(
                    f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}",
                    headers=self._headers,
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
                response = await client.post(url, json=payload, headers=self._headers)
                response.raise_for_status()
                logger.info("github_review_posted", repo=repo_full_name, pr=pr_number, commit=head_sha[:8] if head_sha else "none")
                return response.json()
            except Exception as e:
                logger.error("github_review_post_failed", pr=pr_number, error=str(e))
                raise
