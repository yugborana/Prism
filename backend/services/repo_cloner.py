"""
Prism Repo Cloner — Shallow Clone and Incremental Fetch.

Manages temporary shallow clones of repositories for indexing. Clones are:
  - Shallow (--depth 1) to minimize disk usage (~30MB vs ~500MB for a 100k LOC repo)
  - Shared across PRs for the same repo (clone once, fetch to update)
  - Authenticated via GitHub App installation tokens for private repos
  - Auto-cleaned after 24 hours by a Celery beat task
  - Concurrency-safe via Redis distributed locks
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

from observability.logging import get_logger
from utils.config import settings
from utils.connections import get_redis as get_redis_pool

logger = get_logger(__name__)

# Max time to wait for a concurrent clone to finish (seconds)
CLONE_LOCK_TIMEOUT = 120
CLONE_LOCK_TTL = 300  # Lock auto-expires after 5 min (safety net)


class RepoCloner:
    """Manages shallow clones for repo indexing."""

    def __init__(self) -> None:
        self._base_dir = Path(settings.temp_repo_dir)
        self._redis: Any | None = None

    def _repo_dir(self, repo_name: str) -> Path:
        """Get the local directory path for a repo clone.

        ``repo_name`` is in the format ``owner/repo``.
        """
        # Replace / with os-safe separator
        safe_name = repo_name.replace("/", "__")
        return self._base_dir / safe_name

    async def _get_redis(self) -> Any | None:
        if self._redis is None:
            self._redis = get_redis_pool()
        return self._redis

    async def ensure_clone(
        self,
        repo_name: str,
        base_branch: str,
        installation_id: int,
    ) -> Path:
        """Return the path to a shallow clone of the repo.

        If the clone already exists, performs an incremental fetch + reset.
        If not, performs a fresh shallow clone.

        Uses a Redis lock to prevent parallel clones of the same repo.
        """
        repo_dir = self._repo_dir(repo_name)
        lock_key = f"prism:clone_lock:{repo_name}"

        # Acquire distributed lock
        redis = await self._get_redis()
        lock_acquired = False

        if redis:
            try:
                # Try to acquire lock with NX (set-if-not-exists)
                lock_acquired = await redis.set(lock_key, "1", ex=CLONE_LOCK_TTL, nx=True)

                if not lock_acquired:
                    # Another worker is cloning — wait for it
                    logger.info("clone_waiting_for_lock", repo=repo_name)
                    for _ in range(CLONE_LOCK_TIMEOUT):
                        await asyncio.sleep(1)
                        if not await redis.exists(lock_key):
                            break
                    else:
                        # Lock expired or timed out — proceed anyway
                        logger.warning("clone_lock_timeout", repo=repo_name)

                    # If the clone exists now, just return it
                    if (repo_dir / ".git").exists():
                        return repo_dir
            except Exception as e:
                logger.warning("clone_lock_failed", error=str(e))

        try:
            if (repo_dir / ".git").exists():
                # Incremental update: fetch + reset
                await self._incremental_fetch(repo_dir, base_branch)
            else:
                # Fresh shallow clone
                clone_url = await self._build_clone_url(repo_name, installation_id)
                await self._shallow_clone(clone_url, repo_dir, base_branch)

            # Update last-used timestamp
            self._touch_marker(repo_dir)

            return repo_dir

        finally:
            # Release lock
            if redis and lock_acquired:
                try:
                    await redis.delete(lock_key)
                except Exception:
                    pass

    async def _build_clone_url(self, repo_name: str, installation_id: int) -> str:
        """Build the clone URL, authenticated for private repos."""
        if installation_id:
            try:
                from services.github_service import GitHubService

                gh = GitHubService(installation_id=installation_id)
                token = await gh._get_installation_token()
                return f"https://x-access-token:{token}@github.com/{repo_name}.git"
            except Exception as e:
                logger.warning(
                    "clone_auth_fallback",
                    repo=repo_name,
                    error=str(e),
                )

        # Fallback: use PAT from config (local dev) or unauthenticated (public repos)
        if settings.github_token:
            return f"https://x-access-token:{settings.github_token}@github.com/{repo_name}.git"

        return f"https://github.com/{repo_name}.git"

    async def _shallow_clone(self, clone_url: str, repo_dir: Path, branch: str) -> None:
        """Perform a shallow clone of a specific branch."""
        # Ensure parent directory exists
        repo_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove any partial clone
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)

        # Sanitize URL for logging (remove token)
        safe_url = clone_url.split("@")[-1] if "@" in clone_url else clone_url

        logger.info("clone_starting", repo=safe_url, branch=branch)

        process = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            branch,
            "--quiet",
            clone_url,
            str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error(
                "clone_failed",
                repo=safe_url,
                exit_code=process.returncode,
                error=error_msg,
            )
            raise RuntimeError(f"Git clone failed: {error_msg}")

        logger.info("clone_complete", repo=safe_url)

    async def _incremental_fetch(self, repo_dir: Path, branch: str) -> None:
        """Fetch latest changes and reset to HEAD."""
        logger.info("clone_incremental_fetch", repo=str(repo_dir.name), branch=branch)

        # git fetch origin <branch>
        process = await asyncio.create_subprocess_exec(
            "git",
            "fetch",
            "origin",
            branch,
            "--depth",
            "1",
            "--quiet",
            cwd=str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.warning("fetch_failed", error=error_msg)
            # If fetch fails, try a full re-clone
            raise RuntimeError(f"Git fetch failed: {error_msg}")

        # git reset --hard FETCH_HEAD
        process = await asyncio.create_subprocess_exec(
            "git",
            "reset",
            "--hard",
            "FETCH_HEAD",
            cwd=str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

    def _touch_marker(self, repo_dir: Path) -> None:
        """Update the last-used timestamp marker file."""
        marker = repo_dir / ".prism_last_used"
        marker.write_text(str(time.time()))

    def _get_last_used(self, repo_dir: Path) -> float:
        """Read the last-used timestamp."""
        marker = repo_dir / ".prism_last_used"
        try:
            return float(marker.read_text().strip())
        except (OSError, ValueError):
            return 0.0

    async def cleanup_stale(self, max_age_hours: int = 24) -> int:
        """Delete repo clones older than max_age_hours.

        Called by a Celery beat task (hourly).
        Returns count of deleted clones.
        """
        if not self._base_dir.exists():
            return 0

        deleted = 0
        cutoff = time.time() - (max_age_hours * 3600)

        for entry in self._base_dir.iterdir():
            if not entry.is_dir():
                continue

            last_used = self._get_last_used(entry)
            if last_used < cutoff:
                try:
                    shutil.rmtree(entry, ignore_errors=True)
                    deleted += 1
                    logger.info("clone_cleaned", dir=entry.name)
                except Exception as e:
                    logger.warning("clone_cleanup_failed", dir=entry.name, error=str(e))

        if deleted:
            logger.info("clone_cleanup_complete", deleted=deleted)

        return deleted
