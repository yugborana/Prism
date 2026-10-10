"""
Prism Diff Cache — Skip re-analyzing unchanged files across force-pushes.

When a PR is force-pushed or gets new commits, GitHub fires a `synchronize`
webhook and Prism re-reviews the entire PR. But many files in the diff are
unchanged — re-running 4 LLM agents on them wastes cost and time.

This module hashes each file's diff and stores it in Redis, keyed by
(repo, pr_number, file_path). On the next review of the same PR, files
whose diff hash is identical are skipped, and their previous findings
are carried forward.

Design (inspired by px0's pushedSHA tracking in pr.go):
  - Key:   prism:diff_cache:{repo}:{pr_number}:{file_path}
  - Value: JSON {"hash": "<sha256>", "head_sha": "<commit>", "findings": [...]}
  - TTL:   7 days (PRs rarely stay open longer; old data evicts naturally)

Graceful degradation:
  - If Redis is unavailable, returns empty cache (all files analyzed)
  - If hash computation fails for a file, that file is treated as changed
  - Never blocks the review pipeline — cache misses just mean full analysis
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)

# Redis key TTL: 7 days in seconds
_CACHE_TTL = 7 * 24 * 60 * 60

# Redis key prefix
_KEY_PREFIX = "prism:diff_cache"


@dataclass
class DiffCacheResult:
    """Result of comparing current diffs against the cache."""

    changed_files: list[str]
    """Files whose diffs have changed since the last review (need re-analysis)."""

    cached_files: list[str]
    """Files whose diffs are identical to the last review (skip analysis)."""

    cached_findings: list[dict]
    """Previous findings for cached files, carried forward."""

    total_files: int
    """Total number of files in the PR."""

    cache_hit_rate: float
    """Fraction of files that were cache hits (0.0 to 1.0)."""


def _cache_key(repo: str, pr_number: int, file_path: str) -> str:
    """Build a Redis key for a file's diff cache entry."""
    # Normalize repo name (no slashes in key segments)
    safe_repo = repo.replace("/", ":")
    return f"{_KEY_PREFIX}:{safe_repo}:{pr_number}:{file_path}"


def _hash_diff(diff_text: str) -> str:
    """Compute a stable SHA-256 hash of a file's diff.

    Strips trailing whitespace from each line to avoid false mismatches
    from line-ending differences, but preserves content and indentation.
    """
    # Normalize line endings and strip trailing whitespace per line
    normalized = "\n".join(
        line.rstrip() for line in diff_text.splitlines()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def check_diff_cache(
    redis_client: Any,
    repo: str,
    pr_number: int,
    per_file_diffs: dict[str, str],
    all_changed_files: list[str],
) -> DiffCacheResult:
    """Compare current per-file diffs against cached hashes.

    Args:
        redis_client: Async Redis client (from utils.connections.get_redis).
        repo: Repository full name ("owner/repo").
        pr_number: PR number.
        per_file_diffs: Dict of file_path -> that file's unified diff text.
        all_changed_files: Full list of changed file paths in the PR.

    Returns:
        DiffCacheResult indicating which files changed and which are cached.
    """
    if redis_client is None:
        # No Redis — treat all files as changed
        return DiffCacheResult(
            changed_files=all_changed_files,
            cached_files=[],
            cached_findings=[],
            total_files=len(all_changed_files),
            cache_hit_rate=0.0,
        )

    changed: list[str] = []
    cached: list[str] = []
    cached_findings: list[dict] = []

    for file_path in all_changed_files:
        file_diff = per_file_diffs.get(file_path, "")

        if not file_diff:
            # No diff text available for this file — treat as changed
            changed.append(file_path)
            continue

        current_hash = _hash_diff(file_diff)
        key = _cache_key(repo, pr_number, file_path)

        try:
            cached_json = await redis_client.get(key)
            if cached_json:
                cached_data = json.loads(cached_json)
                stored_hash = cached_data.get("hash", "")

                if stored_hash == current_hash:
                    # Cache hit — diff is identical to last review
                    cached.append(file_path)
                    # Carry forward previous findings
                    prev_findings = cached_data.get("findings", [])
                    if prev_findings:
                        cached_findings.extend(prev_findings)
                    continue
        except Exception as e:
            # Redis error — treat as changed (fail-open)
            logger.debug(
                "diff_cache_check_failed",
                file=file_path,
                error=str(e),
            )

        # Cache miss or hash mismatch — file needs re-analysis
        changed.append(file_path)

    total = len(all_changed_files)
    hit_rate = len(cached) / total if total > 0 else 0.0

    logger.info(
        "diff_cache_checked",
        repo=repo,
        pr=pr_number,
        total_files=total,
        cached_files=len(cached),
        changed_files=len(changed),
        cache_hit_rate=round(hit_rate, 2),
        cached_findings_carried=len(cached_findings),
    )

    return DiffCacheResult(
        changed_files=changed,
        cached_files=cached,
        cached_findings=cached_findings,
        total_files=total,
        cache_hit_rate=hit_rate,
    )


async def update_diff_cache(
    redis_client: Any,
    repo: str,
    pr_number: int,
    per_file_diffs: dict[str, str],
    head_sha: str,
    findings_by_file: dict[str, list[dict]] | None = None,
) -> int:
    """Store current file diff hashes in Redis after a review completes.

    Called at the end of the review pipeline so the *next* review of
    this PR can skip unchanged files.

    Args:
        redis_client: Async Redis client.
        repo: Repository full name.
        pr_number: PR number.
        per_file_diffs: Dict of file_path -> that file's unified diff text.
        head_sha: The HEAD commit SHA this review analyzed.
        findings_by_file: Optional dict of file_path -> list of finding dicts
                          from this review. Stored so cached files can carry
                          their findings forward without re-running agents.

    Returns:
        Number of file cache entries written.
    """
    if redis_client is None:
        return 0

    findings_by_file = findings_by_file or {}
    written = 0

    for file_path, diff_text in per_file_diffs.items():
        if not diff_text:
            continue

        diff_hash = _hash_diff(diff_text)
        key = _cache_key(repo, pr_number, file_path)

        cache_entry = {
            "hash": diff_hash,
            "head_sha": head_sha,
            "findings": findings_by_file.get(file_path, []),
        }

        try:
            await redis_client.set(
                key,
                json.dumps(cache_entry),
                ex=_CACHE_TTL,
            )
            written += 1
        except Exception as e:
            logger.debug(
                "diff_cache_write_failed",
                file=file_path,
                error=str(e),
            )

    if written:
        logger.info(
            "diff_cache_updated",
            repo=repo,
            pr=pr_number,
            files_cached=written,
            head_sha=head_sha[:12] if head_sha else "",
        )

    return written


