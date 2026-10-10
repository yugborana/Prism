"""
Prism Merge-Base Diff Service.

Computes scoped PR diffs using `git merge-base` instead of GitHub's Compare API.

Problem:
  GitHub's PR diff endpoint (`GET /pulls/{n}` with `Accept: .diff`) returns a
  three-dot diff (base...head) that may include commits merged into the base
  branch since the PR was opened, leading to false positives when agents
  review code the PR author didn't write.

Solution (adapted from px0's computeDiffBase):
  1. Clone the PR head commit into a temporary worktree
  2. Fetch the base branch
  3. Compute `git merge-base HEAD origin/<base_branch>`
  4. Generate `git diff <merge-base>..HEAD` — only the PR's own changes

Fallback:
  If merge-base computation fails (deleted base branch, shallow clone depth
  issues), falls back to the GitHub API diff with a warning attached to the
  review state.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass, field

from observability.logging import get_logger
from observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

# Timeout for any single git operation
_GIT_TIMEOUT = 120  # seconds

# Maximum diff size before we truncate (prevent OOM on massive PRs)
_MAX_DIFF_BYTES = 2 * 1024 * 1024  # 2 MB


@dataclass
class MergeBaseDiffResult:
    """Result of a merge-base diff computation."""

    diff: str
    """The unified diff text (merge-base..HEAD)."""

    changed_files: list[str]
    """List of changed file paths relative to repo root."""

    merge_base_sha: str
    """The merge-base commit SHA used as diff anchor."""

    head_sha: str
    """The HEAD commit SHA of the PR branch."""

    method: str = "merge_base"
    """'merge_base' if scoped diff succeeded, 'github_api' if fell back."""

    warning: str = ""
    """Non-empty if merge-base fell back to GitHub API diff, explaining why."""

    worktree_path: str = ""
    """Path to the temporary worktree (caller must clean up)."""

    per_file_diffs: dict[str, str] = field(default_factory=dict)
    """Per-file diff text: file_path -> that file's unified diff."""


async def _run_git(
    *args: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = _GIT_TIMEOUT,
) -> tuple[int, str, str]:
    """Run a git command asynchronously and return (returncode, stdout, stderr).

    All git commands are run with GIT_TERMINAL_PROMPT=0 to prevent interactive
    auth prompts from hanging the worker process.
    """
    full_env = {"GIT_TERMINAL_PROMPT": "0"}
    if env:
        full_env.update(env)

    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace").strip(),
            stderr.decode("utf-8", errors="replace").strip(),
        )
    except asyncio.TimeoutError:
        logger.warning("git_command_timeout", args=list(args), timeout=timeout)
        try:
            process.kill()  # type: ignore[union-attr]
        except Exception:
            pass
        return (-1, "", f"git command timed out after {timeout}s")
    except FileNotFoundError:
        return (-1, "", "git binary not found on PATH")


def _build_clone_url(repo_full_name: str, token: str | None) -> str:
    """Build authenticated or public clone URL."""
    if token:
        return f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
    return f"https://github.com/{repo_full_name}.git"


async def compute_merge_base_diff(
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    base_branch: str,
    token: str | None = None,
    head_ref: str | None = None,
) -> MergeBaseDiffResult:
    """Compute a scoped PR diff using git merge-base.

    This is the core of the merge-base diffing feature, adapted from px0's
    ``computeDiffBase()`` in ``pr.go``.

    Args:
        repo_full_name: "owner/repo" format.
        pr_number: PR number (used for refspec naming).
        head_sha: The HEAD commit SHA of the PR branch.
        base_branch: The PR's target branch (e.g., "main").
        token: GitHub token for authenticated clone (optional).
        head_ref: The PR head branch name. If provided, used for
                  single-branch clone; otherwise falls back to fetch by SHA.

    Returns:
        MergeBaseDiffResult with the scoped diff and metadata.
    """
    with tracer.start_as_current_span(
        "prism.merge_base_diff.compute",
        attributes={
            "github.repo": repo_full_name,
            "github.pr": pr_number,
            "github.base_branch": base_branch,
        },
    ) as span:
        # Create a temporary directory for the worktree
        worktree = tempfile.mkdtemp(prefix=f"prism-pr-{pr_number}-")

        try:
            # ── Step 1: Clone the PR head ──────────────────────────────────
            clone_url = _build_clone_url(repo_full_name, token)
            safe_url = repo_full_name  # for logging (no token)

            logger.info(
                "merge_base_clone_start",
                repo=safe_url,
                pr=pr_number,
                head_ref=head_ref or head_sha[:8],
            )

            # Use GitHub's special pull refspec to fetch the PR head directly.
            # This works even for forks and doesn't require knowing the
            # head branch name.
            rc, stdout, stderr = await _run_git(
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                clone_url,
                worktree,
                timeout=_GIT_TIMEOUT,
            )
            if rc != 0:
                return _fallback_result(
                    head_sha=head_sha,
                    warning=f"Clone failed: {stderr}",
                    worktree=worktree,
                )

            # Fetch the PR head via GitHub's pull refspec
            pr_refspec = f"refs/pull/{pr_number}/head"
            rc, stdout, stderr = await _run_git(
                "fetch", "--no-tags", "origin", pr_refspec,
                cwd=worktree,
            )
            if rc != 0:
                return _fallback_result(
                    head_sha=head_sha,
                    warning=f"Fetch PR head failed: {stderr}",
                    worktree=worktree,
                )

            # Checkout FETCH_HEAD (the PR tip)
            rc, stdout, stderr = await _run_git(
                "checkout", "--detach", "FETCH_HEAD",
                cwd=worktree,
            )
            if rc != 0:
                return _fallback_result(
                    head_sha=head_sha,
                    warning=f"Checkout PR head failed: {stderr}",
                    worktree=worktree,
                )

            # Verify we're at the expected commit
            rc, actual_head, _ = await _run_git(
                "rev-parse", "HEAD",
                cwd=worktree,
            )
            if rc != 0:
                actual_head = head_sha
            else:
                actual_head = actual_head.strip()

            span.set_attribute("git.actual_head", actual_head[:12])

            # ── Step 2: Fetch the base branch ──────────────────────────────
            # Fetch into a namespaced ref to avoid collisions.
            base_refspec = f"refs/heads/{base_branch}:refs/prism/base/{pr_number}"
            rc, stdout, stderr = await _run_git(
                "fetch", "--no-tags", "origin", base_refspec,
                cwd=worktree,
            )

            fetch_err = ""
            if rc != 0:
                fetch_err = stderr or "fetch failed"
                logger.warning(
                    "merge_base_fetch_base_failed",
                    repo=safe_url,
                    base_branch=base_branch,
                    error=fetch_err,
                )

            # ── Step 3: Compute merge-base ─────────────────────────────────
            merge_base = ""
            if rc == 0:
                base_ref = f"refs/prism/base/{pr_number}"
                rc_mb, mb_out, mb_err = await _run_git(
                    "merge-base", "HEAD", base_ref,
                    cwd=worktree,
                )
                if rc_mb == 0 and mb_out:
                    merge_base = mb_out.strip()

            # Fallback: try origin/<base_branch> if the namespaced ref failed
            if not merge_base:
                rc_mb, mb_out, _ = await _run_git(
                    "merge-base", "HEAD", f"origin/{base_branch}",
                    cwd=worktree,
                )
                if rc_mb == 0 and mb_out:
                    merge_base = mb_out.strip()
                    fetch_err = ""  # recovered

            if not merge_base:
                warning = f"Could not resolve merge-base with {base_branch}"
                if fetch_err:
                    warning += f" ({fetch_err})"
                return _fallback_result(
                    head_sha=actual_head,
                    warning=warning,
                    worktree=worktree,
                )

            span.set_attribute("git.merge_base", merge_base[:12])
            logger.info(
                "merge_base_resolved",
                repo=safe_url,
                pr=pr_number,
                merge_base=merge_base[:12],
                head=actual_head[:12],
            )

            # ── Step 4: Generate the scoped diff ───────────────────────────
            rc, diff_out, diff_err = await _run_git(
                "diff", "--no-color", f"{merge_base}...{actual_head}",
                cwd=worktree,
                timeout=60,
            )
            if rc != 0:
                return _fallback_result(
                    head_sha=actual_head,
                    warning=f"git diff failed: {diff_err}",
                    worktree=worktree,
                )

            # Truncate if diff is too large
            if len(diff_out) > _MAX_DIFF_BYTES:
                diff_out = diff_out[:_MAX_DIFF_BYTES]
                logger.warning(
                    "merge_base_diff_truncated",
                    repo=safe_url,
                    pr=pr_number,
                    original_bytes=len(diff_out),
                    max_bytes=_MAX_DIFF_BYTES,
                )

            # ── Step 5: Get changed file list ──────────────────────────────
            rc, files_out, _ = await _run_git(
                "diff", "--name-only", f"{merge_base}...{actual_head}",
                cwd=worktree,
            )
            changed_files = [
                f for f in files_out.split("\n") if f.strip()
            ] if rc == 0 and files_out else []

            # ── Step 6: Build per-file diffs ───────────────────────────────
            per_file_diffs: dict[str, str] = {}
            for file_path in changed_files:
                rc_f, file_diff, _ = await _run_git(
                    "diff", "--no-color", f"{merge_base}...{actual_head}",
                    "--", file_path,
                    cwd=worktree,
                    timeout=30,
                )
                if rc_f == 0 and file_diff:
                    per_file_diffs[file_path] = file_diff

            span.set_attribute("diff.files_count", len(changed_files))
            span.set_attribute("diff.size_bytes", len(diff_out))
            span.set_attribute("diff.method", "merge_base")

            logger.info(
                "merge_base_diff_complete",
                repo=safe_url,
                pr=pr_number,
                files=len(changed_files),
                diff_bytes=len(diff_out),
                merge_base=merge_base[:12],
            )

            return MergeBaseDiffResult(
                diff=diff_out,
                changed_files=changed_files,
                merge_base_sha=merge_base,
                head_sha=actual_head,
                method="merge_base",
                warning="",
                worktree_path=worktree,
                per_file_diffs=per_file_diffs,
            )

        except Exception as e:
            logger.error(
                "merge_base_diff_unexpected_error",
                repo=repo_full_name,
                pr=pr_number,
                error=str(e),
            )
            return _fallback_result(
                head_sha=head_sha,
                warning=f"Unexpected error: {e}",
                worktree=worktree,
            )


def _fallback_result(
    head_sha: str,
    warning: str,
    worktree: str,
) -> MergeBaseDiffResult:
    """Return an empty result indicating fallback to GitHub API diff is needed."""
    logger.warning("merge_base_diff_fallback", warning=warning)
    return MergeBaseDiffResult(
        diff="",
        changed_files=[],
        merge_base_sha="",
        head_sha=head_sha,
        method="github_api",
        warning=warning,
        worktree_path=worktree,
    )


def cleanup_worktree(worktree_path: str) -> None:
    """Remove a temporary worktree created by compute_merge_base_diff.

    Should be called after the review pipeline finishes, regardless of
    success or failure.
    """
    if worktree_path:
        try:
            shutil.rmtree(worktree_path, ignore_errors=True)
            logger.debug("merge_base_worktree_cleaned", path=worktree_path)
        except Exception as e:
            logger.warning("merge_base_worktree_cleanup_failed", error=str(e))
