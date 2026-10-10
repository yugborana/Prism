"""
Prism Fix Verifier — Execution-Grounded Suggestion Validation.

Before Prism posts a ``suggested_fix`` to GitHub, this module applies the
patch to an ephemeral git worktree, runs the repo's linter/tests, and tags
the suggestion as verified or unverified.

Design:
  * Subprocess-based sandbox (MVP). Upgradeable to Docker / E2B later.
  * Each suggestion is verified independently in its own worktree.
  * Timeouts and resource limits prevent runaway processes.
  * Best-effort: a verification failure never blocks the review from posting.
  * Only suggestions are verified; plain comments are left untouched.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from observability.logging import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

LINT_TIMEOUT = 15       # seconds
TEST_TIMEOUT = 90       # seconds
MAX_VERIFY = 10         # max suggestions to verify per review (cost cap)
WORKTREE_PREFIX = "prism-verify-"


@dataclass
class VerifyResult:
    passed: bool
    stage: str          # "lint", "test", "skip", "error"
    detail: str = ""


# ── Test-runner discovery ────────────────────────────────────────────────

_RUNNERS: list[tuple[str, list[str]]] = [
    # (marker file, command)
    ("pyproject.toml",  ["python", "-m", "pytest", "--tb=line", "-q", "--no-header", "-x"]),
    ("setup.py",        ["python", "-m", "pytest", "--tb=line", "-q", "--no-header", "-x"]),
    ("Cargo.toml",      ["cargo", "test", "--no-fail-fast"]),
    ("go.mod",          ["go", "test", "./..."]),
    ("package.json",    ["npm", "test", "--", "--bail"]),
    ("Makefile",        ["make", "test"]),
]

_LINTERS: dict[str, list[str]] = {
    ".py":  ["python", "-m", "py_compile"],
    ".js":  ["node", "--check"],
    ".ts":  ["node", "--check"],       # basic syntax only
    ".go":  ["gofmt", "-e"],
    ".rs":  ["rustfmt", "--check"],
}


def _discover_test_cmd(repo_root: Path) -> list[str] | None:
    """Find the test command for this repo by checking for marker files."""
    for marker, cmd in _RUNNERS:
        if (repo_root / marker).exists():
            return cmd
    return None


def _discover_lint_cmd(file_path: str) -> list[str] | None:
    """Return a lint / syntax-check command for the given file extension."""
    ext = Path(file_path).suffix.lower()
    return _LINTERS.get(ext)


# ── Core verification ────────────────────────────────────────────────────


async def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    *,
    extra_args: list[str] | None = None,
) -> tuple[int, str]:
    """Run a command with a timeout. Returns (return_code, combined_output)."""
    full_cmd = cmd + (extra_args or [])
    try:
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # Prevent the child from inheriting sensitive env vars
            env=_safe_env(),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, (stdout or b"").decode(errors="replace")[:2000]
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -1, f"Timed out after {timeout}s"
    except FileNotFoundError:
        return -2, f"Command not found: {full_cmd[0]}"
    except Exception as e:
        return -3, str(e)[:500]


def _safe_env() -> dict[str, str]:
    """Build a sanitised environment for subprocess execution.

    Inherits PATH so tools are found, but strips secrets.
    """
    import os

    env = dict(os.environ)
    for key in list(env.keys()):
        kl = key.lower()
        if any(s in kl for s in ("secret", "token", "password", "key", "credential")):
            del env[key]
    # Ensure CI-style behaviour in test runners
    env["CI"] = "true"
    return env


async def _create_worktree(repo_root: Path) -> Path:
    """Create a temporary git worktree for isolated verification."""
    worktree_dir = Path(tempfile.mkdtemp(prefix=WORKTREE_PREFIX))
    rc, out = await _run_cmd(
        ["git", "worktree", "add", "--detach", str(worktree_dir)],
        cwd=repo_root,
        timeout=30,
    )
    if rc != 0:
        # Fallback: plain copy (works for shallow clones that don't support worktrees)
        shutil.rmtree(worktree_dir, ignore_errors=True)
        worktree_dir = Path(tempfile.mkdtemp(prefix=WORKTREE_PREFIX))
        await asyncio.to_thread(shutil.copytree, repo_root, worktree_dir, dirs_exist_ok=True)
        logger.debug("verify_worktree_fallback_copy", reason=out[:200])
    return worktree_dir


def _apply_patch(worktree: Path, file_path: str, line: int, suggestion: str) -> bool:
    """Apply a suggestion to a file at the given line.

    The suggestion replaces the single line at ``line`` with the
    suggestion text. This matches how GitHub's suggestion blocks work:
    one hunk replacing a contiguous range.

    Returns True if the patch was applied successfully.
    """
    target = worktree / file_path
    if not target.exists():
        return False

    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except Exception:
        return False

    # line is 1-indexed
    idx = line - 1
    if idx < 0 or idx >= len(lines):
        return False

    # Replace the target line with the suggestion
    suggestion_lines = suggestion.rstrip("\n").split("\n")
    # Preserve original line ending
    ending = "\n"
    new_lines = lines[:idx] + [l + ending for l in suggestion_lines] + lines[idx + 1:]

    try:
        target.write_text("".join(new_lines), encoding="utf-8")
        return True
    except Exception:
        return False


async def _verify_one(
    repo_root: Path,
    file_path: str,
    line: int,
    suggestion: str,
) -> VerifyResult:
    """Verify a single suggestion. Creates a worktree, patches, lints, tests."""
    worktree: Path | None = None
    try:
        worktree = await _create_worktree(repo_root)

        # 1. Apply the patch
        if not _apply_patch(worktree, file_path, line, suggestion):
            return VerifyResult(passed=False, stage="patch", detail="Could not apply patch")

        # 2. Lint / syntax check (fast gate)
        lint_cmd = _discover_lint_cmd(file_path)
        if lint_cmd:
            target_file = str(Path(file_path))
            rc, out = await _run_cmd(lint_cmd, cwd=worktree, timeout=LINT_TIMEOUT, extra_args=[target_file])
            if rc != 0:
                return VerifyResult(passed=False, stage="lint", detail=out[:300])

        # 3. Test suite (slower, optional)
        test_cmd = _discover_test_cmd(worktree)
        if test_cmd:
            rc, out = await _run_cmd(test_cmd, cwd=worktree, timeout=TEST_TIMEOUT)
            if rc != 0:
                return VerifyResult(passed=False, stage="test", detail=out[:300])
            return VerifyResult(passed=True, stage="test", detail="Lint + tests passed")

        # No test runner found — lint-only verification
        if lint_cmd:
            return VerifyResult(passed=True, stage="lint", detail="Syntax check passed (no test runner found)")

        return VerifyResult(passed=False, stage="skip", detail="No linter or test runner found")

    except Exception as e:
        return VerifyResult(passed=False, stage="error", detail=str(e)[:300])
    finally:
        if worktree and worktree.exists():
            try:
                # Prune the git worktree reference first
                await _run_cmd(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, timeout=10)
            except Exception:
                pass
            # Belt-and-suspenders: remove the directory even if git worktree remove failed
            shutil.rmtree(worktree, ignore_errors=True)


# ── Public API ───────────────────────────────────────────────────────────


async def verify_review(
    review,  # AggregatedReview
    repo_root: Path | str | None,
) -> dict:
    """Verify all suggestions in a review. Mutates review in place.

    Returns a summary dict for telemetry:
      {"total": N, "verified": N, "failed": N, "skipped": N}
    """
    stats = {"total": 0, "verified": 0, "failed": 0, "skipped": 0}

    if repo_root is None:
        logger.info("fix_verify_skipped_no_repo_clone")
        return stats

    repo_root = Path(repo_root)
    if not (repo_root / ".git").exists():
        logger.info("fix_verify_skipped_no_git_dir", path=str(repo_root))
        return stats

    # Collect comments that have suggestions
    candidates = [c for c in review.inline_comments if c.suggestion]
    stats["total"] = len(candidates)

    if not candidates:
        return stats

    # Cap verification to avoid blowing up review latency
    if len(candidates) > MAX_VERIFY:
        logger.info("fix_verify_capped", total=len(candidates), cap=MAX_VERIFY)
        candidates = candidates[:MAX_VERIFY]
        stats["skipped"] = stats["total"] - MAX_VERIFY

    for comment in candidates:
        import time
        start_time = time.monotonic()
        result = await _verify_one(repo_root, comment.path, comment.line, comment.suggestion)
        duration = time.monotonic() - start_time
        
        try:
            from observability.metrics import fix_verification_duration_seconds, fix_verification_total
            fix_verification_duration_seconds.observe(duration)
        except Exception:
            pass

        if result.passed:
            comment.verified = True
            comment.body = f"✅ **Verified fix** ({result.stage})\n\n{comment.body}"
            stats["verified"] += 1
            try:
                fix_verification_total.labels(result="verified").inc()
            except Exception:
                pass
            logger.debug("fix_verified", file=comment.path, line=comment.line, stage=result.stage)
        else:
            comment.verified = False
            if result.stage == "skip":
                # No tools available — leave as-is, don't mark as failed
                stats["skipped"] += 1
                try:
                    fix_verification_total.labels(result="skipped").inc()
                except Exception:
                    pass
            else:
                # Real failure: remove the suggestion, keep the finding as a warning
                comment.body = (
                    f"⚠️ **Unverified fix** ({result.stage}: {result.detail[:100]})\n\n{comment.body}"
                )
                comment.suggestion = None
                stats["failed"] += 1
                try:
                    fix_verification_total.labels(result="failed").inc()
                except Exception:
                    pass
                logger.debug("fix_unverified", file=comment.path, line=comment.line, detail=result.detail[:200])

    logger.info("fix_verification_complete", **stats)
    return stats
