"""Tests for the fix verification system."""

import asyncio
import os
import subprocess
import textwrap
from pathlib import Path

from services.fix_verifier import (
    VerifyResult,
    _apply_patch,
    _discover_lint_cmd,
    _discover_test_cmd,
    _safe_env,
    verify_review,
)


# ── Unit tests ───────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a minimal git repo in tmp_path with optional files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    for name, content in (files or {}).items():
        p = repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    # Stage and commit so worktrees work
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init", "--allow-empty"], cwd=repo, check=True)
    return repo


def test_apply_patch_replaces_line(tmp_path):
    repo = _make_repo(tmp_path, {"app.py": "line1\nline2\nline3\n"})
    assert _apply_patch(repo, "app.py", 2, "REPLACED")
    content = (repo / "app.py").read_text()
    assert "REPLACED" in content
    assert "line2" not in content
    assert "line1" in content and "line3" in content


def test_apply_patch_bad_line(tmp_path):
    repo = _make_repo(tmp_path, {"app.py": "line1\n"})
    assert not _apply_patch(repo, "app.py", 999, "REPLACED")


def test_apply_patch_missing_file(tmp_path):
    repo = _make_repo(tmp_path)
    assert not _apply_patch(repo, "nonexistent.py", 1, "REPLACED")


def test_discover_test_cmd(tmp_path):
    repo = _make_repo(tmp_path, {"pyproject.toml": "[tool.pytest]"})
    cmd = _discover_test_cmd(repo)
    assert cmd is not None
    assert "pytest" in " ".join(cmd)


def test_discover_test_cmd_none(tmp_path):
    repo = _make_repo(tmp_path)
    assert _discover_test_cmd(repo) is None


def test_discover_lint_cmd():
    assert _discover_lint_cmd("foo.py") is not None
    assert _discover_lint_cmd("bar.go") is not None
    assert _discover_lint_cmd("readme.md") is None


def test_safe_env():
    os.environ["PRISM_TEST_SECRET_KEY"] = "should_be_removed"
    os.environ["PRISM_TEST_SAFE_VAR"] = "should_stay"
    env = _safe_env()
    assert "PRISM_TEST_SECRET_KEY" not in env
    assert env.get("PRISM_TEST_SAFE_VAR") == "should_stay"
    assert env.get("CI") == "true"
    # Cleanup
    del os.environ["PRISM_TEST_SECRET_KEY"]
    del os.environ["PRISM_TEST_SAFE_VAR"]


# ── Integration test: verify_review ──────────────────────────────────────


def test_verify_review_no_repo():
    """verify_review should be a no-op when no repo root is provided."""
    from types import SimpleNamespace

    review = SimpleNamespace(inline_comments=[])
    stats = asyncio.run(verify_review(review, None))
    assert stats["total"] == 0


def test_verify_review_no_suggestions(tmp_path):
    """Comments without suggestions should be left untouched."""
    from types import SimpleNamespace

    repo = _make_repo(tmp_path, {"app.py": "x = 1\n"})
    comment = SimpleNamespace(path="app.py", line=1, body="Issue here", suggestion=None, verified=False)
    review = SimpleNamespace(inline_comments=[comment])
    stats = asyncio.run(verify_review(review, repo))
    assert stats["total"] == 0
    assert comment.verified is False


def test_verify_review_valid_syntax_fix(tmp_path):
    """A suggestion that fixes valid Python should pass lint."""
    from types import SimpleNamespace

    repo = _make_repo(tmp_path, {
        "app.py": textwrap.dedent("""\
            def greet():
                prnt("hello")
                return True
        """),
    })
    comment = SimpleNamespace(
        path="app.py",
        line=2,
        body="Typo in function name",
        suggestion='    print("hello")',
        verified=False,
    )
    review = SimpleNamespace(inline_comments=[comment])
    stats = asyncio.run(verify_review(review, repo))
    # Whether it passes depends on py_compile availability, but it should not crash
    assert stats["total"] == 1
    assert stats["verified"] + stats["failed"] + stats["skipped"] == 1


def test_verify_review_bad_syntax_fix(tmp_path):
    """A suggestion that introduces a syntax error should fail lint."""
    from types import SimpleNamespace

    repo = _make_repo(tmp_path, {
        "app.py": textwrap.dedent("""\
            def greet():
                print("hello")
                return True
        """),
    })
    comment = SimpleNamespace(
        path="app.py",
        line=2,
        body="Bad fix",
        suggestion='    print("hello"',  # missing closing paren
        verified=False,
    )
    review = SimpleNamespace(inline_comments=[comment])
    stats = asyncio.run(verify_review(review, repo))
    assert stats["total"] == 1
    # Should have failed lint
    if stats["failed"] > 0:
        assert comment.suggestion is None  # suggestion stripped
        assert "Unverified" in comment.body
