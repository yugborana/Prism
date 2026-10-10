"""
Eval Repo Setup — Clone, Index, and Cleanup for eval harness runs.

During eval, Prism needs:
  1. A local repo clone — so agent tools (grep, read_file, find_related) work
  2. A Qdrant index   — so ContextFetcher gets code_graphs, import_files,
                         and SymbolRetriever gets cross-file context
  3. Redis available   — so cross-agent memory and diff caching work

This module provides:
  - setup_repo_for_eval(): Clone + index a repo (full production pipeline)
  - cleanup_repo():        Remove the clone after scoring
  - check_infra():         Verify Docker services (Qdrant, Redis) are up

Authentication:
  - Uses GITHUB_TOKEN env var (PAT) for public repos
  - All ReviewBench and Martian Bench repos are public

Requirements:
  - `git` must be on PATH
  - `docker compose -f docker-compose.eval.yml up -d` must have been run
  - GITHUB_TOKEN env var recommended (avoids GitHub API rate limits)
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from utils.config import settings


# ── Infrastructure Check ─────────────────────────────────────────────────

async def check_infra() -> dict[str, bool]:
    """Check that Qdrant and Redis are reachable.

    Returns {"qdrant": bool, "redis": bool}.
    """
    status = {"qdrant": False, "redis": False}

    # Check Redis
    try:
        from utils.connections import get_redis
        redis = get_redis()
        if redis:
            await redis.ping()
            status["redis"] = True
    except Exception:
        pass

    # Check Qdrant
    try:
        from utils.qdrant_client import get_qdrant_client
        client = get_qdrant_client()
        # get_qdrant_client returns a NoOp client if unreachable;
        # try an actual operation
        collections = await asyncio.to_thread(client.get_collections)
        if collections is not None:
            status["qdrant"] = True
    except Exception:
        pass

    return status


# ── Clone ─────────────────────────────────────────────────────────────────

async def clone_for_eval(
    repo_name: str,
    base_branch: str = "main",
) -> Path | None:
    """Shallow-clone a repo so agent tools (grep, read_file) can work.

    Returns the clone path, or None if cloning fails.
    The clone is placed in settings.temp_repo_dir so that
    ``build_toolkit_for_review()`` picks it up automatically.
    """
    base_dir = Path(settings.temp_repo_dir)
    safe_name = repo_name.replace("/", "__")
    repo_dir = base_dir / safe_name

    # Already cloned from a previous eval run? Skip.
    if (repo_dir / ".git").exists():
        print(f"  [clone] Reusing existing clone at {repo_dir}")
        return repo_dir

    # Build clone URL (PAT or unauthenticated)
    token = settings.github_token or os.environ.get("GITHUB_TOKEN", "")
    if token:
        clone_url = f"https://x-access-token:{token}@github.com/{repo_name}.git"
    else:
        clone_url = f"https://github.com/{repo_name}.git"

    # Ensure parent exists, remove any partial clone
    base_dir.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    print(f"  [clone] Cloning {repo_name} (branch={base_branch}) ...")

    process = await asyncio.create_subprocess_exec(
        "git", "clone",
        "--depth", "1",
        "--single-branch",
        "--branch", base_branch,
        "--quiet",
        clone_url,
        str(repo_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        err = stderr.decode().strip()
        # Common failure: branch name wrong (some repos use 'master')
        if "not found" in err.lower() or "remote branch" in err.lower():
            fallback = "master" if base_branch != "master" else "develop"
            print(f"  [clone] Branch '{base_branch}' not found, trying '{fallback}'...")
            return await clone_for_eval(repo_name, base_branch=fallback)
        print(f"  [clone] FAILED: {err[:200]}")
        return None

    print(f"  [clone] OK ({repo_dir})")
    return repo_dir


# ── Index ─────────────────────────────────────────────────────────────────

async def index_repo_for_eval(repo_name: str, base_branch: str = "main") -> bool:
    """Run the full IndexManager pipeline so Qdrant has repo_chunks.

    This gives ContextFetcher and SymbolRetriever the same data they'd have
    in production after the first PR. Requires Qdrant + Redis to be running.

    Returns True on success, False on failure (eval continues without index).
    """
    try:
        from services.index_manager import IndexManager

        manager = IndexManager()
        stats = await manager.build_full_index(
            repo_name=repo_name,
            base_branch=base_branch,
            installation_id=0,  # eval uses PAT, not GitHub App
        )

        if stats.skipped_reason:
            print(f"  [index] Skipped: {stats.skipped_reason}")
            return False

        print(
            f"  [index] OK — {stats.total_files} files, "
            f"{stats.total_chunks} chunks, "
            f"{stats.chunks_embedded} embedded, "
            f"{stats.chunks_cached} cached "
            f"({stats.duration_seconds:.1f}s)"
        )
        return True

    except Exception as e:
        print(f"  [index] FAILED: {str(e)[:200]}")
        return False


# ── Combined Setup ────────────────────────────────────────────────────────

async def setup_repo_for_eval(
    repo_name: str,
    base_branch: str = "main",
    skip_index: bool = False,
) -> Path | None:
    """Full repo setup: clone + index.

    Args:
        repo_name: e.g. "grafana/grafana"
        base_branch: branch to clone
        skip_index: if True, only clone (no Qdrant indexing)

    Returns the clone path, or None if cloning fails.
    """
    clone_path = await clone_for_eval(repo_name, base_branch)
    if not clone_path:
        return None

    if not skip_index:
        indexed = await index_repo_for_eval(repo_name, base_branch)
        if not indexed:
            print(f"  [setup] Indexing failed — running with clone-only (grep/read_file still work)")

    return clone_path


# ── Cleanup (full flush after each PR) ────────────────────────────────────

async def cleanup_repo_async(repo_name: str) -> None:
    """Flush ALL eval artifacts for a repo: clone, Qdrant vectors, caches.

    This mirrors what production does — no data carries over between PRs.
    """
    # 1. Delete repo_chunks from Qdrant (the full index)
    try:
        from services.index_manager import IndexManager
        manager = IndexManager()
        await manager._delete_all_repo_chunks(repo_name)
        print(f"  [flush] Qdrant repo_chunks deleted for {repo_name}")
    except Exception as e:
        print(f"  [flush] Qdrant cleanup skipped: {str(e)[:100]}")

    # 2. Delete PR-scoped vectors (code_graphs, import_files)
    try:
        from services.vector_indexer import VectorIndexer
        await VectorIndexer.delete_pr_points(repo_name, 0)  # 0 = all PRs
    except Exception:
        pass  # Non-critical

    # 3. Clear the toolkit cache so next PR builds fresh
    try:
        from agents.tools import clear_toolkit_cache
        clear_toolkit_cache()
    except Exception:
        pass

    # 4. Delete SimHash/Merkle metadata from Redis
    try:
        from services.simhash import SimHashIndex
        simhash = SimHashIndex()
        await simhash.delete_index_meta(repo_name, installation_id=0)
        await simhash.close()
    except Exception:
        pass

    # 5. Remove the clone from disk
    base_dir = Path(settings.temp_repo_dir)
    safe_name = repo_name.replace("/", "__")
    repo_dir = base_dir / safe_name
    if repo_dir.exists():
        import os
        import stat
        def remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(repo_dir, onerror=remove_readonly)
        print(f"  [flush] Clone removed: {repo_dir}")


def cleanup_repo(repo_name: str) -> None:
    """Synchronous wrapper for cleanup_repo_async (used in finally blocks)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # We're inside an async context — schedule as a task
        loop.create_task(cleanup_repo_async(repo_name))
    except RuntimeError:
        # No running loop — run synchronously
        asyncio.run(cleanup_repo_async(repo_name))


def cleanup_all_eval_repos() -> None:
    """Remove ALL clones in the temp_repos dir (end-of-eval cleanup)."""
    base_dir = Path(settings.temp_repo_dir)
    if base_dir.exists():
        count = sum(1 for d in base_dir.iterdir() if d.is_dir())
        shutil.rmtree(base_dir, ignore_errors=True)
        print(f"  [cleanup] Removed {count} clones from {base_dir}")

