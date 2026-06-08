"""
Prism Indexing Celery Tasks — Background Repo Indexing Pipeline.

These tasks run on the ``prism.index`` queue, separate from the main review
queue. They handle:
  - ``build_repo_index``: Full index for a never-before-seen repo
  - ``refresh_repo_index``: Incremental update for an already-indexed repo
  - ``cleanup_stale_clones``: Hourly cleanup of old repo clones on disk
  - ``cleanup_stale_indexes``: Daily cleanup of repo indexes not queried in 30 days
"""

from __future__ import annotations

import asyncio

from observability.logging import get_logger
from workers.celery_app import celery_app
from observability.metrics import track_indexing_task

logger = get_logger(__name__)


@celery_app.task(
    name="build_repo_index",
    queue="prism.index",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def build_repo_index(
    repo_name: str,
    base_branch: str,
    installation_id: int,
) -> dict:
    """Full indexing pipeline for a never-before-seen repo.

    Checks SimHash for reusable indexes first (same installation only).
    Runs in background — first PR reviews without cross-file context,
    second PR gets the full index.
    """
    logger.info(
        "index_task_started",
        task="build_repo_index",
        repo=repo_name,
        branch=base_branch,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from services.index_manager import IndexManager

        manager = IndexManager()
        with track_indexing_task("full"):
            stats = loop.run_until_complete(manager.build_full_index(repo_name, base_branch, installation_id))
        loop.run_until_complete(manager.close())

        result = {
            "repo": stats.repo_name,
            "files": stats.total_files,
            "chunks": stats.total_chunks,
            "embedded": stats.chunks_embedded,
            "cached": stats.chunks_cached,
            "copied": stats.chunks_copied,
            "duration": f"{stats.duration_seconds:.1f}s",
            "simhash_source": stats.simhash_source,
            "skipped": stats.skipped_reason,
        }

        logger.info("index_task_complete", task="build_repo_index", **result)
        return result

    except Exception as e:
        logger.error(
            "index_task_failed",
            task="build_repo_index",
            repo=repo_name,
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        loop.close()


@celery_app.task(
    name="refresh_repo_index",
    queue="prism.index",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def refresh_repo_index(
    repo_name: str,
    base_branch: str,
    installation_id: int,
) -> dict:
    """Incremental index update: Merkle diff → re-chunk changed files only."""
    logger.info(
        "index_task_started",
        task="refresh_repo_index",
        repo=repo_name,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from services.index_manager import IndexManager

        manager = IndexManager()
        with track_indexing_task("refresh"):
            stats = loop.run_until_complete(manager.refresh_index(repo_name, base_branch, installation_id))
        loop.run_until_complete(manager.close())

        result = {
            "repo": stats.repo_name,
            "files_changed": stats.files_changed,
            "chunks": stats.total_chunks,
            "embedded": stats.chunks_embedded,
            "cached": stats.chunks_cached,
            "duration": f"{stats.duration_seconds:.1f}s",
        }

        logger.info("index_task_complete", task="refresh_repo_index", **result)
        return result

    except Exception as e:
        logger.error(
            "index_task_failed",
            task="refresh_repo_index",
            repo=repo_name,
            error=str(e),
            exc_info=True,
        )
        raise
    finally:
        loop.close()


@celery_app.task(name="cleanup_stale_clones", queue="prism.index")
def cleanup_stale_clones() -> dict:
    """Hourly: delete repo clones older than 24 hours from disk."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from services.repo_cloner import RepoCloner

        cloner = RepoCloner()
        deleted = loop.run_until_complete(cloner.cleanup_stale(max_age_hours=24))
        return {"deleted_clones": deleted}
    finally:
        loop.close()


@celery_app.task(name="cleanup_stale_indexes", queue="prism.index")
def cleanup_stale_indexes() -> dict:
    """Daily: delete repo indexes not updated in 30 days from Qdrant."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from services.index_manager import IndexManager

        manager = IndexManager()
        with track_indexing_task("cleanup"):
            cleaned = loop.run_until_complete(manager.cleanup_stale_indexes(max_age_days=30))
        loop.run_until_complete(manager.close())
        return {"repos_cleaned": cleaned}
    finally:
        loop.close()
