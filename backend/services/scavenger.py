"""
Prism Memory Scavenger.

Post-review resource reclamation. Runs in a ``finally`` so it executes even
when a review crashes. Every step is best-effort and isolated: a failure in
one never blocks the others, and never fails the review.

Reclaims:
  * PR-scoped Qdrant points (code_graphs / import_files) — otherwise every
    review leaves vectors behind forever.
  * Large strings held on ReviewState (diff, comprehensive context) once the
    final review is produced.
  * In-process garbage (gc.collect) — long-lived Celery workers otherwise
    grow steadily.
"""

from __future__ import annotations

import asyncio
import gc
import shutil

from observability.logging import get_logger

logger = get_logger(__name__)


async def scavenge_review(state, repo_name: str, pr_number: int) -> dict:
    """Release per-review resources. Returns a report of what was done."""
    report = {"qdrant_deleted": False, "state_trimmed": False, "gc_collected": 0}

    # 1. PR-scoped vectors
    try:
        from services.vector_indexer import VectorIndexer

        if repo_name and pr_number:
            report["qdrant_deleted"] = await VectorIndexer.delete_pr_points(repo_name, pr_number)
    except Exception as e:
        logger.warning("scavenge_qdrant_failed", error=str(e))

    # 2. Drop heavyweight state fields no longer needed after aggregation.
    #    Final review + reports are kept; raw inputs are not.
    try:
        if getattr(state, "final_review", None) is not None:
            state.diff_data = {}
            state.comprehensive_context = ""
            state.cross_file_context = ""
            state.file_context = ""
            state.per_file_contexts = {}
            state.code_graphs = []
            state.import_files = []
            report["state_trimmed"] = True
    except Exception as e:
        logger.warning("scavenge_state_failed", error=str(e))

    # 3. Clean up any leftover fix-verification worktrees.
    try:
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.gettempdir())
        for d in tmp.glob("prism-verify-*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                report.setdefault("worktrees_cleaned", 0)
                report["worktrees_cleaned"] = report.get("worktrees_cleaned", 0) + 1
    except Exception as e:
        logger.warning("scavenge_worktrees_failed", error=str(e))

    # 4. GC off the event loop (full collection can take a while).
    try:
        report["gc_collected"] = await asyncio.to_thread(gc.collect)
    except Exception as e:
        logger.warning("scavenge_gc_failed", error=str(e))

    logger.info("review_scavenged", repo=repo_name, pr=pr_number, **report)
    return report
