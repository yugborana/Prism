"""
Prism Vector Retriever — Queries Qdrant for Repo Context.

Retrieves:
1. Code graphs — function/class structure of changed files
2. Import files — source code of dependencies
3. Learnings — past review feedback for the same repo

Used by the ContextFetcher agent to build comprehensive context
before the specialist agents run.
"""

from __future__ import annotations

from typing import Any

from observability.logging import get_logger
from utils.qdrant_client import get_qdrant_client

logger = get_logger(__name__)


def _build_file_filter(file_path: str) -> Any:
    """Build a Qdrant Filter for matching a file_path payload field.

    Returns a proper ``models.Filter`` when the SDK is available,
    or a plain dict when running against the NoOp client.
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        return Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_path),
                )
            ]
        )
    except ImportError:
        # Fallback for NoOp client — never actually used for real queries
        return {"must": [{"key": "file_path", "match": {"value": file_path}}]}


class VectorRetriever:
    """Retrieves stored code context from Qdrant vector collections."""

    def __init__(self):
        self._client = get_qdrant_client()

    def get_code_graphs_by_files(self, file_paths: list[str]) -> list[dict]:
        """Fetch code structure graphs for the changed files."""
        results = []
        for file_path in file_paths:
            try:
                points, _ = self._client.scroll(
                    collection_name="code_graphs",
                    scroll_filter=_build_file_filter(file_path),
                    limit=1,
                    with_payload=True,
                )
                if points:
                    results.append(points[0].payload)
            except Exception as e:
                logger.debug("code_graph_fetch_failed", file=file_path, error=str(e))
        return results

    def get_import_files_by_files(self, file_paths: list[str]) -> list[dict]:
        """Fetch source code of imported dependencies."""
        results = []
        for file_path in file_paths:
            try:
                points, _ = self._client.scroll(
                    collection_name="import_files",
                    scroll_filter=_build_file_filter(file_path),
                    limit=1,
                    with_payload=True,
                )
                if points:
                    results.append(points[0].payload)
            except Exception as e:
                logger.debug("import_file_fetch_failed", file=file_path, error=str(e))
        return results

    def get_related_learnings(self, limit: int = 5) -> list[dict]:
        """Fetch past review learnings (feedback loops)."""
        try:
            points, _ = self._client.scroll(
                collection_name="learnings",
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in points]
        except Exception as e:
            logger.debug("learnings_fetch_failed", error=str(e))
            return []

    def format_for_ai(
        self,
        code_graphs: list[dict],
        import_files: list[dict],
        learnings: list[dict],
    ) -> str:
        """Format all retrieved context into a single string for the LLM agents."""
        parts: list[str] = []

        if learnings:
            parts.append("## Past Review Learnings:")
            for learning in learnings:
                parts.append(f"- Commit: {learning.get('commit_message', 'N/A')}")
                parts.append(f"  Review: {learning.get('bot_comment', 'N/A')[:150]}")
                if learning.get("user_feedback"):
                    parts.append(f"  Feedback: {learning.get('user_feedback', '')[:150]}")

        if code_graphs:
            parts.append("\n## Code Structure:")
            for g in code_graphs:
                parts.append(f"- {g.get('file_path', 'N/A')}")
                parts.append(f"  Functions: {', '.join(g.get('functions', []))}")
                parts.append(f"  Classes: {', '.join(g.get('classes', []))}")

        if import_files:
            parts.append("\n## Dependency Source Code:")
            for f in import_files:
                parts.append(f"- {f.get('file_path', 'N/A')}")
                src = f.get("source_code", "")[:500]
                parts.append(f"  ```\n{src}\n  ```")

        return "\n".join(parts) if parts else ""
