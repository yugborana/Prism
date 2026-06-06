"""
Prism Learning Service — Feedback Loop for Continuous Improvement.

After every review, stores the review→user feedback pair as a "learning"
in Qdrant. This allows future reviews of similar code patterns to benefit
from past human corrections.

Flow:
1. Agent reviews PR and posts findings
2. Human developer responds (approves, requests changes, dismisses)
3. LearningService indexes {review_comment, user_feedback, code_context}
4. Future ContextFetcher queries retrieve relevant learnings
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from observability.logging import get_logger
from utils.config import settings
from utils.qdrant_client import get_qdrant_client

logger = get_logger(__name__)


class LearningService:
    """Indexes review feedback for the continuous learning pipeline."""

    def __init__(self):
        self._client = get_qdrant_client()

    async def index_learning(
        self,
        repo_name: str,
        pr_number: int,
        review_comment: str,
        user_feedback: str | None = None,
        code_context: str = "",
        commit_sha: str = "",
    ) -> str | None:
        """
        Store a learning from a completed review.

        Args:
            repo_name: Full repo name (e.g., "owner/repo")
            pr_number: PR number
            review_comment: What the AI reviewer said
            user_feedback: How the human responded (if any)
            code_context: Relevant code snippet
            commit_sha: Commit hash for traceability

        Returns:
            Point ID if indexed successfully, None otherwise.
        """
        try:
            # Generate embedding for the learning
            embedding = await self._embed_text(
                f"{review_comment} {user_feedback or ''} {code_context[:500]}"
            )
            if not embedding:
                return None

            from qdrant_client.models import PointStruct

            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": "learning",
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "commit_sha": commit_sha,
                    "bot_comment": review_comment[:2000],
                    "user_feedback": user_feedback[:2000] if user_feedback else None,
                    "has_user_feedback": user_feedback is not None,
                    "code_context": code_context[:1000],
                },
            )

            self._client.upsert(collection_name="learnings", points=[point])
            logger.info(
                "learning_indexed",
                repo=repo_name,
                pr=pr_number,
                has_feedback=user_feedback is not None,
            )
            return point_id

        except Exception as e:
            logger.warning("learning_index_failed", error=str(e))
            return None

    async def _embed_text(self, text: str) -> list[float] | None:
        """Generate embedding using Ollama's all-minilm via the LLMClient."""
        try:
            from utils.llm_factory import LLMClient

            client = LLMClient()
            return await client.embed(text[:8000])
        except Exception as e:
            logger.warning("embedding_failed", error=str(e))
            return None

    async def search_similar_learnings(
        self, query: str, repo_name: str | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Semantic search for similar past learnings.
        Optionally filtered by repo.
        """
        try:
            embedding = await self._embed_text(query)
            if not embedding:
                return []

            search_filter = None
            if repo_name:
                search_filter = {
                    "must": [{"key": "repo_name", "match": {"value": repo_name}}]
                }

            results = self._client.search(
                collection_name="learnings",
                query_vector=embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )
            return [r.payload for r in results]

        except Exception as e:
            logger.warning("learning_search_failed", error=str(e))
            return []
