"""
Prism Context Fetcher Agent.

This is NOT an LLM agent — it's a data-fetching step that queries Qdrant
for code graphs, import files, and past learnings, then formats them
as context for the downstream review agents.
"""

import asyncio
from typing import Any

from agents.schemas import AgentStatus, ReviewState
from observability.logging import get_logger

logger = get_logger(__name__)


async def context_fetcher_agent(state: ReviewState) -> dict[str, Any]:
    """
    Fetch vector DB context for the changed files.

    Queries Qdrant for:
    - Code graphs (function/class structure)
    - Import files (source code of dependencies)
    - Past learnings (previous review feedback)

    Returns a dict of state updates.
    """
    try:
        from services.vector_retriever import VectorRetriever

        changed_files = state.changed_files
        retriever = VectorRetriever()

        # Run Qdrant queries in threads (they're blocking SDK calls)
        code_graphs = await asyncio.to_thread(
            retriever.get_code_graphs_by_files, changed_files
        )
        import_files = await asyncio.to_thread(
            retriever.get_import_files_by_files, changed_files
        )
        learnings = await asyncio.to_thread(
            retriever.get_related_learnings, limit=5
        )

        # Format everything into a single context string for the AI agents
        comprehensive_context = retriever.format_for_ai(
            code_graphs, import_files, learnings
        )

        logger.info(
            "context_fetched",
            graphs=len(code_graphs),
            imports=len(import_files),
            learnings=len(learnings),
        )

        return {
            "code_graphs": code_graphs,
            "import_files": import_files,
            "learnings": learnings,
            "comprehensive_context": comprehensive_context,
            "context_fetcher_status": AgentStatus.COMPLETED,
        }

    except Exception as e:
        logger.error("context_fetch_failed", error=str(e))
        return {
            "context_fetcher_status": AgentStatus.FAILED,
            "comprehensive_context": "",
            "errors": [f"Context fetcher failed: {str(e)}"],
        }
