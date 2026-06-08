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
        code_graphs = await asyncio.to_thread(retriever.get_code_graphs_by_files, changed_files)
        import_files = await asyncio.to_thread(retriever.get_import_files_by_files, changed_files)
        learnings = await asyncio.to_thread(retriever.get_related_learnings, limit=5)

        # Format everything into a single context string for the AI agents
        comprehensive_context = retriever.format_for_ai(code_graphs, import_files, learnings)

        # ── Cross-file context from repo symbol index (Cursor pattern) ──
        # Only query if the repo has been indexed (non-blocking: first PR
        # won't have an index, second PR will).
        cross_file_context = ""
        if state.has_repo_index:
            try:
                from services.symbol_retriever import SymbolRetriever

                symbol_retriever = SymbolRetriever()
                cross_file_context = await asyncio.to_thread(
                    symbol_retriever.format_cross_file_context,
                    changed_files,
                    state.repo_full_name,
                )
                logger.info(
                    "cross_file_context_fetched",
                    context_len=len(cross_file_context),
                )
            except Exception as xf_err:
                logger.warning("cross_file_context_failed", error=str(xf_err))
                cross_file_context = ""
        else:
            cross_file_context = "(Repo index building in background — cross-file context will be available on next PR)"

        logger.info(
            "context_fetched",
            graphs=len(code_graphs),
            imports=len(import_files),
            learnings=len(learnings),
        )

        # ── Static Analysis Pre-Processing Layer ───────────────────────
        # Runs tree-sitter over the diff's + lines to extract:
        #   1. OWASP anti-patterns (SQLi, XSS, hardcoded secrets, etc.)
        #   2. Structural metadata (function signatures, complexity, call graph)
        # Results are injected into each agent's prompt as structured context,
        # so the LLM validates what the static tool already found.
        static_analysis: dict = {}
        try:
            from services.static_analyzer import StaticAnalyzer

            diff_text = state.diff_data.get("full_diff", "")
            if diff_text:
                analyzer = StaticAnalyzer()
                static_analysis = await asyncio.to_thread(analyzer.analyze_diff, diff_text)
                logger.info(
                    "static_analysis_complete",
                    security_findings=static_analysis.get("security", {}).get("total_findings", 0),
                    functions_found=len(static_analysis.get("tree_sitter", {}).get("functions", [])),
                )
        except Exception as sa_err:
            logger.warning("static_analysis_failed", error=str(sa_err))
            static_analysis = {}

        return {
            "code_graphs": code_graphs,
            "import_files": import_files,
            "learnings": learnings,
            "comprehensive_context": comprehensive_context,
            "cross_file_context": cross_file_context,
            "static_analysis": static_analysis,
            "context_fetcher_status": AgentStatus.COMPLETED,
        }

    except Exception as e:
        logger.error("context_fetch_failed", error=str(e))
        return {
            "context_fetcher_status": AgentStatus.FAILED,
            "comprehensive_context": "",
            "cross_file_context": "",
            "static_analysis": {},
            "errors": [f"Context fetcher failed: {str(e)}"],
        }
