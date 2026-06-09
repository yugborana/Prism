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

# Files under this line count get full content; above this get compressed context
_FULL_FILE_THRESHOLD = 500

# Number of context lines to include around each diff hunk for large files
_SURROUNDING_LINES = 30


def _build_single_file_context(
    file_path: str,
    content: str,
    diff_text: str,
) -> str:
    """Build context string for a single file using compression strategy.

    - Small files (≤500 lines): include full numbered source
    - Large files (>500 lines): include diff hunks + 30 surrounding lines
    """
    from services.diff_parser import parse_diff_valid_lines

    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= _FULL_FILE_THRESHOLD:
        # Small file → include full numbered source
        parts = [f"### {file_path} ({total_lines} lines — full file)", "```"]
        for i, line in enumerate(lines, 1):
            parts.append(f"L{i:>4}: {line}")
        parts.append("```")
        return "\n".join(parts)

    # Large file → include only hunks + surrounding context
    diff_info = parse_diff_valid_lines(diff_text)
    file_info = diff_info.get(file_path)
    if not file_info or not file_info.hunks:
        return f"### {file_path} ({total_lines} lines — too large, no diff hunks available)"

    parts = [f"### {file_path} ({total_lines} lines — showing diff regions + {_SURROUNDING_LINES} surrounding lines)"]

    # Merge hunk ranges with surrounding context
    shown_ranges: list[tuple[int, int]] = []
    for hunk_start, hunk_end in file_info.hunks:
        range_start = max(1, hunk_start - _SURROUNDING_LINES)
        range_end = min(total_lines, hunk_end + _SURROUNDING_LINES)
        if shown_ranges and range_start <= shown_ranges[-1][1] + 1:
            shown_ranges[-1] = (shown_ranges[-1][0], range_end)
        else:
            shown_ranges.append((range_start, range_end))

    parts.append("```")
    for i, (start, end) in enumerate(shown_ranges):
        if i > 0:
            parts.append(f"  ... (lines {shown_ranges[i - 1][1] + 1}-{start - 1} omitted) ...")
        for line_num in range(start, end + 1):
            if line_num <= total_lines:
                parts.append(f"L{line_num:>4}: {lines[line_num - 1]}")
    parts.append("```")
    return "\n".join(parts)


def _build_per_file_contexts(
    file_contents: dict[str, str],
    diff_text: str,
) -> dict[str, str]:
    """Build per-file context dict for per-file review.

    Returns dict mapping file_path -> that file's formatted source context.
    """
    result: dict[str, str] = {}
    for file_path, content in file_contents.items():
        result[file_path] = _build_single_file_context(file_path, content, diff_text)
    return result


def _build_file_context(
    file_contents: dict[str, str],
    diff_text: str,
) -> str:
    """Build combined file context string (all files together).

    Used as fallback when per-file contexts aren't available.
    """
    if not file_contents:
        return "(Full file context not available)"

    per_file = _build_per_file_contexts(file_contents, diff_text)
    if not per_file:
        return "(Full file context not available)"

    return "\n\n".join(per_file.values())


async def context_fetcher_agent(state: ReviewState) -> dict[str, Any]:
    """
    Fetch vector DB context for the changed files.

    Queries Qdrant for:
    - Code graphs (function/class structure)
    - Import files (source code of dependencies)
    - Past learnings (previous review feedback)

    Also fetches full file contents from GitHub for accurate line references.

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

        # ── Full File Content Fetching (compression strategy) ──────────
        # Fetches source code from GitHub for each changed file so agents
        # can see exact line numbers. Small files get full content, large
        # files get diff hunks + surrounding context only.
        # Builds BOTH aggregated file_context AND per_file_contexts.
        file_context = ""
        per_file_contexts: dict[str, str] = {}
        try:
            if state.installation_id and state.head_sha and changed_files:
                from services.github_service import GitHubService

                gh = GitHubService(installation_id=state.installation_id)
                file_contents = await gh.fetch_file_contents(
                    repo_full_name=state.repo_full_name,
                    file_paths=changed_files[:10],  # Cap at 10 files
                    commit_sha=state.head_sha,
                )
                diff_text = state.diff_data.get("full_diff", "")
                per_file_contexts = _build_per_file_contexts(file_contents, diff_text)
                file_context = "\n\n".join(per_file_contexts.values()) if per_file_contexts else ""
                logger.info(
                    "file_context_built",
                    files_fetched=len(file_contents),
                    per_file_count=len(per_file_contexts),
                    context_len=len(file_context),
                )
        except Exception as fc_err:
            logger.warning("file_context_fetch_failed", error=str(fc_err))
            file_context = ""
            per_file_contexts = {}

        return {
            "code_graphs": code_graphs,
            "import_files": import_files,
            "learnings": learnings,
            "comprehensive_context": comprehensive_context,
            "cross_file_context": cross_file_context,
            "file_context": file_context,
            "per_file_contexts": per_file_contexts,
            "static_analysis": static_analysis,
            "context_fetcher_status": AgentStatus.COMPLETED,
        }

    except Exception as e:
        logger.error("context_fetch_failed", error=str(e))
        return {
            "context_fetcher_status": AgentStatus.FAILED,
            "comprehensive_context": "",
            "cross_file_context": "",
            "file_context": "",
            "per_file_contexts": {},
            "static_analysis": {},
            "errors": [f"Context fetcher failed: {str(e)}"],
        }
