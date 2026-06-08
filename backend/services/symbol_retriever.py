"""
Prism Symbol Retriever — Cross-File Context Queries from Qdrant.

Queries the ``repo_chunks`` collection to answer:
  1. Who calls this function? (caller resolution)
  2. What files import this module? (dependency tracking)
  3. Does this pattern exist elsewhere? (consistency check)

Formats results into a human-readable context string injected into
each agent's prompt as ``{cross_file_context}``.
"""

from __future__ import annotations

from typing import Any

from observability.logging import get_logger
from utils.qdrant_client import get_qdrant_client

logger = get_logger(__name__)


def _build_filter(conditions: list[dict[str, Any]]) -> Any:
    """Build a Qdrant filter from a list of field conditions.

    Uses proper SDK models when available, falls back to dicts for NoOp client.
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        must = []
        for cond in conditions:
            key = cond["key"]
            if "value" in cond:
                must.append(
                    FieldCondition(key=key, match=MatchValue(value=cond["value"]))
                )
            elif "any" in cond:
                must.append(
                    FieldCondition(key=key, match=MatchAny(any=cond["any"]))
                )
        return Filter(must=must)
    except ImportError:
        return {"must": conditions}


class SymbolRetriever:
    """Retrieves cross-file context from the ``repo_chunks`` Qdrant collection."""

    def __init__(self):
        self._client = get_qdrant_client()
        self._llm_client: Any | None = None

    def _get_llm(self) -> Any:
        if self._llm_client is None:
            from utils.llm_factory import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    def find_callers(
        self,
        function_name: str,
        repo_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """Find chunks in the repo whose ``symbols_called`` contains the given function.

        This answers: "Who calls this function?"
        """
        try:
            points, _ = self._client.scroll(
                collection_name="repo_chunks",
                scroll_filter=_build_filter([
                    {"key": "repo_name", "value": repo_name},
                    {"key": "symbols_called", "any": [function_name]},
                ]),
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in points]
        except Exception as e:
            logger.debug("find_callers_failed", func=function_name, error=str(e))
            return []

    def find_dependents(
        self,
        module_path: str,
        repo_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """Find chunks that import from the given module.

        ``module_path`` can be a dotted path like "services.billing" or a
        relative path like "services/billing".
        """
        # Normalize: "services/billing.py" → "services.billing"
        normalized = module_path.replace("/", ".").replace(".py", "")

        try:
            points, _ = self._client.scroll(
                collection_name="repo_chunks",
                scroll_filter=_build_filter([
                    {"key": "repo_name", "value": repo_name},
                    {"key": "imports", "any": [normalized, module_path]},
                ]),
                limit=limit,
                with_payload=True,
            )
            return [p.payload for p in points]
        except Exception as e:
            logger.debug("find_dependents_failed", module=module_path, error=str(e))
            return []

    def find_similar_patterns(
        self,
        code_snippet: str,
        repo_name: str,
        limit: int = 5,
    ) -> list[dict]:
        """Semantic search for similar code patterns in the repo.

        Use case: "does this auth pattern exist elsewhere?"
        Embeds the snippet and finds nearest neighbors.
        """
        try:
            import asyncio

            llm = self._get_llm()
            # Embed the query snippet
            loop = asyncio.new_event_loop()
            try:
                embedding = loop.run_until_complete(llm.embed(code_snippet[:500]))
            finally:
                loop.close()

            results = self._client.search(
                collection_name="repo_chunks",
                query_vector=embedding,
                query_filter=_build_filter([
                    {"key": "repo_name", "value": repo_name},
                ]),
                limit=limit,
                with_payload=True,
            )
            return [r.payload for r in results]
        except Exception as e:
            logger.debug("find_similar_failed", error=str(e))
            return []

    def get_blast_radius(
        self,
        changed_files: list[str],
        changed_symbols: list[str],
        repo_name: str,
    ) -> dict[str, Any]:
        """Compute the blast radius of changes.

        For each changed file/symbol, find:
        - Callers of changed functions
        - Files that import the changed module
        """
        radius: dict[str, Any] = {}

        for file_path in changed_files:
            file_info: dict[str, Any] = {
                "callers": [],
                "dependents": [],
            }

            # Find files that import this module
            dependents = self.find_dependents(file_path, repo_name, limit=15)
            for dep in dependents:
                if dep.get("file_path") != file_path:
                    file_info["dependents"].append({
                        "file": dep.get("file_path", ""),
                        "symbol": dep.get("symbol_name", ""),
                        "line": dep.get("start_line", 0),
                    })

            radius[file_path] = file_info

        # Find callers of changed functions
        for symbol in changed_symbols:
            callers = self.find_callers(symbol, repo_name, limit=10)
            for caller in callers:
                caller_file = caller.get("file_path", "")
                if caller_file in radius:
                    radius[caller_file]["callers"].append({
                        "file": caller_file,
                        "caller": caller.get("symbol_name", ""),
                        "line": caller.get("start_line", 0),
                        "called_function": symbol,
                    })
                else:
                    # Caller is in a file not in changed_files
                    if symbol not in radius:
                        radius[symbol] = {"callers": [], "dependents": []}
                    radius[symbol]["callers"].append({
                        "file": caller_file,
                        "caller": caller.get("symbol_name", ""),
                        "line": caller.get("start_line", 0),
                    })

        return radius

    def format_cross_file_context(
        self,
        changed_files: list[str],
        repo_name: str,
    ) -> str:
        """Build a human-readable cross-file context string for agent prompts.

        Extracts changed function names from the diff-indexed chunks, then
        queries for their callers and dependents.
        """
        parts: list[str] = []

        # Step 1: Find what symbols are defined in the changed files
        changed_symbols: list[str] = []
        for file_path in changed_files:
            try:
                points, _ = self._client.scroll(
                    collection_name="repo_chunks",
                    scroll_filter=_build_filter([
                        {"key": "repo_name", "value": repo_name},
                        {"key": "file_path", "value": file_path},
                    ]),
                    limit=50,
                    with_payload=True,
                )
                for p in points:
                    for sym in p.payload.get("symbols_defined", []):
                        if sym:
                            changed_symbols.append(sym)
            except Exception:
                pass

        if not changed_symbols and not changed_files:
            return ""

        # Step 2: For each changed file, find who depends on it
        for file_path in changed_files[:10]:  # Cap at 10 files
            dependents = self.find_dependents(file_path, repo_name, limit=8)
            external_deps = [
                d for d in dependents
                if d.get("file_path") != file_path
            ]

            if external_deps:
                parts.append(f"\n### {file_path} (CHANGED)")
                parts.append("**Imported by:**")
                for dep in external_deps[:5]:
                    dep_file = dep.get("file_path", "?")
                    parts.append(f"  - `{dep_file}`")

        # Step 3: For each changed function, find its callers
        seen_callers: set[str] = set()  # Dedupe
        for symbol in changed_symbols[:15]:  # Cap at 15 symbols
            callers = self.find_callers(symbol, repo_name, limit=8)
            external_callers = [
                c for c in callers
                if c.get("file_path") not in changed_files
            ]

            if external_callers:
                parts.append(f"\n**`{symbol}()` is called by:**")
                for caller in external_callers[:5]:
                    caller_file = caller.get("file_path", "?")
                    caller_sym = caller.get("symbol_name", "?")
                    caller_line = caller.get("start_line", "?")
                    key = f"{caller_file}:{caller_sym}"
                    if key not in seen_callers:
                        seen_callers.add(key)
                        parts.append(
                            f"  - `{caller_file}:{caller_line}` in `{caller_sym}()`"
                        )

        if not parts:
            return ""

        header = "## Cross-File Dependencies\n"
        return header + "\n".join(parts)
