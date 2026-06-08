"""
Prism Vector Indexer — Populates Qdrant with PR Code Context.

Parses the diff to extract file-level code structure (functions, classes,
imports) and indexes them into Qdrant so the ContextFetcher can retrieve
relevant context for the review agents.

Collections populated:
  - code_graphs: function/class structure per changed file
  - import_files: raw source snippets for dependency tracking
"""

from __future__ import annotations

import re
import uuid

from observability.logging import get_logger
from utils.qdrant_client import get_qdrant_client

# Deterministic namespace for generating reproducible Qdrant point IDs.
# Same (repo + file + PR) always produces the same UUID, so re-running
# /prism-review on the same commit overwrites instead of duplicating.
_QDRANT_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

logger = get_logger(__name__)


class VectorIndexer:
    """Indexes PR diff content into Qdrant for agent context retrieval."""

    def __init__(self):
        self._client = get_qdrant_client()
        # Reuse a single LLMClient for all embedding calls instead of
        # creating a new one per file (avoids re-initializing the SDK
        # and its HTTP connection pool on every call).
        from utils.llm_factory import LLMClient

        self._llm_client = LLMClient()

    async def index_pr_diff(
        self,
        diff: str,
        changed_files: list[str],
        repo_name: str,
        pr_number: int,
    ) -> dict[str, int]:
        """
        Parse the PR diff and index code structure into Qdrant.

        Returns dict with counts of indexed items per collection.
        """
        indexed = {"code_graphs": 0, "import_files": 0}

        # Parse the diff into per-file hunks
        file_diffs = self._split_diff_by_file(diff)

        for file_path, file_diff in file_diffs.items():
            # Only index files that are in the changed_files list
            if not any(file_path.endswith(cf) or cf.endswith(file_path) for cf in changed_files):
                continue

            try:
                # Extract code structure from the diff
                code_graph = self._extract_code_graph(file_path, file_diff)
                if code_graph:
                    await self._index_code_graph(code_graph, repo_name, pr_number)
                    indexed["code_graphs"] += 1

                # Index the raw diff as an import_file entry
                await self._index_import_file(file_path, file_diff, repo_name, pr_number)
                indexed["import_files"] += 1

            except Exception as e:
                logger.warning("file_index_failed", file=file_path, error=str(e))

        logger.info(
            "pr_diff_indexed",
            repo=repo_name,
            pr=pr_number,
            code_graphs=indexed["code_graphs"],
            import_files=indexed["import_files"],
        )
        return indexed

    def _split_diff_by_file(self, diff: str) -> dict[str, str]:
        """Split a unified diff into per-file sections."""
        files: dict[str, str] = {}
        current_file = None
        current_lines: list[str] = []

        for line in diff.split("\n"):
            # Detect file boundaries: "diff --git a/path b/path"
            if line.startswith("diff --git"):
                if current_file and current_lines:
                    files[current_file] = "\n".join(current_lines)
                # Extract file path from "diff --git a/foo b/foo"
                match = re.search(r"b/(.+)$", line)
                current_file = match.group(1) if match else None
                current_lines = [line]
            elif current_file:
                current_lines.append(line)

        # Don't forget the last file
        if current_file and current_lines:
            files[current_file] = "\n".join(current_lines)

        return files

    def _extract_code_graph(self, file_path: str, file_diff: str) -> dict | None:
        """
        Extract function/class/import structure from the added lines in a diff.

        DESIGN NOTE:
        This is a lightweight regex-based parser rather than a full tree-sitter
        AST parser (like simple_ast_parser.py). Since the Celery worker task only
        receives transient unified diff hunks (and does not check out or clone the
        entire repository locally to disk), a traditional tree-sitter parser cannot
        reliably parse these isolated, syntactically incomplete fragments. Using
        this regex-based parser provides robust parsing of added lines.
        """
        functions = []
        classes = []
        imports = []

        # Only look at added lines (lines starting with +, but not +++ header)
        added_lines = [
            line[1:]  # Strip the leading '+'
            for line in file_diff.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        ]

        for line in added_lines:
            stripped = line.strip()

            # Python patterns
            if file_path.endswith(".py"):
                # Function definitions
                match = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", stripped)
                if match:
                    functions.append(match.group(1))
                # Class definitions
                match = re.match(r"class\s+(\w+)[\s(:]", stripped)
                if match:
                    classes.append(match.group(1))
                # Imports
                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append(stripped)

            # JavaScript/TypeScript patterns
            elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
                # Functions
                match = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", stripped)
                if match:
                    functions.append(match.group(1))
                # Arrow functions or constants assigned to functions
                match = re.match(r"(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", stripped)
                if match:
                    functions.append(match.group(1))
                # Class/Interface/Type definitions
                match = re.match(r"(?:export\s+)?(?:class|interface|type)\s+(\w+)", stripped)
                if match:
                    classes.append(match.group(1))
                if stripped.startswith("import "):
                    imports.append(stripped)

            # Go patterns
            elif file_path.endswith(".go"):
                # Functions and methods
                match = re.match(r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", stripped)
                if match:
                    functions.append(match.group(1))
                # Struct / Interface types
                match = re.match(r"type\s+(\w+)\s+(?:struct|interface)", stripped)
                if match:
                    classes.append(match.group(1))

            # Rust patterns
            elif file_path.endswith(".rs"):
                # Functions
                match = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", stripped)
                if match:
                    functions.append(match.group(1))
                # Struct / Trait / Impl blocks
                match = re.match(r"(?:pub\s+)?(?:struct|trait|impl)\s+(\w+)", stripped)
                if match:
                    classes.append(match.group(1))

        if not functions and not classes and not imports:
            return None

        return {
            "file_path": file_path,
            "functions": list(set(functions)),
            "classes": list(set(classes)),
            "imports": list(set(imports)),
            "node_count": len(functions) + len(classes),
        }

    async def _index_code_graph(self, code_graph: dict, repo_name: str, pr_number: int) -> None:
        """Embed and upsert a code graph into the code_graphs collection."""
        from qdrant_client.models import PointStruct

        # Build a text representation for embedding
        text = (
            f"File: {code_graph['file_path']}\n"
            f"Functions: {', '.join(code_graph['functions'])}\n"
            f"Classes: {', '.join(code_graph['classes'])}\n"
            f"Imports: {'; '.join(code_graph['imports'][:10])}"
        )

        embedding = await self._embed_text(text)
        if not embedding:
            return

        point = PointStruct(
            id=str(uuid.uuid5(_QDRANT_NS, f"{repo_name}:{code_graph['file_path']}:cg:{pr_number}")),
            vector=embedding,
            payload={
                **code_graph,
                "repo_name": repo_name,
                "pr_number": pr_number,
            },
        )

        import asyncio

        await asyncio.to_thread(self._client.upsert, collection_name="code_graphs", points=[point])

    async def _index_import_file(self, file_path: str, file_diff: str, repo_name: str, pr_number: int) -> None:
        """Embed and upsert a file's diff content into the import_files collection."""
        from qdrant_client.models import PointStruct

        # Use only added lines as the source context (max 2000 chars)
        added_lines = [
            line[1:] for line in file_diff.split("\n") if line.startswith("+") and not line.startswith("+++")
        ]
        source_code = "\n".join(added_lines)[:2000]

        if not source_code.strip():
            return

        text = f"File: {file_path}\n{source_code[:500]}"
        embedding = await self._embed_text(text)
        if not embedding:
            return

        point = PointStruct(
            id=str(uuid.uuid5(_QDRANT_NS, f"{repo_name}:{file_path}:imp:{pr_number}")),
            vector=embedding,
            payload={
                "file_path": file_path,
                "source_code": source_code,
                "repo_name": repo_name,
                "pr_number": pr_number,
            },
        )

        import asyncio

        await asyncio.to_thread(self._client.upsert, collection_name="import_files", points=[point])

    async def _embed_text(self, text: str) -> list[float] | None:
        """Generate embedding using the shared LLMClient."""
        try:
            return await self._llm_client.embed(text)
        except Exception as e:
            logger.warning("indexer_embedding_failed", error=str(e))
            return None
