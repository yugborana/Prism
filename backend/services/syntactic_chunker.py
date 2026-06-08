"""
Prism Syntactic Chunker — AST-Aware Code Splitting.

Inspired by Cursor's indexing approach: files are split into semantically
meaningful chunks using tree-sitter, not arbitrary byte ranges.

A 200-line function becomes 3-4 chunks (signature, validation, logic, error
handling), each independently embeddable. This enables precise retrieval:
a query about "payment errors" returns the 10-line error handling block,
not the entire 200-line function.

Chunking strategy:
  - Functions ≤ 50 lines → 1 chunk
  - Functions > 50 lines → split at nested block boundaries (if/for/try/with)
  - Classes → header + each method is a separate chunk
  - Import blocks → 1 chunk per contiguous import group
  - Top-level code → grouped into chunks of ≤ 30 lines
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from observability.logging import get_logger
from services.simple_ast_parser import LANGUAGE_MAP, SimpleASTParser

logger = get_logger(__name__)

# Max lines for a single chunk before we try to split it
MAX_CHUNK_LINES = 50
# Max lines for top-level (non-function/class) code chunks
MAX_TOPLEVEL_CHUNK_LINES = 30


@dataclass
class CodeChunk:
    """A semantically meaningful unit of code from a source file."""

    file_path: str  # Relative path: "services/billing.py"
    chunk_index: int  # 0, 1, 2, ... within the file
    chunk_type: str  # "function", "method", "class_header", "imports", "top_level"
    content: str  # The actual source code
    content_hash: str  # SHA-256 of content — used for embedding cache key
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive
    symbol_name: str | None  # "process_payment" for functions, "PaymentService" for classes
    symbols_defined: list[str] = field(default_factory=list)  # Functions/classes defined
    symbols_called: list[str] = field(default_factory=list)  # Functions called from this chunk
    imports: list[str] = field(default_factory=list)  # Modules imported

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_embedding_text(self) -> str:
        """Generate text optimized for embedding.

        Includes metadata (file path, type, symbol name) so the embedding
        captures both semantics AND location.
        """
        parts = [f"File: {self.file_path}"]
        if self.symbol_name:
            parts.append(f"Symbol: {self.symbol_name}")
        parts.append(f"Type: {self.chunk_type}")
        if self.symbols_defined:
            parts.append(f"Defines: {', '.join(self.symbols_defined)}")
        if self.symbols_called:
            parts.append(f"Calls: {', '.join(self.symbols_called[:15])}")
        if self.imports:
            parts.append(f"Imports: {', '.join(self.imports[:10])}")
        parts.append(f"Code:\n{self.content[:1500]}")  # Cap content for embedding
        return "\n".join(parts)

    def to_payload(self) -> dict[str, Any]:
        """Qdrant point payload (metadata stored alongside the vector)."""
        return {
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "content": self.content[:3000],  # Cap stored content
            "content_hash": self.content_hash,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_name": self.symbol_name or "",
            "symbols_defined": self.symbols_defined,
            "symbols_called": self.symbols_called,
            "imports": self.imports,
        }


class SyntacticChunker:
    """Splits source files into AST-aware chunks using tree-sitter.

    Reuses the existing ``SimpleASTParser`` infrastructure and tree-sitter
    language grammars already installed (Python, JS, TS, Go, Rust).
    """

    def __init__(self):
        self._parsers: dict[str, SimpleASTParser] = {}

    def _get_parser(self, language: str) -> SimpleASTParser | None:
        """Lazy-init a parser for the given language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = SimpleASTParser(language)
            except ValueError:
                return None
        return self._parsers[language]

    def chunk_file(self, file_path: str, source_code: str) -> list[CodeChunk]:
        """Split a single source file into semantically meaningful chunks.

        Args:
            file_path: Relative path (e.g., "services/billing.py")
            source_code: The full source code string

        Returns:
            Ordered list of CodeChunks covering the entire file.
        """
        # Determine language from file extension
        ext = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        language = LANGUAGE_MAP.get(ext)
        if not language:
            # Unsupported extension → return the whole file as one chunk
            return [self._make_chunk(file_path, 0, "unknown", source_code, 1)]

        parser = self._get_parser(language)
        if not parser:
            return [self._make_chunk(file_path, 0, "unknown", source_code, 1)]

        # Parse with tree-sitter
        source_bytes = source_code.encode("utf-8")
        tree = parser.parser.parse(source_bytes)
        lines = source_code.split("\n")

        # Extract top-level nodes and classify them
        chunks: list[CodeChunk] = []
        covered_lines: set[int] = set()  # 0-indexed lines covered by a chunk

        # Phase 1: Extract functions and classes (the big structural elements)
        self._extract_structural_chunks(tree.root_node, source_code, lines, file_path, language, chunks, covered_lines)

        # Phase 2: Extract import blocks
        self._extract_import_chunks(tree.root_node, source_code, lines, file_path, language, chunks, covered_lines)

        # Phase 3: Collect remaining top-level code into chunks
        self._extract_toplevel_chunks(lines, file_path, chunks, covered_lines)

        # Sort by start_line and re-index
        chunks.sort(key=lambda c: c.start_line)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def chunk_file_from_disk(self, repo_path: Path, relative_path: str) -> list[CodeChunk]:
        """Read a file from disk and chunk it."""
        full_path = repo_path / relative_path
        try:
            source_code = full_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as e:
            logger.warning("chunk_file_read_failed", file=relative_path, error=str(e))
            return []
        return self.chunk_file(relative_path, source_code)

    # ── Phase 1: Structural Elements ──────────────────────────────────────

    def _extract_structural_chunks(
        self,
        root_node: Any,
        source: str,
        lines: list[str],
        file_path: str,
        language: str,
        chunks: list[CodeChunk],
        covered: set[int],
    ) -> None:
        """Extract function and class chunks from top-level AST nodes."""
        func_types = self._get_function_node_types(language)
        class_types = self._get_class_node_types(language)

        for child in root_node.children:
            node_type = child.type

            # ── Function/Method ────────────────────────────────────────
            if node_type in func_types:
                self._chunk_function(child, source, lines, file_path, language, chunks, covered)

            # ── Class/Struct ───────────────────────────────────────────
            elif node_type in class_types:
                self._chunk_class(child, source, lines, file_path, language, chunks, covered)

            # ── Decorated definitions (Python) ─────────────────────────
            elif node_type == "decorated_definition" and language == "python":
                # The actual definition is inside the decorator wrapper
                for sub in child.children:
                    if sub.type in func_types:
                        self._chunk_function(
                            sub, source, lines, file_path, language, chunks, covered, decorator_node=child
                        )
                    elif sub.type in class_types:
                        self._chunk_class(
                            sub, source, lines, file_path, language, chunks, covered, decorator_node=child
                        )

    def _chunk_function(
        self,
        node: Any,
        source: str,
        lines: list[str],
        file_path: str,
        language: str,
        chunks: list[CodeChunk],
        covered: set[int],
        decorator_node: Any | None = None,
    ) -> None:
        """Create chunk(s) for a function node. Splits large functions."""
        # Use decorator node's span if it exists
        start_node = decorator_node or node
        start_line = start_node.start_point[0]  # 0-indexed
        end_line = node.end_point[0]  # 0-indexed
        func_lines = end_line - start_line + 1

        # Extract function name
        name_node = node.child_by_field_name("name")
        func_name = source[name_node.start_byte : name_node.end_byte] if name_node else None

        # Extract the function source
        func_source = "\n".join(lines[start_line : end_line + 1])

        # Extract call graph info
        calls = self._extract_calls(node, source, language)

        if func_lines <= MAX_CHUNK_LINES:
            # Small function → 1 chunk
            chunk = self._make_chunk(
                file_path,
                0,
                "function",
                func_source,
                start_line + 1,
                symbol_name=func_name,
                symbols_defined=[func_name] if func_name else [],
                symbols_called=calls,
            )
            chunks.append(chunk)
        else:
            # Large function → split at block boundaries
            sub_chunks = self._split_large_function(
                node, source, lines, file_path, func_name, calls, start_line, end_line
            )
            chunks.extend(sub_chunks)

        # Mark lines as covered
        for i in range(start_line, end_line + 1):
            covered.add(i)

    def _split_large_function(
        self,
        node: Any,
        source: str,
        lines: list[str],
        file_path: str,
        func_name: str | None,
        calls: list[str],
        start_line: int,
        end_line: int,
    ) -> list[CodeChunk]:
        """Split a function > 50 lines at nested block boundaries."""
        # Find block boundaries within the function body
        body_node = node.child_by_field_name("body") or node
        split_points: list[int] = [start_line]  # Always include function start

        for child in body_node.children:
            child_start = child.start_point[0]
            # Split at major block statements
            if child.type in {
                "if_statement",
                "for_statement",
                "while_statement",
                "try_statement",
                "with_statement",
                "match_statement",
                "for_in_statement",
                "switch_statement",
            }:
                if child_start > split_points[-1] + 10:  # Min 10 lines between splits
                    split_points.append(child_start)

        split_points.append(end_line + 1)  # End sentinel

        result: list[CodeChunk] = []
        for i in range(len(split_points) - 1):
            chunk_start = split_points[i]
            chunk_end = min(split_points[i + 1] - 1, end_line)
            if chunk_end < chunk_start:
                continue

            chunk_source = "\n".join(lines[chunk_start : chunk_end + 1])
            suffix = f"_part{i + 1}" if len(split_points) > 2 else ""
            chunk = self._make_chunk(
                file_path,
                0,
                "function",
                chunk_source,
                chunk_start + 1,
                symbol_name=f"{func_name}{suffix}" if func_name else None,
                symbols_defined=[func_name] if func_name and i == 0 else [],
                symbols_called=calls if i == 0 else [],
            )
            result.append(chunk)

        return result

    def _chunk_class(
        self,
        node: Any,
        source: str,
        lines: list[str],
        file_path: str,
        language: str,
        chunks: list[CodeChunk],
        covered: set[int],
        decorator_node: Any | None = None,
    ) -> None:
        """Create chunks for a class: 1 for the header, 1 per method."""
        start_node = decorator_node or node
        start_line = start_node.start_point[0]
        end_line = node.end_point[0]

        name_node = node.child_by_field_name("name")
        class_name = source[name_node.start_byte : name_node.end_byte] if name_node else None

        # Find the class body
        body_node = node.child_by_field_name("body")
        func_types = self._get_function_node_types(language)

        method_names: list[str] = []
        method_ranges: list[tuple[int, int]] = []

        if body_node:
            for child in body_node.children:
                if child.type in func_types:
                    m_name_node = child.child_by_field_name("name")
                    if m_name_node:
                        m_name = source[m_name_node.start_byte : m_name_node.end_byte]
                        method_names.append(m_name)
                        method_ranges.append((child.start_point[0], child.end_point[0]))

        # Class header chunk (everything before the first method, or the whole class if no methods)
        if method_ranges:
            header_end = method_ranges[0][0] - 1
        else:
            header_end = end_line

        header_source = "\n".join(lines[start_line : max(header_end + 1, start_line + 1)])
        header_chunk = self._make_chunk(
            file_path,
            0,
            "class_header",
            header_source,
            start_line + 1,
            symbol_name=class_name,
            symbols_defined=[class_name] if class_name else [],
        )
        chunks.append(header_chunk)

        # Each method → separate chunk (reuse function chunking logic)
        if body_node:
            for child in body_node.children:
                if child.type in func_types:
                    self._chunk_function(child, source, lines, file_path, language, chunks, covered)

        # Mark all class lines as covered
        for i in range(start_line, end_line + 1):
            covered.add(i)

    # ── Phase 2: Imports ──────────────────────────────────────────────────

    def _extract_import_chunks(
        self,
        root_node: Any,
        source: str,
        lines: list[str],
        file_path: str,
        language: str,
        chunks: list[CodeChunk],
        covered: set[int],
    ) -> None:
        """Collect contiguous import statements into a single chunk."""
        import_types = self._get_import_node_types(language)
        import_lines: list[int] = []  # 0-indexed line numbers

        for child in root_node.children:
            if child.type in import_types:
                for ln in range(child.start_point[0], child.end_point[0] + 1):
                    import_lines.append(ln)

        if not import_lines:
            return

        # Group contiguous import lines
        import_lines.sort()
        groups: list[list[int]] = []
        current_group: list[int] = [import_lines[0]]

        for ln in import_lines[1:]:
            if ln <= current_group[-1] + 2:  # Allow 1 blank line gap
                current_group.append(ln)
            else:
                groups.append(current_group)
                current_group = [ln]
        groups.append(current_group)

        for group in groups:
            start = group[0]
            end = group[-1]
            import_source = "\n".join(lines[start : end + 1])
            # Extract imported module names
            imported = self._extract_import_names(root_node, source, language, start, end)

            chunk = self._make_chunk(
                file_path,
                0,
                "imports",
                import_source,
                start + 1,
                imports=imported,
            )
            chunks.append(chunk)

            for i in range(start, end + 1):
                covered.add(i)

    # ── Phase 3: Top-Level Code ───────────────────────────────────────────

    def _extract_toplevel_chunks(
        self,
        lines: list[str],
        file_path: str,
        chunks: list[CodeChunk],
        covered: set[int],
    ) -> None:
        """Group remaining uncovered non-blank lines into top-level chunks."""
        uncovered_ranges: list[tuple[int, int]] = []
        current_start: int | None = None

        for i, line in enumerate(lines):
            if i not in covered and line.strip():
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    uncovered_ranges.append((current_start, i - 1))
                    current_start = None

        if current_start is not None:
            uncovered_ranges.append((current_start, len(lines) - 1))

        # Split ranges that exceed MAX_TOPLEVEL_CHUNK_LINES
        for start, end in uncovered_ranges:
            for chunk_start in range(start, end + 1, MAX_TOPLEVEL_CHUNK_LINES):
                chunk_end = min(chunk_start + MAX_TOPLEVEL_CHUNK_LINES - 1, end)
                source = "\n".join(lines[chunk_start : chunk_end + 1])
                if source.strip():
                    chunk = self._make_chunk(
                        file_path,
                        0,
                        "top_level",
                        source,
                        chunk_start + 1,
                    )
                    chunks.append(chunk)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        file_path: str,
        index: int,
        chunk_type: str,
        content: str,
        start_line: int,
        symbol_name: str | None = None,
        symbols_defined: list[str] | None = None,
        symbols_called: list[str] | None = None,
        imports: list[str] | None = None,
    ) -> CodeChunk:
        """Create a CodeChunk with computed hash."""
        end_line = start_line + content.count("\n")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return CodeChunk(
            file_path=file_path,
            chunk_index=index,
            chunk_type=chunk_type,
            content=content,
            content_hash=content_hash,
            start_line=start_line,
            end_line=end_line,
            symbol_name=symbol_name,
            symbols_defined=symbols_defined or [],
            symbols_called=symbols_called or [],
            imports=imports or [],
        )

    def _extract_calls(self, node: Any, source: str, language: str) -> list[str]:
        """Extract function call names from within a node."""
        calls: list[str] = []
        call_types = {"call", "call_expression"}

        def walk(n: Any) -> None:
            if n.type in call_types:
                func_node = n.child_by_field_name("function")
                if func_node:
                    name = source[func_node.start_byte : func_node.end_byte]
                    # Simplify: "self.method" → "method", "module.func" → "func"
                    clean_name = name.rsplit(".", 1)[-1].strip()
                    if clean_name and clean_name.replace("_", "").isalnum():
                        calls.append(clean_name)
            for child in n.children:
                walk(child)

        walk(node)
        return list(dict.fromkeys(calls))  # Dedupe preserving order

    def _extract_import_names(self, root: Any, source: str, language: str, start: int, end: int) -> list[str]:
        """Extract imported module names from import statements in a line range."""
        names: list[str] = []

        def walk(n: Any) -> None:
            if n.start_point[0] < start or n.start_point[0] > end:
                return

            if language == "python":
                if n.type in {"import_from_statement", "import_statement"}:
                    for child in n.children:
                        if child.type == "dotted_name":
                            names.append(source[child.start_byte : child.end_byte])
                            break

            elif language in {"javascript", "typescript", "tsx"}:
                if n.type == "import_statement":
                    for child in n.children:
                        if child.type == "string":
                            text = source[child.start_byte : child.end_byte].strip("\"'")
                            names.append(text)

            elif language == "go":
                if n.type == "import_spec":
                    for child in n.children:
                        if child.type == "interpreted_string_literal":
                            text = source[child.start_byte : child.end_byte].strip('"')
                            names.append(text)

            elif language == "rust":
                if n.type == "use_declaration":
                    arg = n.child_by_field_name("argument")
                    if arg:
                        names.append(source[arg.start_byte : arg.end_byte])

            for child in n.children:
                walk(child)

        walk(root)
        return list(dict.fromkeys(names))

    @staticmethod
    def _get_function_node_types(language: str) -> set[str]:
        """tree-sitter node types that represent function definitions."""
        if language == "python":
            return {"function_definition"}
        elif language in {"javascript", "typescript", "tsx"}:
            return {"function_declaration", "method_definition"}
        elif language == "go":
            return {"function_declaration", "method_declaration"}
        elif language == "rust":
            return {"function_item"}
        return set()

    @staticmethod
    def _get_class_node_types(language: str) -> set[str]:
        if language == "python":
            return {"class_definition"}
        elif language in {"javascript", "typescript", "tsx"}:
            return {"class_declaration"}
        elif language == "go":
            return {"type_declaration"}
        elif language == "rust":
            return {"struct_item", "impl_item", "trait_item"}
        return set()

    @staticmethod
    def _get_import_node_types(language: str) -> set[str]:
        if language == "python":
            return {"import_statement", "import_from_statement"}
        elif language in {"javascript", "typescript", "tsx"}:
            return {"import_statement"}
        elif language == "go":
            return {"import_declaration"}
        elif language == "rust":
            return {"use_declaration"}
        return set()
