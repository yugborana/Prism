"""
Prism Workspace Search — Regex/Literal Search Across Repository Files.

Adapted from px0's search.go: parallel file grepping with snippet generation,
binary file detection, file size caps, and glob filtering.

This service is the backend for the agent "grep" tool. Agents call it during
their Analyze step to ground their findings in actual code context beyond
the diff — e.g., "find all callers of this deprecated function" or "check
if this pattern is used elsewhere."

Design decisions:
  - Uses the WorkspaceIndex for file enumeration (respects .gitignore)
  - Reads files from disk (the cloned repo) — no network calls
  - Bounded concurrency via asyncio.Semaphore (CPU-bound work in threads)
  - Binary detection via null-byte scanning (same as px0/git)
  - Snippet generation: strips indentation, clips long lines, keeps match visible
  - Results capped at max_files × max_per_file to prevent token explosion
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import Any

import asyncio

from observability.logging import get_logger

logger = get_logger(__name__)

# Don't grep files larger than 2MB (binary blobs, minified JS, etc.)
_FILE_SIZE_CAP = 2 * 1024 * 1024

# Snippet formatting
_SNIP_LEAD = 32   # max runes before the match before we elide
_SNIP_KEEP = 16   # runes of lead-in kept when we do elide
_SNIP_MAX = 240   # cap on the whole snippet width

# Lines longer than this are skipped (minified bundles, data blobs). Also a
# cheap guard against pathological regex backtracking on huge lines.
_MAX_LINE_LEN = 4000

# Files are searched in path-ordered batches so results are deterministic
# (same query -> same output -> stable prompts) while still allowing early exit.
_SEARCH_BATCH = 64


@dataclass
class SearchMatch:
    """A single match within a file."""

    line: int
    """1-based line number."""

    pre: str
    """Text before the match on this line (trimmed)."""

    mid: str
    """The matched text itself."""

    post: str
    """Text after the match on this line (trimmed)."""

    is_definition: bool = False
    """True if the line looks like a function/class declaration."""


@dataclass
class FileSearchResult:
    """All matches found in a single file."""

    path: str
    """Workspace-relative path."""

    matches: list[SearchMatch] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete search results across the workspace."""

    results: list[FileSearchResult] = field(default_factory=list)
    """Per-file match groups, sorted by path."""

    total_matches: int = 0
    """Total number of matches across all files."""

    files_matched: int = 0
    """Number of files containing at least one match."""

    truncated: bool = False
    """True if results were capped."""

    query: str = ""
    """The original search query."""


@dataclass
class SearchOptions:
    """Configuration for a workspace search."""

    query: str
    """Search string or regex pattern."""

    is_regex: bool = False
    """If True, interpret query as a regular expression."""

    case_sensitive: bool = False
    """Case-sensitive search."""

    whole_word: bool = False
    """Match whole words only (\\b...\\b)."""

    glob: str = ""
    """Optional glob filter (e.g., '*.py', 'src/')."""

    max_files: int = 50
    """Maximum number of files to return matches from."""

    max_per_file: int = 20
    """Maximum matches per file."""

    context_lines: int = 0
    """Number of surrounding context lines (0 = match line only)."""


def _is_binary(data: bytes) -> bool:
    """Detect binary files by scanning for null bytes (same heuristic as git)."""
    # Check first 8KB for null bytes
    chunk = data[:8192]
    return b"\x00" in chunk


def _snip_match(line: str, start: int, end: int) -> SearchMatch:
    """Create a display-ready match snippet from a line and match offsets.

    Strips leading whitespace, clips long prefixes, and caps total length
    to keep results readable in agent prompts.
    """
    pre = line[:start]
    mid = line[start:end]
    post = line[end:]

    # Strip leading whitespace
    pre = pre.lstrip(" \t")

    # Elide long prefixes
    if len(pre) > _SNIP_LEAD:
        pre = "…" + pre[-_SNIP_KEEP:]

    # Cap match text
    if len(mid) > _SNIP_MAX:
        mid = mid[:_SNIP_MAX] + "…"

    # Cap post text
    budget = _SNIP_MAX - len(pre) - len(mid)
    if budget > 0 and len(post) > budget:
        post = post[:budget] + "…"
    elif budget <= 0:
        post = ""

    post = post.rstrip(" \t")

    return SearchMatch(line=0, pre=pre, mid=mid, post=post)


# Common declaration patterns per language extension for definition detection
_DECL_PATTERNS: dict[str, re.Pattern] = {}


def _get_decl_pattern(ext: str, query: str) -> re.Pattern | None:
    """Get a regex that detects if a line declares `query` as a symbol."""
    key = f"{ext}:{query}"
    if key in _DECL_PATTERNS:
        return _DECL_PATTERNS[key]
    if len(_DECL_PATTERNS) > 2048:
        _DECL_PATTERNS.clear()  # bound memory in long-running workers

    q = re.escape(query)
    js_prefix = r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    js = rf"{js_prefix}(?:function\s*\*?\s*|(?:const|let|var|class)\s+){q}\b"
    ts = rf"{js_prefix}(?:function\s*\*?\s*|(?:const|let|var|class|interface|type|enum)\s+){q}\b"
    patterns = {
        ".py": rf"^\s*(?:def|class|async\s+def)\s+{q}\b",
        ".js": js,
        ".mjs": js,
        ".cjs": js,
        ".jsx": js,
        ".ts": ts,
        ".tsx": ts,
        ".go": rf"^\s*(?:func|type|var|const)\s+(?:\([^)]*\)\s+)?{q}\b",
        ".rs": rf"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|type|const|static|mod)\s+{q}\b",
        ".java": rf"^\s*(?:public|private|protected|static|final|abstract|\s)*(?:class|interface|enum|void|int|String|boolean|long|double|float)\s+{q}\b",
        ".rb": rf"^\s*(?:def|class|module)\s+{q}\b",
        ".c": rf"^\s*(?:\w+\s+)+{q}\s*\(",
        ".h": rf"^\s*(?:\w+\s+)+{q}\s*\(",
        ".cpp": rf"^\s*(?:\w+\s+)+{q}\s*\(",
    }

    ext_lower = ext.lower()
    if ext_lower in patterns:
        try:
            pat = re.compile(patterns[ext_lower])
            _DECL_PATTERNS[key] = pat
            return pat
        except re.error:
            return None
    return None


class WorkspaceSearcher:
    """Searches files in a workspace using the WorkspaceIndex for enumeration.

    Usage::

        from services.workspace_index import build_workspace_index
        index = await build_workspace_index("/path/to/repo")
        searcher = WorkspaceSearcher(index)

        results = await searcher.search(SearchOptions(
            query="authenticate",
            glob="*.py",
            max_files=20,
        ))
    """

    def __init__(self, workspace_index: Any):
        """
        Args:
            workspace_index: A WorkspaceIndex instance (from workspace_index.py).
        """
        self._index = workspace_index
        self._root = workspace_index.root

    async def search(self, opts: SearchOptions) -> SearchResult:
        """Execute a search across all indexed files.

        Returns SearchResult with per-file matches, sorted by path.
        """
        query = opts.query.strip()
        if not query:
            return SearchResult(query=query)

        # Compile the search pattern
        compiled, literal = self._compile_query(opts)
        if compiled is None and literal is None:
            return SearchResult(query=query)

        # Get files to search, filtered by glob, in stable path order
        files = sorted(self._filter_files(opts.glob), key=lambda f: f.path)

        # Search in path-ordered batches with bounded concurrency. Batching
        # (rather than as_completed) keeps results deterministic: the first
        # max_files matching files *by path* win, not whichever finished first.
        sem = asyncio.Semaphore(16)
        results: list[FileSearchResult] = []
        truncated = False

        async def search_file(file_entry: Any) -> FileSearchResult | None:
            async with sem:
                return await asyncio.to_thread(
                    self._search_single_file,
                    file_entry,
                    compiled,
                    literal,
                    opts,
                )

        for i in range(0, len(files), _SEARCH_BATCH):
            batch = files[i : i + _SEARCH_BATCH]
            batch_results = await asyncio.gather(
                *(search_file(f) for f in batch), return_exceptions=True
            )
            for res in batch_results:
                if isinstance(res, BaseException) or not res or not res.matches:
                    continue
                if len(results) >= opts.max_files:
                    truncated = True
                    break
                results.append(res)
            if truncated:
                break
            if len(results) >= opts.max_files and i + _SEARCH_BATCH < len(files):
                # More files remain unsearched — there may be more matches.
                truncated = True
                break

        total_matches = sum(len(r.matches) for r in results)

        logger.info(
            "workspace_search_complete",
            query=query[:50],
            files_matched=len(results),
            total_matches=total_matches,
            truncated=truncated,
        )

        return SearchResult(
            results=results,
            total_matches=total_matches,
            files_matched=len(results),
            truncated=truncated,
            query=query,
        )

    def _compile_query(
        self, opts: SearchOptions
    ) -> tuple[re.Pattern | None, bytes | None]:
        """Compile the search query into a regex or literal needle."""
        try:
            if opts.is_regex or opts.whole_word:
                pat = opts.query
                if not opts.is_regex:
                    pat = re.escape(pat)
                if opts.whole_word:
                    pat = rf"\b(?:{pat})\b"

                flags = 0 if opts.case_sensitive else re.IGNORECASE
                return re.compile(pat, flags), None
            else:
                # Literal fast path
                needle = opts.query
                if not opts.case_sensitive:
                    needle = needle.lower()
                return None, needle.encode("utf-8", errors="replace")
        except re.error as e:
            logger.warning("search_regex_compile_failed", error=str(e))
            return None, None

    def _filter_files(self, glob: str) -> list[Any]:
        """Filter the workspace index files by an optional glob pattern.

        Supported forms:
          - ``*.py``               extension filter
          - ``tests/**/*.py``,     full fnmatch against the workspace path
            ``test_*.py``          (or the basename when no ``/`` present)
          - ``src/``               path prefix
          - ``auth``               case-insensitive path substring
        """
        files = self._index.files
        if not glob:
            return files

        if glob.startswith("*.") and not any(c in glob[2:] for c in "*?[/"):
            # Extension filter (fast path)
            ext = glob[1:]  # e.g., ".py"
            return [f for f in files if f.name.endswith(ext)]
        elif any(c in glob for c in "*?["):
            # Real glob. '**/' should also match zero directories.
            patterns = {glob}
            if "**/" in glob:
                patterns.add(glob.replace("**/", ""))
            if "/" in glob:
                return [f for f in files if any(fnmatch.fnmatchcase(f.path, p) for p in patterns)]
            return [f for f in files if any(fnmatch.fnmatchcase(f.name, p) for p in patterns)]
        elif "/" in glob:
            # Path prefix filter
            prefix = glob.rstrip("/")
            return [f for f in files if f.path == prefix or f.path.startswith(prefix + "/")]
        else:
            # Name contains filter
            glob_lower = glob.lower()
            return [f for f in files if glob_lower in f.path.lower()]

    def _search_single_file(
        self,
        file_entry: Any,
        compiled: re.Pattern | None,
        literal: bytes | None,
        opts: SearchOptions,
    ) -> FileSearchResult | None:
        """Search a single file on disk. Runs in a thread."""
        abs_path = os.path.join(self._root, file_entry.path.replace("/", os.sep))

        # Skip oversized files
        if file_entry.size > _FILE_SIZE_CAP:
            return None

        try:
            with open(abs_path, "rb") as f:
                data = f.read()
        except OSError:
            return None

        # Skip binary files
        if _is_binary(data):
            return None

        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return None

        # Fast whole-file reject for literal searches
        if literal is not None:
            needle = opts.query if opts.case_sensitive else opts.query.lower()
            haystack = text if opts.case_sensitive else text.lower()
            if needle not in haystack:
                return None

        # Get declaration pattern for this file type
        ext = os.path.splitext(file_entry.name)[1]
        decl_pattern = _get_decl_pattern(ext, opts.query) if not opts.is_regex else None

        matches: list[SearchMatch] = []
        lines = text.splitlines()  # handles \n, \r\n (Windows-authored files)

        for line_num, line in enumerate(lines, 1):
            if len(matches) >= opts.max_per_file:
                break
            if len(line) > _MAX_LINE_LEN:
                continue

            line_matches = self._find_matches_in_line(
                line, compiled, literal, opts.case_sensitive
            )

            for start, end in line_matches:
                m = _snip_match(line, start, end)
                m.line = line_num

                # Check if this line is a declaration
                if decl_pattern and decl_pattern.match(line.lstrip()):
                    m.is_definition = True

                matches.append(m)

                if len(matches) >= opts.max_per_file:
                    break

        if not matches:
            return None

        return FileSearchResult(path=file_entry.path, matches=matches)

    def _find_matches_in_line(
        self,
        line: str,
        compiled: re.Pattern | None,
        literal: bytes | None,
        case_sensitive: bool,
    ) -> list[tuple[int, int]]:
        """Find all match positions (start, end) in a single line."""
        if compiled is not None:
            return [(m.start(), m.end()) for m in compiled.finditer(line)]
        else:
            # Literal search
            assert literal is not None
            needle_str = literal.decode("utf-8", errors="replace")
            haystack = line if case_sensitive else line.lower()
            positions = []
            start = 0
            while True:
                idx = haystack.find(needle_str, start)
                if idx < 0:
                    break
                positions.append((idx, idx + len(needle_str)))
                start = idx + len(needle_str)
            return positions


def format_search_results_for_agent(result: SearchResult, max_chars: int = 8000) -> str:
    """Format search results into a text block suitable for agent prompts.

    Produces a compact, readable format like:
        ## Search: "authenticate" (12 matches in 4 files)
        ### src/auth.py
          L23: def authenticate(user, password):    [DEFINITION]
          L45:     if not authenticate(credentials):
        ### src/middleware.py
          L12:     token = authenticate(request.headers)
    """
    if not result.results:
        return f'## Search: "{result.query}" — No matches found.\n'

    lines: list[str] = []
    truncation_note = " (truncated)" if result.truncated else ""
    lines.append(
        f'## Search: "{result.query}" '
        f"({result.total_matches} matches in {result.files_matched} files{truncation_note})"
    )

    chars_used = len(lines[0])

    for file_result in result.results:
        if chars_used > max_chars:
            lines.append(f"\n... ({len(result.results) - result.results.index(file_result)} more files truncated)")
            break

        file_header = f"\n### {file_result.path}"
        lines.append(file_header)
        chars_used += len(file_header)

        for match in file_result.matches:
            def_tag = "  [DEFINITION]" if match.is_definition else ""
            match_line = f"  L{match.line}: {match.pre}{match.mid}{match.post}{def_tag}"
            if chars_used + len(match_line) > max_chars:
                lines.append("  ... (more matches truncated)")
                break
            lines.append(match_line)
            chars_used += len(match_line)

    return "\n".join(lines)
