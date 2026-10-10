"""
Prism Agent Tools — Capabilities Agents Can Invoke During Review.

This module defines the tools available to review agents. Instead of being
limited to what's in the diff and pre-fetched context, agents can now
actively explore the repository:

  - grep_repo:       Search for patterns across the entire codebase
  - read_file:       Read a specific file's contents (beyond the diff)
  - find_related:    Find test files, config siblings, implementations
  - list_directory:  Browse the repo structure

These tools are NOT LLM function-calling tools (we don't do multi-turn
tool loops). Instead, they are invoked BEFORE the reasoning chain runs,
as a "pre-analysis" step: the Analyze step's prompt includes a
"grounding context" section that was gathered by running tools based
on simple heuristics from the diff.

This design avoids the latency and cost of multi-turn LLM tool loops
while still grounding agents in real code context.

Architecture:
  - AgentToolkit: Holds references to WorkspaceIndex + WorkspaceSearcher,
    plus a per-review search cache shared by all agents.
  - ToolResult:   Dataclass wrapping each tool's output
  - extract_diff_symbols(): Classifies symbols touched by a diff into
    removed / signature-changed / modified / added.
  - gather_grounding_context(): The main entry point — analyzes the diff
    to decide which probes to run, runs them under a time budget, and
    returns formatted text that gets injected into the agent's prompt.

Important: the local clone tracks the PR's *base* branch, not the PR head.
Search results therefore reflect the code BEFORE the PR. The probes are
designed around that:
  - A symbol removed by the PR but still referenced in files the PR does
    NOT touch is a strong breaking-change signal.
  - References inside files the PR also modifies are stale (their new
    content is in the diff), so they are labelled or excluded.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from observability.logging import get_logger

logger = get_logger(__name__)

# Cap total grounding context to avoid blowing up the prompt
_MAX_GROUNDING_CHARS = 12000

# Max file read size (don't dump a 50KB file into the prompt)
_MAX_FILE_READ = 4000

# Wall-clock budget for gathering grounding context per file per agent.
# On timeout, whatever was collected so far is still used.
_GROUNDING_TIMEOUT_S = 10.0

# Per-toolkit search result cache (shared by the 4 parallel agents)
_SEARCH_CACHE_SIZE = 256

# Options used for symbol lookups. Kept identical across probes so the
# usage / test-coverage / duplicate probes all hit the same cache entry.
_SYMBOL_MAX_FILES = 30
_SYMBOL_MAX_PER_FILE = 3

# Names too generic to grep meaningfully (would match half the repo)
_GENERIC_NAMES = frozenset({
    "run", "get", "set", "put", "post", "delete", "update", "save", "load",
    "open", "close", "start", "stop", "reset", "init", "main", "test", "setup",
    "teardown", "execute", "process", "handle", "handler", "call", "apply",
    "build", "create", "parse", "render", "validate", "read", "write", "send",
    "to_dict", "from_dict", "to_json", "from_json", "dispatch", "wrapper",
    "inner", "helper", "callback", "default", "value", "data", "config",
    "self", "cls", "new", "next", "value_of", "toString", "equals", "hashCode",
})

# Matches common test file / directory naming conventions
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec|specs)(/|$)"
    r"|(^|/)test_[^/]*$"
    r"|_test\.[A-Za-z]+$"
    r"|\.(test|spec)\.[A-Za-z]+$",
    re.IGNORECASE,
)

# Declaration patterns (applied to a diff line with its +/- prefix removed,
# or to the function context git appends to @@ hunk headers).
_DEF_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)"),                                    # Python
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"),  # JS/TS
    re.compile(r"^\s*(?:export\s+)?(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
               r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"),                                       # JS arrow fn
    re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?(?:public\s+|final\s+)*class\s+([A-Za-z_]\w*)"),  # classes
    re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_]\w*)"),                              # TS/Go-ish
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),                                # Go
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"),             # Rust
    re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)"),       # Rust
]

_HUNK_HEADER_RE = re.compile(r"^@@[^@]*@@\s?(.*)$")


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ToolResult:
    """Result from a single tool invocation."""

    tool: str
    """Tool name (e.g., 'grep_repo')."""

    query: str
    """The input that triggered this tool."""

    output: str
    """Formatted text output."""

    match_count: int = 0
    """Number of matches/results found."""


@dataclass
class DiffSymbols:
    """Symbols touched by a diff, classified by how they changed."""

    removed: list[str] = field(default_factory=list)
    """Declared on '-' lines and not re-declared on '+' lines (deleted/renamed)."""

    signature_changed: list[str] = field(default_factory=list)
    """Declaration line was both removed and re-added (signature edit)."""

    modified: list[str] = field(default_factory=list)
    """Enclosing symbol of an edited hunk (from @@ header) — body changes."""

    added: list[str] = field(default_factory=list)
    """Newly declared on '+' lines only."""

    def usage_targets(self) -> list[str]:
        """Symbols whose existing callers are worth showing, most important first."""
        return _unique(self.signature_changed + self.modified)


@dataclass(frozen=True)
class _Probe:
    """A role-specific search, only run when the diff's added lines match `trigger`."""

    trigger: str
    query: str
    title: str
    note: str = ""
    glob: str = ""
    max_files: int = 6
    max_per_file: int = 2


@dataclass(order=True)
class _Section:
    priority: int
    seq: int
    text: str = field(compare=False)


# ═══════════════════════════════════════════════════════════════════════════
# Toolkit
# ═══════════════════════════════════════════════════════════════════════════


class AgentToolkit:
    """Collection of tools available to review agents.

    Initialized once per review with the repo's workspace index and searcher,
    and shared by all agents. Tools are read-only — they read the cloned repo
    on disk. Search results are memoized so the 4 parallel agents (and the
    per-file runs within each agent) don't repeat identical greps.
    """

    def __init__(
        self,
        repo_path: str,
        workspace_index: Any | None = None,
        workspace_searcher: Any | None = None,
    ):
        self._repo_path = repo_path
        self._index = workspace_index
        self._searcher = workspace_searcher
        self._search_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._inflight: dict[tuple, asyncio.Task] = {}
        self.stats: dict[str, int] = {"searches": 0, "cache_hits": 0}

    @property
    def available(self) -> bool:
        """True if the toolkit has a valid index to work with."""
        return self._index is not None

    # ── Core search (memoized, de-duplicated) ─────────────────────────────

    async def search(self, opts: Any) -> Any | None:
        """Run a workspace search with memoization and in-flight de-duplication.

        Concurrent identical searches (e.g. 4 agents looking up the same
        changed symbol at the same moment) share one underlying scan.
        Returns a ``SearchResult`` or None when search is unavailable.
        """
        if not self._searcher:
            return None

        key = (
            opts.query, opts.is_regex, opts.case_sensitive, opts.whole_word,
            opts.glob, opts.max_files, opts.max_per_file,
        )

        cached = self._search_cache.get(key)
        if cached is not None:
            self._search_cache.move_to_end(key)
            self.stats["cache_hits"] += 1
            return cached

        loop = asyncio.get_running_loop()
        task = self._inflight.get(key)
        if task is None or task.get_loop() is not loop or task.cancelled():
            task = loop.create_task(self._searcher.search(opts))
            self._inflight[key] = task
            self.stats["searches"] += 1
        else:
            self.stats["cache_hits"] += 1

        try:
            # shield: if this caller times out, other waiters still get the result
            result = await asyncio.shield(task)
        finally:
            if task.done() and self._inflight.get(key) is task:
                self._inflight.pop(key, None)

        self._search_cache[key] = result
        while len(self._search_cache) > _SEARCH_CACHE_SIZE:
            self._search_cache.popitem(last=False)
        return result

    async def symbol_search(self, symbol: str) -> Any | None:
        """Whole-word, case-sensitive lookup of a symbol (cache-friendly options)."""
        from services.workspace_search import SearchOptions

        return await self.search(SearchOptions(
            query=symbol,
            is_regex=False,
            case_sensitive=True,
            whole_word=True,
            max_files=_SYMBOL_MAX_FILES,
            max_per_file=_SYMBOL_MAX_PER_FILE,
        ))

    # ── Tools ─────────────────────────────────────────────────────────────

    async def grep_repo(
        self,
        query: str,
        glob: str = "",
        is_regex: bool = False,
        max_files: int = 20,
        max_per_file: int = 10,
        case_sensitive: bool = False,
        whole_word: bool = False,
        exclude_paths: set[str] | None = None,
    ) -> ToolResult:
        """Search for a pattern across the entire codebase.

        This is the "grep" tool — the single most important capability
        for grounding agents in real code context.

        Args:
            query: Search string or regex.
            glob: Optional file filter (e.g., '*.py', 'tests/**/*.py').
            is_regex: Treat query as regex.
            max_files: Max files to return.
            max_per_file: Max matches per file.
            case_sensitive: Case-sensitive matching.
            whole_word: Match whole words only.
            exclude_paths: Workspace paths to drop from the results.

        Returns:
            ToolResult with formatted search output.
        """
        if not self._searcher:
            return ToolResult(
                tool="grep_repo",
                query=query,
                output="(Workspace search not available — no repo clone)",
            )

        from services.workspace_search import SearchOptions, format_search_results_for_agent

        result = await self.search(SearchOptions(
            query=query,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            glob=glob,
            max_files=max_files,
            max_per_file=max_per_file,
        ))
        if exclude_paths:
            result = _filter_result(result, exclude_paths)

        formatted = format_search_results_for_agent(result, max_chars=_MAX_GROUNDING_CHARS // 2)

        return ToolResult(
            tool="grep_repo",
            query=f"{query}" + (f" (glob={glob})" if glob else ""),
            output=formatted,
            match_count=result.total_matches,
        )

    async def read_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> ToolResult:
        """Read a file's contents from the cloned repo.

        Useful for reading the full source of a file that's only partially
        visible in the diff, or for reading related files (e.g., test files).
        Only files present in the workspace index can be read (this also
        blocks path traversal and ignored files like .env).

        Args:
            file_path: Workspace-relative path.
            start_line: 1-based start line.
            end_line: Optional 1-based end line (None = to end of file).
        """
        file_path = file_path.replace("\\", "/").lstrip("/")
        if not self._index or not self._index.contains(file_path):
            return ToolResult(
                tool="read_file",
                query=file_path,
                output=f"(File not found in workspace: {file_path})",
            )

        abs_path = os.path.join(self._repo_path, file_path.replace("/", os.sep))

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.read().splitlines()
        except OSError as e:
            return ToolResult(
                tool="read_file",
                query=file_path,
                output=f"(Cannot read file: {e})",
            )

        # Apply line range
        total = len(all_lines)
        s = min(max(1, start_line), max(total, 1)) - 1
        e = min(total, end_line) if end_line else total
        e = max(e, s)
        selected = all_lines[s:e]

        # Format with line numbers
        lines_with_numbers = []
        char_count = 0
        for i, line in enumerate(selected, start=s + 1):
            formatted = f"{i:4d} | {line.rstrip()}"
            if char_count + len(formatted) > _MAX_FILE_READ:
                lines_with_numbers.append(f"     | ... ({e - i + 1} more lines)")
                break
            lines_with_numbers.append(formatted)
            char_count += len(formatted)

        range_str = f"L{s+1}-{e}" if end_line else f"({total} lines)"
        output = f"### {file_path} {range_str}\n" + "\n".join(lines_with_numbers)

        return ToolResult(
            tool="read_file",
            query=file_path,
            output=output,
            match_count=len(selected),
        )

    async def find_related(self, file_path: str) -> ToolResult:
        """Find files related to the given path.

        Discovers test files, config siblings, and implementations.
        """
        if not self._index:
            return ToolResult(
                tool="find_related",
                query=file_path,
                output="(Workspace index not available)",
            )

        related = self._index.find_related(file_path)
        if not related:
            return ToolResult(
                tool="find_related",
                query=file_path,
                output=f"No related files found for {file_path}",
            )

        lines = [f"### Files related to {file_path}"]
        for f in related[:10]:
            tag = "  [test]" if _is_test_path(f.path) else ""
            lines.append(f"  - {f.path} ({f.size} bytes){tag}")

        return ToolResult(
            tool="find_related",
            query=file_path,
            output="\n".join(lines),
            match_count=len(related),
        )

    async def list_directory(self, dir_path: str = "") -> ToolResult:
        """List the contents of a directory in the repo."""
        if not self._index:
            return ToolResult(
                tool="list_directory",
                query=dir_path,
                output="(Workspace index not available)",
            )

        children = self._index.children(dir_path)
        if not children:
            return ToolResult(
                tool="list_directory",
                query=dir_path or "(root)",
                output=f"Directory not found or empty: {dir_path or '(root)'}",
            )

        lines = [f"### {dir_path or '(root)'}/ ({len(children)} items)"]
        for node in children[:50]:
            if node.ignored:
                continue
            if node.is_dir:
                lines.append(f"  {node.name}/")
            else:
                size_kb = node.size / 1024
                lines.append(f"  {node.name} ({size_kb:.1f}KB)")

        return ToolResult(
            tool="list_directory",
            query=dir_path,
            output="\n".join(lines),
            match_count=len(children),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Diff analysis
# ═══════════════════════════════════════════════════════════════════════════


def extract_diff_symbols(diff_text: str, max_per_kind: int = 6) -> DiffSymbols:
    """Classify the symbols a unified diff touches.

    - Declarations on '-' lines only        → removed (deleted or renamed)
    - Declarations on both '-' and '+'       → signature_changed
    - Function context in @@ hunk headers    → modified (body edits)
    - Declarations on '+' lines only         → added

    Generic names (run, get, save, ...) and dunders are skipped since a
    whole-repo grep for them is noise.
    """
    added_defs: list[str] = []
    removed_defs: list[str] = []
    hunk_ctx: list[str] = []

    for raw in diff_text.splitlines():
        if raw.startswith(("+++", "---", "diff --git", "index ")):
            continue
        if raw.startswith("@@"):
            m = _HUNK_HEADER_RE.match(raw)
            if m and m.group(1):
                name = _declared_name(m.group(1))
                if name:
                    hunk_ctx.append(name)
        elif raw.startswith("+"):
            name = _declared_name(raw[1:])
            if name:
                added_defs.append(name)
        elif raw.startswith("-"):
            name = _declared_name(raw[1:])
            if name:
                removed_defs.append(name)

    added_set, removed_set = set(added_defs), set(removed_defs)

    def keep(names: list[str]) -> list[str]:
        return [n for n in _unique(names) if _is_searchable(n)][:max_per_kind]

    signature_changed = keep([n for n in removed_defs if n in added_set])
    removed = keep([n for n in removed_defs if n not in added_set])
    added = keep([n for n in added_defs if n not in removed_set])
    touched = set(signature_changed) | set(removed) | set(added)
    modified = keep([n for n in hunk_ctx if n not in touched])

    return DiffSymbols(
        removed=removed,
        signature_changed=signature_changed,
        modified=modified,
        added=added,
    )


def _declared_name(line: str) -> str | None:
    for pat in _DEF_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group(1)
    return None


def _is_searchable(name: str) -> bool:
    if len(name) < 3:
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    return name not in _GENERIC_NAMES and name.lower() not in _GENERIC_NAMES


def _is_test_path(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path))


def _added_lines(diff_text: str) -> str:
    return "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Role-specific probes
# ═══════════════════════════════════════════════════════════════════════════

# Probes only fire when the diff's ADDED lines match the trigger, so agents
# aren't fed repo-wide noise unrelated to the change under review.
_ROLE_PROBES: dict[str, list[_Probe]] = {
    "Security": [
        _Probe(
            trigger=r"\b(execute|executemany|raw|text)\s*\(|SELECT\s|INSERT\s|UPDATE\s|DELETE\s",
            query=r"\.execute\(\s*f['\"]|\.execute\([^)]*%\s|\.execute\([^)]*\+\s",
            title="String-built SQL elsewhere in the repo",
            note="Compare with how the rest of the codebase builds queries; flag the new code if it "
                 "deviates from a parameterized convention.",
        ),
        _Probe(
            trigger=r"password|secret|token|api_key|apikey|private_key|credential",
            query=r"(password|secret|api_key|apikey|private_key)\s*[:=]\s*['\"][^'\"]{6,}['\"]",
            title="Hardcoded credential-like literals in the repo",
        ),
        _Probe(
            trigger=r"\b(eval|exec|subprocess|os\.system|pickle|yaml\.load|shell\s*=)",
            query=r"\b(eval|exec)\s*\(|shell\s*=\s*True|os\.system\(|pickle\.loads?\(|yaml\.load\(",
            title="Dangerous execution / deserialization sinks in the repo",
        ),
    ],
    "Performance": [
        _Probe(
            trigger=r"\btime\.sleep\(|\brequests\.(get|post|put|delete|patch)\(|async\s+def",
            query=r"\btime\.sleep\(|\brequests\.(get|post|put|delete|patch)\(",
            glob="*.py",
            title="Blocking calls in the repo (dangerous inside async code)",
        ),
        _Probe(
            trigger=r"\.objects\.|\.query\(|session\.(query|execute)|\.find\(|\.all\(\)",
            query=r"\.objects\.all\(\)|\.query\([^)]*\)\.all\(\)|\.find\(\s*\{\s*\}\s*\)",
            title="Unbounded query patterns in the repo",
        ),
    ],
    "Observability": [
        _Probe(
            trigger=r"\bprint\(|logging\.|logger\.|console\.(log|error)|\bexcept\b|\bcatch\b|raise\s",
            query=r"\bget_logger\(|logging\.getLogger\(|structlog\.get_logger\(",
            title="Logging convention used in this repo",
            note="New code should follow this convention rather than print() or ad-hoc loggers.",
            max_files=4,
            max_per_file=1,
        ),
    ],
    "Quality": [],
}


# ═══════════════════════════════════════════════════════════════════════════
# Grounding context entry point
# ═══════════════════════════════════════════════════════════════════════════


async def gather_grounding_context(
    toolkit: AgentToolkit,
    changed_files: list[str],
    diff_text: str,
    agent_role: str,
    pr_changed_files: list[str] | None = None,
    max_chars: int = _MAX_GROUNDING_CHARS,
    timeout: float = _GROUNDING_TIMEOUT_S,
) -> str:
    """Gather grounding context by running tools based on diff heuristics.

    This is the main entry point called by BaseReviewAgent before the
    reasoning chain runs. It analyzes the diff to decide which probes
    to run, runs them concurrently under a time budget, then formats
    the results into a single prioritized text block.

    Probes (highest priority first):
      0. Removed symbols still referenced in files the PR doesn't touch
         (breaking-change detection) — all agents
      1. Call sites of symbols whose signature changed — all agents
      2. Existing usages of modified symbols — all agents
      3. Test coverage of changed public symbols — Quality
      4. Newly added names that already exist elsewhere — Quality
      5. Role-specific pattern probes (triggered by the diff) — per role
      6. Related files (tests, siblings) — all agents

    Args:
        toolkit: An initialized AgentToolkit.
        changed_files: File(s) whose diff is being reviewed right now.
        diff_text: The diff text for those file(s).
        agent_role: The agent's role (Security, Quality, etc.).
        pr_changed_files: ALL files changed in the PR. Used to tell apart
            stale in-PR references from genuinely external ones.
            Defaults to ``changed_files``.
        max_chars: Character budget for the returned block.
        timeout: Wall-clock budget in seconds; partial results are kept.

    Returns:
        Formatted text block of grounding context (may be empty string).
    """
    if not toolkit.available:
        return ""

    current = {_norm(p) for p in changed_files}
    pr_files = {_norm(p) for p in (pr_changed_files or changed_files)} | current
    symbols = extract_diff_symbols(diff_text)
    added_text = _added_lines(diff_text)

    sections: list[_Section] = []
    counter = iter(range(1_000_000))

    def emit(priority: int, text: str) -> None:
        if text:
            sections.append(_Section(priority, next(counter), text))

    probes = [
        _probe_removed(toolkit, symbols, pr_files, emit),
        _probe_signature_changes(toolkit, symbols, pr_files, emit),
        _probe_usages(toolkit, symbols, current, pr_files, emit),
        _probe_role_patterns(toolkit, agent_role, added_text, current, emit),
        _probe_related(toolkit, changed_files, emit),
    ]
    if agent_role == "Quality":
        probes.append(_probe_test_coverage(toolkit, symbols, emit))
        probes.append(_probe_duplicates(toolkit, symbols, pr_files, emit))

    started = time.monotonic()
    timed_out = False
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*probes, return_exceptions=True),
            timeout=timeout,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.debug("grounding_probe_failed", agent=agent_role, error=str(r))
    except TimeoutError:
        timed_out = True
        logger.warning(
            "grounding_context_timeout",
            agent=agent_role,
            timeout_s=timeout,
            sections_collected=len(sections),
        )

    if not sections:
        return ""

    header = (
        "# Grounding Context (searched the repository for you)\n"
        "Results come from the PR's BASE branch (code before this PR). "
        "References inside files this PR also modifies may already be updated in the diff.\n"
    )
    out: list[str] = [header]
    used = len(header)
    omitted = 0
    for sec in sorted(sections):
        if used + len(sec.text) > max_chars:
            omitted += 1
            continue
        out.append(sec.text)
        used += len(sec.text) + 2
    if omitted:
        out.append(f"(... {omitted} lower-priority section(s) omitted to fit the context budget)")

    grounding = "\n\n".join(out)

    logger.info(
        "grounding_context_gathered",
        agent=agent_role,
        sections=len(sections),
        omitted=omitted,
        context_chars=len(grounding),
        removed_symbols=len(symbols.removed),
        signature_changes=len(symbols.signature_changed),
        timed_out=timed_out,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        searches=toolkit.stats["searches"],
        cache_hits=toolkit.stats["cache_hits"],
    )

    return grounding


# ── Probes ────────────────────────────────────────────────────────────────


async def _probe_removed(toolkit: AgentToolkit, symbols: DiffSymbols, pr_files: set[str], emit) -> None:
    """Removed/renamed symbols that files outside the PR still reference."""
    safe: list[str] = []
    for sym in symbols.removed[:4]:
        result = await toolkit.symbol_search(sym)
        if result is None:
            continue
        external = [r for r in result.results if r.path not in pr_files]
        refs = [_only(r, definitions=False) for r in external]
        refs = [r for r in refs if r.matches]
        other_defs = sorted({r.path for r in external if any(m.is_definition for m in r.matches)})

        if not refs:
            if not result.truncated:
                safe.append(sym)
            continue

        lines = [
            f"## [!] Removed symbol `{sym}` is still referenced outside this PR",
            f"This diff deletes or renames `{sym}`, but these files are NOT modified by the PR "
            f"and still reference it. Unless `{sym}` remains available elsewhere, this is a "
            f"breaking change (ImportError / NameError / missing method at runtime).",
        ]
        if other_defs:
            lines.append(
                f"Note: `{sym}` is also declared in {', '.join(other_defs[:3])} — some references "
                f"may resolve to that declaration instead."
            )
        lines.append(_format_hits(refs))
        emit(0, "\n".join(lines))

    if safe:
        emit(0, "## Removed symbols with no remaining external references\n"
                + ", ".join(f"`{s}`" for s in safe)
                + " — removed by this diff and not referenced by any file outside the PR "
                  "(no breaking-change risk from these removals).")


async def _probe_signature_changes(toolkit: AgentToolkit, symbols: DiffSymbols, pr_files: set[str], emit) -> None:
    """Call sites of symbols whose declaration line changed."""
    for sym in symbols.signature_changed[:3]:
        result = await toolkit.symbol_search(sym)
        if result is None:
            continue
        external = [_only(r, definitions=False) for r in result.results if r.path not in pr_files]
        external = [r for r in external if r.matches]
        in_pr = sorted({r.path for r in result.results if r.path in pr_files})

        if not external:
            emit(1, f"## Signature change: `{sym}`\nNo call sites outside this PR"
                    + (f" (in-PR references: {', '.join(in_pr[:5])})." if in_pr else "."))
            continue

        emit(1, "\n".join([
            f"## Signature change: `{sym}` — verify these call sites (not modified by this PR)",
            f"The declaration of `{sym}` changed in this diff. Check that each caller below still "
            f"passes compatible arguments and handles the return value correctly.",
            _format_hits(external),
        ]))


async def _probe_usages(
    toolkit: AgentToolkit, symbols: DiffSymbols, current: set[str], pr_files: set[str], emit
) -> None:
    """Existing usages of symbols whose bodies were modified."""
    for sym in symbols.modified[:3]:
        result = await toolkit.symbol_search(sym)
        if result is None:
            continue
        hits = [r for r in result.results if r.path not in current]
        if not hits:
            continue
        emit(2, "\n".join([
            f"## Usages of modified `{sym}` (behaviour changes affect these callers)",
            _format_hits(hits, in_pr=pr_files),
        ]))


async def _probe_test_coverage(toolkit: AgentToolkit, symbols: DiffSymbols, emit) -> None:
    """Whether test files reference the changed public symbols."""
    targets = [s for s in _unique(symbols.signature_changed + symbols.modified + symbols.added)
               if not s.startswith("_")][:5]
    if not targets:
        return

    covered: list[str] = []
    uncovered: list[str] = []
    for sym in targets:
        result = await toolkit.symbol_search(sym)
        if result is None:
            continue
        test_files = sorted({r.path for r in result.results if _is_test_path(r.path)})
        if test_files:
            covered.append(f"  - `{sym}`: referenced in {', '.join(test_files[:3])}")
        elif not result.truncated:
            # Only claim "untested" when the search saw the whole repo
            uncovered.append(sym)

    if not covered and not uncovered:
        return
    lines = ["## Test coverage of changed symbols (base branch)"]
    lines.extend(covered)
    if uncovered:
        lines.append(
            "  - No test file references: " + ", ".join(f"`{s}`" for s in uncovered)
            + " (unless tests are added in this PR, these changes are untested)"
        )
    emit(3, "\n".join(lines))


async def _probe_duplicates(toolkit: AgentToolkit, symbols: DiffSymbols, pr_files: set[str], emit) -> None:
    """Newly added names that are already declared elsewhere (possible duplication)."""
    for sym in [s for s in symbols.added if not s.startswith("_")][:3]:
        result = await toolkit.symbol_search(sym)
        if result is None:
            continue
        defs = [_only(r, definitions=True) for r in result.results if r.path not in pr_files]
        defs = [r for r in defs if r.matches]
        if not defs:
            continue
        emit(4, "\n".join([
            f"## `{sym}` is newly added, but a symbol with this name already exists",
            "Possible duplicated logic or confusing name shadowing — check whether the existing "
            "implementation could be reused:",
            _format_hits(defs, max_files=4),
        ]))


async def _probe_role_patterns(
    toolkit: AgentToolkit, agent_role: str, added_text: str, current: set[str], emit
) -> None:
    """Role-specific searches, only when the diff's added lines warrant them."""
    from services.workspace_search import SearchOptions

    for probe in _ROLE_PROBES.get(agent_role, []):
        if not re.search(probe.trigger, added_text, re.IGNORECASE):
            continue
        result = await toolkit.search(SearchOptions(
            query=probe.query,
            is_regex=True,
            glob=probe.glob,
            max_files=probe.max_files,
            max_per_file=probe.max_per_file,
        ))
        if result is None:
            continue
        hits = [r for r in result.results if r.path not in current]
        if not hits:
            continue
        lines = [f"## {probe.title}"]
        if probe.note:
            lines.append(probe.note)
        lines.append(_format_hits(hits, max_files=probe.max_files, max_matches=probe.max_per_file))
        emit(5, "\n".join(lines))


async def _probe_related(toolkit: AgentToolkit, changed_files: list[str], emit) -> None:
    for file_path in changed_files[:3]:
        result = await toolkit.find_related(file_path)
        if result.match_count > 0:
            emit(6, result.output)


# ── Formatting helpers ────────────────────────────────────────────────────


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _only(file_result: Any, definitions: bool) -> Any:
    """Copy of a FileSearchResult keeping only definition (or non-definition) matches."""
    from services.workspace_search import FileSearchResult

    return FileSearchResult(
        path=file_result.path,
        matches=[m for m in file_result.matches if m.is_definition == definitions],
    )


def _filter_result(result: Any, exclude: set[str]) -> Any:
    """Copy of a SearchResult with the given paths removed."""
    from services.workspace_search import SearchResult

    excl = {_norm(p) for p in exclude}
    kept = [r for r in result.results if r.path not in excl]
    return SearchResult(
        results=kept,
        total_matches=sum(len(r.matches) for r in kept),
        files_matched=len(kept),
        truncated=result.truncated,
        query=result.query,
    )


def _format_hits(
    file_results: list[Any],
    max_files: int = 8,
    max_matches: int = 3,
    in_pr: set[str] | None = None,
) -> str:
    """Compact rendering of per-file matches for prompts."""
    lines: list[str] = []
    for fr in file_results[:max_files]:
        tag = "  (also modified in this PR)" if in_pr and fr.path in in_pr else ""
        lines.append(f"### {fr.path}{tag}")
        for m in fr.matches[:max_matches]:
            def_tag = "  [DEFINITION]" if m.is_definition else ""
            lines.append(f"  L{m.line}: {m.pre}{m.mid}{m.post}{def_tag}")
    if len(file_results) > max_files:
        lines.append(f"... and {len(file_results) - max_files} more file(s)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Toolkit factory (shared, freshness-aware cache)
# ═══════════════════════════════════════════════════════════════════════════

# The 4 review agents run in parallel and all call build_toolkit_for_review()
# at once — without a cache they would each walk the repo and build their own
# WorkspaceIndex. Entries are keyed by a clone "freshness stamp" so that a
# git fetch/reset (which rewrites .git/index) invalidates them immediately;
# the TTL and LRU cap just bound memory.
_TOOLKIT_TTL_SECONDS = 600.0
_TOOLKIT_CACHE_MAX = 8
_toolkit_cache: OrderedDict[str, tuple[float, str, AgentToolkit]] = OrderedDict()

# asyncio.Lock binds to the loop that first uses it. Celery tasks create a new
# event loop per task, so locks must be per-loop (a module-level lock would
# raise "bound to a different event loop" on the second review in a worker).
_toolkit_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
    weakref.WeakKeyDictionary()
)


def _clone_stamp(repo_dir: Any) -> str | None:
    """Cheap fingerprint of the clone's checked-out state (None if no clone)."""
    git_dir = repo_dir / ".git"
    try:
        parts = []
        for name in ("index", "HEAD", "FETCH_HEAD"):
            p = git_dir / name
            if p.exists():
                st = p.stat()
                parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        return "|".join(parts) if parts else None
    except OSError:
        return None


def _get_lock(repo_name: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    per_loop = _toolkit_locks.get(loop)
    if per_loop is None:
        per_loop = {}
        _toolkit_locks[loop] = per_loop
    lock = per_loop.get(repo_name)
    if lock is None:
        lock = asyncio.Lock()
        per_loop[repo_name] = lock
    return lock


def clear_toolkit_cache() -> None:
    """Drop all cached toolkits (tests / after clone cleanup)."""
    _toolkit_cache.clear()


async def build_toolkit_for_review(repo_name: str) -> AgentToolkit | None:
    """Build (or reuse) an AgentToolkit for a repo if a clone exists.

    This is the factory function called by each review agent at the start
    of run(). Concurrent callers for the same repo share a single build,
    and the cached toolkit (including its search cache) is reused until the
    clone changes on disk.
    """
    if not repo_name:
        return None

    try:
        from services.repo_cloner import RepoCloner

        repo_dir = RepoCloner()._repo_dir(repo_name)
    except Exception as e:
        logger.warning("toolkit_build_failed", repo=repo_name, error=str(e))
        return None

    async with _get_lock(repo_name):
        stamp = _clone_stamp(repo_dir)
        if stamp is None:
            logger.debug("toolkit_no_clone", repo=repo_name)
            _toolkit_cache.pop(repo_name, None)
            return None

        cached = _toolkit_cache.get(repo_name)
        if cached:
            built_at, cached_stamp, toolkit = cached
            if cached_stamp == stamp and (time.monotonic() - built_at) < _TOOLKIT_TTL_SECONDS:
                _toolkit_cache.move_to_end(repo_name)
                return toolkit

        toolkit = await _build_toolkit_uncached(repo_name, str(repo_dir))
        if toolkit is not None:
            _toolkit_cache[repo_name] = (time.monotonic(), stamp, toolkit)
            _toolkit_cache.move_to_end(repo_name)
            while len(_toolkit_cache) > _TOOLKIT_CACHE_MAX:
                _toolkit_cache.popitem(last=False)
        return toolkit


async def _build_toolkit_uncached(repo_name: str, repo_path: str) -> AgentToolkit | None:
    """Build a fresh AgentToolkit from the local clone (no caching)."""
    try:
        from services.workspace_index import build_workspace_index
        from services.workspace_search import WorkspaceSearcher

        ws_index = await build_workspace_index(repo_path)
        ws_searcher = WorkspaceSearcher(ws_index)

        toolkit = AgentToolkit(
            repo_path=repo_path,
            workspace_index=ws_index,
            workspace_searcher=ws_searcher,
        )

        logger.info(
            "toolkit_built",
            repo=repo_name,
            files_indexed=ws_index.file_count,
        )
        return toolkit

    except Exception as e:
        logger.warning("toolkit_build_failed", repo=repo_name, error=str(e))
        return None
