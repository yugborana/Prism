"""
Prism Triage / Routing.

Cheap, dependency-free classification of a PR (file paths + added diff
lines) used by the orchestrator to skip review agents that cannot
produce meaningful findings — saving LLM calls and latency.

Design rules (safety first):
  * Triage may only *skip* agents, never add behaviour.
  * Anything unknown/ambiguous routes to ALL agents (fail-open).
  * Security is skipped only for PRs that are provably docs/assets-only.

When used at per-file granularity (called from BaseAgent), the same
rules apply to individual files: skip agents that cannot produce
meaningful findings for that file type + diff content.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

ALL_AGENTS = ("Security", "Quality", "Performance", "Observability")

# ── File Classification Sets ──────────────────────────────────────────────

_DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
_ASSET_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm",
    ".pdf", ".zip", ".tar", ".gz", ".bz2",
}

# Files that are machine-generated or vendored — never worth reviewing.
_GENERATED_EXT = {
    ".d.ts", ".min.js", ".min.css", ".map",
    ".snap",         # Jest snapshots
    ".lock",         # lockfiles (package-lock, yarn.lock, etc.)
}
_GENERATED_PATH_RE = re.compile(
    r"(^|/)(dist|build|out|vendor|node_modules|\.next|__pycache__|\.turbo|coverage)(/|$)"
    r"|\.generated\."
    r"|/generated/"
    r"|\.changeset/"
    r"|pnpm-lock\.yaml$"
    r"|package-lock\.json$"
    r"|yarn\.lock$"
    r"|Cargo\.lock$"
    r"|Gemfile\.lock$"
    r"|poetry\.lock$"
    r"|composer\.lock$",
    re.I,
)

_CODE_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".rb", ".php",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".swift", ".scala", ".sh", ".sql",
}

# Style/presentation files — only Quality agent reviews these.
_STYLE_EXT = {".css", ".scss", ".sass", ".less", ".styl"}

# Template files — Quality + Security review, but not Performance/Observability.
_TEMPLATE_EXT = {".html", ".htm", ".astro", ".vue", ".svelte", ".ejs", ".hbs", ".pug", ".njk", ".liquid"}

_CONFIG_EXT = {".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".env", ".tf", ".dockerfile"}

_TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec|specs)(/|$)|(^|/)test_[^/]*$|[._](test|spec)\.[a-z]+$", re.I)

# ── Content Signal Regexes ────────────────────────────────────────────────
# These force-enable an agent even for "safe-looking" files.

_SECURITY_SIGNALS = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|private[_-]?key|eval\(|exec\(|subprocess|os\.system|"
    r"pickle|yaml\.load|innerHTML|dangerouslySetInnerHTML|SELECT\s|INSERT\s|DELETE\s|UPDATE\s|"
    r"verify\s*=\s*False|chmod|sudo|BEGIN [A-Z ]*PRIVATE KEY)",
    re.I,
)
_PERF_SIGNALS = re.compile(
    r"(\bfor\b|\bwhile\b|\.map\(|\.forEach\(|await\s|asyncio|Thread|cache|SELECT\s|\.query\(|\.execute\(|"
    r"requests\.|httpx\.|fetch\()",
    re.I,
)
_OBS_SIGNALS = re.compile(
    r"(logger|logging\.|log\.|print\(|console\.|metrics|tracer|span|except\b|catch\b|raise\b|throw\b)",
    re.I,
)

# ── Symbol Extraction ─────────────────────────────────────────────────────

_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:pub\s+)?(?:async\s+)?(?:def|class|function|fn|func|struct|interface)\s+(\w+)",
    re.M,
)


@dataclass
class TriageResult:
    agents: list[str] = field(default_factory=lambda: list(ALL_AGENTS))
    skipped: dict[str, str] = field(default_factory=dict)  # agent -> reason
    category: str = "code"
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agents": self.agents,
            "skipped": self.skipped,
            "category": self.category,
            "symbols": self.symbols[:50],
        }


# ── Helpers ───────────────────────────────────────────────────────────────

def _classify(path: str) -> str:
    """Classify a file path into a category for triage routing."""
    p = path.replace("\\", "/")
    base = posixpath.basename(p).lower()
    ext = posixpath.splitext(base)[1]

    # Generated / vendored files — skip everything
    if _GENERATED_PATH_RE.search(p):
        return "generated"
    if ext in _GENERATED_EXT:
        return "generated"

    if base in ("dockerfile", "makefile", "jenkinsfile") or ".github/workflows/" in p.lower():
        return "config"
    if _TEST_RE.search(p):
        return "test"
    if ext in _DOC_EXT or base in ("license", "changelog", "codeowners"):
        return "doc"
    if ext in _ASSET_EXT:
        return "asset"
    if ext in _STYLE_EXT:
        return "style"
    if ext in _TEMPLATE_EXT:
        return "template"
    if ext in _CODE_EXT:
        return "code"
    if ext in _CONFIG_EXT:
        return "config"
    return "unknown"


def _added_text(diff: str) -> str:
    """Extract only the added lines from a unified diff."""
    return "\n".join(
        line[1:] for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++")
    )


def _count_added_lines(diff: str) -> int:
    """Count the number of added lines in a unified diff."""
    return sum(1 for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++"))


# ── Main Triage Function ─────────────────────────────────────────────────

def triage_pr(changed_files: list[str], diff: str) -> TriageResult:
    """Decide which review agents to run. Fail-open on any ambiguity."""
    result = TriageResult()
    if not changed_files:
        return result  # nothing to reason about → run everything

    cats = {_classify(f) for f in changed_files}
    added = _added_text(diff or "")
    added_count = _count_added_lines(diff or "")
    result.symbols = list(dict.fromkeys(_SYMBOL_RE.findall(added)))

    skipped: dict[str, str] = {}

    # ── Generated files: skip ALL agents ──────────────────────────────
    if cats == {"generated"}:
        result.category = "generated"
        for a in ALL_AGENTS:
            skipped[a] = "generated/vendored file"
        result.skipped = skipped
        result.agents = []
        return result

    # Remove generated from further consideration
    cats.discard("generated")

    # ── Ambiguous: fail-open ──────────────────────────────────────────
    if "unknown" in cats or (not added.strip() and "code" in cats):
        result.category = "mixed"
        return result

    non_docs = cats - {"doc", "asset"}

    # ── Docs / assets only ────────────────────────────────────────────
    if not non_docs:
        result.category = "docs"
        if not _SECURITY_SIGNALS.search(added):
            skipped["Security"] = "docs/assets only, no security signals"
        skipped["Quality"] = "docs/assets only"
        skipped["Performance"] = "docs/assets only"
        skipped["Observability"] = "docs/assets only"

    # ── Style files only (.css, .scss, etc.) ──────────────────────────
    elif non_docs <= {"style"}:
        result.category = "style"
        skipped["Security"] = "style-only file"
        skipped["Performance"] = "style-only file"
        skipped["Observability"] = "style-only file"
        # Quality reviews styles (typos, dead rules, etc.)

    # ── Template files (.astro, .vue, .html, etc.) ────────────────────
    elif non_docs <= {"template"} or non_docs <= {"template", "style"}:
        result.category = "template"
        skipped["Performance"] = "template/style file"
        skipped["Observability"] = "template/style file"
        # Security reviews templates (XSS, injection)
        # Quality reviews templates (structure, accessibility)

    # ── Test-only changes ─────────────────────────────────────────────
    elif non_docs <= {"test"}:
        result.category = "tests"
        skipped["Performance"] = "test-only change"
        skipped["Observability"] = "test-only change"
        if not _SECURITY_SIGNALS.search(added):
            skipped["Security"] = "test-only change, no security signals"

    # ── Config-only changes ───────────────────────────────────────────
    elif non_docs <= {"config", "test"} and "code" not in cats:
        result.category = "config"
        skipped["Performance"] = "config-only change"
        if not _OBS_SIGNALS.search(added):
            skipped["Observability"] = "config-only change"

    # ── Code changes (with content-signal filtering) ──────────────────
    else:
        result.category = "code"
        if not _PERF_SIGNALS.search(added):
            skipped["Performance"] = "no loops/IO/query patterns in added lines"
        if not _OBS_SIGNALS.search(added):
            skipped["Observability"] = "no logging/error-handling patterns in added lines"

    # ── Tiny diff optimization ────────────────────────────────────────
    # If a file has very few added lines AND no content signals for an
    # agent, skip it. We respect content signals: a 1-line `await fetch()`
    # is still a real perf concern.
    if added_count <= 3 and result.category in ("code", "test", "config", "template", "style"):
        if "Performance" not in skipped and not _PERF_SIGNALS.search(added):
            skipped["Performance"] = f"tiny diff ({added_count} added lines), no perf signals"
        if "Observability" not in skipped and not _OBS_SIGNALS.search(added):
            skipped["Observability"] = f"tiny diff ({added_count} added lines), no obs signals"

    # Never skip everything for a non-docs PR: Quality is the safety net.
    if result.category not in ("docs", "generated"):
        skipped.pop("Quality", None)

    result.skipped = skipped
    result.agents = [a for a in ALL_AGENTS if a not in skipped]
    return result
