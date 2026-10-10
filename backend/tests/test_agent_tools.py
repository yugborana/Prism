"""Tests for workspace regex search and agent grounding tools.

Run:  cd backend && python -m pytest tests/test_agent_tools.py -q
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from agents import tools
from agents.tools import (
    AgentToolkit,
    extract_diff_symbols,
    gather_grounding_context,
)
from services.workspace_index import build_workspace_index
from services.workspace_search import SearchOptions, WorkspaceSearcher


def run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _write(root: Path, rel: str, content: str, newline: str = "\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.replace("\n", newline).encode("utf-8"))


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A small base-branch checkout."""
    _write(tmp_path, "pkg/billing.py", (
        "def compute_invoice(order):\n"
        "    return order.total\n"
        "\n"
        "def legacy_total(x):\n"
        "    return x\n"
    ))
    _write(tmp_path, "pkg/api.py", (
        "from pkg.billing import compute_invoice, legacy_total\n"
        "\n"
        "def handler(o):\n"
        "    legacy_total(1)\n"
        "    return compute_invoice(o)\n"
    ))
    # Also modified by the PR (its new content lives in the diff)
    _write(tmp_path, "pkg/report.py", "from pkg.billing import legacy_total\nlegacy_total(2)\n")
    _write(tmp_path, "tests/test_billing.py", (
        "from pkg.billing import compute_invoice\n"
        "def test_invoice():\n"
        "    assert compute_invoice\n"
    ))
    _write(tmp_path, "pkg/db.py", 'def q(c, uid):\n    c.execute(f"SELECT * FROM u WHERE id={uid}")\n')
    _write(tmp_path, "pkg/crlf.py", "def windows_style():\n    return 1\n", newline="\r\n")
    _write(tmp_path, "pkg/minified.js", "var a=1;" * 1000 + "windows_style\n")
    _write(tmp_path, ".gitignore", "ignored/\n")
    _write(tmp_path, "ignored/secret.py", "legacy_total = 'should never be seen'\n")
    return tmp_path


def _toolkit(root: Path) -> AgentToolkit:
    async def build():
        idx = await build_workspace_index(str(root))
        return AgentToolkit(str(root), idx, WorkspaceSearcher(idx))

    return run(build())


BILLING_DIFF = """diff --git a/pkg/billing.py b/pkg/billing.py
--- a/pkg/billing.py
+++ b/pkg/billing.py
@@ -1,5 +1,3 @@
-def compute_invoice(order):
+def compute_invoice(order, currency):
     return order.total
-
-def legacy_total(x):
-    return x
"""


# ═══════════════════════════════════════════════════════════════════════════
# Diff symbol extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractDiffSymbols:
    def test_classifies_removed_and_signature_changed(self):
        s = extract_diff_symbols(BILLING_DIFF)
        assert s.removed == ["legacy_total"]
        assert s.signature_changed == ["compute_invoice"]
        assert s.added == []

    def test_hunk_header_gives_modified_symbol(self):
        diff = (
            "@@ -10,3 +10,4 @@ def apply_discount(order):\n"
            "     x = 1\n"
            "+    y = 2\n"
        )
        assert extract_diff_symbols(diff).modified == ["apply_discount"]

    def test_added_js_and_go_symbols(self):
        diff = (
            "+export async function fetchUser(id) {\n"
            "+const renderCard = (props) => {\n"
            "+func (s *Server) ServeLogin(w http.ResponseWriter) {\n"
        )
        assert extract_diff_symbols(diff).added == ["fetchUser", "renderCard", "ServeLogin"]

    def test_generic_and_dunder_names_skipped(self):
        diff = "+def run(self):\n+def __init__(self):\n+def get(self):\n+def ok(x):\n"
        assert extract_diff_symbols(diff).added == []

    def test_file_headers_are_not_symbols(self):
        diff = "--- a/def_thing.py\n+++ b/def_thing.py\n"
        s = extract_diff_symbols(diff)
        assert not (s.added or s.removed or s.modified)


# ═══════════════════════════════════════════════════════════════════════════
# Workspace search
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkspaceSearch:
    def test_results_are_deterministic_and_path_sorted(self, repo):
        tk = _toolkit(repo)
        opts = SearchOptions(query="legacy_total", max_files=2)
        a = run(tk._searcher.search(opts))
        b = run(tk._searcher.search(opts))
        paths = [r.path for r in a.results]
        assert paths == [r.path for r in b.results] == sorted(paths)
        assert a.truncated  # 3 matching files, capped at 2

    def test_gitignored_files_never_searched(self, repo):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(query="should never be seen")))
        assert res.total_matches == 0

    def test_crlf_snippets_have_no_carriage_return(self, repo):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(query="windows_style", glob="*.py")))
        m = res.results[0].matches[0]
        assert "\r" not in m.post and m.is_definition

    def test_regex_end_anchor_works_on_crlf(self, repo):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(query=r"windows_style\(\):$", is_regex=True)))
        assert res.files_matched == 1

    def test_huge_lines_are_skipped(self, repo):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(query="windows_style", glob="*.js")))
        assert res.total_matches == 0

    @pytest.mark.parametrize("glob,expected", [
        ("*.py", {"pkg/api.py", "pkg/billing.py", "pkg/report.py", "tests/test_billing.py"}),
        ("tests/", {"tests/test_billing.py"}),
        ("tests/**/*.py", {"tests/test_billing.py"}),
        ("test_*.py", {"tests/test_billing.py"}),
        ("report", {"pkg/report.py"}),
    ])
    def test_glob_forms(self, repo, glob, expected):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(query="legacy_total|compute_invoice",
                                                    is_regex=True, glob=glob)))
        assert {r.path for r in res.results} == expected

    def test_whole_word_case_sensitive(self, repo):
        tk = _toolkit(repo)
        res = run(tk._searcher.search(SearchOptions(
            query="legacy_tota", whole_word=True, case_sensitive=True)))
        assert res.total_matches == 0


# ═══════════════════════════════════════════════════════════════════════════
# Toolkit
# ═══════════════════════════════════════════════════════════════════════════


class TestToolkit:
    def test_concurrent_identical_searches_share_one_scan(self, repo):
        tk = _toolkit(repo)

        async def go():
            await asyncio.gather(*(tk.symbol_search("legacy_total") for _ in range(8)))
            await tk.symbol_search("legacy_total")

        run(go())
        assert tk.stats["searches"] == 1
        assert tk.stats["cache_hits"] == 8

    def test_read_file_blocks_traversal_and_ignored_files(self, repo):
        tk = _toolkit(repo)
        assert "not found" in run(tk.read_file("../etc/passwd")).output
        assert "not found" in run(tk.read_file("ignored/secret.py")).output
        ok = run(tk.read_file("pkg/billing.py", 1, 2))
        assert "def compute_invoice" in ok.output and ok.match_count == 2

    def test_grep_repo_exclude_paths(self, repo):
        tk = _toolkit(repo)
        r = run(tk.grep_repo("legacy_total", exclude_paths={"pkg/report.py"}))
        assert "pkg/report.py" not in r.output and "pkg/api.py" in r.output


# ═══════════════════════════════════════════════════════════════════════════
# Grounding context
# ═══════════════════════════════════════════════════════════════════════════


def _ground(tk, role, diff=BILLING_DIFF, **kw):
    kw.setdefault("pr_changed_files", ["pkg/billing.py", "pkg/report.py"])
    return run(gather_grounding_context(tk, ["pkg/billing.py"], diff, role, **kw))


class TestGrounding:
    def test_breaking_change_flags_only_external_references(self, repo):
        out = _ground(_toolkit(repo), "Security")
        section = out.split("## [!] Removed symbol `legacy_total`")[1].split("\n## ")[0]
        assert "pkg/api.py" in section
        assert "pkg/report.py" not in section  # modified in the PR -> stale reference

    def test_removed_symbol_with_no_external_refs_is_reported_safe(self, repo):
        out = _ground(_toolkit(repo), "Security",
                      pr_changed_files=["pkg/billing.py", "pkg/report.py", "pkg/api.py"])
        assert "[!] Removed symbol" not in out
        assert "no remaining external references" in out

    def test_signature_change_lists_external_call_sites(self, repo):
        out = _ground(_toolkit(repo), "Performance")
        assert "Signature change: `compute_invoice`" in out
        sig = out.split("Signature change: `compute_invoice`")[1].split("\n## ")[0]
        assert "pkg/api.py" in sig and "L5:" in sig

    def test_quality_gets_test_coverage(self, repo):
        out = _ground(_toolkit(repo), "Quality")
        assert "Test coverage" in out and "tests/test_billing.py" in out

    def test_non_quality_roles_skip_test_coverage(self, repo):
        assert "Test coverage" not in _ground(_toolkit(repo), "Security")

    def test_quality_flags_duplicate_new_symbol(self, repo):
        diff = "@@ -0,0 +1,2 @@\n+def legacy_total(amount):\n+    return amount\n"
        out = run(gather_grounding_context(
            _toolkit(repo), ["pkg/new.py"], diff, "Quality", pr_changed_files=["pkg/new.py"]))
        assert "newly added, but a symbol with this name already exists" in out
        assert "pkg/billing.py" in out

    def test_role_probe_only_fires_when_triggered(self, repo):
        tk = _toolkit(repo)
        sql_diff = '@@ -0,0 +1 @@\n+    cursor.execute(f"DELETE FROM t WHERE id={x}")\n'
        hit = run(gather_grounding_context(tk, ["pkg/x.py"], sql_diff, "Security"))
        assert "String-built SQL" in hit and "pkg/db.py" in hit

        miss = run(gather_grounding_context(tk, ["pkg/x.py"], "@@ -1 +1 @@\n+x = 1\n", "Security"))
        assert "String-built SQL" not in miss

    def test_budget_keeps_highest_priority_sections(self, repo):
        out = _ground(_toolkit(repo), "Quality", max_chars=900)
        assert "[!] Removed symbol `legacy_total`" in out
        assert "omitted to fit the context budget" in out

    def test_timeout_returns_quickly(self, repo):
        tk = _toolkit(repo)

        class SlowSearcher:
            async def search(self, opts):
                await asyncio.sleep(5)

        tk._searcher = SlowSearcher()

        async def go():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            out = await gather_grounding_context(
                tk, ["pkg/billing.py"], BILLING_DIFF, "Security", timeout=0.3)
            return out, loop.time() - t0

        out, elapsed = run(go())
        assert elapsed < 2.0
        assert "[!] Removed symbol" not in out

    def test_unavailable_toolkit_returns_empty(self):
        assert run(gather_grounding_context(AgentToolkit("/nope"), ["a.py"], BILLING_DIFF, "Quality")) == ""


# ═══════════════════════════════════════════════════════════════════════════
# Toolkit factory cache
# ═══════════════════════════════════════════════════════════════════════════


class TestToolkitFactory:
    @pytest.fixture(autouse=True)
    def _fake_cloner(self, repo, monkeypatch):
        (repo / ".git").mkdir()
        (repo / ".git" / "index").write_bytes(b"v1")

        class FakeCloner:
            def _repo_dir(self, name):
                return repo

        fake = types.ModuleType("services.repo_cloner")
        fake.RepoCloner = FakeCloner
        monkeypatch.setitem(sys.modules, "services.repo_cloner", fake)

        builds = {"n": 0}
        real_build = tools._build_toolkit_uncached

        async def counting_build(name, path):
            builds["n"] += 1
            return await real_build(name, path)

        monkeypatch.setattr(tools, "_build_toolkit_uncached", counting_build)
        tools.clear_toolkit_cache()
        self.builds = builds
        yield
        tools.clear_toolkit_cache()

    def test_parallel_agents_share_one_build(self):
        async def go():
            return await asyncio.gather(*(tools.build_toolkit_for_review("o/r") for _ in range(4)))

        kits = run(go())
        assert self.builds["n"] == 1 and all(k is kits[0] for k in kits)

    def test_works_across_event_loops(self):
        # Celery runs each task in a fresh event loop — must not raise
        a = run(tools.build_toolkit_for_review("o/r"))
        b = run(tools.build_toolkit_for_review("o/r"))
        assert a is b and self.builds["n"] == 1

    def test_clone_update_invalidates_cache(self, repo):
        a = run(tools.build_toolkit_for_review("o/r"))
        (repo / ".git" / "index").write_bytes(b"v2-after-git-reset")
        b = run(tools.build_toolkit_for_review("o/r"))
        assert a is not b and self.builds["n"] == 2

    def test_no_clone_returns_none(self, repo):
        import shutil

        shutil.rmtree(repo / ".git")
        assert run(tools.build_toolkit_for_review("o/r")) is None
