
from services.triage import triage_pr


def _diff(path, body):
    return f"diff --git a/{path} b/{path}\n+++ b/{path}\n" + "\n".join("+" + l for l in body.split("\n"))


def test_docs_only_skips_everything():
    r = triage_pr(["README.md"], _diff("README.md", "Hello docs"))
    assert r.category == "docs"
    assert r.agents == []


def test_docs_with_secret_keeps_security():
    r = triage_pr(["README.md"], _diff("README.md", "api_key = abc"))
    assert "Security" in r.agents


def test_code_runs_security_and_quality():
    r = triage_pr(["a.py"], _diff("a.py", "def f(x):\n    return x"))
    assert "Security" in r.agents and "Quality" in r.agents
    assert "f" in r.symbols


def test_code_with_loop_and_logging_runs_all():
    r = triage_pr(["a.py"], _diff("a.py", "for i in x:\n    logger.info(i)"))
    assert len(r.agents) == 4


def test_unknown_file_fails_open():
    r = triage_pr(["weird.xyz"], _diff("weird.xyz", "x"))
    assert len(r.agents) == 4


def test_empty_fails_open():
    assert len(triage_pr([], "").agents) == 4


def test_scavenger_trims_state(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from services import scavenger

    class _VI:
        @staticmethod
        async def delete_pr_points(*a):
            return True

    import services.vector_indexer as vi

    monkeypatch.setattr(vi, "VectorIndexer", _VI)
    st = SimpleNamespace(final_review=object(), diff_data={"a": 1}, comprehensive_context="x" * 100,
                         cross_file_context="y", file_context="z", per_file_contexts={"a": "b"},
                         code_graphs=[1], import_files=[1])
    rep = asyncio.run(scavenger.scavenge_review(st, "o/r", 1))
    assert rep["qdrant_deleted"] is True and rep["state_trimmed"] is True
    assert st.comprehensive_context == "" and st.diff_data == {}
