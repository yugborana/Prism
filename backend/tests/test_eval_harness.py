"""Tests for the eval harness — data loading, scoring, and reporting."""

import asyncio
import json
from pathlib import Path

from eval.harness import (
    EvalSummary,
    GoldenFinding,
    PRResult,
    PrismFinding,
    MatchResult,
    compute_metrics,
    generate_report,
)


# ── Scoring Tests ────────────────────────────────────────────────────────


def test_compute_metrics_perfect():
    p, r, f1 = compute_metrics(tp=10, fp=0, fn=0)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_compute_metrics_no_findings():
    p, r, f1 = compute_metrics(tp=0, fp=0, fn=0)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_compute_metrics_all_false_positives():
    p, r, f1 = compute_metrics(tp=0, fp=10, fn=5)
    assert p == 0.0
    assert r == 0.0


def test_compute_metrics_mixed():
    p, r, f1 = compute_metrics(tp=5, fp=3, fn=2)
    assert round(p, 4) == round(5 / 8, 4)
    assert round(r, 4) == round(5 / 7, 4)


def test_compute_metrics_with_matched_golden():
    """matched_golden != tp when multiple Prism findings hit the same golden."""
    # 8 Prism findings matched, but only 4 unique goldens were matched.
    p, r, f1 = compute_metrics(tp=8, fp=2, fn=6, matched_golden=4)
    assert round(p, 4) == round(8 / 10, 4)   # precision uses tp
    assert round(r, 4) == round(4 / 10, 4)    # recall uses matched_golden


# ── Report Tests ─────────────────────────────────────────────────────────


def test_generate_report():
    summary = EvalSummary(
        total_prs=2,
        evaluated_prs=2,
        total_golden=10,
        total_prism=8,
        true_positives=5,
        false_positives=3,
        matched_golden=5,
        missed=5,
        precision=0.625,
        recall=0.5,
        f1=0.5556,
        pr_results=[
            PRResult(
                pr_key="test_repo_1-abc12345",
                repo="test/repo",
                pr_number=1,
                golden_count=5,
                prism_count=4,
                true_positives=3,
                false_positives=1,
                matched_golden=3,
                missed=2,
                duration_seconds=12.5,
            ),
            PRResult(
                pr_key="test_repo_2-def67890",
                repo="test/repo",
                pr_number=2,
                golden_count=5,
                prism_count=4,
                true_positives=2,
                false_positives=2,
                matched_golden=2,
                missed=3,
                duration_seconds=8.3,
            ),
        ],
    )
    report = generate_report(summary, "groq/llama-3.3-70b-versatile")
    assert "Prism Eval Report" in report
    assert "62.50%" in report
    assert "50.00%" in report
    assert "test/repo#1" in report
    assert "test/repo#2" in report


# ── Data Structure Tests ─────────────────────────────────────────────────


def test_golden_finding_dataclass():
    gf = GoldenFinding(
        file="app.py",
        start_line=10,
        end_line=20,
        message="Missing null check",
        severity="high",
        category="correctness",
    )
    assert gf.file == "app.py"
    assert gf.severity == "high"
    assert gf.context_required == ""  # default


def test_prism_finding_dataclass():
    pf = PrismFinding(
        path="app.py",
        line=10,
        body="This could raise a NullPointerException",
    )
    assert pf.path == "app.py"
    assert pf.line == 10


def test_match_result_dataclass():
    pf = PrismFinding(path="a.py", line=1, body="bug")
    mr = MatchResult(
        prism_finding=pf,
        golden_finding=None,
        match_verdict="no_match",
        judge_reasoning="No matching golden finding",
    )
    assert mr.match_verdict == "no_match"
    assert mr.golden_finding is None


# ── Download Tests (cached, no network) ──────────────────────────────────


def test_download_uses_cache(tmp_path):
    """Verify that download_reviewbench uses cached files when present."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    tasks = [{"nwo": "test/repo", "pr_number": 1, "head": "abc12345"}]
    goldens = {"test_repo_1-abc12345": [
        {"file": "a.py", "start_line": 1, "end_line": 5, "message": "bug",
         "severity": "high", "category": "correctness",
         "scope": "", "difficulty": ""}
    ]}

    (cache_dir / "test_set.json").write_text(json.dumps(tasks))
    (cache_dir / "golden_findings.json").write_text(json.dumps(goldens))
    (cache_dir / "manifest.json").write_text(json.dumps(tasks))

    # Import and patch CACHE_DIR to use tmp
    import eval.harness as harness
    original_cache = harness.CACHE_DIR
    harness.CACHE_DIR = cache_dir

    try:
        result_tasks, result_goldens = asyncio.run(harness.download_reviewbench())
        assert len(result_tasks) == 1
        assert "test_repo_1-abc12345" in result_goldens
    finally:
        harness.CACHE_DIR = original_cache
