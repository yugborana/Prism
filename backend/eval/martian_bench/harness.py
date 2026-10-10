"""
Prism Eval Harness — Martian Bench-Based Quality Benchmarking.

Evaluates Prism against the Martian Code Review Benchmark (offline dataset).
Martian Bench uses PRs from 5 major repos (sentry, grafana, cal.com, discourse, keycloak)
with 173 golden comments.

Unlike ReviewBench, Martian's golden comments DO NOT specify exact file paths or lines.
The LLM Judge will match based on the semantic meaning of the finding itself.

Usage:
    python -m eval.martian_bench.harness
    python -m eval.martian_bench.harness --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from eval.harness import (
    PrismFinding,
    compute_metrics,
    EvalSummary,
)

# ── Paths ────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class MartianGolden:
    comment: str
    severity: str
    category: str


@dataclass
class MartianMatchResult:
    prism_finding: PrismFinding
    golden_finding: MartianGolden | None
    match_verdict: str
    judge_reasoning: str


@dataclass
class MartianPRResult:
    repo: str
    pr_number: int
    pr_url: str
    golden_count: int
    prism_count: int
    matches: list[MartianMatchResult] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    matched_golden: int = 0
    missed: int = 0
    judge_errors: int = 0
    duration_seconds: float = 0.0
    error: str = ""


# ── Prism Runner ─────────────────────────────────────────────────────────

async def run_prism_on_pr(repo: str, pr_number: int, skip_index: bool = False) -> list[PrismFinding]:
    """Run Prism's full pipeline on a PR (clone + index + review)."""
    from services.github_service import GitHubService
    from orchestrator.engine import ReviewOrchestrator
    from eval.clone_helper import setup_repo_for_eval, cleanup_repo_async

    gh = GitHubService()

    # fetch_pr_diff/fetch_pr_files swallow errors (return ""/[]); treat as failure
    # so an empty run is never cached or scored.
    pr_details = await gh.fetch_pr_details(repo, pr_number)
    diff = await gh.fetch_pr_diff(repo, pr_number)
    changed_files = await gh.fetch_pr_files(repo, pr_number)
    if not diff or not changed_files:
        raise RuntimeError(f"Could not fetch diff/files for {repo}#{pr_number}")

    base_branch = pr_details["base"]["ref"]

    # Full repo setup: clone + Qdrant index (production-equivalent)
    clone_path = await setup_repo_for_eval(repo, base_branch, skip_index=skip_index)
    has_repo_index = clone_path is not None and not skip_index

    pr_data = {
        "number": pr_number,
        "repo_name": repo,
        "title": pr_details.get("title", ""),
        "body": pr_details.get("body") or "",
        "changed_files": changed_files,
        "diff": diff,
        "head_sha": pr_details["head"]["sha"],
        "base_branch": base_branch,
        "has_repo_index": has_repo_index,
    }

    review_id = f"martian-{repo.replace('/', '-')}-{pr_number}"
    orchestrator = ReviewOrchestrator(review_id=review_id)

    try:
        state = await orchestrator.run_review(pr_data)
    finally:
        # Full flush: clone + Qdrant + toolkit cache
        await cleanup_repo_async(repo)

    findings: list[PrismFinding] = []
    if state.final_review:
        for comment in state.final_review.inline_comments:
            findings.append(PrismFinding(
                path=comment.path,
                line=comment.line,
                body=comment.body,
                severity=getattr(comment, "severity", ""),
                category=getattr(comment, "category", ""),
            ))

    return findings


# ── LLM Judge ────────────────────────────────────────────────────────────

MARTIAN_JUDGE_PROMPT = """You are evaluating an AI code review tool against human-verified "golden" comments from the Martian Code Review Benchmark.

Your job is to determine if a specific finding produced by Prism (the tool) semantically matches any of the golden reference findings.
Note that golden findings in this benchmark DO NOT have file paths or line numbers attached. You must evaluate purely on the semantic meaning of the problem described.

A finding MATCHES if:
1. It identifies the SAME underlying issue or a closely related root cause.
2. The core problem described is equivalent.

A finding PARTIALLY matches if:
1. It touches on the issue but misses a critical nuance.
2. It identifies a symptom of the golden finding's root cause.

A finding does NOT match if:
1. It's about a completely different issue.
2. It's a generic comment that doesn't address the specific bug mentioned in the golden finding.

Respond with EXACTLY this JSON format:
{
  "verdict": "match" | "partial" | "no_match",
  "matched_golden_index": <index or null>,
  "reasoning": "<brief explanation>"
}"""

async def judge_finding(
    prism_finding: PrismFinding,
    golden_findings: list[MartianGolden],
    model: str = "groq/llama-3.3-70b-versatile",
) -> tuple[str, int | None, str]:
    from litellm import acompletion

    golden_summary = "\n".join(
        f"[{i}] Severity: {g.severity}, Category: {g.category}\n"
        f"    Message: {g.comment}"
        for i, g in enumerate(golden_findings)
    )

    user_prompt = f"""## Prism Finding to Evaluate
File: {prism_finding.path}
Line: {prism_finding.line}
Body: {prism_finding.body}

## Golden Reference Findings
{golden_summary}

Does the Prism finding match any golden finding? Respond with JSON."""

    last_err = ""
    for attempt in range(3):
        try:
            response = await acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": MARTIAN_JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            verdict = result.get("verdict", "no_match")
            if verdict not in ("match", "partial", "no_match"):
                verdict = "no_match"
            idx = result.get("matched_golden_index")
            if not isinstance(idx, int):
                idx = None
            return verdict, idx, result.get("reasoning", "")
        except Exception as e:
            last_err = str(e)[:200]
            await asyncio.sleep(2 ** attempt)
    return "error", None, f"Judge error: {last_err}"


# ── Evaluator ────────────────────────────────────────────────────────────

async def evaluate_martian_pr(
    task: dict,
    judge_model: str,
    skip_run: bool = False,
    skip_index: bool = False,
) -> MartianPRResult:
    nwo = task["nwo"]
    pr_number = task["pr_number"]
    
    golden_findings = [MartianGolden(**f) for f in task["golden_findings"]]
    
    result = MartianPRResult(
        repo=nwo,
        pr_number=pr_number,
        pr_url=f"https://github.com/{nwo}/pull/{pr_number}",
        golden_count=len(golden_findings),
        prism_count=0,
    )

    print(f"\nEvaluating {nwo} PR#{pr_number} ({len(golden_findings)} golden findings)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prism_cache = CACHE_DIR / f"prism_{nwo.replace('/', '_')}_{pr_number}.json"

    if skip_run and prism_cache.exists():
        prism_findings_raw = json.loads(prism_cache.read_text(encoding="utf-8"))
        prism_findings = [PrismFinding(**f) for f in prism_findings_raw]
        print(f"  Loaded {len(prism_findings)} cached Prism findings")
    elif skip_run:
        result.error = "No cached Prism findings (--score-only)"
        print(f"  {result.error}")
        return result
    else:
        start = time.monotonic()
        try:
            prism_findings = await run_prism_on_pr(nwo, pr_number, skip_index=skip_index)
            result.duration_seconds = round(time.monotonic() - start, 2)

            prism_cache.write_text(
                json.dumps([asdict(f) for f in prism_findings], indent=2),
                encoding="utf-8",
            )
            print(f"  Prism produced {len(prism_findings)} findings in {result.duration_seconds}s")
        except Exception as e:
            result.error = str(e)[:300]
            print(f"  Error: {result.error}")
            return result

    result.prism_count = len(prism_findings)

    if not prism_findings:
        result.missed = len(golden_findings)
        return result

    matched_golden_indices: set[int] = set()

    for pf in prism_findings:
        verdict, matched_idx, reasoning = await judge_finding(pf, golden_findings, judge_model)

        golden_match = None
        if verdict in ("match", "partial") and matched_idx is not None and 0 <= matched_idx < len(golden_findings):
            golden_match = golden_findings[matched_idx]
            matched_golden_indices.add(matched_idx)

        result.matches.append(MartianMatchResult(
            prism_finding=pf,
            golden_finding=golden_match,
            match_verdict=verdict,
            judge_reasoning=reasoning,
        ))

        if verdict == "error":
            result.judge_errors += 1   # unscored
        elif golden_match is not None:
            result.true_positives += 1  # match or partial (lenient)
        else:
            result.false_positives += 1

    result.matched_golden = len(matched_golden_indices)
    result.missed = len(golden_findings) - result.matched_golden

    p, r, f1 = compute_metrics(
        result.true_positives, result.false_positives, result.missed,
        matched_golden=result.matched_golden,
    )
    print(f"  P={p:.2f} R={r:.2f} F1={f1:.2f} (TP={result.true_positives} FP={result.false_positives} Missed={result.missed})")

    return result

# ── Reporter ─────────────────────────────────────────────────────────────

def generate_report(summary: EvalSummary, judge_model: str, results: list[MartianPRResult]) -> str:
    lines = [
        "# Prism Eval Report — Martian Bench",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Judge Model:** `{judge_model}`",
        f"**PRs Evaluated:** {summary.evaluated_prs} / {summary.total_prs}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Golden Findings | {summary.total_golden} |",
        f"| Total Prism Findings | {summary.total_prism} |",
        f"| True Positives | {summary.true_positives} |",
        f"| False Positives | {summary.false_positives} |",
        f"| Missed (Unmatched Goldens) | {summary.missed} |",
        f"| Judge Errors (unscored) | {summary.judge_errors} |",

        f"| **Precision** | **{summary.precision:.2%}** |",
        f"| **Recall** | **{summary.recall:.2%}** |",
        f"| **F1 Score** | **{summary.f1:.2%}** |",
        "",
        "## Per-PR Results",
        "",
        "| PR | Golden | Prism | TP | FP | Missed | P | R | F1 |",
        "|---|--------|-------|----|----|--------|---|---|----|",
    ]
    
    for pr in results:
        if pr.error:
            lines.append(f"| [{pr.repo}#{pr.pr_number}]({pr.pr_url}) | {pr.golden_count} | ERR | - | - | - | - | - | - |")
        else:
            p, r, f1 = compute_metrics(
                pr.true_positives, pr.false_positives, pr.missed,
                matched_golden=pr.matched_golden,
            )
            lines.append(
                f"| [{pr.repo}#{pr.pr_number}]({pr.pr_url}) | {pr.golden_count} | {pr.prism_count} | "
                f"{pr.true_positives} | {pr.false_positives} | {pr.missed} | "
                f"{p:.2f} | {r:.2f} | {f1:.2f} |"
            )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

async def run_martian_eval(
    limit: int | None = None,
    judge_model: str = "groq/llama-3.3-70b-versatile",
    score_only: bool = False,
    no_index: bool = False,
):
    # Check infra before starting
    if not score_only:
        from eval.clone_helper import check_infra
        infra = await check_infra()
        print(f"\nInfra check: Redis={'✓' if infra['redis'] else '✗'}  Qdrant={'✓' if infra['qdrant'] else '✗'}")
        if not infra['qdrant'] and not no_index:
            print("  ⚠ Qdrant not reachable — run `docker compose -f docker-compose.eval.yml up -d`")
            print("    Continuing without indexing (clone-only mode)")
            no_index = True

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        print("Martian bench data not found! Run `python -m eval.martian_bench.download` first.")
        return

    tasks = json.loads(manifest_path.read_text(encoding="utf-8"))
    if limit:
        tasks = tasks[:limit]

    summary = EvalSummary(total_prs=len(tasks))
    pr_results = []

    for task in tasks:
        pr_result = await evaluate_martian_pr(task, judge_model, score_only, skip_index=no_index)
        pr_results.append(pr_result)
        if pr_result.error:
            continue  # listed in report, excluded from totals

        summary.evaluated_prs += 1
        summary.total_golden += pr_result.golden_count
        summary.total_prism += pr_result.prism_count
        summary.true_positives += pr_result.true_positives
        summary.false_positives += pr_result.false_positives
        summary.matched_golden += pr_result.matched_golden
        summary.missed += pr_result.missed
        summary.judge_errors += pr_result.judge_errors

    summary.precision, summary.recall, summary.f1 = compute_metrics(
        summary.true_positives, summary.false_positives, summary.missed,
        matched_golden=summary.matched_golden,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    report = generate_report(summary, judge_model, pr_results)
    report_path = RESULTS_DIR / f"martian_eval_{stamp}.md"
    report_path.write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"martian_eval_{stamp}.json").write_text(json.dumps({
        "judge_model": judge_model,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "total_golden": summary.total_golden,
        "total_prism": summary.total_prism,
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "matched_golden": summary.matched_golden,
        "missed": summary.missed,
        "judge_errors": summary.judge_errors,
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"MARTIAN EVAL COMPLETE")
    print(f"{'='*60}")
    print(f"  Precision: {summary.precision:.2%}")
    print(f"  Recall:    {summary.recall:.2%}")
    print(f"  F1:        {summary.f1:.2%}")
    print(f"  Report:    {report_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Prism Eval Harness — Martian Bench")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PRs to evaluate")
    parser.add_argument("--judge-model", default="groq/llama-3.3-70b-versatile")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--no-index", action="store_true", help="Skip Qdrant indexing (clone-only mode)")
    args = parser.parse_args()

    asyncio.run(run_martian_eval(
        limit=args.limit,
        judge_model=args.judge_model,
        score_only=args.score_only,
        no_index=args.no_index,
    ))


if __name__ == "__main__":
    main()
