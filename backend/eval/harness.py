"""
Prism Eval Harness — ReviewBench-Based Quality Benchmarking.

Measures Prism's review quality against GitHub's ReviewBench golden set.
Uses the 25-PR test set with LLM-as-judge matching (Groq via LiteLLM).

Usage:
    # Run full eval (requires GROQ_API_KEY or configured LLM provider)
    python -m eval.harness

    # Run on N PRs only (for quick iteration)
    python -m eval.harness --limit 5

    # Use a specific model for judging
    python -m eval.harness --judge-model groq/llama-3.3-70b-versatile

    # Skip the Prism run and just score from cached results
    python -m eval.harness --score-only

Architecture:
    1. DataLoader   — Downloads ReviewBench test set (25 PRs + golden findings)
    2. PrismRunner  — Runs Prism's orchestrator on each PR
    3. LLMJudge     — Uses Groq to match Prism findings ↔ golden findings
    4. Scorer       — Computes precision, recall, F1 per category/severity
    5. Reporter     — Writes a Markdown report to eval/results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx

# ── Paths ────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"

REVIEWBENCH_BASE = "https://raw.githubusercontent.com/review-bench/ReviewBench/main"
TEST_SET_URL = f"{REVIEWBENCH_BASE}/corpus/test/test.json"
GOLDEN_URL = f"{REVIEWBENCH_BASE}/corpus/test/findings.json"
MANIFEST_URL = f"{REVIEWBENCH_BASE}/corpus/manifest.json"


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class GoldenFinding:
    file: str
    start_line: int
    end_line: int
    message: str
    severity: str          # high | medium | low
    category: str          # correctness | security | performance | ...
    scope: str = ""
    difficulty: str = ""
    context_required: str = ""


@dataclass
class PrismFinding:
    path: str
    line: int
    body: str
    severity: str = ""
    category: str = ""


@dataclass
class MatchResult:
    prism_finding: PrismFinding
    golden_finding: GoldenFinding | None
    match_verdict: str      # "match" | "no_match" | "partial"
    judge_reasoning: str = ""


@dataclass
class PRResult:
    pr_key: str
    repo: str
    pr_number: int
    golden_count: int
    prism_count: int
    matches: list[MatchResult] = field(default_factory=list)
    true_positives: int = 0        # Prism findings that matched a golden (precision numerator)
    false_positives: int = 0
    matched_golden: int = 0        # UNIQUE goldens matched (recall numerator)
    matched_indices: list[int] = field(default_factory=list)
    missed: int = 0
    judge_errors: int = 0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class EvalSummary:
    total_prs: int = 0
    evaluated_prs: int = 0
    total_golden: int = 0
    total_prism: int = 0
    true_positives: int = 0
    false_positives: int = 0
    matched_golden: int = 0
    missed: int = 0
    judge_errors: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    by_category: dict[str, dict] = field(default_factory=dict)
    by_severity: dict[str, dict] = field(default_factory=dict)
    pr_results: list[PRResult] = field(default_factory=list)


# ── Data Loader ──────────────────────────────────────────────────────────

async def download_reviewbench(limit: int | None = None, offset: int = 0) -> tuple[list[dict], dict[str, list[dict]]]:
    """Download ReviewBench test set and golden findings.

    Returns:
        (tasks, goldens) where tasks is a list of PR task dicts and
        goldens is {pr_key: [finding_dicts]}.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    test_cache = CACHE_DIR / "test_set.json"
    golden_cache = CACHE_DIR / "golden_findings.json"
    manifest_cache = CACHE_DIR / "manifest.json"

    async with httpx.AsyncClient(timeout=60) as client:
        # Download test set
        if not test_cache.exists():
            print("Downloading ReviewBench test set...")
            resp = await client.get(TEST_SET_URL)
            resp.raise_for_status()
            test_cache.write_text(resp.text, encoding="utf-8")

        # Download golden findings
        if not golden_cache.exists():
            print("Downloading ReviewBench golden findings...")
            resp = await client.get(GOLDEN_URL)
            resp.raise_for_status()
            golden_cache.write_text(resp.text, encoding="utf-8")

        # Download full manifest for PR metadata
        if not manifest_cache.exists():
            print("Downloading ReviewBench manifest...")
            resp = await client.get(MANIFEST_URL)
            resp.raise_for_status()
            manifest_cache.write_text(resp.text, encoding="utf-8")

    tasks = json.loads(test_cache.read_text(encoding="utf-8"))
    golden_raw = json.loads(golden_cache.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_cache.read_text(encoding="utf-8"))

    # Build pr_key -> task mapping from manifest
    manifest_map = {}
    for t in manifest:
        nwo = t["nwo"]
        pr = t["pr_number"]
        head = t["head"][:8]
        key = f"{nwo.replace('/', '_')}_{pr}-{head}"
        manifest_map[key] = t

    # Build goldens dict keyed by pr_key
    goldens: dict[str, list[dict]] = {}
    if isinstance(golden_raw, dict):
        for pr_key, data in golden_raw.items():
            findings = data.get("findings", []) if isinstance(data, dict) else data
            goldens[pr_key] = [f for f in findings]
    elif isinstance(golden_raw, list):
        for entry in golden_raw:
            pr_key = entry.get("pr_key", "")
            findings = entry.get("findings", [])
            goldens[pr_key] = [f for f in findings]

    if offset:
        tasks = tasks[offset:]

    if limit:
        tasks = tasks[:limit]

    print(f"Loaded {len(tasks)} tasks, {sum(len(v) for v in goldens.values())} golden findings")
    return tasks, goldens


# ── Prism Runner ─────────────────────────────────────────────────────────

async def run_prism_on_pr(task: dict, skip_index: bool = False) -> list[PrismFinding]:
    """Run Prism's full review pipeline on a single PR task.

    Clones the repo and (optionally) indexes it into Qdrant so the
    review runs with the same context as production:
      - Agent tools: grep_repo, read_file, find_related (from clone)
      - ContextFetcher: code_graphs, repo_chunks (from Qdrant)

    The clone is removed after caching results to save disk space.
    """
    from services.github_service import GitHubService
    from orchestrator.engine import ReviewOrchestrator
    from eval.clone_helper import setup_repo_for_eval, cleanup_repo_async

    repo = task["nwo"]
    pr_number = task["pr_number"]

    gh = GitHubService()

    # NOTE: fetch_pr_diff/fetch_pr_files swallow errors and return ""/[];
    # treat that as a failure so we never cache or score an empty run.
    pr_details = await gh.fetch_pr_details(repo, pr_number)
    base_sha = task.get("base")
    head_sha = task.get("head")

    if base_sha and head_sha:
        # Use compare API to get the exact diff for the eval task, as the PR might have been modified/squashed
        from utils.connections import get_httpx_client
        client = get_httpx_client()
        headers = await gh._get_headers()
        
        diff_resp = await client.get(
            f"https://api.github.com/repos/{repo}/compare/{base_sha}...{head_sha}",
            headers={**headers, "Accept": "application/vnd.github.v3.diff"},
            timeout=60
        )
        diff_resp.raise_for_status()
        diff = diff_resp.text
        
        json_resp = await client.get(
            f"https://api.github.com/repos/{repo}/compare/{base_sha}...{head_sha}",
            headers={**headers, "Accept": "application/vnd.github+json"},
            timeout=60
        )
        json_resp.raise_for_status()
        changed_files = [f["filename"] for f in json_resp.json().get("files", [])]
    else:
        diff = await gh.fetch_pr_diff(repo, pr_number)
        changed_files = await gh.fetch_pr_files(repo, pr_number)
        
    if not diff or not changed_files:
        raise RuntimeError(f"Could not fetch diff/files for {repo}#{pr_number}")

    # Determine base branch for the clone
    base_branch = pr_details.get("base", {}).get("ref", "main")

    # Full repo setup: clone + Qdrant index (production-equivalent)
    clone_path = await setup_repo_for_eval(repo, base_branch, skip_index=skip_index)
    has_repo_index = clone_path is not None and not skip_index

    pr_data = {
        "number": pr_number,
        "repo_name": repo,
        "title": task.get("title", pr_details.get("title", "")),
        "body": task.get("body", pr_details.get("body", "")),
        "changed_files": changed_files,
        "diff": diff,
        "head_sha": task.get("head", ""),
        "base_branch": base_branch,
        "has_repo_index": has_repo_index,
        "clone_path": str(clone_path) if clone_path else "",
    }

    review_id = f"eval-{repo.replace('/', '-')}-{pr_number}"
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

JUDGE_SYSTEM_PROMPT = """You are a code review evaluation judge. Your job is to determine if a code review finding (Prism finding) semantically matches any of the golden reference findings.

A finding MATCHES if:
1. It identifies the SAME issue or a closely related issue
2. It references the SAME file (or a closely related file)
3. The core problem described is equivalent, even if wording differs

A finding PARTIALLY matches if:
1. It's about the same file and general area
2. It touches on a related but not identical issue

A finding does NOT match if:
1. It's about a completely different issue
2. It's about a different file with no relation
3. It's a generic/obvious comment that doesn't match any specific golden finding

Respond with EXACTLY this JSON format:
{
  "verdict": "match" | "partial" | "no_match",
  "matched_golden_index": <index or null>,
  "reasoning": "<brief explanation>"
}"""


async def judge_finding(
    prism_finding: PrismFinding,
    golden_findings: list[dict],
    model: str = "groq/llama-3.3-70b-versatile",
) -> tuple[str, int | None, str]:
    """Use LLM to judge if a Prism finding matches any golden finding.

    Returns (verdict, matched_index, reasoning).
    """
    from litellm import acompletion

    golden_summary = "\n".join(
        f"[{i}] File: {g['file']}, Lines: {g['start_line']}-{g['end_line']}, "
        f"Severity: {g['severity']}, Category: {g['category']}\n"
        f"    Message: {g['message'][:300]}"
        for i, g in enumerate(golden_findings)
    )

    user_prompt = f"""## Prism Finding to Evaluate
File: {prism_finding.path}
Line: {prism_finding.line}
Body: {prism_finding.body[:500]}

## Golden Reference Findings
{golden_summary}

Does the Prism finding match any golden finding? Respond with JSON."""

    last_err = ""
    for attempt in range(6):
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv(override=True)
            api_key = os.environ.get("GROQ_API_KEY") if model.startswith("groq/") else None
            
            response = await acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
                api_key=api_key,
                timeout=30,
            )
            result = json.loads(response.choices[0].message.content)
            verdict = result.get("verdict", "no_match")
            if verdict not in ("match", "partial", "no_match"):
                verdict = "no_match"
            idx = result.get("matched_golden_index")
            if not isinstance(idx, int):
                idx = None
            return verdict, idx, result.get("reasoning", "")
        except Exception as e:  # rate limits, bad JSON, etc.
            last_err = str(e)[:200]
            print(f"    [judge retry {attempt+1}] {last_err}")
            await asyncio.sleep(4 + (attempt * 5))
    # Distinct verdict so the scorer does not count infra failures as FPs
    return "error", None, f"Judge error: {last_err}"


# ── Scorer ───────────────────────────────────────────────────────────────

def compute_metrics(
    tp: int, fp: int, fn: int, matched_golden: int | None = None
) -> tuple[float, float, float]:
    """Compute precision, recall, F1.

    ``tp`` = Prism findings that matched (precision numerator).
    ``matched_golden`` = unique goldens matched (recall numerator); defaults to
    ``tp`` for callers where the two are the same.
    """
    if matched_golden is None:
        matched_golden = tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = matched_golden / (matched_golden + fn) if (matched_golden + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


async def evaluate_pr(
    task: dict,
    golden_findings: list[dict],
    judge_model: str,
    skip_run: bool = False,
    skip_index: bool = False,
) -> PRResult:
    """Evaluate Prism on a single PR."""
    nwo = task["nwo"]
    pr_number = task["pr_number"]
    head = task.get("head", "")[:8]
    pr_key = f"{nwo.replace('/', '_')}_{pr_number}-{head}"

    result = PRResult(
        pr_key=pr_key,
        repo=nwo,
        pr_number=pr_number,
        golden_count=len(golden_findings),
        prism_count=0,
    )

    print(f"\nEvaluating {nwo} PR#{pr_number} ({len(golden_findings)} golden findings)")

    # Check for cached Prism results
    prism_cache = CACHE_DIR / f"prism_{pr_key}.json"

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
            prism_findings = await run_prism_on_pr(task, skip_index=skip_index)
            result.duration_seconds = round(time.monotonic() - start, 2)

            # Cache results
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
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

    # Judge each Prism finding sequentially to avoid RateLimits
    matched_golden_indices: set[int] = set()

    print(f"Judging {len(prism_findings)} findings sequentially...")
    for pf in prism_findings:
        verdict, matched_idx, reasoning = await judge_finding(pf, golden_findings, judge_model)
        
        golden_match = None
        if verdict in ("match", "partial") and matched_idx is not None and 0 <= matched_idx < len(golden_findings):
            golden_match = GoldenFinding(**{
                k: golden_findings[matched_idx].get(k, "")
                for k in GoldenFinding.__dataclass_fields__
            })
            matched_golden_indices.add(matched_idx)

        result.matches.append(MatchResult(
            prism_finding=pf,
            golden_finding=golden_match,
            match_verdict=verdict,
            judge_reasoning=reasoning,
        ))

        if verdict == "error":
            result.judge_errors += 1   # unscored: neither TP nor FP
        elif verdict in ("match", "partial") and golden_match is not None:
            result.true_positives += 1  # partial counts as TP (lenient)
        else:
            result.false_positives += 1

    result.matched_indices = sorted(matched_golden_indices)
    result.matched_golden = len(matched_golden_indices)
    result.missed = len(golden_findings) - result.matched_golden

    p, r, f1 = compute_metrics(
        result.true_positives, result.false_positives, result.missed,
        matched_golden=result.matched_golden,
    )
    print(f"  P={p:.2f} R={r:.2f} F1={f1:.2f} (TP={result.true_positives} FP={result.false_positives} Missed={result.missed})")

    return result


# ── Reporter ─────────────────────────────────────────────────────────────

def generate_report(summary: EvalSummary, judge_model: str) -> str:
    """Generate a Markdown evaluation report."""
    lines = [
        "# Prism Eval Report — ReviewBench",
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
    ]

    # Per-category breakdown
    if summary.by_category:
        lines.extend([
            "## By Category",
            "",
            "| Category | Golden | Matched | Missed | Recall |",
            "|----------|--------|---------|--------|--------|",
        ])
        for cat, m in sorted(summary.by_category.items()):
            lines.append(
                f"| {cat} | {m['golden']} | {m['tp']} | {m['missed']} | {m['recall']:.2%} |"
            )
        lines.append("")

    # Per-severity breakdown
    if summary.by_severity:
        lines.extend([
            "## By Severity",
            "",
            "| Severity | Golden | TP | Missed | Recall |",
            "|----------|--------|----|----|--------|",
        ])
        for sev, m in sorted(summary.by_severity.items(), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x[0], 3)):
            lines.append(f"| {sev} | {m['golden']} | {m['tp']} | {m['missed']} | {m['recall']:.2%} |")
        lines.append("")

    # Per-PR results
    lines.extend([
        "## Per-PR Results",
        "",
        "| PR | Golden | Prism | TP | FP | Missed | P | R | F1 | Time |",
        "|---|--------|-------|----|----|--------|---|---|----|----|",
    ])
    for pr in summary.pr_results:
        if pr.error:
            lines.append(f"| {pr.repo}#{pr.pr_number} | {pr.golden_count} | ERR | — | — | — | — | — | — | Error |")
        else:
            p, r, f1 = compute_metrics(
                pr.true_positives, pr.false_positives, pr.missed,
                matched_golden=pr.matched_golden,
            )
            lines.append(
                f"| {pr.repo}#{pr.pr_number} | {pr.golden_count} | {pr.prism_count} | "
                f"{pr.true_positives} | {pr.false_positives} | {pr.missed} | "
                f"{p:.2f} | {r:.2f} | {f1:.2f} | {pr.duration_seconds}s |"
            )
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

async def run_eval(
    limit: int | None = None,
    offset: int = 0,
    judge_model: str = "groq/llama-3.3-70b-versatile",
    score_only: bool = False,
    no_index: bool = False,
) -> EvalSummary:
    """Run the full evaluation."""
    # Check infra before starting (non-blocking: warns if services are down)
    if not score_only:
        from eval.clone_helper import check_infra
        infra = await check_infra()
        print(f"\nInfra check: Redis={'✓' if infra['redis'] else '✗'}  Qdrant={'✓' if infra['qdrant'] else '✗'}")
        if not infra['qdrant'] and not no_index:
            print("  ⚠ Qdrant not reachable — run `docker compose -f docker-compose.eval.yml up -d`")
            print("    Continuing without indexing (clone-only mode)")
            no_index = True

    tasks, goldens = await download_reviewbench(limit=limit, offset=offset)

    summary = EvalSummary(total_prs=len(tasks))

    # Category / severity accumulators
    cat_acc: dict[str, dict[str, int]] = {}
    sev_acc: dict[str, dict[str, int]] = {}

    for task in tasks:
        nwo = task["nwo"]
        pr_number = task["pr_number"]
        head = task.get("head", "")[:8]
        pr_key = f"{nwo.replace('/', '_')}_{pr_number}-{head}"

        golden = goldens.get(pr_key, [])
        if not golden:
            print(f"  No golden findings for {pr_key}, skipping")
            continue

        pr_result = await evaluate_pr(task, golden, judge_model, skip_run=score_only, skip_index=no_index)
        summary.pr_results.append(pr_result)
        if pr_result.error:
            # Errored PRs are listed in the report but excluded from totals
            continue
        summary.evaluated_prs += 1
        summary.total_golden += pr_result.golden_count
        summary.total_prism += pr_result.prism_count
        summary.true_positives += pr_result.true_positives
        summary.false_positives += pr_result.false_positives
        summary.matched_golden += pr_result.matched_golden
        summary.missed += pr_result.missed
        summary.judge_errors += pr_result.judge_errors

        # Category / severity recall use UNIQUE matched goldens
        matched = set(pr_result.matched_indices)
        for i, g in enumerate(golden):
            cat = g.get("category") or "unknown"
            sev = g.get("severity") or "unknown"
            c = cat_acc.setdefault(cat, {"golden": 0, "tp": 0})
            s = sev_acc.setdefault(sev, {"golden": 0, "tp": 0})
            c["golden"] += 1
            s["golden"] += 1
            if i in matched:
                c["tp"] += 1
                s["tp"] += 1

    # Compute overall metrics
    summary.precision, summary.recall, summary.f1 = compute_metrics(
        summary.true_positives, summary.false_positives, summary.missed,
        matched_golden=summary.matched_golden,
    )

    for cat, acc in cat_acc.items():
        missed = acc["golden"] - acc["tp"]
        _, r, _ = compute_metrics(acc["tp"], 0, missed)
        summary.by_category[cat] = {**acc, "missed": missed, "recall": r}

    for sev, acc in sev_acc.items():
        missed = acc["golden"] - acc["tp"]
        _, r, _ = compute_metrics(acc["tp"], 0, missed)
        summary.by_severity[sev] = {**acc, "missed": missed, "recall": r}

    # Generate and save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    report = generate_report(summary, judge_model)
    report_path = RESULTS_DIR / f"eval_{stamp}.md"
    report_path.write_text(report, encoding="utf-8")

    # Also save raw JSON (same timestamp so .md and .json always pair)
    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    json_path.write_text(json.dumps({
        "judge_model": judge_model,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "total_golden": summary.total_golden,
        "total_prism": summary.total_prism,
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "missed": summary.missed,
        "matched_golden": summary.matched_golden,
        "judge_errors": summary.judge_errors,
        "by_category": summary.by_category,
        "by_severity": summary.by_severity,
    }, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE")
    print(f"{'='*60}")
    print(f"  Precision: {summary.precision:.2%}")
    print(f"  Recall:    {summary.recall:.2%}")
    print(f"  F1:        {summary.f1:.2%}")
    print(f"  Report:    {report_path}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prism Eval Harness — ReviewBench Benchmarking")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PRs to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N PRs")
    parser.add_argument("--judge-model", default="groq/qwen/qwen3.8-27b", help="LLM model for judging")
    parser.add_argument("--score-only", action="store_true", help="Score from cached results only")
    parser.add_argument("--no-index", action="store_true", help="Skip Qdrant indexing (clone-only mode)")
    args = parser.parse_args()

    asyncio.run(run_eval(
        limit=args.limit,
        offset=args.offset,
        judge_model=args.judge_model,
        score_only=args.score_only,
        no_index=args.no_index,
    ))


if __name__ == "__main__":
    main()
