"""
Prism Aggregator Agent.

Merges structured reports from Security, Quality, Performance, and Observability
agents into a single formatted GitHub PR review comment.

Unlike legacy versions which parse raw LLM text with regex, Prism receives
validated Pydantic models (SecurityReport, QualityReport, PerformanceReport,
ObservabilityReport), making aggregation straightforward.
"""

from typing import Any

from agents.schemas import (
    AgentStatus,
    AggregatedReview,
    CodeFinding,
    InlineComment,
    ReviewEvent,
    ReviewState,
    Severity,
)
from observability.logging import get_logger

logger = get_logger(__name__)


async def aggregator_agent(state: ReviewState) -> dict[str, Any]:
    """
    Merge all agent reports into a final AggregatedReview.
    Returns state updates with the formatted review.
    """
    try:
        # Collect all findings from structured reports
        all_findings: list[CodeFinding] = []
        failed_agents: list[str] = []

        def _get_findings(report) -> list[CodeFinding]:
            if not report: return []
            f_list = report.get("findings", []) if isinstance(report, dict) else getattr(report, "findings", [])
            
            # Convert raw dictionaries to CodeFinding objects for safe attribute access
            parsed = []
            for item in f_list:
                if isinstance(item, dict):
                    # Handle enum conversion errors gracefully by defaulting to INFO
                    if "severity" in item and isinstance(item["severity"], str):
                        item["severity"] = item["severity"].upper()
                        if item["severity"] not in [s.value for s in Severity]:
                            item["severity"] = "INFO"
                    try:
                        parsed.append(CodeFinding(**item))
                    except Exception:
                        pass
                else:
                    parsed.append(item)
            return parsed
            
        def _set_findings(report, f_list):
            if not report: return
            if isinstance(report, dict): report["findings"] = f_list
            else: report.findings = f_list

        if state.security_agent_status == AgentStatus.FAILED:
            failed_agents.append("Security")
        elif state.security_report:
            f = _deduplicate_findings(_get_findings(state.security_report))
            _set_findings(state.security_report, f)
            all_findings.extend(f)

        if state.quality_agent_status == AgentStatus.FAILED:
            failed_agents.append("Quality")
        elif state.quality_report:
            f = _deduplicate_findings(_get_findings(state.quality_report))
            _set_findings(state.quality_report, f)
            all_findings.extend(f)

        if state.performance_agent_status == AgentStatus.FAILED:
            failed_agents.append("Performance")
        elif state.performance_report:
            f = _deduplicate_findings(_get_findings(state.performance_report))
            _set_findings(state.performance_report, f)
            all_findings.extend(f)

        if state.observability_agent_status == AgentStatus.FAILED:
            failed_agents.append("Observability")
        elif state.observability_report:
            f = _deduplicate_findings(_get_findings(state.observability_report))
            _set_findings(state.observability_report, f)
            all_findings.extend(f)

        # Count by severity — normalize to enum to handle mixed types
        # (Pydantic models use Severity enum, raw dicts may use strings)
        counts: dict[Severity, int] = {s: 0 for s in Severity}
        for f in all_findings:
            sev = f.severity
            if isinstance(sev, str):
                try:
                    sev = Severity(sev)
                except ValueError:
                    sev = Severity.INFO
            counts[sev] = counts.get(sev, 0) + 1

        # Build inline comments for GitHub
        inline_comments = _build_inline_comments(all_findings)

        # Build summary comment
        summary = _build_summary(
            state.security_report,
            state.quality_report,
            state.performance_report,
            state.observability_report,
            all_findings,
            failed_agents,
        )

        # Decide review event type
        review_event = ReviewEvent.COMMENT
        if counts.get(Severity.CRITICAL, 0) > 0:
            review_event = ReviewEvent.REQUEST_CHANGES

        review = AggregatedReview(
            summary_comment=summary,
            inline_comments=inline_comments,
            total_issues=len(all_findings),
            critical_count=counts.get(Severity.CRITICAL, 0),
            high_count=counts.get(Severity.HIGH, 0),
            medium_count=counts.get(Severity.MEDIUM, 0),
            low_count=counts.get(Severity.LOW, 0),
            review_event=review_event,
            security_report=state.security_report,
            quality_report=state.quality_report,
            performance_report=state.performance_report,
            observability_report=state.observability_report,
        )

        logger.info(
            "review_aggregated",
            total_issues=len(all_findings),
            critical=counts.get(Severity.CRITICAL, 0),
            review_event=review_event.value,
        )

        return {
            "final_review": review,
            "aggregator_status": AgentStatus.COMPLETED,
        }

    except Exception as e:
        logger.error("aggregation_failed", error=str(e))
        return {
            "aggregator_status": AgentStatus.FAILED,
            "errors": [f"Aggregator failed: {str(e)}"],
        }


# ── Formatting Helpers ───────────────────────────────────────────────────────


def _deduplicate_findings(findings: list[CodeFinding]) -> list[CodeFinding]:
    """
    Remove identical findings occurring on the same file and line.
    Picks the one with the highest severity if multiple exist for the same line.
    """
    grouped: dict[tuple[str, int], list[CodeFinding]] = {}
    general_findings = []

    for f in findings:
        if f.line_number is None or f.line_number <= 0:
            general_findings.append(f)
            continue

        key = (f.file_path, f.line_number)
        if key not in grouped:
            grouped[key] = []

        # Prevent duplicates
        if not any(existing.description.strip().lower() == f.description.strip().lower() for existing in grouped[key]):
            grouped[key].append(f)

    severity_rank = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
        "INFO": 5,
    }

    deduped = list(general_findings)
    for (path, line), line_findings in grouped.items():
        # Sort findings by severity (most severe first)
        line_findings.sort(
            key=lambda f: severity_rank.get(f.severity.value if hasattr(f.severity, "value") else str(f.severity), 99)
        )
        deduped.append(line_findings[0])

    return deduped


def _build_inline_comments(findings: list[CodeFinding]) -> list[InlineComment]:
    """Convert CodeFindings into GitHub inline comments, picking the most severe issue per line."""
    # Group findings by (path, line)
    grouped: dict[tuple[str, int], list[CodeFinding]] = {}
    for f in findings:
        # Ignore findings without a line number
        if f.line_number is None or f.line_number <= 0:
            continue

        key = (f.file_path, f.line_number)
        if key not in grouped:
            grouped[key] = []

        # Prevent exact duplicates (same description)
        if not any(existing.description.strip().lower() == f.description.strip().lower() for existing in grouped[key]):
            grouped[key].append(f)

    # Precedence map for severity (lower number = higher severity)
    severity_rank = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
        "INFO": 5,
    }

    comments = []
    for (path, line), line_findings in grouped.items():
        # Sort findings by severity (most severe first)
        line_findings.sort(key=lambda f: severity_rank.get(f.severity.value, 99))

        # Pick ONLY the most severe finding for this line
        f = line_findings[0]

        body = f"**{f.severity.value}** — {f.category}\n\n{f.description}"
        if f.current_code:
            body += f"\n\n**Current code:**\n```\n{f.current_code}\n```"

        suggestion = f.suggested_fix if f.suggested_fix else None

        comments.append(
            InlineComment(
                path=path,
                line=line,
                body=body,
                suggestion=suggestion,
            )
        )
    return comments


def _build_summary(
    security_report,
    quality_report,
    performance_report,
    observability_report,
    all_findings: list[CodeFinding],
    failed_agents: list[str],
) -> str:
    """Build the main PR review comment with collapsible sections."""
    def _safe_get(obj, key, default=""):
        if not obj: return default
        val = obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
        return val if val is not None else default

    def _safe_enum(val):
        return val.value if hasattr(val, "value") else str(val)

    parts: list[str] = []
    parts.append("## 🔮 Prism Code Review\n")

    if failed_agents:
        parts.append(f"*⚠️ {', '.join(failed_agents)} analysis failed*\n")

    if not all_findings:
        parts.append("✅ **No issues found!** Code looks good.\n")
        parts.append("---\n*Generated by Prism AI Code Reviewer*")
        return "\n".join(parts)

    # Summary counts
    total = len(all_findings)
    parts.append(f"Found **{total}** issue{'s' if total != 1 else ''} across the changed files.\n")

    # Security section
    sec_findings = _safe_get(security_report, "findings", [])
    if security_report and sec_findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>🔒 Security ({len(sec_findings)} issues)</strong></summary>\n")
        parts.append(f"**Risk Level:** {_safe_enum(_safe_get(security_report, 'risk_level'))}\n")
        parts.append(str(_safe_get(security_report, "summary")) + "\n")
        parts.append(_format_findings_section(sec_findings))
        parts.append("\n</details>\n")

    # Quality section
    qual_findings = _safe_get(quality_report, "findings", [])
    if quality_report and qual_findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>🔍 Code Quality ({len(qual_findings)} issues)</strong></summary>\n")
        parts.append(f"**Maintainability Score:** {_safe_get(quality_report, 'maintainability_score')}/10\n")
        parts.append(str(_safe_get(quality_report, "summary")) + "\n")
        parts.append(_format_findings_section(qual_findings))
        parts.append("\n</details>\n")

    # Performance section
    perf_findings = _safe_get(performance_report, "findings", [])
    if performance_report and perf_findings:
        parts.append("<details>")
        parts.append(
            f"<summary><strong>⚡ Performance ({len(perf_findings)} issues)</strong></summary>\n"
        )
        parts.append(f"**Impact:** {_safe_enum(_safe_get(performance_report, 'estimated_impact'))}\n")
        parts.append(str(_safe_get(performance_report, "summary")) + "\n")
        parts.append(_format_findings_section(perf_findings))
        parts.append("\n</details>\n")

    # Observability section
    obs_findings = _safe_get(observability_report, "findings", [])
    if observability_report and obs_findings:
        parts.append("<details>")
        parts.append(
            f"<summary><strong>📡 Observability ({len(obs_findings)} issues)</strong></summary>\n"
        )
        parts.append(f"**Instrumentation Score:** {_safe_get(observability_report, 'instrumentation_score')}/10\n")
        coverage = _safe_get(observability_report, "telemetry_coverage", {})
        spans = _safe_get(coverage, "spans", "N/A")
        logging = _safe_get(coverage, "logging", "N/A")
        metrics = _safe_get(coverage, "metrics", "N/A")
        events = _safe_get(coverage, "events", "N/A")
        parts.append(
            f"**Coverage:** Spans: {spans} | Logging: {logging} | Metrics: {metrics} | Events: {events}\n"
        )
        parts.append(str(_safe_get(observability_report, "summary")) + "\n")
        parts.append(_format_findings_section(obs_findings))
        parts.append("\n</details>\n")

    parts.append("---\n*Generated by Prism AI Code Reviewer*")
    return "\n".join(parts)


def _format_findings_section(findings: list[CodeFinding]) -> str:
    """Format a list of findings into a markdown section with diff blocks."""
    if not findings:
        return ""

    # Group by severity
    by_severity: dict[Severity, list[CodeFinding]] = {}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)

    parts: list[str] = []
    severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]

    for severity in severity_order:
        issues = by_severity.get(severity, [])
        if not issues:
            continue

        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "ℹ️"}
        parts.append(f"\n### {emoji.get(severity.value, '')} {severity.value} ({len(issues)})\n")

        for issue in issues:
            parts.append(f"- `{issue.file_path}:{issue.line_number}` — {issue.description}")
            if issue.current_code and issue.suggested_fix:
                parts.append("```diff")
                parts.append(f"- {issue.current_code.strip()}")
                parts.append(f"+ {issue.suggested_fix.strip()}")
                parts.append("```")

    return "\n".join(parts)
