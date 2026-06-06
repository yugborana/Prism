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

        if state.security_agent_status == AgentStatus.FAILED:
            failed_agents.append("Security")
        elif state.security_report:
            all_findings.extend(state.security_report.findings)

        if state.quality_agent_status == AgentStatus.FAILED:
            failed_agents.append("Quality")
        elif state.quality_report:
            all_findings.extend(state.quality_report.findings)

        if state.performance_agent_status == AgentStatus.FAILED:
            failed_agents.append("Performance")
        elif state.performance_report:
            all_findings.extend(state.performance_report.findings)

        if state.observability_agent_status == AgentStatus.FAILED:
            failed_agents.append("Observability")
        elif state.observability_report:
            all_findings.extend(state.observability_report.findings)

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

def _build_inline_comments(findings: list[CodeFinding]) -> list[InlineComment]:
    """Convert CodeFindings into GitHub inline comments."""
    comments = []
    for f in findings:
        body = f"**{f.severity.value}** — {f.category}\n\n{f.description}"
        if f.current_code:
            body += f"\n\n**Current code:**\n```\n{f.current_code}\n```"

        suggestion = None
        if f.suggested_fix:
            suggestion = f.suggested_fix

        comments.append(InlineComment(
            path=f.file_path,
            line=f.line_number,
            body=body,
            suggestion=suggestion,
        ))
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
    if security_report and security_report.findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>🔒 Security ({len(security_report.findings)} issues)</strong></summary>\n")
        parts.append(f"**Risk Level:** {security_report.risk_level.value}\n")
        parts.append(security_report.summary + "\n")
        parts.append(_format_findings_section(security_report.findings))
        parts.append("\n</details>\n")

    # Quality section
    if quality_report and quality_report.findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>🔍 Code Quality ({len(quality_report.findings)} issues)</strong></summary>\n")
        parts.append(f"**Maintainability Score:** {quality_report.maintainability_score}/10\n")
        parts.append(quality_report.summary + "\n")
        parts.append(_format_findings_section(quality_report.findings))
        parts.append("\n</details>\n")

    # Performance section
    if performance_report and performance_report.findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>⚡ Performance ({len(performance_report.findings)} issues)</strong></summary>\n")
        parts.append(f"**Impact:** {performance_report.estimated_impact.value}\n")
        parts.append(performance_report.summary + "\n")
        parts.append(_format_findings_section(performance_report.findings))
        parts.append("\n</details>\n")

    # Observability section
    if observability_report and observability_report.findings:
        parts.append("<details>")
        parts.append(f"<summary><strong>📡 Observability ({len(observability_report.findings)} issues)</strong></summary>\n")
        parts.append(f"**Instrumentation Score:** {observability_report.instrumentation_score}/10\n")
        coverage = observability_report.telemetry_coverage
        parts.append(f"**Coverage:** Spans: {coverage.spans} | Logging: {coverage.logging} | Metrics: {coverage.metrics} | Events: {coverage.events}\n")
        parts.append(observability_report.summary + "\n")
        parts.append(_format_findings_section(observability_report.findings))
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
