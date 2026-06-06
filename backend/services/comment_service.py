"""
Prism Comment Service — Posts Review Results to GitHub PRs.

Provides methods for:
1. Posting inline code suggestions at exact file + line positions
2. Posting summary comments to the PR conversation
3. Posting dashboard and alert suggestions with action markers
   (parseable by GitHub Actions for one-click creation)

Uses GitHub's comfort-fade preview API for multi-line suggestions.
"""

import httpx
from observability.logging import get_logger
from utils.config import settings

logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"

class CommentService:
    """Posts structured review results to GitHub PRs."""

    def __init__(self, token: str | None = None):
        self.token = token or settings.github_token
        self._headers = {
            "Accept": "application/vnd.github.comfort-fade-preview+json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "Prism-Reviewer-App",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # ── Inline Suggestions ────────────────────────────────────────────────

    async def post_inline_suggestion(
        self,
        repo_full_name: str,
        pr_number: int,
        file_path: str,
        line: int,
        body: str,
        suggestion: str | None = None,
        commit_sha: str | None = None,
    ) -> bool:
        """
        Post an inline comment at a specific file and line on a PR.

        If `suggestion` is provided, it is wrapped in GitHub's suggestion
        block so the reviewer can apply it with one click:
            ```suggestion
            <suggested_code>
            ```
        """
        comment_body = body
        if suggestion:
            comment_body += f"\n\n```suggestion\n{suggestion}\n```"

        payload: dict = {
            "path": file_path,
            "line": line,
            "body": comment_body,
        }
        if commit_sha:
            payload["commit_id"] = commit_sha

        url = f"{GITHUB_API}/repos/{repo_full_name}/pulls/{pr_number}/comments"

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, json=payload, headers=self._headers)
                if response.status_code < 200 or response.status_code > 299:
                    logger.warning(
                        "inline_comment_failed",
                        status=response.status_code,
                        file=file_path,
                        line=line,
                        response=response.text[:300],
                    )
                    return False
                logger.debug("inline_comment_posted", file=file_path, line=line)
                return True
            except Exception as e:
                logger.error("inline_comment_error", file=file_path, line=line, error=str(e))
                return False

    # ── Summary Comments ──────────────────────────────────────────────────

    async def post_summary_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        summary: str,
    ) -> bool:
        """
        Post a top-level summary comment on the PR conversation.

        """
        if not summary:
            logger.debug("no_summary_to_post")
            return True

        url = f"{GITHUB_API}/repos/{repo_full_name}/issues/{pr_number}/comments"
        payload = {"body": summary}

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(url, json=payload, headers=self._headers)
                if response.status_code < 200 or response.status_code > 299:
                    logger.warning(
                        "summary_comment_failed",
                        status=response.status_code,
                        response=response.text[:300],
                    )
                    return False
                logger.info("summary_comment_posted", pr=pr_number)
                return True
            except Exception as e:
                logger.error("summary_comment_error", pr=pr_number, error=str(e))
                return False

    # ── Dashboard Suggestion Comments ─────────────────────────────────────

    async def post_dashboard_suggestions(
        self,
        repo_full_name: str,
        pr_number: int,
        suggestions: list[dict],
    ) -> bool:
        """
        Post dashboard suggestions to the PR with action markers.

        Each suggestion gets a collapsible details block with a parseable
        HTML marker: <!-- DASHBOARD_CREATE:type:name -->
        GitHub Actions workflows can parse these to trigger creation.
        """
        for suggestion in suggestions:
            name = suggestion.get("name", "Unnamed Dashboard")
            dash_type = suggestion.get("type", "grafana")
            priority = suggestion.get("priority", "Medium")
            queries = suggestion.get("queries", "{}")
            panels = suggestion.get("panels", "{}")
            alerts = suggestion.get("alerts", "{}")

            body = f"## Dashboard Suggestion: {name}\n\n"
            body += f"**Type:** {dash_type}\n"
            body += f"**Priority:** {priority}\n\n"
            body += f"### Queries\n```json\n{queries}\n```\n\n"
            body += f"### Panels\n```json\n{panels}\n```\n\n"
            body += f"### Alerts\n```json\n{alerts}\n```\n\n"
            body += "<details>\n"
            body += "<summary>Click to create this dashboard</summary>\n\n"
            body += f"To create this dashboard, comment with:\n\n`prism dashboard --create {name}`\n\n"
            body += f"<!-- DASHBOARD_CREATE:{dash_type}:{name} -->\n"
            body += "</details>\n"

            await self.post_summary_comment(repo_full_name, pr_number, body)

        # Add a "create all" comment
        all_body = "## Create All Dashboards\n\n"
        all_body += "To create all suggested dashboards, comment with:\n\n`prism dashboard --create-all`\n\n"
        await self.post_summary_comment(repo_full_name, pr_number, all_body)

        return True

    # ── Alert Suggestion Comments ─────────────────────────────────────────

    async def post_alert_suggestions(
        self,
        repo_full_name: str,
        pr_number: int,
        suggestions: list[dict],
    ) -> bool:
        """
        Post alert suggestions to the PR with action markers.

        Each suggestion gets a collapsible details block with a parseable
        HTML marker: <!-- ALERT_CREATE:type:name -->
        """
        for suggestion in suggestions:
            name = suggestion.get("name", "Unnamed Alert")
            alert_type = suggestion.get("type", "prometheus")
            priority = suggestion.get("priority", "P1")
            query = suggestion.get("query", "")
            description = suggestion.get("description", "")
            threshold = suggestion.get("threshold", "")
            duration = suggestion.get("duration", "5m")
            notification = suggestion.get("notification", "")
            runbook_link = suggestion.get("runbook_link", "")

            body = f"## Alert Suggestion: {name}\n\n"
            body += f"**Type:** {alert_type}\n"
            body += f"**Priority:** {priority}\n\n"
            body += f"### Query\n```\n{query}\n```\n\n"
            body += f"### Description\n{description}\n\n"
            body += f"### Threshold\n{threshold}\n\n"
            body += f"### Duration\n{duration}\n\n"
            body += f"### Notification\n{notification}\n\n"
            if runbook_link:
                body += f"### Runbook\n[Link to Runbook]({runbook_link})\n\n"
            body += "<details>\n"
            body += "<summary>Click to create this alert</summary>\n\n"
            body += f"To create this alert, comment with:\n\n`prism alert --create {name}`\n\n"
            body += f"<!-- ALERT_CREATE:{alert_type}:{name} -->\n"
            body += "</details>\n"

            await self.post_summary_comment(repo_full_name, pr_number, body)

        # Add a "create all" comment
        all_body = "## Create All Alerts\n\n"
        all_body += "To create all suggested alerts, comment with:\n\n`prism alert --create-all`\n\n"
        await self.post_summary_comment(repo_full_name, pr_number, all_body)

        return True
