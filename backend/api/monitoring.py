"""
Prism Dashboard & Alert API Routes.

Provides endpoints for:
1. Running dashboard analysis on a PR (via the DashboardAgent)
2. Running alert analysis on a PR (via the AlertAgent)
3. Creating dashboards in Grafana/Datadog/Amplitude
4. Creating alerts in Prometheus/Datadog

These endpoints are called by the GitHub Actions workflows
(prism-dashboard-creation.yml, prism-alert-creation.yml) and
can also be triggered manually or via PR comment commands.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from agents.dashboard_agent import DashboardAgent
from agents.alert_agent import AlertAgent
from agents.schemas import ReviewState
from services.dashboard_service import DashboardService
from services.alert_service import AlertService
from services.github_service import GitHubService
from services.comment_service import CommentService
from api.auth import require_api_key
from observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Request/Response Models ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request to analyze a PR for dashboard or alert suggestions."""
    repo: str = Field(description="Full repository name (owner/repo)")
    pr_number: int = Field(description="PR number to analyze")
    installation_id: int | None = Field(default=None, description="GitHub App installation ID (optional for local dev)")


class CreateDashboardRequest(BaseModel):
    """Request to create a specific dashboard."""
    name: str = Field(description="Dashboard name")
    type: str = Field(default="grafana", description="grafana, datadog, or amplitude")
    priority: str = Field(default="Medium")
    queries: str = Field(default="[]", description="JSON string of query definitions")
    panels: str = Field(default="[]", description="JSON string of panel definitions")
    alerts: str = Field(default="[]", description="JSON string of alert definitions")


class CreateAlertRequest(BaseModel):
    """Request to create a specific alert."""
    name: str = Field(description="Alert name")
    type: str = Field(default="prometheus", description="prometheus or datadog")
    priority: str = Field(default="P1")
    query: str = Field(default="", description="PromQL or Datadog query")
    description: str = Field(default="")
    threshold: str = Field(default="")
    duration: str = Field(default="5m")
    notification: str = Field(default="")
    runbook_link: str = Field(default="")
    repo: str = Field(default="", description="Repo for Prometheus commit (optional)")
    branch: str = Field(default="", description="Branch for Prometheus commit (optional)")


# ── Dashboard Endpoints ──────────────────────────────────────────────────

@router.post("/dashboards/analyze")
async def analyze_dashboards(request: AnalyzeRequest, _=Depends(require_api_key)) -> dict[str, Any]:
    """
    Run the Dashboard Agent on a PR to generate dashboard suggestions.
    Returns suggestions that can then be created via /dashboards/create.
    """
    logger.info("dashboard_analysis_started", repo=request.repo, pr=request.pr_number)

    try:
        # Fetch PR data
        gh = GitHubService(installation_id=request.installation_id)
        diff = await gh.fetch_pr_diff(request.repo, request.pr_number)
        changed_files = await gh.fetch_pr_files(request.repo, request.pr_number)
        pr_details = await gh.fetch_pr_details(request.repo, request.pr_number)

        # Build state for the agent
        state = ReviewState(
            pr_title=pr_details.get("title", ""),
            pr_description=pr_details.get("body", ""),
            repo_full_name=request.repo,
            pr_number=request.pr_number,
            changed_files=changed_files,
            diff_data={"full_diff": diff},
        )

        # Populate context so the agent has full repo awareness
        from agents.context_fetcher import context_fetcher_agent
        ctx_updates = await context_fetcher_agent(state)
        for k, v in ctx_updates.items():
            if hasattr(state, k):
                setattr(state, k, v)

        # Run dashboard agent
        agent = DashboardAgent()
        result = await agent.run(state)

        # Post suggestions as PR comments
        suggestions = result.get("suggestions", [])
        if suggestions:
            comment_service = CommentService(installation_id=request.installation_id)
            await comment_service.post_dashboard_suggestions(
                request.repo, request.pr_number, suggestions
            )

        logger.info(
            "dashboard_analysis_complete",
            repo=request.repo,
            pr=request.pr_number,
            suggestions=len(suggestions),
        )

        return {
            "status": "success",
            "suggestions": suggestions,
            "summary": result.get("summary", ""),
        }

    except Exception as e:
        logger.error("dashboard_analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboards/create")
async def create_dashboard(request: CreateDashboardRequest, _=Depends(require_api_key)) -> dict[str, Any]:
    """
    Create a dashboard in the specified platform (Grafana/Datadog/Amplitude).
    Called by the prism-dashboard-creation.yml GitHub Actions workflow.
    """
    logger.info("dashboard_creation_started", name=request.name, type=request.type)

    service = DashboardService()
    result = await service.create_dashboard(request.model_dump())

    if "error" in result:
        logger.error("dashboard_creation_failed", name=request.name, error=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])

    logger.info("dashboard_creation_complete", name=request.name)
    return result


# ── Alert Endpoints ──────────────────────────────────────────────────────

@router.post("/alerts/analyze")
async def analyze_alerts(request: AnalyzeRequest, _=Depends(require_api_key)) -> dict[str, Any]:
    """
    Run the Alert Agent on a PR to generate alert suggestions.
    Returns suggestions that can then be created via /alerts/create.
    """
    logger.info("alert_analysis_started", repo=request.repo, pr=request.pr_number)

    try:
        gh = GitHubService(installation_id=request.installation_id)
        diff = await gh.fetch_pr_diff(request.repo, request.pr_number)
        changed_files = await gh.fetch_pr_files(request.repo, request.pr_number)
        pr_details = await gh.fetch_pr_details(request.repo, request.pr_number)

        state = ReviewState(
            pr_title=pr_details.get("title", ""),
            pr_description=pr_details.get("body", ""),
            repo_full_name=request.repo,
            pr_number=request.pr_number,
            changed_files=changed_files,
            diff_data={"full_diff": diff},
        )

        # Populate context so the agent has full repo awareness
        from agents.context_fetcher import context_fetcher_agent
        ctx_updates = await context_fetcher_agent(state)
        for k, v in ctx_updates.items():
            if hasattr(state, k):
                setattr(state, k, v)

        agent = AlertAgent()
        result = await agent.run(state)

        suggestions = result.get("suggestions", [])
        if suggestions:
            comment_service = CommentService(installation_id=request.installation_id)
            await comment_service.post_alert_suggestions(
                request.repo, request.pr_number, suggestions
            )

        logger.info(
            "alert_analysis_complete",
            repo=request.repo,
            pr=request.pr_number,
            suggestions=len(suggestions),
        )

        return {
            "status": "success",
            "suggestions": suggestions,
            "summary": result.get("summary", ""),
        }

    except Exception as e:
        logger.error("alert_analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/create")
async def create_alert(request: CreateAlertRequest, _=Depends(require_api_key)) -> dict[str, Any]:
    """
    Create an alert in the specified platform (Prometheus/Datadog).
    Called by the prism-alert-creation.yml GitHub Actions workflow.

    For Prometheus alerts, if repo and branch are provided, the alert rule
    YAML is committed directly to the PR branch.
    """
    logger.info("alert_creation_started", name=request.name, type=request.type)

    service = AlertService()
    result = await service.create_alert(
        suggestion=request.model_dump(),
        repo_full_name=request.repo or None,
        pr_branch=request.branch or None,
    )

    if "error" in result:
        logger.error("alert_creation_failed", name=request.name, error=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])

    logger.info("alert_creation_complete", name=request.name)
    return result
