"""
Prism Background Tasks.

This file defines the Celery tasks that run in the worker processes.
The primary task 'process_pr_review' orchestrates the entire agentic pipeline.
"""

import asyncio
import uuid
from typing import Any

from workers.celery_app import celery_app
from orchestrator.engine import ReviewOrchestrator
from observability.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(name="process_pr_review", bind=True, max_retries=3)
def process_pr_review(self, pr_data: dict[str, Any]):
    """
    Background task to run the full agentic review pipeline.
    
    Args:
        pr_data: Dictionary containing PR details (diff, title, etc)
    """
    review_id = str(uuid.uuid4())
    logger.info("celery_task_started", task="process_pr_review", review_id=review_id)
    
    # Celery tasks are synchronous by default, so we run the async orchestrator
    # using a dedicated event loop. Always create a new one — Celery worker
    # processes don't have a running asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        orchestrator = ReviewOrchestrator(review_id=review_id)
        # Run the full review
        result_state = loop.run_until_complete(orchestrator.run_review(pr_data))
        
        # Post the final aggregated review back to GitHub (Bug 1 Fix)
        if result_state.final_review:
            try:
                from services.github_service import GitHubService
                github_service = GitHubService()
                logger.info(
                    "posting_review_to_github",
                    review_id=review_id,
                    repo=result_state.repo_full_name,
                    pr=result_state.pr_number,
                )
                loop.run_until_complete(
                    github_service.post_review(
                        repo_full_name=result_state.repo_full_name,
                        pr_number=result_state.pr_number,
                        review_data=result_state.final_review.model_dump(),
                    )
                )
                logger.info("review_posted_to_github_successfully", review_id=review_id)
            except Exception as post_err:
                logger.error("github_post_review_failed", review_id=review_id, error=str(post_err))

            # Post dashboard/alert suggestions if observability found issues
            try:
                from services.comment_service import CommentService
                from agents.dashboard_agent import DashboardAgent
                from agents.alert_agent import AlertAgent

                comment_svc = CommentService()
                repo = result_state.repo_full_name
                pr_num = result_state.pr_number

                # Run Dashboard Agent and post suggestions
                try:
                    dashboard_agent = DashboardAgent()
                    dash_result = loop.run_until_complete(dashboard_agent.run(result_state))
                    dash_suggestions = dash_result.get("suggestions", [])
                    if dash_suggestions:
                        loop.run_until_complete(
                            comment_svc.post_dashboard_suggestions(repo, pr_num, dash_suggestions)
                        )
                        logger.info("dashboard_suggestions_posted", count=len(dash_suggestions))
                except Exception as dash_err:
                    logger.warning("dashboard_suggestions_failed", error=str(dash_err))

                # Run Alert Agent and post suggestions
                try:
                    alert_agent = AlertAgent()
                    alert_result = loop.run_until_complete(alert_agent.run(result_state))
                    alert_suggestions = alert_result.get("suggestions", [])
                    if alert_suggestions:
                        loop.run_until_complete(
                            comment_svc.post_alert_suggestions(repo, pr_num, alert_suggestions)
                        )
                        logger.info("alert_suggestions_posted", count=len(alert_suggestions))
                except Exception as alert_err:
                    logger.warning("alert_suggestions_failed", error=str(alert_err))

            except Exception as suggestion_err:
                logger.warning("suggestion_posting_failed", error=str(suggestion_err))

        else:
            logger.warning("no_final_review_available_to_post", review_id=review_id)

        logger.info(
            "celery_task_finished",
            review_id=review_id,
            issues=result_state.final_review.total_issues if result_state.final_review else 0,
        )
        
        return {
            "review_id": review_id,
            "status": "success",
            "total_issues": result_state.final_review.total_issues if result_state.final_review else 0
        }

    except Exception as exc:
        import json
        from pydantic import ValidationError
        
        # Classify and separate permanent errors from transient operational failures (Bug 5 Fix)
        permanent_exceptions = (
            ValidationError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        )
        
        if isinstance(exc, permanent_exceptions):
            logger.error(
                "celery_task_failed_permanent",
                review_id=review_id,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            return {
                "review_id": review_id,
                "status": "failed_permanent",
                "error": f"{exc.__class__.__name__}: {str(exc)}",
            }
            
        logger.error(
            "celery_task_failed_transient",
            review_id=review_id,
            error_type=exc.__class__.__name__,
            error=str(exc),
            retries=self.request.retries,
        )
        # Exponential backoff for transient retries (network dropouts, LLM timeouts, Qdrant/Postgres down)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

    finally:
        loop.close()
