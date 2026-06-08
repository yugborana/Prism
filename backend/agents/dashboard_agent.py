"""
Prism Dashboard Generation Agent.

Analyzes PR diffs to identify telemetry instrumentation and suggests
dashboards for Grafana, Datadog, and Amplitude with complete
API-ready JSON configurations.

Uses the 4-step reasoning chain (analyze → generate → critique → refine)
to ensure dashboard suggestions reference actual telemetry from the diff.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    DASHBOARD_ANALYZE,
    DASHBOARD_CRITIQUE,
    DASHBOARD_GENERATE,
    DASHBOARD_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import DashboardReport


class DashboardAgent(BaseReviewAgent):
    """Generates dashboard suggestions based on telemetry in the PR diff."""

    ROLE = "Dashboard"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert observability engineer specializing in dashboard "
            "creation for Grafana, Datadog, and Amplitude. You analyze code diffs "
            "to identify OpenTelemetry spans, Prometheus metrics, structured logs, "
            "and analytics events, then generate complete dashboard configurations "
            "with valid queries, panels, and alert rules. You only suggest dashboards "
            "for telemetry that actually exists in the code."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=DASHBOARD_ANALYZE),
                ReasoningStep(name="generate", prompt_template=DASHBOARD_GENERATE),
                ReasoningStep(name="critique", prompt_template=DASHBOARD_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=DASHBOARD_REFINE),
            ],
            output_schema=DashboardReport,
        )
