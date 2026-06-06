"""
Prism Alert Generation Agent.

Analyzes PR diffs to identify critical code paths and telemetry,
then suggests Prometheus PromQL alert rules and Datadog monitors
with priorities (P0/P1/P2), thresholds, durations, and runbook links.

Uses the 4-step reasoning chain (analyze → generate → critique → refine)
to avoid noisy alerts and ensure queries reference real metrics.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    ALERT_ANALYZE,
    ALERT_CRITIQUE,
    ALERT_GENERATE,
    ALERT_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import AlertReport

class AlertAgent(BaseReviewAgent):
    """Generates alert suggestions based on telemetry in the PR diff."""

    ROLE = "Alert"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert SRE specializing in alert rule design for "
            "Prometheus and Datadog. You analyze code diffs to identify "
            "critical code paths, error patterns, and performance bottlenecks, "
            "then generate actionable alert rules with PromQL queries, "
            "appropriate thresholds, durations, and runbook links. You prioritize "
            "alerts to minimize noise: P0 for critical issues, P1 for warnings, "
            "P2 for informational. You never suggest alerts for metrics that "
            "don't exist in the code."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=ALERT_ANALYZE),
                ReasoningStep(name="generate", prompt_template=ALERT_GENERATE),
                ReasoningStep(name="critique", prompt_template=ALERT_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=ALERT_REFINE),
            ],
            output_schema=AlertReport,
        )
