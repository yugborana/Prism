"""
Prism Observability Instrumentation Agent.

Checks PR diffs for missing observability instrumentation:
- OpenTelemetry spans, attributes, and error tracking
- Logging at entry/exit/error points
- Metrics collection (counters, histograms, gauges)
- Event tracking (Amplitude / analytics)

Uses the 4-step reasoning chain (analyze → generate → critique → refine)
to produce high-quality suggestions constrained to the diff.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    OBSERVABILITY_ANALYZE,
    OBSERVABILITY_CRITIQUE,
    OBSERVABILITY_GENERATE,
    OBSERVABILITY_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import ObservabilityReport

class ObservabilityAgent(BaseReviewAgent):
    """Reviews code changes for missing observability instrumentation."""

    ROLE = "Observability"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert observability engineer specializing in "
            "OpenTelemetry instrumentation, structured logging, metrics collection, "
            "and event tracking. You review code diffs and identify missing or "
            "inadequate observability instrumentation. You suggest concrete code "
            "changes to add spans, logging, metrics, and event tracking. "
            "You only suggest changes to code visible in the diff and never "
            "introduce imports that aren't already present."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=OBSERVABILITY_ANALYZE),
                ReasoningStep(name="generate", prompt_template=OBSERVABILITY_GENERATE),
                ReasoningStep(name="critique", prompt_template=OBSERVABILITY_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=OBSERVABILITY_REFINE),
            ],
            output_schema=ObservabilityReport,
        )
