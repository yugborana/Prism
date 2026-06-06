"""
Prism Performance Review Agent.

Inherits BaseReviewAgent, defines performance-specific system prompt
and 4-step reasoning chain targeting bottlenecks and inefficiencies.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    PERFORMANCE_ANALYZE,
    PERFORMANCE_CRITIQUE,
    PERFORMANCE_GENERATE,
    PERFORMANCE_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import PerformanceReport


class PerformanceAgent(BaseReviewAgent):
    """Reviews code changes for performance bottlenecks and inefficiencies."""

    ROLE = "Performance"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a performance engineer who identifies bottlenecks in code. "
            "You focus on algorithm complexity, N+1 queries, memory leaks, "
            "unnecessary I/O, and missed caching opportunities. "
            "You only flag issues that would cause measurable slowdowns at scale. "
            "You always provide exact file paths, line numbers, and optimized alternatives."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=PERFORMANCE_ANALYZE),
                ReasoningStep(name="generate", prompt_template=PERFORMANCE_GENERATE),
                ReasoningStep(name="critique", prompt_template=PERFORMANCE_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=PERFORMANCE_REFINE),
            ],
            output_schema=PerformanceReport,
        )
