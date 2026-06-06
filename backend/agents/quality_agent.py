"""
Prism Code Quality Review Agent.

Inherits BaseReviewAgent, defines quality-specific system prompt
and 4-step reasoning chain targeting bugs, smells, and design issues.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    QUALITY_ANALYZE,
    QUALITY_CRITIQUE,
    QUALITY_GENERATE,
    QUALITY_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import QualityReport


class QualityAgent(BaseReviewAgent):
    """Reviews code changes for bugs, code smells, and design violations."""

    ROLE = "Quality"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior software engineer specializing in code quality. "
            "You find real bugs, logic errors, type mismatches, and design violations. "
            "You focus on issues that will cause runtime failures or make code unmaintainable. "
            "You always provide exact file paths, line numbers, and concrete fixes."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=QUALITY_ANALYZE),
                ReasoningStep(name="generate", prompt_template=QUALITY_GENERATE),
                ReasoningStep(name="critique", prompt_template=QUALITY_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=QUALITY_REFINE),
            ],
            output_schema=QualityReport,
        )
