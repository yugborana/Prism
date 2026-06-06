"""
Prism Security Review Agent.

Inherits BaseReviewAgent, defines security-specific system prompt
and 4-step reasoning chain targeting OWASP vulnerabilities.
"""

from agents.base_agent import BaseReviewAgent
from agents.prompts import (
    SECURITY_ANALYZE,
    SECURITY_CRITIQUE,
    SECURITY_GENERATE,
    SECURITY_REFINE,
)
from agents.reasoning import ReasoningChain, ReasoningStep
from agents.schemas import SecurityReport


class SecurityAgent(BaseReviewAgent):
    """Reviews code changes for security vulnerabilities."""

    ROLE = "Security"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert application security engineer. "
            "You specialize in identifying OWASP Top 10 vulnerabilities, "
            "injection attacks, authentication flaws, and data exposure risks. "
            "You review code diffs and report only real, actionable vulnerabilities "
            "with exact file paths and line numbers. Never report false positives."
        )

    def build_reasoning_chain(self) -> ReasoningChain:
        return ReasoningChain(
            steps=[
                ReasoningStep(name="analyze", prompt_template=SECURITY_ANALYZE),
                ReasoningStep(name="generate", prompt_template=SECURITY_GENERATE),
                ReasoningStep(name="critique", prompt_template=SECURITY_CRITIQUE, temperature=0.1),
                ReasoningStep(name="refine", prompt_template=SECURITY_REFINE),
            ],
            output_schema=SecurityReport,
        )
