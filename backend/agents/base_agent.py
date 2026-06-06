"""
Prism BaseReviewAgent — Abstract Base Class for All Review Agents.

Focused purely on code review responsibilities:
  1. Multi-provider LLM calls with retry
  2. ReasoningChain integration (4-step deliberation)
  3. Clean abstract interface for concrete agents
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from agents.reasoning import ReasoningChain
from agents.schemas import ReviewState
from observability.logging import get_logger
from utils.config import settings
from utils.llm_factory import LLMClient

logger = get_logger(__name__)


def _clean_json_response(text: str) -> str:
    """Strip Markdown fences (```json ... ```) from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class BaseReviewAgent(ABC):
    """
    Abstract base class for all Prism review agents.

    Each concrete agent (SecurityAgent, QualityAgent, PerformanceAgent)
    inherits this and defines:
      - ROLE: str
      - system_prompt: property
      - build_reasoning_chain(): returns a configured ReasoningChain
    """

    ROLE: str = "BaseAgent"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        provider: str | None = None,
    ):
        self.provider = provider or settings.llm_provider
        self._llm_client = llm_client or LLMClient(provider=self.provider)

        logger.info(
            "agent_initialized",
            role=self.ROLE,
            provider=self.provider,
            model=self._llm_client.model,
        )

    # ── Abstract Interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Each agent defines its domain-specific system prompt."""
        ...

    @abstractmethod
    def build_reasoning_chain(self) -> ReasoningChain:
        """
        Each agent defines its own 4-step reasoning chain
        with domain-specific step templates and output schema.
        """
        ...

    # ── Main Execution ────────────────────────────────────────────────────

    async def run(self, state: ReviewState, **kwargs: Any) -> dict[str, Any]:
        """
        Execute this agent's review using the ReasoningChain.

        Args:
            state: The shared ReviewState with context and diff data.

        Returns:
            Dict with the agent's structured report.
        """
        chain = self.build_reasoning_chain()

        try:
            result = await chain.execute(
                call_llm=self.call_llm,
                diff=state.diff_data.get("full_diff", ""),
                context=state.comprehensive_context,
                pr_title=state.pr_title,
                changed_files=", ".join(state.changed_files),
            )
            return result

        except Exception as e:
            logger.error("agent_run_failed", agent=self.ROLE, error=str(e))
            raise

    # ── LLM Interface ─────────────────────────────────────────────────────

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
        response_format: str | None = None,
    ) -> str:
        """
        Unified LLM call. Signature matches what ReasoningChain expects.
        Retry is handled by _execute_with_retry().
        """
        text = await self._execute_with_retry(
            messages, temperature, max_tokens, response_format
        )
        return text

    @retry(
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _execute_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: str | None,
    ) -> str:
        """Execute the LLM call with automatic retry on transient failures."""
        prompt = "\n".join(m["content"] for m in messages)

        system = self.system_prompt
        if response_format == "json_object":
            system += (
                "\n\nIMPORTANT: Return ONLY a valid JSON object. "
                "Do not include markdown formatting like ```json or any other text."
            )

        try:
            response_text = await self._llm_client.generate(
                prompt=prompt,
                system_prompt=system,
            )

            if response_format == "json_object":
                response_text = _clean_json_response(response_text)

            logger.debug(
                "llm_call_completed",
                agent=self.ROLE,
                response_length=len(response_text),
            )
            return response_text

        except Exception as e:
            logger.error(
                "llm_call_failed",
                agent=self.ROLE,
                provider=self.provider,
                error=str(e),
            )
            raise
