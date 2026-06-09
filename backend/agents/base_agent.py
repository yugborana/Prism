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


from agents.reasoning import ReasoningChain
from agents.schemas import ReviewState
from observability.logging import get_logger
from utils.config import settings
from utils.llm_factory import LLMClient
from observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


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
            # Format static analysis as a concise JSON string for prompts
            static_analysis_str = ""
            if state.static_analysis:
                static_analysis_str = json.dumps(state.static_analysis, indent=2)

            # Build annotated diff with clear line numbers
            raw_diff = state.diff_data.get("full_diff", "")
            try:
                from services.diff_parser import build_annotated_diff

                annotated_diff = build_annotated_diff(raw_diff)
            except Exception:
                annotated_diff = raw_diff  # Fallback to raw diff

            # Build file context string
            file_context = state.file_context or "(Full file context not available)"

            result = await chain.execute(
                call_llm=self.call_llm,
                diff=annotated_diff,
                context=state.comprehensive_context,
                cross_file_context=state.cross_file_context,
                file_context=file_context,
                static_analysis=static_analysis_str or "(No static analysis results available)",
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
        Retries, backoff, and fallbacks are now handled by the LiteLLM Gateway.
        """
        prompt = "\n".join(m["content"] for m in messages)

        system = self.system_prompt
        if response_format == "json_object":
            system += (
                "\n\nIMPORTANT: Return ONLY a valid JSON object. "
                "Do not include markdown formatting like ```json or any other text."
            )

        # OTel span for every LLM call — shows model, agent, and token usage
        with tracer.start_as_current_span(
            "prism.llm.call",
            attributes={
                "llm.agent": self.ROLE,
                "llm.model": self._llm_client.model,
                "llm.provider": self.provider,
                "llm.temperature": temperature,
                "llm.max_tokens": max_tokens,
            },
        ) as llm_span:
            try:
                response_text = await self._llm_client.generate(
                    prompt=prompt,
                    system_prompt=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if response_format == "json_object":
                    response_text = _clean_json_response(response_text)

                llm_span.set_attribute("llm.response_length", len(response_text))
                logger.debug(
                    "llm_call_completed",
                    agent=self.ROLE,
                    response_length=len(response_text),
                )
                return response_text

            except Exception as e:
                llm_span.set_attribute("llm.status", "error")
                llm_span.record_exception(e)
                logger.error(
                    "llm_call_failed",
                    agent=self.ROLE,
                    provider=self.provider,
                    error=str(e),
                )
                raise
