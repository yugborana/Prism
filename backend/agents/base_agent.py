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
        Execute this agent's review using per-file strategy.

        Files are reviewed SEQUENTIALLY within each agent to avoid
        rate-limit explosions (4 agents already run in parallel via
        the DAG — adding file-level parallelism caused 8+ concurrent
        LLM streams which exceeded Groq's 30 RPM limit).

        Falls back to single-pass for single-file PRs or if splitting fails.
        """
        try:
            from services.diff_parser import split_diff_by_file

            raw_diff = state.diff_data.get("full_diff", "")
            per_file_diffs = split_diff_by_file(raw_diff)

            # Single file or empty → use single-pass (no overhead)
            if len(per_file_diffs) <= 1:
                return await self._run_single_pass(state, raw_diff)

            # Multiple files → run per-file SEQUENTIALLY
            # (4 agents run in parallel via DAG — that's enough concurrency)
            logger.info(
                "per_file_review_start",
                agent=self.ROLE,
                file_count=len(per_file_diffs),
            )

            file_results: list[dict[str, Any]] = []
            failed_files: list[str] = []

            for file_path, file_diff in per_file_diffs.items():
                try:
                    result = await self._run_for_file(
                        file_path=file_path,
                        file_diff=file_diff,
                        state=state,
                    )
                    file_results.append(result)
                except Exception as file_err:
                    logger.error(
                        "per_file_review_failed",
                        agent=self.ROLE,
                        file=file_path,
                        error=str(file_err),
                    )
                    failed_files.append(file_path)
                    # Continue to next file — don't lose all findings
                    continue

            if not file_results:
                # All files failed — fall back to single-pass
                logger.warning(
                    "all_per_file_reviews_failed_fallback",
                    agent=self.ROLE,
                    failed_files=failed_files,
                )
                return await self._run_single_pass(state, raw_diff)

            # Merge results from all successful files
            merged = self._merge_file_results(file_results)

            logger.info(
                "per_file_review_complete",
                agent=self.ROLE,
                files_reviewed=len(file_results),
                files_failed=len(failed_files),
                total_findings=len(merged.get("findings", [])),
            )

            return merged

        except Exception as e:
            logger.error("agent_run_failed", agent=self.ROLE, error=str(e))
            raise

    async def _run_for_file(
        self,
        file_path: str,
        file_diff: str,
        state: ReviewState,
    ) -> dict[str, Any]:
        """Run the reasoning chain for a single file."""
        chain = self.build_reasoning_chain()

        try:
            from services.diff_parser import build_annotated_diff

            annotated_diff = build_annotated_diff(file_diff)
        except Exception:
            annotated_diff = file_diff

        # Get per-file context (full source or compressed)
        file_context = state.per_file_contexts.get(file_path, "")
        if not file_context:
            file_context = "(Full file context not available for this file)"

        # Format static analysis — filter to this file where possible
        static_analysis_str = ""
        if state.static_analysis:
            static_analysis_str = json.dumps(state.static_analysis, indent=2)

        result = await chain.execute(
            call_llm=self.call_llm,
            diff=annotated_diff,
            context=state.comprehensive_context,
            cross_file_context=state.cross_file_context,
            file_context=file_context,
            static_analysis=static_analysis_str or "(No static analysis results available)",
            pr_title=state.pr_title,
            changed_files=file_path,  # Only this file
        )

        logger.info(
            "per_file_review_done",
            agent=self.ROLE,
            file=file_path,
            findings=len(result.get("findings", [])),
        )
        return result

    async def _run_single_pass(
        self,
        state: ReviewState,
        raw_diff: str,
    ) -> dict[str, Any]:
        """Original single-pass review for single-file PRs."""
        chain = self.build_reasoning_chain()

        static_analysis_str = ""
        if state.static_analysis:
            static_analysis_str = json.dumps(state.static_analysis, indent=2)

        try:
            from services.diff_parser import build_annotated_diff

            annotated_diff = build_annotated_diff(raw_diff)
        except Exception:
            annotated_diff = raw_diff

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

    def _merge_file_results(
        self,
        file_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge per-file results into a single agent report.

        Concatenates findings lists and joins summaries.
        Only receives successful results (failures handled in run()).
        """
        merged_findings: list[dict] = []
        summaries: list[str] = []

        # Collect non-finding fields from the first result as template
        template: dict[str, Any] = {}

        for result in file_results:
            if not isinstance(result, dict):
                continue

            # Collect findings
            findings = result.get("findings", [])
            merged_findings.extend(findings)

            # Collect summary
            summary = result.get("summary", "")
            if summary:
                summaries.append(summary)

            # Use first result as template for non-list fields
            if not template:
                template = {k: v for k, v in result.items() if k not in ("findings", "summary")}

        # Build merged result
        merged = dict(template)
        merged["findings"] = merged_findings
        merged["summary"] = " | ".join(summaries) if summaries else "No findings."

        return merged

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
