"""
Prism Orchestrator Engine.

Source: Proximus orchestrator/planner.py (adapted for PR review)

Central coordinator that:
1. Builds the review DAG (ContextFetcher -> [Security, Quality, Performance] -> Aggregator)
2. Executes tasks in parallel where possible
3. Manages shared ReviewState
4. Maintains ReviewMemory for cross-agent awareness
5. Logs decisions for audit/debugging
6. Saves audit trail to PostgreSQL for permanent storage
7. Emits Prometheus metrics for observability
"""

import asyncio
import time
from typing import Any, cast

from agents.schemas import (
    AgentStatus,
    ObservabilityReport,
    PerformanceReport,
    QualityReport,
    ReviewState,
    SecurityReport,
)
from agents.context_fetcher import context_fetcher_agent
from agents.security_agent import SecurityAgent
from agents.quality_agent import QualityAgent
from agents.performance_agent import PerformanceAgent
from agents.observability_agent import ObservabilityAgent
from agents.aggregator import aggregator_agent
from orchestrator.task_graph import Task, TaskGraph, TaskStatus
from orchestrator.memory.review_memory import ReviewMemory
from orchestrator.memory.decision_log import ReviewDecisionLog

from observability.logging import get_logger
from db.postgres import get_db_session
from db.repository import ReviewRepository
from observability.metrics import (
    track_review,
    track_agent_task,
    findings_total,
)
from observability.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ReviewOrchestrator:
    """
    Manages the lifecycle of a single PR review.

    Like Proximus OrchestratorEngine, this initializes shared memory systems
    at bootstrap time and passes them through the execution context.
    """

    def __init__(self, review_id: str):
        self.review_id = review_id
        self.state = ReviewState()

        # ── Memory Systems (adapted from Proximus) ────────────────────
        redis_client = self._connect_redis()
        self.memory = ReviewMemory(review_id, redis_client=redis_client)
        self.decision_log = ReviewDecisionLog(review_id)

        # ── Agent Registry ────────────────────────────────────────────
        self.agents = {
            "Security": SecurityAgent(),
            "Quality": QualityAgent(),
            "Performance": PerformanceAgent(),
            "Observability": ObservabilityAgent(),
        }

    @staticmethod
    def _connect_redis():
        """Get the shared async Redis client for ReviewMemory's L2 cache.

        Returns None if the pool isn't initialized — ReviewMemory will
        gracefully fall back to L1 (in-process dict) only.
        """
        try:
            from utils.connections import get_redis

            redis = get_redis()
            if redis is None:
                logger.debug("redis_memory_pool_not_initialized_using_l1")
            return redis
        except Exception as e:
            logger.warning("redis_memory_connect_failed", error=str(e))
            return None

    async def run_review(self, pr_data: dict[str, Any]) -> ReviewState:
        """
        Full review lifecycle execution.
        """
        start_time = time.monotonic()

        # 1. Initialize State
        self.state.pr_title = pr_data.get("title", "")
        self.state.pr_description = pr_data.get("body", "")
        self.state.repo_full_name = pr_data.get("repo_name", "")
        self.state.pr_number = pr_data.get("number", 0)
        self.state.installation_id = pr_data.get("installation_id", 0)
        self.state.head_sha = pr_data.get("head_sha", "")
        self.state.changed_files = pr_data.get("changed_files", [])
        self.state.diff_data = {"full_diff": pr_data.get("diff", "")}
        self.state.has_repo_index = pr_data.get("has_repo_index", False)

        with tracer.start_as_current_span(
            "prism.orchestrator.run_review",
            attributes={
                "review.id": self.review_id,
                "github.repo": self.state.repo_full_name,
                "github.pr": self.state.pr_number,
                "review.files_count": len(self.state.changed_files),
            },
        ) as review_span:
            # 1a. Load PRD content if configured
            prd_content = ""
            try:
                from utils.config import settings as _settings

                if _settings.prd_file_path:
                    import os

                    if os.path.isfile(_settings.prd_file_path):

                        def _read_prd():
                            with open(_settings.prd_file_path, "r", encoding="utf-8") as f:
                                return f.read()

                        prd_content = await asyncio.to_thread(_read_prd)
                        logger.info("prd_loaded", path=_settings.prd_file_path, size=len(prd_content))
                    else:
                        logger.warning("prd_file_not_found", path=_settings.prd_file_path)
            except Exception as e:
                logger.warning("prd_load_failed", error=str(e))

            # Store PRD so it can be appended to context after ContextFetcher runs
            self._prd_content = prd_content

            # Store PR context in memory for cross-agent access
            await self.memory.set(
                "pr_context",
                {
                    "title": self.state.pr_title,
                    "description": self.state.pr_description,
                    "repo": self.state.repo_full_name,
                    "files": self.state.changed_files,
                },
            )

            # Log the review kickoff decision
            self.decision_log.log(
                agent_role="Orchestrator",
                decision_type="review_started",
                description=f"Review started for PR #{self.state.pr_number}",
                rationale=f"PR event received for {self.state.repo_full_name}",
                confidence=1.0,
                metadata={"files_count": len(self.state.changed_files)},
            )

            # 1b. Index PR diff into Qdrant so ContextFetcher has data to retrieve
            try:
                from services.vector_indexer import VectorIndexer

                indexer = VectorIndexer()
                index_counts = await indexer.index_pr_diff(
                    diff=pr_data.get("diff", ""),
                    changed_files=self.state.changed_files,
                    repo_name=self.state.repo_full_name,
                    pr_number=self.state.pr_number,
                )
                logger.info("vector_index_complete", **index_counts)
            except Exception as e:
                logger.warning("vector_index_failed", error=str(e))

            # 2. Build DAG
            graph = self._build_graph()

            logger.info(
                "review_started",
                review_id=self.review_id,
                pr=self.state.pr_number,
                tasks=len(graph.tasks),
            )

            # 3. Execute DAG (with Prometheus tracking)
            with track_review(self.state.repo_full_name):
                while not graph.is_complete():
                    ready_tasks = graph.get_ready_tasks()
                    if not ready_tasks:
                        # Deadlock protection: if nothing is ready but the graph
                        # isn't complete, skip all remaining PENDING tasks to
                        # prevent an infinite spin.
                        pending = [t for t in graph.tasks.values() if t.status == TaskStatus.PENDING]
                        if pending:
                            for t in pending:
                                logger.warning("task_skipped_deadlock", task=t.name)
                                t.status = TaskStatus.SKIPPED
                        break

                    await asyncio.gather(*[self._execute_task(task, graph) for task in ready_tasks])

                # 5. Log completion
                duration_ms = int((time.monotonic() - start_time) * 1000)
                summary = graph.get_status_summary()
                self.decision_log.log(
                    agent_role="Orchestrator",
                    decision_type="review_completed",
                    description=f"Review pipeline finished: {summary}",
                    rationale="All DAG tasks resolved",
                    confidence=1.0,
                )

                import json

                review_span.set_attribute("review.duration_ms", duration_ms)
                review_span.set_attribute("review.dag_summary", json.dumps(summary))

                logger.info(
                    "review_completed",
                    review_id=self.review_id,
                    dag_summary=summary,
                    duration_ms=duration_ms,
                    decision_summary=self.decision_log.summary(),
                )

                # 6. Persist audit trail to PostgreSQL
                await self._persist_audit_trail(duration_ms)

                # 7. Cleanup session memory
                await self.memory.cleanup()

                return self.state

    def _build_graph(self) -> TaskGraph:
        """
        Builds the standard review DAG:
        Fetcher -> (Security, Quality, Performance) -> Aggregator
        """
        graph = TaskGraph(self.review_id)

        # Task 1: Fetch Context
        fetcher = Task(name="Fetch Context", agent_role="ContextFetcher")
        f_id = graph.add_task(fetcher)

        # Parallel Tasks: Reviews
        sec = Task(name="Security Review", agent_role="Security", dependencies=[f_id])
        qual = Task(name="Quality Review", agent_role="Quality", dependencies=[f_id])
        perf = Task(name="Performance Review", agent_role="Performance", dependencies=[f_id])
        obs = Task(name="Observability Review", agent_role="Observability", dependencies=[f_id])

        s_id = graph.add_task(sec)
        q_id = graph.add_task(qual)
        p_id = graph.add_task(perf)
        o_id = graph.add_task(obs)

        # Task 3: Aggregate
        agg = Task(name="Aggregate Review", agent_role="Aggregator", dependencies=[s_id, q_id, p_id, o_id])
        graph.add_task(agg)

        return graph

    async def _execute_task(self, task: Task, graph: TaskGraph):
        """Execute a single task, log decisions, track metrics, and update state."""
        # Skip non-Aggregator tasks whose dependencies failed.
        # The Aggregator is special — it must always run so it can produce
        # a partial review from whatever agents succeeded.
        if task.agent_role != "Aggregator" and graph.has_failed_dependency(task):
            logger.warning(
                "task_skipped_failed_dep",
                task=task.name,
                role=task.agent_role,
            )
            task.status = TaskStatus.SKIPPED
            status_key = f"{task.agent_role.lower()}_agent_status"
            self._apply_updates({status_key: AgentStatus.FAILED})
            self.decision_log.log(
                agent_role=task.agent_role,
                decision_type="task_skipped",
                description=f"{task.name} skipped: upstream dependency failed",
                rationale="Cannot proceed without required input",
                confidence=0.0,
            )
            return

        task.mark_started()
        logger.info("task_started", task=task.name, role=task.agent_role)

        # OTel span for each agent task — shows up as a child of run_review
        with tracer.start_as_current_span(
            f"prism.agent.{task.agent_role.lower()}",
            attributes={
                "agent.role": task.agent_role,
                "agent.task_name": task.name,
            },
        ) as agent_span:
            with track_agent_task(task.agent_role):
                try:
                    if task.agent_role == "ContextFetcher":
                        updates = await context_fetcher_agent(self.state)
                        self._apply_updates(updates)

                        # Append PRD content to context
                        if self._prd_content:
                            self.state.comprehensive_context += (
                                "\n\n## Product Requirements Document (PRD)\n" + self._prd_content
                            )
                            logger.info("prd_injected_into_context")

                        # Log context fetcher decision
                        self.decision_log.log(
                            agent_role="ContextFetcher",
                            decision_type="context_fetched",
                            description=f"Fetched context for {len(self.state.changed_files)} files",
                            rationale="Required by downstream review agents",
                        )

                    elif task.agent_role in self.agents:
                        agent = self.agents[task.agent_role]
                        report_dict = await agent.run(self.state)

                        # Coerce raw dict into the proper Pydantic report model
                        # so the Aggregator can access .findings directly
                        report_model = self._coerce_report(task.agent_role, report_dict)

                        # Read findings from the validated model when available,
                        # falling back to the raw dict if coercion returned a dict.
                        if hasattr(report_model, "findings"):
                            findings = report_model.findings  # list[CodeFinding]
                        else:
                            findings = report_dict.get("findings", [])

                        for finding in findings:
                            # Share with other agents via memory
                            finding_dict = finding.model_dump() if hasattr(finding, "model_dump") else finding
                            await self.memory.share_finding(task.agent_role, finding_dict)
                            # Emit Prometheus metric per finding
                            sev = (
                                finding.severity.value
                                if hasattr(finding, "severity") and hasattr(finding.severity, "value")
                                else finding_dict.get("severity", "info")
                            )
                            findings_total.labels(
                                agent=task.agent_role,
                                severity=sev,
                            ).inc()

                        # Log the agent's review decision
                        self.decision_log.log(
                            agent_role=task.agent_role,
                            decision_type="review_completed",
                            description=f"{task.agent_role} found {len(findings)} issues",
                            rationale=report_dict.get("summary", "Analysis complete"),
                            confidence=0.85,
                            metadata={"findings_count": len(findings)},
                        )

                        # Update status and report in state (Pydantic model, not raw dict)
                        status_key = f"{task.agent_role.lower()}_agent_status"
                        report_key = f"{task.agent_role.lower()}_report"
                        self._apply_updates(
                            {
                                status_key: AgentStatus.COMPLETED,
                                report_key: report_model,
                            }
                        )

                    elif task.agent_role == "Aggregator":
                        updates = await aggregator_agent(self.state)
                        self._apply_updates(updates)

                        self.decision_log.log(
                            agent_role="Aggregator",
                            decision_type="aggregation_completed",
                            description="Merged all agent reports into final review",
                            rationale="All specialist agents completed",
                        )

                    task.mark_completed({})
                    agent_span.set_attribute("agent.status", "completed")
                    logger.info("task_completed", task=task.name)

                except Exception as e:
                    logger.error("task_failed", task=task.name, error=str(e))
                    task.mark_failed(str(e))
                    agent_span.set_attribute("agent.status", "failed")
                    agent_span.record_exception(e)

                    # Log the failure decision
                    self.decision_log.log(
                        agent_role=task.agent_role,
                        decision_type="task_failed",
                        description=f"{task.name} failed: {str(e)[:100]}",
                        rationale="Exception during execution",
                        confidence=0.0,
                    )

                    status_key = f"{task.agent_role.lower()}_agent_status"
                    self._apply_updates({status_key: AgentStatus.FAILED})

    def _apply_updates(self, updates: dict[str, Any]):
        """Helper to update Pydantic state via direct attribute setting."""
        for k, v in updates.items():
            if hasattr(self.state, k):
                setattr(self.state, k, v)

    @staticmethod
    def _coerce_report(agent_role: str, report_dict: dict) -> Any:
        """Convert a raw dict from the reasoning chain into the correct Pydantic report model."""
        schema_map = {
            "Security": SecurityReport,
            "Quality": QualityReport,
            "Performance": PerformanceReport,
            "Observability": ObservabilityReport,
        }
        schema = schema_map.get(agent_role)
        if schema is None:
            return report_dict
        try:
            return cast(Any, schema).model_validate(report_dict)
        except Exception:
            # If validation fails, return the raw dict as a last resort
            return report_dict

    # ── Postgres Audit Trail ──────────────────────────────────────────

    async def _persist_audit_trail(self, duration_ms: int):
        """
        Save the review record, decisions, and findings to PostgreSQL.
        Degrades gracefully if DB is unavailable (review still completes).
        """
        try:
            from utils.config import settings

            async with get_db_session() as session:
                repo = ReviewRepository(session)

                # Create the review record
                review_record = await repo.create_review(
                    review_id=self.review_id,
                    repo_name=self.state.repo_full_name,
                    pr_number=self.state.pr_number,
                    pr_title=self.state.pr_title,
                    files_changed=len(self.state.changed_files),
                    llm_provider=settings.llm_provider,
                    llm_model="prism-review",
                )

                # Persist all decision log entries
                decision_entries = self.decision_log.all_entries()
                if decision_entries:
                    await repo.save_decisions(review_record.id, decision_entries)

                # Persist all findings from shared memory
                shared_findings = await self.memory.get_shared_findings()
                if shared_findings:
                    await repo.save_findings(
                        review_record.id,
                        [
                            {
                                "agent": f["agent"],
                                **f["finding"],
                            }
                            for f in shared_findings
                        ],
                    )

                # Mark the review as completed
                await repo.complete_review(
                    review_id=self.review_id,
                    status="completed",
                    findings_count=len(shared_findings),
                    duration_ms=duration_ms,
                )

                logger.info(
                    "audit_trail_persisted",
                    review_id=self.review_id,
                    decisions=len(decision_entries),
                    findings=len(shared_findings),
                )

        except Exception as e:
            # Don't fail the review if audit persistence fails
            logger.warning(
                "audit_trail_persist_failed",
                review_id=self.review_id,
                error=str(e),
            )
