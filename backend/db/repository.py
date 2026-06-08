"""
Prism Review Repository — Postgres CRUD for Audit Trail.

Writes review records, decisions, and findings to PostgreSQL.
Called by the ReviewOrchestrator after each review completes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DecisionRecord_DB, FindingRecord, ReviewRecord
from observability.logging import get_logger

logger = get_logger(__name__)


class ReviewRepository:
    """Async repository for permanent review audit storage."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # ── Reviews ───────────────────────────────────────────────────────

    async def create_review(
        self,
        review_id: str,
        repo_name: str,
        pr_number: int,
        pr_title: str = "",
        files_changed: int = 0,
        llm_provider: str = "",
        llm_model: str = "",
    ) -> ReviewRecord:
        record = ReviewRecord(
            review_id=review_id,
            repo_name=repo_name,
            pr_number=pr_number,
            pr_title=pr_title,
            status="running",
            files_changed=files_changed,
            llm_provider=llm_provider,
            llm_model=llm_model,
            started_at=datetime.now(UTC),
        )
        self._session.add(record)
        await self._session.flush()
        logger.debug("review_record_created", review_id=review_id)
        return record

    async def complete_review(
        self,
        review_id: str,
        status: str = "completed",
        findings_count: int = 0,
        duration_ms: int = 0,
    ) -> None:
        result = await self._session.execute(select(ReviewRecord).where(ReviewRecord.review_id == review_id))
        record = result.scalar_one_or_none()
        if record:
            record.status = status
            record.findings_count = findings_count
            record.duration_ms = duration_ms
            record.completed_at = datetime.now(UTC)
            logger.debug("review_record_completed", review_id=review_id, status=status)

    async def get_reviews_by_repo(self, repo_name: str, limit: int = 20) -> list[ReviewRecord]:
        result = await self._session.execute(
            select(ReviewRecord)
            .where(ReviewRecord.repo_name == repo_name)
            .order_by(ReviewRecord.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_review_stats(self, repo_name: str) -> dict[str, Any]:
        """Aggregate stats for a repo — useful for dashboards."""
        total = await self._session.execute(select(func.count()).where(ReviewRecord.repo_name == repo_name))
        avg_duration = await self._session.execute(
            select(func.avg(ReviewRecord.duration_ms))
            .where(ReviewRecord.repo_name == repo_name)
            .where(ReviewRecord.status == "completed")
        )
        avg_findings = await self._session.execute(
            select(func.avg(ReviewRecord.findings_count)).where(ReviewRecord.repo_name == repo_name)
        )
        return {
            "total_reviews": total.scalar() or 0,
            "avg_duration_ms": round(avg_duration.scalar() or 0, 1),
            "avg_findings": round(avg_findings.scalar() or 0, 1),
        }

    # ── Decisions ─────────────────────────────────────────────────────

    async def save_decisions(self, db_review_id, decisions: list[dict[str, Any]]) -> int:
        """Bulk-insert decision log entries."""
        records = []
        for d in decisions:
            records.append(
                DecisionRecord_DB(
                    review_id=db_review_id,
                    agent_role=d.get("agent", ""),
                    decision_type=d.get("type", ""),
                    description=d.get("description", "")[:500],
                    rationale=d.get("rationale", "")[:500],
                    confidence=d.get("confidence", 1.0),
                    metadata_json=d.get("metadata", {}),
                )
            )
        self._session.add_all(records)
        await self._session.flush()
        logger.debug("decisions_saved", count=len(records))
        return len(records)

    # ── Findings ──────────────────────────────────────────────────────

    async def save_findings(self, db_review_id, findings: list[dict[str, Any]]) -> int:
        """Bulk-insert agent findings."""
        records = []
        for f in findings:
            records.append(
                FindingRecord(
                    review_id=db_review_id,
                    agent_role=f.get("agent", ""),
                    severity=f.get("severity", "info"),
                    file_path=f.get("file_path", ""),
                    line_number=f.get("line_number"),
                    title=f.get("category", f.get("title", "")),
                    description=f.get("description", "")[:2000],
                    suggestion=f.get("suggested_fix", f.get("suggestion", ""))[:2000],
                )
            )
        self._session.add_all(records)
        await self._session.flush()
        logger.debug("findings_saved", count=len(records))
        return len(records)

    async def get_findings_by_severity(self, repo_name: str) -> dict[str, int]:
        """Findings breakdown by severity — for Grafana dashboards."""
        result = await self._session.execute(
            select(FindingRecord.severity, func.count())
            .join(ReviewRecord)
            .where(ReviewRecord.repo_name == repo_name)
            .group_by(FindingRecord.severity)
        )
        return dict(result.all())
