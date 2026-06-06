"""
Prism Decision Log — Immutable Audit Trail of Agent Decisions.

Source: Proximus orchestrator/memory/decision_log.py (adapted)

Records every decision made during a review:
- Which agent analyzed which file
- What severity it assigned and why
- What alternatives it considered (from the critique step)
- Confidence score from the reasoning chain

Used for:
1. Debugging — understand why an agent flagged or missed something
2. Observability — feed into dashboards and alerting
3. Continuous improvement — identify systematic false positive patterns
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import uuid

from observability.logging import get_logger

logger = get_logger(__name__)


class DecisionRecord:
    """A single decision made by an agent during review."""

    def __init__(
        self,
        agent_role: str,
        decision_type: str,  # finding, skip, severity_change, false_positive_removal
        description: str,
        rationale: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.agent_role = agent_role
        self.decision_type = decision_type
        self.description = description
        self.rationale = rationale
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent_role,
            "type": self.decision_type,
            "description": self.description[:200],
            "rationale": self.rationale[:200],
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class ReviewDecisionLog:
    """
    Append-only log of every agent decision for a single PR review.
    
    Enables:
    - Timeline replay of the review process
    - Filtering decisions by agent or type
    - Surfacing low-confidence decisions for human review
    """

    def __init__(self, review_id: str):
        self.review_id = review_id
        self._records: list[DecisionRecord] = []

    def log(
        self,
        agent_role: str,
        decision_type: str,
        description: str,
        rationale: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Log a decision and return its ID."""
        record = DecisionRecord(
            agent_role=agent_role,
            decision_type=decision_type,
            description=description,
            rationale=rationale,
            confidence=confidence,
            metadata=metadata,
        )
        self._records.append(record)
        logger.debug(
            "decision_logged",
            agent=agent_role,
            type=decision_type,
            desc=description[:80],
        )
        return record.id

    def get_by_agent(self, agent_role: str) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records if r.agent_role == agent_role]

    def get_timeline(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in sorted(self._records, key=lambda x: x.timestamp)]

    def get_low_confidence(self, threshold: float = 0.7) -> list[dict[str, Any]]:
        """Decisions that should be flagged for human attention."""
        return [r.to_dict() for r in self._records if r.confidence < threshold]

    def summary(self) -> dict[str, Any]:
        agent_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for r in self._records:
            agent_counts[r.agent_role] = agent_counts.get(r.agent_role, 0) + 1
            type_counts[r.decision_type] = type_counts.get(r.decision_type, 0) + 1

        return {
            "review_id": self.review_id,
            "total_decisions": len(self._records),
            "by_agent": agent_counts,
            "by_type": type_counts,
            "low_confidence_count": len(self.get_low_confidence()),
            "avg_confidence": (
                sum(r.confidence for r in self._records) / len(self._records)
                if self._records else 0.0
            ),
        }

    def all_entries(self) -> list[dict[str, Any]]:
        """Return all decision records as dicts — used for Postgres persistence."""
        return [
            {
                "agent": r.agent_role,
                "type": r.decision_type,
                "description": r.description,
                "rationale": r.rationale,
                "confidence": r.confidence,
                "metadata": r.metadata,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self._records
        ]
