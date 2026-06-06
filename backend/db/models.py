"""
Prism SQLAlchemy Models — Permanent Audit Trail.

Source: Proximus go-backend/migrations/001_init.up.sql (schema adapted for code review)

Tables:
- reviews:     One row per PR review (metadata, status, timing)
- decisions:   Immutable log of every agent decision during a review
- findings:    Individual code issues found by agents
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ReviewRecord(Base):
    """
    Permanent record of every PR review.
    Adapted from Proximus `projects` table — one review = one "project".
    """
    __tablename__ = "reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(String(64), unique=True, nullable=False, index=True)
    repo_name = Column(String(255), nullable=False)
    pr_number = Column(Integer, nullable=False)
    pr_title = Column(Text, default="")
    status = Column(
        String(20), nullable=False, default="pending",
        # pending → running → completed → failed
    )
    files_changed = Column(Integer, default=0)
    findings_count = Column(Integer, default=0)
    duration_ms = Column(Integer)
    llm_provider = Column(String(50))
    llm_model = Column(String(100))

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    decisions = relationship("DecisionRecord_DB", back_populates="review", cascade="all, delete-orphan")
    findings = relationship("FindingRecord", back_populates="review", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_reviews_repo_status", "repo_name", "status"),
        Index("idx_reviews_created", "created_at"),
    )


class DecisionRecord_DB(Base):
    """
    Immutable decision log — persisted version of in-memory DecisionLog.
    Adapted from Proximus `tasks` table — each decision = one task record.
    """
    __tablename__ = "decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False)
    agent_role = Column(String(50), nullable=False)
    decision_type = Column(String(50), nullable=False)
    description = Column(Text, default="")
    rationale = Column(Text, default="")
    confidence = Column(Float, default=1.0)
    metadata_json = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    review = relationship("ReviewRecord", back_populates="decisions")

    __table_args__ = (
        Index("idx_decisions_review", "review_id"),
        Index("idx_decisions_agent", "agent_role"),
    )


class FindingRecord(Base):
    """
    Individual code issue found by an agent.
    Persists the output of Security/Quality/Performance agents.
    """
    __tablename__ = "findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False)
    agent_role = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, default="info")  # critical|high|medium|low|info
    file_path = Column(Text, default="")
    line_number = Column(Integer)
    title = Column(Text, nullable=False)
    description = Column(Text, default="")
    suggestion = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    review = relationship("ReviewRecord", back_populates="findings")

    __table_args__ = (
        Index("idx_findings_review", "review_id"),
        Index("idx_findings_severity", "severity"),
        Index("idx_findings_agent", "agent_role"),
    )
