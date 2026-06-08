"""
Prism PostgreSQL Database Layer.

Source: Proximus go-backend/internal/shared/db/postgres.go (pattern adapted to Python)

Provides async SQLAlchemy engine + session factory for permanent storage of:
- Review audit trails
- Decision logs
- Agent findings
- Review metadata

Uses asyncpg driver for high-performance async Postgres access.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from observability.logging import get_logger
from utils.config import settings

logger = get_logger(__name__)

# ── Engine (lazy singleton) ───────────────────────────────────────────────
_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=(settings.environment == "development"),
            pool_size=settings.asyncpg_pool_size,
            max_overflow=settings.asyncpg_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30,
        )
        logger.info(
            "postgres_engine_created",
            url=settings.database_url.split("@")[-1],
            pool_size=settings.asyncpg_pool_size,
            max_overflow=settings.asyncpg_max_overflow,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency: yields an async DB session with auto-commit/rollback."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables on startup."""
    from db.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("postgres_tables_initialized")


async def close_db():
    """Cleanup on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("postgres_engine_closed")
