"""
Prism AI Code Reviewer — FastAPI Application Entry Point.

"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from observability.logging import configure_logging, get_logger
from utils.config import settings
from api.webhook import router as webhook_router
from api.monitoring import router as monitoring_router
from api.dlq import router as dlq_router

# Configure logging FIRST — before any other imports that might log
configure_logging(settings.environment)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    logger.info(
        "prism_starting",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        port=settings.port,
    )

    if settings.environment == "production" and not settings.github_webhook_secret:
        logger.error("CRITICAL: GITHUB_WEBHOOK_SECRET not set in production! All incoming webhooks will be rejected.")

    # ── Initialize OpenTelemetry Tracing ──────────────────────────────
    try:
        from observability.tracing import init_tracing
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        init_tracing("prism-api")
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        logger.info("otel_auto_instrumentation_enabled")
    except Exception as e:
        logger.warning("otel_init_failed", error=str(e), msg="Continuing without tracing")

    # ── Initialize PostgreSQL (Audit Trail) ───────────────────────────
    try:
        from db.postgres import init_db

        await init_db()
        logger.info("postgres_initialized")
    except Exception as e:
        logger.warning("postgres_init_failed", error=str(e), msg="Continuing without audit DB")

    # ── Initialize Vector Database ────────────────────────────────────
    try:
        from db.index import initialize_collections

        initialize_collections()
        logger.info("vector_db_initialized")
    except Exception as e:
        logger.warning("vector_db_init_failed", error=str(e), msg="Continuing without vector DB")

    # ── Note: Prometheus metrics have been replaced with CloudWatch Metrics.
    # No standalone metrics server or endpoint is needed.

    # ── Initialize Connection Pools ───────────────────────────────────
    from utils.connections import (
        init_redis_pool,
        close_redis_pool,
        init_httpx_client,
        close_httpx_client,
    )

    try:
        await init_redis_pool()
    except Exception as e:
        logger.warning("redis_pool_init_failed", error=str(e))

    try:
        await init_httpx_client()
    except Exception as e:
        logger.warning("httpx_client_init_failed", error=str(e))

    logger.info("prism_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    await close_redis_pool()
    await close_httpx_client()
    try:
        from db.postgres import close_db

        await close_db()
    except Exception:
        pass
    logger.info("prism_shutting_down")


app = FastAPI(
    title="Prism AI Code Reviewer",
    description="Production-grade multi-agent AI code review system",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ──────────────────────────────────────────────────────────────
app.include_router(webhook_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(dlq_router, prefix="/api/v1")


# ── Health Check ─────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Liveness probe for Kubernetes / Docker health checks."""
    checks = {"postgres": False, "redis": False}

    # 1. Check PostgreSQL
    try:
        from db.postgres import get_db_session
        from sqlalchemy import text

        async with get_db_session() as s:
            await s.execute(text("SELECT 1"))
            checks["postgres"] = True
    except Exception as e:
        logger.warning("health_check_postgres_failed", error=str(e))

    # 2. Check Redis (uses shared pool — no per-request connection)
    try:
        from utils.connections import get_redis

        redis = get_redis()
        if redis is not None:
            await redis.ping()
            checks["redis"] = True
        else:
            logger.warning("health_check_redis_pool_not_initialized")
    except Exception as e:
        logger.warning("health_check_redis_failed", error=str(e))

    status = "healthy" if all(checks.values()) else "degraded"

    return {
        "status": status,
        "service": "prism",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "checks": checks,
    }


# ── Metrics replaced with CloudWatch ────────────────────────────────────


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
    )
