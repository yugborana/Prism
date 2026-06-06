"""
Prism AI Code Reviewer — FastAPI Application Entry Point.

"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response, Depends
from fastapi.middleware.cors import CORSMiddleware

from observability.logging import configure_logging, get_logger
from utils.config import settings
from api.webhook import router as webhook_router
from api.monitoring import router as monitoring_router
from api.auth import require_api_key

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

    # ── Initialize LLM Client ────────────────────────────────────────
    try:
        from utils.llm_factory import LLMClient
        app.state.llm_client = LLMClient()
        logger.info("llm_client_ready", provider=settings.llm_provider, model=settings.get_model_for_provider())
    except Exception as e:
        logger.warning("llm_client_init_failed", error=str(e))

    # ── Start Prometheus Metrics Server ────────────────────────────────
    try:
        from observability.metrics import start_metrics_server
        start_metrics_server(port=settings.prometheus_port_api)
    except Exception as e:
        logger.warning("metrics_server_failed", error=str(e))

    logger.info("prism_ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────
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
        
    # 2. Check Redis
    try:
        import redis.asyncio as redis_async
        r = redis_async.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        checks["redis"] = True
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


# ── Prometheus Metrics Endpoint ──────────────────────────────────────────
@app.get("/api/metrics")
async def metrics(_=Depends(require_api_key)):
    """Expose Prometheus metrics for scraping."""
    from observability.metrics import get_metrics_text
    return Response(content=get_metrics_text(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.environment == "development",
    )
