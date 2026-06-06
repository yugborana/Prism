"""
Prism Celery Configuration.

Source: Proximus infrastructure patterns (standard production task queue)

Uses Redis as the broker and result backend.
Configured for production-grade reliability with prefetch limits and retries.
"""

from celery import Celery
from utils.config import settings

celery_app = Celery(
    "prism",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    
    # Limits
    worker_max_tasks_per_child=100,
    task_time_limit=1800,  # 30 minute timeout for full review (Ollama/CPU friendly)
    task_soft_time_limit=1600,
)


# ── Start Prometheus metrics server when worker boots ─────────────────
# This exposes metrics on port 9091 so Prometheus can scrape the worker.
# Without this, all review/agent/LLM metrics emitted by tasks would be lost.
from celery.signals import worker_process_init

@worker_process_init.connect
def _start_worker_metrics(**kwargs):
    try:
        from observability.metrics import start_metrics_server
        start_metrics_server(port=settings.prometheus_port_worker)
    except Exception as e:
        print(f"Warning: Worker metrics server failed to start: {e}")
