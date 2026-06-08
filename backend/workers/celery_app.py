"""
Prism Celery Configuration.

Source: Proximus infrastructure patterns (standard production task queue)

Uses Redis as the broker and result backend.
Configured for production-grade reliability with:
  - Late acknowledgement (task_acks_late) — messages stay in Redis until the
    worker confirms success, so a killed worker doesn't lose the task.
  - Worker-lost rejection (task_reject_on_worker_lost) — if a worker process
    dies mid-task, the message is requeued instead of acked.
  - Dead Letter Queue (prism.dlq) — tasks that exhaust all retries are routed
    here for manual inspection and replay.

References:
  - https://docs.celeryq.dev/en/stable/userguide/configuration.html#task-reject-on-worker-lost
  - https://inventivehq.com/blog/webhook-best-practices-guide
"""

from celery import Celery
from kombu import Exchange, Queue
from celery.signals import worker_process_init
from utils.config import settings

# ── Queue Definitions ─────────────────────────────────────────────────────
default_exchange = Exchange("prism", type="direct")

TASK_QUEUES = (
    Queue("celery", default_exchange, routing_key="celery"),
    Queue("prism.dlq", default_exchange, routing_key="prism.dlq"),
    Queue("prism.index", default_exchange, routing_key="prism.index"),
)


celery_app = Celery(
    "prism", broker=settings.redis_url, backend=settings.redis_url, include=["workers.tasks", "workers.indexing_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # ── Reliability ───────────────────────────────────────────────────
    # Late ack: message stays in broker until task succeeds/fails
    task_acks_late=True,
    # If worker is killed (OOM, SIGKILL), reject the message so broker
    # redelivers it to another worker instead of silently dropping it
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # ── Queues ────────────────────────────────────────────────────────
    task_queues=TASK_QUEUES,
    task_default_queue="celery",
    task_default_exchange="prism",
    task_default_routing_key="celery",
    # ── Limits ────────────────────────────────────────────────────────
    worker_max_tasks_per_child=50,
    task_time_limit=1800,  # 30 minute hard kill
    task_soft_time_limit=1600,  # 26m40s — gives task time to clean up
    # ── Task Routing ──────────────────────────────────────────────────
    task_routes={
        "build_repo_index": {"queue": "prism.index"},
        "refresh_repo_index": {"queue": "prism.index"},
        "cleanup_stale_clones": {"queue": "prism.index"},
        "cleanup_stale_indexes": {"queue": "prism.index"},
    },
    # ── Beat Schedule (periodic tasks) ────────────────────────────────
    beat_schedule={
        "cleanup-stale-clones-hourly": {
            "task": "cleanup_stale_clones",
            "schedule": 3600.0,  # Every hour
        },
        "cleanup-stale-indexes-daily": {
            "task": "cleanup_stale_indexes",
            "schedule": 86400.0,  # Every 24 hours
        },
        "update-dlq-depth-every-minute": {
            "task": "update_dlq_depth",
            "schedule": 60.0,  # Every minute — keeps DLQ gauge accurate for alerts
        },
    },
)


# ── Start OTel tracing when worker child boots ───────────────────────
# OTel: initializes TracerProvider in EACH child process via worker_process_init,
# because trace context is per-process.


@worker_process_init.connect
def _init_worker_child(**kwargs):
    """Initialize OTel tracing in each child process."""
    try:
        from observability.tracing import init_tracing
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        init_tracing("prism-worker")
        CeleryInstrumentor().instrument()
    except Exception as e:
        print(f"Warning: Worker OTel tracing failed to initialize: {e}")
