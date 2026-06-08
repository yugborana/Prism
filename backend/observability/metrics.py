"""
Prism Prometheus Metrics Registry.

Source: Proximus observability/metrics.py (adapted for code review domain)

Defines all Prometheus counters, histograms, and gauges for the review pipeline.
Gracefully degrades to no-ops if prometheus_client is not installed.

Usage:
    from observability.metrics import review_total, track_review, track_llm_call
    review_total.labels(repo="owner/repo", status="completed").inc()
"""

from contextlib import contextmanager
import time
from observability.logging import get_logger

PROMETHEUS_AVAILABLE = False
_MULTIPROCESS_MODE = False

logger = get_logger(__name__)

# Stub classes since prometheus_client is removed
Counter = None
Gauge = None
Histogram = None
Info = None


# ── No-Op Fallback ────────────────────────────────────────────────────────
class _NoOpMetric:
    """Stub metric when prometheus_client is absent."""
    def labels(self, **kwargs):
        return self
    def inc(self, amount=1):
        pass
    def dec(self, amount=1):
        pass
    def set(self, value):
        pass
    def observe(self, value):
        pass
    def time(self):
        return _NoOpCtx()
    def info(self, val):
        pass


class _NoOpCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _metric(constructor, *args, **kwargs):
    if PROMETHEUS_AVAILABLE:
        return constructor(*args, **kwargs)
    return _NoOpMetric()


# ══════════════════════════════════════════════════════════════════════════
# REVIEW PIPELINE METRICS
# ══════════════════════════════════════════════════════════════════════════

review_total = _metric(
    Counter,
    "prism_review_total",
    "Total PR reviews processed",
    ["repo", "status"],  # status: completed | failed
)

review_duration_seconds = _metric(
    Histogram,
    "prism_review_duration_seconds",
    "Full review pipeline duration (webhook to GitHub comment)",
    ["repo"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

reviews_in_flight = _metric(
    Gauge,
    "prism_reviews_in_flight",
    "Number of reviews currently being processed",
    multiprocess_mode='livesum',
)

findings_total = _metric(
    Counter,
    "prism_findings_total",
    "Total findings reported by agents",
    ["agent", "severity"],  # severity: critical | high | medium | low | info
)

# End-to-end review latency: from Celery task start to completion.
# This is distinct from review_duration_seconds which only measures the
# orchestrator's run_review(). The e2e metric includes Celery queue wait,
# diff fetching from GitHub, all agent runs, and the final GitHub post.
# Buckets tuned for tail latency detection (p95, p99).
review_e2e_seconds = _metric(
    Histogram,
    "prism_review_e2e_seconds",
    "End-to-end review latency from Celery task start to completion",
    ["repo"],
    buckets=[5, 10, 30, 60, 120, 300, 600, 900],
)


# ══════════════════════════════════════════════════════════════════════════
# AGENT TASK METRICS
# ══════════════════════════════════════════════════════════════════════════

agent_task_total = _metric(
    Counter,
    "prism_agent_task_total",
    "Total agent tasks executed",
    ["agent", "status"],  # status: completed | failed
)

agent_task_duration_seconds = _metric(
    Histogram,
    "prism_agent_task_duration_seconds",
    "Individual agent task duration",
    ["agent"],
    buckets=[1, 5, 10, 30, 60, 120],
)

agent_tasks_in_flight = _metric(
    Gauge,
    "prism_agent_tasks_in_flight",
    "Number of agent tasks currently running",
    ["agent"],
    multiprocess_mode='livesum',
)


# ══════════════════════════════════════════════════════════════════════════
# LLM METRICS
# ══════════════════════════════════════════════════════════════════════════

llm_calls_total = _metric(
    Counter,
    "prism_llm_calls_total",
    "Total LLM API calls",
    ["agent", "model", "status"],  # status: success | error | timeout
)

llm_latency_seconds = _metric(
    Histogram,
    "prism_llm_latency_seconds",
    "LLM API call latency",
    ["agent", "model"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
)

llm_tokens_total = _metric(
    Counter,
    "prism_llm_tokens_total",
    "Total LLM tokens consumed",
    ["agent", "model", "direction"],  # direction: input | output
)


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH METRICS
# ══════════════════════════════════════════════════════════════════════════

webhook_requests_total = _metric(
    Counter,
    "prism_webhook_requests_total",
    "Total GitHub webhook requests received",
    ["event_type", "status"],  # status: accepted | rejected | error
)

system_errors_total = _metric(
    Counter,
    "prism_system_errors_total",
    "Total system errors",
    ["component", "error_type"],
)

db_query_duration_seconds = _metric(
    Histogram,
    "prism_db_query_duration_seconds",
    "Database query duration",
    ["operation"],  # operation: insert_review | insert_decision | select_reviews
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)


# ══════════════════════════════════════════════════════════════════════════
# DEAD LETTER QUEUE (DLQ) METRICS
# ══════════════════════════════════════════════════════════════════════════

dlq_tasks_total = _metric(
    Counter,
    "prism_dlq_tasks_total",
    "Total tasks routed to the Dead Letter Queue",
)

dlq_depth = _metric(
    Gauge,
    "prism_dlq_depth",
    "Current number of messages in the Dead Letter Queue",
    multiprocess_mode='max',
)

celery_task_retries_total = _metric(
    Counter,
    "prism_celery_task_retries_total",
    "Total Celery task retries",
    ["task", "retry_number"],
)


# ══════════════════════════════════════════════════════════════════════════
# REPO INDEXING METRICS
# ══════════════════════════════════════════════════════════════════════════

indexing_tasks_total = _metric(
    Counter,
    "prism_indexing_tasks_total",
    "Total repo indexing tasks executed",
    ["task_type", "status"],  # task_type: full | refresh | cleanup, status: completed | failed | skipped
)

index_build_duration_seconds = _metric(
    Histogram,
    "prism_index_build_duration_seconds",
    "Time taken to build or refresh a repo index",
    ["task_type"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)

index_chunks_embedded = _metric(
    Counter,
    "prism_index_chunks_embedded",
    "Number of chunks embedded during indexing",
    ["cache_status"],  # hit | miss
)


# ══════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS (adapted from Proximus)
# ══════════════════════════════════════════════════════════════════════════


@contextmanager
def track_review(repo: str):
    """Track full review pipeline duration."""
    reviews_in_flight.inc()
    start = time.monotonic()
    status = "completed"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.monotonic() - start
        reviews_in_flight.dec()
        review_total.labels(repo=repo, status=status).inc()
        review_duration_seconds.labels(repo=repo).observe(duration)
        logger.info(
            "review_completed",
            repo=repo,
            status=status,
            duration_seconds=round(duration, 2)
        )


@contextmanager
def track_agent_task(agent: str):
    """Track individual agent task duration."""
    agent_tasks_in_flight.labels(agent=agent).inc()
    start = time.monotonic()
    status = "completed"
    try:
        yield
    except Exception:
        status = "failed"
        system_errors_total.labels(component=agent, error_type="task_failure").inc()
        raise
    finally:
        duration = time.monotonic() - start
        agent_tasks_in_flight.labels(agent=agent).dec()
        agent_task_total.labels(agent=agent, status=status).inc()
        agent_task_duration_seconds.labels(agent=agent).observe(duration)
        logger.info(
            "agent_task_completed",
            agent=agent,
            status=status,
            duration_seconds=round(duration, 2)
        )


@contextmanager
def track_llm_call(agent: str, model: str):
    """Track LLM API call latency and outcome."""
    start = time.monotonic()
    status = "success"

    class _Tracker:
        def record_tokens(self, input_tokens: int, output_tokens: int):
            llm_tokens_total.labels(agent=agent, model=model, direction="input").inc(input_tokens)
            llm_tokens_total.labels(agent=agent, model=model, direction="output").inc(output_tokens)

    tracker = _Tracker()
    try:
        yield tracker
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        logger.info(
            "llm_call_completed",
            agent=agent,
            model=model,
            status=status,
            duration_seconds=round(duration, 2)
        )


@contextmanager
def track_indexing_task(task_type: str):
    """Track background indexing task duration and outcome."""
    start = time.monotonic()
    status = "completed"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.monotonic() - start
        indexing_tasks_total.labels(task_type=task_type, status=status).inc()
        index_build_duration_seconds.labels(task_type=task_type).observe(duration)


# Metrics migrated to CloudWatch Logs.

