"""
Prism Structured Logging Configuration.

Adapted from: Autonomous-Multi-Agent desktop_nova.py (L23-32)
Pattern: structlog with ISO timestamps, colored console output for dev,
         JSON output for production.

Enhancements:
  - bind_correlation_id(): injects GitHub delivery ID into every log line
    in the current async context via structlog.contextvars.
  - OTel integration: automatically injects trace_id and span_id into
    every log line so logs can be correlated with Jaeger/X-Ray traces.

Usage:
    from observability.logging import configure_logging, get_logger, bind_correlation_id
    configure_logging()
    bind_correlation_id("abc-123-delivery-id")
    logger = get_logger(__name__)
    logger.info("review_started", pr_number=42)
    # Output: {"correlation_id": "abc-123-delivery-id", "trace_id": "...", ...}
"""

import os
import sys

import structlog


def _inject_otel_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Structlog processor that injects OTel trace_id and span_id into every log line.

    This allows log lines to be correlated with distributed traces in Jaeger/X-Ray.
    If no active span exists, the fields are omitted (not set to empty strings).
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id != 0:
            # Format as hex strings (standard OTel format)
            event_dict["trace_id"] = f"{ctx.trace_id:032x}"
            event_dict["span_id"] = f"{ctx.span_id:016x}"
    except Exception:
        # OTel not initialized or import failed — skip silently
        pass

    return event_dict


def configure_logging(environment: str | None = None) -> None:
    """
    Configure structlog for the entire application.

    - Development: Colored console output (human-readable)
    - Production:  JSON output (machine-parseable for CloudWatch/Datadog)
    """
    env = environment or os.getenv("ENVIRONMENT", "development")

    # Shared processors — always applied
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_otel_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if env == "production":
        # JSON output for log aggregation (CloudWatch, Datadog, ELK)
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for local development
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation_id (GitHub delivery ID) to all log lines in this context.

    Uses structlog's contextvars integration so the ID automatically flows
    through every function call in the same async context — including agents,
    LLM calls, and GitHub API calls — without passing it as an argument.

    Call this once at the start of:
      - The webhook handler (using X-GitHub-Delivery header)
      - The Celery task (using pr_data["correlation_id"])
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_context() -> None:
    """Clear all bound context vars. Call at the end of a request/task."""
    structlog.contextvars.clear_contextvars()
