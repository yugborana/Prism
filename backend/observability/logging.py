"""
Prism Structured Logging Configuration.

Adapted from: Autonomous-Multi-Agent desktop_nova.py (L23-32)
Pattern: structlog with ISO timestamps, colored console output for dev,
         JSON output for production.

Usage:
    from observability.logging import configure_logging, get_logger
    configure_logging()
    logger = get_logger(__name__)
    logger.info("review_started", pr_number=42, repo="owner/repo")
"""

import os
import sys

import structlog


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
