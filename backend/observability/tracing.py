"""
Prism OpenTelemetry Tracing Configuration.

Initializes a TracerProvider with a ConsoleSpanExporter.
Traces are shipped to:
  - Local development: stdout
  - AWS production: CloudWatch Logs (via Docker awslogs driver)

Usage:
    from observability.tracing import init_tracing, get_tracer
    init_tracing("prism-api")
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("my_operation"):
        ...

References:
    - https://opentelemetry.io/docs/languages/python/getting-started/
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

from observability.logging import get_logger

logger = get_logger(__name__)

_initialized = False


def init_tracing(service_name: str = "prism") -> None:
    """Initialize OpenTelemetry tracing for this process.

    Call once at startup — either in FastAPI lifespan or Celery worker_process_init.
    Safe to call multiple times (subsequent calls are no-ops).

    Args:
        service_name: Identifies this process in traces ("prism-api" or "prism-worker").
    """
    global _initialized
    if _initialized:
        return

    from utils.config import settings

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "deployment.environment": settings.environment,
        }
    )

    provider = TracerProvider(resource=resource)

    # ── Console Exporter (for CloudWatch Logs via awslogs driver) ─────
    # Outputs traces to stdout, which the Docker awslogs driver sends
    # directly to CloudWatch Logs.
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(
            "otel_exporter_configured",
            type="console",
            service=service_name,
        )
    except Exception as e:
        logger.warning("otel_exporter_failed", error=str(e))

    # ── Set Global Provider ───────────────────────────────────────────
    trace.set_tracer_provider(provider)

    # ── W3C Trace Context propagation ─────────────────────────────────
    # Ensures trace context flows through HTTP headers and Celery messages.
    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )

    _initialized = True
    logger.info("otel_tracing_initialized", service=service_name)


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer for creating spans.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        An OTel Tracer (returns a no-op tracer if init_tracing hasn't been called).
    """
    return trace.get_tracer(name, tracer_provider=trace.get_tracer_provider())


def inject_trace_context() -> dict[str, str]:
    """Extract the current trace context into a dict for serialization.

    Use this to propagate trace context across process boundaries
    (e.g., webhook → Celery task via pr_data["trace_context"]).
    """
    from opentelemetry.propagate import inject

    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_trace_context(carrier: dict[str, str]):
    """Restore trace context from a serialized dict.

    Call this at the start of a Celery task to link the worker span
    as a child of the webhook span.

    Returns:
        An OTel Context object to attach.
    """
    from opentelemetry.propagate import extract

    return extract(carrier)
