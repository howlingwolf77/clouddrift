"""
OpenTelemetry distributed tracing for the CloudDrift API.

Design:
    TracerProvider is configured at startup with a ConsoleSpanExporter
    that writes completed spans to stdout. This makes traces visible in
    the server logs during development and in Docker Compose log streams.

    For production deployment, swap ConsoleSpanExporter for
    OTLPSpanExporter (pointing to Jaeger, Zipkin, or Grafana Tempo)
    without changing any application code — just the exporter config.

Instrumentation:
    FastAPIInstrumentor.instrument_app() adds middleware that automatically
    creates a span for every HTTP request: method, URL, status code, and
    server-side latency are captured without any manual span creation.

    Manual spans are added in /detect and /batch_detect to capture
    CloudDrift-specific attributes: severity_label, anomaly_score, and
    n_snapshots. These go into the span as attributes so they're queryable
    in the trace backend.

Guard against double-instrumentation:
    setup_tracing() checks whether the app has already been instrumented
    before calling FastAPIInstrumentor. This prevents errors in test
    environments where the app module is imported multiple times.
"""

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_TRACER_NAME = "clouddrift.api"
_tracer_provider: TracerProvider | None = None


def setup_tracing(app: "FastAPI") -> TracerProvider:
    """
    Configure OpenTelemetry tracing and instrument the FastAPI application.

    Creates a TracerProvider with a ConsoleSpanExporter (dev/staging) and
    registers it as the global provider. Then instruments the FastAPI app
    so all HTTP requests automatically generate spans.

    Idempotent: calling this function more than once (e.g., in test
    environments where the module is re-imported) safely no-ops after
    the first successful call.

    Args:
        app: The FastAPI application instance.

    Returns:
        The configured TracerProvider.
    """
    global _tracer_provider

    if _tracer_provider is not None:
        logger.debug("Tracing already configured — skipping re-init")
        return _tracer_provider

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    # Instrument FastAPI — adds middleware to capture HTTP-level spans
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        if not FastAPIInstrumentor().is_instrumented_by_opentelemetry:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented with OpenTelemetry")
        else:
            logger.debug("FastAPI already instrumented")
    except Exception:
        logger.exception(
            "FastAPIInstrumentor failed — tracing disabled for this session"
        )

    return provider


def get_tracer() -> trace.Tracer:
    """
    Return the module-level tracer for creating manual spans.

    If setup_tracing() has not been called yet (e.g., in unit tests
    that don't run the full lifespan), returns a no-op tracer so
    application code does not need defensive checks around every
    span.with_span() call.

    Returns:
        trace.Tracer instance.
    """
    return trace.get_tracer(_TRACER_NAME)
