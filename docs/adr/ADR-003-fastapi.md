# ADR-003: FastAPI for Model Serving

**Date:** June 2026
**Status:** Accepted
**Author:** Rainel (Ryan) Lobora

## Context

CloudDrift needed a Python web framework for the anomaly detection API:
five endpoints (/health, /ready, /detect, /batch_detect, /metrics),
Pydantic v2 request validation, OpenTelemetry instrumentation, and
Prometheus metrics exposition.

## Decision

Selected **FastAPI** over Flask, Django REST Framework, and Litestar.

## Rationale

| Criterion | FastAPI | Flask | DRF |
|-----------|---------|-------|-----|
| Async native | ✓ | Partial | ✗ |
| Pydantic v2 | Native | Plugin | Plugin |
| OpenAPI auto-docs | ✓ | Plugin | Plugin |
| Type hint inference | ✓ | ✗ | ✗ |
| Lifespan context | ✓ (native) | ✗ | ✗ |
| Community/ecosystem | Large | Very large | Large |

FastAPI's `lifespan` context manager (used in `api/main.py`) loads all
model artifacts once at startup and stores them in `app.state`, making
them available to all request handlers without global variables or
repeated file I/O.

## Readiness Gate Pattern

The `/ready` endpoint returns 503 until all artifacts have loaded
successfully. Container orchestration (Docker Compose `depends_on:
condition: service_healthy`) prevents traffic from reaching the
dashboard before the API is ready to serve.

## Single-Point Detection Mode

The `/detect` endpoint uses z-score attribution (Track 1) rather than
the full IF+TCN ensemble. The IF was trained on 13 NAB rolling features;
the TCN requires 30-timestep sequences. Neither can be applied directly
to a single raw Alibaba-style telemetry snapshot. z-score attribution
against the training distribution is the correct single-point signal.
Full ensemble is available via `/batch_detect` with sufficient context.

## Consequences

- Swagger UI auto-generated at /docs
- Pydantic v2 validates all input before any model inference
- Pandera adds a second validation layer for data quality checks
- Lifespan pattern eliminates repeated artifact loading overhead
