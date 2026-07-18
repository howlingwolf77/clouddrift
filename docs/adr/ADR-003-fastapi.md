# ADR-003: FastAPI for Model Serving

**Date:** June 2026
**Status:** Accepted (amended July 2026 — detection serving architecture updated)
**Author:** Rainel (Ryan) Lobo

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

## Two-Track Detection Serving

Two detection paths are served by the API:

**Track 1 — Z-score attribution (\`/detect\` and \`/batch_detect\` fallback):**
Single-point stateless scoring. Computes |value − training_mean| / training_std
per metric against the SMD training distribution. No feature engineering
or model inference required. Latency < 10ms.

**Track 2 — IF + TCN Ensemble (`/batch_detect` with ≥ 30 snapshots):**
When a `machine_id` group provides ≥ 30 sequential snapshots,
`_score_group_ensemble()` in `api/services/detection.py` builds the
68-feature matrix using `build_alibaba_features()`, applies the fitted
normalization pipeline, computes IF anomaly scores (IF trained on 68 SMD
features) and TCN reconstruction errors (seq_length=30), and combines at
IF=0.40 / TCN=0.60. AUC-ROC validated at 0.899 on the SMD test set.
Latency 3–8 seconds. Ensemble failures fall back to z-score silently.

The `detection_mode` field in every result confirms which path ran.

## Consequences

- Swagger UI auto-generated at /docs
- Pydantic v2 validates all input before any model inference
- Pandera adds a second validation layer for data quality checks
- Lifespan pattern eliminates repeated artifact loading overhead
