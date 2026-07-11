"""
CloudDrift FastAPI application entry point.

Endpoints:
    GET  /health        — liveness check
    GET  /ready         — readiness check (artifacts loaded?)
    POST /detect        — single telemetry snapshot → anomaly result
    POST /batch_detect  — list of snapshots → ranked results
    GET  /metrics       — Prometheus metrics scrape endpoint (stub;
                          full implementation on Day 9)

Startup:
    All model artifacts are loaded in the lifespan context manager.
    The /ready endpoint returns 503 until all artifacts are present.

OpenTelemetry and Prometheus instrumentation are added on Day 9.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.routers.detection import router as detection_router
from api.routers.health import router as health_router
from api.services.observability import setup_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup: load all model artifacts into app.state.artifacts.
    Shutdown: release resources (artifacts are GC-collected automatically).

    Using lifespan rather than the deprecated @app.on_event("startup")
    per FastAPI best practices (FastAPI >= 0.93).
    """
    logger.info("CloudDrift API starting up — loading artifacts...")
    from api.services.detection import load_all_artifacts

    app.state.artifacts = load_all_artifacts()

    # Initialise OpenTelemetry tracing
    setup_tracing(app)
    logger.info("OpenTelemetry tracing initialised (ConsoleSpanExporter)")

    if app.state.artifacts["loaded"]:
        logger.info("All artifacts loaded — service is ready")
    else:
        failed = [k for k, v in app.state.artifacts["artifact_status"].items() if not v]
        logger.warning(
            "Startup complete with missing artifacts: %s. "
            "/ready will return 503 until resolved.",
            failed,
        )

    yield  # API is running

    logger.info("CloudDrift API shutting down")
    app.state.artifacts = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CloudDrift — Cloud Infrastructure Anomaly Detector",
    description=(
        "ML-powered detection of cloud infrastructure drift before it "
        "becomes an outage. Combines Isolation Forest and TCN Autoencoder "
        "ensemble (AUC-ROC=0.899, IF=0.40/TCN=0.60) with lightweight "
        "z-score attribution for real-time anomaly explanations.\n\n"
        "**Dataset:** Server Machine Dataset (SMD) — 7 machines, "
        "68-dimensional engineered features from CPU, memory, network, "
        "and disk I/O telemetry.\n\n"
        "**Explainability:** Two-track design — z-score attribution in "
        "every /detect response (Track 1); SHAP TreeExplainer in "
        "notebooks/06_shap_analysis.ipynb (Track 2)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register routers
app.include_router(health_router)
app.include_router(detection_router)


# ---------------------------------------------------------------------------
# /metrics — Prometheus scrape endpoint
# Day 9 will replace this stub with full prometheus-client instrumentation.
# ---------------------------------------------------------------------------


@app.get(
    "/metrics",
    include_in_schema=True,
    response_class=PlainTextResponse,
)
async def metrics() -> PlainTextResponse:
    """
    Prometheus metrics scrape endpoint.

    Returns all registered prometheus_client metrics in the standard
    Prometheus text exposition format. Prometheus scrapes this endpoint
    on a configurable interval (default 15s in the Day 12 Compose config).

    Metrics exposed:
        clouddrift_requests_total          Counter by endpoint + status_code
        clouddrift_anomalies_total         Counter by severity_label
        clouddrift_prediction_latency_seconds Histogram by endpoint
        clouddrift_schema_violations_total Counter by endpoint
    """
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
