"""
/detect and /batch_detect endpoints — with observability.

Layers added in Day 9:
    1. Pandera schema validation on /detect: a thin second validation layer
       that catches data quality issues Pydantic cannot, such as out-of-range
       sensor values that are valid Python floats but physically impossible
       telemetry readings. Violations increment clouddrift_schema_violations_total.

    2. Prometheus metrics: every request records to REQUEST_COUNTER (by
       endpoint and status_code), LATENCY_HISTOGRAM (by endpoint), and
       ANOMALY_COUNTER (by severity_label for detected anomalies).

    3. OpenTelemetry spans: each request gets a manual child span with
       CloudDrift-specific attributes (severity_label, anomaly_score,
       n_snapshots). The parent HTTP span is added automatically by
       FastAPIInstrumentor middleware.
"""

import time
from typing import Any

import pandas as pd
import pandera.pandas as pa
from fastapi import APIRouter, HTTPException, Request
from opentelemetry.trace import StatusCode

from api.schemas.telemetry import (
    AnomalyResponse,
    BatchDetectItem,
    BatchDetectRequest,
    BatchDetectResponse,
    TelemetrySnapshot,
)
from api.services.detection import score_batch, score_snapshot
from api.services.metrics import (
    ANOMALY_COUNTER,
    LATENCY_HISTOGRAM,
    REQUEST_COUNTER,
    SCHEMA_VIOLATION_COUNTER,
)
from api.services.observability import get_tracer

router = APIRouter(tags=["detection"])


# ---------------------------------------------------------------------------
# Pandera schema — second validation layer for telemetry data quality
# ---------------------------------------------------------------------------

_TELEMETRY_SCHEMA = pa.DataFrameSchema(
    {
        "cpu_util": pa.Column(float, pa.Check.between(0.0, 100.0)),
        "mem_util": pa.Column(float, pa.Check.between(0.0, 100.0)),
        "net_io_in": pa.Column(float, pa.Check.between(0.0, 100.0)),
        "net_io_out": pa.Column(float, pa.Check.between(0.0, 100.0)),
        "disk_io": pa.Column(float, pa.Check.between(0.0, 100.0), nullable=True),
    },
    coerce=True,
)


def _validate_snapshot(snapshot: TelemetrySnapshot) -> None:
    """
    Apply Pandera schema validation to a single telemetry snapshot.

    Converts the snapshot to a 1-row DataFrame and validates column
    types and value ranges. Raises HTTPException(422) on failure and
    increments the schema violations counter.

    Args:
        snapshot: The incoming TelemetrySnapshot Pydantic model.

    Raises:
        HTTPException(422): if the snapshot fails Pandera validation.
    """
    row: dict[str, Any] = {
        "cpu_util": snapshot.cpu_util,
        "mem_util": snapshot.mem_util,
        "net_io_in": snapshot.net_io_in,
        "net_io_out": snapshot.net_io_out,
        "disk_io": snapshot.disk_io,
    }
    df = pd.DataFrame([row])

    try:
        _TELEMETRY_SCHEMA.validate(df)
    except pa.errors.SchemaError as exc:
        SCHEMA_VIOLATION_COUNTER.labels(endpoint="/detect").inc()
        raise HTTPException(
            status_code=422,
            detail={
                "error": "schema_validation_failed",
                "message": str(exc),
                "hint": "Check for out-of-range values or sentinel readings "
                "(-1, 101) that passed Pydantic validation but violate "
                "the expected telemetry data contract.",
            },
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_artifacts(request: Request) -> dict:
    """Extract artifacts from app.state or raise 503."""
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None or not artifacts.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail="Service not ready — call /ready to check artifact status",
        )
    return artifacts


# ---------------------------------------------------------------------------
# /detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=AnomalyResponse,
    summary="Single-snapshot anomaly detection",
    description=(
        "Score one telemetry reading. "
        "Uses z-score attribution against the Alibaba training distribution "
        "(single-point mode — no rolling window required). "
        "For ensemble scoring (IF+TCN), use /batch_detect with >= 30 "
        "sequential snapshots from the same machine."
    ),
)
async def detect(snapshot: TelemetrySnapshot, request: Request) -> AnomalyResponse:
    t_start = time.perf_counter()
    tracer = get_tracer()

    with tracer.start_as_current_span("clouddrift.detect") as span:
        span.set_attribute("endpoint", "/detect")

        # Pandera validation — second layer after Pydantic
        _validate_snapshot(snapshot)

        artifacts = _get_artifacts(request)
        api_ref = artifacts.get("api_reference_stats", {})
        thresholds = artifacts.get("thresholds", {})

        result = score_snapshot(
            snapshot.model_dump(),
            api_ref,
            thresholds,
            n_top=5,
        )

        latency_s = time.perf_counter() - t_start

        # Record OTel span attributes
        span.set_attribute("anomaly_score", result["anomaly_score"])
        span.set_attribute("severity_label", result["severity_label"])
        span.set_attribute("latency_ms", round(latency_s * 1000, 2))
        span.set_status(StatusCode.OK)

        # Record Prometheus metrics
        REQUEST_COUNTER.labels(endpoint="/detect", status_code="200").inc()
        LATENCY_HISTOGRAM.labels(endpoint="/detect").observe(latency_s)
        if result["severity_label"] != "Normal":
            ANOMALY_COUNTER.labels(severity_label=result["severity_label"]).inc()

    return AnomalyResponse(
        anomaly_score=result["anomaly_score"],
        severity_label=result["severity_label"],
        top_contributing_features=result["top_contributing_features"],
        feature_deviation_scores=result["feature_deviation_scores"],
        inference_latency_ms=round(latency_s * 1000, 2),
        detection_mode="single_point_zscore",
    )


# ---------------------------------------------------------------------------
# /batch_detect
# ---------------------------------------------------------------------------


@router.post(
    "/batch_detect",
    response_model=BatchDetectResponse,
    summary="Batch anomaly detection",
    description=(
        "Score a list of telemetry snapshots and return results "
        "ranked by anomaly score descending. "
        "Accepts 1–1000 snapshots. Results above the calibrated threshold "
        "are flagged."
    ),
)
async def batch_detect(
    payload: BatchDetectRequest, request: Request
) -> BatchDetectResponse:
    t_start = time.perf_counter()
    tracer = get_tracer()

    with tracer.start_as_current_span("clouddrift.batch_detect") as span:
        span.set_attribute("endpoint", "/batch_detect")
        span.set_attribute("n_snapshots", len(payload.snapshots))

        artifacts = _get_artifacts(request)
        api_ref = artifacts.get("api_reference_stats", {})
        thresholds = artifacts.get("thresholds", {})

        snapshot_dicts = [s.model_dump() for s in payload.snapshots]
        ranked, threshold_val = score_batch(snapshot_dicts, api_ref, thresholds)

        latency_s = time.perf_counter() - t_start
        n_flagged = sum(1 for r in ranked if r["anomaly_score"] >= threshold_val)

        # Count anomalies by severity for batch
        for r in ranked:
            if r["severity_label"] != "Normal":
                ANOMALY_COUNTER.labels(severity_label=r["severity_label"]).inc()

        REQUEST_COUNTER.labels(endpoint="/batch_detect", status_code="200").inc()
        LATENCY_HISTOGRAM.labels(endpoint="/batch_detect").observe(latency_s)

        span.set_attribute("n_flagged", n_flagged)
        span.set_attribute("latency_ms", round(latency_s * 1000, 2))
        span.set_status(StatusCode.OK)

    results = [
        BatchDetectItem(
            rank=r["rank"],
            timestamp=r["timestamp"],
            machine_id=r.get("machine_id"),
            anomaly_score=r["anomaly_score"],
            severity_label=r["severity_label"],
            top_contributing_features=r["top_contributing_features"],
            feature_deviation_scores=r["feature_deviation_scores"],
        )
        for r in ranked
    ]

    return BatchDetectResponse(
        n_snapshots=len(results),
        n_flagged=n_flagged,
        threshold=threshold_val,
        results=results,
    )
