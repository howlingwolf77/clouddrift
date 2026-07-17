"""
/detect and /batch_detect endpoints — with observability.

Detection modes:
    /detect:
        Always uses z-score attribution (Track 1). Single-point mode —
        no rolling window required. Fast and stateless.

    /batch_detect:
        Routes per machine_id group:
        - machine_id present AND >= 30 sequential snapshots:
            Full IF + TCN ensemble (Track 2) — AUC-ROC validated at 0.899.
        - < 30 snapshots OR no machine_id:
            Z-score fallback (same as /detect).

        The detection_mode field in each result item indicates which path ran.

Observability layers:
    1. Pandera schema validation on /detect — catches out-of-range sensor
       values that pass Pydantic but violate the telemetry data contract.
    2. Prometheus metrics — REQUEST_COUNTER, LATENCY_HISTOGRAM, ANOMALY_COUNTER.
    3. OpenTelemetry spans — per-request child spans with CloudDrift attributes.
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
                "hint": (
                    "Check for out-of-range values or sentinel readings "
                    "(-1, 101) that passed Pydantic validation but violate "
                    "the expected telemetry data contract."
                ),
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
        "Score one telemetry reading using z-score attribution (Track 1). "
        "Fast and stateless — no rolling window required.\n\n"
        "**Detection mode:** `single_point_zscore`. "
        "Computes |value − training_mean| / training_std per metric against "
        "the SMD training distribution. Returns the top contributing metrics "
        "ranked by deviation score.\n\n"
        "**For ensemble scoring (IF + TCN, AUC-ROC 0.899):** use `/batch_detect` "
        "with ≥ 30 sequential snapshots from the same `machine_id`."
    ),
)
async def detect(snapshot: TelemetrySnapshot, request: Request) -> AnomalyResponse:
    t_start = time.perf_counter()
    tracer = get_tracer()

    with tracer.start_as_current_span("clouddrift.detect") as span:
        span.set_attribute("endpoint", "/detect")

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

        span.set_attribute("anomaly_score", result["anomaly_score"])
        span.set_attribute("severity_label", result["severity_label"])
        span.set_attribute("latency_ms", round(latency_s * 1000, 2))
        span.set_status(StatusCode.OK)

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
    summary="Batch anomaly detection with automatic ensemble routing",
    description=(
        "Score a list of telemetry snapshots and return results ranked by "
        "anomaly score descending.\n\n"
        "**Routing logic (per machine_id group):**\n"
        "- `machine_id` present **AND** ≥ 30 sequential snapshots from that "
        "machine → **Full IF + TCN ensemble** (AUC-ROC 0.899, `detection_mode: "
        "ensemble_if_tcn`). Feature engineering builds 68 rolling and "
        "cross-metric features; IF and TCN scores are combined at "
        "IF=0.40 / TCN=0.60.\n"
        "- < 30 snapshots **OR** no `machine_id` → **Z-score fallback** "
        "(`detection_mode: single_point_zscore`).\n\n"
        "Mixed batches (multiple machines, different group sizes) are fully "
        "supported — each group is routed independently.\n\n"
        "The `detection_mode` field in each result item indicates which path ran. "
        "`ensemble_scored` and `zscore_scored` in the response show the split.\n\n"
        "**TCN warm-up note:** with exactly 30 snapshots, only the last row "
        "has a full TCN reconstruction error. Earlier rows are IF-dominant. "
        "60+ snapshots give most rows a full TCN score.\n\n"
        "Accepts 1–1000 snapshots per request."
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

        ranked, threshold_val, ensemble_count, zscore_count = score_batch(
            snapshot_dicts,
            api_ref,
            thresholds,
            artifacts=artifacts,
        )

        latency_s = time.perf_counter() - t_start
        n_flagged = sum(
            1 for r in ranked if r.get("anomaly_score", 0.0) >= threshold_val
        )

        for r in ranked:
            if r.get("severity_label", "Normal") != "Normal":
                ANOMALY_COUNTER.labels(severity_label=r["severity_label"]).inc()

        REQUEST_COUNTER.labels(endpoint="/batch_detect", status_code="200").inc()
        LATENCY_HISTOGRAM.labels(endpoint="/batch_detect").observe(latency_s)

        span.set_attribute("n_flagged", n_flagged)
        span.set_attribute("ensemble_count", ensemble_count)
        span.set_attribute("zscore_count", zscore_count)
        span.set_attribute("latency_ms", round(latency_s * 1000, 2))
        span.set_status(StatusCode.OK)

    results = [
        BatchDetectItem(
            rank=r["rank"],
            timestamp=r.get("timestamp", ""),
            machine_id=r.get("machine_id"),
            anomaly_score=r["anomaly_score"],
            severity_label=r["severity_label"],
            top_contributing_features=r.get("top_contributing_features", []),
            feature_deviation_scores=r.get("feature_deviation_scores", {}),
            detection_mode=r.get("detection_mode", "single_point_zscore"),
        )
        for r in ranked
    ]

    return BatchDetectResponse(
        n_snapshots=len(results),
        n_flagged=n_flagged,
        threshold=threshold_val,
        results=results,
        ensemble_scored=ensemble_count,
        zscore_scored=zscore_count,
    )
