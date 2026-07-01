"""
/detect and /batch_detect endpoints.

/detect — single telemetry snapshot → anomaly result
    Scores one telemetry reading using z-score deviation from the
    Alibaba training distribution (Track 1 explainability).
    No rolling window context is required.

/batch_detect — list of snapshots → ranked results
    Scores a list of telemetry readings and returns them ranked by
    anomaly score descending. Designed to be extended with the full
    IF+TCN ensemble when sequential context (>= 30 snapshots per
    machine in time order) is available.
"""

import time

from fastapi import APIRouter, HTTPException, Request

from api.schemas.telemetry import (
    AnomalyResponse,
    BatchDetectItem,
    BatchDetectRequest,
    BatchDetectResponse,
    TelemetrySnapshot,
)
from api.services.detection import score_batch, score_snapshot

router = APIRouter(tags=["detection"])


def _get_artifacts(request: Request) -> dict:
    """Extract artifacts from app.state or raise 503 if not loaded."""
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None or not artifacts.get("loaded"):
        raise HTTPException(
            status_code=503,
            detail="Service not ready — call /ready to check artifact status",
        )
    return artifacts


@router.post(
    "/detect",
    response_model=AnomalyResponse,
    summary="Single-snapshot anomaly detection",
    description="Score one telemetry reading. "
    "Uses z-score attribution against the Alibaba training distribution "
    "(single-point mode — no rolling window required). "
    "For ensemble scoring (IF+TCN), use /batch_detect with >= 30 "
    "sequential snapshots from the same machine.",
)
async def detect(snapshot: TelemetrySnapshot, request: Request) -> AnomalyResponse:
    t_start = time.perf_counter()
    artifacts = _get_artifacts(request)

    api_ref = artifacts.get("api_reference_stats", {})
    thresholds = artifacts.get("thresholds", {})

    result = score_snapshot(
        snapshot.model_dump(),
        api_ref,
        thresholds,
        n_top=5,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000

    return AnomalyResponse(
        anomaly_score=result["anomaly_score"],
        severity_label=result["severity_label"],
        top_contributing_features=result["top_contributing_features"],
        feature_deviation_scores=result["feature_deviation_scores"],
        inference_latency_ms=round(latency_ms, 2),
        detection_mode="single_point_zscore",
    )


@router.post(
    "/batch_detect",
    response_model=BatchDetectResponse,
    summary="Batch anomaly detection",
    description="Score a list of telemetry snapshots and return results "
    "ranked by anomaly score descending. "
    "Accepts 1–1000 snapshots. Results above the calibrated threshold "
    "are flagged. "
    "For full IF+TCN ensemble scoring, provide >= 30 sequential snapshots "
    "per machine in timestamp order.",
)
async def batch_detect(
    payload: BatchDetectRequest, request: Request
) -> BatchDetectResponse:
    artifacts = _get_artifacts(request)

    api_ref = artifacts.get("api_reference_stats", {})
    thresholds = artifacts.get("thresholds", {})

    snapshot_dicts = [s.model_dump() for s in payload.snapshots]
    ranked, threshold_val = score_batch(snapshot_dicts, api_ref, thresholds)

    n_flagged = sum(1 for r in ranked if r["anomaly_score"] >= threshold_val)

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
