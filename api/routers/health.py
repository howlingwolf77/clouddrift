"""
/health and /ready endpoints.

/health — liveness check
    Returns HTTP 200 as long as the container process is running.
    Container orchestration (Kubernetes, Docker Compose health checks)
    uses this to confirm the process hasn't crashed. It does NOT check
    whether model artifacts have been loaded.

/ready — readiness check
    Returns HTTP 200 only after all required artifacts have been loaded
    into app.state. Returns HTTP 503 (Service Unavailable) if any
    artifact failed to load. Traffic should not be routed to the service
    until /ready returns 200.
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas.telemetry import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns 200 if the process is running. "
    "Does not check artifact loading status.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Returns 200 when all model artifacts are loaded. "
    "Returns 503 if any artifact failed to load at startup. "
    "Route traffic here only after this endpoint returns 200.",
    responses={
        503: {"description": "One or more artifacts not loaded"},
    },
)
async def ready(request: Request) -> ReadinessResponse:
    artifacts = getattr(request.app.state, "artifacts", None)

    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Artifacts not initialized — startup may still be in progress",
        )

    artifact_status: dict[str, bool] = artifacts.get("artifact_status", {})
    all_ready = artifacts.get("loaded", False)

    response = ReadinessResponse(
        status="ready" if all_ready else "degraded",
        artifacts_loaded=artifact_status,
        all_ready=all_ready,
    )

    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail=response.model_dump(),
        )

    return response
