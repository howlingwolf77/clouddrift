"""
Pydantic v2 request and response schemas for the CloudDrift API.

TelemetrySnapshot  — input for /detect and /batch_detect
AnomalyResponse    — output from /detect
BatchDetectRequest — input for /batch_detect
BatchDetectItem    — one item in the /batch_detect response list
HealthResponse     — output from /health
ReadinessResponse  — output from /ready
"""

from pydantic import BaseModel, Field, field_validator


class TelemetrySnapshot(BaseModel):
    """
    A single multi-metric telemetry snapshot from one machine
    at one point in time.

    All metric fields represent percentage utilization [0, 100]
    or normalized throughput [0, 100] as per the Alibaba Cluster
    Trace 2018 schema.
    """

    cpu_util: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="CPU utilization percentage [0, 100]",
        examples=[41.0],
    )
    mem_util: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Memory utilization percentage [0, 100]",
        examples=[72.0],
    )
    net_io_in: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Normalized inbound network traffic [0, 100]",
        examples=[43.04],
    )
    net_io_out: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Normalized outbound network traffic [0, 100]",
        examples=[33.08],
    )
    disk_io: float | None = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Disk I/O utilization percentage [0, 100]. "
        "Null is acceptable — sentinel values (-1, 101) from the "
        "Alibaba trace are excluded here at the client layer.",
        examples=[5.0],
    )
    timestamp: str = Field(
        ...,
        description="ISO-8601 timestamp of the reading",
        examples=["2026-06-25T14:30:00Z"],
    )
    machine_id: str | None = Field(
        None,
        description="Optional machine identifier for multi-machine batch requests",
        examples=["m_1932"],
    )

    @field_validator("cpu_util", "mem_util", "net_io_in", "net_io_out")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("Value must be a finite number")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cpu_util": 85.3,
                    "mem_util": 72.1,
                    "net_io_in": 43.04,
                    "net_io_out": 33.08,
                    "disk_io": 5.0,
                    "timestamp": "2026-06-25T14:30:00Z",
                    "machine_id": "m_1932",
                }
            ]
        }
    }


class AnomalyResponse(BaseModel):
    """
    Anomaly detection result for one telemetry snapshot.

    anomaly_score is in [0, 1]:
        < 0.50  → Normal
        0.50-0.80 → Warning
        >= 0.80 → Critical

    top_contributing_features lists the metric names that deviated most
    from their training normal distribution (z-score attribution,
    Track 1 explainability per CloudDrift design).
    """

    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ensemble anomaly score in [0, 1]",
        examples=[0.87],
    )
    severity_label: str = Field(
        ...,
        description="'Critical', 'Warning', or 'Normal'",
        examples=["Critical"],
    )
    top_contributing_features: list[str] = Field(
        ...,
        description="Metric names ranked by z-score deviation (descending)",
        examples=[["cpu_util", "mem_util"]],
    )
    feature_deviation_scores: dict[str, float] = Field(
        ...,
        description="Z-score deviation per metric (absolute value)",
        examples=[{"cpu_util": 3.42, "mem_util": 2.81}],
    )
    inference_latency_ms: float = Field(
        ...,
        description="End-to-end inference time in milliseconds",
        examples=[4.7],
    )
    detection_mode: str = Field(
        default="single_point_zscore",
        description="'single_point_zscore' for /detect (z-score attribution only). "
        "Full IF+TCN ensemble requires /batch_detect with >= 30 snapshots.",
    )


class BatchDetectRequest(BaseModel):
    """Input for /batch_detect — list of telemetry snapshots to score."""

    snapshots: list[TelemetrySnapshot] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of telemetry snapshots to score and rank",
    )


class BatchDetectItem(BaseModel):
    """One item in the /batch_detect response."""

    rank: int = Field(..., description="Rank by anomaly score (1 = highest)")
    timestamp: str
    machine_id: str | None
    anomaly_score: float
    severity_label: str
    top_contributing_features: list[str]
    feature_deviation_scores: dict[str, float]


class BatchDetectResponse(BaseModel):
    """Output from /batch_detect — ranked list of anomaly results."""

    n_snapshots: int = Field(..., description="Total snapshots scored")
    n_flagged: int = Field(..., description="Snapshots above threshold")
    threshold: float = Field(..., description="Anomaly score threshold used")
    results: list[BatchDetectItem]


class HealthResponse(BaseModel):
    """Output from /health — liveness check."""

    status: str = Field(default="ok")


class ReadinessResponse(BaseModel):
    """Output from /ready — readiness check with artifact loading status."""

    status: str
    artifacts_loaded: dict[str, bool]
    all_ready: bool
