"""
Pydantic v2 schemas for the CloudDrift API.
TelemetrySnapshot — input schema for /detect
AnomalyResponse   — output schema for /detect
Implemented: Day 8
"""

from pydantic import BaseModel


class TelemetrySnapshot(BaseModel):
    """Input: one telemetry snapshot for anomaly detection."""

    cpu_util: float
    mem_util: float
    net_io_in: float
    net_io_out: float
    disk_io: float
    timestamp: str


class AnomalyResponse(BaseModel):
    """Output: anomaly detection result with lightweight attribution."""

    anomaly_score: float
    severity_label: str
    top_contributing_features: list[str]
    feature_deviation_scores: dict[str, float]
    inference_latency_ms: float
