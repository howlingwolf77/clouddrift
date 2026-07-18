"""
Prometheus metrics definitions for the CloudDrift API.

Four metrics following the RED observability pattern
(Rate, Errors/Violations, Duration):

    clouddrift_requests_total
        Counter — total HTTP requests by endpoint and HTTP status code.
        Rate metric: `rate(clouddrift_requests_total[5m])` shows
        request throughput per endpoint.

    clouddrift_anomalies_total
        Counter — anomalies detected, broken out by severity label.
        Useful for alerting: if Critical count spikes, page on-call.

    clouddrift_prediction_latency_seconds
        Histogram — end-to-end request latency in seconds for /detect
        and /batch_detect. Bucket boundaries chosen around the ≤200ms
        p95 SLA target for /detect from the project spec.

    clouddrift_schema_violations_total
        Counter — Pandera validation failures in /detect. A sustained
        non-zero rate signals a client sending malformed telemetry.

All metrics are registered in the default prometheus_client REGISTRY
on first import. Subsequent imports reuse the same objects (Python
module caching guarantees single instantiation).

Day 12 Docker Compose adds a Prometheus scraper container that polls
/metrics on a configurable interval.
"""

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

REQUEST_COUNTER = Counter(
    "clouddrift_requests_total",
    "Total HTTP requests processed by the CloudDrift API",
    ["endpoint", "status_code"],
)

ANOMALY_COUNTER = Counter(
    "clouddrift_anomalies_total",
    "Anomalies detected by the CloudDrift ensemble, broken out by severity",
    ["severity_label"],
)

SCHEMA_VIOLATION_COUNTER = Counter(
    "clouddrift_schema_violations_total",
    "Pandera schema validation failures in /detect requests",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

# Buckets chosen around the ≤200ms p95 API latency SLA:
# first four buckets (1ms–50ms) cover the normal fast path;
# 100ms and 200ms are the warning and SLA breach markers;
# 500ms and 1s catch outliers without blowing up the histogram.
LATENCY_HISTOGRAM = Histogram(
    "clouddrift_prediction_latency_seconds",
    "End-to-end prediction latency in seconds for detection endpoints",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.010, 0.050, 0.100, 0.200, 0.500, 1.0, 5.0],
)
