# CloudDrift API Documentation

**Version:** 1.0.0
**Base URL (local):** http://localhost:8000
**Base URL (Docker):** http://localhost:8000
**Swagger UI:** http://localhost:8000/docs
**OpenAPI spec:** http://localhost:8000/openapi.json

---

## Authentication

No authentication required. CloudDrift is designed for internal
infrastructure monitoring. Add an API gateway or OAuth2 layer for
production multi-tenant deployments.

---

## Endpoints

### GET /health

**Purpose:** Liveness check. Container orchestration (Docker Compose,
Kubernetes) polls this endpoint to confirm the process is running.

**Always returns HTTP 200** regardless of artifact loading status.
If the process is running, you get 200. If the process is dead, you
get a connection error.

```bash
curl http://localhost:8000/health
```

**Response (200):**
```json
{"status": "ok"}
```

---

### GET /ready

**Purpose:** Readiness check. Returns 200 only when all model artifacts
have been loaded into memory. Returns 503 during the startup artifact-
loading window or if any artifact failed to load.

Use this endpoint to gate traffic: do not route requests to `/detect`
until `/ready` returns 200.

```bash
curl http://localhost:8000/ready
```

**Response (200 — all artifacts loaded):**
```json
{
  "status": "ready",
  "artifacts_loaded": {
    "isolation_forest": true,
    "feature_pipeline": true,
    "thresholds": true,
    "tcn_autoencoder": true,
    "ensemble_meta": true,
    "feature_meta": true,
    "reference_stats": true,
    "api_reference_stats": true
  },
  "all_ready": true
}
```

**Response (503 — artifacts not loaded):**
```json
{
  "detail": {
    "status": "degraded",
    "artifacts_loaded": {"isolation_forest": false, ...},
    "all_ready": false
  }
}
```

---

### POST /detect

**Purpose:** Score a single telemetry snapshot. Uses z-score
attribution against the SMD training distribution (single-point
mode). For full IF+TCN ensemble scoring, use `/batch_detect` with
≥30 sequential snapshots.

**Request body:**

| Field | Type | Required | Range | Description |
|-------|------|----------|-------|-------------|
| cpu_util | float | ✓ | [0, 100] | CPU utilization % |
| mem_util | float | ✓ | [0, 100] | Memory utilization % |
| net_io_in | float | ✓ | [0, 100] | Inbound network traffic (normalized to %) |
| net_io_out | float | ✓ | [0, 100] | Outbound network traffic (normalized to %) |
| disk_io | float | ✗ | [0, 100] | Disk I/O utilization % (null accepted) |
| timestamp | str | ✓ | ISO-8601 | Reading timestamp |
| machine_id | str | ✗ | — | Optional machine identifier |

**Note on input scale:** The API accepts values in [0, 100] (percentage
scale) for operator ergonomics. SMD training data is pre-normalized to
[0, 1]. The z-score reference statistics in `artifacts/api_reference_stats.json`
are scaled to [0, 100] to match — a reading of `cpu_util=45.0` is
compared against a reference mean of ~30.0 (30% CPU from SMD training).

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_util": 85.3,
    "mem_util": 72.1,
    "net_io_in": 43.04,
    "net_io_out": 33.08,
    "disk_io": 5.0,
    "timestamp": "2026-07-04T14:30:00Z",
    "machine_id": "m_1932"
  }'
```

**Response (200):**

| Field | Type | Description |
|-------|------|-------------|
| anomaly_score | float [0,1] | Composite anomaly score |
| severity_label | str | "Critical" (≥0.8), "Warning" (≥0.5), "Normal" (<0.5) |
| top_contributing_features | list[str] | Metrics ranked by z-score deviation |
| feature_deviation_scores | dict[str,float] | Z-score deviation per metric |
| inference_latency_ms | float | End-to-end latency in ms |
| detection_mode | str | Always "single_point_zscore" for /detect |

**Error responses:**
- `422 Unprocessable Entity` — Pydantic or Pandera validation failed
  (e.g., cpu_util > 100, missing required field)
- `503 Service Unavailable` — Artifacts not loaded (check `/ready`)

---

### POST /batch_detect

**Purpose:** Score a list of telemetry snapshots and return results
ranked by anomaly score descending.

```bash
curl -X POST http://localhost:8000/batch_detect \
  -H "Content-Type: application/json" \
  -d '{
    "snapshots": [
      {
        "cpu_util": 41.0, "mem_util": 60.0,
        "net_io_in": 43.0, "net_io_out": 33.0,
        "timestamp": "2026-07-04T14:30:00Z"
      },
      {
        "cpu_util": 99.0, "mem_util": 98.0,
        "net_io_in": 95.0, "net_io_out": 90.0,
        "timestamp": "2026-07-04T14:31:00Z"
      }
    ]
  }'
```

**Response (200):**
```json
{
  "n_snapshots": 2,
  "n_flagged": 1,
  "threshold": 0.4829,
  "results": [
    {
      "rank": 1,
      "timestamp": "2026-07-04T14:31:00Z",
      "machine_id": null,
      "anomaly_score": 0.9312,
      "severity_label": "Critical",
      "top_contributing_features": ["cpu_util", "net_io_in", "mem_util"],
      "feature_deviation_scores": {"cpu_util": 2.95, "net_io_in": 3.47, "mem_util": 1.90}
    },
    {
      "rank": 2,
      "timestamp": "2026-07-04T14:30:00Z",
      "anomaly_score": 0.0312,
      "severity_label": "Normal",
      "top_contributing_features": ["cpu_util"],
      "feature_deviation_scores": {"cpu_util": 0.05}
    }
  ]
}
```

Limits: 1–1000 snapshots per request.

---

### GET /metrics

**Purpose:** Prometheus scrape endpoint. Returns all registered
metrics in the standard Prometheus text exposition format.

```bash
curl http://localhost:8000/metrics
```

**Metrics exposed:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `clouddrift_requests_total` | Counter | endpoint, status_code | Total HTTP requests |
| `clouddrift_anomalies_total` | Counter | severity_label | Anomalies by severity |
| `clouddrift_prediction_latency_seconds` | Histogram | endpoint | Request latency |
| `clouddrift_schema_violations_total` | Counter | endpoint | Pandera validation failures |

Prometheus should be configured to scrape this endpoint every 15
seconds. See `monitoring/prometheus.yml`.

---

## Input Validation

CloudDrift uses a two-layer validation strategy:

**Layer 1 — Pydantic v2:** Type checking and field-level range
validation (e.g., 0 ≤ cpu_util ≤ 100). Violations return HTTP 422
before any model inference runs.

**Layer 2 — Pandera:** DataFrame-level schema validation after Pydantic.
Catches data quality issues not expressible as field-level constraints.
Violations increment `clouddrift_schema_violations_total` counter and
return HTTP 422 with a structured error body.

---

## OpenTelemetry Tracing

Every `/detect` and `/batch_detect` request generates an OpenTelemetry
span with attributes:

| Attribute | Value |
|-----------|-------|
| `endpoint` | "/detect" or "/batch_detect" |
| `anomaly_score` | The computed ensemble score |
| `severity_label` | "Critical", "Warning", or "Normal" |
| `latency_ms` | End-to-end latency |
| `n_snapshots` | (batch_detect only) Number of snapshots |

Spans are emitted to `ConsoleSpanExporter` by default (visible in
server logs). Swap to `OTLPSpanExporter` for Jaeger/Zipkin in production.
