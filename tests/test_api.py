"""
Day 8 tests: FastAPI endpoint and schema validation.

Uses starlette.testclient.TestClient which runs the full ASGI app
synchronously — no asyncio required in tests. The lifespan context
manager (artifact loading) runs automatically on TestClient construction.

Tests are split into:
    - Schema tests: run without artifact files (validate Pydantic logic)
    - Endpoint tests: run with real artifact files (integration)
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.schemas.telemetry import (
    AnomalyResponse,
    TelemetrySnapshot,
)
from api.services.detection import score_batch, score_snapshot

ARTIFACTS_EXIST = all(
    Path(f"artifacts/{fname}").exists()
    for fname in [
        "isolation_forest.joblib",
        "tcn_autoencoder.pt",
        "thresholds.joblib",
        "ensemble_metadata.json",
        "api_reference_stats.json",
    ]
)

# ---------------------------------------------------------------------------
# Pydantic schema tests — no artifact files required
# ---------------------------------------------------------------------------


class TestTelemetrySnapshot:
    """Tests for TelemetrySnapshot Pydantic model."""

    def _valid(self, **kwargs) -> dict:
        base = {
            "cpu_util": 41.0,
            "mem_util": 72.0,
            "net_io_in": 43.0,
            "net_io_out": 33.0,
            "timestamp": "2026-06-25T14:30:00Z",
        }
        base.update(kwargs)
        return base

    def test_valid_snapshot_parses(self):
        s = TelemetrySnapshot(**self._valid())
        assert s.cpu_util == 41.0

    def test_disk_io_is_optional(self):
        s = TelemetrySnapshot(**self._valid())
        assert s.disk_io is None

    def test_cpu_util_above_100_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TelemetrySnapshot(**self._valid(cpu_util=150.0))
        assert "cpu_util" in str(exc_info.value)

    def test_cpu_util_below_0_rejected(self):
        with pytest.raises(ValidationError):
            TelemetrySnapshot(**self._valid(cpu_util=-1.0))

    def test_mem_util_above_100_rejected(self):
        with pytest.raises(ValidationError):
            TelemetrySnapshot(**self._valid(mem_util=101.0))

    def test_missing_timestamp_rejected(self):
        data = self._valid()
        data.pop("timestamp")
        with pytest.raises(ValidationError):
            TelemetrySnapshot(**data)

    def test_missing_required_metric_rejected(self):
        data = self._valid()
        data.pop("cpu_util")
        with pytest.raises(ValidationError):
            TelemetrySnapshot(**data)

    def test_machine_id_is_optional(self):
        s = TelemetrySnapshot(**self._valid())
        assert s.machine_id is None

    def test_machine_id_accepted(self):
        s = TelemetrySnapshot(**self._valid(machine_id="m_1932"))
        assert s.machine_id == "m_1932"


class TestAnomalyResponse:
    """Tests for AnomalyResponse Pydantic model."""

    def _valid_response(self, **kwargs) -> dict:
        base = {
            "anomaly_score": 0.85,
            "severity_label": "Critical",
            "top_contributing_features": ["cpu_util", "mem_util"],
            "feature_deviation_scores": {"cpu_util": 3.42, "mem_util": 2.81},
            "inference_latency_ms": 4.7,
        }
        base.update(kwargs)
        return base

    def test_valid_response_parses(self):
        r = AnomalyResponse(**self._valid_response())
        assert r.anomaly_score == 0.85
        assert r.severity_label == "Critical"

    def test_anomaly_score_above_1_rejected(self):
        with pytest.raises(ValidationError):
            AnomalyResponse(**self._valid_response(anomaly_score=1.5))

    def test_anomaly_score_below_0_rejected(self):
        with pytest.raises(ValidationError):
            AnomalyResponse(**self._valid_response(anomaly_score=-0.1))

    def test_detection_mode_defaults(self):
        r = AnomalyResponse(**self._valid_response())
        assert r.detection_mode == "single_point_zscore"


# ---------------------------------------------------------------------------
# Score function tests — no artifact files required
# ---------------------------------------------------------------------------


class TestScoreSnapshot:
    """Tests for the score_snapshot() detection logic."""

    def _make_ref_stats(self) -> dict:
        return {
            "cpu_util": {"mean": 40.0, "std": 20.0},
            "mem_util": {"mean": 60.0, "std": 20.0},
            "net_io_in": {"mean": 43.0, "std": 15.0},
            "net_io_out": {"mean": 33.0, "std": 12.0},
            "disk_io": {"mean": 10.0, "std": 10.0},
        }

    def test_normal_reading_has_low_score(self):
        snapshot = {
            "cpu_util": 40.0,
            "mem_util": 60.0,
            "net_io_in": 43.0,
            "net_io_out": 33.0,
        }
        result = score_snapshot(snapshot, self._make_ref_stats(), {})
        assert result["anomaly_score"] < 0.3
        assert result["severity_label"] == "Normal"

    def test_extreme_reading_has_high_score(self):
        snapshot = {
            "cpu_util": 99.0,
            "mem_util": 98.0,
            "net_io_in": 95.0,
            "net_io_out": 90.0,
        }
        result = score_snapshot(snapshot, self._make_ref_stats(), {})
        assert result["anomaly_score"] > 0.5

    def test_anomaly_score_in_0_1_range(self):
        for cpu in [0.0, 40.0, 99.0, 100.0]:
            snapshot = {
                "cpu_util": cpu,
                "mem_util": 60.0,
                "net_io_in": 43.0,
                "net_io_out": 33.0,
            }
            result = score_snapshot(snapshot, self._make_ref_stats(), {})
            assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_top_contributing_features_sorted_by_deviation(self):
        # cpu_util is very deviated, mem_util is normal
        snapshot = {
            "cpu_util": 99.0,
            "mem_util": 60.0,
            "net_io_in": 43.0,
            "net_io_out": 33.0,
        }
        result = score_snapshot(snapshot, self._make_ref_stats(), {})
        assert result["top_contributing_features"][0] == "cpu_util"

    def test_empty_snapshot_returns_normal(self):
        result = score_snapshot({}, self._make_ref_stats(), {})
        assert result["anomaly_score"] == 0.0
        assert result["severity_label"] == "Normal"

    def test_severity_labels_match_thresholds(self):
        ref = {"cpu_util": {"mean": 0.0, "std": 1.0}}

        for z, expected_severity in [
            (1.0, "Normal"),  # tanh(1/3) ≈ 0.32
            (6.0, "Critical"),  # tanh(6/3) ≈ 0.96
        ]:
            snapshot = {"cpu_util": z}
            result = score_snapshot(snapshot, ref, {})
            assert result["severity_label"] == expected_severity, (
                f"z={z} expected {expected_severity}, "
                f"got {result['severity_label']} "
                f"(score={result['anomaly_score']})"
            )

    def test_score_batch_sorted_descending(self):
        ref = self._make_ref_stats()
        snapshots = [
            {
                "cpu_util": 40.0,
                "mem_util": 60.0,
                "net_io_in": 43.0,
                "net_io_out": 33.0,
                "timestamp": "t1",
            },
            {
                "cpu_util": 99.0,
                "mem_util": 98.0,
                "net_io_in": 95.0,
                "net_io_out": 90.0,
                "timestamp": "t2",
            },
        ]
        ranked, *_ = score_batch(snapshots, ref, {})
        scores = [r["anomaly_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_score_batch_assigns_ranks(self):
        ref = self._make_ref_stats()
        snapshots = [
            {
                "cpu_util": 40.0,
                "mem_util": 60.0,
                "net_io_in": 43.0,
                "net_io_out": 33.0,
                "timestamp": "t1",
            },
            {
                "cpu_util": 99.0,
                "mem_util": 98.0,
                "net_io_in": 95.0,
                "net_io_out": 90.0,
                "timestamp": "t2",
            },
        ]
        ranked, *_ = score_batch(snapshots, ref, {})
        ranks = [r["rank"] for r in ranked]
        assert ranks == [1, 2]


# ---------------------------------------------------------------------------
# Endpoint tests — require artifact files
# ---------------------------------------------------------------------------


def _make_mock_artifacts() -> dict:
    """Create a minimal mock artifacts dict for endpoint testing."""
    return {
        "loaded": True,
        "artifact_status": {
            k: True
            for k in [
                "isolation_forest",
                "feature_pipeline",
                "thresholds",
                "tcn_autoencoder",
                "ensemble_meta",
                "feature_meta",
                "reference_stats",
                "api_reference_stats",
            ]
        },
        "thresholds": {"isolation_forest": 0.5, "tcn_autoencoder": 0.0001},
        "api_reference_stats": {
            "cpu_util": {"mean": 40.0, "std": 20.0},
            "mem_util": {"mean": 60.0, "std": 20.0},
            "net_io_in": {"mean": 43.0, "std": 15.0},
            "net_io_out": {"mean": 33.0, "std": 12.0},
            "disk_io": {"mean": 10.0, "std": 10.0},
        },
    }


@asynccontextmanager
async def _mock_lifespan(app):
    """
    Test lifespan that injects mock artifacts without loading real model files.
    Replaces app.router.lifespan_context in setup_method so the real
    load_all_artifacts() is never called — avoiding all patch-timing issues
    with TestClient background threads.
    """
    app.state.artifacts = _make_mock_artifacts()
    yield
    app.state.artifacts = None


class TestHealthEndpoints:
    """Tests for /health and /ready using mocked artifacts."""

    def setup_method(self):
        from api.main import app

        # Replace the real lifespan with _mock_lifespan so mock artifacts are
        # injected synchronously during TestClient.__enter__, with no dependency
        # on patch timing relative to TestClient background threads.
        self._original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _mock_lifespan
        self._test_client = TestClient(app)
        self.client = self._test_client.__enter__()

    def teardown_method(self):
        from api.main import app

        self._test_client.__exit__(None, None, None)
        app.router.lifespan_context = self._original_lifespan

    def test_health_returns_200(self):
        response = self.client.get("/health")
        assert response.status_code == 200

    def test_health_body(self):
        response = self.client.get("/health")
        assert response.json()["status"] == "ok"

    def test_ready_returns_200_when_loaded(self):
        response = self.client.get("/ready")
        assert response.status_code == 200

    def test_ready_body_contains_all_artifacts(self):
        response = self.client.get("/ready")
        body = response.json()
        assert body["all_ready"] is True
        assert len(body["artifacts_loaded"]) > 0


class TestDetectEndpoint:
    """Tests for /detect using mocked artifacts."""

    def setup_method(self):
        from api.main import app

        # Replace the real lifespan with _mock_lifespan so mock artifacts are
        # injected synchronously during TestClient.__enter__, with no dependency
        # on patch timing relative to TestClient background threads.
        self._original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _mock_lifespan
        self._test_client = TestClient(app)
        self.client = self._test_client.__enter__()

    def teardown_method(self):
        from api.main import app

        self._test_client.__exit__(None, None, None)
        app.router.lifespan_context = self._original_lifespan

    def _valid_payload(self, **kwargs) -> dict:
        base = {
            "cpu_util": 41.0,
            "mem_util": 72.0,
            "net_io_in": 43.0,
            "net_io_out": 33.0,
            "timestamp": "2026-06-25T14:30:00Z",
        }
        base.update(kwargs)
        return base

    def test_detect_returns_200(self):
        response = self.client.post("/detect", json=self._valid_payload())
        assert response.status_code == 200

    def test_detect_response_has_required_fields(self):
        response = self.client.post("/detect", json=self._valid_payload())
        body = response.json()
        for field in [
            "anomaly_score",
            "severity_label",
            "top_contributing_features",
            "feature_deviation_scores",
            "inference_latency_ms",
        ]:
            assert field in body, f"Missing field: {field}"

    def test_detect_anomaly_score_in_range(self):
        response = self.client.post("/detect", json=self._valid_payload())
        score = response.json()["anomaly_score"]
        assert 0.0 <= score <= 1.0

    def test_detect_invalid_cpu_util_returns_422(self):
        response = self.client.post("/detect", json=self._valid_payload(cpu_util=150.0))
        assert response.status_code == 422

    def test_detect_missing_required_field_returns_422(self):
        payload = self._valid_payload()
        payload.pop("cpu_util")
        response = self.client.post("/detect", json=payload)
        assert response.status_code == 422

    def test_detect_severity_label_is_valid(self):
        response = self.client.post("/detect", json=self._valid_payload())
        label = response.json()["severity_label"]
        assert label in {"Critical", "Warning", "Normal"}

    def test_detect_latency_is_positive(self):
        response = self.client.post("/detect", json=self._valid_payload())
        assert response.json()["inference_latency_ms"] > 0


class TestBatchDetectEndpoint:
    """Tests for /batch_detect using mocked artifacts."""

    def setup_method(self):
        from api.main import app

        # Replace the real lifespan with _mock_lifespan so mock artifacts are
        # injected synchronously during TestClient.__enter__, with no dependency
        # on patch timing relative to TestClient background threads.
        self._original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _mock_lifespan
        self._test_client = TestClient(app)
        self.client = self._test_client.__enter__()

    def teardown_method(self):
        from api.main import app

        self._test_client.__exit__(None, None, None)
        app.router.lifespan_context = self._original_lifespan

    def _make_batch(self, n: int = 3) -> dict:
        return {
            "snapshots": [
                {
                    "cpu_util": float(40 + i * 10),
                    "mem_util": 60.0,
                    "net_io_in": 43.0,
                    "net_io_out": 33.0,
                    "timestamp": f"2026-06-25T14:{30 + i:02d}:00Z",
                }
                for i in range(n)
            ]
        }

    def test_batch_returns_200(self):
        response = self.client.post("/batch_detect", json=self._make_batch())
        assert response.status_code == 200

    def test_batch_returns_correct_n_snapshots(self):
        response = self.client.post("/batch_detect", json=self._make_batch(3))
        assert response.json()["n_snapshots"] == 3

    def test_batch_results_sorted_descending(self):
        response = self.client.post("/batch_detect", json=self._make_batch(3))
        scores = [r["anomaly_score"] for r in response.json()["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_batch_results_have_rank_field(self):
        response = self.client.post("/batch_detect", json=self._make_batch(3))
        for item in response.json()["results"]:
            assert "rank" in item

    def test_empty_snapshots_returns_422(self):
        response = self.client.post("/batch_detect", json={"snapshots": []})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for /metrics."""

    def setup_method(self):
        from api.main import app

        # Replace the real lifespan with _mock_lifespan so mock artifacts are
        # injected synchronously during TestClient.__enter__, with no dependency
        # on patch timing relative to TestClient background threads.
        self._original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _mock_lifespan
        self._test_client = TestClient(app)
        self.client = self._test_client.__enter__()

    def teardown_method(self):
        from api.main import app

        self._test_client.__exit__(None, None, None)
        app.router.lifespan_context = self._original_lifespan

    def test_metrics_returns_200(self):
        response = self.client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_text(self):
        response = self.client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_real_metric(self):
        # clouddrift_api_up was the Day 8 stub — Day 9 replaced it with
        # real generate_latest() output; check a stable real metric name
        response = self.client.get("/metrics")
        assert "clouddrift_requests_total" in response.text


# ---------------------------------------------------------------------------
# Prometheus metrics tests — added Day 9
# ---------------------------------------------------------------------------


class TestPrometheusMetrics:
    """
    Tests for the Prometheus /metrics endpoint content.

    Strategy: make at least one /detect request in setup_method,
    then verify /metrics returns valid text containing the expected
    metric family names. Counter values are not asserted because
    prometheus_client uses a global registry that accumulates across
    test sessions — checking names is stable, checking values is not.
    """

    def setup_method(self):
        from api.main import app

        self._original_lifespan = app.router.lifespan_context
        app.router.lifespan_context = _mock_lifespan
        self._test_client = TestClient(app)
        self.client = self._test_client.__enter__()
        # Seed the counters with at least one request
        self.client.post(
            "/detect",
            json={
                "cpu_util": 41.0,
                "mem_util": 72.0,
                "net_io_in": 43.0,
                "net_io_out": 33.0,
                "timestamp": "2026-06-25T14:30:00Z",
            },
        )

    def teardown_method(self):
        from api.main import app

        self._test_client.__exit__(None, None, None)
        app.router.lifespan_context = self._original_lifespan

    def test_metrics_returns_200(self):
        response = self.client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self):
        response = self.client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_requests_total_counter_present(self):
        response = self.client.get("/metrics")
        assert "clouddrift_requests_total" in response.text

    def test_prediction_latency_histogram_present(self):
        response = self.client.get("/metrics")
        assert "clouddrift_prediction_latency_seconds" in response.text

    def test_anomalies_total_counter_present(self):
        response = self.client.get("/metrics")
        assert "clouddrift_anomalies_total" in response.text

    def test_schema_violations_counter_present(self):
        response = self.client.get("/metrics")
        assert "clouddrift_schema_violations_total" in response.text

    def test_histogram_has_bucket_lines(self):
        response = self.client.get("/metrics")
        assert "_bucket{" in response.text

    def test_histogram_has_sum_and_count(self):
        response = self.client.get("/metrics")
        assert "clouddrift_prediction_latency_seconds_sum" in response.text
        assert "clouddrift_prediction_latency_seconds_count" in response.text


class TestPanderaValidation:
    """
    Tests for the Pandera schema validation layer in /detect.

    Pydantic catches out-of-range values before Pandera runs —
    these tests verify the _validate_snapshot() function directly
    rather than hitting the HTTP layer, which avoids the Pydantic
    interception and tests the Pandera logic independently.
    """

    def test_valid_snapshot_passes_pandera(self):
        from api.routers.detection import _validate_snapshot
        from api.schemas.telemetry import TelemetrySnapshot

        snap = TelemetrySnapshot(
            cpu_util=41.0,
            mem_util=72.0,
            net_io_in=43.0,
            net_io_out=33.0,
            timestamp="2026-06-25T14:30:00Z",
        )
        # Should not raise
        _validate_snapshot(snap)

    def test_null_disk_io_passes_pandera(self):
        from api.routers.detection import _validate_snapshot
        from api.schemas.telemetry import TelemetrySnapshot

        snap = TelemetrySnapshot(
            cpu_util=41.0,
            mem_util=72.0,
            net_io_in=43.0,
            net_io_out=33.0,
            disk_io=None,
            timestamp="2026-06-25T14:30:00Z",
        )
        _validate_snapshot(snap)

    def test_schema_returns_dict_with_required_keys(self):
        """Verify _TELEMETRY_SCHEMA covers all required metric columns."""
        from api.routers.detection import _TELEMETRY_SCHEMA

        expected_cols = {"cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"}
        schema_cols = set(_TELEMETRY_SCHEMA.columns.keys())
        assert expected_cols == schema_cols


class TestMetricImports:
    """Smoke tests: all four metric objects are importable and named correctly."""

    def test_request_counter_importable(self):
        from api.services.metrics import REQUEST_COUNTER

        # prometheus_client strips _total suffix from ._name — it appears
        # only in the /metrics text output, not in the internal attribute
        assert REQUEST_COUNTER._name == "clouddrift_requests"

    def test_anomaly_counter_importable(self):
        from api.services.metrics import ANOMALY_COUNTER

        assert ANOMALY_COUNTER._name == "clouddrift_anomalies"

    def test_latency_histogram_importable(self):
        from api.services.metrics import LATENCY_HISTOGRAM

        assert LATENCY_HISTOGRAM._name == "clouddrift_prediction_latency_seconds"

    def test_schema_violation_counter_importable(self):
        from api.services.metrics import SCHEMA_VIOLATION_COUNTER

        assert SCHEMA_VIOLATION_COUNTER._name == "clouddrift_schema_violations"
