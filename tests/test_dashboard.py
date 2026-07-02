"""
Day 10 tests: drift monitoring logic.

Streamlit itself is not unit-tested here — Streamlit provides
`streamlit.testing.v1.AppTest` for UI testing which requires the full
app to render; too heavy for the sprint timeline. These tests cover
the pure-Python drift monitoring functions in dashboard/drift_monitor.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.drift_monitor import (
    MIN_CURRENT_ROWS,
    compute_zscore_drift,
    generate_drift_report,
    load_reference_data,
)

ARTIFACTS_EXIST = Path("artifacts/api_reference_stats.json").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(cpu: float = 41.0, mem: float = 60.0) -> dict:
    return {
        "cpu_util": cpu,
        "mem_util": mem,
        "net_io_in": 43.0,
        "net_io_out": 33.0,
        "disk_io": 5.0,
    }


def _make_snapshots(n: int, anomalous: bool = False) -> list[dict]:
    rng = np.random.default_rng(42)
    # Generate all n values per metric at once (returns ndarray, not scalar)
    # so .clip() is available on the array.
    if anomalous:
        return [
            {
                "cpu_util": float(rng.normal(88, 5, n).clip(0, 100)[i]),
                "mem_util": float(rng.normal(92, 4, n).clip(0, 100)[i]),
                "net_io_in": float(rng.normal(85, 8, n).clip(0, 100)[i]),
                "net_io_out": float(rng.normal(78, 7, n).clip(0, 100)[i]),
                "disk_io": float(rng.normal(60, 10, n).clip(0, 100)[i]),
            }
            for i in range(n)
        ]
    return [
        {
            "cpu_util": float(rng.normal(40, 10, n).clip(0, 100)[i]),
            "mem_util": float(rng.normal(60, 10, n).clip(0, 100)[i]),
            "net_io_in": float(rng.normal(43, 8, n).clip(0, 100)[i]),
            "net_io_out": float(rng.normal(33, 6, n).clip(0, 100)[i]),
            "disk_io": float(rng.normal(10, 8, n).clip(0, 100)[i]),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# load_reference_data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="api_reference_stats.json missing")
class TestLoadReferenceData:
    def test_returns_dataframe(self):
        df = load_reference_data(n_rows=50)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_reference_data(n_rows=50)
        # At least three metric columns must be present
        expected = {"cpu_util", "mem_util", "net_io_in"}
        assert expected.issubset(set(df.columns))

    def test_respects_n_rows(self):
        df = load_reference_data(n_rows=30)
        assert len(df) <= 30

    def test_no_all_nan_rows(self):
        df = load_reference_data(n_rows=50)
        # At least 80% of rows must have non-NaN cpu_util and mem_util
        valid_rows = df[["cpu_util", "mem_util"]].notna().all(axis=1).sum()
        assert valid_rows / len(df) >= 0.8


# ---------------------------------------------------------------------------
# compute_zscore_drift
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ARTIFACTS_EXIST, reason="api_reference_stats.json missing")
class TestComputeZscoreDrift:
    def test_returns_dict(self):
        snaps = _make_snapshots(10)
        result = compute_zscore_drift(snaps)
        assert isinstance(result, dict)

    def test_empty_input_returns_empty(self):
        result = compute_zscore_drift([])
        assert result == {}

    def test_keys_are_metric_names(self):
        snaps = _make_snapshots(10)
        result = compute_zscore_drift(snaps)
        valid = {"cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"}
        assert set(result.keys()).issubset(valid)

    def test_all_values_are_non_negative(self):
        snaps = _make_snapshots(20)
        result = compute_zscore_drift(snaps)
        for val in result.values():
            assert val >= 0.0

    def test_anomalous_readings_produce_higher_drift(self):
        normal_drift = compute_zscore_drift(_make_snapshots(30, anomalous=False))
        anomalous_drift = compute_zscore_drift(_make_snapshots(30, anomalous=True))

        # At least one metric should show more drift in anomalous mode
        common_metrics = set(normal_drift.keys()) & set(anomalous_drift.keys())
        assert any(anomalous_drift[m] > normal_drift[m] for m in common_metrics), (
            "Expected anomalous readings to produce higher drift than normal"
        )

    def test_values_are_finite(self):
        snaps = _make_snapshots(20)
        result = compute_zscore_drift(snaps)
        for val in result.values():
            assert np.isfinite(val)


# ---------------------------------------------------------------------------
# generate_drift_report
# ---------------------------------------------------------------------------


class TestGenerateDriftReport:
    def _make_ref_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "cpu_util": rng.normal(40, 10, n).clip(0, 100),
                "mem_util": rng.normal(60, 10, n).clip(0, 100),
                "net_io_in": rng.normal(43, 8, n).clip(0, 100),
                "net_io_out": rng.normal(33, 6, n).clip(0, 100),
            }
        )

    def test_raises_on_insufficient_current_data(self):
        snaps = _make_snapshots(5)  # below MIN_CURRENT_ROWS
        ref_df = self._make_ref_df()
        with pytest.raises(ValueError, match="at least"):
            generate_drift_report(snaps, reference_df=ref_df)

    def test_returns_html_path_and_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.drift_monitor.DRIFT_LOG_DIR", tmp_path)
        snaps = _make_snapshots(MIN_CURRENT_ROWS + 5)
        ref_df = self._make_ref_df(n=100)
        html_path, summary = generate_drift_report(snaps, reference_df=ref_df)

        assert Path(html_path).exists(), "HTML report file not created"
        assert Path(html_path).suffix == ".html"
        assert isinstance(summary, dict)

    def test_html_file_is_non_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.drift_monitor.DRIFT_LOG_DIR", tmp_path)
        snaps = _make_snapshots(MIN_CURRENT_ROWS + 5)
        ref_df = self._make_ref_df(n=100)
        html_path, _ = generate_drift_report(snaps, reference_df=ref_df)
        assert Path(html_path).stat().st_size > 1000

    def test_summary_is_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dashboard.drift_monitor.DRIFT_LOG_DIR", tmp_path)
        snaps = _make_snapshots(MIN_CURRENT_ROWS + 5)
        ref_df = self._make_ref_df(n=100)
        _, summary = generate_drift_report(snaps, reference_df=ref_df)
        assert isinstance(summary, dict)
        assert "dataset_drifted" in summary
        assert "n_drifted" in summary


# ---------------------------------------------------------------------------
# MIN_CURRENT_ROWS constant
# ---------------------------------------------------------------------------


class TestConstants:
    def test_min_current_rows_is_positive_integer(self):
        assert isinstance(MIN_CURRENT_ROWS, int)
        assert MIN_CURRENT_ROWS > 0

    def test_min_current_rows_is_at_least_30(self):
        assert MIN_CURRENT_ROWS >= 30
