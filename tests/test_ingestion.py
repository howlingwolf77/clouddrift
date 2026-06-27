"""
Day 2 tests: data ingestion and validation pipeline.
Tests run against real NAB data — confirms the pipeline works
end-to-end, not just that functions are importable.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    _label_anomalies,
    get_dataset_summary,
    load_nab_dataset,
)
from src.data.validation import (
    define_temporal_split,
    generate_data_quality_report,
    validate_null_rates,
    validate_timestamp_continuity,
)

NAB_ROOT = Path("data/raw/nab")
NAB_AVAILABLE = NAB_ROOT.exists() and (NAB_ROOT / "data").exists()


# ---------------------------------------------------------------------------
# Unit tests — no real data required
# ---------------------------------------------------------------------------


class TestLabelAnomalies:
    """Tests for _label_anomalies helper."""

    def test_no_windows_returns_all_false(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=5, freq="5min"))
        result = _label_anomalies(timestamps, [])
        assert not result.any()

    def test_window_labels_correct_rows(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=10, freq="5min"))
        windows = [["2024-01-01 00:10:00", "2024-01-01 00:25:00"]]
        result = _label_anomalies(timestamps, windows)
        # rows at 00:10, 00:15, 00:20, 00:25 should be True
        assert result.sum() == 4

    def test_multiple_windows(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=20, freq="5min"))
        windows = [
            ["2024-01-01 00:05:00", "2024-01-01 00:10:00"],
            ["2024-01-01 00:30:00", "2024-01-01 00:35:00"],
        ]
        result = _label_anomalies(timestamps, windows)
        assert result.sum() == 4  # 2 rows per window

    def test_result_is_boolean_series(self):
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=5, freq="5min"))
        result = _label_anomalies(timestamps, [])
        assert result.dtype == bool


class TestValidateNullRates:
    """Tests for validate_null_rates."""

    def test_all_nulls_below_threshold(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = validate_null_rates(df, max_null_pct=5.0)
        assert all(v["pass"] for v in result.values())

    def test_column_exceeding_threshold_fails(self):
        df = pd.DataFrame({"a": [1.0, None, None, None, None, 6.0]})
        result = validate_null_rates(df, max_null_pct=5.0)
        assert not result["a"]["pass"]

    def test_null_pct_calculation_correct(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0]})
        result = validate_null_rates(df, max_null_pct=50.0)
        assert result["x"]["null_pct"] == 25.0
        assert result["x"]["pass"]


class TestValidateTimestampContinuity:
    """Tests for validate_timestamp_continuity."""

    def test_regular_series_passes(self):
        df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=100, freq="5min")}
        )
        result = validate_timestamp_continuity(df)
        assert result["pass"]
        assert result["large_gap_count"] == 0

    def test_large_gap_detected(self):
        ts = list(pd.date_range("2024-01-01", periods=50, freq="5min"))
        # Insert a 60-minute gap
        ts += list(pd.date_range("2024-01-01 05:00:00", periods=50, freq="5min"))
        df = pd.DataFrame({"timestamp": ts})
        result = validate_timestamp_continuity(df)
        # Continuity is informational — pass is always True regardless of gaps.
        # The gap is surfaced via large_gap_count for downstream awareness.
        assert result["pass"]
        assert result["large_gap_count"] >= 1

    def test_missing_column_returns_error(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = validate_timestamp_continuity(df, timestamp_col="timestamp")
        assert "error" in result
        assert not result["pass"]


class TestDefineTemporalSplit:
    """Tests for define_temporal_split."""

    def _make_df(self, n: int = 1000) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
                "value": range(n),
            }
        )

    def test_correct_split_sizes(self):
        df = self._make_df(1000)
        train, val, test = define_temporal_split(df, train_pct=0.70, val_pct=0.15)
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_no_data_leakage(self):
        df = self._make_df(1000)
        train, val, test = define_temporal_split(df)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_all_rows_accounted_for(self):
        df = self._make_df(1000)
        train, val, test = define_temporal_split(df)
        assert len(train) + len(val) + len(test) == 1000

    def test_invalid_pct_raises(self):
        df = self._make_df(100)
        with pytest.raises(AssertionError):
            define_temporal_split(df, train_pct=0.90, val_pct=0.20)


class TestGetDatasetSummary:
    """Tests for get_dataset_summary."""

    def test_returns_expected_keys(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
                "value": range(10),
                "is_anomaly": [False] * 9 + [True],
            }
        )
        summary = get_dataset_summary(df, name="test")
        assert summary["rows"] == 10
        assert "null_rates_pct" in summary
        assert "time_range" in summary
        assert "anomaly_distribution" in summary

    def test_anomaly_rate_calculated_correctly(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
                "value": range(10),
                "is_anomaly": [True] * 2 + [False] * 8,
            }
        )
        summary = get_dataset_summary(df)
        assert summary["anomaly_distribution"]["anomaly_rate_pct"] == 20.0


# ---------------------------------------------------------------------------
# Integration tests — require real NAB data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NAB_AVAILABLE, reason="NAB dataset not downloaded")
class TestNABIntegration:
    """Integration tests that run against the real NAB dataset."""

    def test_nab_loads_without_error(self):
        df = load_nab_dataset()
        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "is_anomaly" in df.columns

    def test_nab_has_expected_columns(self):
        df = load_nab_dataset()
        expected = {
            "timestamp",
            "value",
            "metric_name",
            "category",
            "source_file",
            "is_anomaly",
        }
        assert expected.issubset(set(df.columns))

    def test_nab_has_anomaly_labels(self):
        df = load_nab_dataset()
        assert df["is_anomaly"].sum() > 0, "No anomaly rows found"
        assert df["is_anomaly"].mean() < 0.20, "Anomaly rate seems too high"

    def test_nab_timestamps_are_datetime(self):
        df = load_nab_dataset()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_nab_quality_report_passes(self):
        df = load_nab_dataset()
        report = generate_data_quality_report(df, dataset_name="NAB")
        assert report["overall_pass"], f"Data quality report failed:\n{report}"

    def test_processed_parquet_files_exist(self):
        expected_files = [
            "data/processed/nab_labeled.parquet",
            "data/processed/nab_train.parquet",
            "data/processed/nab_val.parquet",
            "data/processed/nab_test.parquet",
        ]
        for f in expected_files:
            assert Path(f).exists(), f"Missing processed file: {f}"

    def test_parquet_roundtrip(self):
        original = load_nab_dataset()
        reloaded = pd.read_parquet("data/processed/nab_labeled.parquet")
        assert len(original) == len(reloaded)
        assert set(original.columns) == set(reloaded.columns)


# ---------------------------------------------------------------------------
# Alibaba-specific unit tests
# ---------------------------------------------------------------------------


class TestNormaliseAlibabaColumns:
    """Tests for Alibaba column normalisation and cleaning."""

    def _make_alibaba_chunk(self) -> pd.DataFrame:
        """Minimal Alibaba-format DataFrame for testing."""
        return pd.DataFrame(
            {
                "machine_id": ["m_1", "m_1", "m_1"],
                "time_stamp": [386640.0, 386670.0, 386700.0],
                "cpu_util_percent": [41.0, 43.0, 44.0],
                "mem_util_percent": [92.0, 92.0, 93.0],
                "mem_gps": [float("nan"), float("nan"), float("nan")],
                "mkpi": [float("nan"), float("nan"), float("nan")],
                "net_in": [43.04, 43.04, 43.05],
                "net_out": [33.08, 33.08, 33.09],
                "disk_io_percent": [5.0, -1.0, 101.0],  # -1 and 101 are sentinels
            }
        )

    def test_sentinel_disk_values_become_nan(self):
        from src.data.ingestion import _normalise_alibaba_columns

        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        # Row 0 has valid disk_io=5.0; rows 1 and 2 have sentinels
        assert result["disk_io"].iloc[0] == 5.0
        assert pd.isna(result["disk_io"].iloc[1])  # was -1
        assert pd.isna(result["disk_io"].iloc[2])  # was 101

    def test_timestamp_converted_from_seconds(self):
        from src.data.ingestion import ALIBABA_BASE_TIME, _normalise_alibaba_columns

        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "timestamp" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        # First timestamp: base + 386640 seconds
        expected_first = ALIBABA_BASE_TIME + pd.to_timedelta(386640, unit="s")
        assert result["timestamp"].iloc[0] == expected_first

    def test_columns_renamed_to_clouddrift_standard(self):
        from src.data.ingestion import _normalise_alibaba_columns

        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "cpu_util" in result.columns
        assert "mem_util" in result.columns
        assert "net_io_in" in result.columns
        assert "net_io_out" in result.columns
        assert "disk_io" in result.columns
        # Original names should be gone
        assert "cpu_util_percent" not in result.columns
        assert "disk_io_percent" not in result.columns

    def test_sparse_columns_dropped(self):
        from src.data.ingestion import _normalise_alibaba_columns

        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "mem_gps" not in result.columns
        assert "mkpi" not in result.columns
        assert "time_stamp" not in result.columns

    def test_missing_required_columns_returns_none(self):
        from src.data.ingestion import _normalise_alibaba_columns

        df = pd.DataFrame({"machine_id": ["m_1"], "time_stamp": [100.0]})
        result = _normalise_alibaba_columns(df)
        assert result is None
