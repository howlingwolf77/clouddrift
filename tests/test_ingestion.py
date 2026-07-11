"""
Data ingestion and validation tests.

Unit tests (no data required):
    TestValidateNullRates           — null-rate validation
    TestValidateTimestampContinuity — gap detection in time-series
    TestDefineTemporalSplit         — single-series 70/15/15 split
    TestGetDatasetSummary           — summary dict structure
    TestNormaliseAlibabaColumns     — Alibaba column cleaning
    TestSMDLoader                   — SMD column selection and metadata

Integration tests (skip when data absent):
    TestSMDIntegration              — real SMD dataset
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    _normalise_alibaba_columns,
    _select_smd_columns,
    get_dataset_summary,
    load_smd_dataset,
)
from src.data.validation import (
    define_temporal_split,
    validate_null_rates,
    validate_smd_schema,
    validate_timestamp_continuity,
)

# ---------------------------------------------------------------------------
# Dataset availability guard
# ---------------------------------------------------------------------------

SMD_ROOT = Path("data/raw/smd/ServerMachineDataset")
SMD_AVAILABLE = (SMD_ROOT / "train").exists() and (SMD_ROOT / "test_label").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smd_raw_df(n_rows: int = 10, n_cols: int = 38) -> pd.DataFrame:
    """
    Create a synthetic 38-column raw SMD file DataFrame.
    Simulates output of pd.read_csv() on an SMD .txt file.
    Values are in [0, 1] as per the SMD published dataset.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(n_rows, n_cols)),
        columns=list(range(n_cols)),
    )


# ---------------------------------------------------------------------------
# Unit tests — no real data required
# ---------------------------------------------------------------------------


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
        ts += list(pd.date_range("2024-01-01 05:00:00", periods=50, freq="5min"))
        df = pd.DataFrame({"timestamp": ts})
        result = validate_timestamp_continuity(df)
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


class TestNormaliseAlibabaColumns:
    """Tests for Alibaba column normalisation and cleaning."""

    def _make_alibaba_chunk(self) -> pd.DataFrame:
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
                "disk_io_percent": [5.0, -1.0, 101.0],
            }
        )

    def test_sentinel_disk_values_become_nan(self):
        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert result["disk_io"].iloc[0] == 5.0
        assert pd.isna(result["disk_io"].iloc[1])
        assert pd.isna(result["disk_io"].iloc[2])

    def test_timestamp_converted_from_seconds(self):
        from src.data.ingestion import ALIBABA_BASE_TIME

        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "timestamp" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        expected_first = ALIBABA_BASE_TIME + pd.to_timedelta(386640, unit="s")
        assert result["timestamp"].iloc[0] == expected_first

    def test_columns_renamed_to_clouddrift_standard(self):
        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "cpu_util" in result.columns
        assert "mem_util" in result.columns
        assert "net_io_in" in result.columns
        assert "net_io_out" in result.columns
        assert "disk_io" in result.columns
        assert "cpu_util_percent" not in result.columns
        assert "disk_io_percent" not in result.columns

    def test_sparse_columns_dropped(self):
        df = self._make_alibaba_chunk()
        result = _normalise_alibaba_columns(df)
        assert "mem_gps" not in result.columns
        assert "mkpi" not in result.columns
        assert "time_stamp" not in result.columns

    def test_missing_required_columns_returns_none(self):
        df = pd.DataFrame({"machine_id": ["m_1"], "time_stamp": [100.0]})
        result = _normalise_alibaba_columns(df)
        assert result is None


class TestSMDLoader:
    """
    Unit tests for SMD-specific ingestion logic.
    No real SMD data required — uses synthetic 38-column DataFrames.
    """

    def test_select_smd_columns_returns_five_columns(self):
        raw = _make_smd_raw_df()
        result = _select_smd_columns(raw)
        assert set(result.columns) == {
            "cpu_util",
            "net_io_in",
            "net_io_out",
            "disk_io",
            "mem_util",
        }

    def test_select_smd_columns_clips_to_0_1(self):
        raw = _make_smd_raw_df()
        raw.iloc[0, 0] = 1.5
        raw.iloc[1, 0] = -0.3
        result = _select_smd_columns(raw)
        assert result["cpu_util"].iloc[0] == 1.0
        assert result["cpu_util"].iloc[1] == 0.0

    def test_select_smd_columns_values_in_range(self):
        raw = _make_smd_raw_df(n_rows=100)
        result = _select_smd_columns(raw)
        for col in result.columns:
            assert result[col].between(0.0, 1.0).all()

    def test_select_smd_columns_correct_mapping(self):
        """Column 0 → cpu_util, column 5 → mem_util (clipped to 1.0)."""
        raw = pd.DataFrame({i: [float(i)] for i in range(38)})
        result = _select_smd_columns(raw)
        assert result["cpu_util"].iloc[0] == pytest.approx(0.0)
        assert result["mem_util"].iloc[0] == pytest.approx(1.0)

    @pytest.mark.skipif(not SMD_AVAILABLE, reason="SMD dataset not downloaded")
    def test_load_smd_dataset_source_file_equals_machine_id(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        assert (df["source_file"] == df["machine_id"]).all()

    @pytest.mark.skipif(not SMD_AVAILABLE, reason="SMD dataset not downloaded")
    def test_load_smd_dataset_single_machine_shape(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        assert len(df) > 50_000
        assert df.shape[1] == 9

    @pytest.mark.skipif(not SMD_AVAILABLE, reason="SMD dataset not downloaded")
    def test_load_smd_dataset_anomaly_rate_reasonable(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        rate = df["is_anomaly"].mean()
        assert 0.01 < rate < 0.20


# ---------------------------------------------------------------------------
# Integration tests — require real SMD data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SMD_AVAILABLE, reason="SMD dataset not downloaded")
class TestSMDIntegration:
    """Integration tests against the real SMD dataset."""

    def test_smd_loads_without_error(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        assert len(df) > 0

    def test_smd_has_expected_columns(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        expected = {
            "machine_id",
            "source_file",
            "timestamp",
            "cpu_util",
            "mem_util",
            "net_io_in",
            "net_io_out",
            "disk_io",
            "is_anomaly",
        }
        assert expected == set(df.columns)

    def test_smd_values_in_0_1_range(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        for col in ["cpu_util", "mem_util", "net_io_in", "net_io_out", "disk_io"]:
            col_data = df[col].dropna()
            assert col_data.min() >= 0.0
            assert col_data.max() <= 1.0

    def test_smd_timestamps_are_datetime(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_smd_timestamps_monotonically_increasing(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        assert df["timestamp"].is_monotonic_increasing

    def test_smd_schema_validation_passes(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        validated = validate_smd_schema(df)
        assert validated is not None
        assert len(validated) == len(df)

    def test_smd_train_rows_all_normal(self):
        df = load_smd_dataset(machines=["machine-1-1"])
        n_train_approx = len(df) // 2
        train_half = df.iloc[:n_train_approx]
        assert not train_half["is_anomaly"].any()

    def test_smd_multi_machine_load(self):
        machines = ["machine-1-1", "machine-1-2"]
        df = load_smd_dataset(machines=machines)
        assert df["machine_id"].nunique() == 2
        assert set(df["machine_id"].unique()) == set(machines)
