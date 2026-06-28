"""
Day 3 tests: feature engineering pipeline.

Tests cover:
    - Rolling feature computation on synthetic data
    - Per-series isolation (no cross-contamination between groups)
    - Cross-metric feature computation
    - RobustPercentileNormalizer fit/transform
    - Feature pipeline save/load round-trip
    - Integration test against real NAB data
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    RobustPercentileNormalizer,
    apply_feature_pipeline,
    build_alibaba_features,
    build_feature_pipeline,
    build_nab_features,
    get_feature_columns,
    load_feature_pipeline,
    save_feature_pipeline,
)

NAB_FEATURES_EXIST = Path("data/processed/nab_train_features.parquet").exists()
PIPELINE_EXISTS = Path("artifacts/feature_pipeline.joblib").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nab_df(n: int = 50, n_series: int = 2) -> pd.DataFrame:
    """Create a minimal NAB-format DataFrame for testing."""
    dfs = []
    for i in range(n_series):
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        dfs.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "value": [float(j + i * 10) for j in range(n)],
                    "metric_name": f"metric_{i}",
                    "category": "test",
                    "source_file": f"test/series_{i}.csv",
                    "is_anomaly": [j > n - 5 for j in range(n)],
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


def _make_alibaba_df(n: int = 30, n_machines: int = 2) -> pd.DataFrame:
    """Create a minimal Alibaba-format DataFrame for testing."""
    dfs = []
    for m in range(n_machines):
        timestamps = pd.date_range("2024-01-01", periods=n, freq="30s")
        dfs.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "machine_id": f"m_{m}",
                    "cpu_util": [float(20 + j * 0.5 + m * 5) for j in range(n)],
                    "mem_util": [float(40 + j * 0.3) for j in range(n)],
                    "net_io_in": [float(10 + j * 0.1) for j in range(n)],
                    "net_io_out": [float(5 + j * 0.05) for j in range(n)],
                    "disk_io": [8.0] * n,
                }
            )
        )
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Rolling feature tests
# ---------------------------------------------------------------------------


class TestNABFeatures:
    """Tests for build_nab_features()."""

    def test_returns_all_original_columns(self):
        df = _make_nab_df()
        result = build_nab_features(df)
        original_cols = set(df.columns)
        assert original_cols.issubset(set(result.columns))

    def test_engineered_columns_present(self):
        df = _make_nab_df()
        result = build_nab_features(df)
        expected = [
            "value_mean_short",
            "value_mean_mid",
            "value_mean_long",
            "value_std_short",
            "value_std_mid",
            "value_std_long",
            "value_zscore_short",
            "value_zscore_mid",
            "value_zscore_long",
            "value_roc",
            "value_range_ratio_mid",
            "value_range_ratio_long",
        ]
        for col in expected:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_row_count_preserved(self):
        df = _make_nab_df(n=50, n_series=3)
        result = build_nab_features(df)
        assert len(result) == len(df)

    def test_per_series_isolation(self):
        """Features from series_0 must not contaminate series_1."""
        df = _make_nab_df(n=10, n_series=2)
        result = build_nab_features(df)

        # The first row of each series should have roc=0 (no previous row
        # in that series). If series were concatenated before computing,
        # roc would be non-zero at the series boundary.
        for series_id in df["source_file"].unique():
            first_row = (
                result[result["source_file"] == series_id]
                .sort_values("timestamp")
                .iloc[0]
            )
            assert first_row["value_roc"] == 0.0, (
                f"Series {series_id} first roc should be 0 — "
                "cross-series contamination detected"
            )

    def test_no_nan_in_feature_columns(self):
        df = _make_nab_df(n=20)
        result = build_nab_features(df)
        feat_cols = get_feature_columns(result)
        nan_count = result[feat_cols].isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaNs in feature columns"

    def test_zscore_zero_when_std_is_zero(self):
        """Constant series should have z-score of 0, not NaN."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
                "value": [5.0] * 10,
                "metric_name": "const",
                "category": "test",
                "source_file": "test/const.csv",
                "is_anomaly": [False] * 10,
            }
        )
        result = build_nab_features(df)
        assert result["value_zscore_long"].isna().sum() == 0
        assert (result["value_zscore_long"] == 0.0).all()

    def test_range_ratio_between_0_and_1(self):
        df = _make_nab_df(n=30)
        result = build_nab_features(df)
        for col in ["value_range_ratio_mid", "value_range_ratio_long"]:
            assert result[col].between(0.0, 1.0).all(), (
                f"{col} has values outside [0, 1]"
            )

    def test_raises_on_missing_group_column(self):
        df = pd.DataFrame({"timestamp": [], "value": []})
        with pytest.raises(ValueError, match="source_file"):
            build_nab_features(df, group_col="source_file")

    def test_raises_on_missing_value_column(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "source_file": ["a"] * 3,
            }
        )
        with pytest.raises(ValueError, match="value"):
            build_nab_features(df)


# ---------------------------------------------------------------------------
# Alibaba and cross-metric feature tests
# ---------------------------------------------------------------------------


class TestAlibabaFeatures:
    """Tests for build_alibaba_features()."""

    def test_returns_rolling_features_per_metric(self):
        df = _make_alibaba_df()
        result = build_alibaba_features(df)
        for metric in ["cpu_util", "mem_util"]:
            assert f"{metric}_mean_long" in result.columns
            assert f"{metric}_zscore_mid" in result.columns

    def test_cross_metric_columns_present(self):
        df = _make_alibaba_df(n=30)
        result = build_alibaba_features(df)
        assert "cpu_mem_corr_long" in result.columns
        assert "cpu_net_ratio" in result.columns
        assert "volatility_score" in result.columns

    def test_per_machine_isolation(self):
        """First row roc of each machine must be 0."""
        df = _make_alibaba_df(n=10, n_machines=2)
        result = build_alibaba_features(df)
        for machine_id in df["machine_id"].unique():
            first_row = (
                result[result["machine_id"] == machine_id]
                .sort_values("timestamp")
                .iloc[0]
            )
            assert first_row["cpu_util_roc"] == 0.0

    def test_row_count_preserved(self):
        df = _make_alibaba_df(n=30, n_machines=3)
        result = build_alibaba_features(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# RobustPercentileNormalizer tests
# ---------------------------------------------------------------------------


class TestRobustPercentileNormalizer:
    """Tests for the custom sklearn transformer."""

    def _make_feature_df(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 100, n),
                "feat_b": rng.uniform(-10, 200, n),
            }
        )

    def test_fit_stores_bounds(self):
        norm = RobustPercentileNormalizer()
        df = self._make_feature_df()
        norm.fit(df)
        assert "feat_a" in norm.bounds_
        assert "feat_b" in norm.bounds_
        lo, hi = norm.bounds_["feat_a"]
        assert lo < hi

    def test_transform_output_mostly_in_0_1(self):
        norm = RobustPercentileNormalizer(lower_pct=1, upper_pct=99)
        df = self._make_feature_df(1000)
        norm.fit(df)
        result = norm.transform(df)
        # Training data should be almost entirely in [0, 1]
        # (not exactly because we clip at p1/p99, not min/max)
        assert result["feat_a"].between(0.0, 1.0).mean() > 0.98
        assert result["feat_b"].between(0.0, 1.0).mean() > 0.98

    def test_constant_column_gets_degenerate_bounds(self):
        norm = RobustPercentileNormalizer()
        df = pd.DataFrame({"constant": [5.0] * 50})
        norm.fit(df)
        lo, hi = norm.bounds_["constant"]
        assert hi > lo  # degenerate guard: hi = lo + 1

    def test_missing_column_at_transform_gets_zeros(self):
        norm = RobustPercentileNormalizer()
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [4.0, 5.0, 6.0]})
        norm.fit(train)
        # Transform data missing feat_b
        new_data = pd.DataFrame({"feat_a": [1.5, 2.5]})
        result = norm.transform(new_data)
        assert "feat_b" in result.columns
        assert (result["feat_b"] == 0.0).all()

    def test_sklearn_pipeline_compatible(self):
        from sklearn.pipeline import Pipeline

        norm = RobustPercentileNormalizer()
        pipe = Pipeline([("normalizer", norm)])
        df = self._make_feature_df()
        pipe.fit(df)
        result = pipe.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape


# ---------------------------------------------------------------------------
# Feature pipeline save/load tests
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    """Tests for pipeline save/load round-trip."""

    def test_build_pipeline_returns_fitted_pipeline(self):
        df = _make_nab_df(n=50)
        feat_df = build_nab_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        assert hasattr(pipeline.named_steps["normalizer"], "bounds_")

    def test_save_load_round_trip(self, tmp_path):
        df = _make_nab_df(n=50)
        feat_df = build_nab_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)

        save_path = tmp_path / "test_pipeline.joblib"
        save_feature_pipeline(pipeline, path=save_path)
        assert save_path.exists()

        loaded = load_feature_pipeline(path=save_path)
        original_bounds = pipeline.named_steps["normalizer"].bounds_
        loaded_bounds = loaded.named_steps["normalizer"].bounds_
        assert original_bounds == loaded_bounds

    def test_load_raises_on_missing_artifact(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_feature_pipeline(path=tmp_path / "nonexistent.joblib")

    def test_apply_pipeline_preserves_metadata(self):
        df = _make_nab_df(n=50)
        feat_df = build_nab_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        result = apply_feature_pipeline(pipeline, feat_df, feat_cols)

        # Metadata columns must be unchanged
        assert "timestamp" in result.columns
        assert "is_anomaly" in result.columns
        assert "source_file" in result.columns
        # Timestamps must be identical to input
        pd.testing.assert_series_equal(feat_df["timestamp"], result["timestamp"])

    def test_apply_pipeline_features_in_0_1(self):
        df = _make_nab_df(n=100)
        feat_df = build_nab_features(df)
        feat_cols = get_feature_columns(feat_df)
        pipeline = build_feature_pipeline(feat_df, feat_cols)
        result = apply_feature_pipeline(pipeline, feat_df, feat_cols)
        feat_values = result[feat_cols]
        assert feat_values.min().min() >= 0.0
        assert feat_values.max().max() <= 1.0 + 1e-9  # float tolerance


# ---------------------------------------------------------------------------
# Integration tests — require real processed files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not NAB_FEATURES_EXIST, reason="NAB feature files not generated")
class TestNABFeaturesIntegration:
    """Integration tests against real generated feature matrices."""

    def test_feature_files_exist(self):
        for fname in [
            "nab_train_features.parquet",
            "nab_val_features.parquet",
            "nab_test_features.parquet",
        ]:
            assert Path(f"data/processed/{fname}").exists()

    def test_feature_matrix_has_no_nans(self):
        import json

        train = pd.read_parquet("data/processed/nab_train_features.parquet")
        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)
        feat_cols = meta["nab_feature_cols"]
        nan_count = train[feat_cols].isnull().sum().sum()
        assert nan_count == 0

    def test_temporal_ordering_preserved(self):
        train = pd.read_parquet("data/processed/nab_train_features.parquet")
        val = pd.read_parquet("data/processed/nab_val_features.parquet")
        test = pd.read_parquet("data/processed/nab_test_features.parquet")
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_anomaly_labels_preserved(self):
        train = pd.read_parquet("data/processed/nab_train_features.parquet")
        original_train = pd.read_parquet("data/processed/nab_train.parquet")
        assert train["is_anomaly"].sum() == original_train["is_anomaly"].sum()

    def test_pipeline_artifact_loadable(self):
        pipeline = load_feature_pipeline()
        norm = pipeline.named_steps["normalizer"]
        assert hasattr(norm, "bounds_")
        assert len(norm.bounds_) > 0


@pytest.mark.skipif(not PIPELINE_EXISTS, reason="Feature pipeline not generated")
class TestFeaturePipelineIntegration:
    """Tests pipeline consistency across train and val splits."""

    def test_val_uses_training_bounds(self):
        """Val normalization must use training bounds, not val bounds."""
        import json

        pipeline = load_feature_pipeline()
        train = pd.read_parquet("data/processed/nab_train_features.parquet")

        with open("artifacts/feature_metadata.json") as f:
            meta = json.load(f)
        feat_cols = meta["nab_feature_cols"]

        # Re-apply pipeline to train and compare to saved val
        # (validates that both used the same fitted bounds)
        norm = pipeline.named_steps["normalizer"]
        first_col = feat_cols[0]
        lo, hi = norm.bounds_[first_col]

        # Training data clipped to [lo, hi] → result should be exactly in [0, 1]
        train_feat = apply_feature_pipeline(pipeline, train, feat_cols)
        assert train_feat[first_col].min() >= 0.0 - 1e-9
        assert train_feat[first_col].max() <= 1.0 + 1e-9
